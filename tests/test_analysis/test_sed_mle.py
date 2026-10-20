#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Fisher information matrix computation, log-distance reparameterization,
diagonal preconditioning, and the Monte Carlo draws of ``logpost_grid``.

These tests validate the numerical correctness of the precision matrix
(inverse covariance) computed in ``_get_sed_mle``, the (ln d, Av, Rv) draws,
weights and effective sample size of ``BruteForce.logpost_grid``, the
``inverse3`` utility with regularization, and the jitted helpers of
``loglike_grid``.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats

from brutus.analysis.individual import (
    BruteForce,
    _chi2_from_resid,
    _get_sed_mle,
    _init_mag_resid,
)
from brutus.core.individual import StarGrid
from brutus.core.sed_utils import _get_seds
from brutus.utils.math import LOG_ZERO, inverse3, isPSD
from brutus.utils.sampling import quantile

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def simple_mag_coeffs():
    """
    Create simple magnitude coefficients for a single model with 5 bands.

    Returns mag_coeffs of shape (1, 5, 3) representing
    (base_mag, reddening_vector_R0, dR/dRv) for each band.
    """
    Nmodels, Nbands = 1, 5

    mag_coeffs = np.zeros((Nmodels, Nbands, 3))
    # Base magnitudes (typical stellar SED getting fainter at shorter wavelengths)
    mag_coeffs[0, :, 0] = np.array([16.0, 15.5, 15.2, 15.0, 14.9])
    # Reddening vector R0 (larger in bluer bands)
    mag_coeffs[0, :, 1] = np.array([1.5, 1.2, 1.0, 0.8, 0.7])
    # dR/dRv (differential reddening w.r.t. Rv)
    mag_coeffs[0, :, 2] = np.array([0.15, 0.12, 0.10, 0.08, 0.07])

    return mag_coeffs


@pytest.fixture
def multi_model_mag_coeffs():
    """
    Create magnitude coefficients for multiple models (5 models, 5 bands).
    """
    Nmodels, Nbands = 5, 5

    mag_coeffs = np.zeros((Nmodels, Nbands, 3))
    for i in range(Nmodels):
        base = 14.0 + i * 0.5
        mag_coeffs[i, :, 0] = base + np.array([0, -0.2, -0.3, -0.4, -0.45])
        mag_coeffs[i, :, 1] = np.array([1.5, 1.2, 1.0, 0.8, 0.7])
        mag_coeffs[i, :, 2] = np.array([0.15, 0.12, 0.10, 0.08, 0.07])

    return mag_coeffs


def _make_synthetic_data(mag_coeffs, av_true, rv_true, scale_true, snr=100.0):
    """
    Helper: create synthetic observed flux from mag_coeffs at given
    (scale, Av, Rv), with Gaussian noise at the specified SNR.

    Returns (data, tot_var, resid_placeholder) ready for ``_get_sed_mle``.
    """
    Nmodels, Nbands = mag_coeffs.shape[0], mag_coeffs.shape[1]
    av_arr = np.full(Nmodels, av_true)
    rv_arr = np.full(Nmodels, rv_true)

    # Compute model flux at the given Av/Rv
    seds_flux, _, _ = _get_seds(mag_coeffs, av_arr, rv_arr, return_flux=True)

    # Scale to the true distance
    true_flux = seds_flux[0] * scale_true  # shape (Nbands,)

    # Noise
    flux_err = true_flux / snr
    rng = np.random.RandomState(999)
    data = true_flux + rng.normal(0, flux_err)

    tot_var = flux_err**2
    resid = np.zeros((Nmodels, Nbands))

    return data, tot_var, resid


# ============================================================================
# Part 1 -- Fisher information / ar_mix validation
# ============================================================================


class TestFisherInformation:
    """Test the Fisher information matrix computed in _get_sed_mle."""

    def test_ar_mix_matches_finite_difference(self, simple_mag_coeffs):
        """
        Verify that ar_mix (the Av-Rv cross-term in the precision matrix)
        matches a finite-difference numerical Hessian of the log-likelihood
        evaluated in the full (scale, Av, Rv) space at the MLE point.

        The code computes a Gauss-Newton approximation to the Hessian.
        We compare against finite-difference derivatives of the full
        log-likelihood (without profiling out scale) at the MLE.
        """
        av_true, rv_true, scale_true = 1.0, 3.3, 1.0
        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs, av_true, rv_true, scale_true, snr=500.0
        )

        av_arr = np.array([av_true])
        rv_arr = np.array([rv_true])
        # Use very flat priors so the prior contribution is negligible
        av_gauss = (0.0, 1e6)
        rv_gauss = (3.32, 1e6)

        _, _, _, scale_ml, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            simple_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
            av_gauss=av_gauss,
            rv_gauss=rv_gauss,
        )

        # The analytic ar_mix from icov_sar
        ar_mix_analytic = icov_sar[0, 1, 2]
        s_ml = scale_ml[0]

        # Full log-likelihood as a function of (scale, Av, Rv) -- no profiling
        def logL(s, av_val, rv_val):
            av_a = np.array([av_val])
            rv_a = np.array([rv_val])
            seds_f, _, _ = _get_seds(simple_mag_coeffs, av_a, rv_a, return_flux=True)
            model = s * seds_f[0]
            residual = data - model
            return -0.5 * np.sum(residual**2 / tot_var)

        # Finite-difference Hessian for d^2L / dAv dRv (indices 1, 2)
        h_av = 1e-4 * max(abs(av_true), 0.01)
        h_rv = 1e-4 * max(abs(rv_true), 0.01)

        f_pp = logL(s_ml, av_true + h_av, rv_true + h_rv)
        f_pm = logL(s_ml, av_true + h_av, rv_true - h_rv)
        f_mp = logL(s_ml, av_true - h_av, rv_true + h_rv)
        f_mm = logL(s_ml, av_true - h_av, rv_true - h_rv)
        d2L_dAv_dRv = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_av * h_rv)

        assert abs(ar_mix_analytic) > 0, "ar_mix should be non-zero for non-zero Av"
        # Gauss-Newton and exact Hessian agree well for high-SNR data
        npt.assert_allclose(-d2L_dAv_dRv, ar_mix_analytic, rtol=0.15)

    def test_all_fisher_elements_match_finite_differences(self, simple_mag_coeffs):
        """
        Verify all 6 unique elements of the 3x3 symmetric precision matrix
        against numerical second derivatives of the (profiled) log-likelihood.
        """
        av_true, rv_true, scale_true = 1.0, 3.3, 1.0
        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs, av_true, rv_true, scale_true, snr=500.0
        )

        av_arr = np.array([av_true])
        rv_arr = np.array([rv_true])
        # Use very flat priors to isolate the data term
        av_gauss = (0.0, 1e6)
        rv_gauss = (3.32, 1e6)

        _, _, _, scale_ml, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            simple_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
            av_gauss=av_gauss,
            rv_gauss=rv_gauss,
        )
        s_ml = scale_ml[0]

        # Build a log-likelihood as a function of (scale, Av, Rv)
        def logL(s, av_val, rv_val):
            av_a = np.array([av_val])
            rv_a = np.array([rv_val])
            seds_f, _, _ = _get_seds(simple_mag_coeffs, av_a, rv_a, return_flux=True)
            model = s * seds_f[0]
            residual = data - model
            return -0.5 * np.sum(residual**2 / tot_var)

        params = np.array([s_ml, av_true, rv_true])
        h_factors = np.array([1e-4, 1e-4, 1e-4])
        h = h_factors * np.maximum(np.abs(params), np.array([1e-3, 0.01, 0.01]))

        # Compute full numerical Hessian
        hessian_num = np.zeros((3, 3))
        for a in range(3):
            for b in range(a, 3):
                p_pp = params.copy()
                p_pp[a] += h[a]
                p_pp[b] += h[b]
                p_pm = params.copy()
                p_pm[a] += h[a]
                p_pm[b] -= h[b]
                p_mp = params.copy()
                p_mp[a] -= h[a]
                p_mp[b] += h[b]
                p_mm = params.copy()
                p_mm[a] -= h[a]
                p_mm[b] -= h[b]
                d2 = (logL(*p_pp) - logL(*p_pm) - logL(*p_mp) + logL(*p_mm)) / (
                    4 * h[a] * h[b]
                )
                hessian_num[a, b] = d2
                hessian_num[b, a] = d2

        # The precision matrix is the negative Hessian (GN approximation).
        # With flat priors the prior terms are negligible.
        precision_num = -hessian_num
        precision_code = icov_sar[0]

        # Diagonal elements (these should match well)
        for idx in range(3):
            npt.assert_allclose(
                precision_code[idx, idx],
                precision_num[idx, idx],
                rtol=0.15,
                err_msg=f"Diagonal element [{idx},{idx}] mismatch",
            )

        # Off-diagonal elements (larger tolerance for GN vs exact)
        for a in range(3):
            for b in range(a + 1, 3):
                if abs(precision_num[a, b]) > 1e-10:
                    npt.assert_allclose(
                        precision_code[a, b],
                        precision_num[a, b],
                        rtol=0.3,
                        err_msg=f"Off-diagonal [{a},{b}] mismatch",
                    )

    @pytest.mark.parametrize(
        "av_val, rv_val",
        [
            (0.01, 3.3),  # Very low Av
            (5.0, 3.3),  # High Av
            (1.0, 1.5),  # Low Rv
            (1.0, 7.0),  # High Rv
            (0.0, 3.3),  # Zero Av (edge case)
        ],
    )
    def test_precision_matrix_positive_semidefinite(
        self, simple_mag_coeffs, av_val, rv_val
    ):
        """
        Verify the precision matrix is PSD for various (Av, Rv) combinations,
        including edge cases.
        """
        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs, av_val, rv_val, scale_true=1.0, snr=100.0
        )
        av_arr = np.array([av_val])
        rv_arr = np.array([rv_val])

        _, _, _, _, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            simple_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
        )

        P = icov_sar[0]
        # Should be symmetric
        npt.assert_allclose(P, P.T, atol=1e-10)
        # Should be PSD
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(
            eigvals >= -1e-10
        ), f"Precision matrix not PSD: eigenvalues = {eigvals}"

    def test_precision_structure_s_den_uses_unscaled_models(self, simple_mag_coeffs):
        """
        Verify that s_den (the scale-scale element of the precision matrix)
        uses the *unscaled* models: s_den = sum(model_j^2 / var_j).
        """
        av_true, rv_true = 0.5, 3.3
        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs, av_true, rv_true, scale_true=1.0, snr=200.0
        )
        av_arr = np.array([av_true])
        rv_arr = np.array([rv_true])

        # Run _get_sed_mle (which scales models internally)
        _, _, _, scale_ml, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            simple_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
            av_gauss=(0.0, 1e6),
            rv_gauss=(3.32, 0.18),
        )

        # Recompute s_den manually from unscaled flux models
        seds_flux, _, _ = _get_seds(simple_mag_coeffs, av_arr, rv_arr, return_flux=True)
        inv_var = 1.0 / tot_var
        s_den_manual = np.sum(seds_flux[0] ** 2 * inv_var)

        s_den_code = icov_sar[0, 0, 0]
        npt.assert_allclose(s_den_code, s_den_manual, rtol=1e-10)

    def test_sa_mix_uses_scaled_models(self, simple_mag_coeffs):
        """
        Verify that sa_mix (the scale-Av cross-term) uses scaled models:
        sa_mix = sum(scale*model_j * rvec_j / var_j).
        """
        av_true, rv_true = 1.0, 3.3
        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs, av_true, rv_true, scale_true=1.0, snr=200.0
        )
        av_arr = np.array([av_true])
        rv_arr = np.array([rv_true])

        models_out, rvecs_out, _, scale_ml, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            simple_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
            av_gauss=(0.0, 1e6),
            rv_gauss=(3.32, 0.18),
        )

        # After _get_sed_mle, models_out are already scaled (models[i][j] *= scale[i]).
        # sa_mix is computed BEFORE scaling using models[i][j] (unscaled) and
        # rvecs[i][j] (unscaled). But looking at the code:
        #   sa_mix[i] += models[i][j] * rvecs[i][j] * inv_tot_var[j]
        # This is computed at lines 667 *before* the rescaling at line 670.
        # So sa_mix uses the unscaled models and unscaled rvecs.
        #
        # Actually, looking more carefully at the code:
        #   Line 657: models[i][j] = models[i][j] * scale[i]  (scales the model)
        #   Line 667: sa_mix uses models[i][j] (already scaled!)
        #   Line 670: rvecs[i][j] = rvecs[i][j] * scale[i] (scales rvec)
        #
        # So sa_mix = sum( (scale * model_unscaled) * rvec_unscaled / var )

        seds_flux, rvecs_flux, _ = _get_seds(
            simple_mag_coeffs, av_arr, rv_arr, return_flux=True
        )
        inv_var = 1.0 / tot_var
        s = scale_ml[0]

        # sa_mix uses: scaled_model * unscaled_rvec
        sa_mix_manual = np.sum(s * seds_flux[0] * rvecs_flux[0] * inv_var)

        sa_mix_code = icov_sar[0, 0, 1]
        npt.assert_allclose(sa_mix_code, sa_mix_manual, rtol=1e-10)


# ============================================================================
# Part 2 -- logpost_grid: log-distance draws and effective sample size
# ============================================================================


def _lnd_precision(icov_sar, scale):
    """Return the (ln d, Av, Rv) precision, J^-T P J^-1 with J^-1 = diag(-2s, 1, 1).

    ``ln d = -0.5 ln s``, so ``ds / d(ln d) = -2s``.
    """
    jinv = np.diag([-2.0 * scale, 1.0, 1.0])
    return jinv.T @ icov_sar @ jinv


def _sar_precision(sd_scale, sd_av, sd_rv, corr):
    """Return the (scale, Av, Rv) precision of a Gaussian with these marginals."""
    sd = np.array([sd_scale, sd_av, sd_rv])
    return np.linalg.inv(corr * np.outer(sd, sd))


def _results_tuple(lnlike, scales, avs, rvs, icovs_sar):
    """Build a `loglike_grid(..., return_vals=True)` tuple by hand."""
    lnlike = np.asarray(lnlike, dtype=float)
    return (
        lnlike,
        5,
        -2.0 * lnlike,
        np.asarray(scales, dtype=float),
        np.asarray(avs, dtype=float),
        np.asarray(rvs, dtype=float),
        np.asarray(icovs_sar, dtype=float),
    )


# Correlations of (scale, Av, Rv). Scale and A(V) are strongly correlated, as
# for a reddened star, so a sign error in the (ln d, Av) term is visible.
SAR_CORR = np.array(
    [
        [1.0, 0.9, 0.3],
        [0.9, 1.0, 0.2],
        [0.3, 0.2, 1.0],
    ]
)


@pytest.fixture(scope="module")
def logpost_fitter():
    """BruteForce on a 20-model grid with 5 bands.

    Without ``coord``, `logpost_grid` reads only its ``results`` argument, so
    the hand-built results tuples below need not match these models.
    """
    nmodels = 20
    models = np.zeros((nmodels, 5, 3))
    models[:, :, 0] = 15.0
    models[:, :, 1] = [1.5, 1.2, 1.0, 0.8, 0.7]
    models[:, :, 2] = [0.15, 0.12, 0.10, 0.08, 0.07]
    labels = np.zeros(nmodels, dtype=[("mini", "f8"), ("eep", "f8"), ("feh", "f8")])
    labels["mini"] = np.linspace(0.5, 2.0, nmodels)
    grid = StarGrid(models, labels, filters=["g", "r", "i", "z", "y"], verbose=False)
    return BruteForce(grid, verbose=False)


class TestLogpostGridDraws:
    """Test the (ln d, Av, Rv) draws and mc_ess that logpost_grid returns."""

    scales = np.array([0.25, 4.0, 1e-2])  # distances of 2, 0.5 and 10 kpc

    def _results(self):
        # 4% scale errors are 2% distance errors.
        icovs = [_sar_precision(0.04 * s, 0.1, 0.2, SAR_CORR) for s in self.scales]
        return _results_tuple(
            np.zeros(3), self.scales, [1.0, 2.0, 0.5], [3.3, 3.1, 3.5], icovs
        )

    def test_draws_follow_the_log_distance_gaussian_of_each_model(self, logpost_fitter):
        """The (ln d, Av, Rv) draws have mean (-ln(s)/2, Av, Rv) and
        covariance inv(J^-T P J^-1)."""
        results = self._results()
        nmc = 4000
        sel, _, _, dist_mc, a_mc, r_mc, _, _ = logpost_fitter.logpost_grid(
            results, Nmc_prior=nmc, rstate=np.random.RandomState(118)
        )
        npt.assert_array_equal(sel, np.arange(3))
        icovs = results[-1]
        for i in range(3):
            draws = np.vstack([np.log(dist_mc[i]), a_mc[i], r_mc[i]])
            mean = [-0.5 * np.log(self.scales[i]), results[4][i], results[5][i]]
            # Antithetic pairs (z, -z) make the sample mean exact.
            npt.assert_allclose(draws.mean(axis=1), mean, rtol=0, atol=1e-12)

            cov = np.linalg.inv(_lnd_precision(icovs[i], self.scales[i]))
            # The standard error of a sample covariance is
            # sqrt((S_ii S_jj + S_ij^2) / n). A pair (z, -z) contributes the
            # same (x - mean)(x - mean)^T twice, so n = nmc / 2 pairs.
            var = np.diag(cov)
            stderr = np.sqrt((np.outer(var, var) + cov**2) / (nmc / 2))
            npt.assert_array_less(np.abs(np.cov(draws) - cov), 5 * stderr)

    def test_mc_ess_is_inverse_sum_of_squared_normalized_weights(self, logpost_fitter):
        """mc_ess equals 1 / sum(w**2) of the normalized lnp_mc weights."""
        nmc = 200
        out = logpost_fitter.logpost_grid(
            self._results(),
            parallax=1.0,
            parallax_err=0.2,
            Nmc_prior=nmc,
            rstate=np.random.RandomState(7),
        )
        lnp_mc, mc_ess = out[6], out[7]
        weights = np.exp(lnp_mc - lnp_mc.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        expected = 1.0 / np.sum(weights**2, axis=1)
        # The parallax and the distance Jacobian make the weights unequal.
        assert np.all(expected < nmc)
        npt.assert_allclose(mc_ess, expected, rtol=1e-12)


class TestLogDistanceReparameterization:
    """Test the inversion of the (ln d, Av, Rv) precision matrix."""

    @pytest.mark.parametrize(
        "s, label",
        [
            (1e-5, "very distant star"),
            (100.0, "very nearby star"),
            (1.0, "moderate distance"),
        ],
    )
    def test_transform_invert_gives_finite_psd_covariance(self, s, label):
        """
        Transform -> invert gives well-behaved covariance for extreme scales.
        """
        # Build a plausible precision matrix
        icov_sar = np.array(
            [
                [1e4, 50.0, 5.0],
                [50.0, 200.0, 3.0],
                [5.0, 3.0, 31.0],
            ]
        )

        icov_lnd = _lnd_precision(icov_sar, s)
        # Use regularized inversion (as the real code does)
        cov_lnd = inverse3(icov_lnd, regularize=True)

        assert np.all(
            np.isfinite(cov_lnd)
        ), f"Covariance has non-finite values for s={s} ({label})"
        assert isPSD(cov_lnd), f"Covariance is not PSD for s={s} ({label})"


# ============================================================================
# Part 3 -- Diagonal preconditioning (inverse3 with regularize=True)
# ============================================================================


class TestDiagonalPreconditioning:
    """Test inverse3 with regularize=True."""

    def test_condition_number_reduction(self):
        """
        Create a matrix with condition number ~30,000 (realistic for
        scale/Av/Rv precision) and verify preconditioning helps produce
        a valid inverse.
        """
        # Construct a precision matrix with large condition number.
        # s_den ~ 1e5, a_den ~ 100, r_den ~ 31
        P = np.array(
            [
                [1e5, 500.0, 50.0],
                [500.0, 100.0, 5.0],
                [50.0, 5.0, 31.0],
            ]
        )
        cond_original = np.linalg.cond(P)
        assert (
            cond_original > 1000
        ), f"Test setup: condition number {cond_original} should be > 1000"

        # The preconditioned inverse should be finite and PSD
        C = inverse3(P, regularize=True)
        assert np.all(np.isfinite(C)), "Inverse should be finite"
        assert isPSD(C), "Inverse should be PSD"

        # It should approximate np.linalg.inv reasonably
        C_ref = np.linalg.inv(P)
        npt.assert_allclose(C, C_ref, rtol=1e-3)

    def test_preserves_correct_inverse_well_conditioned(self):
        """
        For a well-conditioned matrix, inverse3(A, regularize=True) should
        give the same result as np.linalg.inv(A).
        """
        A = np.array(
            [
                [10.0, 1.0, 0.5],
                [1.0, 8.0, 0.3],
                [0.5, 0.3, 6.0],
            ]
        )
        C_reg = inverse3(A, regularize=True)
        C_ref = np.linalg.inv(A)
        npt.assert_allclose(C_reg, C_ref, rtol=1e-6)

    def test_handles_near_singular_matrix(self):
        """
        Create a rank-2 matrix (one zero eigenvalue). Verify the result
        is finite and PSD.
        """
        # Build a rank-2 matrix from two outer products
        v1 = np.array([1.0, 0.5, 0.2])
        v2 = np.array([0.3, 1.0, 0.4])
        P = 100 * np.outer(v1, v1) + 50 * np.outer(v2, v2)

        # Verify it is indeed near-singular
        eigvals = np.linalg.eigvalsh(P)
        assert (
            min(abs(eigvals)) < 1e-10
        ), "Test setup: matrix should have a near-zero eigenvalue"

        C = inverse3(P, regularize=True)
        assert np.all(
            np.isfinite(C)
        ), "Inverse of near-singular matrix should be finite with regularization"
        assert isPSD(
            C
        ), "Inverse of near-singular matrix should be PSD with regularization"

    def test_different_parameter_scales(self):
        """
        Create a precision matrix with diagonal elements spanning 5 orders
        of magnitude (like s_den=1e5, a_den=100, r_den=31). Verify the
        inverse is accurate.
        """
        P = np.array(
            [
                [1e5, 200.0, 20.0],
                [200.0, 100.0, 3.0],
                [20.0, 3.0, 31.0],
            ]
        )

        C = inverse3(P, regularize=True)
        C_ref = np.linalg.inv(P)

        # Check each element
        for i in range(3):
            for j in range(3):
                if abs(C_ref[i, j]) > 1e-15:
                    npt.assert_allclose(
                        C[i, j],
                        C_ref[i, j],
                        rtol=1e-3,
                        err_msg=f"Element [{i},{j}] mismatch",
                    )

    def test_batch_inverse_with_regularization(self):
        """
        Verify batch (N, 3, 3) inversion with regularize=True produces
        correct results for each matrix.
        """
        N = 4
        A_batch = np.zeros((N, 3, 3))
        for i in range(N):
            diag = np.array([10.0 * (i + 1), 5.0 * (i + 1), 3.0 * (i + 1)])
            off = 0.5 * (i + 1)
            A_batch[i] = np.diag(diag)
            A_batch[i, 0, 1] = A_batch[i, 1, 0] = off
            A_batch[i, 0, 2] = A_batch[i, 2, 0] = off * 0.3
            A_batch[i, 1, 2] = A_batch[i, 2, 1] = off * 0.2

        C_batch = inverse3(A_batch, regularize=True)
        assert C_batch.shape == (N, 3, 3)

        for i in range(N):
            C_ref = np.linalg.inv(A_batch[i])
            npt.assert_allclose(C_batch[i], C_ref, rtol=1e-5)


# ============================================================================
# Part 4 -- Integration: _get_sed_mle with multiple models
# ============================================================================


class TestGetSedMleMultiModel:
    """Integration tests using multiple models simultaneously."""

    def test_returns_correct_shapes(self, multi_model_mag_coeffs):
        """Verify output shapes of _get_sed_mle."""
        Nmodels = multi_model_mag_coeffs.shape[0]
        Nbands = multi_model_mag_coeffs.shape[1]

        av_true, rv_true = 0.5, 3.3
        data, tot_var, resid = _make_synthetic_data(
            multi_model_mag_coeffs, av_true, rv_true, scale_true=1.0, snr=100.0
        )

        av_arr = np.full(Nmodels, av_true)
        rv_arr = np.full(Nmodels, rv_true)

        models, rvecs, drvecs, scale, icov_sar, resid_out = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            multi_model_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
        )

        assert models.shape == (Nmodels, Nbands)
        assert rvecs.shape == (Nmodels, Nbands)
        assert drvecs.shape == (Nmodels, Nbands)
        assert scale.shape == (Nmodels,)
        assert icov_sar.shape == (Nmodels, 3, 3)
        assert resid_out.shape == (Nmodels, Nbands)

    def test_all_precision_matrices_psd(self, multi_model_mag_coeffs):
        """All precision matrices from a multi-model run should be PSD."""
        Nmodels = multi_model_mag_coeffs.shape[0]
        av_true, rv_true = 1.0, 3.3
        data, tot_var, resid = _make_synthetic_data(
            multi_model_mag_coeffs, av_true, rv_true, scale_true=1.0, snr=100.0
        )

        av_arr = np.full(Nmodels, av_true)
        rv_arr = np.full(Nmodels, rv_true)

        _, _, _, _, icov_sar, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            multi_model_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
        )

        for i in range(Nmodels):
            P = icov_sar[i]
            npt.assert_allclose(
                P, P.T, atol=1e-10, err_msg=f"icov_sar[{i}] not symmetric"
            )
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(
                eigvals >= -1e-10
            ), f"icov_sar[{i}] not PSD: eigenvalues = {eigvals}"

    def test_scale_positive(self, multi_model_mag_coeffs):
        """All MLE scale factors should be positive."""
        Nmodels = multi_model_mag_coeffs.shape[0]
        data, tot_var, resid = _make_synthetic_data(
            multi_model_mag_coeffs, 0.5, 3.3, scale_true=1.0, snr=100.0
        )

        av_arr = np.full(Nmodels, 0.5)
        rv_arr = np.full(Nmodels, 3.3)

        _, _, _, scale, _, _ = _get_sed_mle(
            data,
            tot_var,
            resid.copy(),
            multi_model_mag_coeffs,
            av_arr.copy(),
            rv_arr.copy(),
        )

        assert np.all(scale > 0), "All scale factors should be positive"


# ============================================================================
# Part 5 -- Convergence test for _optimize_fit_mag (M20)
# ============================================================================


class TestOptimizeFitMagConvergence:
    """Test that _optimize_fit_mag recovers known synthetic star parameters."""

    def test_recovers_known_synthetic_star(self, simple_mag_coeffs):
        """
        Create a known synthetic star (model + noise), run the full
        loglike_grid pipeline, and verify the optimized (scale, Av, Rv)
        are close to the true values.
        """
        av_true, rv_true, scale_true = 0.8, 3.3, 1.0
        snr = 200.0

        data, tot_var, resid = _make_synthetic_data(
            simple_mag_coeffs,
            av_true,
            rv_true,
            scale_true,
            snr=snr,
        )

        av_arr = np.array([av_true * 0.5])  # Start away from truth
        rv_arr = np.array([3.0])  # Start away from truth

        from brutus.analysis.individual import _optimize_fit_mag
        from brutus.core.sed_utils import _get_seds
        from brutus.utils.photometry import magnitude

        Nmodels, Nbands = simple_mag_coeffs.shape[0], simple_mag_coeffs.shape[1]

        # Compute initial model predictions in magnitudes
        models_init, rvecs_init, drvecs_init = _get_seds(
            simple_mag_coeffs,
            av_arr,
            rv_arr,
            return_flux=False,
        )

        # Convert data to magnitudes
        with np.errstate(divide="ignore", invalid="ignore"):
            mags = -2.5 * np.log10(data)
            mags_var = (2.5 / np.log(10.0)) ** 2 * tot_var / data**2

        resid_init = mags - models_init
        stepsize = np.ones(Nmodels)

        results = _optimize_fit_mag(
            data,
            tot_var,
            models_init,
            rvecs_init,
            drvecs_init,
            av_arr.copy(),
            rv_arr.copy(),
            simple_mag_coeffs,
            resid_init,
            stepsize,
            mags,
            mags_var,
            avlim=(0.0, 20.0),
            av_gauss=(0.0, 1e6),
            rvlim=(1.0, 8.0),
            rv_gauss=(3.32, 0.18),
        )
        (
            models_out,
            rvecs_out,
            drvecs_out,
            scale_opt,
            av_opt,
            rv_opt,
            icov_sar,
            resid_out,
        ) = results

        # The optimized values should be close to the true values.
        npt.assert_allclose(
            scale_opt[0],
            scale_true,
            rtol=0.1,
            err_msg="Scale should converge near true value",
        )
        npt.assert_allclose(
            av_opt[0], av_true, atol=0.3, err_msg="Av should converge near true value"
        )
        npt.assert_allclose(
            rv_opt[0], rv_true, atol=0.5, err_msg="Rv should converge near true value"
        )


# ============================================================================
# Part 6 -- Jitted helpers of loglike_grid against numpy
# ============================================================================


class TestJittedLikelihoodHelpers:
    """Compare the jitted loglike_grid helpers with direct numpy."""

    def test_init_mag_resid_matches_numpy(self):
        """_init_mag_resid returns the reddened SEDs, R(V)-dependent
        reddening vectors, their R(V) derivatives and mags - SED."""
        rng = np.random.RandomState(166)
        mag_coeffs = rng.normal(size=(7, 5, 3))
        av = rng.uniform(0.0, 3.0, 7)
        rv = rng.uniform(2.0, 5.0, 7)
        mags = rng.normal(15.0, 1.0, 5)

        models, rvecs, drvecs, resid = _init_mag_resid(mag_coeffs, av, rv, mags)

        expected_rvecs = mag_coeffs[:, :, 1] + rv[:, None] * mag_coeffs[:, :, 2]
        expected_models = mag_coeffs[:, :, 0] + av[:, None] * expected_rvecs
        npt.assert_array_equal(drvecs, mag_coeffs[:, :, 2])
        npt.assert_array_equal(rvecs, expected_rvecs)
        npt.assert_array_equal(models, expected_models)
        npt.assert_array_equal(resid, mags - expected_models)

    def test_chi2_from_resid_matches_numpy(self):
        """_chi2_from_resid returns sum_j resid[i, j]**2 / tot_var[j]."""
        rng = np.random.RandomState(145)
        resid = rng.normal(size=(7, 5))
        tot_var = rng.uniform(0.01, 1.0, 5)

        npt.assert_array_equal(
            _chi2_from_resid(resid, tot_var), np.sum(resid**2 / tot_var, axis=1)
        )


# ============================================================================
# Part 7 -- logpost_grid: analytic distance posterior and scale guards
# ============================================================================


class TestLogpostGridDistancePosterior:
    """Compare logpost_grid with an analytic posterior and test its guards."""

    def test_weighted_distance_quantiles_match_the_analytic_posterior(
        self, logpost_fitter
    ):
        """With flat priors in d, the weighted ln d draws follow
        N(eta0 + sigma**2, sigma**2).

        The draws are N(eta0, sigma**2) in eta = ln d. A flat prior in d is
        the weight |dd/deta| = exp(eta), and
        N(eta; eta0, sigma**2) * exp(eta) is proportional to
        N(eta; eta0 + sigma**2, sigma**2).
        """
        scale = 0.25  # 2 kpc
        icov = _sar_precision(2 * scale * 0.3, 0.1, 0.2, SAR_CORR)  # sd(ln d) = 0.3
        results = _results_tuple([0.0], [scale], [5.0], [3.3], [icov])
        _, _, _, dist_mc, _, _, lnp_mc, _ = logpost_fitter.logpost_grid(
            results, Nmc_prior=20000, rstate=np.random.RandomState(214)
        )
        weights = np.exp(lnp_mc[0] - lnp_mc[0].max())
        weights /= weights.sum()

        sigma2 = np.linalg.inv(_lnd_precision(icov, scale))[0, 0]
        sigma = np.sqrt(sigma2)
        q = np.array([0.16, 0.5, 0.84])
        z_q = scipy.stats.norm.ppf(q)
        expected = -0.5 * np.log(scale) + sigma2 + sigma * z_q
        # Standard error of a sample quantile from n independent draws:
        # sigma * sqrt(q (1 - q)) / (sqrt(n) * phi(z_q)), with n the Kish
        # size 1 / sum(w**2). Antithetic pairs anticorrelate the indicators
        # 1(eta < eta_q), so this bound is conservative. The Jacobian shifts
        # the quantiles by sigma2 = 0.09, 27 to 32 standard errors.
        n_eff = 1.0 / np.sum(weights**2)
        stderr = (
            sigma * np.sqrt(q * (1 - q)) / (np.sqrt(n_eff) * scipy.stats.norm.pdf(z_q))
        )
        found = quantile(np.log(dist_mc[0]), q, weights=weights)
        npt.assert_array_less(np.abs(found - expected), 5 * stderr)

    def test_nonpositive_scale_gives_finite_posterior(self, logpost_fitter):
        """Models with scale 0 or < 0 give finite lnp and no NaN draws."""
        scales = [0.0, -0.5, 0.25]
        icovs = [_sar_precision(0.01, 0.1, 0.2, SAR_CORR)] * 3
        results = _results_tuple(np.zeros(3), scales, [1.0] * 3, [3.3] * 3, icovs)
        sel, _, lnp, dist_mc, a_mc, r_mc, lnp_mc, mc_ess = logpost_fitter.logpost_grid(
            results, Nmc_prior=50, rstate=np.random.RandomState(1634)
        )
        npt.assert_array_equal(sel, np.arange(3))
        assert np.all(np.isfinite(lnp))
        # Every draw of the clamped models leaves the distance support.
        npt.assert_array_equal(lnp[:2], LOG_ZERO)
        npt.assert_array_equal(mc_ess[:2], 0.0)
        for draws in (dist_mc, a_mc, r_mc, lnp_mc, mc_ess):
            assert not np.any(np.isnan(draws))


# ============================================================================
# Part 8 -- logpost_grid: max_models subsampling
# ============================================================================


class TestLogpostGridSubsampling:
    """Test the max_models subsampling of the models above the threshold."""

    max_models = 5

    def _results(self):
        # 20 models with ln L = -0.5 * rank, in shuffled order. With
        # wt_thresh=1e-3, ranks 0-13 pass the cut (0.5 * 13 < ln 1e3 = 6.9).
        rank = np.random.RandomState(228).permutation(20)
        icovs = [_sar_precision(0.01, 0.1, 0.2, SAR_CORR)] * 20
        lnlike = -0.5 * rank
        results = _results_tuple(lnlike, [0.25] * 20, [1.0] * 20, [3.3] * 20, icovs)
        return results, rank

    def _logpost(self, fitter, results, mode):
        return fitter.logpost_grid(
            results,
            Nmc_prior=10,
            max_models=self.max_models,
            subsample_mode=mode,
            rstate=np.random.RandomState(1581),
        )

    def test_topk_keeps_the_models_with_the_highest_log_probability(
        self, logpost_fitter
    ):
        """subsample_mode="topk" keeps the max_models best models."""
        results, rank = self._results()
        sel, _, lnp, *_ = self._logpost(logpost_fitter, results, "topk")
        npt.assert_array_equal(np.sort(rank[sel]), np.arange(self.max_models))
        assert lnp.shape == (self.max_models,)

    def test_representative_keeps_max_models_distinct_models_above_threshold(
        self, logpost_fitter
    ):
        """subsample_mode="representative" keeps max_models distinct models,
        all from the models that pass the threshold."""
        results, rank = self._results()
        sel, _, lnp, *_ = self._logpost(logpost_fitter, results, "representative")
        assert np.unique(sel).size == self.max_models
        assert np.all(rank[sel] <= 13)
        assert lnp.shape == (self.max_models,)

    def test_unknown_subsample_mode_raises(self, logpost_fitter):
        """An unknown subsample_mode raises ValueError naming the argument."""
        results, _ = self._results()
        with pytest.raises(ValueError, match="subsample_mode"):
            self._logpost(logpost_fitter, results, "random")
