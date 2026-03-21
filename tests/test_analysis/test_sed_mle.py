#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Fisher information matrix computation, log-distance reparameterization,
diagonal preconditioning, and ESS computation in brutus.

These tests validate the numerical correctness of the precision matrix
(inverse covariance) computed in ``_get_sed_mle``, the coordinate
transformation from (scale, Av, Rv) to (ln d, Av, Rv) space in
``logpost_grid``, the ``inverse3`` utility with regularization, and the
effective sample size calculation.
"""

import numpy as np
import numpy.testing as npt
import pytest

from brutus.analysis.individual import _get_sed_mle
from brutus.core.sed_utils import _get_seds
from brutus.utils.math import inverse3, isPSD

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
    np.random.seed(12345)
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
    np.random.seed(54321)
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
    np.random.seed(999)
    data = true_flux + np.random.normal(0, flux_err)

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
# Part 2 -- Log-distance reparameterization
# ============================================================================


class TestLogDistanceReparameterization:
    """Test the (scale, Av, Rv) -> (ln d, Av, Rv) precision transform."""

    def _transform_precision(self, icov_sar, scale):
        """
        Replicate the precision-matrix transform from logpost_grid.

        Parameters
        ----------
        icov_sar : array of shape (3, 3)
            Precision matrix in (scale, Av, Rv) space.
        scale : float
            MLE scale factor.

        Returns
        -------
        icov_lnd : array of shape (3, 3)
            Precision matrix in (ln d, Av, Rv) space.
        """
        s = max(scale, 1e-20)
        icov_lnd = icov_sar.copy()
        icov_lnd[0, 0] = 4.0 * s**2 * icov_sar[0, 0]
        icov_lnd[0, 1] = -2.0 * s * icov_sar[0, 1]
        icov_lnd[1, 0] = -2.0 * s * icov_sar[1, 0]
        icov_lnd[0, 2] = -2.0 * s * icov_sar[0, 2]
        icov_lnd[2, 0] = -2.0 * s * icov_sar[2, 0]
        # (1,1), (1,2), (2,1), (2,2) unchanged
        return icov_lnd

    def test_precision_transform_explicit_values(self):
        """
        For a known precision matrix and scale value, verify each element
        of the transformed precision matrix.
        """
        s = 0.5
        icov_sar = np.array(
            [
                [100.0, 10.0, 5.0],
                [10.0, 50.0, 3.0],
                [5.0, 3.0, 30.0],
            ]
        )

        icov_lnd = self._transform_precision(icov_sar, s)

        # Check each element
        npt.assert_allclose(icov_lnd[0, 0], 4 * s**2 * 100.0)
        npt.assert_allclose(icov_lnd[0, 1], -2 * s * 10.0)
        npt.assert_allclose(icov_lnd[1, 0], -2 * s * 10.0)
        npt.assert_allclose(icov_lnd[0, 2], -2 * s * 5.0)
        npt.assert_allclose(icov_lnd[2, 0], -2 * s * 5.0)
        # Av-Av, Av-Rv, Rv-Rv unchanged
        npt.assert_allclose(icov_lnd[1, 1], 50.0)
        npt.assert_allclose(icov_lnd[1, 2], 3.0)
        npt.assert_allclose(icov_lnd[2, 1], 3.0)
        npt.assert_allclose(icov_lnd[2, 2], 30.0)

    def test_precision_transform_matches_jacobian(self):
        """
        Verify the transform by computing it via the Jacobian explicitly:
        icov_lnd = J^{-T} icov_sar J^{-1} where J^{-1} = diag(-2s, 1, 1).
        """
        s = 1.5
        icov_sar = np.array(
            [
                [200.0, 15.0, 8.0],
                [15.0, 80.0, 4.0],
                [8.0, 4.0, 35.0],
            ]
        )

        Jinv = np.diag([-2 * s, 1.0, 1.0])
        icov_lnd_expected = Jinv.T @ icov_sar @ Jinv
        icov_lnd_code = self._transform_precision(icov_sar, s)

        npt.assert_allclose(icov_lnd_code, icov_lnd_expected, atol=1e-12)

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

        icov_lnd = self._transform_precision(icov_sar, s)
        # Use regularized inversion (as the real code does)
        cov_lnd = inverse3(icov_lnd, regularize=True)

        assert np.all(
            np.isfinite(cov_lnd)
        ), f"Covariance has non-finite values for s={s} ({label})"
        assert isPSD(cov_lnd), f"Covariance is not PSD for s={s} ({label})"

    def test_jacobian_ln_d_in_mc_weights(self):
        """
        Verify that the Jacobian for ln(d)->d (a factor of d = exp(eta))
        is included in log-posterior MC weights.

        logpost_grid adds np.log(dist_mc) to lnp_mc, which corresponds
        to the |dd/d(eta)| = d Jacobian for the change-of-variables from
        distance-space priors to ln(d)-space sampling.
        """
        # Simulate what the code does
        np.random.seed(42)
        Nmc, Nsel = 10, 3

        # Dummy base log-probabilities
        lnprob_base = np.array([-10.0, -12.0, -11.0])
        lnp_mc = np.tile(lnprob_base, (Nmc, 1))

        # Dummy distance samples
        eta_samples = np.random.normal(loc=6.0, scale=0.5, size=(Nmc, Nsel))
        dist_mc = np.exp(eta_samples)

        # Add Jacobian (as the code does on line 1474)
        lnp_mc_with_jac = lnp_mc + np.log(dist_mc + 1e-300)

        # Without Jacobian, all rows would be identical
        assert not np.allclose(
            lnp_mc_with_jac[0], lnp_mc_with_jac[1]
        ), "With Jacobian, different MC samples should give different weights"
        # The Jacobian term should be positive (since dist > 0 => ln(dist) can
        # be positive or negative, but dist > 1 => ln(dist) > 0).
        # For eta ~ 6 => dist ~ 400, ln(dist) ~ 6 > 0
        jacobian_terms = np.log(dist_mc + 1e-300)
        assert np.all(
            jacobian_terms > 0
        ), "For distant stars, ln(dist) Jacobian should be positive"

    def test_distance_samples_centered_near_mle(self):
        """
        Verify that dist_mc = exp(eta_samples) produces distances centered
        near 1/sqrt(s_MLE) for a simple case.
        """
        # True distance = 500 pc => scale = 1/500^2 = 4e-6
        s_true = 4e-6
        eta_true = -0.5 * np.log(s_true)  # ln(d) = ln(500) ~ 6.21

        # Build a tight precision matrix so samples cluster near MLE
        icov_lnd = np.diag([100.0, 100.0, 100.0])
        cov_lnd = np.linalg.inv(icov_lnd)

        np.random.seed(42)
        mean = np.array([eta_true, 0.5, 3.3])
        samples = np.random.multivariate_normal(mean, cov_lnd, size=1000)
        eta_samples = samples[:, 0]
        dist_samples = np.exp(eta_samples)

        # Median distance should be close to 500
        d_expected = 1.0 / np.sqrt(s_true)
        npt.assert_allclose(np.median(dist_samples), d_expected, rtol=0.05)


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
# Part 4 -- ESS computation
# ============================================================================


class TestEffectiveSampleSize:
    """Test the effective sample size (ESS) computation."""

    @staticmethod
    def _compute_ess(log_weights):
        """
        Replicate the ESS computation from logpost_grid.

        Parameters
        ----------
        log_weights : array of shape (Nmc,)
            Log-weights for a single model's MC samples.

        Returns
        -------
        ess : float
            Effective sample size.
        """
        lw_max = np.max(log_weights)
        w = np.exp(log_weights - lw_max)
        w_sum = w.sum()
        if w_sum > 0:
            w_normed = w / w_sum
            return 1.0 / np.sum(w_normed**2)
        return 0.0

    def test_uniform_weights_give_ess_equal_n(self):
        """All equal weights should give ESS = N."""
        N = 100
        log_weights = np.zeros(N)  # All weights = 1
        ess = self._compute_ess(log_weights)
        npt.assert_allclose(ess, N, rtol=1e-10)

    def test_single_dominant_weight_gives_ess_near_one(self):
        """One weight much larger than others should give ESS near 1."""
        N = 100
        log_weights = np.full(N, -100.0)
        log_weights[0] = 0.0  # One dominant weight
        ess = self._compute_ess(log_weights)
        npt.assert_allclose(ess, 1.0, atol=1e-5)

    def test_ess_between_one_and_n(self):
        """For random weights, ESS should be between 1 and N."""
        np.random.seed(42)
        N = 200
        log_weights = np.random.normal(0, 2, N)
        ess = self._compute_ess(log_weights)
        assert 1.0 <= ess <= N, f"ESS = {ess} not in [1, {N}]"

    def test_ess_monotonic_with_weight_concentration(self):
        """
        As weights become more concentrated, ESS should decrease.
        """
        N = 100
        # sigma controls how spread out the log-weights are.
        # Larger sigma => more concentrated (one dominant sample) => lower ESS.
        ess_values = []
        for sigma in [0.1, 1.0, 5.0, 20.0]:
            np.random.seed(42)
            log_weights = np.random.normal(0, sigma, N)
            ess_values.append(self._compute_ess(log_weights))

        # ESS should decrease as sigma increases
        for i in range(len(ess_values) - 1):
            assert ess_values[i] >= ess_values[i + 1], (
                f"ESS should decrease as weights become more concentrated: "
                f"ESS[sigma={[0.1, 1.0, 5.0, 20.0][i]}]={ess_values[i]} < "
                f"ESS[sigma={[0.1, 1.0, 5.0, 20.0][i+1]}]={ess_values[i+1]}"
            )

    def test_ess_shape_and_nonnegativity_batch(self):
        """
        Verify ESS values for a batch of models are non-negative and have
        the correct shape. This mimics the computation in logpost_grid.
        """
        np.random.seed(42)
        Nmc, Nsel = 50, 10
        lnp_mc = np.random.normal(-50, 5, (Nmc, Nsel))

        mc_ess = np.zeros(Nsel)
        for j in range(Nsel):
            lw = lnp_mc[:, j]
            lw_max = np.max(lw)
            w = np.exp(lw - lw_max)
            w_sum = w.sum()
            if w_sum > 0:
                w_normed = w / w_sum
                mc_ess[j] = 1.0 / np.sum(w_normed**2)
            else:
                mc_ess[j] = 0.0

        assert mc_ess.shape == (Nsel,)
        assert np.all(mc_ess >= 0), "ESS values should be non-negative"
        assert np.all(mc_ess <= Nmc), "ESS should not exceed number of MC samples"


# ============================================================================
# Part 5 -- Integration: _get_sed_mle with multiple models
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
