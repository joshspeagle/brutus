#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression and equivalence tests for the audited fixes in
brutus.analysis.offsets:

1. Vectorized bootstrap (inverted-CDF model draws): distribution-preserving
   equivalence against the legacy per-object rng.choice implementation.
2. Leave-one-out importance reweighting (w = L_-i / L_full): one-pass
   equivalence against phot_loglike, plus injected-offset recovery for both
   fitted and unfitted bands.
3. Input handling: masked NaN/zero placeholders, float masks, non-positive
   fluxes, degenerate weights, legacy band-count thresholds, priors for
   bands without data, chunked SED generation.
"""

import numpy as np
import pytest

from brutus.analysis.offsets import (
    PhotometricOffsetsConfig,
    _generate_seds,
    _loo_log_weights,
    _validate_inputs,
    _vectorized_bootstrap_median,
    photometric_offsets,
)
from brutus.core.sed_utils import get_seds
from brutus.utils.photometry import phot_loglike


def _reference_bootstrap_median(ratios, weights, obj_weights, n_bootstrap, rng):
    """Verbatim pre-fix implementation (per-object rng.choice draws)."""
    n_objects = len(ratios)
    bootstrap_medians = np.zeros(n_bootstrap)
    obj_indices = rng.choice(n_objects, size=(n_bootstrap, n_objects), p=obj_weights)
    for i in range(n_bootstrap):
        selected_ratios = ratios[obj_indices[i]]
        selected_weights = weights[obj_indices[i]]
        model_indices = np.array(
            [rng.choice(len(w), p=w) if np.sum(w) > 0 else 0 for w in selected_weights]
        )
        final_ratios = selected_ratios[np.arange(n_objects), model_indices]
        bootstrap_medians[i] = np.median(final_ratios)
    return bootstrap_medians


def _softmax_rows(lnw):
    m = np.max(lnw, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    w = np.exp(lnw - m)
    return w / w.sum(axis=1, keepdims=True)


class TestBootstrapVectorization:
    """The rewritten bootstrap must preserve the sampling distribution."""

    def test_inverted_cdf_matches_choice_probabilities(self):
        """Per-row inverted-CDF sampling reproduces rng.choice(p=w) exactly
        in distribution: frequencies match the weights, zero-weight entries
        are never drawn, and the comparison count equals searchsorted."""
        w = np.array([0.1, 0.2, 0.0, 0.45, 0.25])
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        rng = np.random.default_rng(7)
        n_draws = 200_000
        u = rng.random(n_draws)

        idx_count = np.sum(cdf[None, :] < u[:, None], axis=1)
        idx_searchsorted = np.searchsorted(cdf, u, side="left")
        assert np.array_equal(idx_count, idx_searchsorted)

        freqs = np.bincount(idx_count, minlength=len(w)) / n_draws
        # exact: zero-probability category never sampled
        assert freqs[2] == 0.0
        # empirical frequencies match the target probabilities within 5 SE
        se = np.sqrt(w * (1.0 - w) / n_draws)
        assert np.all(np.abs(freqs - w) < 5.0 * se + 1e-12)

        # and match rng.choice's own empirical distribution
        idx_choice = np.random.default_rng(11).choice(len(w), size=n_draws, p=w)
        freqs_choice = np.bincount(idx_choice, minlength=len(w)) / n_draws
        assert np.all(np.abs(freqs - freqs_choice) < 5.0 * np.sqrt(2.0) * se + 1e-12)

    def test_bootstrap_distribution_matches_reference(self):
        """Old and new implementations sample the same bootstrap-median
        distribution (different random streams): with a fixed seed, medians
        and IQRs over 2000 bootstraps agree within Monte-Carlo error."""
        rng_data = np.random.default_rng(3)
        n_objects, nsamps = 400, 25
        ratios = rng_data.lognormal(0.0, 0.15, (n_objects, nsamps))
        weights = rng_data.random((n_objects, nsamps)) ** 3
        weights[5] = 0.0  # zero-weight row exercises the sample-0 fallback
        sums = weights.sum(axis=1)
        nz = sums > 0
        weights[nz] /= sums[nz, None]
        obj_weights = np.full(n_objects, 1.0 / n_objects)

        n_bootstrap = 2000
        ref = _reference_bootstrap_median(
            ratios, weights, obj_weights, n_bootstrap, np.random.default_rng(42)
        )
        new = _vectorized_bootstrap_median(
            ratios, weights, obj_weights, n_bootstrap, np.random.default_rng(42)
        )

        # location: standard error of the median ~ 1.2533 * sigma / sqrt(B)
        se_med = 1.2533 * np.std(ref) / np.sqrt(n_bootstrap)
        assert abs(np.median(ref) - np.median(new)) < 6.0 * se_med
        se_mean = np.std(ref) / np.sqrt(n_bootstrap)
        assert abs(np.mean(ref) - np.mean(new)) < 6.0 * np.sqrt(2.0) * se_mean
        # spread: IQRs agree within a loose relative tolerance
        iqr_ref = np.subtract(*np.percentile(ref, [75, 25]))
        iqr_new = np.subtract(*np.percentile(new, [75, 25]))
        assert abs(iqr_ref - iqr_new) < 0.25 * max(abs(iqr_ref), abs(iqr_new))

    def test_zero_weight_rows_fall_back_to_sample_zero(self):
        """Rows with zero total weight deterministically yield sample 0,
        matching the legacy per-object behavior."""
        ratios = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        weights = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        obj_weights = np.array([1.0, 0.0])  # always resample object 0
        medians = _vectorized_bootstrap_median(
            ratios, weights, obj_weights, 20, np.random.default_rng(0)
        )
        assert np.all(medians == 1.0)


class TestLeaveOneOutReweighting:
    """Corrected importance weights (w = L_-i / L_full) and their one-pass
    computation."""

    @pytest.mark.parametrize("dim_prior", [True, False])
    def test_loo_log_weights_match_phot_loglike_reference(self, dim_prior):
        """One-pass weights equal softmax(lnl_loo - lnl_full) computed with
        two full phot_loglike passes (the reference formulation)."""
        rng = np.random.default_rng(5)
        n, nsamps, nfilt = 40, 30, 6
        phot = rng.uniform(0.5, 2.0, (n, nfilt))
        err = rng.uniform(0.05, 0.2, (n, nfilt))
        seds = phot[:, None, :] * rng.uniform(0.8, 1.2, (n, nsamps, nfilt))
        mask = rng.random((n, nfilt)) < 0.7
        mask[:, :4] = True  # ndim >= 4 so LOO dof stays positive
        band = 2

        mask_full = mask.astype(float)
        mask_loo = mask_full.copy()
        mask_loo[:, band] = 0.0
        lnl_full = phot_loglike(
            phot, err, seds, mask=mask_full, dim_prior=dim_prior, dof_reduction=1
        )
        lnl_loo = phot_loglike(
            phot, err, seds, mask=mask_loo, dim_prior=dim_prior, dof_reduction=1
        )
        ref_weights = _softmax_rows(lnl_loo - lnl_full)

        var = err**2
        contrib = (phot[:, None, :] - seds) ** 2 / var[:, None, :] * mask[:, None, :]
        chi2_full = contrib.sum(axis=2)
        lnw = _loo_log_weights(
            chi2_full, contrib[:, :, band], mask.sum(axis=1), dim_prior
        )
        new_weights = _softmax_rows(lnw)

        assert np.allclose(new_weights, ref_weights, rtol=1e-9, atol=1e-12)

    @staticmethod
    def _make_recovery_data(delta, fit_band0, seed=17, n_obj=150, nsamps=500):
        """Synthetic single-parameter problem with a known multiplicative
        offset (1 + delta) injected into band 0.

        Models form a 1D family (overall brightness t); posterior draws per
        object are exact categorical samples from the likelihood over the
        bands used in fitting. The correct offset for band 0 is
        1 / (1 + delta): the leave-one-out (or band-0-free) posterior
        centers on the uncontaminated truth.
        """
        rng = np.random.default_rng(seed)
        nfilt = 5
        n_models = 121
        t = np.linspace(-0.3, 0.3, n_models)
        base = np.array([16.0, 15.5, 15.2, 15.0, 14.8])
        models = np.zeros((n_models, nfilt, 3))
        models[:, :, 0] = base[None, :] + t[:, None]
        models[:, :, 1] = 1.0
        models[:, :, 2] = 0.1
        fluxes = 10 ** (-0.4 * models[:, :, 0])  # at 1 kpc, A_V = 0
        f_true = 10 ** (-0.4 * base)
        err = np.tile(0.02 * f_true, (n_obj, 1))
        phot = f_true[None, :] + err * rng.standard_normal((n_obj, nfilt))
        phot[:, 0] *= 1.0 + delta

        fit_bands = np.arange(nfilt) if fit_band0 else np.arange(1, nfilt)
        idxs = np.empty((n_obj, nsamps), dtype=int)
        for k in range(n_obj):
            resid = (phot[k, fit_bands][None, :] - fluxes[:, fit_bands]) / err[
                k, fit_bands
            ][None, :]
            lnp = -0.5 * np.sum(resid**2, axis=1)
            lnp -= lnp.max()
            p = np.exp(lnp)
            p /= p.sum()
            idxs[k] = rng.choice(n_models, size=nsamps, p=p)

        mask = np.ones((n_obj, nfilt), dtype=int)
        reds = np.zeros((n_obj, nsamps))
        dreds = np.full((n_obj, nsamps), 3.1)
        dists = np.ones((n_obj, nsamps))
        return phot, err, mask, models, idxs, reds, dreds, dists

    @pytest.mark.parametrize("dim_prior,tol", [(False, 0.006), (True, 0.008)])
    def test_injected_offset_recovered_fitted_band(self, dim_prior, tol):
        """A band used in fitting: draws come from the contaminated full
        posterior, so recovery requires the corrected importance weights.

        Regression: the pre-fix weighting (w = L_-i) removes only ~2/3 of
        the circularity and misses the true correction by ~0.011-0.012 here,
        outside these tolerances.
        """
        delta = 0.10
        true_corr = 1.0 / (1.0 + delta)
        data = self._make_recovery_data(delta, fit_band0=True)
        config = PhotometricOffsetsConfig(
            n_bootstrap=100, progress_interval=0, random_seed=99
        )
        offsets, errors, n_used = photometric_offsets(
            *data, dim_prior=dim_prior, config=config, verbose=False
        )
        assert n_used[0] == len(data[0])
        assert abs(offsets[0] - true_corr) < tol

    def test_injected_offset_recovered_unfitted_band(self):
        """A band NOT used in fitting: draws already target the correct
        (band-0-free) posterior, so the plain weighted median recovers the
        injected offset."""
        delta = 0.10
        true_corr = 1.0 / (1.0 + delta)
        data = self._make_recovery_data(delta, fit_band0=False)
        mask_fit = np.array([False, True, True, True, True])
        config = PhotometricOffsetsConfig(
            n_bootstrap=100, progress_interval=0, random_seed=99
        )
        offsets, errors, n_used = photometric_offsets(
            *data, mask_fit=mask_fit, dim_prior=False, config=config, verbose=False
        )
        assert n_used[0] == len(data[0])
        assert abs(offsets[0] - true_corr) < 0.005


def _small_setup(n_obj=40, nfilt=5, nsamps=15, n_models=30, seed=42):
    """Small, well-behaved dataset using the real get_seds."""
    rng = np.random.default_rng(seed)
    models = np.zeros((n_models, nfilt, 3))
    models[:, :, 0] = rng.uniform(15, 18, (n_models, nfilt))
    models[:, :, 1] = rng.uniform(0.6, 1.8, (n_models, nfilt))
    models[:, :, 2] = rng.uniform(0.0, 0.25, (n_models, nfilt))
    idxs = rng.integers(0, n_models, (n_obj, nsamps))
    reds = rng.uniform(0.0, 1.0, (n_obj, nsamps))
    dreds = rng.uniform(3.0, 3.5, (n_obj, nsamps))
    dists = rng.uniform(0.9, 1.1, (n_obj, nsamps))
    seds = get_seds(
        models[idxs.ravel()], av=reds.ravel(), rv=dreds.ravel(), return_flux=True
    )
    seds = (seds / dists.ravel()[:, None] ** 2).reshape(n_obj, nsamps, nfilt)
    phot = seds.mean(axis=1) * (1.0 + 0.02 * rng.standard_normal((n_obj, nfilt)))
    err = 0.03 * np.abs(phot)
    mask = np.ones((n_obj, nfilt), dtype=int)
    return phot, err, mask, models, idxs, reds, dreds, dists


class TestInputHandlingFixes:
    """Masked placeholders, float masks, non-positive fluxes."""

    def test_masked_nonfinite_inputs_accepted(self):
        """The canonical BruteForce input format (NaN flux, zero error in
        unobserved bands) must pass validation and fit end-to-end."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        mask = mask.copy()
        mask[:10, 1] = 0
        phot = phot.copy()
        err = err.copy()
        phot[:10, 1] = np.nan  # unobserved placeholders
        err[:10, 1] = 0.0

        _validate_inputs(phot, err, mask, models, idxs, reds, dreds, dists)

        config = PhotometricOffsetsConfig(
            n_bootstrap=20, progress_interval=0, random_seed=1
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )
        assert np.all(np.isfinite(offsets))
        assert np.all(errors >= 0)

    def test_nonfinite_observed_values_still_rejected(self):
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        bad_phot = phot.copy()
        bad_phot[0, 0] = np.nan  # mask[0, 0] == 1: observed
        with pytest.raises(ValueError, match="phot contains non-finite values"):
            _validate_inputs(bad_phot, err, mask, models, idxs, reds, dreds, dists)
        bad_err = err.copy()
        bad_err[0, 0] = 0.0
        with pytest.raises(ValueError, match="err must be positive"):
            _validate_inputs(phot, bad_err, mask, models, idxs, reds, dreds, dists)

    def test_float_mask_accepted(self):
        """A plain np.ones float mask (blessed by validation) must not crash
        on bitwise ops."""
        phot, err, _, models, idxs, reds, dreds, dists = _small_setup()
        mask = np.ones(phot.shape)  # float64
        config = PhotometricOffsetsConfig(
            n_bootstrap=10, progress_interval=0, random_seed=1
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )
        assert np.all(np.isfinite(offsets))
        assert np.all(n_used == len(phot))

    def test_nonpositive_fluxes_excluded(self):
        """Zero/negative observed fluxes must not enter the ratio median
        (they would produce inf or sign-flipped offsets)."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        phot = phot.copy()
        phot[:8, 0] = 0.0
        phot[8:12, 0] = -0.5
        config = PhotometricOffsetsConfig(
            n_bootstrap=20, progress_interval=0, random_seed=1
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )
        assert np.all(np.isfinite(offsets))
        assert offsets[0] > 0
        assert n_used[0] == len(phot) - 12


class TestDegenerateWeightHandling:
    """Objects with collapsed weights must be excluded, not fabricated."""

    def test_collapsed_weights_reported_not_fabricated(self):
        """When every selected object's LOO weights collapse (dof <= 0 under
        the dimensionality prior), the filter must report n_used = 0 and the
        placeholder offset, instead of the median of posterior draw 0."""
        rng = np.random.default_rng(2)
        n_obj, nfilt, nsamps, n_models = 12, 4, 8, 20
        models = np.zeros((n_models, nfilt, 3))
        models[:, :, 0] = rng.uniform(15, 18, (n_models, nfilt))
        models[:, :, 1] = 1.0
        models[:, :, 2] = 0.1
        idxs = rng.integers(0, n_models, (n_obj, nsamps))
        reds = rng.uniform(0.0, 1.0, (n_obj, nsamps))
        dreds = np.full((n_obj, nsamps), 3.1)
        dists = np.ones((n_obj, nsamps))
        phot = np.abs(rng.uniform(0.5, 2.0, (n_obj, nfilt)))
        err = 0.05 * phot
        # every object observes exactly bands {0, 1}: LOO dof = 2 - 1 - 1 = 0
        mask = np.zeros((n_obj, nfilt), dtype=int)
        mask[:, :2] = 1

        config = PhotometricOffsetsConfig(
            n_bootstrap=10, min_bands_used=1, progress_interval=0, random_seed=3
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            dim_prior=True,
            config=config,
            verbose=False,
        )
        assert n_used[0] == 0
        assert offsets[0] == 1.0
        assert errors[0] == 0.0


class TestBandCountThresholds:
    """Selection cuts must match the documented legacy behavior."""

    def test_fitted_band_requires_min_bands_excluding_current(self):
        """min_bands_used=4 (default) means >= 4 bands besides the one being
        calibrated (>= 5 in total, the legacy > 3 + 1 cut); 4-band objects
        are excluded."""
        phot, err, _, models, idxs, reds, dreds, dists = _small_setup()
        n_obj, nfilt = phot.shape
        mask = np.ones((n_obj, nfilt), dtype=int)
        mask[:10, 4] = 0  # first 10 objects observe only 4 bands (incl. band 0)

        config = PhotometricOffsetsConfig(
            n_bootstrap=10, progress_interval=0, random_seed=1
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
        )
        assert n_used[0] == n_obj - 10


class TestPriorHandling:
    """Prior combination for bands without data."""

    def test_prior_respected_for_band_without_data(self):
        """A band with no valid objects must return the prior, not the
        1.0 +/- 0.0 placeholder acting as an infinitely precise
        measurement."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        mask = mask.copy()
        mask[:, 0] = 0  # band 0 never observed
        prior_mean = np.full(phot.shape[1], 1.05)
        prior_std = np.full(phot.shape[1], 0.02)

        config = PhotometricOffsetsConfig(
            n_bootstrap=20, progress_interval=0, random_seed=1
        )
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            prior_mean=prior_mean,
            prior_std=prior_std,
            config=config,
            verbose=False,
        )
        assert n_used[0] == 0
        assert np.isclose(offsets[0], 1.05)
        assert np.isclose(errors[0], 0.02)
        # estimated bands still get the product-of-Gaussians combination
        assert np.all(errors[1:] > 0)
        assert np.all(errors[1:] < 0.02 + 1e-12)


class TestChunkedSedGeneration:
    """Chunked SED generation must be bitwise identical to a single call."""

    def test_chunked_matches_single_call(self):
        rng = np.random.default_rng(9)
        n_obj, nfilt, nsamps, n_models = 37, 5, 7, 50
        models = np.zeros((n_models, nfilt, 3))
        models[:, :, 0] = rng.uniform(15, 18, (n_models, nfilt))
        models[:, :, 1] = rng.uniform(0.6, 1.8, (n_models, nfilt))
        models[:, :, 2] = rng.uniform(0.0, 0.25, (n_models, nfilt))
        idxs = rng.integers(0, n_models, (n_obj, nsamps))
        reds = rng.uniform(0.0, 1.0, (n_obj, nsamps))
        dreds = rng.uniform(3.0, 3.5, (n_obj, nsamps))
        dists = rng.uniform(0.5, 2.0, (n_obj, nsamps))

        full = get_seds(
            models[idxs.ravel()], av=reds.ravel(), rv=dreds.ravel(), return_flux=True
        )
        full = (full / dists.ravel()[:, None] ** 2).reshape(n_obj, nsamps, nfilt)

        # chunk_rows=13 with nsamps=7 -> 1-object blocks; also try mid-size
        for chunk_rows in [13, 50, 10**9]:
            chunked = _generate_seds(
                models, idxs, reds, dreds, dists, chunk_rows=chunk_rows
            )
            assert np.array_equal(full, chunked)


class TestImportHygiene:
    """The placeholder-import fallback is gone: brutus internals import
    unconditionally."""

    def test_no_placeholder_import_fallback(self):
        import brutus.analysis.offsets as mod

        assert mod.get_seds.__module__ == "brutus.core.sed_utils"
        assert not hasattr(mod, "phot_loglike")
        assert not hasattr(mod, "logsumexp")


if __name__ == "__main__":
    pytest.main([__file__])
