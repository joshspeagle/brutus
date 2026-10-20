"""Tests of brutus.analysis.offsets."""

import numpy as np
import pytest

from brutus.analysis.offsets import (
    PhotometricOffsetsConfig,
    _generate_seds,
    _loo_weights,
    _validate_inputs,
    _vectorized_bootstrap_median,
    photometric_offsets,
)
from brutus.core.sed_utils import get_seds
from brutus.utils.photometry import phot_loglike


@pytest.fixture
def mock_data():
    """Create realistic mock data for testing."""
    rng = np.random.RandomState(42)

    n_obj, n_filt, n_samp, n_models = 50, 5, 20, 100

    # Mock photometry
    phot = rng.uniform(0.1, 10.0, (n_obj, n_filt))
    err = 0.05 * phot + 0.01  # Realistic errors
    mask = rng.choice([0, 1], (n_obj, n_filt), p=[0.1, 0.9])

    # Mock models with realistic coefficients
    models = rng.random((n_models, n_filt, 3))
    models[:, :, 0] = rng.uniform(14, 20, (n_models, n_filt))  # Base mags
    models[:, :, 1] = rng.uniform(0.5, 2.0, (n_models, n_filt))  # A(V) coeffs
    models[:, :, 2] = rng.uniform(0.0, 0.3, (n_models, n_filt))  # R(V) coeffs

    # Mock fitted parameters
    idxs = rng.randint(0, n_models, (n_obj, n_samp))
    reds = rng.uniform(0.0, 3.0, (n_obj, n_samp))
    dreds = rng.uniform(2.5, 4.5, (n_obj, n_samp))
    dists = rng.uniform(0.5, 10.0, (n_obj, n_samp))

    return {
        "phot": phot,
        "err": err,
        "mask": mask,
        "models": models,
        "idxs": idxs,
        "reds": reds,
        "dreds": dreds,
        "dists": dists,
    }


TRUE_OFFSETS = np.array([1.10, 0.90, 1.08, 0.92, 1.05])


@pytest.fixture
def injected_offsets():
    """Synthetic fluxes from real `get_seds` SEDs with known offsets injected.

    Each object's samples sit at its true model, A(V), R(V) and distance, so
    `photometric_offsets` should return ``1 / TRUE_OFFSETS``.
    """
    rng = np.random.RandomState(0)
    n_obj, n_filt, n_models, n_samp = 80, len(TRUE_OFFSETS), 20, 10
    models = np.zeros((n_models, n_filt, 3))
    models[:, :, 0] = rng.uniform(15, 18, (n_models, n_filt))
    models[:, :, 1] = rng.uniform(0.7, 1.6, (n_models, n_filt))
    models[:, :, 2] = rng.uniform(0.0, 0.2, (n_models, n_filt))
    idx = rng.randint(0, n_models, n_obj)
    av = rng.uniform(0.0, 1.0, n_obj)
    rv = rng.uniform(3.15, 3.45, n_obj)
    dist = rng.uniform(0.95, 1.05, n_obj)

    flux = get_seds(models[idx], av=av, rv=rv, return_flux=True) / dist[:, None] ** 2
    err = 0.02 * flux
    phot = flux * TRUE_OFFSETS + rng.normal(0.0, err)

    def per_sample(x):
        return np.repeat(x[:, None], n_samp, axis=1)

    return {
        "phot": phot,
        "err": err,
        "mask": np.ones((n_obj, n_filt), dtype=int),
        "models": models,
        "idxs": per_sample(idx),
        "reds": per_sample(av),
        "dreds": per_sample(rv),
        "dists": per_sample(dist),
    }


# The bootstrap errors are about 0.003, so 0.015 is about 5 sigma; the
# smallest injected offset moves the answer by 0.048.
RECOVERY_ATOL = 0.015


def _assert_recovered(offsets):
    np.testing.assert_allclose(offsets, 1.0 / TRUE_OFFSETS, atol=RECOVERY_ATOL)


class TestPhotometricOffsetsConfig:
    """Test the configuration class."""

    def test_default_config(self):
        config = PhotometricOffsetsConfig()

        assert config.min_bands_used == 4
        assert config.min_bands_unused == 3
        assert config.n_bootstrap == 300
        assert config.uncertainty_method == "bootstrap_iqr"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid min_bands_used
        with pytest.raises(ValueError, match="min_bands_used must be >= 1"):
            PhotometricOffsetsConfig(min_bands_used=0)

        # Test invalid uncertainty_method
        with pytest.raises(ValueError, match="Unknown uncertainty_method"):
            PhotometricOffsetsConfig(uncertainty_method="invalid")

        # Test invalid n_bootstrap
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            PhotometricOffsetsConfig(n_bootstrap=0)


class TestPhotometricOffsetsCore:
    """photometric_offsets recovers offsets injected into synthetic data."""

    @staticmethod
    def _run(data, uncertainty_method="bootstrap_iqr", **kwargs):
        config = PhotometricOffsetsConfig(
            n_bootstrap=50, uncertainty_method=uncertainty_method
        )
        return photometric_offsets(
            **data, config=config, verbose=False, rng=np.random.default_rng(1), **kwargs
        )

    @pytest.mark.parametrize("method", ["bootstrap_std", "bootstrap_iqr"])
    def test_injected_offsets_are_recovered(self, injected_offsets, method):
        offsets, errors, n_used = self._run(injected_offsets, uncertainty_method=method)
        _assert_recovered(offsets)
        assert np.all((errors > 0) & (errors < RECOVERY_ATOL))
        np.testing.assert_array_equal(n_used, len(injected_offsets["phot"]))

    def test_selection_and_weights_restrict_the_objects_used(self, injected_offsets):
        n_obj, n_samp = injected_offsets["idxs"].shape
        sel = np.arange(n_obj) % 4 != 0
        weights = np.random.RandomState(2).uniform(0.5, 1.5, (n_obj, n_samp))
        offsets, _, n_used = self._run(injected_offsets, sel=sel, weights=weights)
        _assert_recovered(offsets)
        np.testing.assert_array_equal(n_used, sel.sum())

    def test_tight_prior_pulls_offsets_to_its_mean(self, injected_offsets):
        n_filt = len(TRUE_OFFSETS)
        offsets, errors, _ = self._run(
            injected_offsets,
            prior_mean=np.ones(n_filt),
            prior_std=np.full(n_filt, 1e-4),
        )
        np.testing.assert_allclose(offsets, 1.0, atol=1e-3)
        assert np.all(errors < 1e-4)


class TestInputValidation:
    """Test input validation functionality."""

    def test_array_shape_validation(self, mock_data):
        """Test validation of array shapes."""
        from brutus.analysis.offsets import _validate_inputs

        # Test correct shapes (should not raise)
        _validate_inputs(
            mock_data["phot"],
            mock_data["err"],
            mock_data["mask"],
            mock_data["models"],
            mock_data["idxs"],
            mock_data["reds"],
            mock_data["dreds"],
            mock_data["dists"],
        )

        # Test incorrect err shape
        with pytest.raises(ValueError, match="err shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"][:25],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test incorrect mask shape
        with pytest.raises(ValueError, match="mask shape"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"][:25],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

    def test_array_value_validation(self, mock_data):
        """Test validation of array values."""
        from brutus.analysis.offsets import _validate_inputs

        # Test negative errors
        bad_err = -np.abs(mock_data["err"])
        with pytest.raises(ValueError, match="err must be positive"):
            _validate_inputs(
                mock_data["phot"],
                bad_err,
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test invalid mask values
        bad_mask = mock_data["mask"].copy().astype(float)
        bad_mask[0, 0] = 0.5  # Should be 0 or 1
        with pytest.raises(ValueError, match="mask must contain only 0s and 1s"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                bad_mask,
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )

        # Test negative distances
        bad_dists = -np.abs(mock_data["dists"])
        with pytest.raises(ValueError, match="dists must be positive"):
            _validate_inputs(
                mock_data["phot"],
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                bad_dists,
            )

        # Test non-finite photometry
        bad_phot = mock_data["phot"].copy()
        bad_phot[0, 0] = np.nan
        with pytest.raises(ValueError, match="phot contains non-finite values"):
            _validate_inputs(
                bad_phot,
                mock_data["err"],
                mock_data["mask"],
                mock_data["models"],
                mock_data["idxs"],
                mock_data["reds"],
                mock_data["dreds"],
                mock_data["dists"],
            )


class TestParameterRecovery:
    """Test parameter recovery using real get_seds function."""

    def test_unity_recovery_real_get_seds(self):
        """Test unity recovery using real get_seds function."""
        # Import should always work
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            get_seds,
            photometric_offsets,
        )

        n_obj, n_filt = 60, 5
        n_models, n_samp = 25, 12

        # Create realistic model grid
        rng = np.random.RandomState(42)
        models = rng.random((n_models, n_filt, 3))
        models[:, :, 0] = rng.uniform(15, 19, (n_models, n_filt))
        models[:, :, 1] = rng.uniform(0.6, 1.8, (n_models, n_filt))
        models[:, :, 2] = rng.uniform(0.0, 0.25, (n_models, n_filt))

        # True parameters for data generation
        true_model_idxs = rng.randint(0, n_models, n_obj)
        true_avs = rng.uniform(0.0, 1.2, n_obj)
        true_rvs = rng.uniform(3.1, 3.5, n_obj)
        true_dists = rng.uniform(0.9, 1.1, n_obj)

        # Generate observed photometry using real get_seds
        observed_phot = np.zeros((n_obj, n_filt))
        for i in range(n_obj):
            true_sed = get_seds(
                models[true_model_idxs[i : i + 1]],
                av=np.array([true_avs[i]]),
                rv=np.array([true_rvs[i]]),
                return_flux=True,
            )
            observed_phot[i] = true_sed[0] / (true_dists[i] ** 2)

        # Add realistic noise
        noise = rng.normal(0, 0.025 * observed_phot)
        observed_phot += noise
        observed_phot = np.maximum(observed_phot, 0.1 * np.abs(observed_phot))

        err = 0.03 * observed_phot
        mask = np.ones((n_obj, n_filt), dtype=int)

        # Create samples centered on true values
        idxs = np.zeros((n_obj, n_samp), dtype=int)
        reds = np.zeros((n_obj, n_samp))
        dreds = np.zeros((n_obj, n_samp))
        dists = np.zeros((n_obj, n_samp))

        for i in range(n_obj):
            idxs[i] = true_model_idxs[i]
            reds[i] = rng.normal(true_avs[i], 0.015, n_samp)
            dreds[i] = rng.normal(true_rvs[i], 0.008, n_samp)
            dists[i] = rng.normal(true_dists[i], 0.008, n_samp)

            reds[i] = np.clip(reds[i], 0.0, 2.5)
            dreds[i] = np.clip(dreds[i], 2.9, 3.9)
            dists[i] = np.clip(dists[i], 0.6, 1.5)

        # Run the algorithm (real likelihood reweighting)
        config = PhotometricOffsetsConfig(
            n_bootstrap=20,
            min_bands_used=3,
            min_bands_unused=2,
        )

        offsets, errors, n_used = photometric_offsets(
            observed_phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=config,
            verbose=False,
            rng=np.random.default_rng(1),
        )

        # Check unity recovery
        max_deviation = np.max(np.abs(offsets - 1.0))
        max_rel_error = np.max(np.abs((offsets - 1.0) / 1.0))

        # Unity recovery should be very accurate
        assert (
            max_rel_error < 0.05
        ), f"Unity recovery failed: max relative error {max_rel_error:.3f} > 0.05"
        assert np.all(np.isfinite(offsets)), "Offsets contain non-finite values"
        assert np.all(errors >= 0), "Errors must be non-negative"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_valid_objects(self):
        """Test behavior when no objects meet selection criteria."""
        from brutus.analysis.offsets import (
            PhotometricOffsetsConfig,
            photometric_offsets,
        )

        rng = np.random.RandomState(3)
        n_obj, n_filt = 10, 5
        phot = rng.uniform(0.1, 1.0, (n_obj, n_filt))
        err = 0.1 * phot
        mask = np.zeros((n_obj, n_filt), dtype=bool)  # No detections

        models = rng.random_sample((10, n_filt, 3))
        idxs = rng.randint(0, 10, (n_obj, 5))
        reds = rng.uniform(0, 1, (n_obj, 5))
        dreds = rng.uniform(3, 4, (n_obj, 5))
        dists = rng.uniform(1, 5, (n_obj, 5))

        config = PhotometricOffsetsConfig(
            n_bootstrap=5,
            min_bands_used=3,
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
            rng=np.random.default_rng(1),
        )

        # Should handle gracefully
        assert offsets.shape == (n_filt,)
        assert np.all(n_used == 0)  # No objects used
        assert np.all(offsets == 1.0)  # Default values
        # No data and no prior: infinite (not zero) uncertainty, so
        # the placeholder offsets cannot pose as measurements
        assert np.all(np.isinf(errors))


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

    def test_bootstrap_distribution_matches_reference(self):
        """Old and new implementations sample the same bootstrap-median
        distribution (different random streams): with a fixed seed, medians
        and IQRs over 500 bootstraps agree within Monte-Carlo error."""
        rng_data = np.random.default_rng(3)
        n_objects, nsamps = 100, 25
        ratios = rng_data.lognormal(0.0, 0.15, (n_objects, nsamps))
        weights = rng_data.random((n_objects, nsamps)) ** 3
        weights[5] = 0.0  # zero-weight row exercises the sample-0 fallback
        sums = weights.sum(axis=1)
        nz = sums > 0
        weights[nz] /= sums[nz, None]
        obj_weights = np.full(n_objects, 1.0 / n_objects)

        n_bootstrap = 500
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

    def test_loo_weights_match_phot_loglike_reference(self):
        """For the objects that observe the band, the weights equal
        softmax(lnl_loo - lnl_full) computed with two Gaussian phot_loglike
        passes (the reference formulation)."""
        rng = np.random.default_rng(5)
        n, nsamps, nfilt = 40, 30, 6
        phot = rng.uniform(0.5, 2.0, (n, nfilt))
        err = rng.uniform(0.05, 0.2, (n, nfilt))
        seds = phot[:, None, :] * rng.uniform(0.8, 1.2, (n, nsamps, nfilt))
        mask = rng.random((n, nfilt)) < 0.7
        band = 2

        mask_full = mask.astype(float)
        mask_loo = mask_full.copy()
        mask_loo[:, band] = 0.0
        lnl_full = phot_loglike(phot, err, seds, mask=mask_full)
        lnl_loo = phot_loglike(phot, err, seds, mask=mask_loo)
        ref_weights = _softmax_rows(lnl_loo - lnl_full)

        obs = mask[:, band]
        new_weights = _loo_weights(
            phot[obs, band],
            err[obs, band],
            seds[obs, :, band],
            np.ones((obs.sum(), nsamps)),
        )
        new_weights /= new_weights.sum(axis=1, keepdims=True)

        assert np.allclose(new_weights, ref_weights[obs], rtol=1e-9, atol=1e-12)

    def test_default_offsets_use_gaussian_loo_weights(self):
        """With default arguments, each fitted band's offset is the
        bootstrap median of the ratios under w = exp(+chi2_i / 2) times the
        sample weights, computed here directly with the same random stream.

        Regression: the default used to weight by a ratio of chi-square
        densities, which does not match the BruteForce posterior.
        """
        phot, err, mask, models, idxs, reds, dreds, dists = self._make_recovery_data(
            0.1, fit_band0=True, n_obj=40, nsamps=50
        )
        old_offsets = np.array([1.02, 0.98, 1.0, 1.01, 0.99])
        weights = np.random.default_rng(4).uniform(0.5, 1.5, idxs.shape)
        config = PhotometricOffsetsConfig(n_bootstrap=30)
        offsets, _, _ = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            weights=weights,
            old_offsets=old_offsets,
            config=config,
            verbose=False,
            rng=np.random.default_rng(7),
        )

        seds = _generate_seds(models, idxs, reds, dreds, dists)
        rng = np.random.default_rng(7)
        n = len(phot)
        expected = np.empty(phot.shape[1])
        for i in range(phot.shape[1]):
            var_i = err[:, i] ** 2 * old_offsets[i] ** 2
            chi2_i = (phot[:, None, i] * old_offsets[i] - seds[:, :, i]) ** 2
            lnw = 0.5 * chi2_i / var_i[:, None] + np.log(weights)
            w = np.exp(lnw - lnw.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            medians = _vectorized_bootstrap_median(
                seds[:, :, i] / phot[:, None, i], w, np.full(n, 1.0 / n), 30, rng
            )
            expected[i] = np.median(medians)
        np.testing.assert_allclose(offsets, expected, rtol=1e-12)

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

    def test_injected_offset_recovered_fitted_band(self):
        """A band used in fitting: draws come from the contaminated full
        posterior, so recovery requires the corrected importance weights.

        Regression: the pre-fix weighting (w = L_-i) removes only ~2/3 of
        the circularity and misses the true correction by ~0.011-0.012 here,
        outside the tolerance.
        """
        delta = 0.10
        true_corr = 1.0 / (1.0 + delta)
        data = self._make_recovery_data(delta, fit_band0=True)
        config = PhotometricOffsetsConfig(n_bootstrap=100)
        offsets, errors, n_used = photometric_offsets(
            *data, config=config, rng=np.random.default_rng(99), verbose=False
        )
        assert n_used[0] == len(data[0])
        assert abs(offsets[0] - true_corr) < 0.006

    def test_injected_offset_recovered_unfitted_band(self):
        """A band NOT used in fitting: draws already target the correct
        (band-0-free) posterior, so the plain weighted median recovers the
        injected offset."""
        delta = 0.10
        true_corr = 1.0 / (1.0 + delta)
        data = self._make_recovery_data(delta, fit_band0=False)
        mask_fit = np.array([False, True, True, True, True])
        config = PhotometricOffsetsConfig(n_bootstrap=100)
        offsets, errors, n_used = photometric_offsets(
            *data,
            mask_fit=mask_fit,
            config=config,
            rng=np.random.default_rng(99),
            verbose=False,
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

        config = PhotometricOffsetsConfig(n_bootstrap=20)
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
            rng=np.random.default_rng(1),
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
        config = PhotometricOffsetsConfig(n_bootstrap=10)
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
            rng=np.random.default_rng(1),
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
        config = PhotometricOffsetsConfig(n_bootstrap=20)
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
            rng=np.random.default_rng(1),
            verbose=False,
        )
        assert np.all(np.isfinite(offsets))
        assert offsets[0] > 0
        assert n_used[0] == len(phot) - 12


class TestDegenerateWeightHandling:
    """Objects with collapsed weights must be excluded, not fabricated."""

    def test_collapsed_weights_reported_not_fabricated(self):
        """When every object's leave-one-out weights are undefined (an
        infinite band chi-square), the filter must report n_used = 0 and the
        placeholder offset, instead of the median of posterior draw 0."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        err = 1e-200 * phot  # err**2 underflows to 0: chi2 is infinite
        offsets, errors, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            config=PhotometricOffsetsConfig(n_bootstrap=10),
            verbose=False,
            rng=np.random.default_rng(3),
        )
        assert n_used[0] == 0
        assert offsets[0] == 1.0
        # collapsed weights leave no measurement: infinite uncertainty
        assert np.isinf(errors[0])

    def test_only_weighted_draw_survives_a_large_chi2(self):
        """A draw that holds all the sample weight keeps its weight even
        when other draws have far larger leave-one-out weights.

        Regression: the importance weights were scaled by their row maximum
        before the sample weights multiplied them, so the product underflowed
        to 0 and the object was dropped as degenerate.
        """
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        err = 1e-3 * phot  # band chi-squares of order 1e4: exp(-chi2/2) = 0
        weights = np.zeros(idxs.shape)
        weights[:, 0] = 1.0
        seds = _generate_seds(models, idxs, reds, dreds, dists)
        phot = phot.copy()
        phot[:, 0] = seds[:, 0, 0]  # draw 0 fits band 0 exactly
        offsets, _, n_used = photometric_offsets(
            phot,
            err,
            mask,
            models,
            idxs,
            reds,
            dreds,
            dists,
            weights=weights,
            config=PhotometricOffsetsConfig(n_bootstrap=10),
            verbose=False,
            rng=np.random.default_rng(3),
        )
        assert n_used[0] == len(phot)
        assert offsets[0] == 1.0


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

        config = PhotometricOffsetsConfig(n_bootstrap=10)
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
            rng=np.random.default_rng(1),
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

        config = PhotometricOffsetsConfig(n_bootstrap=20)
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
            rng=np.random.default_rng(1),
            verbose=False,
        )
        assert n_used[0] == 0
        assert np.isclose(offsets[0], 1.05)
        assert np.isclose(errors[0], 0.02)
        # estimated bands still get the product-of-Gaussians combination
        assert np.all(errors[1:] > 0)
        assert np.all(errors[1:] < 0.02 + 1e-12)

    def test_band_without_data_and_without_prior_has_infinite_error(self):
        """No data and no prior means no measurement: the placeholder
        offset of 1 must carry infinite (not zero) uncertainty."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        mask = mask.copy()
        mask[:, 0] = 0  # band 0 never observed

        # Relax the band-count thresholds so the OTHER bands still qualify
        # with one band masked (the small fixture has few filters).
        config = PhotometricOffsetsConfig(
            n_bootstrap=20,
            min_bands_used=3,
            min_bands_unused=3,
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
            rng=np.random.default_rng(1),
        )
        assert n_used[0] == 0
        assert offsets[0] == 1.0
        assert np.isinf(errors[0])
        assert np.all(np.isfinite(errors[1:]))

    def test_one_sided_prior_raises(self):
        """Supplying only one of prior_mean/prior_std must fail fast
        instead of silently ignoring the prior."""
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        with pytest.raises(ValueError, match="provided together"):
            photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                prior_mean=np.ones(phot.shape[1]),
                verbose=False,
            )


class TestPerFilterInputValidation:
    """mask_fit / old_offsets must be validated up front."""

    def test_wrong_length_mask_fit_raises(self):
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        with pytest.raises(ValueError, match="mask_fit must have shape"):
            photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                mask_fit=np.ones(3, dtype=bool),
                verbose=False,
            )

    def test_scalar_old_offsets_raises(self):
        phot, err, mask, models, idxs, reds, dreds, dists = _small_setup()
        with pytest.raises(ValueError, match="old_offsets must have shape"):
            photometric_offsets(
                phot,
                err,
                mask,
                models,
                idxs,
                reds,
                dreds,
                dists,
                old_offsets=1.0,
                verbose=False,
            )

    def test_integer_sel_of_zeros_and_ones_acts_as_a_bool_mask(self):
        sel = np.arange(40) % 3 > 0
        results = [
            photometric_offsets(
                *_small_setup(),
                sel=s,
                config=PhotometricOffsetsConfig(n_bootstrap=5),
                verbose=False,
                rng=np.random.default_rng(0),
            )
            for s in (sel, sel.astype(int))
        ]
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"sel": np.full(40, 2)}, "sel must be a bool or 0/1 array"),
            ({"sel": np.array([0, 3, 7])}, "sel must be a bool or 0/1 array"),
            ({"weights": np.ones((40, 3))}, "weights must have shape"),
            ({"weights": -np.ones((40, 15))}, "weights must be finite"),
            ({"prior_mean": 1.0, "prior_std": 0.1}, "must have shape"),
            (
                {"prior_mean": np.ones(5), "prior_std": np.zeros(5)},
                "prior_std must be finite and > 0",
            ),
        ],
    )
    def test_invalid_sel_weights_and_priors_raise_before_seds(
        self, monkeypatch, kwargs, match
    ):
        """Bad `sel`, `weights` or priors raise a ValueError that names the
        argument, before any model SED is generated."""
        import brutus.analysis.offsets as mod

        def fail(*args, **kwargs):
            raise AssertionError("validation ran after SED generation")

        monkeypatch.setattr(mod, "_generate_seds", fail)
        with pytest.raises(ValueError, match=match):
            photometric_offsets(*_small_setup(), verbose=False, **kwargs)


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
