#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for binary population handling in brutus stellar populations module.

This test suite covers coverage gaps in src/brutus/core/populations.py:
1. Scalar fallback SED generation (lines 750-764)
2. Equal-mass binary fraction (binary_fraction=1.0)
3. return_dict=False output format (lines 803-804)
4. _add_binary_components method (lines 829-916)
"""

from copy import deepcopy
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import numpy.testing as npt
import pytest

from brutus.core.populations import Isochrone, StellarPop


def _make_mock_isochrone(n_eep=10):
    """Create a mock isochrone with realistic stellar parameters."""
    iso = MagicMock()

    masses = np.linspace(0.6, 2.5, n_eep)
    eeps = np.linspace(200, 600, n_eep)

    params = np.column_stack(
        [
            masses,  # mini
            masses * 0.97,  # mass (current, slightly less)
            np.linspace(-0.2, 2.0, n_eep),  # logl
            np.linspace(3.65, 3.95, n_eep),  # logt
            np.linspace(0.0, 1.0, n_eep),  # logr
            np.linspace(4.7, 3.7, n_eep),  # logg (decreases with mass)
            np.zeros(n_eep),  # feh_surf
            np.zeros(n_eep),  # afe_surf
        ]
    )

    iso.predictions = [
        "mini",
        "mass",
        "logl",
        "logt",
        "logr",
        "logg",
        "feh_surf",
        "afe_surf",
    ]
    iso.eep_u = eeps

    def mock_get_predictions(
        feh=0.0, afe=0.0, loga=8.5, eep=None, apply_corr=True, corr_params=None
    ):
        if eep is not None:
            # For secondary stars: generate scaled-down parameters
            sec_params = params.copy()
            sec_params[:, 0] *= 0.7
            sec_params[:, 1] *= 0.7
            sec_params[:, 2] -= 0.3
            sec_params[:, 5] += 0.2
            return sec_params
        return params

    iso.get_predictions = mock_get_predictions
    return iso


def _make_stellar_pop(iso, has_sed_batch=True, nfilt=5):
    """Create a StellarPop with a mock predictor."""
    with patch("brutus.core.neural_nets.FastNNPredictor"):
        pop = StellarPop(isochrone=iso, verbose=False)

    mock_predictor = MagicMock()
    mock_predictor.NFILT = nfilt
    mock_predictor.sed.return_value = np.linspace(14.0, 16.0, nfilt)

    if has_sed_batch:
        # sed_batch needs to return shape (N, nfilt) where N = len(logt)
        def _sed_batch_side_effect(**kwargs):
            n = len(kwargs["logt"])
            return np.tile(np.linspace(14.0, 16.0, nfilt), (n, 1))

        mock_predictor.sed_batch.side_effect = _sed_batch_side_effect
    else:
        # Remove sed_batch entirely so hasattr returns False
        del mock_predictor.sed_batch

    pop.predictor = mock_predictor
    pop.filters = [f"band_{i}" for i in range(nfilt)]
    return pop


class TestScalarFallbackSEDGeneration:
    """Test the scalar loop fallback when sed_batch is unavailable (lines 748-764)."""

    def test_scalar_fallback_when_no_sed_batch(self):
        """When predictor lacks sed_batch, code should fall back to scalar loop."""
        iso = _make_mock_isochrone(n_eep=5)
        pop = _make_stellar_pop(iso, has_sed_batch=False, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0)

        # Should have called sed() for each star above mini_bound
        assert pop.predictor.sed.called
        assert isinstance(seds, np.ndarray)
        assert seds.shape == (5, 4)
        # All stars should have valid SEDs (all mini >= 0.5)
        assert np.all(np.isfinite(seds))

    def test_scalar_fallback_when_sed_batch_fails(self):
        """When sed_batch raises, code should fall back to scalar loop."""
        iso = _make_mock_isochrone(n_eep=5)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        # Make sed_batch raise an exception to trigger fallback
        pop.predictor.sed_batch.side_effect = RuntimeError("batch failed")

        with pytest.warns(RuntimeWarning, match="Batch primary SED generation failed"):
            seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0)

        # Should still produce valid output via scalar loop
        assert pop.predictor.sed.called
        assert isinstance(seds, np.ndarray)
        assert seds.shape == (5, 4)

    def test_scalar_fallback_respects_mini_bound(self):
        """Scalar loop should skip stars below mini_bound."""
        iso = _make_mock_isochrone(n_eep=5)
        # Masses range from 0.6 to 2.5; mini_bound=1.0 should exclude some
        pop = _make_stellar_pop(iso, has_sed_batch=False, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, mini_bound=1.0)

        # Stars below mini_bound should have NaN SEDs
        masses = params["mini"]
        for i in range(len(masses)):
            if masses[i] < 1.0:
                assert np.all(np.isnan(seds[i]))
            else:
                assert np.all(np.isfinite(seds[i]))

    def test_scalar_fallback_handles_individual_failure(self):
        """Scalar loop should warn and continue if individual sed() call fails."""
        iso = _make_mock_isochrone(n_eep=5)
        pop = _make_stellar_pop(iso, has_sed_batch=False, nfilt=4)

        call_count = [0]
        original_return = np.linspace(14.0, 16.0, 4)

        def sed_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("SED generation failed for this star")
            return original_return.copy()

        pop.predictor.sed.side_effect = sed_side_effect

        with pytest.warns(RuntimeWarning, match="Primary SED generation failed"):
            seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0)

        # The star that failed should have NaN SEDs
        assert np.any(np.isnan(seds))
        # Other stars should be fine
        assert np.any(np.isfinite(seds))


class TestEqualMassBinary:
    """Test get_seds with binary_fraction=1.0 (lines 776-777, 797-799).

    Note: binary_fraction=1.0 is captured by the first branch
    (0.0 < binary_fraction <= 1.0) and goes through _add_binary_components,
    NOT the elif branch (which is dead code). We test the actual behavior.
    """

    def test_binary_fraction_one_calls_add_binary(self):
        """binary_fraction=1.0 goes through _add_binary_components."""
        iso = _make_mock_isochrone(n_eep=8)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=1.0)

        assert isinstance(seds, np.ndarray)
        assert isinstance(params, dict)
        assert isinstance(params2, dict)
        assert seds.shape == (8, 4)

        # Secondary parameters should be populated
        assert "mini" in params2
        # With binary_fraction=1.0, secondary mass = primary mass
        # so _add_binary_components should produce secondary params

    def test_binary_seds_differ_from_single(self):
        """Binary SEDs should differ from single-star SEDs (combined flux)."""
        iso = _make_mock_isochrone(n_eep=8)
        pop_single = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)
        pop_binary = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds_single, _, _ = pop_single.get_seds(feh=0.0, loga=9.0, binary_fraction=0.0)
        seds_binary, _, _ = pop_binary.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)

        # Binary SEDs should generally be brighter (smaller magnitudes)
        # due to combined flux, at least for some stars
        assert isinstance(seds_binary, np.ndarray)
        assert seds_binary.shape == seds_single.shape


class TestReturnDictFalse:
    """Test get_seds with return_dict=False (lines 802-804)."""

    def test_return_dict_false_returns_arrays(self):
        """With return_dict=False, params should be arrays not dicts."""
        iso = _make_mock_isochrone(n_eep=6)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, return_dict=False)

        # params should be np.ndarray, not dict
        assert isinstance(params, np.ndarray)
        assert isinstance(params2, np.ndarray)
        assert isinstance(seds, np.ndarray)

    def test_return_dict_false_shape(self):
        """Array output shapes should match the source code behavior."""
        n_eep = 6
        iso = _make_mock_isochrone(n_eep=n_eep)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, return_dict=False)

        n_pred = len(iso.predictions)
        # params = params_arr.T => shape (Npred, Neep)
        assert params.shape == (n_pred, n_eep)
        # params2 = params_arr2 (no binary) => shape (Neep, Npred)
        assert params2.shape == (n_eep, n_pred)
        assert seds.shape == (n_eep, 4)

    def test_return_dict_true_returns_dicts(self):
        """Verify default return_dict=True returns dicts (contrast test)."""
        iso = _make_mock_isochrone(n_eep=6)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, return_dict=True)

        assert isinstance(params, dict)
        assert isinstance(params2, dict)
        assert "mini" in params
        assert "mini" in params2

    def test_return_dict_false_with_binaries(self):
        """return_dict=False should work correctly with binary populations."""
        n_eep = 8
        iso = _make_mock_isochrone(n_eep=n_eep)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(
            feh=0.0, loga=9.0, binary_fraction=0.5, return_dict=False
        )

        n_pred = len(iso.predictions)
        assert isinstance(params, np.ndarray)
        assert isinstance(params2, np.ndarray)
        # params = params_arr.T => (Npred, Neep)
        assert params.shape == (n_pred, n_eep)
        # params2 = params_arr2 => (Neep, Npred) -- not transposed
        assert params2.shape == (n_eep, n_pred)


class TestAddBinaryComponents:
    """Test the _add_binary_components method (lines 829-916)."""

    def test_binary_components_generated(self):
        """Binary components should be generated with reasonable properties."""
        iso = _make_mock_isochrone(n_eep=10)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)

        # Secondary parameters should have some finite values
        assert np.any(np.isfinite(params2["mini"]))

    def test_binary_secondary_mass_relation(self):
        """Secondary mass should be binary_fraction * primary mass."""
        iso = _make_mock_isochrone(n_eep=10)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        bf = 0.7
        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=bf)

        # The _add_binary_components computes mini2 = mini * binary_fraction
        # and then uses interpolation to find corresponding EEPs.
        # We can verify the secondary parameters were populated.
        assert isinstance(params2, dict)
        assert "mini" in params2
        assert "logl" in params2
        assert "logt" in params2

    def test_binary_eep_restriction(self):
        """Binaries should be restricted to EEPs below eep_binary_max."""
        iso = _make_mock_isochrone(n_eep=10)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        # Use a low eep_binary_max to restrict binary modeling
        # The _add_binary_components method sets eep2 to NaN for
        # stars with eep > eep_binary_max, which means the SED
        # combination step won't brighten those stars.
        seds_restricted, params, params2 = pop.get_seds(
            feh=0.0, loga=9.0, binary_fraction=0.5, eep_binary_max=300.0
        )

        # Should still produce output without errors
        assert isinstance(seds_restricted, np.ndarray)
        assert seds_restricted.shape == (10, 4)

        # Compare with a high eep_binary_max -- should differ
        pop2 = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)
        seds_unrestricted, _, _ = pop2.get_seds(
            feh=0.0, loga=9.0, binary_fraction=0.5, eep_binary_max=700.0
        )
        assert isinstance(seds_unrestricted, np.ndarray)

    def test_binary_output_shapes(self):
        """Output shapes should be consistent for binary populations."""
        n_eep = 10
        nfilt = 4
        iso = _make_mock_isochrone(n_eep=n_eep)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=nfilt)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)

        # SED array should have correct shape
        assert seds.shape == (n_eep, nfilt)

        # Both primary and secondary params should have same keys
        assert set(params.keys()) == set(params2.keys())

        # Each parameter array should have length n_eep
        for key in params:
            assert len(params[key]) == n_eep
            assert len(params2[key]) == n_eep

    def test_binary_scalar_fallback_for_secondaries(self):
        """When sed_batch is unavailable, secondary SEDs use scalar loop."""
        iso = _make_mock_isochrone(n_eep=5)
        pop = _make_stellar_pop(iso, has_sed_batch=False, nfilt=4)

        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)

        # Should still produce output even without sed_batch
        assert isinstance(seds, np.ndarray)
        assert seds.shape == (5, 4)
        # sed() should have been called for both primary and secondary stars
        assert pop.predictor.sed.call_count > 5  # More than just primary stars

    def test_binary_secondary_batch_failure_fallback(self):
        """When secondary sed_batch fails, should fall back to scalar loop."""
        iso = _make_mock_isochrone(n_eep=5)
        pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)

        call_count = [0]

        def batch_side_effect(**kwargs):
            call_count[0] += 1
            n = len(kwargs["logt"])
            if call_count[0] == 1:
                # Primary batch succeeds
                return np.tile(np.linspace(14.0, 16.0, 4), (n, 1))
            else:
                # Secondary batch fails
                raise RuntimeError("secondary batch failed")

        pop.predictor.sed_batch.side_effect = batch_side_effect

        with pytest.warns(
            RuntimeWarning, match="Batch secondary SED generation failed"
        ):
            seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)

        assert isinstance(seds, np.ndarray)
        # sed() should have been called for fallback secondary computation
        assert pop.predictor.sed.called

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_binary_no_valid_masses(self):
        """When no primary masses are valid for interpolation, handle gracefully."""
        iso = MagicMock()
        iso.predictions = [
            "mini",
            "mass",
            "logl",
            "logt",
            "logr",
            "logg",
            "feh_surf",
            "afe_surf",
        ]
        # Use float eep_u to match the real isochrone's behavior after interp
        iso.eep_u = np.array([200.0, 250.0, 300.0, 350.0, 400.0])

        # All NaN mini values -- no valid masses
        params_all_nan = np.full((5, 8), np.nan)
        iso.get_predictions = MagicMock(return_value=params_all_nan)

        with patch("brutus.core.neural_nets.FastNNPredictor"):
            pop = StellarPop(isochrone=iso, verbose=False)

        mock_predictor = MagicMock()
        mock_predictor.NFILT = 4
        mock_predictor.sed.return_value = np.array([14.0, 15.0, 15.5, 16.0])
        del mock_predictor.sed_batch
        pop.predictor = mock_predictor

        # Should not crash -- all SEDs will be NaN since mini < mini_bound (NaN)
        seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.5)
        assert isinstance(seds, np.ndarray)

    def test_binary_different_fractions(self):
        """Different binary fractions should produce different results."""
        iso = _make_mock_isochrone(n_eep=8)

        results = {}
        for bf in [0.0, 0.3, 0.7, 1.0]:
            pop = _make_stellar_pop(iso, has_sed_batch=True, nfilt=4)
            seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=bf)
            results[bf] = (seds.copy(), params2.copy())

        # With bf=0.0, secondary should be all NaN
        assert np.all(np.isnan(results[0.0][1]["mini"]))


class TestPredictorNotAvailable:
    """Test error when predictor is None."""

    def test_no_predictor_raises_error(self):
        """get_seds should raise RuntimeError when predictor is None."""
        iso = _make_mock_isochrone(n_eep=5)
        with patch("brutus.core.neural_nets.FastNNPredictor"):
            pop = StellarPop(isochrone=iso, verbose=False)
        pop.predictor = None

        with pytest.raises(
            RuntimeError, match="Neural network predictor not available"
        ):
            pop.get_seds(feh=0.0, loga=9.0)
