#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for coverage gaps in brutus neural_nets module.

Covers:
1. FastNNPredictor initialization with filters=None (line 354)
2. Out-of-bounds parameter handling returning NaN (line 505)
"""

import os

import numpy as np
import numpy.testing as npt
import pytest
from conftest import find_brutus_data_file


def _find_nn_file():
    """Find the neural network data file."""
    nn_file = find_brutus_data_file("nn_c3k.h5")
    if nn_file is None:
        nn_file = find_brutus_data_file("nnMIST_BC.h5")
    return nn_file


_nn_file = _find_nn_file()
_has_nn_data = _nn_file is not None


@pytest.mark.skipif(not _has_nn_data, reason="Neural network data file not available")
class TestFastNNPredictorDefaultFilters:
    """Test FastNNPredictor initialization with filters=None (line 354)."""

    def test_default_filters_initialization(self):
        """When filters=None, predictor should use all FILTERS."""
        from brutus.core.neural_nets import FastNNPredictor
        from brutus.data.filters import FILTERS

        predictor = FastNNPredictor(filters=None, nnfile=_nn_file, verbose=False)

        # Should have loaded all filters
        assert predictor.NFILT == len(FILTERS)
        npt.assert_array_equal(predictor.filters, np.array(FILTERS))

    def test_default_filters_produce_valid_seds(self):
        """Default filter predictor should produce valid SED output."""
        from brutus.core.neural_nets import FastNNPredictor

        predictor = FastNNPredictor(filters=None, nnfile=_nn_file, verbose=False)

        # Solar-type star at 1 kpc
        sed = predictor.sed(
            logt=3.76,
            logg=4.44,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
        )

        assert isinstance(sed, np.ndarray)
        assert len(sed) == predictor.NFILT
        assert np.all(np.isfinite(sed))


@pytest.mark.skipif(not _has_nn_data, reason="Neural network data file not available")
class TestFastNNPredictorOutOfBounds:
    """Test out-of-bounds parameter handling (line 505)."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor with a few filters."""
        from brutus.core.neural_nets import FastNNPredictor

        # Use a small subset of filters for speed
        filters = ["PS_g", "PS_r", "PS_i"]
        return FastNNPredictor(filters=filters, nnfile=_nn_file, verbose=False)

    def test_out_of_bounds_logt_returns_nan(self, predictor):
        """Parameters outside training bounds should return NaN."""
        # Use an extremely high temperature (well beyond training range)
        sed = predictor.sed(
            logt=6.0,  # 10^6 K -- way beyond bounds
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
        )

        assert isinstance(sed, np.ndarray)
        assert len(sed) == predictor.NFILT
        assert np.all(np.isnan(sed))

    def test_out_of_bounds_logg_returns_nan(self, predictor):
        """Out-of-bounds logg should return NaN."""
        sed = predictor.sed(
            logt=3.76,
            logg=20.0,  # Way beyond bounds
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
        )

        assert np.all(np.isnan(sed))

    def test_out_of_bounds_feh_returns_nan(self, predictor):
        """Out-of-bounds metallicity should return NaN."""
        sed = predictor.sed(
            logt=3.76,
            logg=4.4,
            feh_surf=-100.0,  # Way beyond bounds
            logl=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
        )

        assert np.all(np.isnan(sed))

    def test_out_of_bounds_av_returns_nan(self, predictor):
        """Out-of-bounds extinction should return NaN."""
        sed = predictor.sed(
            logt=3.76,
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=1000.0,  # Way beyond bounds
            rv=3.3,
            dist=1000.0,
        )

        assert np.all(np.isnan(sed))

    def test_nan_input_returns_nan(self, predictor):
        """NaN input parameters should return NaN output."""
        sed = predictor.sed(
            logt=np.nan,
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.0,
            rv=3.3,
            dist=1000.0,
        )

        assert np.all(np.isnan(sed))

    def test_in_bounds_returns_finite(self, predictor):
        """In-bounds parameters should return finite output (contrast test)."""
        sed = predictor.sed(
            logt=3.76,
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )

        assert np.all(np.isfinite(sed))

    def test_sed_batch_out_of_bounds(self, predictor):
        """sed_batch should return NaN for out-of-bounds stars."""
        # Mix of valid and invalid parameters
        logt = np.array([3.76, 6.0, 3.80])  # Second is out of bounds
        logg = np.array([4.4, 4.4, 4.3])
        feh_surf = np.array([0.0, 0.0, -0.5])
        logl = np.array([0.0, 0.0, 0.5])
        afe = np.array([0.0, 0.0, 0.0])

        seds = predictor.sed_batch(
            logt=logt,
            logg=logg,
            feh_surf=feh_surf,
            logl=logl,
            afe=afe,
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )

        assert seds.shape == (3, predictor.NFILT)
        # First and third should be finite
        assert np.all(np.isfinite(seds[0]))
        assert np.all(np.isfinite(seds[2]))
        # Second should be NaN (out of bounds)
        assert np.all(np.isnan(seds[1]))
