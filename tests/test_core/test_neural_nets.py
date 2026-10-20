"""Tests of brutus.core.neural_nets."""

import h5py
import numpy as np
import numpy.testing as npt
import pytest
from conftest import requires_data

from brutus.core.neural_nets import FastNN, FastNNPredictor


@pytest.fixture
def nn_file():
    return requires_data("nn_c3k.h5")


class TestFastNNEncode:
    """Regression tests for ``FastNN.encode`` input-dimensionality dispatch.

    These do not require the neural-network data file -- they exercise the
    pure rescaling logic on a bare ``FastNN`` instance with synthetic bounds.
    """

    @staticmethod
    def _make_nn():
        from brutus.core.neural_nets import FastNN

        nn = FastNN.__new__(FastNN)
        # Six parameters with distinct, asymmetric bounds so that a
        # wrong-axis normalization yields a clearly different result.
        nn.xmin = np.array([3.0, 0.1, -2.0, -0.2, 0.0, 2.0])
        nn.xmax = np.array([4.0, 5.0, 0.5, 0.4, 6.0, 8.0])
        nn.xspan = nn.xmax - nn.xmin
        return nn

    def test_encode_1d_contract(self):
        """A 1D input of shape (Ninput,) returns (Ninput, 1)."""
        nn = self._make_nn()
        x = nn.xmin + 0.5 * nn.xspan
        out = nn.encode(x)
        assert out.shape == (6, 1)
        npt.assert_allclose(out[:, 0], (x - nn.xmin) / nn.xspan)

    @pytest.mark.parametrize("nsamp", [5, 6, 7])
    def test_encode_2d_per_parameter_normalization(self, nsamp):
        """2D batches normalize per parameter (row) for every batch size.

        Regression for the silent N == 6 bug: a (6, 6) batch used to
        broadcast against ``xmin[None, :]`` without raising, so the wrong
        (sample-axis) normalization was applied only when exactly 6 samples
        were supplied.
        """
        nn = self._make_nn()
        x = np.stack(
            [nn.xmin + ((j + 1) / (nsamp + 2)) * nn.xspan for j in range(nsamp)],
            axis=1,
        )  # shape (6, nsamp)
        out = nn.encode(x)
        expected = (x - nn.xmin[:, None]) / nn.xspan[:, None]
        assert out.shape == (6, nsamp)
        npt.assert_allclose(out, expected)


class TestFastNNPredictorDefaultFilters:
    """Test FastNNPredictor initialization with filters=None (line 354)."""

    def test_default_filters_initialization(self, nn_file):
        """When filters=None, predictor should use all FILTERS."""
        from brutus.core.neural_nets import FastNNPredictor
        from brutus.data.filters import FILTERS

        predictor = FastNNPredictor(filters=None, nnfile=nn_file, verbose=False)

        # Should have loaded all filters
        assert predictor.NFILT == len(FILTERS)
        npt.assert_array_equal(predictor.filters, np.array(FILTERS))

    def test_default_filters_produce_valid_seds(self, nn_file):
        """Default filter predictor should produce valid SED output."""
        from brutus.core.neural_nets import FastNNPredictor

        predictor = FastNNPredictor(filters=None, nnfile=nn_file, verbose=False)

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


class TestFastNNPredictorOutOfBounds:
    """Test out-of-bounds parameter handling (line 505)."""

    @pytest.fixture
    def predictor(self, nn_file):
        """Create a predictor with a few filters."""
        from brutus.core.neural_nets import FastNNPredictor

        # Use a small subset of filters for speed
        filters = ["PS_g", "PS_r", "PS_i"]
        return FastNNPredictor(filters=filters, nnfile=nn_file, verbose=False)

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


def _write_nn_file(path, per_filter_bounds, seed=42):
    """Write a synthetic NN file in the FastNN HDF5 layout."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for fname, (xmin, xmax) in per_filter_bounds.items():
            g = f.create_group(fname)
            g["w1"] = rng.random((16, 6))
            g["b1"] = rng.random((16, 1))
            g["w2"] = rng.random((16, 16))
            g["b2"] = rng.random((16, 1))
            g["w3"] = rng.random((1, 16))
            g["b3"] = rng.random((1, 1))
            g["xmin"] = np.asarray(xmin, dtype=float)
            g["xmax"] = np.asarray(xmax, dtype=float)


class TestLoadNNBoundsCheck:
    """The bounds check must compare per-filter bounds elementwise."""

    def test_coincident_bound_values_accepted(self, tmp_path):
        """A consistent file where two parameters share a bound value
        (afe_min == av_min == 0) must load; the old value-counting check
        rejected it because len(np.unique(xmin)) < 6."""
        path = str(tmp_path / "coincident.h5")
        bounds = (
            [2500.0, -1.0, -4.0, 0.0, 0.0, 2.0],  # afe_min == av_min == 0.0
            [5.0e4, 5.5, 0.5, 0.6, 6.0, 5.0],
        )
        _write_nn_file(path, {"f1": bounds, "f2": bounds})
        nn = FastNN(filters=["f1", "f2"], nnfile=path, verbose=False)
        np.testing.assert_array_equal(nn.xmin, bounds[0])
        np.testing.assert_array_equal(nn.xmax, bounds[1])

    def test_permuted_bounds_rejected(self, tmp_path):
        """Per-filter bounds that are permutations of the same six values
        are inconsistent and must raise; the old check accepted them."""
        path = str(tmp_path / "permuted.h5")
        b1 = (
            [2500.0, -1.0, -4.0, -0.2, 0.0, 2.0],
            [5.0e4, 5.5, 0.5, 0.6, 6.0, 5.0],
        )
        # Teff/logg bounds swapped for the second filter
        b2 = (
            [-1.0, 2500.0, -4.0, -0.2, 0.0, 2.0],
            [5.5, 5.0e4, 0.5, 0.6, 6.0, 5.0],
        )
        _write_nn_file(path, {"f1": b1, "f2": b2})
        with pytest.raises(ValueError, match="different"):
            FastNN(filters=["f1", "f2"], nnfile=path, verbose=False)

    def test_real_file_still_loads(self, nn_file):
        nn = FastNN(filters=["SDSS_g", "SDSS_r"], nnfile=nn_file, verbose=False)
        assert nn.xmin.shape == (6,)


class TestSedBatchSingleFilter:
    """sed_batch must work when the predictor has a single filter."""

    def test_single_filter_batch_matches_scalar(self, nn_file):
        pred = FastNNPredictor(filters=["SDSS_g"], nnfile=nn_file, verbose=False)
        n = 5
        seds = pred.sed_batch(
            logt=np.full(n, 3.76),
            logg=np.full(n, 4.4),
            feh_surf=np.zeros(n),
            logl=np.zeros(n),
            afe=np.zeros(n),
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        assert seds.shape == (n, 1)
        assert np.all(np.isfinite(seds))
        ref = pred.sed(
            logt=3.76,
            logg=4.4,
            feh_surf=0.0,
            logl=0.0,
            afe=0.0,
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        np.testing.assert_allclose(seds[:, 0], ref[0], rtol=1e-12)

    def test_single_valid_star_edge(self, nn_file):
        """n_valid == 1 (squeeze also drops the sample axis)."""
        pred = FastNNPredictor(filters=["SDSS_g"], nnfile=nn_file, verbose=False)
        seds = pred.sed_batch(
            logt=np.array([3.76, 99.0]),  # second star far out of bounds
            logg=np.array([4.4, 4.4]),
            feh_surf=np.zeros(2),
            logl=np.zeros(2),
            afe=np.zeros(2),
            av=0.1,
            rv=3.3,
            dist=1000.0,
        )
        assert seds.shape == (2, 1)
        assert np.isfinite(seds[0, 0])
        assert np.isnan(seds[1, 0])


def test_sed_batch_matches_loop_over_sed(nn_file):
    predictor = FastNNPredictor(filters=None, nnfile=nn_file, verbose=False)
    logt = np.array([3.76, 3.70, 3.85, 3.60, 5.50, 3.76])  # star 4 out of bounds
    logg = np.array([4.44, 4.60, 4.20, 2.50, 4.40, 9.00])  # star 5 out of bounds
    feh = np.array([0.0, -0.5, 0.2, -1.0, 0.0, 0.0])
    logl = np.array([0.0, -0.3, 0.5, 1.8, 0.0, 0.0])
    afe = np.zeros(6)
    filt_idxs = [0, 3, 5, 20]
    batch = predictor.sed_batch(
        logt, logg, feh, logl, afe, 0.4, 3.1, 2500.0, filt_idxs=filt_idxs
    )
    loop = np.array(
        [
            predictor.sed(
                logt=t,
                logg=g,
                feh_surf=f,
                logl=lum,
                afe=a,
                av=0.4,
                rv=3.1,
                dist=2500.0,
                filt_idxs=filt_idxs,
            )
            for t, g, f, lum, a in zip(logt, logg, feh, logl, afe)
        ]
    )
    assert np.isfinite(batch[:4]).all() and np.isnan(batch[4:]).all()
    npt.assert_allclose(batch, loop, rtol=1e-12, equal_nan=True)


def test_missing_filter_raises_value_error(tmp_path):
    """Report missing filters by name instead of exposing an HDF5 KeyError."""
    path = str(tmp_path / "missing-filter.h5")
    bounds = (
        [2500.0, -1.0, -4.0, -0.2, 0.0, 2.0],
        [5.0e4, 5.5, 0.5, 0.6, 6.0, 5.0],
    )
    _write_nn_file(path, {"SDSS_g": bounds})

    with pytest.raises(ValueError, match=r"filters: BOGUS_V"):
        FastNNPredictor(filters=["BOGUS_V"], nnfile=path, verbose=False)
