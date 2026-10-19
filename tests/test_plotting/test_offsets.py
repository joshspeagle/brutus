"""Tests of `photometric_offsets` and `photometric_offsets_2d`.

The data are synthetic but run through the real `get_seds`, `magnitude` and
`phot_loglike`, so the tests check the plotted numbers.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.plotting.offsets import photometric_offsets, photometric_offsets_2d
from brutus.utils.photometry import inv_magnitude


def _offsets_dataset(nobj=30, nfilt=6, nsamps=8, seed=3):
    """Deterministic synthetic dataset exercised with the real (unmocked)
    get_seds / magnitude / phot_loglike stack."""
    rng = np.random.RandomState(seed)
    nmodels = 40
    models = np.zeros((nmodels, nfilt, 3))
    models[:, :, 0] = rng.uniform(4.0, 7.0, (nmodels, nfilt))
    models[:, :, 1] = rng.uniform(0.8, 1.2, (nmodels, nfilt))
    models[:, :, 2] = rng.uniform(-0.05, 0.05, (nmodels, nfilt))
    idxs = rng.randint(0, nmodels, (nobj, nsamps))
    reds = rng.uniform(0.1, 0.5, (nobj, nsamps))
    dreds = rng.normal(3.3, 0.1, (nobj, nsamps))
    dists = rng.uniform(0.8, 2.0, (nobj, nsamps))
    mag = rng.uniform(5.0, 8.0, (nobj, nfilt))
    magerr = np.full((nobj, nfilt), 0.05)
    flux, fluxerr = inv_magnitude(mag, magerr)
    mask = np.ones((nobj, nfilt), dtype=bool)
    return dict(
        models=models,
        idxs=idxs,
        reds=reds,
        dreds=dreds,
        dists=dists,
        mag=mag,
        magerr=magerr,
        flux=flux,
        fluxerr=fluxerr,
        mask=mask,
        nobj=nobj,
        nfilt=nfilt,
    )


def _panel_hist_arrays(axes, nfilt):
    """Extract the per-panel hist2d QuadMesh arrays from photometric_offsets."""
    ax = axes.flatten()
    out = []
    for i in range(nfilt):
        arr = np.ma.filled(np.asanyarray(ax[i].collections[0].get_array()), np.nan)
        out.append(np.array(arr, dtype=float))
    return out


class TestPhotometricOffsetsFixes:
    """Magnitude-space offset arithmetic, integer masks, and finiteness
    selection in photometric_offsets / photometric_offsets_2d."""

    SPANS = dict(xspan=None, yspan=None)

    def _run(self, d, phot, err, mask, flux, **extra):
        xspan = [(4.0, 9.0)] * d["nfilt"]
        yspan = [(-3.0, 3.0)] * d["nfilt"]
        fig, axes = photometric_offsets(
            phot,
            err,
            mask,
            d["models"],
            d["idxs"],
            d["reds"],
            d["dreds"],
            d["dists"],
            flux=flux,
            xspan=xspan,
            yspan=yspan,
            bins=20,
            **extra,
        )
        arrays = _panel_hist_arrays(axes, d["nfilt"])
        plt.close(fig)
        return arrays

    def test_flux_false_matches_flux_true(self):
        """The same photometry passed as magnitudes (flux=False) or fluxes
        (flux=True) must produce identical diagnostics; the old code added
        the multiplicative flux offset (default 1.0) to every magnitude."""
        d = _offsets_dataset()
        arrays_flux = self._run(d, d["flux"], d["fluxerr"], d["mask"], flux=True)
        arrays_mag = self._run(d, d["mag"], d["magerr"], d["mask"], flux=False)
        for a, b in zip(arrays_flux, arrays_mag):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)

    def test_flux_false_applies_offset_in_mag_space(self):
        """A multiplicative flux offset must enter magnitudes as
        -2.5 log10(offset), matching the flux=True path exactly."""
        d = _offsets_dataset()
        offset = np.linspace(0.95, 1.05, d["nfilt"])
        arrays_flux = self._run(
            d, d["flux"], d["fluxerr"], d["mask"], flux=True, offset=offset
        )
        arrays_mag = self._run(
            d, d["mag"], d["magerr"], d["mask"], flux=False, offset=offset
        )
        for a, b in zip(arrays_flux, arrays_mag):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)

    def test_integer_mask_matches_boolean_mask(self):
        """A documented 0/1 integer mask must behave like a boolean mask
        instead of fancy-indexing rows 0/1."""
        d = _offsets_dataset()
        rng = np.random.RandomState(11)
        mask_bool = np.ones((d["nobj"], d["nfilt"]), dtype=bool)
        # Knock out one random band per object (>= 4 observed bands remain).
        mask_bool[np.arange(d["nobj"]), rng.randint(0, d["nfilt"], d["nobj"])] = False
        # Give masked bands finite values so only mask semantics differ.
        arrays_bool = self._run(d, d["flux"], d["fluxerr"], mask_bool, flux=True)
        arrays_int = self._run(
            d, d["flux"], d["fluxerr"], mask_bool.astype(int), flux=True
        )
        for a, b in zip(arrays_bool, arrays_int):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)

    def test_nonfinite_flux_in_masked_bands_does_not_empty_selection(self):
        """Objects with nonpositive flux in *unobserved* bands (the case masks
        exist for) must still populate every panel."""
        d = _offsets_dataset()
        rng = np.random.RandomState(13)
        flux = d["flux"].copy()
        mask = np.ones((d["nobj"], d["nfilt"]), dtype=bool)
        # Each object is missing one band, encoded as flux = 0 (-> mag = inf).
        missing = rng.randint(0, d["nfilt"], d["nobj"])
        mask[np.arange(d["nobj"]), missing] = False
        flux[np.arange(d["nobj"]), missing] = 0.0

        arrays = self._run(d, flux, d["fluxerr"], mask, flux=True)
        for arr in arrays:
            assert np.nansum(arr) > 0  # every panel keeps objects

    def test_2d_integer_mask_matches_boolean_mask(self):
        """photometric_offsets_2d: integer masks must not bitwise-invert
        (`~1 == -2`) onto the wrong objects."""
        d = _offsets_dataset()
        rng = np.random.RandomState(17)
        x = rng.uniform(-2, 2, d["nobj"])
        y = rng.uniform(-1, 1, d["nobj"])
        mask_bool = np.ones((d["nobj"], d["nfilt"]), dtype=bool)
        mask_bool[np.arange(d["nobj"]), rng.randint(0, d["nfilt"], d["nobj"])] = False

        def run(mask):
            fig, axes = photometric_offsets_2d(
                d["flux"],
                d["fluxerr"],
                mask,
                d["models"],
                d["idxs"],
                d["reds"],
                d["dreds"],
                d["dists"],
                x,
                y,
                bins=4,
                plot_thresh=1.0,
            )
            ax = axes.flatten()
            arrays = [
                np.ma.filled(np.asanyarray(ax[i].images[0].get_array()), np.nan)
                for i in range(d["nfilt"])
            ]
            plt.close(fig)
            return arrays

        arrays_bool = run(mask_bool)
        arrays_int = run(mask_bool.astype(int))
        for a, b in zip(arrays_bool, arrays_int):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)


def _panel_image_arrays(axes, nfilt):
    """Extract the per-panel offset maps from photometric_offsets_2d."""
    ax = axes.flatten()
    return [
        np.ma.filled(np.asanyarray(ax[i].images[0].get_array()), np.nan)
        for i in range(nfilt)
    ]


def _run_1d(d, **kwargs):
    kwargs.setdefault("xspan", [(-10.0, 20.0)] * d["nfilt"])
    kwargs.setdefault("yspan", [(-20.0, 20.0)] * d["nfilt"])
    fig, axes = photometric_offsets(
        d["flux"],
        d["fluxerr"],
        d["mask"],
        d["models"],
        d["idxs"],
        d["reds"],
        d["dreds"],
        d["dists"],
        **kwargs,
    )
    arrays = _panel_hist_arrays(axes, d["nfilt"])
    plt.close(fig)
    return arrays


def _run_2d(d, x, y, **kwargs):
    kwargs.setdefault("bins", 3)
    kwargs.setdefault("plot_thresh", 1.0)
    fig, axes = photometric_offsets_2d(
        d["flux"],
        d["fluxerr"],
        d["mask"],
        d["models"],
        d["idxs"],
        d["reds"],
        d["dreds"],
        d["dists"],
        x,
        y,
        **kwargs,
    )
    arrays = _panel_image_arrays(axes, d["nfilt"])
    plt.close(fig)
    return arrays


class TestPhotometricOffsets1D:
    """Histogram weights and binning of `photometric_offsets`."""

    def test_each_object_adds_unit_weight_to_each_panel(self):
        d = _offsets_dataset()
        for arr in _run_1d(d, bins=10):
            assert np.nansum(arr) == pytest.approx(d["nobj"])

    def test_per_object_weights_scale_each_object(self):
        d = _offsets_dataset()
        weights = np.where(np.arange(d["nobj"]) < 10, 1.0, 0.0)
        for arr in _run_1d(d, bins=10, weights=weights):
            assert np.nansum(arr) == pytest.approx(10.0)

    def test_constant_x_falls_in_one_column(self):
        d = _offsets_dataset()
        arrays = _run_1d(
            d, x=np.zeros(d["nobj"]), xspan=[(-1.0, 1.0)] * d["nfilt"], bins=2
        )
        for arr in arrays:  # shape (ny, nx); x = 0 falls in the second column
            assert np.nansum(arr[:, 0]) == 0.0
            assert np.nansum(arr[:, 1]) == pytest.approx(d["nobj"])

    def test_bins_per_filter_set_each_panel_shape(self):
        d = _offsets_dataset()
        bins = [3, 4, 5, 6, 7, 8]
        for arr, nbin in zip(_run_1d(d, bins=bins), bins):
            assert arr.size == nbin * nbin

    def test_given_figure_is_returned(self):
        d = _offsets_dataset()
        figure = plt.subplots(2, 3)
        fig, axes = photometric_offsets(
            d["flux"],
            d["fluxerr"],
            d["mask"],
            d["models"],
            d["idxs"],
            d["reds"],
            d["dreds"],
            d["dists"],
            bins=5,
            fig=figure,
        )
        assert fig is figure[0] and axes is figure[1]
        plt.close(fig)


class TestPhotometricOffsets2D:
    """Offset maps of `photometric_offsets_2d`."""

    @staticmethod
    def _xy(d):
        rng = np.random.RandomState(5)
        return rng.uniform(-2, 2, d["nobj"]), rng.uniform(-1, 1, d["nobj"])

    def test_offset_in_one_band_shifts_only_its_map(self):
        """A flux offset c in band 0 adds 2.5 log10(c) to that band's map.

        The weights for band 0 use only the other bands, so they do not
        change.
        """
        d = _offsets_dataset()
        x, y = self._xy(d)
        offset = np.ones(d["nfilt"])
        offset[0] = 1.1
        base = _run_2d(d, x, y)
        shifted = _run_2d(d, x, y, offset=offset)
        np.testing.assert_allclose(
            shifted[0], base[0] + 2.5 * np.log10(1.1), rtol=1e-10, equal_nan=True
        )

    def test_shared_1d_weights_match_per_sample_weights(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        w1 = np.random.RandomState(6).uniform(0.5, 1.5, d["nobj"])
        w2 = np.repeat(w1[:, None], d["idxs"].shape[1], axis=1)
        for a, b in zip(_run_2d(d, x, y, weights=w1), _run_2d(d, x, y, weights=w2)):
            np.testing.assert_array_equal(a, b)

    def test_bins_with_too_few_objects_are_masked(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        for arr in _run_2d(d, x, y, plot_thresh=d["nobj"] + 1):
            assert np.all(np.isnan(arr))

    def test_two_element_bins_set_the_map_shape(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        for arr in _run_2d(d, x, y, bins=(4, 2)):
            assert arr.shape == (2, 4)  # imshow of offset2d.T
