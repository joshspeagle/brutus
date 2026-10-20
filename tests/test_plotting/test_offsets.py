"""Tests of `photometric_offsets` and `photometric_offsets_2d`.

The data are synthetic but run through the real `get_seds` and `magnitude`,
so the tests check the plotted numbers.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.analysis import photometric_offsets as analysis_photometric_offsets
from brutus.plotting.offsets import photometric_offsets, photometric_offsets_2d
from brutus.utils.photometry import inv_magnitude


@pytest.fixture
def no_tight_layout(monkeypatch):
    """Skip text layout, which dominates the run time of the data checks."""
    monkeypatch.setattr(plt, "tight_layout", lambda *args, **kwargs: None)


def _offsets_dataset(nobj=30, nfilt=6, nsamps=8, seed=3):
    """Deterministic synthetic dataset exercised with the real (unmocked)
    get_seds and magnitude."""
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


def _contaminated_band0_dataset(delta=0.10, n_obj=200, nsamps=300, seed=17):
    """Posterior draws fitted to all 5 bands, with a flux offset of
    1 + delta injected into band 0.

    The models form a 1-D family in overall brightness, and the draws of
    each object are exact samples from its Gaussian likelihood.
    """
    rng = np.random.default_rng(seed)
    nfilt, n_models = 5, 121
    base = np.array([16.0, 15.5, 15.2, 15.0, 14.8])
    models = np.zeros((n_models, nfilt, 3))
    models[:, :, 0] = base + np.linspace(-0.3, 0.3, n_models)[:, None]
    models[:, :, 1] = 1.0
    models[:, :, 2] = 0.1
    fluxes = 10 ** (-0.4 * models[:, :, 0])  # at 1 kpc with A(V) = 0
    err = np.tile(0.02 * 10 ** (-0.4 * base), (n_obj, 1))
    phot = 10 ** (-0.4 * base) + err * rng.standard_normal((n_obj, nfilt))
    phot[:, 0] *= 1.0 + delta
    lnp = -0.5 * np.sum(((phot[:, None] - fluxes) / err[:, None]) ** 2, axis=2)
    p = np.exp(lnp - lnp.max(axis=1, keepdims=True))
    idxs = np.array([rng.choice(n_models, nsamps, p=pk / pk.sum()) for pk in p])
    return (
        phot,
        err,
        np.ones((n_obj, nfilt), dtype=bool),
        models,
        idxs,
        np.zeros((n_obj, nsamps)),
        np.full((n_obj, nsamps), 3.1),
        np.ones((n_obj, nsamps)),
    )


def _panel_hist_arrays(axes, nfilt):
    """Extract the per-panel hist2d QuadMesh arrays from photometric_offsets."""
    ax = axes.flatten()
    out = []
    for i in range(nfilt):
        arr = np.ma.filled(np.asanyarray(ax[i].collections[0].get_array()), np.nan)
        out.append(np.array(arr, dtype=float))
    return out


@pytest.mark.usefixtures("no_tight_layout")
class TestPhotometricOffsetsFixes:
    """Magnitude-space offset arithmetic, integer masks, and finiteness
    selection in photometric_offsets / photometric_offsets_2d."""

    @staticmethod
    def _run(d, phot, err, mask, flux, **extra):
        return _run_1d(
            dict(d, flux=phot, fluxerr=err, mask=mask),
            flux=flux,
            xspan=[(4.0, 9.0)] * d["nfilt"],
            yspan=[(-3.0, 3.0)] * d["nfilt"],
            bins=20,
            **extra,
        )

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
            return _run_2d(dict(d, mask=mask), x, y, bins=4)

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
    if "fig" not in kwargs:
        kwargs["fig"] = plt.subplots(2, 3)
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
    if "fig" not in kwargs:
        kwargs["fig"] = plt.subplots(2, 3)
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


@pytest.mark.usefixtures("no_tight_layout")
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

    def test_two_bins_for_many_bands_raise(self):
        """Regression (#170): a length-2 `bins` raised TypeError."""
        with pytest.raises(ValueError, match="`bins`"):
            _run_1d(_offsets_dataset(), bins=(10, 12))

    def test_band_that_no_object_observes_is_left_blank(self):
        """Regression (#170): an empty band raised IndexError."""
        d = _offsets_dataset()
        d["mask"][:, 5] = False
        fig, axes = plt.subplots(2, 3)
        photometric_offsets(
            d["flux"],
            d["fluxerr"],
            d["mask"],
            d["models"],
            d["idxs"],
            d["reds"],
            d["dreds"],
            d["dists"],
            bins=5,
            titles=list("abcdef"),
            fig=(fig, axes),
        )
        panel = axes.flatten()[5]
        plt.close(fig)
        assert panel.get_title() == "f"
        assert len(panel.collections) == 0
        assert all(len(ax.collections) == 1 for ax in axes.flatten()[:5])

    @pytest.mark.parametrize("plot", ["1d", "2d"])
    def test_per_sample_weights_without_objects_raise(self, plot):
        """Regression (#170): `(Nsamps,)` weights were read as per-object
        weights when Nsamps == Nobj, and failed in `reshape` otherwise."""
        d = _offsets_dataset()
        weights = np.ones(d["idxs"].shape[1])
        with pytest.raises(ValueError, match="`weights`"):
            if plot == "1d":
                _run_1d(d, weights=weights)
            else:
                _run_2d(d, np.zeros(d["nobj"]), np.zeros(d["nobj"]), weights=weights)

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


@pytest.mark.usefixtures("no_tight_layout")
class TestPhotometricOffsets2D:
    """Offset maps of `photometric_offsets_2d`."""

    @staticmethod
    def _xy(d):
        rng = np.random.RandomState(5)
        return rng.uniform(-2, 2, d["nobj"]), rng.uniform(-1, 1, d["nobj"])

    def test_offset_in_one_band_shifts_only_its_map(self):
        """A flux offset c in band 0 adds 2.5 log10(c) to that band's map
        and leaves the others unchanged: the weights use the data as given."""
        d = _offsets_dataset()
        x, y = self._xy(d)
        offset = np.ones(d["nfilt"])
        offset[0] = 1.1
        base = _run_2d(d, x, y)
        shifted = _run_2d(d, x, y, offset=offset)
        np.testing.assert_allclose(
            shifted[0], base[0] + 2.5 * np.log10(1.1), rtol=1e-10, equal_nan=True
        )
        np.testing.assert_array_equal(shifted[1], base[1])

    def test_plotted_median_matches_analysis_offset(self):
        """With one bin, the map of a fitted band is the weighted median of
        mag_pred - mag_obs. It equals -2.5 log10 of the offset from
        `brutus.analysis.photometric_offsets` within the bootstrap error.

        Regression: weighting by L_-i alone, as the plots did, keeps part of
        the pull of band 0 on the draws and misses by about 0.01 mag here.
        """
        data = _contaminated_band0_dataset()
        offsets, errors, _ = analysis_photometric_offsets(
            *data, verbose=False, rng=np.random.default_rng(1)
        )
        x = np.linspace(0.0, 1.0, len(data[0]))
        d = dict(
            zip(["flux", "fluxerr", "mask", "models", "idxs", "reds", "dreds"], data)
        )
        d.update(dists=data[7], nfilt=data[0].shape[1])
        maps = _run_2d(d, x, x, bins=1)
        expected = -2.5 * np.log10(offsets[0])
        tol = 3 * 2.5 / np.log(10) * errors[0] / offsets[0]
        assert abs(maps[0][0, 0] - expected) < tol

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

    def test_bins_with_zero_total_weight_are_masked(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        weights = (x > 0).astype(float)
        for arr in _run_2d(d, x, y, weights=weights, plot_thresh=1.0):
            assert np.all(np.isnan(arr[:, 0]))
            assert np.any(np.isfinite(arr))

    def test_two_element_bins_set_the_map_shape(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        for arr in _run_2d(d, x, y, bins=(4, 2)):
            assert arr.shape == (2, 4)  # imshow of offset2d.T

    def test_titles_give_the_offset_in_fixed_point(self):
        """Regression (#170): a 12% offset was titled `1.2e+01%`."""
        d = _offsets_dataset()
        x, y = self._xy(d)
        offset = np.ones(d["nfilt"])
        offset[0] = 1.12
        fig, axes = plt.subplots(2, 3)
        photometric_offsets_2d(
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
            bins=3,
            offset=offset,
            fig=(fig, axes),
        )
        titles = [ax.get_title() for ax in axes.flatten()[:2]]
        plt.close(fig)
        assert titles == ["Band 0 (+12.0% offset)", "Band 1 (+0.0% offset)"]

    def test_bins_of_length_nfilt_are_per_band_with_two_bands(self):
        """Regression (#170): with 2 bands, per-band `bins` became a pair."""
        d = _offsets_dataset(nfilt=2)
        x, y = self._xy(d)
        shapes = [arr.shape for arr in _run_2d(d, x, y, bins=[2, 3])]
        assert shapes == [(2, 2), (3, 3)]

    def test_magnitude_input_matches_flux_input(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        flux_maps = _run_2d(d, x, y)
        mag = dict(d, flux=d["mag"], fluxerr=d["magerr"])
        mag_maps = _run_2d(mag, x, y, flux=False)
        for a, b in zip(flux_maps, mag_maps):
            np.testing.assert_allclose(a, b, rtol=1e-10, equal_nan=True)

    def test_bins_per_filter_set_each_map_shape(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        bins = [2, 3, 4, 2, 3, 4]
        for arr, nbin in zip(_run_2d(d, x, y, bins=bins), bins):
            assert arr.shape == (nbin, nbin)

    def test_given_figure_is_returned(self):
        d = _offsets_dataset()
        x, y = self._xy(d)
        figure = plt.subplots(2, 3)
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
            bins=3,
            fig=figure,
        )
        assert fig is figure[0] and axes is figure[1]
        plt.close(fig)


class TestDefaultFigure:
    """The default figure and layout path, without the layout patch."""

    def test_xlabel_labels_every_panel(self):
        d = _offsets_dataset()
        fig, axes = photometric_offsets(
            d["flux"],
            d["fluxerr"],
            d["mask"],
            d["models"],
            d["idxs"],
            d["reds"],
            d["dreds"],
            d["dists"],
            x=np.zeros(d["nobj"]),
            xlabel="Color",
            bins=5,
        )  # default figure: 2 x 5 panels, then tight_layout
        labels = [ax.get_xlabel() for ax in axes.flatten()[: d["nfilt"]]]
        plt.close(fig)
        assert labels == ["Color"] * d["nfilt"]

    def test_2d_default_figure_has_a_panel_per_filter(self):
        d = _offsets_dataset()
        rng = np.random.RandomState(5)
        fig, axes = photometric_offsets_2d(
            d["flux"],
            d["fluxerr"],
            d["mask"],
            d["models"],
            d["idxs"],
            d["reds"],
            d["dreds"],
            d["dists"],
            rng.uniform(-2, 2, d["nobj"]),
            rng.uniform(-1, 1, d["nobj"]),
            bins=3,
            plot_thresh=1.0,
        )
        n_images = [len(ax.images) for ax in axes.flatten()]
        plt.close(fig)
        assert axes.shape == (2, 5)
        assert n_images == [1] * d["nfilt"] + [0] * (10 - d["nfilt"])
