#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for audit fixes in `brutus.plotting`.

Each test here fails on the pre-fix code:
- missing scale-to-distance Jacobian (d^3) when reweighting SAR draws
  (`cornerplot`, `bin_pdfs_distred`),
- `dist_vs_red` silently ignoring `weights` and multi-object input,
- CDF accumulated along the wrong axis in `bin_pdfs_distred`,
- SAR path crash with a custom `lndistprior` and `coord=None`,
- `cornerplot` mutating caller kwargs dicts / freezing per-panel contour flags,
- `photometric_offsets*` magnitude-space offset arithmetic, integer masks,
  and finiteness selection over unobserved bands.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from brutus.plotting.binning import bin_pdfs_distred
from brutus.plotting.corner import cornerplot
from brutus.plotting.distance import dist_vs_red
from brutus.plotting.offsets import photometric_offsets, photometric_offsets_2d
from brutus.utils.photometry import inv_magnitude

# np.trapezoid is NumPy 2.0+; fall back to np.trapz on NumPy 1.x.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class TestScaleToDistanceJacobian:
    """The SAR regeneration paths draw from a Gaussian proposal in scale
    space (s = d^-2) and reweight by a *distance*-space prior, which requires
    the |dd/ds| ~ d^3 Jacobian. Without it the recovered distance posterior
    is biased low. These tests push draws from a known scale-space Gaussian
    through the reweighting and compare against the analytic expectation."""

    # Scale-space Gaussian likelihood ~ N(S0, SIG_S^2); distance prior
    # ~ N(D_MU, D_SIG^2); posterior restricted to [DLO, DHI].
    S0, SIG_S = 1.0, 0.3
    D_MU, D_SIG = 2.0, 0.5
    DLO, DHI = 0.5, 4.0

    @classmethod
    def _lndistprior(cls, d, coord):
        return -0.5 * ((d - cls.D_MU) / cls.D_SIG) ** 2

    @classmethod
    def _analytic_means(cls):
        """Quadrature means of the correct posterior and of the biased
        (Jacobian-less) posterior the old code produced."""
        d = np.linspace(cls.DLO, cls.DHI, 200001)
        like = np.exp(-0.5 * ((d**-2 - cls.S0) / cls.SIG_S) ** 2)
        prior = np.exp(-0.5 * ((d - cls.D_MU) / cls.D_SIG) ** 2)
        p = like * prior
        mean_correct = _trapezoid(d * p, d) / _trapezoid(p, d)
        # Old code: draws land in d with density ~ like * d^-3 (from the
        # scale-space proposal) but are reweighted by the prior alone.
        p_biased = like * prior * d**-3.0
        mean_biased = _trapezoid(d * p_biased, d) / _trapezoid(p_biased, d)
        return mean_correct, mean_biased

    def test_bin_pdfs_distred_sar_matches_analytic_distance_posterior(self):
        """bin_pdfs_distred SAR path: binned distance marginal must match the
        analytic posterior mean, not the d^-3-biased one."""
        rstate = np.random.RandomState(42)
        nsamps = 1
        scales = np.full((1, nsamps), self.S0)
        avs = np.full((1, nsamps), 1.0)
        rvs = np.full((1, nsamps), 3.3)
        covs = np.tile(np.diag([self.SIG_S**2, 1e-10, 1e-10]), (1, nsamps, 1, 1))

        nxbin = 400
        binned, xedges, yedges = bin_pdfs_distred(
            (scales, avs, rvs, covs),
            dist_type="distance",
            lndistprior=self._lndistprior,
            coord=[(90.0, 20.0)],
            avlim=(0.5, 1.5),
            span=((0.5, 1.5), (self.DLO, self.DHI)),
            bins=(nxbin, 4),
            smooth=1e-6,
            Nr=100000,
            rstate=rstate,
        )

        marg = binned[0].sum(axis=1)  # marginal over reddening
        centers = 0.5 * (xedges[1:] + xedges[:-1])
        mean_rec = np.sum(centers * marg) / np.sum(marg)

        mean_correct, mean_biased = self._analytic_means()
        # Sanity check the test has power: the bias is >> the tolerance.
        assert abs(mean_correct - mean_biased) > 0.2
        assert abs(mean_rec - mean_correct) < 0.05
        assert abs(mean_rec - mean_biased) > 0.15

    def test_cornerplot_sar_matches_analytic_distance_posterior(self):
        """cornerplot SAR path: the plotted distance histogram must match the
        analytic posterior mean, not the d^-3-biased one."""
        rstate = np.random.RandomState(7)
        nsamps = 3000
        nmodels = 10

        params = np.empty(nmodels, dtype=np.dtype([("mass", "f4")]))
        params["mass"] = np.linspace(0.8, 1.2, nmodels)
        idxs = rstate.choice(nmodels, nsamps)

        scales = np.full(nsamps, self.S0)
        avs = np.full(nsamps, 1.0)
        rvs = np.full(nsamps, 3.3)
        covs = np.tile(np.diag([self.SIG_S**2, 1e-10, 1e-10]), (nsamps, 1, 1))

        # Dimensions: mass, Av, Rv, Parallax, Distance.
        span = [
            (0.7, 1.3),
            (0.9, 1.1),
            (3.2, 3.4),
            (0.0, 2.5),
            (self.DLO, self.DHI),
        ]
        fig, axes = cornerplot(
            idxs,
            (scales, avs, rvs, covs),
            params,
            lndistprior=self._lndistprior,
            applied_parallax=False,
            Nr=200,
            smooth=50,
            span=span,
            quantiles=None,
            rstate=rstate,
        )
        # Read the distance histogram back off the last diagonal panel.
        ax = axes[4, 4]
        heights = np.array([p.get_height() for p in ax.patches])
        centers = np.array([p.get_x() + 0.5 * p.get_width() for p in ax.patches])
        plt.close(fig)
        mean_rec = np.sum(centers * heights) / np.sum(heights)

        mean_correct, mean_biased = self._analytic_means()
        assert abs(mean_rec - mean_correct) < 0.08
        assert abs(mean_rec - mean_biased) > 0.15


class TestBinPdfsDistredFixes:
    """CDF axis, custom-prior coord handling, and sample weights."""

    def test_cdf_accumulates_along_reddening_axis(self):
        """cdf=True must equal the PDF cumulated along reddening (axis 2)
        within each distance column, per the docstring semantics."""
        rng = np.random.RandomState(42)
        nobj, nsamps = 2, 200
        dists = rng.lognormal(np.log(2.0), 0.3, (nobj, nsamps))
        reds = rng.uniform(0.0, 2.0, (nobj, nsamps))
        dreds = np.full((nobj, nsamps), 3.3)
        data = (dists, reds, dreds)
        kwargs = dict(bins=(15, 12), smooth=0.01, dist_type="distance")

        pdf_vals, xe, ye = bin_pdfs_distred(data, cdf=False, **kwargs)
        cdf_vals, xe2, ye2 = bin_pdfs_distred(data, cdf=True, **kwargs)

        expected = pdf_vals.cumsum(axis=2)
        np.testing.assert_allclose(cdf_vals, expected, rtol=1e-5, atol=1e-7)
        # Monotone non-decreasing along reddening within each distance column.
        assert np.all(np.diff(cdf_vals, axis=2) >= -1e-6)

    def test_sar_custom_prior_without_coord(self):
        """A custom lndistprior must be usable with coord=None (the guard
        only requires coord for the default galactic prior)."""
        rstate = np.random.RandomState(0)
        nobj, nsamps = 2, 5
        scales = np.full((nobj, nsamps), 1.0)
        avs = np.full((nobj, nsamps), 1.0)
        rvs = np.full((nobj, nsamps), 3.3)
        covs = np.tile(np.diag([0.05**2, 1e-6, 1e-6]), (nobj, nsamps, 1, 1))

        binned, xe, ye = bin_pdfs_distred(
            (scales, avs, rvs, covs),
            lndistprior=lambda d, c: np.zeros_like(d),
            bins=(10, 10),
            Nr=20,
            rstate=rstate,
        )
        assert binned.shape == (nobj, 10, 10)
        assert np.all(np.isfinite(binned))

    def test_sar_weights_zero_out_samples(self):
        """Per-sample weights in the SAR path: zeroing the second sample must
        reproduce (up to normalization) the run containing only the first."""
        span = ((0.5, 1.5), (0.5, 4.0))
        kwargs = dict(
            dist_type="distance",
            lndistprior=lambda d, c: np.zeros_like(d),
            avlim=(0.5, 1.5),
            span=span,
            bins=(20, 5),
            smooth=1e-6,
            Nr=200,
        )
        scales = np.array([[1.0, 0.25]])
        avs = np.full((1, 2), 1.0)
        rvs = np.full((1, 2), 3.3)
        covs = np.tile(np.diag([0.05**2, 1e-8, 1e-8]), (1, 2, 1, 1))

        b_w, _, _ = bin_pdfs_distred(
            (scales, avs, rvs, covs),
            weights=np.array([[1.0, 0.0]]),
            rstate=np.random.RandomState(3),
            **kwargs,
        )
        b_first, _, _ = bin_pdfs_distred(
            (scales[:, :1], avs[:, :1], rvs[:, :1], covs[:, :1]),
            rstate=np.random.RandomState(3),
            **kwargs,
        )
        assert b_w.sum() > 0
        np.testing.assert_allclose(
            b_w[0] / b_w[0].sum(),
            b_first[0] / b_first[0].sum(),
            rtol=1e-4,
            atol=1e-7,
        )

    def test_weights_shape_validated(self):
        data = (np.ones((2, 5)), np.ones((2, 5)), np.full((2, 5), 3.3))
        with pytest.raises(ValueError, match="weights"):
            bin_pdfs_distred(data, weights=np.ones(5), bins=(5, 5))


class TestDistVsRedFixes:
    """`weights` forwarding and multi-object combination."""

    @staticmethod
    def _bimodal_data(seed=42, n=400):
        rng = np.random.RandomState(seed)
        dists = np.concatenate(
            [rng.uniform(0.6, 1.0, n // 2), rng.uniform(3.0, 4.0, n // 2)]
        )
        reds = rng.uniform(0.1, 1.9, n)
        dreds = np.full(n, 3.3)
        return dists, reds, dreds

    def test_weights_are_applied(self):
        """Zero/one weights must reproduce the histogram of the kept subset;
        previously `weights` was silently ignored."""
        n = 400
        data = self._bimodal_data(n=n)
        kwargs = dict(
            dist_type="distance",
            span=((0.0, 2.0), (0.4, 4.5)),
            bins=(20, 10),
            smooth=0.01,
        )
        w = np.zeros(n)
        w[: n // 2] = 1.0

        plt.figure()
        H_unw, _, _, _ = dist_vs_red(data, **kwargs)
        plt.close()
        plt.figure()
        H_w, _, _, _ = dist_vs_red(data, weights=w, **kwargs)
        plt.close()
        plt.figure()
        H_sub, _, _, _ = dist_vs_red(tuple(a[: n // 2] for a in data), **kwargs)
        plt.close()

        assert not np.allclose(H_w, H_unw)
        np.testing.assert_allclose(
            H_w / H_w.sum(), H_sub / H_sub.sum(), rtol=1e-4, atol=1e-7
        )

    def test_multi_object_combines_all_objects(self):
        """Multi-object input must average the per-object PDFs rather than
        silently plotting only object 0."""
        rng = np.random.RandomState(1)
        nsamps = 200
        # Two objects with clearly separated distance distributions.
        dists = np.vstack(
            [rng.uniform(0.6, 1.0, nsamps), rng.uniform(3.0, 4.0, nsamps)]
        )
        reds = rng.uniform(0.1, 1.9, (2, nsamps))
        dreds = np.full((2, nsamps), 3.3)
        data = (dists, reds, dreds)
        kwargs = dict(
            dist_type="distance",
            span=((0.0, 2.0), (0.4, 4.5)),
            bins=(20, 10),
            smooth=0.015,
        )

        plt.figure()
        H, xe, ye, _ = dist_vs_red(data, **kwargs)
        plt.close()
        binned, _, _ = bin_pdfs_distred(data, cdf=False, verbose=False, **kwargs)

        assert not np.allclose(H, binned[0])
        np.testing.assert_allclose(H, binned.mean(axis=0), rtol=1e-5, atol=1e-8)

    def test_multi_object_accepts_shared_1d_weights(self):
        """A 1-D `(Nsamps,)` weights array is documented as shared across all
        objects; previously the multi-object path forwarded it unbroadcast to
        `bin_pdfs_distred`, which raised a shape ValueError."""
        rng = np.random.RandomState(7)
        nsamps = 200
        dists = np.vstack(
            [rng.uniform(0.6, 1.0, nsamps), rng.uniform(3.0, 4.0, nsamps)]
        )
        reds = rng.uniform(0.1, 1.9, (2, nsamps))
        dreds = np.full((2, nsamps), 3.3)
        data = (dists, reds, dreds)
        kwargs = dict(
            dist_type="distance",
            span=((0.0, 2.0), (0.4, 4.5)),
            bins=(20, 10),
            smooth=0.015,
        )
        w = rng.uniform(0.5, 1.5, nsamps)

        plt.figure()
        H_shared, _, _, _ = dist_vs_red(data, weights=w, **kwargs)
        plt.close()
        plt.figure()
        H_tiled, _, _, _ = dist_vs_red(data, weights=np.tile(w, (2, 1)), **kwargs)
        plt.close()

        np.testing.assert_allclose(H_shared, H_tiled, rtol=1e-12, atol=0.0)

        # A 1-D weights array of the wrong length must fail loudly.
        with pytest.raises(ValueError, match="weights"):
            dist_vs_red(data, weights=w[:-1], **kwargs)


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


class TestCornerplotKwargsHandling:
    """cornerplot must not mutate caller kwargs dicts, and contour flags must
    be re-derived per panel from that panel's smoothing."""

    @staticmethod
    def _basic_inputs(nsamps=40, nmodels=20, seed=42):
        rng = np.random.RandomState(seed)
        params = np.empty(nmodels, dtype=np.dtype([("mass", "f4")]))
        params["mass"] = rng.uniform(0.5, 2.0, nmodels)
        idxs = rng.choice(nmodels, nsamps)
        dists = rng.lognormal(np.log(2.0), 0.3, nsamps)
        reds = rng.exponential(0.5, nsamps)
        dreds = rng.normal(3.3, 0.3, nsamps)
        return idxs, (dists, reds, dreds), params

    def test_caller_kwargs_dicts_not_mutated(self):
        idxs, data, params = self._basic_inputs()
        hist_kwargs = {}
        hist2d_kwargs = {}
        truth_kwargs = {}
        fig, _ = cornerplot(
            idxs,
            data,
            params,
            applied_parallax=False,
            span=[0.9] * 5,
            hist_kwargs=hist_kwargs,
            hist2d_kwargs=hist2d_kwargs,
            truth_kwargs=truth_kwargs,
        )
        plt.close(fig)
        assert hist_kwargs == {}
        assert hist2d_kwargs == {}
        assert truth_kwargs == {}

    def test_contour_flags_rederived_per_panel(self, monkeypatch):
        """With mixed int/float smoothing, int/int panels must get plain
        histograms (no contours) even after a float panel was drawn first."""
        idxs, data, params = self._basic_inputs()
        calls = []

        def spy_hist2d(y, x, ax=None, fill_contours=None, plot_contours=None, **kw):
            calls.append((fill_contours, plot_contours))

        monkeypatch.setattr("brutus.plotting.corner.hist2d", spy_hist2d)

        smooth = [0.05, 10, 10, 10, 10]
        fig, _ = cornerplot(
            idxs,
            data,
            params,
            applied_parallax=False,
            span=[0.9] * 5,
            smooth=smooth,
        )
        plt.close(fig)

        expected = []
        for i in range(5):
            for j in range(i):
                flag = not (isinstance(smooth[i], int) and isinstance(smooth[j], int))
                expected.append((flag, flag))
        assert calls == expected
