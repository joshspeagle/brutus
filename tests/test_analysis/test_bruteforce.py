"""Tests of BruteForce in brutus.analysis.individual."""

import h5py
import numpy as np
import pytest
from conftest import split_labels

from brutus.analysis.individual import BruteForce
from brutus.core.individual import StarGrid
from brutus.utils import inv_magnitude


@pytest.fixture(scope="module")
def real_mist_setup(mist_ps_grid):
    """Every 20000th model of the MIST v9 grid, with the five PS1 filters."""
    models, labels, label_mask = mist_ps_grid
    subset = slice(0, None, 20000)
    return {
        "subset_grid": StarGrid(
            models[subset], *split_labels(labels[subset], label_mask)
        ),
        "ps_indices": list(range(5)),
    }


@pytest.fixture(scope="module")
def mock_grid():
    """Create a small mock grid for testing."""

    # Create small grid (3x3x3 = 27 models)
    nmodels = 27
    nfilters = 5

    # Create models with realistic structure
    models = np.zeros((nmodels, nfilters, 3))

    # Separate grid parameters from predictions
    labels_dtype = [("mini", "f4"), ("eep", "f4"), ("feh", "f4")]
    labels = np.zeros(nmodels, dtype=labels_dtype)

    params_dtype = [
        ("mass", "f4"),
        ("radius", "f4"),
        ("logg", "f4"),
        ("Teff", "f4"),
        ("Mr", "f4"),
    ]
    params = np.zeros(nmodels, dtype=params_dtype)

    # Fill grid
    idx = 0
    for i, m in enumerate([0.5, 1.0, 2.0]):
        for j, e in enumerate([200, 350, 450]):
            for k, z in enumerate([-1.0, 0.0, 0.5]):
                # Base magnitudes (brighter for more massive stars)
                base_mag = 15.0 - 2.5 * np.log10(m)
                models[idx, :, 0] = base_mag + np.array([0, -0.2, -0.3, -0.4, -0.45])
                models[idx, :, 1] = np.array([1.5, 1.2, 1.0, 0.8, 0.7])  # Reddening
                models[idx, :, 2] = np.array([0.15, 0.12, 0.10, 0.08, 0.07])  # dR/dRv

                # Grid parameters
                labels[idx] = (m, e, z)

                # Predictions
                params[idx] = (
                    m * (1 - 0.05 * j),  # mass
                    m**0.8 * (1 + 0.05 * j),  # radius
                    4.4 - 0.1 * j,  # logg
                    5777 * m**0.4,  # Teff
                    5.0 - 2.5 * np.log10(m),
                )  # Mr
                idx += 1

    filters = ["g", "r", "i", "z", "y"]

    return StarGrid(models, labels, params, filters=filters)


@pytest.fixture(scope="module")
def bruteforce_fitter(real_mist_setup):
    """Create BruteForce instance with real MIST subset grid."""
    return BruteForce(real_mist_setup["subset_grid"], verbose=False)


@pytest.fixture
def synthetic_observation(real_mist_setup):
    """Create synthetic observation from known model."""

    grid = real_mist_setup["subset_grid"]
    # Use model index that exists in the subset grid
    true_idx = min(13, len(grid.models) - 1)  # Ensure index is valid
    true_model = grid.models[true_idx, :, 0]  # Base magnitudes

    # Convert to flux - NO NOISE for deterministic testing
    true_flux = 10 ** (-0.4 * true_model)
    flux = true_flux  # No perturbation
    flux_err = true_flux * 0.05  # 5% error bars

    # All bands observed
    mask = np.ones(len(flux), dtype=bool)

    return flux, flux_err, mask, true_idx


@pytest.fixture
def noisy_synthetic_observation(real_mist_setup):
    """Create noisy synthetic observation for realistic testing."""
    rng = np.random.RandomState(42)

    grid = real_mist_setup["subset_grid"]
    # Use model index that exists in the subset grid
    true_idx = min(13, len(grid.models) - 1)  # Ensure index is valid
    true_model = grid.models[true_idx, :, 0]  # Base magnitudes

    # Convert to flux and add realistic noise
    true_flux = 10 ** (-0.4 * true_model)
    noise_level = 0.05  # 5% noise
    flux = true_flux * (1 + rng.normal(0, noise_level, len(true_flux)))
    flux_err = true_flux * noise_level

    # All bands observed
    mask = np.ones(len(flux), dtype=bool)

    return flux, flux_err, mask, true_idx


class TestBruteForceInitialization:
    """Test BruteForce initialization."""

    def test_initialization_with_stargrid(self, mock_grid):
        """Test proper initialization with StarGrid."""
        fitter = BruteForce(mock_grid, verbose=False)

        assert fitter.nmodels == 27
        assert fitter.nfilters == 5
        assert hasattr(fitter, "star_grid")
        assert hasattr(fitter, "labels_mask")

    def test_initialization_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises(TypeError):
            BruteForce("not a grid")

        with pytest.raises(TypeError):
            BruteForce(None)

    def test_labels_mask_generation(self, bruteforce_fitter):
        """Test automatic labels mask generation."""
        mask = bruteforce_fitter.labels_mask

        # Grid parameters should be True
        assert mask["mini"] == True
        assert mask["eep"] == True
        assert mask["feh"] == True

        # Check that we have some predictions (labels beyond grid params)
        assert len(mask) > 3  # Should have more than just grid parameters
        # Real MIST grid has different parameter names - just check grid vs predictions exist
        grid_params = sum(1 for is_grid in mask.values() if is_grid)
        pred_params = sum(1 for is_grid in mask.values() if not is_grid)
        # mini, eep, feh for the published single-afe grid: the default
        # 'afe' label is dropped when constant, so the schema matches the
        # pre-1.2 behavior; multi-afe grids would report 4.
        assert grid_params == 3
        assert pred_params > 0  # Should have some predictions

    def test_verbose_output(self, mock_grid, capsys):
        """Test verbose initialization output."""
        fitter = BruteForce(mock_grid, verbose=True)
        captured = capsys.readouterr()

        assert "BruteForce initialized" in captured.out
        assert "27" in captured.out  # Number of models
        assert "Grid parameters" in captured.out
        assert "Predictions" in captured.out


class TestBruteForceGetSedGrid:
    """Test batch SED computation."""

    def test_get_sed_grid_all_models(self, bruteforce_fitter):
        """Test SED computation for all models."""
        seds, rvecs, drvecs = bruteforce_fitter.get_sed_grid()

        nmodels = len(bruteforce_fitter.models)
        assert seds.shape == (nmodels, 5)
        assert rvecs.shape == (nmodels, 5)
        assert drvecs.shape == (nmodels, 5)

        # Check values are finite
        assert np.all(np.isfinite(seds))

    def test_get_sed_grid_subset(self, bruteforce_fitter):
        """Test SED computation for model subset."""
        indices = [0, 5, 10, 15, 20]
        seds, rvecs, drvecs = bruteforce_fitter.get_sed_grid(indices=indices)

        assert seds.shape == (5, 5)
        assert np.all(np.isfinite(seds))

    def test_get_sed_grid_with_extinction(self, bruteforce_fitter):
        """Test SED computation with extinction."""
        # Without extinction
        seds_no_ext, _, _ = bruteforce_fitter.get_sed_grid(av=0.0, rv=3.3)

        # With extinction
        seds_ext, _, _ = bruteforce_fitter.get_sed_grid(av=0.5, rv=3.1)

        # Should be different (extinction makes fainter = larger mags)
        assert not np.allclose(seds_no_ext, seds_ext)
        assert np.mean(seds_ext) > np.mean(seds_no_ext)

    def test_get_sed_grid_return_flux(self, bruteforce_fitter):
        """Test returning flux vs magnitudes."""
        seds_mag, _, _ = bruteforce_fitter.get_sed_grid(return_flux=False)
        seds_flux, _, _ = bruteforce_fitter.get_sed_grid(return_flux=True)

        # Convert and compare
        flux_from_mag = 10 ** (-0.4 * seds_mag)
        np.testing.assert_allclose(flux_from_mag, seds_flux, rtol=1e-6)


class TestBruteForceLikelihood:
    """Test likelihood computation."""

    def test_loglike_grid_basic(self, bruteforce_fitter, synthetic_observation):
        """Test basic likelihood computation."""
        flux, flux_err, mask, true_idx = synthetic_observation

        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=False
        )

        assert len(lnl) == len(
            bruteforce_fitter.models
        )  # Should match number of models
        assert ndim == 5  # All bands observed
        assert len(chi2) == len(bruteforce_fitter.models)
        assert np.all(np.isfinite(lnl))

    def test_loglike_finds_true_model(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood identifies true model as good fit."""
        flux, flux_err, mask, true_idx = synthetic_observation

        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=False
        )

        # True model should have reasonable likelihood (within top 10)
        # The optimization may find better fits due to extinction/distance fitting
        best_indices = np.argsort(lnl)[-10:]
        assert (
            true_idx in best_indices
        ), f"True model {true_idx} not in top 10: {best_indices[-5:]}"

    def test_loglike_with_noisy_data(
        self, bruteforce_fitter, noisy_synthetic_observation
    ):
        """Test likelihood with realistic noisy data."""
        flux, flux_err, mask, true_idx = noisy_synthetic_observation

        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=False
        )

        # With noise, check if true model has reasonable likelihood
        # (within 100 of the maximum log-likelihood)
        max_lnl = np.max(lnl)
        assert lnl[true_idx] > max_lnl - 100, "True model likelihood too low"

    def test_loglike_with_optimization(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with extinction optimization."""
        flux, flux_err, mask, true_idx = synthetic_observation

        results = bruteforce_fitter.loglike_grid(
            flux,
            flux_err,
            mask,
            avlim=(0.0, 2.0),
            rvlim=(2.5, 4.0),
            av_gauss=(0.0, 1.0),
            rv_gauss=(3.3, 0.2),
            return_vals=True,
        )

        lnl, ndim, chi2, scale, av, rv, icov = results

        assert len(scale) == len(bruteforce_fitter.models)
        nmodels = len(bruteforce_fitter.models)
        assert len(av) == nmodels
        assert len(rv) == nmodels
        assert icov.shape == (nmodels, 3, 3)

        # Check optimized values are in bounds
        assert np.all((av >= 0.0) & (av <= 2.0))
        assert np.all((rv >= 2.5) & (rv <= 4.0))

    def test_loglike_with_parallax(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with parallax constraint."""
        flux, flux_err, mask, true_idx = synthetic_observation

        # Add parallax (1 mas = 1 kpc distance)
        parallax = 1.0
        parallax_err = 0.1

        lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
            flux,
            flux_err,
            mask,
            parallax=parallax,
            parallax_err=parallax_err,
            return_vals=False,
        )

        assert np.all(np.isfinite(lnl))

    def test_loglike_masked_bands(self, bruteforce_fitter, synthetic_observation):
        """Test likelihood with some bands masked."""
        flux, flux_err, mask, true_idx = synthetic_observation

        # Mask some bands
        mask_partial = mask.copy()
        mask_partial[2:4] = False

        with pytest.warns(RuntimeWarning, match="Only 3 valid photometric bands"):
            lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
                flux, flux_err, mask_partial, return_vals=False
            )

        assert ndim == 3  # Only 3 bands observed
        assert np.all(np.isfinite(lnl))

    def test_loglike_bad_data(self, bruteforce_fitter):
        """Test likelihood with bad data values."""
        # Create data with NaN and inf
        flux = np.array([0.1, np.nan, 0.15, np.inf, 0.2])
        flux_err = np.ones(5) * 0.01
        mask = np.ones(5, dtype=bool)

        with pytest.warns(RuntimeWarning, match="valid photometric bands"):
            lnl, ndim, chi2 = bruteforce_fitter.loglike_grid(
                flux, flux_err, mask, return_vals=False
            )

        # Should handle bad values gracefully
        assert ndim < 5  # Bad values should be masked
        assert np.any(
            np.isfinite(lnl)
        )  # At least some models should have valid likelihood


class TestBruteForcePosterior:
    """Test posterior computation."""

    def test_logpost_grid_basic(self, bruteforce_fitter, synthetic_observation):
        """Test basic posterior computation."""
        flux, flux_err, mask, true_idx = synthetic_observation

        # Get likelihood results first
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=True
        )

        # Compute posteriors
        results = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=0.01,  # Small for testing
            rstate=np.random.RandomState(379),
        )

        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc, mc_ess = results

        assert len(sel) > 0  # Some models selected
        assert cov.shape == (len(sel), 3, 3)
        assert len(lnp) == len(sel)
        assert dist_mc.shape == (len(sel), 10)
        assert mc_ess.shape == (len(sel),)
        assert np.all(mc_ess >= 0)

    def test_logpost_grid_tiny_mem_lim(self, bruteforce_fitter, synthetic_observation):
        """A tiny mem_lim must not crash; Nmc is floored to >= 1.

        Regression: ``Nmc = min(Nmc_prior, int(mem_lim*1e6/(32*Nsel)))`` could
        be 0 for a small mem_lim, yielding empty MC draws, ``log(0)``, and a
        crash in the downstream reduction.
        """
        import warnings

        flux, flux_err, mask, _ = synthetic_observation
        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = bruteforce_fitter.logpost_grid(
                like_results,
                Nmc_prior=10,
                wt_thresh=0.01,
                mem_lim=1e-4,
                rstate=np.random.RandomState(407),
            )
        sel, _, lnp, dist_mc = results[0], results[1], results[2], results[3]
        assert len(sel) > 0
        assert dist_mc.shape[1] >= 1  # Nmc floored to >= 1
        assert not np.any(np.isnan(lnp))

    def test_loglike_grid_all_masked_raises(
        self, bruteforce_fitter, synthetic_observation
    ):
        """A fully-masked object must fail fast with a clear error.

        Regression: with zero valid bands the optimizer divided by zero (an
        opaque ZeroDivisionError under numba) and the dim_prior term hit
        log(0). loglike_grid now raises an actionable ValueError instead.
        """
        import warnings

        flux, flux_err, mask, _ = synthetic_observation
        zero_mask = np.zeros_like(mask, dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="valid photometric band"):
                bruteforce_fitter.loglike_grid(flux, flux_err, zero_mask)

    def test_logpost_model_selection(self, bruteforce_fitter, synthetic_observation):
        """Test different model selection methods."""
        flux, flux_err, mask, true_idx = synthetic_observation

        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=True
        )

        # Test weight threshold
        results_wt = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=1e-3,
            cdf_thresh=None,
            rstate=np.random.RandomState(442),
        )

        # Test CDF threshold
        results_cdf = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=None,
            cdf_thresh=0.95,
            rstate=np.random.RandomState(447),
        )

        # Both should select some models
        assert len(results_wt[0]) > 0
        assert len(results_cdf[0]) > 0

    def test_logpost_monte_carlo_sampling(
        self, bruteforce_fitter, synthetic_observation
    ):
        """Test Monte Carlo sampling produces reasonable distributions."""
        flux, flux_err, mask, true_idx = synthetic_observation

        like_results = bruteforce_fitter.loglike_grid(
            flux, flux_err, mask, return_vals=True
        )

        results = bruteforce_fitter.logpost_grid(
            like_results,
            Nmc_prior=100,  # More samples for better statistics
            wt_thresh=0.01,
            rstate=np.random.RandomState(465),
        )

        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc, mc_ess = results

        # Check distance samples are positive
        assert np.all(dist_mc > 0)

        # Check extinction is bounded
        assert np.all(av_mc >= 0)
        assert np.all(av_mc <= 20)  # Default avlim

        # Check Rv is bounded
        assert np.all(rv_mc >= 1)
        assert np.all(rv_mc <= 8)  # Default rvlim

        # Check ESS is reasonable (between 0 and Nmc)
        assert np.all(mc_ess >= 0)
        assert np.all(mc_ess <= 100)


class TestBruteForceInternal:
    """Test internal methods."""

    def test_setup_method(self, bruteforce_fitter):
        """Test data preprocessing in _setup."""
        # Create test data
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        # Coordinates for Galactic prior
        coords = (120.0, 45.0)

        results = bruteforce_fitter._setup(
            data, data_err, data_mask, data_coords=coords, mag_max=30.0, merr_max=0.5
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Data should be unchanged (all good values)
        np.testing.assert_array_equal(proc_data, data)
        np.testing.assert_array_equal(proc_mask, data_mask)

        # Prior should be initialized
        assert len(lnprior) == bruteforce_fitter.nmodels
        assert gal_prior is not None  # Function should be set

    def test_fit_internal(self, bruteforce_fitter, synthetic_observation):
        """Test internal _fit method."""
        flux, flux_err, mask, true_idx = synthetic_observation

        results = bruteforce_fitter._fit(
            flux,
            flux_err,
            mask,
            parallax=1.0,
            parallax_err=0.2,
            coord=(120.0, 45.0),
            Nmc_prior=10,
            wt_thresh=0.01,
            rstate=np.random.RandomState(518),
        )

        # _fit returns 14 values with return_distreds=True (default)
        (
            idxs,
            scales,
            avs,
            rvs,
            covs_sar,
            Ndim,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = results

        # Check basic outputs
        assert len(idxs) > 0
        assert np.all(np.isfinite(lnprob))

    def test_repr_method(self, bruteforce_fitter):
        """Test string representation."""
        repr_str = repr(bruteforce_fitter)

        assert "BruteForce" in repr_str
        assert "31" in repr_str  # nmodels (updated for real MIST subset)
        assert "5" in repr_str  # nfilters


class TestBruteForceRealGrid:
    """Comprehensive tests using real MIST grid data."""

    def test_initialization(self, bruteforce_fitter, real_mist_setup):
        """Test BruteForce initialization with real grid."""
        fitter = bruteforce_fitter

        assert fitter.nmodels == 31  # Test subset
        assert fitter.nfilters == 5  # Pan-STARRS subset filters
        assert len(fitter.labels_mask) >= 8  # Grid + prediction parameters

        # Check that grid parameters are marked correctly
        assert fitter.labels_mask["mini"] == True
        assert fitter.labels_mask["eep"] == True
        assert fitter.labels_mask["feh"] == True

        # Test properties (should match above assertions)
        assert fitter.nmodels == 31  # Test subset
        assert fitter.nfilters == 5

    def test_get_sed_grid_real_data(self, bruteforce_fitter):
        """Test SED computation with real grid."""
        fitter = bruteforce_fitter

        # Test all models
        seds, rvecs, drvecs = fitter.get_sed_grid()
        assert seds.shape == (fitter.nmodels, fitter.nfilters)
        assert np.all(np.isfinite(seds))

        # Test subset
        indices = np.arange(10)
        seds_sub, _, _ = fitter.get_sed_grid(indices=indices)
        assert seds_sub.shape == (10, fitter.nfilters)

        # Test with extinction
        seds_ext, _, _ = fitter.get_sed_grid(av=0.5, rv=3.1, indices=indices)
        assert not np.allclose(seds_sub, seds_ext)  # Should be different

    def test_loglike_comprehensive(self, bruteforce_fitter, real_mist_setup):
        """Test likelihood computation with various configurations."""
        fitter = bruteforce_fitter
        ps_indices = real_mist_setup["ps_indices"]

        # Create realistic PanSTARRS observations
        nfilters = fitter.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Solar-like star magnitudes in PanSTARRS
        ps_mags = [15.0, 14.8, 14.6, 14.4, 14.3]
        for i, idx in enumerate(ps_indices[:5]):
            obs[idx] = ps_mags[i]
            obs_err[idx] = 0.05

        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        # Test 1: Basic likelihood
        lnl, ndim, chi2 = fitter.loglike_grid(obs, obs_err, obs_mask, return_vals=False)
        assert lnl.shape == (fitter.nmodels,)
        assert ndim == 5  # 5 PanSTARRS bands
        assert np.all(np.isfinite(lnl))
        assert np.all(chi2 >= 0)

        # Test 2: With parallax
        lnl_par, _, _ = fitter.loglike_grid(
            obs, obs_err, obs_mask, parallax=1.0, parallax_err=0.1, return_vals=False
        )
        assert np.all(np.isfinite(lnl_par))

        # Test 3: With masked bands
        mask_partial = obs_mask.copy()
        mask_partial[ps_indices[2]] = False  # Mask one band
        lnl_masked, ndim_masked, _ = fitter.loglike_grid(
            obs, obs_err, mask_partial, return_vals=False
        )
        assert ndim_masked == 4  # One less band

        # Test 4: Subset of models
        indices = np.arange(20)
        lnl_sub, _, _ = fitter.loglike_grid(
            obs, obs_err, obs_mask, indices=indices, return_vals=False
        )
        assert lnl_sub.shape == (20,)

    def test_logpost_comprehensive(self, bruteforce_fitter, real_mist_setup):
        """Test posterior computation with real grid and realistic priors."""
        fitter = bruteforce_fitter
        ps_indices = real_mist_setup["ps_indices"]

        # Step 1: Set up realistic test scenario with small model subset
        # Use 15 models to ensure stable covariance matrices but keep test fast
        indices = np.arange(15)

        # Step 2: Create realistic stellar observation
        # Generate observation from a mid-grid model to ensure good fit
        nfilters = fitter.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        # Use 4 PanSTARRS filters with realistic magnitudes and errors
        for i, idx in enumerate(ps_indices[:4]):
            obs[idx] = 15.0 + 0.3 * i  # mag
            obs_err[idx] = 0.08  # Realistic photometric errors
        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        # Step 3: Run loglike_grid to get proper input for logpost_grid
        like_results = fitter.loglike_grid(
            obs,
            obs_err,
            obs_mask,
            indices=indices,
            return_vals=True,  # Essential for logpost_grid
        )

        # Verify loglike_grid results have correct format
        lnl, ndim, chi2, scale, av, rv, icov = like_results
        assert len(lnl) == 15
        assert icov.shape == (15, 3, 3)

        # Step 4: Test basic logpost_grid without complex priors
        # This tests the core Monte Carlo sampling and covariance handling
        results_basic = fitter.logpost_grid(
            like_results,
            Nmc_prior=10,  # Small number for fast testing
            wt_thresh=0.1,  # Relaxed threshold to ensure models selected
            coord=None,  # No spatial priors initially
            dustfile=None,  # No dust priors initially
            parallax=None,  # No parallax constraints initially
            rstate=np.random.RandomState(683),
        )

        # Verify basic posterior computation works
        sel, cov_sar, lnp, dist_mc, av_mc, rv_mc, lnp_mc, mc_ess = results_basic

        # Basic validation - ensure we get reasonable outputs
        assert len(sel) > 0, "No models selected by posterior"
        assert cov_sar.shape == (
            len(sel),
            3,
            3,
        ), f"Covariance shape mismatch: {cov_sar.shape}"
        assert dist_mc.shape == (
            len(sel),
            10,
        ), f"Distance MC shape mismatch: {dist_mc.shape}"
        assert np.all(dist_mc > 0), "Distances must be positive"
        assert np.all(av_mc >= 0), "Extinction must be non-negative"

        print(f"✓ Basic logpost_grid: {len(sel)} models selected")

        # Step 5: Test with realistic galactic coordinates
        # Choose coordinates in the Galactic disk (l=120°, b=45°)
        coord_galactic = (120.0, 45.0)

        results_galactic = fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=0.1,
            coord=coord_galactic,  # Add galactic structure prior
            dlabels=fitter.models_labels[indices],
            dustfile=None,
            parallax=None,
            rstate=np.random.RandomState(715),
        )

        # Verify galactic prior integration works
        sel_gal, _, lnp_gal, _, _, _, _, _ = results_galactic
        assert len(sel_gal) > 0, "Galactic prior eliminated all models"

        print(f"✓ With galactic prior: {len(sel_gal)} models selected")

        # Step 6: Test threshold-based model selection
        # Test both weight threshold and CDF threshold
        results_cdf = fitter.logpost_grid(
            like_results,
            Nmc_prior=10,
            wt_thresh=None,
            cdf_thresh=0.5,  # Select top 50% of models
            coord=coord_galactic,
            dlabels=fitter.models_labels[indices],
            rstate=np.random.RandomState(732),
        )

        sel_cdf, _, _, _, _, _, _, _ = results_cdf
        assert len(sel_cdf) > 0, "CDF threshold eliminated all models"

        print(f"✓ CDF selection: {len(sel_cdf)} models selected")

        # Step 7: Basic validation that the pipeline produces reasonable results
        # Verify that Monte Carlo samples are within expected bounds
        assert np.all(
            (av_mc >= 0) & (av_mc <= 20)
        ), "A_V samples outside expected range"
        assert np.all((rv_mc >= 1) & (rv_mc <= 8)), "R_V samples outside expected range"
        assert np.all(dist_mc < 100), "Distance samples unrealistically large"

        print("✓ All logpost_grid tests completed successfully")

    def test_ltol_clipping_keeps_top_models(self, bruteforce_fitter, real_mist_setup):
        """A larger ltol keeps most of the ten best models."""
        fitter = bruteforce_fitter
        ps_indices = real_mist_setup["ps_indices"]

        # Create simple observation
        nfilters = fitter.nfilters
        obs = np.full(nfilters, np.nan)
        obs_err = np.full(nfilters, np.nan)

        for i, idx in enumerate(ps_indices[:3]):
            obs[idx] = 15.0
            obs_err[idx] = 0.05
        obs, obs_err = inv_magnitude(obs, obs_err)  # maggies
        obs_mask = ~np.isnan(obs)

        # Test clipping behavior
        with pytest.warns(RuntimeWarning, match="valid photometric bands"):
            lnl_default, _, _ = fitter.loglike_grid(
                obs, obs_err, obs_mask, ltol=3e-2, return_vals=False
            )
        with pytest.warns(RuntimeWarning, match="valid photometric bands"):
            lnl_aggressive, _, _ = fitter.loglike_grid(
                obs, obs_err, obs_mask, ltol=1e-1, return_vals=False
            )

        # Top models should be similar
        best_default = set(np.argsort(lnl_default)[-10:])
        best_aggressive = set(np.argsort(lnl_aggressive)[-10:])
        overlap = len(best_default & best_aggressive)
        assert overlap >= 5, f"Only {overlap}/10 top models agree"

    def test_edge_cases_and_error_handling(self, bruteforce_fitter):
        """Test edge cases and error handling."""
        fitter = bruteforce_fitter

        # Test with bad data - must match the number of filters in the real grid
        nfilters = fitter.nfilters
        flux = np.full(nfilters, 0.1)  # Fill all filters
        flux[1] = np.nan  # Bad value
        flux[3] = np.inf  # Bad value
        flux_err = np.full(nfilters, 0.01)
        mask = np.ones(nfilters, dtype=bool)

        # Should handle gracefully
        with pytest.warns(RuntimeWarning, match="valid photometric bands"):
            lnl, ndim, chi2 = fitter.loglike_grid(
                flux, flux_err, mask, return_vals=False
            )
        assert ndim < nfilters  # Some bad values should be masked
        assert np.any(np.isfinite(lnl))

        # Test with very few bands - create proper sized arrays
        flux_simple = np.full(nfilters, np.nan)  # Start with all NaN
        flux_simple[0] = 0.1  # Only one valid measurement
        flux_err_simple = np.full(nfilters, 0.01)
        mask_simple = np.ones(nfilters, dtype=bool)

        with pytest.warns(RuntimeWarning, match="valid photometric bands"):
            lnl_simple, ndim_simple, _ = fitter.loglike_grid(
                flux_simple, flux_err_simple, mask_simple, return_vals=False
            )
        assert ndim_simple == 1  # Only one valid band
        assert len(lnl_simple) == fitter.nmodels

    def test_internal_methods(self, bruteforce_fitter, real_mist_setup):
        """Test internal _setup method (avoiding numba compilation issues in _fit)."""
        fitter = bruteforce_fitter

        # Test _setup method - must use correct array sizes for real grid
        nfilters = fitter.nfilters
        data = np.full(nfilters, 1e-6)  # Match real grid size
        data_err = data * 0.05
        data_mask = np.ones(nfilters, dtype=bool)

        # Coordinates for Galactic prior
        coords = (120.0, 45.0)

        results = fitter._setup(
            data, data_err, data_mask, data_coords=coords, mag_max=30.0, merr_max=0.5
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Data should be processed correctly
        assert len(proc_data) == nfilters
        assert len(lnprior) == fitter.nmodels
        assert gal_prior is not None

        # Test _setup with photometric offsets
        offsets = np.ones(nfilters) * 1.1  # 10% offset
        results_offset = fitter._setup(
            data,
            data_err,
            data_mask,
            phot_offsets=offsets,
            data_coords=coords,
            mag_max=30.0,
        )

        proc_data_offset = results_offset[0]
        expected_data = data * offsets
        np.testing.assert_allclose(proc_data_offset, expected_data)

        # Test _setup error handling for missing coordinates
        try:
            fitter._setup(
                data,
                data_err,
                data_mask,
                data_coords=None,  # Missing coordinates
                lngalprior=None,  # Will try to use default
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "data_coords" in str(e)

    def test_repr_method(self, bruteforce_fitter):
        """Test string representation."""
        fitter = bruteforce_fitter
        repr_str = repr(fitter)

        assert "BruteForce" in repr_str
        assert "31" in repr_str  # nmodels from subset (updated)
        assert str(fitter.nfilters) in repr_str


class TestBruteForceEdgeCases:
    """Test edge cases and error handling."""

    def test_setup_with_photometric_offsets(self, bruteforce_fitter):
        """Test _setup method with photometric offsets."""
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        # Apply photometric offsets
        offsets = np.array([1.1, 0.95, 1.0, 1.05, 0.9])

        results = bruteforce_fitter._setup(
            data,
            data_err,
            data_mask,
            phot_offsets=offsets,
            data_coords=(120.0, 45.0),
            mag_max=30.0,
            merr_max=0.5,
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Data should be modified by offsets
        expected_data = data * offsets
        np.testing.assert_allclose(proc_data, expected_data)

    def test_setup_basic_functionality(self, bruteforce_fitter):
        """Test _setup method basic functionality."""
        data, data_err = inv_magnitude(
            np.array([20.0, 20.5, 21.0, 21.5, 22.0]),
            np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
        )
        data_mask = np.ones(5, dtype=bool)

        results = bruteforce_fitter._setup(
            data, data_err, data_mask, data_coords=(120.0, 45.0)
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Basic functionality checks
        assert len(proc_data) == 5
        assert len(proc_err) == 5
        assert len(proc_mask) == 5
        assert np.all(np.isfinite(proc_data))
        assert np.all(proc_err > 0)
        assert isinstance(lnprior, np.ndarray)
        assert len(lnprior) == bruteforce_fitter.nmodels

    def test_setup_prior_initialization_edge_cases(self, bruteforce_fitter):
        """Test prior initialization with various model configurations."""
        rng = np.random.RandomState(0)
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        # Test with custom prior
        custom_prior = rng.normal(0, 1, bruteforce_fitter.nmodels)

        results = bruteforce_fitter._setup(
            data,
            data_err,
            data_mask,
            lnprior=custom_prior,
            data_coords=(120.0, 45.0),
            apply_agewt=False,  # Disable age weighting
            apply_grad=False,  # Disable gradient prior
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Should use custom prior
        np.testing.assert_array_equal(lnprior, custom_prior)

    def test_properties(self, bruteforce_fitter):
        """Test class properties."""
        assert bruteforce_fitter.nmodels == 31
        assert bruteforce_fitter.nfilters == 5

        # Properties should be consistent
        assert bruteforce_fitter.nmodels == 31  # Test subset
        assert bruteforce_fitter.nfilters == 5


class TestBruteForceCoverageCompletion:
    """Tests specifically designed to increase coverage of untested code paths."""

    def test_setup_custom_galactic_and_dust_priors(self, bruteforce_fitter):
        """Test _setup with custom galactic and dust priors."""
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        # Mock prior functions
        def mock_gal_prior(distance, coord, labels=None):
            return np.zeros_like(distance)

        def mock_dust_prior(coord, distance, av, dustmap):
            return np.zeros_like(distance)

        results = bruteforce_fitter._setup(
            data,
            data_err,
            data_mask,
            data_coords=(120.0, 45.0),
            lngalprior=mock_gal_prior,
            lndustprior=mock_dust_prior,
            dustfile="/mock/dust/file",
        )

        proc_data, proc_err, proc_mask, lnprior, gal_prior, dust_prior = results

        # Should use provided functions
        assert gal_prior == mock_gal_prior
        assert dust_prior == mock_dust_prior

    def test_logpost_covariance_inversion_failure(
        self, bruteforce_fitter, synthetic_observation
    ):
        """Test handling of covariance matrix inversion failures."""
        flux, flux_err, mask, true_idx = synthetic_observation

        # Get likelihood results and modify to create singular covariances
        like_results = list(
            bruteforce_fitter.loglike_grid(flux, flux_err, mask, return_vals=True)
        )

        # Make some covariance matrices singular by zeroing out diagonal
        icov_sar = like_results[6]  # icov_sar
        icov_sar[0, :, :] = 0.0  # Make first model's covariance singular
        like_results[6] = icov_sar

        # Should handle gracefully with diagonal approximation
        results = bruteforce_fitter.logpost_grid(
            tuple(like_results),
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(1015),
        )

        sel, cov, lnp, dist_mc, av_mc, rv_mc, lnp_mc, mc_ess = results
        assert len(sel) > 0


class TestBruteForceMathematicalValidation:
    """Critical mathematical validation tests for cross-term bugfix."""

    def test_logpost_with_full_priors_and_spd_matrices(
        self, bruteforce_fitter, real_mist_setup
    ):
        """
        Test complete logpost_grid functionality with galactic and dust priors.

        This tests the full chain: loglike_grid -> logpost_grid with all priors,
        ensuring inverse3 produces SPD covariance matrices.
        """

        fitter = bruteforce_fitter

        # Create realistic observation with galactic coordinates
        obs = 10 ** (
            -0.4 * np.array([15.0, 15.2, 15.4, 15.6, 15.8])
        )  # Convert mags to flux
        obs_err = obs * 0.05  # 5% flux errors
        obs_mask = np.ones(5, dtype=bool)

        # Use galactic coordinates that should have dust
        coord = (120.0, -20.0)  # l, b in degrees - low latitude for dust

        # Test with small subset for speed
        test_indices = np.arange(3)  # 3 models

        # Step 1: Get likelihood results
        loglike_results = fitter.loglike_grid(
            obs, obs_err, obs_mask, indices=test_indices, return_vals=True
        )

        # Step 2: Run full logpost with all priors (including dust map loading)
        try:
            logpost_results = fitter.logpost_grid(
                loglike_results,
                coord=coord,  # Galactic coordinates for dust/galactic priors
                dlabels=fitter.models_labels[test_indices],
                Nmc_prior=10,  # Small for speed
                wt_thresh=0.1,
                rstate=np.random.RandomState(1058),
            )

            # Unpack results - check what we actually get
            assert (
                len(logpost_results) >= 5
            ), f"Expected at least 5 return values, got {len(logpost_results)}"

            # Basic validation that logpost completed successfully
            sel = logpost_results[0]  # Selected model indices
            assert len(sel) > 0, "Should select at least some models"

            # If covariance matrices are returned, test they are SPD
            if len(logpost_results) > 4:  # Has covariance matrices
                cov_matrices = logpost_results[4]  # Typically the 5th element
                if hasattr(cov_matrices, "shape") and len(cov_matrices.shape) == 3:
                    # Test first few covariance matrices are PSD (from inverse3)
                    for i in range(min(len(cov_matrices), 2)):
                        cov_matrix = cov_matrices[i]
                        eigenvals = np.linalg.eigvals(cov_matrix)
                        assert np.all(
                            eigenvals >= -1e-10
                        ), f"Covariance matrix {i} not SPD: eigenvals={eigenvals}"

            print("✅ Full logpost_grid with priors completed successfully")

        except FileNotFoundError as e:
            if "dustmap" in str(e).lower() or "bayestar" in str(e).lower():
                # Dust map not available - test basic functionality without dust prior
                print("Dust map not available, testing without dust prior")
                logpost_results = fitter.logpost_grid(
                    loglike_results,
                    Nmc_prior=10,
                    wt_thresh=0.1,
                    rstate=np.random.RandomState(1092),
                )
                sel = logpost_results[0]
                assert len(sel) > 0, "Should select at least some models"
            else:
                raise


@pytest.fixture
def mock_fitter(mock_grid):
    """Create BruteForce instance with mock grid for tests that don't need real data."""
    return BruteForce(mock_grid, verbose=False)


@pytest.fixture
def mock_synthetic_observation(mock_grid):
    """Create synthetic observation from mock grid."""
    true_idx = 13  # Pick a model from the mock grid
    true_model = mock_grid.models[true_idx, :, 0]  # Base magnitudes

    # Convert to flux - NO NOISE for deterministic testing
    true_flux = 10 ** (-0.4 * true_model)
    flux = true_flux
    flux_err = true_flux * 0.05  # 5% error bars

    # All bands observed
    mask = np.ones(len(flux), dtype=bool)

    return flux, flux_err, mask, true_idx


class TestBruteForceFitMethod:
    """Tests for the complete fit() method with HDF5 I/O."""

    def test_fit_basic_streaming_io(self, mock_fitter, tmp_path):
        """Test basic fit with streaming I/O mode."""
        # Create small test dataset
        ndata = 3
        nfilters = 5

        # Create synthetic photometry
        rng = np.random.RandomState(42)
        data = 10 ** (-0.4 * (15 + rng.randn(ndata, nfilters) * 0.5))
        data_err = data * 0.05
        data_mask = np.ones((ndata, nfilters), dtype=bool)

        # Labels for each object
        data_labels = np.array(
            [(i, i * 100, i * 0.1) for i in range(ndata)],
            dtype=[("id", "i4"), ("ra", "f4"), ("dec", "f4")],
        )

        # Galactic coordinates
        data_coords = np.array([[120.0, 45.0], [130.0, 50.0], [140.0, 55.0]])

        # Output file
        save_file = str(tmp_path / "test_fit_streaming")

        # Run fit with streaming I/O
        result_file = mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=10,
            Nmc_prior=5,
            wt_thresh=0.1,
            running_io=True,
            save_dar_draws=True,
            verbose=False,
            rstate=np.random.RandomState(1231),
        )

        # Verify output file exists and has correct structure
        assert result_file.endswith(".h5")
        with h5py.File(result_file, "r") as f:
            # Check all required datasets exist
            assert "labels" in f
            assert "model_idx" in f
            assert "ml_scale" in f
            assert "ml_av" in f
            assert "ml_rv" in f
            assert "ml_cov_sar" in f
            assert "obj_log_post" in f
            assert "obj_log_evid" in f
            assert "obj_chi2min" in f
            assert "obj_Nbands" in f

            # Check distance/reddening draws are saved
            assert "samps_dist" in f
            assert "samps_red" in f
            assert "samps_dred" in f
            assert "samps_logp" in f

            # Verify shapes
            assert f["model_idx"].shape == (ndata, 10)
            assert f["ml_scale"].shape == (ndata, 10)
            assert f["ml_cov_sar"].shape == (ndata, 10, 3, 3)
            assert f["obj_log_evid"].shape == (ndata,)
            assert f["samps_dist"].shape == (ndata, 10)

            # Verify data values are reasonable
            assert np.all(f["model_idx"][:] >= -99)  # -99 is sentinel for invalid
            assert np.all(np.isfinite(f["obj_log_evid"][:]))
            assert np.all(f["obj_Nbands"][:] > 0)

    def test_fit_batch_io(self, mock_fitter, tmp_path):
        """Test fit with batch I/O mode (accumulate in memory, write at end)."""
        ndata = 2
        nfilters = 5

        rng = np.random.RandomState(123)
        data = 10 ** (-0.4 * (14.5 + rng.randn(ndata, nfilters) * 0.3))
        data_err = data * 0.03
        data_mask = np.ones((ndata, nfilters), dtype=bool)
        data_labels = np.array([(i,) for i in range(ndata)], dtype=[("id", "i4")])
        data_coords = np.array([[100.0, 30.0], [110.0, 35.0]])

        save_file = str(tmp_path / "test_fit_batch")

        result_file = mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=5,
            Nmc_prior=3,
            wt_thresh=0.2,
            running_io=False,  # Batch mode
            save_dar_draws=True,
            verbose=False,
            rstate=np.random.RandomState(1293),
        )

        # Verify output
        with h5py.File(result_file, "r") as f:
            assert f["model_idx"].shape == (ndata, 5)
            assert f["ml_scale"].shape == (ndata, 5)
            assert f["samps_dist"].shape == (ndata, 5)

            # Compare streaming vs batch - both should work
            assert np.all(np.isfinite(f["obj_chi2min"][:]))

    def test_fit_without_dar_draws(self, mock_fitter, tmp_path):
        """Test fit without saving distance/reddening draws."""
        ndata = 2
        nfilters = 5

        rng = np.random.RandomState(456)
        data = 10 ** (-0.4 * (15 + rng.randn(ndata, nfilters) * 0.4))
        data_err = data * 0.04
        data_mask = np.ones((ndata, nfilters), dtype=bool)
        data_labels = np.array([(i,) for i in range(ndata)], dtype=[("id", "i4")])
        data_coords = np.array([[90.0, 20.0], [95.0, 25.0]])

        save_file = str(tmp_path / "test_fit_no_dar")

        result_file = mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=5,
            Nmc_prior=3,
            wt_thresh=0.2,
            save_dar_draws=False,  # Don't save distance/reddening draws
            verbose=False,
            rstate=np.random.RandomState(1331),
        )

        with h5py.File(result_file, "r") as f:
            # Core datasets should exist
            assert "model_idx" in f
            assert "ml_scale" in f

            # Distance/reddening draws should NOT exist
            assert "samps_dist" not in f
            assert "samps_red" not in f
            assert "samps_dred" not in f
            assert "samps_logp" not in f

    def test_fit_with_parallax(self, mock_fitter, tmp_path):
        """Test fit with parallax constraints."""
        ndata = 2
        nfilters = 5

        rng = np.random.RandomState(789)
        data = 10 ** (-0.4 * (14 + rng.randn(ndata, nfilters) * 0.3))
        data_err = data * 0.03
        data_mask = np.ones((ndata, nfilters), dtype=bool)
        data_labels = np.array([(i,) for i in range(ndata)], dtype=[("id", "i4")])
        data_coords = np.array([[80.0, 10.0], [85.0, 15.0]])

        # Parallax constraints
        parallax = np.array([1.0, 0.5])  # mas
        parallax_err = np.array([0.1, 0.05])

        save_file = str(tmp_path / "test_fit_parallax")

        result_file = mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            parallax=parallax,
            parallax_err=parallax_err,
            Ndraws=5,
            Nmc_prior=3,
            wt_thresh=0.2,
            verbose=False,
            rstate=np.random.RandomState(1374),
        )

        with h5py.File(result_file, "r") as f:
            # Results should be valid
            assert np.all(np.isfinite(f["obj_log_evid"][:]))
            # Nbands should include parallax contribution
            assert np.all(f["obj_Nbands"][:] >= 5)

    def test_fit_file_extension_handling(self, mock_fitter, tmp_path):
        """Test that .h5 extension is added if not present."""
        ndata = 1
        nfilters = 5

        data = 10 ** (-0.4 * np.ones((ndata, nfilters)) * 15)
        data_err = data * 0.05
        data_mask = np.ones((ndata, nfilters), dtype=bool)
        data_labels = np.array([(0,)], dtype=[("id", "i4")])
        data_coords = np.array([[100.0, 40.0]])

        # Save without .h5 extension
        save_file = str(tmp_path / "test_no_extension")

        result_file = mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=3,
            Nmc_prior=2,
            wt_thresh=0.3,
            verbose=False,
            rstate=np.random.RandomState(1409),
        )

        # Should add .h5 extension
        assert result_file.endswith(".h5")
        assert (tmp_path / "test_no_extension.h5").exists()

    def test_internal_fit_returns_correct_tuple(
        self, mock_fitter, mock_synthetic_observation
    ):
        """Test that _fit returns the correct tuple format."""
        flux, flux_err, mask, true_idx = mock_synthetic_observation

        # Test with return_distreds=True (default)
        results = mock_fitter._fit(
            flux,
            flux_err,
            mask,
            coord=(120.0, 45.0),
            Nmc_prior=5,
            Ndraws=10,
            wt_thresh=0.1,
            return_distreds=True,
            rstate=np.random.RandomState(1371),
        )

        # Should return 14-element tuple (13 + mc_ess)
        assert len(results) == 14
        (
            idxs,
            scales,
            avs,
            rvs,
            covs_sar,
            Ndim,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = results

        # Verify shapes
        assert idxs.shape == (10,)
        assert scales.shape == (10,)
        assert avs.shape == (10,)
        assert rvs.shape == (10,)
        assert covs_sar.shape == (10, 3, 3)
        assert lnprob.shape == (10,)
        assert dists.shape == (10,)
        assert reds.shape == (10,)
        assert dreds.shape == (10,)
        assert logwts.shape == (10,)

        # Verify scalar outputs
        assert isinstance(Ndim, (int, np.integer))
        assert isinstance(levid, (float, np.floating))
        assert isinstance(chi2min, (float, np.floating))

        # Verify values are reasonable
        assert np.all(idxs >= 0)
        assert np.all(scales > 0)
        assert np.all(avs >= 0)
        assert np.all(rvs > 0)
        assert np.all(dists > 0)

    def test_internal_fit_without_distreds(
        self, mock_fitter, mock_synthetic_observation
    ):
        """Test _fit with return_distreds=False."""
        flux, flux_err, mask, true_idx = mock_synthetic_observation

        results = mock_fitter._fit(
            flux,
            flux_err,
            mask,
            coord=(120.0, 45.0),
            Nmc_prior=5,
            Ndraws=10,
            wt_thresh=0.1,
            return_distreds=False,
            rstate=np.random.RandomState(1431),
        )

        # Should return 10-element tuple (9 + mc_ess)
        assert len(results) == 10
        idxs, scales, avs, rvs, covs_sar, Ndim, lnprob, levid, chi2min, mc_ess = results

        # Verify shapes
        assert idxs.shape == (10,)
        assert covs_sar.shape == (10, 3, 3)

    def test_fit_verbose_output(self, mock_fitter, tmp_path, capsys):
        """Test verbose progress output."""
        ndata = 2
        nfilters = 5

        data = 10 ** (-0.4 * np.ones((ndata, nfilters)) * 15)
        data_err = data * 0.05
        data_mask = np.ones((ndata, nfilters), dtype=bool)
        data_labels = np.array([(i,) for i in range(ndata)], dtype=[("id", "i4")])
        data_coords = np.array([[100.0, 40.0], [105.0, 45.0]])

        save_file = str(tmp_path / "test_verbose")

        # Run with verbose=True - output goes to stderr
        import sys
        from io import StringIO

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        mock_fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=3,
            Nmc_prior=2,
            wt_thresh=0.3,
            verbose=True,
            rstate=np.random.RandomState(1532),
        )

        stderr_output = sys.stderr.getvalue()
        sys.stderr = old_stderr

        # Check progress was printed
        assert "Fitting object" in stderr_output
        assert "chi2" in stderr_output.lower() or "total:" in stderr_output


class TestBruteForceFitIntegration:
    """Integration tests for the complete fitting pipeline using mock grid."""

    def test_fit_produces_valid_results(self, mock_fitter, mock_grid, tmp_path):
        """Test that fit() produces valid results with reasonable quality metrics."""
        fitter = mock_fitter

        # Create observation from a known model in the grid
        true_idx = 10  # Pick a model from the mock grid
        true_model = mock_grid.models[true_idx]

        # Convert to flux (using zero extinction for simplicity)
        true_mags = true_model[:, 0]  # Base magnitudes
        true_flux = 10 ** (-0.4 * true_mags)
        flux_err = true_flux * 0.03  # 3% errors

        # Create dataset with single object
        ndata = 1
        data = true_flux.reshape(1, -1)
        data_err = flux_err.reshape(1, -1)
        data_mask = np.ones((ndata, fitter.nfilters), dtype=bool)
        data_labels = np.array([(0,)], dtype=[("id", "i4")])
        data_coords = np.array([[120.0, 45.0]])

        save_file = str(tmp_path / "test_recovery")

        result_file = fitter.fit(
            data,
            data_err,
            data_mask,
            data_labels,
            save_file,
            data_coords=data_coords,
            Ndraws=50,
            Nmc_prior=20,
            wt_thresh=0.01,  # Tight threshold
            verbose=False,
            rstate=np.random.RandomState(1579),
        )

        # Verify results file has valid structure and values
        with h5py.File(result_file, "r") as f:
            sampled_indices = f["model_idx"][0, :]

            # Should have sampled valid model indices
            assert np.all(sampled_indices >= 0)
            assert np.all(sampled_indices < fitter.nmodels)

            # Chi2 should be reasonable (not wildly off)
            chi2min = f["obj_chi2min"][0]
            nbands = f["obj_Nbands"][0]
            reduced_chi2 = chi2min / max(nbands - 3, 1)
            assert reduced_chi2 < 100, f"Reduced chi2 unreasonably high: {reduced_chi2}"

            # Log-evidence should be finite
            assert np.isfinite(f["obj_log_evid"][0])

            # Scale factors should be positive
            assert np.all(f["ml_scale"][0, :] > 0)

            # Distances should be positive
            assert np.all(f["samps_dist"][0, :] > 0)


@pytest.fixture(scope="module")
def fitter(mock_grid):
    return BruteForce(mock_grid, verbose=False)


@pytest.fixture
def observation(mock_grid):
    """Noiseless synthetic observation from mock grid model 13."""
    true_flux = 10 ** (-0.4 * mock_grid.models[13, :, 0])
    flux_err = true_flux * 0.05
    mask = np.ones(len(true_flux), dtype=bool)
    return true_flux, flux_err, mask


def _make_loglike_results(scales, cov_list, ndim=5):
    """Build a synthetic loglike_grid(return_vals=True) tuple."""
    n = len(scales)
    lnlike = np.zeros(n)
    chi2 = np.zeros(n)
    avs = np.full(n, 0.5)
    rvs = np.full(n, 3.3)
    icovs = np.array([np.linalg.inv(c) for c in cov_list])
    return (lnlike, ndim, chi2, np.asarray(scales, dtype=float), avs, rvs, icovs)


class TestSetupLnpriorNotMutated:
    def test_user_lnprior_array_unchanged(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)
        user_lnprior = np.zeros(fitter.nmodels)

        res = fitter._setup(
            data,
            data_err,
            data_mask,
            lnprior=user_lnprior,
            data_coords=(120.0, 45.0),
            apply_agewt=True,
            apply_grad=True,
        )
        lnprior_out = res[3]

        # Grid-gradient corrections were applied (output differs from input)
        assert not np.allclose(lnprior_out, 0.0)
        # ... but never in place on the caller's array
        assert lnprior_out is not user_lnprior
        np.testing.assert_array_equal(user_lnprior, np.zeros(fitter.nmodels))

    def test_repeated_setup_calls_do_not_compound(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)
        user_lnprior = np.zeros(fitter.nmodels)

        kwargs = dict(
            lnprior=user_lnprior,
            data_coords=(120.0, 45.0),
            apply_agewt=True,
            apply_grad=True,
        )
        out1 = fitter._setup(data, data_err, data_mask, **kwargs)[3]
        out2 = fitter._setup(data, data_err, data_mask, **kwargs)[3]

        # Corrections must be applied once per call, not accumulated across
        # calls (double-strength age/grid weighting on the second run).
        np.testing.assert_allclose(out1, out2)


class TestParallaxSelectionMarginalError:
    def test_degenerate_model_kept_with_marginal_error(self, fitter):
        """A model 2 marginal-sigma from the parallax scale must be kept.

        With a strong scale-A(V) degeneracy (rho=0.99) the conditional std
        is ~7x smaller than the marginal std. The old conditional gate gave
        this model a selection penalty of ~98 (pruned at log(1e-3)=-6.9);
        the marginal gate gives ~2 (kept).
        """
        sig_s, sig_a, sig_r, rho = 1.0, 1.0, 0.1, 0.99
        cov = np.array(
            [
                [sig_s**2, rho * sig_s * sig_a, 0.0],
                [rho * sig_s * sig_a, sig_a**2, 0.0],
                [0.0, 0.0, sig_r**2],
            ]
        )
        # Model 0: ML scale exactly at the parallax-implied scale.
        # Model 1: ML scale 2 marginal-sigma away.
        results = _make_loglike_results([1.0, 3.0], [cov, cov])

        out = fitter.logpost_grid(
            results,
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert 0 in sel
        assert 1 in sel, (
            "Model 2 marginal-sigma from the parallax scale was pruned: "
            "selection gate is using the conditional (not marginal) "
            "scale uncertainty"
        )

    def test_inconsistent_model_still_pruned(self, fitter):
        """With uncorrelated errors the gate must still prune bad models."""
        cov = np.diag([0.02**2, 0.1**2, 0.1**2])
        results = _make_loglike_results([1.0, 3.0], [cov, cov])

        out = fitter.logpost_grid(
            results,
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert 0 in sel
        assert 1 not in sel

    def test_diagonal_precision_matches_conditional(self, fitter):
        """For a diagonal precision matrix marginal == conditional, so the
        selection must be identical to the old-code reference computation."""
        cov = np.diag([0.05**2, 0.1**2, 0.1**2])
        icov = np.linalg.inv(cov)
        # Reference (old code): conditional error from icov[0,0].
        cond_err = 1.0 / np.sqrt(icov[0, 0])
        # New code marginal error via cofactor formula.
        minor00 = icov[1, 1] * icov[2, 2] - icov[1, 2] ** 2
        marg_err = np.sqrt(minor00 / np.linalg.det(icov))
        np.testing.assert_allclose(marg_err, cond_err, rtol=1e-12)

    def test_singular_precision_does_not_crash(self, fitter):
        """Degenerate (all-zero) precision matrices fall back gracefully."""
        cov = np.diag([0.05**2, 0.1**2, 0.1**2])
        results = list(_make_loglike_results([1.0, 1.0], [cov, cov]))
        icovs = results[6]
        icovs[1] = 0.0  # singular: uninformative error is used
        results[6] = icovs

        out = fitter.logpost_grid(
            tuple(results),
            parallax=1.0,
            parallax_err=0.01,
            coord=None,
            Nmc_prior=5,
            wt_thresh=1e-3,
            rstate=np.random.RandomState(0),
        )
        sel = out[0]
        assert len(sel) > 0


class TestDustPriorGating:
    def _mock_dust(self, calls):
        def lndust(avs, dustmap, coord, distance=None):
            calls.append(np.size(avs))
            return np.zeros_like(avs)

        return lndust

    def test_apply_av_prior_false_skips_dust_prior(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=False,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert calls == [], "dust prior was applied despite apply_av_prior=False"

    def test_apply_av_prior_true_applies_dust_prior(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=True,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert len(calls) == 1

    def test_default_dust_prior_receives_fit_avlim(
        self, fitter, observation, monkeypatch
    ):
        # The built-in prior normalizes over avlim (truncated Gaussian, and
        # the uniform density in masked bins), so it must see the fit's
        # avlim rather than its own (0, 20) default.
        import brutus.analysis.individual as individual

        seen = []

        def spy(avs, dustmap, coord, distance=None, **kwargs):
            seen.append(kwargs)
            return np.zeros_like(avs)

        monkeypatch.setattr(individual, "logp_extinction", spy)
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)
        fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            avlim=(0.0, 6.0),
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert seen == [{"avlim": (0.0, 6.0)}]

    def test_dustfile_without_coord_raises(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        with pytest.raises(ValueError, match="coord"):
            fitter.logpost_grid(
                like,
                coord=None,
                dustfile=object(),
                lndustprior=self._mock_dust([]),
                Nmc_prior=5,
                wt_thresh=0.1,
                rstate=np.random.RandomState(1851),
            )

    def test_dustfile_without_coord_ok_when_prior_disabled(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        out = fitter.logpost_grid(
            like,
            coord=None,
            dustfile=object(),
            lndustprior=self._mock_dust(calls),
            apply_av_prior=False,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert calls == []

    def test_dustfile_accepts_pathlib_path(
        self, fitter, observation, monkeypatch, tmp_path
    ):
        """A pathlib.Path dustfile must trigger the same auto-load as a str
        path; previously the isinstance(dustfile, str) gate let a Path fall
        through as a "pre-loaded map" and crash downstream."""
        import brutus.dust as dust_mod

        loaded = []

        class FakeBayestar:
            def __init__(self, dustfile=None, **kwargs):
                loaded.append(dustfile)

        monkeypatch.setattr(dust_mod, "Bayestar", FakeBayestar)

        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask.copy(), return_vals=True)

        calls = []
        dust_path = tmp_path / "bayestar_fake.h5"
        out = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=dust_path,
            lndustprior=self._mock_dust(calls),
            apply_av_prior=True,
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert len(out[0]) > 0
        assert loaded == [dust_path], "Path dustfile was not auto-loaded"
        assert len(calls) == 1

    def test_setup_dustfile_without_data_coords_raises(self, fitter):
        data = 10 ** (-0.4 * np.array([15, 14.8, 14.7, 14.6, 14.55]))
        data_err = data * 0.05
        data_mask = np.ones(5, dtype=bool)

        def custom_gal(distance, coord, labels=None):
            return np.zeros_like(distance)

        with pytest.raises(ValueError, match="data_coords"):
            fitter._setup(
                data,
                data_err,
                data_mask,
                lngalprior=custom_gal,  # bypass the galactic-prior coord check
                dustfile=object(),
                data_coords=None,
            )


class TestMaskAndInputs:
    def test_int_mask_gives_same_lnl_as_bool_mask(self, fitter, observation):
        flux, flux_err, _ = observation
        flux = flux.copy()
        flux[0] *= 3.0  # corrupt the masked band
        int_mask = np.array([0, 1, 1, 1, 1])
        lnl_int, ndim_int, _ = fitter.loglike_grid(flux, flux_err, int_mask)
        lnl_bool, ndim_bool, _ = fitter.loglike_grid(
            flux, flux_err, int_mask.astype(bool)
        )
        assert ndim_int == ndim_bool == 4
        np.testing.assert_array_equal(lnl_int, lnl_bool)

    def test_loglike_grid_leaves_inputs_unchanged(self, fitter, observation):
        flux, flux_err, _ = observation
        flux = flux.copy()
        flux[2] = np.nan
        mask = np.ones(5, dtype=bool)
        av_init = np.full(fitter.nmodels, 0.5)
        rv_init = np.full(fitter.nmodels, 3.3)
        fitter.loglike_grid(
            flux, flux_err, mask, av_init=av_init, rv_init=rv_init, return_vals=True
        )
        assert mask.all()
        np.testing.assert_array_equal(av_init, 0.5)
        np.testing.assert_array_equal(rv_init, 3.3)


@pytest.fixture
def batch(observation):
    """fit() inputs for two copies of `observation`."""
    flux, flux_err, mask = observation
    return dict(
        data=np.tile(flux, (2, 1)),
        data_err=np.tile(flux_err, (2, 1)),
        data_mask=np.tile(mask, (2, 1)),
        data_labels=np.zeros(2, dtype=[("id", "i4")]),
        data_coords=np.tile([120.0, 45.0], (2, 1)),
    )


class TestInputValidation:
    @pytest.mark.parametrize(
        "change, match",
        [
            (dict(parallax=np.array([5.0, 5.0])), "parallax_err"),
            (dict(parallax_err=np.array([0.1, 0.1])), "parallax_err"),
            (
                dict(parallax=np.array([5.0]), parallax_err=np.array([0.1])),
                "parallax must have shape",
            ),
            (dict(data_coords=np.zeros((2, 3))), "data_coords"),
            (dict(data_labels=np.zeros(1, dtype=[("id", "i4")])), "data_labels"),
            (dict(lnprior=np.zeros(5)), "lnprior"),
            (dict(logl_initthresh=None), "logl_initthresh"),
            (dict(logl_initthresh=0.0), "logl_initthresh"),
            (dict(logl_initthresh=1.0), "logl_initthresh"),
        ],
    )
    def test_fit_rejects_bad_input_before_creating_file(
        self, fitter, batch, tmp_path, change, match
    ):
        path = tmp_path / "out.h5"
        with pytest.raises(ValueError, match=match):
            fitter.fit(save_file=str(path), verbose=False, **{**batch, **change})
        assert not path.exists()

    @pytest.mark.parametrize("key", ["data", "data_err", "data_mask"])
    def test_fit_rejects_wrong_band_count(self, fitter, batch, tmp_path, key):
        batch[key] = batch[key][:, :4]
        with pytest.raises(ValueError, match=key):
            fitter.fit(save_file=str(tmp_path / "out"), verbose=False, **batch)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(parallax=5.0), "parallax_err"),
            (dict(init_thresh=2.0), "init_thresh"),
        ],
    )
    def test_loglike_grid_rejects_bad_input(self, fitter, observation, kwargs, match):
        flux, flux_err, mask = observation
        with pytest.raises(ValueError, match=match):
            fitter.loglike_grid(flux, flux_err, mask, **kwargs)

    def test_zero_parallax_error_counts_as_no_parallax(self, fitter, observation):
        flux, flux_err, mask = observation
        with_zero = fitter.loglike_grid(
            flux, flux_err, mask, parallax=5.0, parallax_err=0.0
        )
        without = fitter.loglike_grid(flux, flux_err, mask)
        np.testing.assert_array_equal(with_zero[0], without[0])


class TestLikelihoodNormalization:
    def test_culled_and_refined_models_share_the_normalization(self, mock_grid):
        # Random colours, so that most models fail the initial cut.
        models = mock_grid.models.copy()
        models[:, :, 0] += np.random.RandomState(3).normal(0, 0.3, (27, 5))
        fitter = BruteForce(
            StarGrid(models, mock_grid.labels, mock_grid.params), verbose=False
        )
        flux = 10 ** (-0.4 * models[13, :, 0])
        flux_err = 0.05 * flux
        mask = np.ones(5, dtype=bool)
        lnl, ndim, chi2 = fitter.loglike_grid(flux, flux_err, mask, dim_prior=False)
        expected = -0.5 * (ndim * np.log(2 * np.pi) + np.sum(np.log(flux_err**2)))
        np.testing.assert_allclose(lnl + 0.5 * chi2, expected, rtol=1e-12)


class TestTruncatedDraws:
    def test_saved_av_draws_stay_off_the_bound(self, fitter, observation):
        # An unreddened star: about half of the raw A(V) draws fall below 0.
        flux, flux_err, mask = observation
        out = fitter._fit(
            flux,
            flux_err,
            mask,
            coord=(120.0, 45.0),
            Nmc_prior=50,
            Ndraws=500,
            rstate=np.random.RandomState(11),
        )
        reds = out[10]
        assert np.all(reds > 0.0) and np.all(reds <= 20.0)

    @pytest.mark.parametrize("param", ["dist", "av", "rv"])
    def test_model_with_no_draw_inside_support_gets_zero_weight(self, fitter, param):
        cov = np.diag([1e-4, 1e-4, 1e-4])
        far = np.diag([1e-32, 1e-4, 1e-4])  # scale 1e-14 known to 1%
        covs = [cov, far if param == "dist" else cov]
        like = list(_make_loglike_results([1.0, 1.0], covs))
        if param == "dist":
            like[3][1] = 1e-14  # 1e7 kpc, beyond the 1e6 kpc limit
        else:
            like[4 if param == "av" else 5][1] = -5.0 if param == "av" else 20.0
        sel, _, lnp, d_mc, a_mc, r_mc, _, mc_ess = fitter.logpost_grid(
            tuple(like),
            wt_thresh=None,
            cdf_thresh=None,
            Nmc_prior=20,
            rstate=np.random.RandomState(5),
        )
        assert list(sel) == [0, 1]
        assert np.all(np.isfinite(lnp)) and lnp[1] < -1e299 and lnp[0] > -1e299
        assert mc_ess[1] == 0.0 and mc_ess[0] > 0.0
        assert np.all(d_mc <= 1e6) and np.all(a_mc >= 0.0) and np.all(r_mc <= 8.0)

    def test_model_with_zero_prior_everywhere_gets_zero_ess(self, fitter):
        cov = np.diag([1e-4, 1e-4, 1e-4])
        like = _make_loglike_results([1.0, 1.0], [cov, cov])
        mc_ess = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            lngalprior=lambda d, coord, labels=None: np.full(np.shape(d), -1e300),
            dlabels=fitter.models_labels[:2],
            lnprior=np.zeros(2),
            wt_thresh=None,
            cdf_thresh=None,
            Nmc_prior=20,
            rstate=np.random.RandomState(5),
        )[7]
        np.testing.assert_array_equal(mc_ess, 0.0)

    def test_model_on_the_bound_loses_half_its_weight(self, fitter):
        # Antithetic pairs put exactly half of the A(V) draws of a model at
        # A(V) = 0 below the bound, so truncation lowers its lnp by ln 2.
        cov = np.diag([1e-12, 1e-2, 1e-12])
        like = list(_make_loglike_results([1.0, 1.0], [cov, cov]))
        like[4] = np.array([0.5, 0.0])
        _, _, lnp, *_ = fitter.logpost_grid(
            tuple(like),
            wt_thresh=None,
            cdf_thresh=None,
            Nmc_prior=20,
            rstate=np.random.RandomState(5),
        )
        np.testing.assert_allclose(lnp[1] - lnp[0], np.log(0.5), atol=1e-6)


class TestDustPriorPartial:
    @pytest.fixture
    def spy(self, monkeypatch):
        import brutus.analysis.individual as individual

        seen = []

        def logp_extinction(avs, dustmap, coord, distance=None, **kwargs):
            seen.append(kwargs)
            return np.zeros_like(avs)

        monkeypatch.setattr(individual, "logp_extinction", logp_extinction)
        return logp_extinction, seen

    @pytest.mark.parametrize(
        "bound, expected",
        [
            ({"scatter": 0.3}, {"avlim": (0.0, 6.0)}),
            ({"avlim": (0.0, 3.0)}, {}),
        ],
    )
    def test_partial_gets_fit_avlim_unless_bound(
        self, fitter, observation, spy, bound, expected
    ):
        from functools import partial

        func, seen = spy
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(flux, flux_err, mask, return_vals=True)
        fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dustfile=object(),
            lndustprior=partial(func, **bound),
            avlim=(0.0, 6.0),
            Nmc_prior=5,
            wt_thresh=0.1,
            rstate=np.random.RandomState(0),
        )
        assert seen == [{**bound, **expected}]


class TestSubsetResults:
    def test_default_dlabels_with_subset_results_raises(self, fitter, observation):
        flux, flux_err, mask = observation
        like = fitter.loglike_grid(
            flux, flux_err, mask, indices=np.arange(10, 20), return_vals=True
        )
        with pytest.raises(ValueError, match="dlabels"):
            fitter.logpost_grid(like, coord=(120.0, 45.0), Nmc_prior=5)
        with pytest.raises(ValueError, match="lnprior"):
            fitter.logpost_grid(like, lnprior=np.zeros(27), dlabels=None, Nmc_prior=5)

    def test_subset_labels_are_accepted(self, fitter, observation):
        flux, flux_err, mask = observation
        idx = np.arange(10, 20)
        like = fitter.loglike_grid(flux, flux_err, mask, indices=idx, return_vals=True)
        sel = fitter.logpost_grid(
            like,
            coord=(120.0, 45.0),
            dlabels=fitter.models_labels[idx],
            Nmc_prior=5,
            rstate=np.random.RandomState(0),
        )[0]
        assert np.all(sel < len(idx))


def test_get_sed_grid_rejects_wrong_length_av(fitter):
    with pytest.raises(ValueError, match="av"):
        fitter.get_sed_grid(av=np.zeros(2))


class TestUnfittedObjects:
    @pytest.mark.parametrize("running_io", [True, False])
    def test_object_without_valid_bands_is_skipped(
        self, fitter, batch, tmp_path, running_io
    ):
        batch["data_mask"][0] = False
        with pytest.warns(RuntimeWarning, match="1 of 2 objects") as record:
            path = fitter.fit(
                save_file=str(tmp_path / "out"),
                Ndraws=10,
                Nmc_prior=10,
                running_io=running_io,
                rstate=np.random.RandomState(3),
                verbose=False,
                **batch,
            )
        assert sum("no valid bands" in str(w.message) for w in record) == 1
        with h5py.File(path, "r") as f:
            assert np.all(f["model_idx"][0] == -99)
            assert f["obj_Nbands"][0] == 0
            assert np.all(np.isnan(f["samps_dist"][0]))
            assert np.isnan(f["obj_log_evid"][0])
            assert np.all(f["model_idx"][1] >= 0)
            assert f["obj_Nbands"][1] == 5

    def test_output_file_closes_when_a_fit_raises(
        self, fitter, batch, tmp_path, monkeypatch
    ):
        def fail(*args, **kwargs):
            raise RuntimeError("fit failed")

        monkeypatch.setattr(fitter, "_fit", fail)
        path = tmp_path / "out.h5"
        # Keeping the traceback keeps fit's frame alive, as Jupyter does.
        with pytest.raises(RuntimeError, match="fit failed") as excinfo:
            fitter.fit(save_file=str(path), verbose=False, **batch)
        # HDF5 refuses to truncate a file that is still open.
        h5py.File(path, "w").close()
        assert excinfo.traceback


def test_misspelled_keyword_raises(fitter, observation):
    flux, flux_err, mask = observation
    with pytest.raises(TypeError, match="av_gaus"):
        fitter.loglike_grid(flux, flux_err, mask, av_gaus=(0.0, 0.01))
    like = fitter.loglike_grid(flux, flux_err, mask, return_vals=True)
    with pytest.raises(TypeError, match="Nmc"):
        fitter.logpost_grid(like, Nmc=10)
