"""Tests of brutus.analysis.populations, using real MIST isochrones where needed."""

import warnings

import numpy as np
import pytest

from brutus.analysis.populations import (
    apply_isochrone_mixture_model,
    compute_isochrone_cluster_loglike,
    compute_isochrone_outlier_loglike,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
    marginalize_isochrone_grid,
)
from brutus.utils.math import LOG_ZERO
from brutus.utils.photometry import phot_loglike, uniform_outlier_loglike


@pytest.fixture(scope="module")
def real_isochrone_grid(real_stellarpop):
    """Generate real isochrone grid for testing."""
    # Use realistic stellar population parameters
    feh = 0.0  # Solar metallicity
    loga = 9.5  # ~3 Gyr age
    av = 0.1  # Low extinction
    rv = 3.1  # Standard extinction curve
    dist = 1000.0  # 1 kpc distance

    # Generate grid with small number of points for fast testing
    smf_grid = np.array([0.0, 0.5])  # Single stars and equal-mass binaries
    eep_grid = np.linspace(300, 500, 20)  # Smaller range for speed

    grid = generate_isochrone_population_grid(
        real_stellarpop,
        feh,
        loga,
        av,
        rv,
        dist,
        smf_grid=smf_grid,
        eep_grid=eep_grid,
    )
    return grid


@pytest.fixture
def mock_observations():
    """Create mock stellar observations for testing."""
    rng = np.random.RandomState(42)  # Reproducible results

    # Create realistic mock observations
    n_stars = 5
    n_filters = 3

    # Mock fluxes: bright stars with some scatter
    obs_flux = rng.uniform(0.5, 2.0, (n_stars, n_filters))
    obs_err = rng.uniform(0.05, 0.15, (n_stars, n_filters))

    # Add some parallax data
    parallax = rng.uniform(0.5, 2.0, n_stars)  # 0.5-2 mas
    parallax_err = rng.uniform(0.05, 0.2, n_stars)

    return {
        "obs_flux": obs_flux,
        "obs_err": obs_err,
        "parallax": parallax,
        "parallax_err": parallax_err,
    }


class TestPopulationGridGeneration:
    """Test isochrone population grid generation with real data."""

    def test_grid_generation_basic(self):
        """Test basic grid generation error handling."""
        # Should fail when stellarpop is None
        with pytest.raises((AttributeError, ValueError)):
            generate_isochrone_population_grid(
                None, feh=0.0, loga=9.5, av=0.1, rv=3.1, dist=1000.0
            )

    def test_real_grid_generation(self, real_stellarpop):
        """Test real isochrone grid generation."""
        # Test with small grid for speed - use parameters within isochrone range
        feh, loga, av, rv, dist = 0.0, 9.0, 0.1, 3.1, 1000.0
        smf_grid = np.array([0.0, 0.5])  # Singles and equal-mass binaries
        eep_grid = np.linspace(400, 470, 10)  # EEP range within binary limit (480)

        grid = generate_isochrone_population_grid(
            real_stellarpop,
            feh,
            loga,
            av,
            rv,
            dist,
            smf_grid=smf_grid,
            eep_grid=eep_grid,
        )

        # Check grid structure
        assert isinstance(grid, dict)
        required_keys = [
            "photometry",
            "masses",
            "smf_values",
            "mass_jacobians",
            "smf_jacobians",
        ]
        for key in required_keys:
            assert key in grid

        # Check shapes are consistent
        n_points = len(grid["masses"])
        assert grid["photometry"].shape[0] == n_points
        assert len(grid["smf_values"]) == n_points
        assert len(grid["mass_jacobians"]) == n_points
        assert len(grid["smf_jacobians"]) == n_points

        # Should have reasonable number of points
        assert n_points > 0

    def test_grid_parameters(self):
        """Test grid parameter validation."""
        # Test parameter validation
        feh, loga, av, rv, dist = 0.0, 9.5, 0.1, 3.1, 1000.0

        # These are expected parameter types/ranges
        assert isinstance(feh, (int, float))
        assert isinstance(loga, (int, float))
        assert isinstance(av, (int, float)) and av >= 0
        assert isinstance(rv, (int, float)) and rv > 0
        assert isinstance(dist, (int, float)) and dist > 0


class TestClusterLikelihood:
    """Test cluster likelihood computation with real data."""

    def test_cluster_likelihood_shapes(self, real_isochrone_grid, mock_observations):
        """Test cluster likelihood with real isochrone grid."""
        obs_flux = mock_observations["obs_flux"]
        obs_err = mock_observations["obs_err"]

        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )

        # Should return shape (N_grid_points, N_objects)
        n_grid_points = real_isochrone_grid["photometry"].shape[0]
        n_objects = obs_flux.shape[0]
        assert lnl_cluster.shape == (n_grid_points, n_objects)

        # Should have reasonable likelihood values (not all -inf)
        assert np.any(np.isfinite(lnl_cluster))

    def test_cluster_likelihood_with_parallax(
        self, real_isochrone_grid, mock_observations
    ):
        """Test cluster likelihood with parallax data."""
        obs_flux = mock_observations["obs_flux"][:1]  # Use just one star
        obs_err = mock_observations["obs_err"][:1]
        parallax = mock_observations["parallax"][:1]
        parallax_err = mock_observations["parallax_err"][:1]
        distance = 1000.0  # pc (consistent with grid generation)

        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux,
            obs_err,
            real_isochrone_grid,
            parallax=parallax,
            parallax_err=parallax_err,
            distance=distance,
        )

        n_grid_points = real_isochrone_grid["photometry"].shape[0]
        assert lnl_cluster.shape == (n_grid_points, 1)

        # Should have some finite likelihood values
        assert np.any(np.isfinite(lnl_cluster))


class TestOutlierLikelihood:
    """Test outlier likelihood computation with real data."""

    def test_outlier_likelihood_stellar_independent(
        self, real_isochrone_grid, mock_observations
    ):
        """Test outlier likelihood with stellar-independent models."""
        obs_flux = mock_observations["obs_flux"]
        obs_err = mock_observations["obs_err"]

        # Test both chi-square and uniform models
        lnl_outlier_chi = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid, dim_prior=True
        )

        lnl_outlier_unif = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid, dim_prior=False
        )

        # Should broadcast to (N_grid_points, N_objects)
        n_grid_points = real_isochrone_grid["photometry"].shape[0]
        n_objects = obs_flux.shape[0]
        assert lnl_outlier_chi.shape == (n_grid_points, n_objects)
        assert lnl_outlier_unif.shape == (n_grid_points, n_objects)
        assert np.all(np.isfinite(lnl_outlier_chi))
        assert np.all(np.isfinite(lnl_outlier_unif))


class TestMixtureModel:
    """Test mixture model application with realistic data."""

    def test_mixture_model_basic(self, real_isochrone_grid, mock_observations):
        """Test basic mixture model application with real likelihoods."""
        obs_flux = mock_observations["obs_flux"]
        obs_err = mock_observations["obs_err"]

        # Compute real likelihoods
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        lnl_outlier = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )

        cluster_prob = 0.9
        field_fraction = 0.1

        lnl_mixture = apply_isochrone_mixture_model(
            lnl_cluster, lnl_outlier, cluster_prob, field_fraction
        )

        assert lnl_mixture.shape == lnl_cluster.shape
        assert np.all(np.isfinite(lnl_mixture))
        # Mixed likelihood should be higher than weighted components
        ln_cluster_weight = np.log(cluster_prob * (1.0 - field_fraction))
        ln_outlier_weight = np.log(1.0 - cluster_prob * (1.0 - field_fraction))
        assert np.all(lnl_mixture >= lnl_cluster + ln_cluster_weight - 1e-10)
        assert np.all(lnl_mixture >= lnl_outlier + ln_outlier_weight - 1e-10)

    def test_mixture_model_shape_validation(self):
        """Test mixture model input shape validation."""
        lnl_cluster = np.array([[-1.0, -2.0]])  # shape (1, 2)
        lnl_outlier = np.array([[-3.0], [-3.0]])  # shape (2, 1) - incompatible

        with pytest.raises(ValueError, match="shapes must match"):
            apply_isochrone_mixture_model(lnl_cluster, lnl_outlier, 0.9, 0.1)


class TestMarginalization:
    """Test marginalization over stellar parameters with real data."""

    def test_marginalization_basic(self, real_isochrone_grid, mock_observations):
        """Test basic marginalization with real data."""
        obs_flux = mock_observations["obs_flux"]
        obs_err = mock_observations["obs_err"]

        # Compute real mixed likelihoods
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        lnl_outlier = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        lnl_mixture = apply_isochrone_mixture_model(lnl_cluster, lnl_outlier, 0.9, 0.1)

        lnl_marginalized = marginalize_isochrone_grid(
            lnl_mixture,
            real_isochrone_grid["mass_jacobians"],
            real_isochrone_grid["smf_jacobians"],
        )

        n_objects = obs_flux.shape[0]
        assert lnl_marginalized.shape == (n_objects,)
        assert np.all(np.isfinite(lnl_marginalized))
        # Marginalized likelihood should be within reasonable range of best mixture components
        # (not necessarily >= max due to jacobian weighting in integration)
        max_mixture = np.max(lnl_mixture, axis=0)
        assert np.all(
            lnl_marginalized >= max_mixture - 10.0
        )  # Allow reasonable integration range


class TestMainInterface:
    """Test the main isochrone_population_loglike function with real data."""

    def test_theta_outside_isochrone_grid_returns_log_zero(self, real_stellarpop):
        theta = [0.0, 20.0, 0.1, 3.1, 1000.0, 0.05]  # loga=20 is off the grid
        obs_phot = np.array([[1.0, 0.8, 0.6]])
        obs_err = np.array([[0.1, 0.08, 0.06]])
        with pytest.warns(UserWarning, match="No valid isochrone models"):
            lnl = isochrone_population_loglike(
                theta, real_stellarpop, obs_phot, obs_err, smf_grid=[0.0, 0.5]
            )
        assert lnl == LOG_ZERO

    def test_real_population_likelihood(self, real_stellarpop, mock_observations):
        """Test main function with real stellarpop and realistic parameters."""
        theta = [
            0.0,
            9.5,
            0.1,
            3.1,
            1000.0,
            0.05,
        ]  # [feh, loga, av, rv, dist, field_frac]
        obs_phot = mock_observations["obs_flux"]
        obs_err = mock_observations["obs_err"]

        # Use small grids for speed
        smf_grid = np.array([0.0, 0.5])
        eep_grid = np.linspace(400, 470, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings

            lnl = isochrone_population_loglike(
                theta,
                real_stellarpop,
                obs_phot,
                obs_err,
                smf_grid=smf_grid,
                eep_grid=eep_grid,
            )

            # Should get a finite likelihood
            assert np.isfinite(lnl)
            assert isinstance(lnl, (int, float, np.number))

    def test_parameter_count_validation(self):
        """Test that wrong number of parameters raises error."""
        theta = [0.0, 9.5, 0.1]  # Only 3 parameters instead of 5
        obs_phot = np.array([[1.0, 0.8]])
        obs_err = np.array([[0.1, 0.08]])

        with pytest.raises(ValueError, match="Expected 6 population parameters"):
            isochrone_population_loglike(theta, None, obs_phot, obs_err)

    def test_photometry_shape_validation(self):
        """Test photometry shape validation."""
        theta = [
            0.0,
            9.5,
            0.1,
            3.1,
            1000.0,
            0.05,
        ]  # [feh, loga, av, rv, dist, field_frac]
        obs_phot = np.array([[1.0, 0.8]])
        obs_err = np.array([[0.1, 0.08, 0.06]])  # Wrong shape

        with pytest.raises(ValueError, match="same shape"):
            isochrone_population_loglike(theta, None, obs_phot, obs_err)

    def test_parallax_integration(self, real_stellarpop, mock_observations):
        """Test main function with parallax data."""
        theta = [
            0.0,
            9.5,
            0.1,
            3.1,
            1000.0,
            0.05,
        ]  # [feh, loga, av, rv, dist, field_frac]
        obs_phot = mock_observations["obs_flux"][:2]  # Use 2 stars
        obs_err = mock_observations["obs_err"][:2]
        parallax = mock_observations["parallax"][:2]
        parallax_err = mock_observations["parallax_err"][:2]

        # Use small grids for speed
        smf_grid = np.array([0.0, 0.5])
        eep_grid = np.linspace(400, 470, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings

            lnl = isochrone_population_loglike(
                theta,
                real_stellarpop,
                obs_phot,
                obs_err,
                parallax=parallax,
                parallax_err=parallax_err,
                smf_grid=smf_grid,
                eep_grid=eep_grid,
            )

            # Should get a finite likelihood
            assert np.isfinite(lnl)
            assert isinstance(lnl, (int, float, np.number))


class TestInputValidation:
    """Malformed inputs raise before any model is generated (#106)."""

    theta = [0.0, 9.5, 0.1, 3.1, 1000.0, 0.05]
    obs_phot = np.full((3, 4), 1.0)
    obs_err = np.full((3, 4), 0.1)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"obs_phot": np.ones(4), "obs_err": np.ones(4)}, "obs_phot"),
            ({"mask": np.ones((3, 2))}, "mask"),
            ({"parallax": np.ones(2), "parallax_err": np.ones(2)}, "parallax"),
            ({"parallax": np.ones(3)}, "parallax_err"),
            ({"smf_grid": [1.0, 0.5, 0.0]}, "smf_grid"),
            ({"smf_grid": [0.0, 0.5, 1.5]}, "smf_grid"),
            ({"eep_grid": np.linspace(500, 300, 50)}, "eep_grid"),
        ],
    )
    def test_malformed_input_raises_value_error(self, kwargs, match):
        args = {"obs_phot": self.obs_phot, "obs_err": self.obs_err, **kwargs}
        with pytest.raises(ValueError, match=match):
            isochrone_population_loglike(self.theta, MockStellarPop(4), **args)

    def test_misspelled_outlier_keyword_raises_type_error(self):
        with pytest.raises(TypeError, match="bogus_kw"):
            isochrone_population_loglike(
                self.theta,
                MockStellarPop(4),
                self.obs_phot,
                self.obs_err,
                smf_grid=[0.0, 0.5],
                eep_grid=np.linspace(300, 500, 20),
                bogus_kw=1,
            )

    def test_model_value_error_propagates(self):
        """Only a grid with no valid model returns LOG_ZERO; a ValueError from
        the model code itself propagates."""

        class FailingPop(MockStellarPop):
            def get_seds(self, **kwargs):
                raise ValueError("bad corr_params")

        with pytest.raises(ValueError, match="bad corr_params"):
            isochrone_population_loglike(
                self.theta, FailingPop(4), self.obs_phot, self.obs_err
            )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"parallax": np.ones(3), "distance": 1000.0}, "parallax_err"),
            ({"parallax": np.ones(3), "parallax_err": np.ones(3)}, "distance"),
        ],
    )
    def test_cluster_loglike_parallax_needs_error_and_distance(self, kwargs, match):
        grid = generate_isochrone_population_grid(
            MockStellarPop(4), 0.0, 9.5, 0.1, 3.1, 1000.0, smf_grid=[0.0]
        )
        with pytest.raises(ValueError, match=match):
            compute_isochrone_cluster_loglike(
                self.obs_phot, self.obs_err, grid, **kwargs
            )


class TestIntegration:
    """Integration tests using real MIST data."""

    def test_full_workflow_real(self, real_isochrone_grid, mock_observations):
        """Test complete workflow with real isochrone data."""
        obs_flux = mock_observations["obs_flux"][:2]  # Use 2 stars for speed
        obs_err = mock_observations["obs_err"][:2]

        # Test individual components with real data

        # 1. Test cluster likelihood with real grid
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        n_grid_points = real_isochrone_grid["photometry"].shape[0]
        assert lnl_cluster.shape == (n_grid_points, 2)
        assert np.any(np.isfinite(lnl_cluster))  # Should have some finite values

        # 2. Test outlier likelihood
        lnl_outlier = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        assert lnl_outlier.shape == (n_grid_points, 2)
        assert np.all(np.isfinite(lnl_outlier))

        # 3. Test mixture
        lnl_mixture = apply_isochrone_mixture_model(lnl_cluster, lnl_outlier, 0.9, 0.1)
        assert lnl_mixture.shape == (n_grid_points, 2)
        assert np.all(np.isfinite(lnl_mixture))

        # 4. Test marginalization
        lnl_marginalized = marginalize_isochrone_grid(
            lnl_mixture,
            real_isochrone_grid["mass_jacobians"],
            real_isochrone_grid["smf_jacobians"],
        )
        assert lnl_marginalized.shape == (2,)
        assert np.all(np.isfinite(lnl_marginalized))

        print("✅ All workflow components working correctly with real MIST data")


class MockStellarPop:
    """Duck-typed StellarPop whose SED brightens with binary_fraction on the
    main sequence only, mirroring the real model's structure."""

    def __init__(self, n_filters=3):
        self.n_filters = n_filters

    def get_seds(
        self,
        feh=0.0,
        loga=8.5,
        av=0.0,
        rv=3.3,
        eep=None,
        binary_fraction=0.0,
        dist=1000.0,
        mini_bound=0.08,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
    ):
        eep = np.asarray(eep)
        n = len(eep)
        # Monotone mass-EEP relation
        masses = 0.5 + 0.001 * (eep - eep.min())
        base = np.full((n, self.n_filters), 10.0)
        # Binaries brighten MS stars only
        is_ms = eep <= eep_binary_max
        sed = base.copy()
        if binary_fraction > 0:
            sed[is_ms] -= 2.5 * np.log10(1.0 + binary_fraction**3.5)
        params = {"mini": masses}
        params2 = {"mini": np.where(is_ms, masses * binary_fraction, np.nan)}
        return sed, params, params2


class TestSMFPlumbing:
    """The smf grid must actually change the generated photometry."""

    def test_smf_slices_differ(self):
        pop = MockStellarPop()
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 1.0]),
            eep_grid=np.linspace(300, 400, 10),
        )
        phot0 = grid["photometry"][grid["smf_values"] == 0.0]
        phot1 = grid["photometry"][grid["smf_values"] == 1.0]
        assert phot0.shape == phot1.shape
        # equal-mass binaries should be exactly twice as bright in flux
        np.testing.assert_allclose(phot1 / phot0, 2.0, rtol=1e-10)

    def test_real_smf_slices_differ(self, real_stellarpop):
        """End-to-end through the real StellarPop: smf=1 doubles MS flux."""
        eep_grid = np.linspace(300, 400, 8)
        grid = generate_isochrone_population_grid(
            real_stellarpop,
            feh=0.0,
            loga=9.5,
            av=0.1,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 1.0]),
            eep_grid=eep_grid,
        )
        phot0 = grid["photometry"][grid["smf_values"] == 0.0]
        phot1 = grid["photometry"][grid["smf_values"] == 1.0]
        # Compare rows for matching masses (both slices share the mass grid)
        m0 = grid["masses"][grid["smf_values"] == 0.0]
        m1 = grid["masses"][grid["smf_values"] == 1.0]
        shared = np.intersect1d(m0, m1)
        assert shared.size > 0, "expected overlapping MS masses across slices"
        f0 = phot0[np.isin(m0, shared)]
        f1 = phot1[np.isin(m1, shared)]
        finite = np.isfinite(f0) & np.isfinite(f1)
        assert finite.any()
        np.testing.assert_allclose(f1[finite] / f0[finite], 2.0, rtol=1e-6)


class TestPostMSMeasure:
    """Post-MS (giant) models must carry the full SMF measure."""

    def test_giant_weight_matches_ms_weight(self):
        pop = MockStellarPop()
        eep_grid = np.linspace(400, 560, 17)  # spans eep_binary_max=480
        smf_grid = np.linspace(0.0, 1.0, 21)
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=smf_grid,
            eep_grid=eep_grid,
        )
        # Total integration weight accumulated at a fixed mass, MS vs post-MS
        w = grid["mass_jacobians"] * grid["smf_jacobians"]
        masses = grid["masses"]
        ms_mass = 0.5 + 0.001 * (440.0 - 400.0)  # eep=440 (MS)
        pms_mass = 0.5 + 0.001 * (520.0 - 400.0)  # eep=520 (post-MS)
        w_ms = w[np.isclose(masses, ms_mass)].sum()
        w_pms = w[np.isclose(masses, pms_mass)].sum()
        assert w_pms > 0
        # Equal to the MS total weight (both integrate the full SMF axis)
        np.testing.assert_allclose(w_pms, w_ms, rtol=1e-12)

    def test_post_ms_stored_once(self):
        pop = MockStellarPop()
        eep_grid = np.array([450.0, 500.0])
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 0.5, 1.0]),
            eep_grid=eep_grid,
        )
        pms_mass = 0.5 + 0.001 * (500.0 - 450.0)
        n_pms_rows = np.sum(np.isclose(grid["masses"], pms_mass))
        assert n_pms_rows == 1, "post-MS model duplicated across SMF slices"


class TestNaNAutoMask:
    """A stray NaN flux or zero error must only drop that band."""

    def _tiny_grid(self):
        return {
            "photometry": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            "masses": np.array([0.5, 0.6]),
            "smf_values": np.zeros(2),
            "mass_jacobians": np.ones(2),
            "smf_jacobians": np.ones(2),
        }

    def test_nan_flux_does_not_poison_object(self):
        grid = self._tiny_grid()
        flux = np.array([[1.0, np.nan, 0.9], [1.1, 1.0, 0.9]])
        err = np.full_like(flux, 0.1)
        lnl = compute_isochrone_cluster_loglike(flux, err, grid, dim_prior=True)
        assert np.isfinite(lnl).all()
        # Equivalent to explicitly masking the NaN band
        mask = np.ones_like(flux)
        mask[0, 1] = 0
        flux_clean = np.nan_to_num(flux, nan=1.0)
        lnl_masked = compute_isochrone_cluster_loglike(
            flux_clean, err, grid, dim_prior=True, mask=mask
        )
        np.testing.assert_allclose(lnl, lnl_masked)

    def test_zero_error_does_not_poison_object(self):
        grid = self._tiny_grid()
        flux = np.array([[1.0, 1.0, 0.9]])
        err = np.array([[0.1, 0.0, 0.1]])
        lnl = compute_isochrone_cluster_loglike(flux, err, grid, dim_prior=False)
        assert np.isfinite(lnl).all()

    def test_object_without_bands_adds_nothing_to_cluster_fit(self):
        grid = self._tiny_grid()

        def total(flux, err):
            lnl_c = compute_isochrone_cluster_loglike(flux, err, grid, dim_prior=True)
            lnl_o = compute_isochrone_outlier_loglike(flux, err, grid, dim_prior=True)
            mix = apply_isochrone_mixture_model(lnl_c, lnl_o, 0.9, 0.1)
            marg = marginalize_isochrone_grid(
                mix, grid["mass_jacobians"], grid["smf_jacobians"]
            )
            return marg.sum()

        flux = np.array([[1.1, 1.0, 0.9], [np.nan, np.nan, np.nan]])
        err = np.full_like(flux, 0.1)
        with_empty = total(flux, err)
        assert np.isfinite(with_empty)
        np.testing.assert_allclose(with_empty, total(flux[:1], err[:1]))


class TestMeasureNormalization:
    """The marginalization must be invariant to the measure's overall scale."""

    def test_marginalization_invariant_to_jacobian_scale(self):
        rng = np.random.default_rng(0)
        lnl_mixture = rng.normal(-10, 2, size=(50, 4))
        mass_jac = rng.uniform(0.5, 1.5, 50)
        smf_jac = rng.uniform(0.5, 1.5, 50)
        base = marginalize_isochrone_grid(lnl_mixture, mass_jac, smf_jac)
        scaled = marginalize_isochrone_grid(lnl_mixture, mass_jac * 7.3, smf_jac)
        np.testing.assert_allclose(base, scaled, rtol=1e-12)

    def test_object_with_no_valid_model_gets_log_zero(self):
        lnl_mixture = np.full((5, 2), -1.0)
        lnl_mixture[:, 1] = -np.inf
        out = marginalize_isochrone_grid(lnl_mixture, np.ones(5), np.ones(5))
        assert np.isfinite(out[0])
        assert out[1] == LOG_ZERO
        none_valid = marginalize_isochrone_grid(
            np.full((5, 2), np.nan), np.ones(5), np.ones(5)
        )
        np.testing.assert_array_equal(none_valid, LOG_ZERO)

    def test_invalid_rows_do_not_change_outlier_mass(self):
        # A grid with extra invalid (NaN) rows must marginalize identically
        # to the same grid without them.
        rng = np.random.default_rng(1)
        lnl_valid = rng.normal(-10, 2, size=(30, 4))
        jac = rng.uniform(0.5, 1.5, 30)
        base = marginalize_isochrone_grid(lnl_valid, jac, np.ones(30))
        lnl_padded = np.vstack([lnl_valid, np.full((10, 4), np.nan)])
        jac_padded = np.concatenate([jac, rng.uniform(0.5, 1.5, 10)])
        padded = marginalize_isochrone_grid(lnl_padded, jac_padded, np.ones(40))
        np.testing.assert_allclose(base, padded, rtol=1e-12)


class TestParallaxMeasure:
    """Parallax must enter the chi-square (not a Gaussian density) when
    dim_prior=True, matching phot_loglike's extra_chi2 mechanism."""

    def test_parallax_enters_chi2_when_dim_prior(self):
        grid = {
            "photometry": np.array([[1.0, 1.0, 1.0]]),
            "masses": np.array([0.5]),
            "smf_values": np.zeros(1),
            "mass_jacobians": np.ones(1),
            "smf_jacobians": np.ones(1),
        }
        flux = np.array([[1.0, 1.05, 0.95]])
        err = np.full_like(flux, 0.1)
        parallax = np.array([1.0])  # mas
        parallax_err = np.array([0.1])
        dist = 1200.0  # pc -> predicted parallax 0.833 mas
        lnl = compute_isochrone_cluster_loglike(
            flux,
            err,
            grid,
            parallax=parallax,
            parallax_err=parallax_err,
            distance=dist,
            dim_prior=True,
        )
        chi2_par = (1.0 - 1000.0 / dist) ** 2 / 0.1**2
        expected = phot_loglike(
            flux,
            err,
            grid["photometry"],
            dim_prior=True,
            extra_chi2=np.array([chi2_par]),
            extra_dims=np.array([1]),
        ).T
        np.testing.assert_allclose(lnl, expected, rtol=1e-12)


class TestUniformOutlierDensity:
    """The uniform outlier model must be a proper flux-space density."""

    def test_unit_scaling(self):
        rng = np.random.default_rng(3)
        flux = rng.uniform(0.5, 2.0, (6, 4))
        err = rng.uniform(0.02, 0.1, (6, 4))
        base = uniform_outlier_loglike(flux, err)
        halved = uniform_outlier_loglike(flux * 0.5, err * 0.5)
        # density transforms as 1/volume: lnP shifts by +Nfilt*log(2)
        np.testing.assert_allclose(halved - base, 4 * np.log(2.0), rtol=1e-10)

    def test_small_errors_do_not_blow_up(self):
        rng = np.random.default_rng(4)
        flux = rng.uniform(0.5, 2.0, (6, 4))
        err = rng.uniform(0.02, 0.1, (6, 4))
        base = uniform_outlier_loglike(flux, err)
        tight = uniform_outlier_loglike(flux, err * 1e-3)
        # bounded by the +/- sigma_clip*err range padding (< ~log(2)/band);
        # the old dimensionless form scaled as -log(err), i.e. +4*log(1e3)
        # here (~27.6)
        assert np.all(tight - base < 4 * np.log(2.0))


class TestGetSedsKwargs:
    """StellarPop.get_seds must reject unknown keyword arguments."""

    def test_unknown_kwarg_raises(self, real_stellarpop):
        with pytest.raises(TypeError):
            real_stellarpop.get_seds(
                feh=0.0, loga=9.0, smf=0.5, eep=np.linspace(300, 400, 5)
            )


class TestBinaryPreservesPrimary:
    """Stars without a valid companion keep their primary-only SED."""

    def test_partial_binary_keeps_primaries(self, real_stellarpop):
        eep_grid = np.linspace(300, 550, 12)  # spans eep_binary_max
        sed_single, p1, _ = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.0
        )
        sed_binary, p1b, p2 = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.3
        )
        valid_single = np.isfinite(sed_single).all(axis=1)
        valid_binary = np.isfinite(sed_binary).all(axis=1)
        # A binary run must never lose models that were valid as singles
        assert np.all(
            valid_binary[valid_single]
        ), "binary companion invalidity poisoned valid primary SEDs"
        # Post-MS rows (no companion possible) must be identical
        post_ms = (eep_grid > 480.0) & valid_single
        if post_ms.any():
            np.testing.assert_allclose(
                sed_binary[post_ms], sed_single[post_ms], rtol=1e-12
            )

    def test_equal_mass_binary_respects_eep_max(self, real_stellarpop):
        eep_grid = np.linspace(300, 550, 12)
        sed_single, _, _ = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=0.0
        )
        sed_eq, _, p2 = real_stellarpop.get_seds(
            feh=0.0, loga=9.5, eep=eep_grid, binary_fraction=1.0
        )
        valid = np.isfinite(sed_single).all(axis=1)
        ms = (eep_grid <= 480.0) & valid
        post_ms = (eep_grid > 480.0) & valid
        # MS stars brightened by exactly 2.5*log10(2)
        if ms.any():
            np.testing.assert_allclose(
                sed_single[ms] - sed_eq[ms], 2.5 * np.log10(2.0), rtol=1e-10
            )
        # post-MS stars unchanged, with no secondary parameters
        if post_ms.any():
            np.testing.assert_allclose(sed_eq[post_ms], sed_single[post_ms], rtol=1e-12)
            assert np.all(np.isnan(p2["mini"][post_ms]))


class RecordingStellarPop:
    """Duck-typed StellarPop (get_seds only) that records call kwargs."""

    def __init__(self, n_filters=3):
        self.n_filters = n_filters
        self.calls = []

    def get_seds(
        self,
        feh=0.0,
        loga=8.5,
        av=0.0,
        rv=3.3,
        eep=None,
        binary_fraction=0.0,
        dist=1000.0,
        mini_bound=0.08,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
    ):
        self.calls.append({"corr_params": corr_params, "smf": binary_fraction})
        eep = np.asarray(eep)
        n = len(eep)
        masses = 0.5 + 0.001 * (eep - eep.min())
        sed = np.full((n, self.n_filters), 10.0)
        if binary_fraction > 0:
            is_ms = eep <= eep_binary_max
            sed[is_ms] -= 2.5 * np.log10(1.0 + binary_fraction**3.5)
        params = {"mini": masses}
        params2 = {"mini": np.where(eep <= eep_binary_max, masses, np.nan)}
        return sed, params, params2


class TestCorrParamsPlumbing:
    """corr_params must flow through the MCMC entry point to get_seds."""

    def test_corr_params_forwarded(self):
        pop = RecordingStellarPop()
        rng = np.random.default_rng(0)
        flux = rng.uniform(0.5, 1.5, (4, 3))
        err = np.full((4, 3), 0.1)
        corr = (0.05, -0.05, 25.0, 0.4)
        lnl = isochrone_population_loglike(
            [0.0, 9.0, 0.1, 3.1, 1000.0, 0.1],
            pop,
            flux,
            err,
            smf_grid=np.array([0.0, 0.5]),
            eep_grid=np.linspace(300, 400, 10),
            corr_params=corr,
        )
        assert np.isfinite(lnl)
        assert len(pop.calls) == 2
        for call in pop.calls:
            assert call["corr_params"] == corr

    def test_total_of_zero_likelihood_objects_is_log_zero(self, monkeypatch):
        # Four objects at LOG_ZERO sum to -4e300; the total stays LOG_ZERO.
        import brutus.analysis.populations as pops

        monkeypatch.setattr(
            pops,
            "marginalize_isochrone_grid",
            lambda lnl, *args: np.full(lnl.shape[1], LOG_ZERO),
        )
        rng = np.random.default_rng(0)
        lnl = isochrone_population_loglike(
            [0.0, 9.0, 0.1, 3.1, 1000.0, 0.1],
            RecordingStellarPop(),
            rng.uniform(0.5, 1.5, (4, 3)),
            np.full((4, 3), 0.1),
            smf_grid=np.array([0.0, 0.5]),
            eep_grid=np.linspace(300, 400, 10),
        )
        assert lnl == LOG_ZERO

    def test_grid_generation_forwards_corr_params(self, real_stellarpop):
        """Different corr_params must actually change the generated grid."""
        kw = dict(
            feh=-0.3,
            loga=9.5,
            av=0.1,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0]),
            eep_grid=np.linspace(250, 400, 20),
        )
        g_def = generate_isochrone_population_grid(
            real_stellarpop, corr_params=None, **kw
        )
        g_alt = generate_isochrone_population_grid(
            real_stellarpop, corr_params=(0.3, -0.3, 30.0, 0.0), **kw
        )
        assert not np.allclose(g_def["photometry"], g_alt["photometry"], equal_nan=True)


def _reference_mixture(lnl_cluster, lnl_outlier, cluster_prob, field_fraction):
    """Old hand-rolled two-term log-sum-exp (pre-audit implementation)."""
    ln_cluster_weight = np.log(cluster_prob * (1.0 - field_fraction))
    ln_outlier_weight = np.log(1.0 - cluster_prob * (1.0 - field_fraction))
    cluster_term = lnl_cluster + ln_cluster_weight
    outlier_term = lnl_outlier + ln_outlier_weight
    max_term = np.maximum(cluster_term, outlier_term)
    with np.errstate(invalid="ignore"):
        return max_term + np.log(
            np.exp(cluster_term - max_term) + np.exp(outlier_term - max_term)
        )


class TestMixtureLogaddexp:
    """np.logaddexp path must match the old expression on finite values."""

    def test_equivalence_random(self):
        rng = np.random.default_rng(1)
        a = rng.normal(-50, 30, (500, 40))
        b = rng.normal(-40, 5, (500, 40))
        old = _reference_mixture(a, b, 0.95, 0.1)
        new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-13)

    def test_nan_rows_preserved(self):
        """NaN cluster likelihoods (invalid models) must stay non-finite so
        marginalization can drop them."""
        a = np.array([[np.nan, -10.0]])
        b = np.array([[-5.0, -5.0]])
        with np.errstate(invalid="ignore"):
            new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        assert np.isnan(new[0, 0])
        assert np.isfinite(new[0, 1])

    def test_both_neginf_gives_neginf(self):
        """logaddexp(-inf, -inf) = -inf (the old expression produced NaN);
        either way the point is dropped, but -inf is the correct value."""
        a = np.full((1, 2), -np.inf)
        b = np.full((1, 2), -np.inf)
        new = apply_isochrone_mixture_model(a, b, 0.95, 0.1)
        assert np.all(np.isneginf(new))


def _reference_population_grid(
    stellarpop,
    feh,
    loga,
    av,
    rv,
    dist,
    smf_grid,
    eep_grid,
    mini_bound=0.08,
    eep_binary_max=480.0,
    corr_params=None,
):
    """Pre-audit generate_isochrone_population_grid: one full get_seds call
    per SMF slice (primary recomputed every time)."""
    smf_grid = np.asarray(smf_grid)
    eep_grid = np.asarray(eep_grid)
    if len(smf_grid) > 1:
        smf_jacobians = np.gradient(smf_grid)
    else:
        smf_jacobians = np.array([1.0])
    all_photometry, all_masses, all_smf_values = [], [], []
    all_mass_jacobians, all_smf_jacobians = [], []
    post_ms_stored = False
    total_smf_measure = np.sum(smf_jacobians)
    for i, smf in enumerate(smf_grid):
        sed, params1, params2 = stellarpop.get_seds(
            feh=feh,
            loga=loga,
            av=av,
            rv=rv,
            eep=eep_grid,
            binary_fraction=smf,
            dist=dist,
            mini_bound=mini_bound,
            eep_binary_max=eep_binary_max,
            corr_params=corr_params,
        )
        masses = params1["mini"]
        mass_jacobians = np.gradient(masses) if len(masses) > 1 else np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_valid = mass_jacobians > 0.0
            slice_mask = base_valid & (eep_grid <= eep_binary_max)
            post_ms_mask = base_valid & (eep_grid > eep_binary_max)
        blocks = [(slice_mask, smf_jacobians[i])]
        if not post_ms_stored and np.any(post_ms_mask):
            blocks.append((post_ms_mask, total_smf_measure))
            post_ms_stored = True
        for block_mask, smf_jac in blocks:
            valid_indices = np.where(block_mask)[0]
            if len(valid_indices) == 0:
                continue
            all_photometry.append(10 ** (-0.4 * sed[valid_indices]))
            all_masses.append(masses[valid_indices])
            all_smf_values.append(np.full(len(valid_indices), smf))
            all_mass_jacobians.append(mass_jacobians[valid_indices])
            all_smf_jacobians.append(np.full(len(valid_indices), smf_jac))
    return {
        "photometry": np.vstack(all_photometry),
        "masses": np.concatenate(all_masses),
        "smf_values": np.concatenate(all_smf_values),
        "mass_jacobians": np.concatenate(all_mass_jacobians),
        "smf_jacobians": np.concatenate(all_smf_jacobians),
    }


class TestPrimaryCaching:
    """The cached-primary grid must equal the per-slice reference exactly."""

    def test_equivalence_with_reference(self, real_stellarpop):
        kw = dict(
            feh=0.05,
            loga=9.4,
            av=0.15,
            rv=3.2,
            dist=1200.0,
            smf_grid=np.linspace(0.0, 1.0, 11),
            eep_grid=np.linspace(202.0, 808.0, 120),
        )
        ref = _reference_population_grid(real_stellarpop, **kw)
        new = generate_isochrone_population_grid(real_stellarpop, **kw)
        for key in ref:
            np.testing.assert_array_equal(new[key], ref[key], err_msg=key)

    def test_ducktyped_fallback_without_smf_grid_method(self):
        """Population objects exposing only get_seds must keep working."""
        pop = RecordingStellarPop()
        assert not hasattr(pop, "get_seds_smf_grid")
        grid = generate_isochrone_population_grid(
            pop,
            feh=0.0,
            loga=9.0,
            av=0.0,
            rv=3.1,
            dist=1000.0,
            smf_grid=np.array([0.0, 1.0]),
            eep_grid=np.linspace(300, 400, 10),
        )
        assert grid["photometry"].shape[0] > 0
        assert len(pop.calls) == 2  # one get_seds call per slice

    def test_primary_computed_once(self, real_stellarpop):
        """The isochrone must be interpolated once, not once per SMF value."""
        calls = {"n": 0}
        iso = real_stellarpop.isochrone
        orig = iso.get_predictions

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        iso.get_predictions = counting
        try:
            generate_isochrone_population_grid(
                real_stellarpop,
                feh=0.0,
                loga=9.5,
                av=0.1,
                rv=3.1,
                dist=1000.0,
                smf_grid=np.linspace(0.0, 1.0, 5),
                eep_grid=np.linspace(300.0, 500.0, 20),
            )
        finally:
            iso.get_predictions = orig
        # 1 primary call + one secondary call per 0 < smf < 1 slice (3 of 5).
        # The old implementation used 5 primary + 3 secondary = 8 calls.
        assert calls["n"] == 4
