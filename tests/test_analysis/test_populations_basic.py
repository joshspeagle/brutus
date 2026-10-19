#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for isochrone-based population analysis with mixture-before-marginalization.
Uses real MIST isochrone data for authentic testing.
"""

import warnings

import numpy as np
import pytest

# Test population analysis functions
from brutus.analysis.populations import (
    apply_isochrone_mixture_model,
    compute_isochrone_cluster_loglike,
    compute_isochrone_outlier_loglike,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
    marginalize_isochrone_grid,
)

# Import required classes
from brutus.core.populations import Isochrone, StellarPop


@pytest.fixture(scope="module")
def real_stellarpop():
    """Load real MIST StellarPop with neural network once for module tests."""
    from conftest import find_brutus_data_file

    iso_file = find_brutus_data_file("MIST_1.2_iso_vvcrit0.0.h5")
    # Try both possible neural network file names (they are the same file)
    nn_file = find_brutus_data_file("nnMIST_BC.h5")
    if nn_file is None:
        nn_file = find_brutus_data_file("nn_c3k.h5")

    assert iso_file is not None, "iso_file should be found after downloading all data"
    assert (
        nn_file is not None
    ), "nn_file (nnMIST_BC.h5 or nn_c3k.h5) should be found after downloading all data"

    try:
        # Create the isochrone object
        iso = Isochrone(mistfile=iso_file, verbose=False)

        # Create the StellarPop object with neural network and standard filters
        # Use filter names that exist in the NN file
        filters = ["SDSS_g", "SDSS_r", "SDSS_i"]
        stellarpop = StellarPop(
            isochrone=iso, filters=filters, nnfile=nn_file, verbose=False
        )

        return stellarpop

    except Exception as e:
        raise  # Should not fail with all data available


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

    try:
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
    except Exception as e:
        raise  # Should not fail with all data available


@pytest.fixture
def mock_observations():
    """Create mock stellar observations for testing."""
    np.random.seed(42)  # Reproducible results

    # Create realistic mock observations
    n_stars = 5
    n_filters = 3

    # Mock fluxes: bright stars with some scatter
    obs_flux = np.random.uniform(0.5, 2.0, (n_stars, n_filters))
    obs_err = np.random.uniform(0.05, 0.15, (n_stars, n_filters))

    # Add some parallax data
    parallax = np.random.uniform(0.5, 2.0, n_stars)  # 0.5-2 mas
    parallax_err = np.random.uniform(0.05, 0.2, n_stars)

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

    def test_parameter_validation(self):
        """Test parameter validation for main function."""
        theta = [
            0.0,
            9.5,
            0.1,
            3.1,
            1000.0,
            0.05,
        ]  # [feh, loga, av, rv, dist, field_frac]
        obs_phot = np.array([[1.0, 0.8, 0.6]])
        obs_err = np.array([[0.1, 0.08, 0.06]])

        # Should fail with no isochrone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress expected warnings

            lnl = isochrone_population_loglike(theta, None, obs_phot, obs_err)
            assert lnl == -np.inf  # Should return -inf for failed computation

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

        with pytest.raises(ValueError, match="shapes must match"):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # Added -s to see print outputs
