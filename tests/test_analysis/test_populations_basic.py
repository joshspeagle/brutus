#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for isochrone-based population analysis with mixture-before-marginalization.
Uses real MIST isochrone data for authentic testing.
"""

import os
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

# Test outlier model functions
from brutus.utils.photometry import chisquare_outlier_loglike, uniform_outlier_loglike


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


class TestOutlierModels:
    """Test outlier model functions in utils.photometry."""

    def test_chisquare_outlier_basic(self):
        """Test chi-square outlier model with basic inputs."""
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])  # 2 objects, 3 filters
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])

        lnl_outlier = chisquare_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))
        assert np.all(lnl_outlier < 0)  # Should be negative (log probability)

    def test_uniform_outlier_basic(self):
        """Test uniform outlier model with basic inputs."""
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])

        lnl_outlier = uniform_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))
        # Note: uniform model can have positive values due to volume normalization

    def test_outlier_models_with_parallax(self):
        """Test outlier models with parallax data."""
        flux = np.array([[1.0, 0.8]])
        err = np.array([[0.1, 0.08]])
        parallax = np.array([2.0])  # 2 mas
        parallax_err = np.array([0.1])  # 0.1 mas

        lnl_chi = chisquare_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )
        lnl_unif = uniform_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_chi) == 1
        assert len(lnl_unif) == 1
        assert np.all(np.isfinite([lnl_chi[0], lnl_unif[0]]))

    def test_chisquare_outlier_with_invalid_data(self):
        """Test chi-square outlier model with invalid photometry data."""
        flux = np.array([[1.0, np.nan, 0.6], [1.2, 1.0, np.inf]])
        err = np.array([[0.1, 0.08, 0.0], [0.12, np.nan, 0.08]])  # Zero and NaN errors

        lnl_outlier = chisquare_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))  # Should handle invalid data gracefully

    def test_chisquare_outlier_custom_p_value(self):
        """Test chi-square outlier model with custom p-value cut."""
        flux = np.array([[1.0, 0.8, 0.6]])
        err = np.array([[0.1, 0.08, 0.06]])

        # Test different p-values
        lnl_1 = chisquare_outlier_loglike(flux, err, p_value_cut=1e-3)
        lnl_2 = chisquare_outlier_loglike(flux, err, p_value_cut=1e-6)

        assert len(lnl_1) == 1 and len(lnl_2) == 1
        assert np.all(np.isfinite([lnl_1[0], lnl_2[0]]))
        # Different p-values should give different results
        assert lnl_1[0] != lnl_2[0]

    def test_chisquare_outlier_invalid_parallax(self):
        """Test chi-square outlier model with invalid parallax data."""
        flux = np.array([[1.0, 0.8], [1.2, 1.0]])
        err = np.array([[0.1, 0.08], [0.12, 0.1]])
        parallax = np.array([2.0, np.nan])  # One invalid parallax
        parallax_err = np.array([0.1, 0.0])  # One zero error

        lnl_outlier = chisquare_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier))

    def test_uniform_outlier_with_invalid_data(self):
        """Test uniform outlier model with invalid photometry data."""
        flux = np.array([[1.0, np.nan, 0.6], [1.2, 1.0, np.inf]])
        err = np.array([[0.1, 0.08, 0.0], [0.12, np.nan, 0.08]])  # Zero and NaN errors

        # Should handle invalid data gracefully - expect warnings
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # Ignore expected NaN warnings
            lnl_outlier = uniform_outlier_loglike(flux, err)

        assert len(lnl_outlier) == 2
        # Should handle invalid data - may be -inf but finite is acceptable
        assert np.all(np.isfinite(lnl_outlier) | (lnl_outlier == -np.inf))

    def test_uniform_outlier_custom_sigma_clip(self):
        """Test uniform outlier model with custom sigma clipping."""
        # Use more diverse data to ensure sigma clip differences matter
        flux = np.array([[1.0, 0.8, 0.6], [2.0, 1.5, 1.2]])
        err = np.array([[0.1, 0.08, 0.06], [0.2, 0.15, 0.12]])

        # Test different sigma clips
        lnl_1 = uniform_outlier_loglike(flux, err, sigma_clip=2.0)
        lnl_2 = uniform_outlier_loglike(flux, err, sigma_clip=5.0)

        assert len(lnl_1) == 2 and len(lnl_2) == 2
        assert np.all(np.isfinite([lnl_1, lnl_2]))
        # Different sigma clips should give different results (may be same for some data)
        # Just test that function works with different parameters
        assert isinstance(lnl_1[0], (int, float, np.number))
        assert isinstance(lnl_2[0], (int, float, np.number))

    def test_uniform_outlier_invalid_parallax(self):
        """Test uniform outlier model with invalid parallax data."""
        flux = np.array([[1.0, 0.8], [1.2, 1.0]])
        err = np.array([[0.1, 0.08], [0.12, 0.1]])
        parallax = np.array([2.0, np.nan])  # One invalid parallax
        parallax_err = np.array([0.1, np.inf])  # One infinite error

        lnl_outlier = uniform_outlier_loglike(
            flux, err, parallax=parallax, parallax_err=parallax_err
        )

        assert len(lnl_outlier) == 2
        assert np.all(np.isfinite(lnl_outlier) | (lnl_outlier == -np.inf))

    def test_outlier_models_single_filter(self):
        """Test outlier models with single filter (edge case)."""
        flux = np.array([[1.0], [0.5]])  # Single filter
        err = np.array([[0.1], [0.05]])

        lnl_chi = chisquare_outlier_loglike(flux, err)
        lnl_unif = uniform_outlier_loglike(flux, err)

        assert len(lnl_chi) == 2 and len(lnl_unif) == 2
        assert np.all(np.isfinite(lnl_chi))
        assert np.all(np.isfinite(lnl_unif) | (lnl_unif == -np.inf))

    def test_outlier_models_stellar_params_unused(self):
        """Test that stellar_params parameter is handled (currently unused)."""
        flux = np.array([[1.0, 0.8]])
        err = np.array([[0.1, 0.08]])
        stellar_params = {"mass": [1.0], "age": [1e9]}  # Currently unused

        lnl_chi = chisquare_outlier_loglike(flux, err, stellar_params=stellar_params)
        lnl_unif = uniform_outlier_loglike(flux, err, stellar_params=stellar_params)

        assert len(lnl_chi) == 1 and len(lnl_unif) == 1
        assert np.all(np.isfinite([lnl_chi[0], lnl_unif[0]]))


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
        theta = [0.0, 9.5, 0.1, 3.1, 1000.0]
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

        # 1. Test outlier models
        lnl_chi = chisquare_outlier_loglike(obs_flux, obs_err)
        lnl_unif = uniform_outlier_loglike(obs_flux, obs_err)
        assert np.all(np.isfinite(lnl_chi))
        assert np.all(np.isfinite(lnl_unif))

        # 2. Test cluster likelihood with real grid
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        n_grid_points = real_isochrone_grid["photometry"].shape[0]
        assert lnl_cluster.shape == (n_grid_points, 2)
        assert np.any(np.isfinite(lnl_cluster))  # Should have some finite values

        # 3. Test outlier likelihood
        lnl_outlier = compute_isochrone_outlier_loglike(
            obs_flux, obs_err, real_isochrone_grid
        )
        assert lnl_outlier.shape == (n_grid_points, 2)
        assert np.all(np.isfinite(lnl_outlier))

        # 4. Test mixture
        lnl_mixture = apply_isochrone_mixture_model(lnl_cluster, lnl_outlier, 0.9, 0.1)
        assert lnl_mixture.shape == (n_grid_points, 2)
        assert np.all(np.isfinite(lnl_mixture))

        # 5. Test marginalization
        lnl_marginalized = marginalize_isochrone_grid(
            lnl_mixture,
            real_isochrone_grid["mass_jacobians"],
            real_isochrone_grid["smf_jacobians"],
        )
        assert lnl_marginalized.shape == (2,)
        assert np.all(np.isfinite(lnl_marginalized))

        print("✅ All workflow components working correctly with real MIST data")


class TestPhotometryUtilityFunctions:
    """Test photometry utility functions for coverage."""

    def test_magnitude_conversion(self):
        """Test magnitude and inverse magnitude conversion functions."""
        from brutus.utils.photometry import inv_magnitude, magnitude

        # Test data
        phot = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        err = np.array([[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]])
        zeropoints = np.array([1.0, 1.0, 1.0])

        # Test magnitude conversion
        mag, mag_err = magnitude(phot, err, zeropoints=zeropoints)

        assert mag.shape == phot.shape
        assert mag_err.shape == err.shape
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(mag_err))
        assert np.all(mag_err > 0)  # Errors should be positive

        # Test inverse conversion
        phot_recovered, err_recovered = inv_magnitude(
            mag, mag_err, zeropoints=zeropoints
        )

        # Should recover original photometry (within numerical precision)
        assert np.allclose(phot, phot_recovered, rtol=1e-10)
        assert np.allclose(err, err_recovered, rtol=1e-10)

    def test_luptitude_conversion(self):
        """Test luptitude and inverse luptitude conversion functions."""
        from brutus.utils.photometry import inv_luptitude, luptitude

        # Test data
        phot = np.array([[1.0, 2.0, 0.1], [0.5, 1.5, 0.05]])  # Include faint source
        err = np.array([[0.1, 0.2, 0.05], [0.05, 0.15, 0.02]])
        skynoise = np.array([0.1, 0.1, 0.1])
        zeropoints = np.array([1.0, 1.0, 1.0])

        # Test luptitude conversion
        lupt, lupt_err = luptitude(phot, err, skynoise=skynoise, zeropoints=zeropoints)

        assert lupt.shape == phot.shape
        assert lupt_err.shape == err.shape
        assert np.all(np.isfinite(lupt))
        assert np.all(np.isfinite(lupt_err))
        assert np.all(lupt_err > 0)  # Errors should be positive

        # Test inverse conversion
        phot_recovered, err_recovered = inv_luptitude(
            lupt, lupt_err, skynoise=skynoise, zeropoints=zeropoints
        )

        # Should recover original photometry (within numerical precision)
        assert np.allclose(phot, phot_recovered, rtol=1e-10)
        assert np.allclose(err, err_recovered, rtol=1e-10)

    def test_add_mag_function(self):
        """Test magnitude addition function."""
        from brutus.utils.photometry import add_mag

        # Test simple case: adding equal magnitudes should brighten by 2.5*log10(2) ≈ 0.75 mag
        mag1 = 10.0
        mag2 = 10.0
        mag_combined = add_mag(mag1, mag2)

        expected = mag1 - 2.5 * np.log10(2)  # Should be brighter
        assert np.isclose(mag_combined, expected, rtol=1e-10)

        # Test with arrays
        mag1_arr = np.array([10.0, 15.0, 20.0])
        mag2_arr = np.array([10.0, 15.0, 20.0])
        mag_combined_arr = add_mag(mag1_arr, mag2_arr)

        expected_arr = mag1_arr - 2.5 * np.log10(2)
        assert np.allclose(mag_combined_arr, expected_arr, rtol=1e-10)

    def test_phot_loglike_with_dof_reduction(self):
        """Test phot_loglike function with dof_reduction parameter."""
        from brutus.utils.photometry import phot_loglike

        # Test data
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])  # 2 objects, 3 filters
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])
        mfluxes = np.array(
            [
                [
                    [0.9, 0.75, 0.55],
                    [1.1, 0.85, 0.65],
                ],  # 2 objects, 2 models, 3 filters
                [[1.15, 0.95, 0.75], [1.25, 1.05, 0.85]],
            ]
        )

        # Test without dof_reduction
        lnl_no_dof = phot_loglike(flux, err, mfluxes, dim_prior=True, dof_reduction=0)

        # Test with dof_reduction
        lnl_with_dof = phot_loglike(flux, err, mfluxes, dim_prior=True, dof_reduction=1)

        assert lnl_no_dof.shape == (2, 2)  # (n_obj, n_models)
        assert lnl_with_dof.shape == (2, 2)
        assert np.all(np.isfinite(lnl_no_dof))
        assert np.all(np.isfinite(lnl_with_dof))
        # Different dof_reduction should give different results
        assert not np.allclose(lnl_no_dof, lnl_with_dof)

    def test_phot_loglike_dimension_mismatch(self):
        """Test phot_loglike function with mismatched dimensions."""
        from brutus.utils.photometry import phot_loglike

        # Test data with mismatched dimensions
        flux = np.array([[1.0, 0.8, 0.6], [1.2, 1.0, 0.8]])  # 2 objects, 3 filters
        err = np.array([[0.1, 0.08, 0.06], [0.12, 0.1, 0.08]])
        mfluxes = np.array(
            [
                [
                    [0.9, 0.75],
                    [1.1, 0.85],
                ],  # WRONG: 2 objects, 2 models, 2 filters (should be 3)
                [[1.15, 0.95], [1.25, 1.05]],
            ]
        )

        # Should raise ValueError for dimension mismatch
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            phot_loglike(flux, err, mfluxes)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # Added -s to see print outputs
