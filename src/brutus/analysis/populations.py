#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Isochrone-based stellar population analysis with mixture-before-marginalization.

This module provides functions for Bayesian inference of coeval stellar population
parameters using isochrone fitting. The implementation uses the mathematically
correct approach of applying mixture models before marginalization over stellar
parameters (mass, secondary mass fraction).

The key innovation is the mixture-before-marginalization approach, which properly
accounts for field contamination by applying the mixture model at each grid point
before integrating over stellar parameters. This differs from traditional approaches
that marginalize first and mix later, which can produce biased results.

Functions
---------
isochrone_population_loglike : Main likelihood function
    Compute log-likelihood for coeval stellar population
generate_isochrone_population_grid : Grid generation
    Generate (mass, SMF) grid for population modeling
compute_isochrone_cluster_loglike : Cluster likelihood
    Compute membership likelihood for each grid point
compute_isochrone_outlier_loglike : Outlier likelihood
    Compute field contamination likelihood
apply_isochrone_mixture_model : Mixture model
    Apply mixture before marginalization
marginalize_isochrone_grid : Marginalization
    Integrate over stellar parameters with geometric jacobians

See Also
--------
brutus.core.populations.StellarPop : Stellar population synthesis
brutus.utils.photometry : Photometric likelihood functions
brutus.priors : Prior probability distributions

Notes
-----
The workflow follows these steps:

1. Generate isochrone grid over (mass, SMF) parameter space
2. Compute cluster likelihood for each (grid_point, object) pair
3. Compute outlier likelihood for each (grid_point, object) pair
4. Apply mixture model: P(data|mass,SMF) = w_c * P_c + w_o * P_o
5. Marginalize over (mass, SMF) with proper geometric jacobians
6. Sum log-likelihoods over all objects

This approach is designed for use with external MCMC or optimization codes
(e.g., emcee, dynesty, scipy.optimize) that vary the population parameters
[Fe/H], log(age), A_V, R_V, distance.

Examples
--------
Basic usage with emcee:

>>> from brutus.core.populations import StellarPop, Isochrone
>>> from brutus.analysis.populations import isochrone_population_loglike
>>>
>>> # Initialize stellar population model
>>> iso = Isochrone()
>>> pop = StellarPop(isochrone=iso)
>>>
>>> # Define log-likelihood function for MCMC
>>> def lnprob(theta):
...     return isochrone_population_loglike(
...         theta, pop, obs_flux, obs_err,
...         parallax=parallax, parallax_err=parallax_err
...     )
>>>
>>> # Run MCMC
>>> import emcee
>>> sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
>>> sampler.run_mcmc(initial_pos, nsteps)
"""

from __future__ import division, print_function

import warnings

import numpy as np
from scipy.special import logsumexp

# Import photometry utilities
from ..utils.photometry import (
    chisquare_outlier_loglike,
    phot_loglike,
    uniform_outlier_loglike,
)

__all__ = [
    "isochrone_population_loglike",
    "generate_isochrone_population_grid",
    "compute_isochrone_cluster_loglike",
    "compute_isochrone_outlier_loglike",
    "apply_isochrone_mixture_model",
    "marginalize_isochrone_grid",
]


def generate_isochrone_population_grid(
    stellarpop,
    feh,
    loga,
    av,
    rv,
    dist,
    smf_grid=None,
    eep_grid=None,
    mini_bound=0.08,
    eep_binary_max=480.0,
    corr_params=None,
):
    r"""
    Generate isochrone population grid over (mass, SMF) parameter space.

    Parameters
    ----------
    stellarpop : StellarPop object
        StellarPop model from core.populations module with get_seds method
    feh, loga, av, rv, dist : float
        Stellar population parameters (metallicity, log age, extinction, distance)
    smf_grid : array-like, optional
        Secondary mass fraction grid. Default is adaptive grid from 0.0 to 1.0
    eep_grid : array-like, optional
        EEP grid for isochrone evaluation. Default is 2000 points from 202 to 808
    mini_bound : float, optional
        Minimum initial mass for evaluation. Default 0.08 solar masses
    eep_binary_max : float, optional
        Maximum EEP for binary modeling. Default 480.0
    corr_params : array-like, optional
        Empirical correction parameters [dtdm, drdm, msto_smooth, feh_scale]

    Returns
    -------
    grid : dict
        Dictionary containing:
        - 'photometry': array, shape (N_total_points, N_filters) - model photometry
        - 'masses': array, shape (N_total_points,) - stellar masses
        - 'smf_values': array, shape (N_total_points,) - SMF values for each point
        - 'mass_jacobians': array, shape (N_total_points,) - mass grid spacing
        - 'smf_jacobians': array, shape (N_total_points,) - SMF grid spacing
        - 'grid_info': dict with SMF grid structure information

    See Also
    --------
    compute_isochrone_cluster_loglike : Use this grid for likelihood computation
    StellarPop.get_seds : Underlying SED generation

    Notes
    -----
    The grid is constructed by:

    1. Looping over SMF values (binary mass ratios)
    2. For each SMF, computing isochrone along EEP dimension
    3. Extracting masses from the isochrone
    4. Computing geometric jacobians (grid spacings) for proper integration
    5. Filtering invalid models (NaN photometry from impossible binaries)

    The jacobians are critical for proper marginalization - they represent
    the geometric factors :math:`dm` and :math:`d({\\rm SMF})` in the integral:

    .. math::
        P({\\rm data}) = \\int \\int P({\\rm data}|m, {\\rm SMF}) \\, dm \\, d({\\rm SMF})

    Binary models are only computed for EEP ≤ eep_binary_max (typically
    main sequence) to avoid unphysical binary configurations.
    """
    # Set default grids
    if smf_grid is None:
        smf_grid = np.array(
            [
                0.0,
                0.2,
                0.35,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        )
    if eep_grid is None:
        eep_grid = np.linspace(202.0, 808.0, 2000)

    smf_grid = np.asarray(smf_grid)
    eep_grid = np.asarray(eep_grid)

    # Compute SMF jacobians (grid spacing)
    if len(smf_grid) > 1:
        smf_jacobians = np.gradient(smf_grid)
    else:
        smf_jacobians = np.array([1.0])

    # Storage for combined grid
    all_photometry = []
    all_masses = []
    all_smf_values = []
    all_mass_jacobians = []
    all_smf_jacobians = []

    # Track which models have been computed (for binary masking)
    identical_models_computed = False

    # Loop over SMF grid
    for i, smf in enumerate(smf_grid):

        # Generate isochrone for this SMF
        try:
            sed, params1, params2 = stellarpop.get_seds(
                feh=feh,
                loga=loga,
                av=av,
                rv=rv,
                eep=eep_grid,
                smf=smf,
                dist=dist,
                mini_bound=mini_bound,
                eep_binary_max=eep_binary_max,
                corr_params=corr_params,
            )
        except Exception as e:
            warnings.warn(f"Failed to generate isochrone for SMF={smf}: {e}")
            continue

        # Extract mass grid and compute jacobians
        masses = params1["mini"]
        if len(masses) > 1:
            mass_jacobians = np.gradient(masses)
        else:
            mass_jacobians = np.array([1.0])

        # Create mask for valid models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if identical_models_computed:
                # Mask out repeated single-star models for SMF > 0
                # Don't check for finite SED - let likelihood handle NaNs
                valid_mask = (mass_jacobians > 0.0) & (eep_grid <= eep_binary_max)
            else:
                # First time - only require positive mass jacobian
                # Don't check for finite SED - let likelihood handle NaNs
                valid_mask = mass_jacobians > 0.0
                identical_models_computed = True

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # Store valid models
            sed_valid = sed[valid_indices]
            masses_valid = masses[valid_indices]
            mass_jacobians_valid = mass_jacobians[valid_indices]

            # Convert magnitudes to fluxes
            photometry_valid = 10 ** (-0.4 * sed_valid)

            # Store in combined arrays
            all_photometry.append(photometry_valid)
            all_masses.append(masses_valid)
            all_smf_values.append(np.full(len(masses_valid), smf))
            all_mass_jacobians.append(mass_jacobians_valid)
            all_smf_jacobians.append(np.full(len(masses_valid), smf_jacobians[i]))

    # Combine all arrays
    if len(all_photometry) == 0:
        raise ValueError("No valid isochrone models generated")

    combined_photometry = np.vstack(all_photometry)
    combined_masses = np.concatenate(all_masses)
    combined_smf_values = np.concatenate(all_smf_values)
    combined_mass_jacobians = np.concatenate(all_mass_jacobians)
    combined_smf_jacobians = np.concatenate(all_smf_jacobians)

    return {
        "photometry": combined_photometry,
        "masses": combined_masses,
        "smf_values": combined_smf_values,
        "mass_jacobians": combined_mass_jacobians,
        "smf_jacobians": combined_smf_jacobians,
        "grid_info": {
            "smf_grid": smf_grid,
            "eep_grid": eep_grid,
            "n_total_points": len(combined_masses),
        },
    }


def compute_isochrone_cluster_loglike(
    obs_flux,
    obs_err,
    isochrone_grid,
    parallax=None,
    parallax_err=None,
    distance=None,
    dim_prior=True,
    mask=None,
):
    r"""
    Compute cluster membership likelihood using existing photometry infrastructure.

    Parameters
    ----------
    obs_flux : array-like, shape (N_objects, N_filters)
        Observed flux densities
    obs_err : array-like, shape (N_objects, N_filters)
        Flux errors
    isochrone_grid : dict
        Isochrone grid from generate_isochrone_population_grid()
    parallax : array-like, shape (N_objects,), optional
        Parallax measurements (mas)
    parallax_err : array-like, shape (N_objects,), optional
        Parallax errors (mas)
    distance : float, optional
        Population distance (pc). Required if parallax provided
    dim_prior : bool, optional
        Whether to use chi-square (True) or normal (False) likelihood
    mask : array-like, shape (N_objects, N_filters), optional
        Data mask (1=use, 0=skip)

    Returns
    -------
    lnl_cluster : array-like, shape (N_grid_points, N_objects)
        Cluster membership log-likelihood for each grid point and object.
        Invalid models (NaN photometry) are assigned NaN likelihood.

    See Also
    --------
    brutus.utils.photometry.phot_loglike : Photometric likelihood function
    compute_isochrone_outlier_loglike : Complementary outlier likelihood
    generate_isochrone_population_grid : Creates the isochrone_grid input

    Notes
    -----
    The likelihood includes both photometric and parallax components:

    .. math::
        \\ln L_{\\rm cluster} = \\ln L_{\\rm phot} + \\ln L_{\\rm parallax}

    where the photometric likelihood uses either chi-square (dim_prior=True)
    or Gaussian (dim_prior=False) formulation.

    Invalid models with NaN photometry are preserved as NaN in the output
    to be properly handled during marginalization (they contribute zero
    probability via logsumexp).
    """
    obs_flux = np.asarray(obs_flux)
    obs_err = np.asarray(obs_err)
    n_objects, n_filters = obs_flux.shape

    model_photometry = isochrone_grid["photometry"]  # shape (N_grid_points, N_filters)
    n_grid_points = model_photometry.shape[0]

    # Check for invalid models (NaN photometry from impossible binary configs)
    model_valid_mask = np.all(
        np.isfinite(model_photometry), axis=1
    )  # shape (N_grid_points,)

    # Replace NaN models with finite values for phot_loglike (will set to NaN after)
    model_photometry_clean = np.where(
        np.isfinite(model_photometry),
        model_photometry,
        0.0,  # Temporary replacement, will be masked out
    )

    # Reshape models for phot_loglike: (N_objects, N_grid_points, N_filters)
    model_photometry_reshaped = np.broadcast_to(
        model_photometry_clean[None, :, :], (n_objects, n_grid_points, n_filters)
    )

    # Compute photometric likelihood using existing infrastructure
    lnl_phot = phot_loglike(
        obs_flux, obs_err, model_photometry_reshaped, mask=mask, dim_prior=dim_prior
    )  # shape (N_objects, N_grid_points)

    # Transpose to get correct orientation for masking
    lnl_phot = lnl_phot.T  # Now shape (N_grid_points, N_objects)

    # For invalid models, set likelihood to NaN (will be handled in marginalization)
    lnl_phot[~model_valid_mask, :] = np.nan

    # Add parallax contribution if provided
    lnl_parallax = 0.0
    if parallax is not None and parallax_err is not None and distance is not None:
        parallax = np.asarray(parallax)
        parallax_err = np.asarray(parallax_err)

        # Parallax prediction from distance
        parallax_pred = 1000.0 / distance  # mas

        # Parallax mask
        parallax_mask = (
            np.isfinite(parallax) & np.isfinite(parallax_err) & (parallax_err > 0)
        )

        if np.any(parallax_mask):
            # Parallax chi-square contribution
            chi2_parallax = (parallax - parallax_pred) ** 2 / parallax_err**2
            lnl_parallax = np.where(
                parallax_mask,
                -0.5 * (chi2_parallax + np.log(2 * np.pi * parallax_err**2)),
                0.0,
            )
            # Broadcast to grid shape (now lnl_phot is already transposed)
            lnl_parallax = lnl_parallax[None, :]  # shape (1, N_objects)

    # Combine photometric and parallax likelihoods
    lnl_cluster = lnl_phot + lnl_parallax  # shape (N_grid_points, N_objects)

    # Already in standard grid-first ordering
    return lnl_cluster  # shape (N_grid_points, N_objects)


def compute_isochrone_outlier_loglike(
    obs_flux,
    obs_err,
    isochrone_grid=None,
    parallax=None,
    parallax_err=None,
    dim_prior=True,
    outlier_model_func=None,
    **outlier_kwargs,
):
    """
    Compute outlier likelihood with stellar-parameter-aware interface.

    Parameters
    ----------
    obs_flux : array-like, shape (N_objects, N_filters)
        Observed flux densities
    obs_err : array-like, shape (N_objects, N_filters)
        Flux errors
    isochrone_grid : dict, optional
        Isochrone grid containing stellar parameters for potential dependence
    parallax : array-like, shape (N_objects,), optional
        Parallax measurements (mas)
    parallax_err : array-like, shape (N_objects,), optional
        Parallax errors (mas)
    dim_prior : bool, optional
        Use chi-square (True) or uniform (False) outlier model
    outlier_model_func : callable, optional
        Custom outlier model function
    **outlier_kwargs : dict
        Additional arguments for outlier model

    Returns
    -------
    lnl_outlier : array-like, shape (N_grid_points, N_objects)
        Outlier likelihood for each grid point and object
    """
    obs_flux = np.asarray(obs_flux)
    obs_err = np.asarray(obs_err)
    n_objects = obs_flux.shape[0]

    # Extract stellar parameters for potential use
    stellar_params = None
    if isochrone_grid is not None:
        stellar_params = {
            "masses": isochrone_grid["masses"],
            "smf_values": isochrone_grid["smf_values"],
        }

    # Compute outlier likelihood
    if outlier_model_func is not None:
        # Custom outlier model
        lnl_outlier = outlier_model_func(
            obs_flux,
            obs_err,
            stellar_params=stellar_params,
            parallax=parallax,
            parallax_err=parallax_err,
            **outlier_kwargs,
        )
    elif dim_prior:
        # Default chi-square outlier model
        lnl_outlier = chisquare_outlier_loglike(
            obs_flux,
            obs_err,
            stellar_params=stellar_params,
            parallax=parallax,
            parallax_err=parallax_err,
            **outlier_kwargs,
        )
    else:
        # Default uniform outlier model
        lnl_outlier = uniform_outlier_loglike(
            obs_flux,
            obs_err,
            stellar_params=stellar_params,
            parallax=parallax,
            parallax_err=parallax_err,
            **outlier_kwargs,
        )

    # Handle broadcasting to grid shape
    lnl_outlier = np.asarray(lnl_outlier)

    if isochrone_grid is not None:
        n_grid_points = len(isochrone_grid["masses"])

        if lnl_outlier.shape == (n_objects,):
            # Stellar-independent: broadcast over grid
            lnl_outlier = np.broadcast_to(
                lnl_outlier[None, :], (n_grid_points, n_objects)
            )
        elif lnl_outlier.shape == (n_grid_points, n_objects):
            # Stellar-dependent: already correct shape
            pass
        else:
            raise ValueError(
                f"Outlier likelihood shape {lnl_outlier.shape} incompatible "
                f"with expected ({n_grid_points}, {n_objects}) or ({n_objects},)"
            )
    else:
        # No grid provided - assume stellar-independent
        if lnl_outlier.ndim == 1:
            lnl_outlier = lnl_outlier[None, :]  # shape (1, N_objects)

    return lnl_outlier


def apply_isochrone_mixture_model(
    lnl_cluster, lnl_outlier, cluster_prob, field_fraction
):
    r"""
    Apply mixture model at each grid point: mixture before marginalization.

    For each (grid_point, object) pair:
    P(data|mass,SMF) = P_cluster * P(data|cluster) + P_outlier * P(data|outlier)

    Parameters
    ----------
    lnl_cluster : array-like, shape (N_grid_points, N_objects)
        Cluster membership likelihoods
    lnl_outlier : array-like, shape (N_grid_points, N_objects)
        Outlier model likelihoods
    cluster_prob : float
        Prior probability of cluster membership (external)
    field_fraction : float
        Fraction of cluster stars that are field contaminants (fitted parameter)

    Returns
    -------
    lnl_mixture : array-like, shape (N_grid_points, N_objects)
        Mixed log-likelihoods for each grid point and object

    See Also
    --------
    marginalize_isochrone_grid : Next step after mixture application
    compute_isochrone_cluster_loglike : Cluster likelihood component
    compute_isochrone_outlier_loglike : Outlier likelihood component

    Notes
    -----
    The mixture model is applied as:

    .. math::
        P({\\rm data}|m, {\\rm SMF}) = w_c \\cdot P_c + w_o \\cdot P_o

    where:
    - :math:`w_c = P_{\\rm cluster} \\cdot (1 - f_{\\rm field})`
    - :math:`w_o = 1 - w_c`
    - :math:`P_{\\rm cluster}` is the prior probability (cluster_prob)
    - :math:`f_{\\rm field}` is the field contamination fraction

    This is computed in log-space using logsumexp for numerical stability:

    .. math::
        \\ln L_{\\rm mix} = \\ln(\\exp(\\ln L_c + \\ln w_c) + \\exp(\\ln L_o + \\ln w_o))

    The key distinction from traditional approaches is that this mixture
    is applied **before** marginalization over stellar parameters, which
    is mathematically correct for contaminated populations.
    """
    lnl_cluster = np.asarray(lnl_cluster)
    lnl_outlier = np.asarray(lnl_outlier)

    # Ensure compatible shapes
    if lnl_cluster.shape != lnl_outlier.shape:
        raise ValueError(
            f"Cluster and outlier likelihood shapes must match: "
            f"{lnl_cluster.shape} vs {lnl_outlier.shape}"
        )

    # Compute mixture probabilities
    # P(cluster member & not field) = cluster_prob * (1 - field_fraction)
    # P(outlier OR field) = 1 - cluster_prob * (1 - field_fraction)
    ln_cluster_weight = np.log(cluster_prob * (1.0 - field_fraction))
    ln_outlier_weight = np.log(1.0 - cluster_prob * (1.0 - field_fraction))

    # Apply mixture model using logsumexp for numerical stability
    cluster_term = lnl_cluster + ln_cluster_weight
    outlier_term = lnl_outlier + ln_outlier_weight

    # Stack for logsumexp: shape (2, N_grid_points, N_objects)
    terms = np.stack([cluster_term, outlier_term], axis=0)
    lnl_mixture = logsumexp(terms, axis=0)

    return lnl_mixture


def marginalize_isochrone_grid(lnl_mixture, mass_jacobians, smf_jacobians):
    r"""
    Marginalize mixed likelihoods over (mass, SMF) grid with geometric jacobians.

    Performs: P(data|population_params) = ∫∫ P(data|mass,SMF) dm d(SMF)

    Parameters
    ----------
    lnl_mixture : array-like, shape (N_grid_points, N_objects)
        Mixed likelihoods at each grid point
    mass_jacobians : array-like, shape (N_grid_points,)
        Mass grid spacing (geometric factors for integration)
    smf_jacobians : array-like, shape (N_grid_points,)
        SMF grid spacing (geometric factors for integration)

    Returns
    -------
    lnl_marginalized : array-like, shape (N_objects,)
        Marginalized log-likelihoods for each object

    See Also
    --------
    apply_isochrone_mixture_model : Previous step before marginalization
    generate_isochrone_population_grid : Provides the jacobians

    Notes
    -----
    Performs the integration:

    .. math::
        P({\\rm data}|\\theta) = \\int \\int P({\\rm data}|m, {\\rm SMF}, \\theta) \\, dm \\, d({\\rm SMF})

    numerically using a grid-based approach:

    .. math::
        \\ln P \\approx \\ln \\sum_{i,j} \\exp(\\ln L_{i,j}) \\cdot \\Delta m_i \\cdot \\Delta({\\rm SMF})_j

    where:
    - :math:`\\ln L_{i,j}` is the mixed likelihood at grid point (i,j)
    - :math:`\\Delta m_i` is the mass grid spacing (mass_jacobians)
    - :math:`\\Delta({\\rm SMF})_j` is the SMF grid spacing (smf_jacobians)

    Invalid models (with NaN likelihood) are converted to -∞ before the
    logsumexp operation, so they contribute zero probability.

    The jacobians represent geometric integration weights and are crucial
    for obtaining unbiased parameter estimates.
    """
    lnl_mixture = np.asarray(lnl_mixture)
    mass_jacobians = np.asarray(mass_jacobians)
    smf_jacobians = np.asarray(smf_jacobians)

    # Compute total geometric jacobian
    geometric_jacobian = mass_jacobians * smf_jacobians  # shape (N_grid_points,)
    ln_jacobian = np.log(geometric_jacobian)  # shape (N_grid_points,)

    # Add jacobian to likelihoods for proper integration
    lnl_with_jacobian = (
        lnl_mixture + ln_jacobian[:, None]
    )  # shape (N_grid_points, N_objects)

    # Convert NaN to -inf for logsumexp (invalid models contribute nothing to marginalization)
    lnl_with_jacobian = np.where(
        np.isfinite(lnl_with_jacobian), lnl_with_jacobian, -np.inf
    )

    # Marginalize over grid using logsumexp
    lnl_marginalized = logsumexp(lnl_with_jacobian, axis=0)  # shape (N_objects,)

    return lnl_marginalized


def isochrone_population_loglike(
    theta,
    stellarpop,
    obs_phot,
    obs_err,
    parallax=None,
    parallax_err=None,
    cluster_prob=0.95,
    dim_prior=True,
    outlier_model_func=None,
    smf_grid=None,
    eep_grid=None,
    mini_bound=0.08,
    eep_binary_max=480.0,
    return_components=False,
    mask=None,
    **outlier_kwargs,
):
    r"""
    Compute log-likelihood for coeval stellar population using isochrone fitting.

    Uses the mathematically correct mixture-before-marginalization approach:
    1. Generate isochrone grid over (mass, SMF)
    2. Compute cluster and outlier likelihoods at each grid point
    3. Apply mixture model at each grid point
    4. Marginalize over (mass, SMF) with proper geometric factors
    5. Sum over all objects

    Parameters
    ----------
    theta : array-like, shape (6,)
        Population parameters: [feh, loga, av, rv, dist, field_frac]
        where field_frac is the field contamination fraction (0 to 1).
    stellarpop : StellarPop object
        StellarPop model from core.populations module with get_seds method
    obs_phot : array-like, shape (N_objects, N_filters)
        Observed flux densities in units of 10**(-0.4 * mag)
    obs_err : array-like, shape (N_objects, N_filters)
        Flux density errors in same units
    parallax : array-like, shape (N_objects,), optional
        Parallax measurements (mas)
    parallax_err : array-like, shape (N_objects,), optional
        Parallax errors (mas)
    cluster_prob : float, optional
        Prior probability of cluster membership. Default 0.95
    dim_prior : bool, optional
        Use chi-square (True) or normal (False) likelihood. Default True
    outlier_model_func : callable, optional
        Custom outlier model function
    smf_grid : array-like, optional
        Secondary mass fraction grid for binary modeling
    eep_grid : array-like, optional
        EEP grid for isochrone evaluation
    mini_bound : float, optional
        Minimum initial mass for isochrone. Default 0.08
    eep_binary_max : float, optional
        Maximum EEP for binary modeling. Default 480.0
    return_components : bool, optional
        Return intermediate results for debugging. Default False
    mask : array-like, shape (N_objects, N_filters), optional
        Data validity mask
    **outlier_kwargs : dict
        Additional arguments passed to outlier model

    Returns
    -------
    lnl_total : float
        Total log-likelihood summed over all objects
    components : dict, optional
        Intermediate results (if return_components=True) containing:
        - 'lnl_total': same as primary return value
        - 'lnl_per_object': array of per-object likelihoods
        - 'isochrone_grid': the generated grid dictionary
        - 'lnl_cluster': cluster likelihood array
        - 'lnl_outlier': outlier likelihood array
        - 'lnl_mixture': mixed likelihood array

    See Also
    --------
    generate_isochrone_population_grid : Step 1 - Grid generation
    compute_isochrone_cluster_loglike : Step 2 - Cluster likelihood
    compute_isochrone_outlier_loglike : Step 3 - Outlier likelihood
    apply_isochrone_mixture_model : Step 4 - Mixture model
    marginalize_isochrone_grid : Step 5 - Marginalization
    brutus.core.populations.StellarPop : Stellar population model

    Notes
    -----
    This function implements the complete mixture-before-marginalization
    workflow:

    .. math::
        \\ln L(\\theta) = \\sum_i \\ln \\left[ \\int \\int \\left( w_c P_c(d_i|m, s, \\theta) + w_o P_o(d_i) \\right) dm \\, ds \\right]

    where:
    - :math:`\\theta = [{\\rm Fe/H}, \\log {\\rm age}, A_V, R_V, d, f_{\\rm field}]` are population parameters
    - :math:`m` is stellar mass
    - :math:`s` is secondary mass fraction (SMF)
    - :math:`w_c, w_o` are mixture weights
    - :math:`P_c, P_o` are cluster and outlier likelihoods
    - :math:`d_i` is data for object i

    The function is designed for use with MCMC samplers like emcee or
    nested sampling codes like dynesty. It handles errors gracefully by
    returning -∞ for failed computations.

    Examples
    --------
    Use with emcee for MCMC sampling:

    >>> def lnprob(theta):
    ...     if not in_prior_bounds(theta):
    ...         return -np.inf
    ...     return isochrone_population_loglike(theta, pop, flux, err)
    >>>
    >>> sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    >>> sampler.run_mcmc(p0, nsteps)

    Extract intermediate results for diagnostics:

    >>> lnl, components = isochrone_population_loglike(
    ...     theta, pop, flux, err, return_components=True
    ... )
    >>> print(f"Per-object likelihoods: {components['lnl_per_object']}")
    """
    theta = np.asarray(theta)
    if len(theta) != 6:
        raise ValueError(f"Expected 6 population parameters, got {len(theta)}")

    feh, loga, av, rv, dist, field_fraction = theta

    # Check inputs
    obs_phot = np.asarray(obs_phot)
    obs_err = np.asarray(obs_err)
    if obs_phot.shape != obs_err.shape:
        raise ValueError("Photometry and error shapes must match")

    try:
        # 1. Generate isochrone population grid
        isochrone_grid = generate_isochrone_population_grid(
            stellarpop,
            feh,
            loga,
            av,
            rv,
            dist,
            smf_grid=smf_grid,
            eep_grid=eep_grid,
            mini_bound=mini_bound,
            eep_binary_max=eep_binary_max,
        )

        # 2. Compute cluster likelihood
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_phot,
            obs_err,
            isochrone_grid,
            parallax=parallax,
            parallax_err=parallax_err,
            distance=dist,
            dim_prior=dim_prior,
            mask=mask,
        )

        # 3. Compute outlier likelihood
        lnl_outlier = compute_isochrone_outlier_loglike(
            obs_phot,
            obs_err,
            isochrone_grid,
            parallax=parallax,
            parallax_err=parallax_err,
            dim_prior=dim_prior,
            outlier_model_func=outlier_model_func,
            **outlier_kwargs,
        )

        # 4. Apply mixture model
        lnl_mixture = apply_isochrone_mixture_model(
            lnl_cluster, lnl_outlier, cluster_prob, field_fraction
        )

        # 5. Marginalize over stellar parameters
        lnl_marginalized = marginalize_isochrone_grid(
            lnl_mixture,
            isochrone_grid["mass_jacobians"],
            isochrone_grid["smf_jacobians"],
        )

        # 6. Sum over all objects
        lnl_total = np.sum(lnl_marginalized)

        if not np.isfinite(lnl_total):
            lnl_total = -np.inf

    except Exception as e:
        warnings.warn(f"Likelihood computation failed: {e}")
        lnl_total = -np.inf
        lnl_marginalized = None

    if return_components:
        components = {
            "lnl_total": lnl_total,
            "lnl_per_object": (
                lnl_marginalized
                if lnl_marginalized is not None
                else np.full(obs_phot.shape[0], -np.inf)
            ),
            "isochrone_grid": isochrone_grid if "isochrone_grid" in locals() else None,
            "lnl_cluster": lnl_cluster if "lnl_cluster" in locals() else None,
            "lnl_outlier": lnl_outlier if "lnl_outlier" in locals() else None,
            "lnl_mixture": lnl_mixture if "lnl_mixture" in locals() else None,
        }
        return lnl_total, components

    return lnl_total
