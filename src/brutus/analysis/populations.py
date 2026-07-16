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
        Secondary mass fraction grid. Default is 21 uniform points from 0.0 to 1.0
    eep_grid : array-like, optional
        EEP grid for isochrone evaluation. Default is 1000 points from 202 to 808
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

    1. Looping over SMF values (binary mass ratios, passed to
       ``StellarPop.get_seds`` as ``binary_fraction``)
    2. For each SMF, computing isochrone along EEP dimension
    3. Extracting masses from the isochrone
    4. Computing geometric jacobians (grid spacings) for proper integration
    5. Filtering invalid models (NaN photometry from impossible binaries)

    The jacobians are critical for proper marginalization - they represent
    the geometric factors :math:`dm` and :math:`d({\\rm SMF})` in the integral:

    .. math::
        P({\\rm data}) = \\int \\int P({\\rm data}|m, {\\rm SMF}) \\, dm \\, d({\\rm SMF})

    Binary models are only computed for EEP ≤ eep_binary_max (typically
    main sequence) to avoid unphysical binary configurations. Models above
    ``eep_binary_max`` are independent of SMF, so they are stored once (from
    the first SMF slice) carrying the **full** SMF measure — the sum of all
    SMF grid spacings — so that the SMF marginalization weights them
    consistently with main-sequence models that accumulate over every slice.
    Their recorded ``smf_values`` entry is the slice they were generated from
    and is not meaningful for these SMF-independent models.
    """
    # Set default grids
    if smf_grid is None:
        smf_grid = np.linspace(0.0, 1.0, 21)
    if eep_grid is None:
        eep_grid = np.linspace(202.0, 808.0, 1000)

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

    # Track whether the SMF-independent post-MS block has been stored yet.
    # Models with eep > eep_binary_max never have a binary companion, so their
    # SEDs are identical across all SMF slices; they are stored once, carrying
    # the *full* SMF measure (see below).
    post_ms_stored = False
    total_smf_measure = np.sum(smf_jacobians)

    # Loop over SMF grid
    for i, smf in enumerate(smf_grid):

        # Generate isochrone for this SMF. `smf` is the secondary mass
        # fraction, i.e. StellarPop's `binary_fraction` (mass-ratio) argument.
        try:
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
        except Exception as e:
            warnings.warn(f"Failed to generate isochrone for SMF={smf}: {e}")
            continue

        # Extract mass grid and compute jacobians
        masses = params1["mini"]
        if len(masses) > 1:
            mass_jacobians = np.gradient(masses)
        else:
            mass_jacobians = np.array([1.0])

        # Create masks for valid models.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_valid = mass_jacobians > 0.0
            # Per-slice (binary-capable) block: eep <= eep_binary_max.
            # Don't check for finite SED - let likelihood handle NaNs.
            slice_mask = base_valid & (eep_grid <= eep_binary_max)
            # SMF-independent post-MS block (no binary companion possible).
            post_ms_mask = base_valid & (eep_grid > eep_binary_max)

        blocks = [(slice_mask, smf_jacobians[i])]
        if not post_ms_stored and np.any(post_ms_mask):
            # Store the post-MS models once with the total SMF measure.
            # P(data | m, smf) is constant in SMF for these models, so
            # integrating over the SMF axis contributes the full measure
            # (sum of all SMF grid spacings), not a single slice's spacing.
            # Storing one slice's spacing instead would underweight post-MS
            # (e.g. red-giant) models by ~len(smf_grid) relative to
            # main-sequence models, which accumulate over every slice.
            blocks.append((post_ms_mask, total_smf_measure))
            post_ms_stored = True

        for block_mask, smf_jac in blocks:
            valid_indices = np.where(block_mask)[0]
            if len(valid_indices) == 0:
                continue

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
            all_smf_jacobians.append(np.full(len(masses_valid), smf_jac))

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

    # Build the effective data mask: combine any user mask with a finiteness/
    # positivity check so that a stray NaN flux or zero error in one band
    # excludes just that band instead of poisoning the object's likelihood at
    # every grid point (which would drive the total log-likelihood to -inf
    # for every theta). The same mask is shared with the outlier component
    # by the caller so both mixture components see identical data.
    with np.errstate(invalid="ignore"):
        data_valid = np.isfinite(obs_flux) & np.isfinite(obs_err) & (obs_err > 0)
    if mask is None:
        mask = data_valid.astype(float)
    else:
        mask = np.asarray(mask) * data_valid

    # Parallax contribution. Under the chi-square dimensionality prior the
    # parallax enters the *chi-square* (with one extra degree of freedom per
    # measured parallax), matching the BruteForce convention; mixing a
    # Gaussian density into a chi-square log-pdf would combine incompatible
    # measures. Without the dimensionality prior, everything is a Gaussian
    # density and the parallax term is added as one.
    extra_chi2 = None
    extra_dims = None
    lnl_parallax = 0.0
    if parallax is not None and parallax_err is not None and distance is not None:
        parallax = np.asarray(parallax)
        parallax_err = np.asarray(parallax_err)

        # Parallax prediction from distance
        parallax_pred = 1000.0 / distance  # mas

        # Parallax mask
        with np.errstate(invalid="ignore"):
            parallax_mask = (
                np.isfinite(parallax) & np.isfinite(parallax_err) & (parallax_err > 0)
            )

        if np.any(parallax_mask):
            # Parallax chi-square contribution
            with np.errstate(invalid="ignore", divide="ignore"):
                chi2_parallax = np.where(
                    parallax_mask,
                    (parallax - parallax_pred) ** 2
                    / np.where(parallax_mask, parallax_err, 1.0) ** 2,
                    0.0,
                )
            if dim_prior:
                extra_chi2 = chi2_parallax
                extra_dims = parallax_mask.astype(int)
            else:
                lnl_parallax = np.where(
                    parallax_mask,
                    -0.5
                    * (
                        chi2_parallax
                        + np.log(
                            2 * np.pi * np.where(parallax_mask, parallax_err, 1.0) ** 2
                        )
                    ),
                    0.0,
                )[None, :]

    # Compute photometric likelihood using existing infrastructure. Passing
    # the shared 2-D model grid triggers phot_loglike's matrix-multiplication
    # fast path (no (Nobj, Ngrid, Nfilt) temporaries).
    lnl_phot = phot_loglike(
        obs_flux,
        obs_err,
        model_photometry_clean,
        mask=mask,
        dim_prior=dim_prior,
        extra_chi2=extra_chi2,
        extra_dims=extra_dims,
    )  # shape (N_objects, N_grid_points)

    # Transpose to get correct orientation for masking
    lnl_phot = lnl_phot.T  # Now shape (N_grid_points, N_objects)

    # For invalid models, set likelihood to NaN (will be handled in marginalization)
    lnl_phot[~model_valid_mask, :] = np.nan

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
    mask=None,
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
        Custom outlier model function. Must accept a ``mask`` keyword (the
        same band mask seen by the cluster likelihood) in addition to
        ``stellar_params``, ``parallax``, and ``parallax_err``.
    mask : array-like, shape (N_objects, N_filters), optional
        Data mask (1=use, 0=skip). Should be the same mask used for the
        cluster likelihood so both mixture components describe the same data.
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
            mask=mask,
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
            mask=mask,
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
            mask=mask,
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

    # Apply mixture model using numerically stable log-sum-exp
    cluster_term = lnl_cluster + ln_cluster_weight
    outlier_term = lnl_outlier + ln_outlier_weight

    # For two terms, direct numpy is faster than scipy.special.logsumexp
    max_term = np.maximum(cluster_term, outlier_term)
    lnl_mixture = max_term + np.log(
        np.exp(cluster_term - max_term) + np.exp(outlier_term - max_term)
    )

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
        \\ln P \\approx \\ln \\frac{\\sum_{i,j} \\exp(\\ln L_{i,j}) \\cdot \\Delta m_i \\cdot \\Delta({\\rm SMF})_j}{\\sum_{i,j} \\Delta m_i \\cdot \\Delta({\\rm SMF})_j}

    where:
    - :math:`\\ln L_{i,j}` is the mixed likelihood at grid point (i,j)
    - :math:`\\Delta m_i` is the mass grid spacing (mass_jacobians)
    - :math:`\\Delta({\\rm SMF})_j` is the SMF grid spacing (smf_jacobians)

    Invalid models (with NaN likelihood) are converted to -∞ before the
    logsumexp operation, so they contribute zero probability. The measure is
    normalized over the valid grid points (denominator above), making the
    flat measure a proper uniform prior p(m, SMF | θ): without this, the
    θ-dependent grid volume multiplies every mixture component — including
    the θ-independent outlier model — and biases both the population
    parameters and the field fraction.

    The jacobians represent geometric integration weights and are crucial
    for obtaining unbiased parameter estimates.
    """
    lnl_mixture = np.asarray(lnl_mixture)
    mass_jacobians = np.asarray(mass_jacobians)
    smf_jacobians = np.asarray(smf_jacobians)

    # Compute total geometric jacobian
    geometric_jacobian = mass_jacobians * smf_jacobians  # shape (N_grid_points,)
    ln_jacobian = np.log(geometric_jacobian)  # shape (N_grid_points,)

    # Normalize the integration measure over the *valid* grid points, turning
    # the flat measure into a proper (uniform) prior p(m, SMF | theta). The
    # grid volume Z(theta) = sum of jacobians changes with the population
    # parameters (isochrone mass range, invalid-model region); without this
    # normalization every component of the mixture — including the
    # theta-independent outlier model — is multiplied by Z(theta), biasing
    # the inferred population parameters and the field fraction.
    valid_rows = np.any(np.isfinite(lnl_mixture), axis=1) & np.isfinite(ln_jacobian)
    if not np.any(valid_rows):
        n_objects = lnl_mixture.shape[1]
        return np.full(n_objects, -np.inf)
    ln_measure_norm = logsumexp(ln_jacobian[valid_rows])

    # Add jacobian to likelihoods for proper integration
    lnl_with_jacobian = (
        lnl_mixture + ln_jacobian[:, None] - ln_measure_norm
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

        # Effective data mask shared by BOTH mixture components, so the
        # cluster and outlier likelihoods always describe the same bands.
        with np.errstate(invalid="ignore"):
            data_valid = np.isfinite(obs_phot) & np.isfinite(obs_err) & (obs_err > 0)
        eff_mask = data_valid.astype(float) if mask is None else mask * data_valid

        # 2. Compute cluster likelihood
        lnl_cluster = compute_isochrone_cluster_loglike(
            obs_phot,
            obs_err,
            isochrone_grid,
            parallax=parallax,
            parallax_err=parallax_err,
            distance=dist,
            dim_prior=dim_prior,
            mask=eff_mask,
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
            mask=eff_mask,
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
