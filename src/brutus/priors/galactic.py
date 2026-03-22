#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Galactic structure priors for Bayesian stellar parameter estimation.

This module provides log-prior functions for Galactic structure modeling
including disk and halo number densities, metallicity distributions,
and age-metallicity relations. These priors encode our knowledge of the
Milky Way's stellar populations and are essential for distance estimation.

Functions
---------
logn_disk : Disk number density
    Exponential disk model
logn_halo : Halo number density
    Flattened power-law halo
logp_feh : Metallicity distribution
    Gaussian metallicity distribution
logp_age_from_feh : Age-metallicity relation
    Age distribution conditional on metallicity
logp_galactic_structure : Combined prior
    Full Galactic structure prior combining disk and halo

See Also
--------
brutus.analysis.individual.BruteForce : Uses Galactic priors for distance fitting
brutus.priors.stellar : Stellar population priors
brutus.priors.astrometric : Parallax and proper motion priors

Notes
-----
These priors provide critical constraints on stellar distances by
incorporating knowledge of Galactic structure:

- **Number density priors** (disk and halo) weight distance based on
  expected stellar distributions in the Galaxy

- **Metallicity priors** provide realistic [Fe/H] distributions for
  disk and halo populations

- **Age-metallicity relations** link stellar age and composition

The combined prior `logp_galactic_structure` integrates disk and halo
models with appropriate mixing fractions.

The priors use Galactocentric coordinates, which requires coordinate
transformations from equatorial coordinates and distances.

Examples
--------
Evaluate disk number density:

>>> import numpy as np
>>> from brutus.priors.galactic import logn_disk
>>>
>>> # Position in Galactic disk
>>> R = np.array([8.0])  # kpc from Galactic center
>>> Z = np.array([0.1])  # kpc above midplane
>>> log_density = logn_disk(R, Z)
>>> print(f"Log-density: {log_density[0]:.3f}")

Combined Galactic structure prior:

>>> from astropy.coordinates import SkyCoord
>>> from brutus.priors.galactic import logp_galactic_structure
>>>
>>> # Sky position
>>> coords = SkyCoord(ra=180, dec=30, unit='deg', frame='icrs')
>>> distances = np.array([1.0, 2.0, 5.0])  # kpc
>>>
>>> # Evaluate prior
>>> log_prior = logp_galactic_structure(
...     coords, distances, feh=0.0, disk_frac=0.9
... )
"""

import numpy as np

# Import utility functions from brutus.utils
from brutus.utils import truncnorm_logpdf
from brutus.utils.math import galactic_to_galactocentric_cyl


def _logsumexp3(a, b, c):
    """Fast logsumexp of 3 arrays. Avoids scipy overhead for this common case."""
    mx = np.maximum(np.maximum(a, b), c)
    return mx + np.log(np.exp(a - mx) + np.exp(b - mx) + np.exp(c - mx))


__all__ = [
    "logn_disk",
    "logn_halo",
    "logp_feh",
    "logp_age_from_feh",
    "logp_galactic_structure",
]


def logn_disk(R, Z, R_solar=8.2, Z_solar=0.025, R_scale=2.6, Z_scale=0.3, R_smooth=2.0):
    r"""
    Log-number density for the Galactic disk stellar population.

    Implements an exponential disk model with separate radial and vertical
    scale lengths, smoothed near the Galactic center to avoid singularities.

    Parameters
    ----------
    R : array_like
        Galactocentric cylindrical radius in kpc.
    Z : array_like
        Height above the Galactic midplane in kpc.
    R_solar : float, optional
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
    R_scale : float, optional
        Disk radial scale length in kpc. Default is 2.6.
    Z_scale : float, optional
        Disk vertical scale height in kpc. Default is 0.3.
    R_smooth : float, optional
        Smoothing radius to avoid central singularity in kpc. Default is 2.0.

    Returns
    -------
    logn : array_like
        Normalized log-number density relative to Solar neighborhood.

    See Also
    --------
    logn_halo : Halo number density
    logp_galactic_structure : Combined disk+halo model

    Notes
    -----
    The disk number density follows:

    .. math::
        n_{\\text{disk}}(R, Z) \\propto \\exp\\left(-\\frac{R_{\\text{eff}} - R_\\odot}{R_{\\text{scale}}} - \\frac{|Z| - |Z_\\odot|}{Z_{\\text{scale}}}\\right)

    where :math:`R_{\\text{eff}} = \\sqrt{R^2 + R_{\\text{smooth}}^2}` provides
    smoothing near the Galactic center.

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    R = np.asarray(R)
    Z = np.asarray(Z)

    # Smoothed effective radius
    R_eff = np.sqrt(R**2 + R_smooth**2)

    # Exponential disk components
    radial_term = (R_eff - R_solar) / R_scale
    vertical_term = (np.abs(Z) - np.abs(Z_solar)) / Z_scale

    return -(radial_term + vertical_term)


def logn_halo(
    R,
    Z,
    R_solar=8.2,
    Z_solar=0.025,
    R_smooth=2.0,
    eta=4.2,
    q_ctr=0.2,
    q_inf=0.8,
    r_q=6.0,
):
    r"""
    Log-number density for the Galactic halo stellar population.

    Implements a flattened power-law halo model with radius-dependent
    oblateness following observational constraints.

    Parameters
    ----------
    R : array_like
        Galactocentric cylindrical radius in kpc.
    Z : array_like
        Height above the Galactic midplane in kpc.
    R_solar : float, optional
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
    R_smooth : float, optional
        Smoothing radius to avoid central singularity in kpc. Default is 2.0.
    eta : float, optional
        Power-law index for halo density profile. Default is 4.2.
    q_ctr : float, optional
        Halo oblateness at Galactic center. Default is 0.2.
    q_inf : float, optional
        Halo oblateness at large radii. Default is 0.8.
    r_q : float, optional
        Scale radius for oblateness transition in kpc. Default is 6.0.

    Returns
    -------
    logn : array_like
        Normalized log-number density relative to Solar neighborhood.

    See Also
    --------
    logn_disk : Disk number density
    logp_galactic_structure : Combined disk+halo model

    Notes
    -----
    The halo follows a flattened power-law profile:

    .. math::
        n_{\\text{halo}}(R, Z) \\propto R_{\\text{eff}}^{-\\eta}

    where the effective radius includes radius-dependent flattening:

    .. math::
        R_{\\text{eff}} = \\sqrt{R^2 + (Z/q)^2 + R_{\\text{smooth}}^2}

        q(r) = q_\\infty - (q_\\infty - q_{\\text{ctr}}) e^{1 - r'/r_q}

        r' = \\sqrt{r^2 + r_q^2}, \\quad r = \\sqrt{R^2 + Z^2}

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    Bell et al. (2008) - Stellar Halo Properties from SDSS
    """
    R = np.asarray(R)
    Z = np.asarray(Z)

    # Spherical radius from Galactic center
    r = np.sqrt(R**2 + Z**2)

    # Radius-dependent oblateness
    r_prime = np.sqrt(r**2 + r_q**2)
    q = q_inf - (q_inf - q_ctr) * np.exp(1.0 - r_prime / r_q)

    # Effective radius with flattening and smoothing
    R_eff = np.sqrt(R**2 + (Z / q) ** 2 + R_smooth**2)

    # Solar normalization values
    r_solar = np.sqrt(R_solar**2 + Z_solar**2)
    r_prime_solar = np.sqrt(r_solar**2 + r_q**2)
    q_solar = q_inf - (q_inf - q_ctr) * np.exp(1.0 - r_prime_solar / r_q)
    R_eff_solar = np.sqrt(R_solar**2 + (Z_solar / q_solar) ** 2 + R_smooth**2)

    # Power-law halo profile
    logn = -eta * np.log(R_eff / R_eff_solar)

    return logn


def logp_feh(feh, feh_mean=-0.2, feh_sigma=0.3):
    r"""
    Log-prior for stellar metallicity in Galactic components.

    Implements a Gaussian metallicity distribution appropriate for
    different Galactic stellar populations (disk, thick disk, halo).

    Parameters
    ----------
    feh : array_like
        Stellar metallicity [Fe/H] in dex.
    feh_mean : float, optional
        Mean metallicity of the population in dex. Default is -0.2 (thin disk).
    feh_sigma : float, optional
        Metallicity dispersion in dex. Default is 0.3.

    Returns
    -------
    logp : array_like
        Normalized log-probability density for the input metallicities.

    Notes
    -----
    The metallicity prior follows a normal distribution:

    .. math::
        \\log p([\\text{Fe/H}]) = -\\frac{1}{2}\\left[\\frac{([\\text{Fe/H}] - \\mu_{\\text{Fe/H}})^2}{\\sigma_{\\text{Fe/H}}^2} + \\log(2\\pi\\sigma_{\\text{Fe/H}}^2)\\right]

    Typical values for different Galactic components:
    - Thin disk: feh_mean = -0.2, feh_sigma = 0.3
    - Thick disk: feh_mean = -0.7, feh_sigma = 0.4
    - Halo: feh_mean = -1.6, feh_sigma = 0.5

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    feh = np.asarray(feh)

    # Gaussian log-prior
    chi2 = (feh - feh_mean) ** 2 / feh_sigma**2
    log_norm = np.log(2.0 * np.pi * feh_sigma**2)
    logp = -0.5 * (chi2 + log_norm)

    return logp


def logp_age_from_feh(
    age,
    feh_mean=-0.2,
    max_age=13.8,
    min_age=0.0,
    feh_age_ctr=-0.5,
    feh_age_scale=0.5,
    nsigma_from_max_age=2.0,
    max_sigma=4.0,
    min_sigma=1.0,
):
    r"""
    Log-prior for stellar age based on metallicity-age relation.

    Implements the age-metallicity relation observed in the Galactic disk,
    where older stars tend to be more metal-poor. Uses truncated normal
    distribution bounded by physically reasonable ages.

    Parameters
    ----------
    age : array_like
        Stellar ages in Gyr.
    feh_mean : float, optional
        Mean metallicity of the population in dex. Default is -0.2.
    max_age : float, optional
        Maximum allowed stellar age in Gyr. Default is 13.8 (age of Universe).
    min_age : float, optional
        Minimum allowed stellar age in Gyr. Default is 0.0.
    feh_age_ctr : float, optional
        Metallicity where mean age is halfway between min/max. Default is -0.5.
    feh_age_scale : float, optional
        Scale length for metallicity-age relation in dex. Default is 0.5.
    nsigma_from_max_age : float, optional
        Number of σ the mean age is below max_age. Default is 2.0.
    max_sigma : float, optional
        Maximum age dispersion in Gyr. Default is 4.0.
    min_sigma : float, optional
        Minimum age dispersion in Gyr. Default is 1.0.

    Returns
    -------
    logp : array_like
        Normalized log-probability density for the input ages.

    Notes
    -----
    The age-metallicity relation follows a logistic function:

    .. math::
        \\langle t \\rangle = \\frac{t_{\\max} - t_{\\min}}{1 + \\exp\\left(\\frac{[\\text{Fe/H}] - c}{s}\\right)} + t_{\\min}

    where c is the central metallicity and s is the scale length.

    The age dispersion decreases for younger (more metal-rich) stars:

    .. math::
        \\sigma_t = \\min\\left(\\max\\left(\\frac{t_{\\max} - \\langle t \\rangle}{n\\sigma}, \\sigma_{\\min}\\right), \\sigma_{\\max}\\right)

    Ages are drawn from a truncated normal distribution bounded by [min_age, max_age].

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    Nordström et al. (2004) - Age-metallicity relation in Solar neighborhood
    """
    age = np.asarray(age)

    # Predicted mean age from metallicity
    age_mean_pred = (max_age - min_age) / (
        1.0 + np.exp((feh_mean - feh_age_ctr) / feh_age_scale)
    ) + min_age

    # Age dispersion (younger stars have smaller dispersion)
    age_sigma_pred = (max_age - age_mean_pred) / nsigma_from_max_age
    age_sigma_pred = np.clip(age_sigma_pred, min_sigma, max_sigma)

    # Truncated normal distribution bounds
    a = (min_age - age_mean_pred) / age_sigma_pred  # Lower bound
    b = (max_age - age_mean_pred) / age_sigma_pred  # Upper bound

    # Compute truncated normal log-probability
    logp = truncnorm_logpdf(age, a, b, loc=age_mean_pred, scale=age_sigma_pred)

    return logp


def logp_galactic_structure(
    dists,
    coord,
    labels=None,
    R_solar=8.2,
    Z_solar=0.025,
    R_thin=2.6,
    Z_thin=0.3,
    Rs_thin=2.0,
    R_thick=2.0,
    Z_thick=0.9,
    f_thick=0.04,
    Rs_thick=2.0,
    Rs_halo=2.0,
    q_halo_ctr=0.2,
    q_halo_inf=0.8,
    r_q_halo=6.0,
    eta_halo=4.2,
    f_halo=0.005,
    feh_thin=-0.2,
    feh_thin_sigma=0.3,
    feh_thick=-0.7,
    feh_thick_sigma=0.4,
    feh_halo=-1.6,
    feh_halo_sigma=0.5,
    max_age=13.8,
    min_age=0.0,
    feh_age_ctr=-0.5,
    feh_age_scale=0.5,
    nsigma_from_max_age=2.0,
    max_sigma=4.0,
    min_sigma=1.0,
    return_components=False,
):
    """
    Complete Galactic structure log-prior with thin disk, thick disk, and halo.

    Implements a sophisticated three-component Galactic model based on
    Bland-Hawthorn & Gerhard (2016). Combines spatial number density priors
    with optional metallicity and age priors for realistic stellar populations.

    Parameters
    ----------
    dists : array_like
        Distance from observer in kpc.
    coord : tuple of floats
        Galactic coordinates (l, b) in degrees.
    labels : structured array, optional
        Stellar labels containing 'feh' and/or 'loga' for metallicity/age priors.
    R_solar : float, optional
        Solar Galactocentric radius in kpc. Default is 8.2.
    Z_solar : float, optional
        Solar height above midplane in kpc. Default is 0.025.
    R_thin : float, optional
        Thin disk radial scale length in kpc. Default is 2.6.
    Z_thin : float, optional
        Thin disk vertical scale height in kpc. Default is 0.3.
    Rs_thin : float, optional
        Thin disk smoothing radius in kpc. Default is 2.0.
    R_thick : float, optional
        Thick disk radial scale length in kpc. Default is 2.0.
    Z_thick : float, optional
        Thick disk vertical scale height in kpc. Default is 0.9.
    f_thick : float, optional
        Thick disk relative normalization. Default is 0.04.
    Rs_thick : float, optional
        Thick disk smoothing radius in kpc. Default is 2.0.
    Rs_halo : float, optional
        Halo smoothing radius in kpc. Default is 2.0.
    q_halo_ctr : float, optional
        Halo central oblateness. Default is 0.2.
    q_halo_inf : float, optional
        Halo asymptotic oblateness. Default is 0.8.
    r_q_halo : float, optional
        Halo oblateness transition radius in kpc. Default is 6.0.
    eta_halo : float, optional
        Halo power-law index. Default is 4.2.
    f_halo : float, optional
        Halo relative normalization. Default is 0.005.
    feh_thin : float, optional
        Thin disk mean metallicity in dex. Default is -0.2.
    feh_thin_sigma : float, optional
        Thin disk metallicity dispersion in dex. Default is 0.3.
    feh_thick : float, optional
        Thick disk mean metallicity in dex. Default is -0.7.
    feh_thick_sigma : float, optional
        Thick disk metallicity dispersion in dex. Default is 0.4.
    feh_halo : float, optional
        Halo mean metallicity in dex. Default is -1.6.
    feh_halo_sigma : float, optional
        Halo metallicity dispersion in dex. Default is 0.5.
    max_age : float, optional
        Maximum stellar age in Gyr. Default is 13.8.
    min_age : float, optional
        Minimum stellar age in Gyr. Default is 0.0.
    feh_age_ctr : float, optional
        Central metallicity for age-metallicity relation. Default is -0.5.
    feh_age_scale : float, optional
        Scale length for age-metallicity relation. Default is 0.5.
    nsigma_from_max_age : float, optional
        Age dispersion parameter. Default is 2.0.
    max_sigma : float, optional
        Maximum age dispersion in Gyr. Default is 4.0.
    min_sigma : float, optional
        Minimum age dispersion in Gyr. Default is 1.0.
    return_components : bool, optional
        Whether to return individual component contributions. Default is False.

    Returns
    -------
    logp : array_like
        Total log-prior probability density.
    components : dict, optional
        Individual component contributions (if return_components=True).

    Notes
    -----
    The Galactic model combines three stellar populations:

    1. **Thin Disk**: Young, metal-rich stars with small scale height
    2. **Thick Disk**: Intermediate-age, metal-poor stars with larger scale height
    3. **Halo**: Old, very metal-poor stars with flattened power-law profile

    Each component has distinct spatial, metallicity, and age distributions
    calibrated from observations. The model accounts for:

    - Coordinate transformations from observer to Galactocentric frame
    - Volume correction factors (dV ∝ distance²)
    - Component membership probabilities
    - Conditional metallicity and age priors

    When stellar labels are provided, applies population-specific priors:
    - Metallicity: Gaussian distributions with different means/dispersions
    - Age: Age-metallicity relation with truncated normal distributions

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    dists = np.asarray(dists)

    # Volume correction factor (dV ∝ r² dr)
    vol_factor = 2.0 * np.log(dists + 1e-300)

    # Convert to galactocentric cylindrical coordinates
    if hasattr(coord, "galactic"):
        # coord is already a SkyCoord object
        ell = coord.galactic.l.deg
        b = coord.galactic.b.deg
    else:
        # coord is a tuple of (l, b) in degrees
        ell, b = coord[0], coord[1]

    # Convert to Galactocentric cylindrical coordinates using fast NumPy math
    R, Z = galactic_to_galactocentric_cyl(
        dists, ell, b, R_solar=R_solar, Z_solar=Z_solar
    )

    # Thin disk component
    logp_thin = logn_disk(
        R,
        Z,
        R_solar=R_solar,
        Z_solar=Z_solar,
        R_scale=R_thin,
        Z_scale=Z_thin,
        R_smooth=Rs_thin,
    )
    logp_thin += vol_factor

    # Thick disk component
    logp_thick = logn_disk(
        R,
        Z,
        R_solar=R_solar,
        Z_solar=Z_solar,
        R_scale=R_thick,
        Z_scale=Z_thick,
        R_smooth=Rs_thick,
    )
    logp_thick += vol_factor + np.log(f_thick)

    # Halo component
    logp_halo = logn_halo(
        R,
        Z,
        R_solar=R_solar,
        Z_solar=Z_solar,
        R_smooth=Rs_halo,
        eta=eta_halo,
        q_ctr=q_halo_ctr,
        q_inf=q_halo_inf,
        r_q=r_q_halo,
    )
    logp_halo += vol_factor + np.log(f_halo)

    # Combined number density prior
    logp = _logsumexp3(logp_thin, logp_thick, logp_halo)

    # Component tracking
    components = {"number_density": [logp_thin, logp_thick, logp_halo]}

    # Apply metallicity and age priors if labels provided
    if labels is not None:
        # Component membership probabilities
        lnprior_thin = logp_thin - logp
        lnprior_thick = logp_thick - logp
        lnprior_halo = logp_halo - logp

        # Metallicity prior
        if "feh" in labels.dtype.names:
            try:
                feh = labels["feh"]

                # Component-specific metallicity priors
                feh_lnp_thin = (
                    logp_feh(feh, feh_mean=feh_thin, feh_sigma=feh_thin_sigma)
                    + lnprior_thin
                )
                feh_lnp_thick = (
                    logp_feh(feh, feh_mean=feh_thick, feh_sigma=feh_thick_sigma)
                    + lnprior_thick
                )
                feh_lnp_halo = (
                    logp_feh(feh, feh_mean=feh_halo, feh_sigma=feh_halo_sigma)
                    + lnprior_halo
                )

                # Combined metallicity prior
                feh_lnp = _logsumexp3(feh_lnp_thin, feh_lnp_thick, feh_lnp_halo)
                logp += feh_lnp
                components["feh"] = [feh_lnp_thin, feh_lnp_thick, feh_lnp_halo]
            except (KeyError, IndexError, ValueError):
                pass

        # Age prior
        if "loga" in labels.dtype.names:
            try:
                age = 10 ** labels["loga"] / 1e9  # Convert log(age) to Gyr

                # Component-specific age priors
                age_lnp_thin = (
                    logp_age_from_feh(
                        age,
                        feh_mean=feh_thin,
                        max_age=max_age,
                        min_age=min_age,
                        feh_age_ctr=feh_age_ctr,
                        feh_age_scale=feh_age_scale,
                        nsigma_from_max_age=nsigma_from_max_age,
                        max_sigma=max_sigma,
                        min_sigma=min_sigma,
                    )
                    + lnprior_thin
                )

                age_lnp_thick = (
                    logp_age_from_feh(
                        age,
                        feh_mean=feh_thick,
                        max_age=max_age,
                        min_age=min_age,
                        feh_age_ctr=feh_age_ctr,
                        feh_age_scale=feh_age_scale,
                        nsigma_from_max_age=nsigma_from_max_age,
                        max_sigma=max_sigma,
                        min_sigma=min_sigma,
                    )
                    + lnprior_thick
                )

                age_lnp_halo = (
                    logp_age_from_feh(
                        age,
                        feh_mean=feh_halo,
                        max_age=max_age,
                        min_age=min_age,
                        feh_age_ctr=feh_age_ctr,
                        feh_age_scale=feh_age_scale,
                        nsigma_from_max_age=nsigma_from_max_age,
                        max_sigma=max_sigma,
                        min_sigma=min_sigma,
                    )
                    + lnprior_halo
                )

                # Combined age prior
                age_lnp = _logsumexp3(age_lnp_thin, age_lnp_thick, age_lnp_halo)
                logp += age_lnp
                components["age"] = [age_lnp_thin, age_lnp_thick, age_lnp_halo]
            except (KeyError, IndexError, ValueError):
                pass

    if return_components:
        return logp, components
    else:
        return logp
