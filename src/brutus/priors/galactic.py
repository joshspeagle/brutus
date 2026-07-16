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

>>> from brutus.priors.galactic import logp_galactic_structure
>>>
>>> # Galactic coordinates (l, b) in degrees and distances in kpc
>>> coord = (90.0, 30.0)
>>> distances = np.array([1.0, 2.0, 5.0])
>>>
>>> # Evaluate the combined thin disk + thick disk + halo prior
>>> log_prior = logp_galactic_structure(distances, coord)
"""

import warnings
from math import erf, exp, log, pi, sqrt

import numpy as np
from numba import jit, prange

# Import utility functions from brutus.utils
from brutus.utils import truncnorm_logpdf
from brutus.utils.math import galactic_to_galactocentric_cyl

_LOG_2PI = log(2.0 * pi)
_SQRT2 = sqrt(2.0)


def _logsumexp3(a, b, c):
    """Fast logsumexp of 3 arrays using numpy's logaddexp ufunc."""
    return np.logaddexp(np.logaddexp(a, b), c)


@jit(nopython=True, parallel=True, cache=True)
def _galactic_prior_fused(
    dists,
    sin_l,
    cos_l,
    sin_b,
    cos_b,
    feh_arr,
    loga_arr,
    has_feh,
    has_loga,
    R_solar,
    Z_solar,
    # Thin disk
    R_thin,
    Z_thin,
    Rs_thin,
    # Thick disk
    R_thick,
    Z_thick,
    Rs_thick,
    f_thick,
    # Halo
    Rs_halo,
    eta_halo,
    q_halo_ctr,
    q_halo_inf,
    r_q_halo,
    f_halo,
    # Metallicity
    feh_thin_mean,
    feh_thin_sigma,
    feh_thick_mean,
    feh_thick_sigma,
    feh_halo_mean,
    feh_halo_sigma,
    # Age
    max_age,
    min_age,
    feh_age_ctr,
    feh_age_scale,
    nsigma_from_max_age,
    max_sigma_age,
    min_sigma_age,
):
    """Fused galactic prior: coords + density + vol + met + age in one pass.

    ``sin_l``/``cos_l``/``sin_b``/``cos_b`` are the precomputed sine/cosine
    of the (scalar) Galactic longitude/latitude; the Galactocentric (R, Z)
    conversion is folded into the parallel loop with the exact operation
    order of ``galactic_to_galactocentric_cyl`` (bitwise-identical results,
    no full-size temporaries).
    """
    N = len(dists)
    logp_out = np.empty(N)

    # --- Loop-invariant constants (hoisted out of the prange loop) ---

    # Solar halo normalization
    r_solar = sqrt(R_solar**2 + Z_solar**2)
    r_prime_solar = sqrt(r_solar**2 + r_q_halo**2)
    q_solar = q_halo_inf - (q_halo_inf - q_halo_ctr) * exp(
        1.0 - r_prime_solar / r_q_halo
    )
    R_eff_solar_halo = sqrt(R_solar**2 + (Z_solar / q_solar) ** 2 + Rs_halo**2)

    # Solar disk normalizations use the SMOOTHED solar radius so the disk
    # density is exactly 1 at the solar position (consistent with the halo
    # normalization above and with ``logn_disk``); otherwise the f_thick /
    # f_halo mixture fractions are silently rescaled by ~10%.
    R_solar_eff_thin = sqrt(R_solar**2 + Rs_thin**2)
    R_solar_eff_thick = sqrt(R_solar**2 + Rs_thick**2)
    abs_Z_solar = abs(Z_solar)
    ln_f_thick = log(f_thick)
    ln_f_halo = log(f_halo)

    # Age-prior constants depend only on the component metallicity means,
    # not on the grid point: 3 exp + 6 erf + logs hoisted out of the loop.
    age_mean_c = np.empty(3)
    age_sigma_c = np.empty(3)
    lndenom_c = np.empty(3)
    for comp_idx in range(3):
        if comp_idx == 0:
            fm = feh_thin_mean
        elif comp_idx == 1:
            fm = feh_thick_mean
        else:
            fm = feh_halo_mean
        age_mean = (max_age - min_age) / (
            1.0 + exp((fm - feh_age_ctr) / feh_age_scale)
        ) + min_age
        age_sigma = (max_age - age_mean) / nsigma_from_max_age
        if age_sigma < min_sigma_age:
            age_sigma = min_sigma_age
        if age_sigma > max_sigma_age:
            age_sigma = max_sigma_age
        alpha = (min_age - age_mean) / age_sigma
        beta = (max_age - age_mean) / age_sigma
        denom = max(erf(beta / _SQRT2) - erf(alpha / _SQRT2), 1e-300)
        age_mean_c[comp_idx] = age_mean
        age_sigma_c[comp_idx] = age_sigma
        lndenom_c[comp_idx] = log(age_sigma / 2.0) + log(denom)

    for i in prange(N):
        d = dists[i]

        # Galactocentric cylindrical coordinates (same operation order as
        # galactic_to_galactocentric_cyl for bitwise-identical results;
        # explicit multiplication instead of **2 matches numpy's squaring
        # fast path in both JIT and NUMBA_DISABLE_JIT modes)
        x = d * cos_b * cos_l - R_solar
        y = d * cos_b * sin_l
        R = sqrt(x * x + y * y)
        Z = d * sin_b + Z_solar

        # Volume factor
        vol = 2.0 * log(d + 1e-300)

        # Thin disk
        R_eff_thin = sqrt(R**2 + Rs_thin**2)
        lnp_thin = (
            -(R_eff_thin - R_solar_eff_thin) / R_thin
            - (abs(Z) - abs_Z_solar) / Z_thin
            + vol
        )

        # Thick disk
        R_eff_thick = sqrt(R**2 + Rs_thick**2)
        lnp_thick = (
            -(R_eff_thick - R_solar_eff_thick) / R_thick
            - (abs(Z) - abs_Z_solar) / Z_thick
            + vol
            + ln_f_thick
        )

        # Halo
        r = sqrt(R**2 + Z**2)
        r_prime = sqrt(r**2 + r_q_halo**2)
        q = q_halo_inf - (q_halo_inf - q_halo_ctr) * exp(1.0 - r_prime / r_q_halo)
        R_eff_halo = sqrt(R**2 + (Z / q) ** 2 + Rs_halo**2)
        lnp_halo = -eta_halo * log(R_eff_halo / R_eff_solar_halo) + vol + ln_f_halo

        # logsumexp3
        mx = max(lnp_thin, max(lnp_thick, lnp_halo))
        logp_total = mx + log(
            exp(lnp_thin - mx) + exp(lnp_thick - mx) + exp(lnp_halo - mx)
        )

        # Metallicity prior
        feh_lnp = 0.0
        feh_lnp_thin = 0.0
        feh_lnp_thick = 0.0
        feh_lnp_halo = 0.0
        if has_feh:
            feh_val = feh_arr[i]
            # Component membership
            ln_w_thin = lnp_thin - logp_total
            ln_w_thick = lnp_thick - logp_total
            ln_w_halo = lnp_halo - logp_total

            # Gaussian logpdf for each component
            feh_lnp_thin = (
                -0.5
                * (
                    (feh_val - feh_thin_mean) ** 2 / feh_thin_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_thin_sigma)
                )
                + ln_w_thin
            )
            feh_lnp_thick = (
                -0.5
                * (
                    (feh_val - feh_thick_mean) ** 2 / feh_thick_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_thick_sigma)
                )
                + ln_w_thick
            )
            feh_lnp_halo = (
                -0.5
                * (
                    (feh_val - feh_halo_mean) ** 2 / feh_halo_sigma**2
                    + _LOG_2PI
                    + 2 * log(feh_halo_sigma)
                )
                + ln_w_halo
            )

            mx2 = max(feh_lnp_thin, max(feh_lnp_thick, feh_lnp_halo))
            feh_lnp = mx2 + log(
                exp(feh_lnp_thin - mx2)
                + exp(feh_lnp_thick - mx2)
                + exp(feh_lnp_halo - mx2)
            )
            logp_total += feh_lnp

        # Age prior
        if has_loga:
            age_val = 10.0 ** loga_arr[i] / 1e9  # Gyr
            if has_feh:
                # Metallicity-updated membership weights: the joint mixture
                # prior is sum_c w_c(pos) p(feh|c) p(age|c), so the age term
                # must weight components by w_c(pos) p(feh|c) / Z_feh rather
                # than by the position-only w_c(pos).
                ln_w_thin = feh_lnp_thin - feh_lnp
                ln_w_thick = feh_lnp_thick - feh_lnp
                ln_w_halo = feh_lnp_halo - feh_lnp
            else:
                ln_w_thin = lnp_thin - logp_total
                ln_w_thick = lnp_thick - logp_total
                ln_w_halo = lnp_halo - logp_total

            # Compute truncated normal for each component
            age_lnp_total = -1e300
            for comp_idx in range(3):
                if comp_idx == 0:
                    ln_w = ln_w_thin
                elif comp_idx == 1:
                    ln_w = ln_w_thick
                else:
                    ln_w = ln_w_halo

                xi = (age_val - age_mean_c[comp_idx]) / age_sigma_c[comp_idx]
                lnphi = -0.5 * (_LOG_2PI + xi * xi)
                age_comp = lnphi - lndenom_c[comp_idx] + ln_w

                # Bounds check
                if age_val < min_age or age_val > max_age:
                    age_comp = -1e300

                # Accumulate logsumexp
                if age_comp > age_lnp_total:
                    age_lnp_total = (
                        age_comp + log(1.0 + exp(age_lnp_total - age_comp))
                        if age_lnp_total > -1e200
                        else age_comp
                    )
                else:
                    age_lnp_total = (
                        age_lnp_total + log(1.0 + exp(age_comp - age_lnp_total))
                        if age_comp > -1e200
                        else age_lnp_total
                    )

            logp_total += age_lnp_total

        logp_out[i] = logp_total

    return logp_out


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
        n_{\\text{disk}}(R, Z) \\propto \\exp\\left(-\\frac{R_{\\text{eff}} - R_{\\text{eff},\\odot}}{R_{\\text{scale}}} - \\frac{|Z| - |Z_\\odot|}{Z_{\\text{scale}}}\\right)

    where :math:`R_{\\text{eff}} = \\sqrt{R^2 + R_{\\text{smooth}}^2}` provides
    smoothing near the Galactic center and
    :math:`R_{\\text{eff},\\odot} = \\sqrt{R_\\odot^2 + R_{\\text{smooth}}^2}`
    is the smoothed solar radius, so the normalized density is exactly 1 at
    the solar position (consistent with :func:`logn_halo`). This keeps the
    mixture fractions (``f_thick``, ``f_halo``) in
    :func:`logp_galactic_structure` exact at the solar position.

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    R = np.asarray(R)
    Z = np.asarray(Z)

    # Smoothed effective radius
    R_eff = np.sqrt(R**2 + R_smooth**2)

    # Exponential disk components, normalized at the SMOOTHED solar radius
    # so that logn_disk(R_solar, Z_solar) == 0 (as for logn_halo).
    R_eff_solar = np.sqrt(R_solar**2 + R_smooth**2)
    radial_term = (R_eff - R_eff_solar) / R_scale
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
    feh=None,
    loga=None,
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
    feh : array_like, optional
        Per-point [Fe/H] as a plain float array, an alternative to ``labels``
        for the metallicity prior. Lets callers (e.g. ``logpost_grid``) avoid
        tiling a structured ``labels`` array; equivalent to passing the same
        values via ``labels``. Honored on every code path (fused kernel,
        numpy fallback, and small arrays). If both ``labels`` and
        ``feh``/``loga`` are given, the plain arrays take precedence.
    loga : array_like, optional
        Per-point log10(age/yr) as a plain float array, an alternative to
        ``labels`` for the age prior (see ``feh``).
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

    When both metallicity and age are supplied, the joint prior is the
    proper mixture ``sum_c w_c(pos) p(feh|c) p(age|c)``: the age term
    weights each component by its metallicity-updated membership
    probability ``w_c(pos) p(feh|c) / sum_c' w_c'(pos) p(feh|c')``, not by
    the position-only weight. This matters for chemically extreme stars
    (e.g. nearby metal-poor stars are correctly weighted toward old,
    halo-like ages).

    References
    ----------
    Bland-Hawthorn & Gerhard (2016) - The Galaxy in Context
    """
    dists = np.asarray(dists, dtype=float)

    # Extract Galactic (l, b) in degrees
    if hasattr(coord, "galactic"):
        # coord is already a SkyCoord object
        ell = coord.galactic.l.deg
        b = coord.galactic.b.deg
    else:
        # coord is a tuple of (l, b) in degrees
        ell, b = coord[0], coord[1]

    # Callers may pass the per-point ``feh`` / ``loga`` as plain float arrays
    # directly (instead of a structured ``labels`` array). This is the hot path
    # from logpost_grid, where it avoids tiling a structured array across all MC
    # samples -- a numpy structured-dtype ``np.tile`` is ~100x slower than the
    # equivalent float tile. The values handed to the fused kernel are identical
    # either way, so results are bitwise-identical to the structured path.
    use_arrays = feh is not None or loga is not None
    if use_arrays:
        # Normalize to contiguous float64 up front so the numba kernel types
        # cleanly (lists, float32, object arrays) and misuse gives an
        # actionable error rather than failing later inside the kernel.
        if feh is not None:
            feh = np.ascontiguousarray(feh, dtype=np.float64)
            if feh.ndim != 1 or feh.shape[0] != len(dists):
                raise ValueError(
                    f"`feh` must be a 1D array matching len(dists)="
                    f"{len(dists)}; got shape {np.shape(feh)}."
                )
        if loga is not None:
            loga = np.ascontiguousarray(loga, dtype=np.float64)
            if loga.ndim != 1 or loga.shape[0] != len(dists):
                raise ValueError(
                    f"`loga` must be a 1D array matching len(dists)="
                    f"{len(dists)}; got shape {np.shape(loga)}."
                )

    # Fast path: use fused numba kernel when labels are provided and
    # return_components is not needed. Eliminates ~15 temporary arrays
    # (the coordinate conversion and volume factor are computed per point
    # inside the kernel). Requires a single (scalar) sky position.
    if (
        (labels is not None or use_arrays)
        and not return_components
        and len(dists) > 1000
        and np.ndim(ell) == 0
        and np.ndim(b) == 0
    ):
        if use_arrays:
            has_feh = feh is not None
            has_loga = loga is not None
            feh_arr = feh if has_feh else np.empty(0)
            loga_arr = loga if has_loga else np.empty(0)
        else:
            has_feh = "feh" in labels.dtype.names
            has_loga = "loga" in labels.dtype.names
            feh_arr = labels["feh"] if has_feh else np.empty(0)
            loga_arr = labels["loga"] if has_loga else np.empty(0)
        ell_rad = np.deg2rad(float(ell))
        b_rad = np.deg2rad(float(b))
        try:
            return _galactic_prior_fused(
                dists,
                np.sin(ell_rad),
                np.cos(ell_rad),
                np.sin(b_rad),
                np.cos(b_rad),
                feh_arr,
                loga_arr,
                has_feh,
                has_loga,
                R_solar,
                Z_solar,
                R_thin,
                Z_thin,
                Rs_thin,
                R_thick,
                Z_thick,
                Rs_thick,
                f_thick,
                Rs_halo,
                eta_halo,
                q_halo_ctr,
                q_halo_inf,
                r_q_halo,
                f_halo,
                feh_thin,
                feh_thin_sigma,
                feh_thick,
                feh_thick_sigma,
                feh_halo,
                feh_halo_sigma,
                max_age,
                min_age,
                feh_age_ctr,
                feh_age_scale,
                nsigma_from_max_age,
                max_sigma,
                min_sigma,
            )
        except Exception as e:
            warnings.warn(
                f"Numba fused galactic prior failed, falling back to numpy: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---- Numpy path (small arrays, return_components, array coordinates,
    # or fused-kernel fallback; also the NUMBA_DISABLE_JIT coverage path) ----

    # Volume correction factor (dV ∝ r² dr); the fused kernel computes this
    # per point, so it is only materialized here.
    vol_factor = 2.0 * np.log(dists + 1e-300)

    # Convert to Galactocentric cylindrical coordinates using fast NumPy math
    R, Z = galactic_to_galactocentric_cyl(
        dists, ell, b, R_solar=R_solar, Z_solar=Z_solar
    )

    # Plain feh/loga arrays must receive the exact same priors as the fused
    # kernel: synthesize a structured labels array so the code below applies
    # them (previously these were silently dropped on this path).
    if use_arrays:
        _dtype = []
        if feh is not None:
            _dtype.append(("feh", np.float64))
        if loga is not None:
            _dtype.append(("loga", np.float64))
        labels = np.empty(len(dists), dtype=_dtype)
        if feh is not None:
            labels["feh"] = feh
        if loga is not None:
            labels["loga"] = loga

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
        # Component membership probabilities (reuse logp_thin/thick/halo)
        lnprior_thin = logp_thin - logp
        lnprior_thick = logp_thick - logp
        lnprior_halo = logp_halo - logp

        # Membership weights for the age mixture. Position-only by default;
        # updated with the metallicity evidence below when 'feh' is present,
        # so the joint prior is sum_c w_c(pos) p(feh|c) p(age|c) (matching
        # the fused kernel) rather than a product of marginal mixtures.
        age_w_thin = lnprior_thin
        age_w_thick = lnprior_thick
        age_w_halo = lnprior_halo

        # Metallicity prior — fused computation to minimize temporaries
        if "feh" in labels.dtype.names:
            try:
                feh = labels["feh"]

                # Inline Gaussian logpdf for all 3 components at once
                # logp_feh(feh, mu, sigma) = -0.5*((feh-mu)^2/sigma^2 + log(2*pi*sigma^2))
                feh_lnp_thin = (
                    -0.5
                    * (
                        (feh - feh_thin) ** 2 / feh_thin_sigma**2
                        + np.log(2.0 * np.pi * feh_thin_sigma**2)
                    )
                    + lnprior_thin
                )
                feh_lnp_thick = (
                    -0.5
                    * (
                        (feh - feh_thick) ** 2 / feh_thick_sigma**2
                        + np.log(2.0 * np.pi * feh_thick_sigma**2)
                    )
                    + lnprior_thick
                )
                feh_lnp_halo = (
                    -0.5
                    * (
                        (feh - feh_halo) ** 2 / feh_halo_sigma**2
                        + np.log(2.0 * np.pi * feh_halo_sigma**2)
                    )
                    + lnprior_halo
                )

                feh_lnp = _logsumexp3(feh_lnp_thin, feh_lnp_thick, feh_lnp_halo)
                logp += feh_lnp
                components["feh"] = [feh_lnp_thin, feh_lnp_thick, feh_lnp_halo]

                # Metallicity-updated membership weights for the age mixture
                age_w_thin = feh_lnp_thin - feh_lnp
                age_w_thick = feh_lnp_thick - feh_lnp
                age_w_halo = feh_lnp_halo - feh_lnp
            except (KeyError, IndexError, ValueError) as e:
                warnings.warn(
                    f"Metallicity prior computation failed: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Age prior — fused computation
        if "loga" in labels.dtype.names:
            try:
                age = 10.0 ** labels["loga"] / 1e9  # Convert log(age) to Gyr

                # Compute age priors for all 3 components with shared parameters.
                # logp_age_from_feh computes a truncated normal; inline the core
                # computation to avoid 3 separate function calls + temporaries.
                age_params = []
                for fm, lnp_comp in [
                    (feh_thin, age_w_thin),
                    (feh_thick, age_w_thick),
                    (feh_halo, age_w_halo),
                ]:
                    age_mean = (max_age - min_age) / (
                        1.0 + np.exp((fm - feh_age_ctr) / feh_age_scale)
                    ) + min_age
                    age_sigma = np.clip(
                        (max_age - age_mean) / nsigma_from_max_age,
                        min_sigma,
                        max_sigma,
                    )
                    age_params.append(
                        truncnorm_logpdf(
                            age,
                            (min_age - age_mean) / age_sigma,
                            (max_age - age_mean) / age_sigma,
                            loc=age_mean,
                            scale=age_sigma,
                        )
                        + lnp_comp
                    )

                age_lnp_thin, age_lnp_thick, age_lnp_halo = age_params
                age_lnp = _logsumexp3(age_lnp_thin, age_lnp_thick, age_lnp_halo)
                logp += age_lnp
                components["age"] = [age_lnp_thin, age_lnp_thick, age_lnp_halo]
            except (KeyError, IndexError, ValueError) as e:
                warnings.warn(
                    f"Age prior computation failed: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if return_components:
        return logp, components
    else:
        return logp
