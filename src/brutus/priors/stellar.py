#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stellar priors for Bayesian parameter estimation.

This module provides log-prior functions for stellar properties including
the initial mass function (IMF) and luminosity functions. These priors
are used in Bayesian inference of stellar parameters to incorporate
physical constraints from stellar populations.

Functions
---------
logp_imf : Initial mass function prior
    Kroupa-like broken power-law IMF
logp_ps1_luminosity_function : Luminosity function prior
    Pan-STARRS 1 r-band luminosity function

See Also
--------
brutus.analysis.individual.BruteForce : Uses these priors for stellar fitting
brutus.priors.galactic : Galactic structure priors
brutus.priors.extinction : Extinction priors

Notes
-----
These priors provide physically-motivated probability distributions for
stellar parameters:

- **IMF priors** weight stellar masses according to population statistics,
  ensuring realistic mass distributions in Bayesian fits

- **Luminosity function priors** weight absolute magnitudes according to
  observed stellar populations, useful when fitting distance and extinction

The priors are normalized and return log-probabilities suitable for direct
use in MCMC or nested sampling codes.

Examples
--------
Basic IMF prior usage:

>>> import numpy as np
>>> from brutus.priors.stellar import logp_imf
>>>
>>> # Evaluate IMF prior for solar-mass star
>>> masses = np.array([1.0])
>>> log_prior = logp_imf(masses)
>>> print(f"Log-prior for 1 solar mass: {log_prior[0]:.3f}")
>>>
>>> # Binary system with 1.0 + 0.5 solar mass components
>>> log_prior_binary = logp_imf(masses, mgrid2=np.array([0.5]))
>>> print(f"Binary log-prior: {log_prior_binary[0]:.3f}")

Luminosity function prior:

>>> from brutus.priors.stellar import logp_ps1_luminosity_function
>>>
>>> # Absolute r-band magnitude for main sequence star
>>> Mr = np.array([5.0])
>>> log_prior = logp_ps1_luminosity_function(Mr)
>>> print(f"Log-prior for Mr=5: {log_prior[0]:.3f}")
"""

import numpy as np

__all__ = ["logp_imf", "logp_ps1_luminosity_function"]


def _powerlaw_mass_integral(m_lo, m_hi, alpha):
    """Integral of ``M**(-alpha) dM`` from ``m_lo`` to ``m_hi``.

    Uses the logarithmic form ``ln(m_hi / m_lo)`` when ``alpha`` is (near) 1,
    where the usual ``(m**(1 - alpha)) / (1 - alpha)`` expression is singular
    (0/0). This keeps the IMF normalization finite and correct for the
    physically meaningful flat-in-log slope ``alpha == 1`` (and tames the
    catastrophic cancellation that occurs for slopes very close to 1).
    """
    if np.isclose(alpha, 1.0):
        return np.log(m_hi / m_lo)
    return (m_hi ** (1.0 - alpha) - m_lo ** (1.0 - alpha)) / (1.0 - alpha)


def logp_imf(
    mgrid,
    alpha_low=1.3,
    alpha_high=2.3,
    mass_break=0.5,
    mgrid2=None,
    mass_min=0.08,
    mass_max=100.0,
):
    r"""
    Log-prior for a Kroupa-like broken initial mass function.

    Implements a broken power-law IMF with separate slopes for low and high
    stellar masses, following Kroupa (2001). Supports binary systems with
    a secondary mass component.

    Parameters
    ----------
    mgrid : array_like
        Grid of initial stellar masses in solar units. Must be > 0.
    alpha_low : float, optional
        Power-law slope for low-mass stars (M ≤ mass_break).
        Default is 1.3 (Kroupa 2001).
    alpha_high : float, optional
        Power-law slope for high-mass stars (M > mass_break).
        Default is 2.3 (Kroupa 2001).
    mass_break : float, optional
        Transition mass between low and high mass regimes in solar units.
        Default is 0.5.
    mgrid2 : array_like, optional
        Grid of secondary stellar masses for binary systems in solar units.
        If provided, computes joint prior for binary system.
    mass_min : float, optional
        Minimum mass for normalization (hydrogen burning limit).
        Default is 0.08 solar masses.
    mass_max : float, optional
        Maximum mass for normalization. Default is 100.0 solar masses.

    Returns
    -------
    logp : array_like
        Normalized log-prior probability density for the input mass grid(s).
        Returns -inf for masses below mass_min or above mass_max.

    See Also
    --------
    logp_ps1_luminosity_function : Alternative luminosity-based prior
    brutus.analysis.individual.BruteForce : Uses IMF priors for fitting

    Notes
    -----
    The IMF follows the form:

    .. math::
        \\xi(M) \\propto M^{-\\alpha}

    where α = α_low for M ≤ M_break and α = α_high for M > M_break.

    For binary systems, assumes independent sampling from the same IMF
    for both components.

    References
    ----------
    Kroupa, P. (2001), MNRAS, 322, 231
    """
    mgrid = np.asarray(mgrid)

    # Initialize log-prior with -inf for invalid masses
    logp = np.full_like(mgrid, -np.inf, dtype=float)

    # Valid mass range
    valid_mass = (mgrid >= mass_min) & (mgrid <= mass_max)

    # Low-mass regime: M ≤ mass_break
    low_mass = valid_mass & (mgrid <= mass_break)
    logp[low_mass] = -alpha_low * np.log(mgrid[low_mass])

    # High-mass regime: M > mass_break
    high_mass = valid_mass & (mgrid > mass_break)
    logp[high_mass] = -alpha_high * np.log(mgrid[high_mass]) + (
        alpha_high - alpha_low
    ) * np.log(mass_break)

    # Compute normalization factor over [mass_min, mass_max]
    # Low-mass regime: mass_min to min(mass_break, mass_max)
    if mass_max >= mass_break:
        # Both regimes present
        norm_low = _powerlaw_mass_integral(mass_min, mass_break, alpha_low)
        # High-mass regime: mass_break to mass_max
        # The high-mass PDF includes continuity factor: mass_break^(alpha_high - alpha_low)
        continuity_factor = mass_break ** (alpha_high - alpha_low)
        norm_high = continuity_factor * _powerlaw_mass_integral(
            mass_break, mass_max, alpha_high
        )
        norm = norm_low + norm_high
    else:
        # Only low-mass regime present
        norm = _powerlaw_mass_integral(mass_min, mass_max, alpha_low)

    # Handle binary component if provided
    if mgrid2 is not None:
        mgrid2 = np.asarray(mgrid2)

        # Compute prior for secondary
        logp2 = np.full_like(mgrid2, -np.inf, dtype=float)

        valid_mass2 = (mgrid2 >= mass_min) & (mgrid2 <= mass_max)
        low_mass2 = valid_mass2 & (mgrid2 <= mass_break)
        high_mass2 = valid_mass2 & (mgrid2 > mass_break)

        logp2[low_mass2] = -alpha_low * np.log(mgrid2[low_mass2])
        logp2[high_mass2] = -alpha_high * np.log(mgrid2[high_mass2]) + (
            alpha_high - alpha_low
        ) * np.log(mass_break)

        # Combined log-prior for binary system
        logp = logp + logp2

        # Updated normalization for binary (independent sampling)
        norm = norm**2

    # Apply normalization
    return logp - np.log(norm)


# Global interpolator for PS1 luminosity function (loaded on first use)
_ps1_lf_interpolator = None


def logp_ps1_luminosity_function(Mr):
    """
    Pan-STARRS 1 r-band luminosity function log-prior.

    Implements the stellar luminosity function derived from Pan-STARRS 1
    observations, specifically calibrated for use with Bayestar stellar
    evolutionary models and dust maps.

    Parameters
    ----------
    Mr : array_like
        Absolute r-band magnitudes in the Pan-STARRS 1 photometric system.

    Returns
    -------
    logp : array_like
        Log-prior probability density for the given absolute magnitudes.
        Interpolates from empirical Pan-STARRS 1 luminosity function.

    See Also
    --------
    logp_imf : Alternative mass-based IMF prior
    brutus.analysis.individual.BruteForce : Uses luminosity priors for fitting

    Notes
    -----
    This prior is designed for integration with:

    - Bayestar stellar evolutionary tracks
    - Bayestar 3D dust extinction maps
    - Pan-STARRS 1 photometric system

    The luminosity function is based on empirical measurements from the
    Pan-STARRS 1 survey and provides realistic stellar population weights
    for Bayesian inference.

    References
    ----------
    Green et al. (2015) - 3D Dust Mapping with Pan-STARRS 1
    Green et al. (2018) - Bayestar dust maps
    """
    global _ps1_lf_interpolator

    # Load data file on first use
    if _ps1_lf_interpolator is None:
        import os

        from scipy.interpolate import interp1d

        # Get path to data file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(module_dir, "PSMrLF_lnprior.dat")

        # Load PS1 luminosity function data
        grid_Mr, grid_lnp = np.loadtxt(data_path).T

        # Create interpolator with extrapolation
        _ps1_lf_interpolator = interp1d(
            grid_Mr, grid_lnp, fill_value="extrapolate", kind="linear"
        )

    # Evaluate log-prior
    return _ps1_lf_interpolator(Mr)
