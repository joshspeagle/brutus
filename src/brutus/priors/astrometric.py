#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Astrometric priors for Bayesian stellar parameter estimation.

This module provides log-prior functions for astrometric measurements
including parallax priors and coordinate transformations. These priors
are essential for incorporating Gaia astrometry into stellar parameter
estimation.

Functions
---------
logp_parallax : Parallax prior
    Gaussian prior from measured parallax
logp_parallax_scale : Scale factor prior
    Prior on distance scale (s = 1/d^2)
convert_parallax_to_scale : Coordinate transform
    Convert parallax to flux scale factor

See Also
--------
brutus.priors.galactic : Galactic structure distance priors
brutus.analysis.individual.BruteForce : Uses parallax priors for fitting

Notes
-----
These priors incorporate astrometric information from missions like Gaia
to constrain stellar distances and luminosities.

The parallax prior is straightforward Gaussian, but care must be taken
with the coordinate transformation when using scale factors (s = 1/d^2)
rather than distances directly. The Jacobian must be properly accounted for.

Examples
--------
>>> from brutus.priors.astrometric import logp_parallax
>>> import numpy as np
>>>
>>> # Gaia parallax measurement
>>> p_meas = 2.5  # mas
>>> p_err = 0.1   # mas
>>>
>>> # Evaluate prior for model parallaxes
>>> parallaxes = np.linspace(1.0, 4.0, 100)
>>> log_prior = logp_parallax(parallaxes, p_meas, p_err)
"""

import numpy as np

__all__ = ["logp_parallax", "logp_parallax_scale", "convert_parallax_to_scale"]

# Numerical constants for uninformative prior bounds
SCALE_MIN = 1e-20  # Minimum scale factor (essentially zero)
SCALE_MAX = 1e20  # Maximum scale factor (essentially infinity)


def logp_parallax(parallaxes, p_meas, p_err):
    r"""
    Log-prior for parallax measurements assuming Gaussian errors.

    Implements a Gaussian log-prior based on observed parallax and
    measurement uncertainty. Returns uniform prior when measurements
    are invalid or unavailable.

    Parameters
    ----------
    parallaxes : array_like
        Model parallax values in milliarcseconds (mas).
    p_meas : float
        Measured parallax in milliarcseconds (mas).
    p_err : float
        Parallax measurement uncertainty in milliarcseconds (mas).

    Returns
    -------
    logp : array_like
        Log-prior probability density for the input parallax values.
        Returns 0 (uniform prior) if measurements are invalid.

    Notes
    -----
    The log-prior follows a normal distribution:

    .. math::
        \\log p(\\pi | \\pi_{\\text{obs}}, \\sigma_\\pi) =
        -\\frac{1}{2} \\left[ \\frac{(\\pi - \\pi_{\\text{obs}})^2}{\\sigma_\\pi^2} +
        \\log(2\\pi\\sigma_\\pi^2) \\right]

    For invalid measurements (non-finite values), returns uniform prior.
    """
    parallaxes = np.asarray(parallaxes)

    # Check for valid measurements
    if np.isfinite(p_meas) and np.isfinite(p_err) and p_err > 0:
        # Gaussian log-prior
        chi2 = (parallaxes - p_meas) ** 2 / p_err**2
        log_norm = np.log(2.0 * np.pi * p_err**2)
        logp = -0.5 * (chi2 + log_norm)
    else:
        # Uniform prior for invalid measurements
        logp = np.zeros_like(parallaxes, dtype=float)

    return logp


def logp_parallax_scale(scales, scale_errs, p_meas, p_err, snr_lim=4.0):
    r"""
    Log-prior for flux scale factors derived from parallax measurements.

    Applies parallax constraints to flux density scale factors where
    scale ~ parallax². Uses Gaussian approximation for high signal-to-noise
    parallax measurements.

    Parameters
    ----------
    scales : array_like
        Flux density scale factors (proportional to parallax²).
    scale_errs : array_like
        Scale factor measurement uncertainties.
    p_meas : float
        Measured parallax in milliarcseconds (mas).
    p_err : float
        Parallax measurement uncertainty in milliarcseconds (mas).
    snr_lim : float, optional
        Minimum signal-to-noise ratio for applying Gaussian approximation.
        Below this threshold, returns uniform prior. Default is 4.0.

    Returns
    -------
    logp : array_like
        Log-prior probability density for the input scale factors.

    Notes
    -----
    For high SNR measurements (p_meas/p_err > snr_lim), the scale factor
    prior is derived from the parallax measurement using error propagation:

    .. math::
        s = \\pi^2 + \\sigma_\\pi^2

        \\sigma_s = \\sqrt{2\\sigma_\\pi^4 + 4\\pi^2\\sigma_\\pi^2}

    The total uncertainty combines model and measurement errors in quadrature.
    """
    scales = np.asarray(scales)
    scale_errs = np.asarray(scale_errs)

    # Check SNR threshold and measurement validity
    if (
        np.isfinite(p_meas)
        and np.isfinite(p_err)
        and p_err > 0
        and abs(p_meas / p_err) > snr_lim
    ):

        # Convert parallax to scale factor statistics
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err, snr_lim)

        # Total variance (model + measurement)
        total_var = s_std**2 + scale_errs**2

        # Gaussian log-prior
        chi2 = (scales - s_mean) ** 2 / total_var
        log_norm = np.log(2.0 * np.pi * total_var)
        logp = -0.5 * (chi2 + log_norm)
    else:
        # Uniform prior for low SNR or invalid measurements
        logp = np.zeros_like(scales, dtype=float)

    return logp


def convert_parallax_to_scale(p_meas, p_err, snr_lim=4.0):
    r"""
    Convert parallax measurement to flux density scale factor statistics.

    Transforms parallax measurements and uncertainties to scale factor
    (s ~ π²) mean and standard deviation using error propagation.

    Parameters
    ----------
    p_meas : float
        Measured parallax in milliarcseconds (mas).
    p_err : float
        Parallax measurement uncertainty in milliarcseconds (mas).
    snr_lim : float, optional
        Minimum signal-to-noise ratio for conversion. Below this threshold,
        returns uninformative scale factor statistics. Default is 4.0.

    Returns
    -------
    s_mean : float
        Mean of the scale factor distribution.
    s_std : float
        Standard deviation of the scale factor distribution.

    Notes
    -----
    For high SNR measurements, uses error propagation:

    .. math::
        s_{\\text{mean}} = \\max(0, \\pi_{\\text{meas}})^2 + \\sigma_\\pi^2

        s_{\\text{std}} = \\sqrt{2\\sigma_\\pi^4 + 4\\pi_{\\text{meas}}^2\\sigma_\\pi^2}

    The parallax is floored at zero to handle negative measurements.
    For low SNR or invalid measurements (non-finite values, or
    ``p_err <= 0``), returns uninformative statistics (tiny mean, huge std).

    Examples
    --------
    >>> s_mean, s_std = convert_parallax_to_scale(1.0, 0.1)  # 10-sigma detection
    >>> print(f"Scale factor: {s_mean:.3f} ± {s_std:.3f}")
    """
    # Guard p_err > 0 before dividing (mirrors logp_parallax /
    # logp_parallax_scale): p_err == 0 would raise ZeroDivisionError for
    # Python floats or silently return a delta-function prior for numpy
    # scalars (inf > snr_lim), instead of the documented uninformative
    # fallback.
    if (
        np.isfinite(p_meas)
        and np.isfinite(p_err)
        and p_err > 0
        and abs(p_meas / p_err) > snr_lim
    ):
        # Floor parallax at zero to handle negative measurements
        p_positive = max(0.0, p_meas)

        # Scale factor statistics via error propagation
        s_mean = p_positive**2 + p_err**2
        s_std = np.sqrt(2 * p_err**4 + 4 * p_positive**2 * p_err**2)
    else:
        # Uninformative prior for low SNR measurements
        s_mean, s_std = SCALE_MIN, SCALE_MAX

    return s_mean, s_std
