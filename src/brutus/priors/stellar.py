"""
Stellar initial mass function (IMF) and luminosity function priors.

See Also
--------
brutus.analysis.individual.BruteForce : Uses these priors for stellar fitting
brutus.priors.galactic : Galactic structure priors
brutus.priors.extinction : Extinction priors

Notes
-----
- **IMF priors** (:func:`logp_imf`) return a normalized log-density over initial
  stellar mass (and optional secondary mass for binary systems).
- **Luminosity function priors** (:func:`logp_ps1_luminosity_function`) return
  an unnormalized log-density over Pan-STARRS 1 absolute :math:`r`-band
  magnitude :math:`M_r`.

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

from ..utils.math import LOG_ZERO

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


def _log_imf_unnormalized(mgrid, alpha_low, alpha_high, mass_min, mass_break, mass_max):
    mgrid = np.asarray(mgrid)
    logp = np.full_like(mgrid, LOG_ZERO, dtype=float)
    valid_mass = (mgrid >= mass_min) & (mgrid <= mass_max)
    low_mass = valid_mass & (mgrid <= mass_break)
    logp[low_mass] = -alpha_low * np.log(mgrid[low_mass])
    # The high-mass piece is scaled to be continuous at mass_break.
    high_mass = valid_mass & (mgrid > mass_break)
    logp[high_mass] = -alpha_high * np.log(mgrid[high_mass]) + (
        alpha_high - alpha_low
    ) * np.log(mass_break)
    return logp


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
    Compute the log-prior for a Kroupa-like broken power-law initial mass function.

    Evaluates a two-part power-law IMF following Kroupa (2001), with optional
    independent draws for a binary companion.

    Parameters
    ----------
    mgrid : array_like
        Initial stellar masses in solar masses (:math:`M_\odot`).
    alpha_low : float, optional
        Power-law slope for ``mgrid <= mass_break``. Default is 1.3.
    alpha_high : float, optional
        Power-law slope for ``mgrid > mass_break``. Default is 2.3.
    mass_break : float, optional
        Transition mass between the low- and high-mass regimes in
        :math:`M_\odot`. Default is 0.5.
    mgrid2 : array_like, optional
        Secondary stellar masses for binary systems in :math:`M_\odot`,
        broadcastable with ``mgrid``.
    mass_min : float, optional
        Minimum mass for normalization in :math:`M_\odot`. Default is 0.08.
    mass_max : float, optional
        Maximum mass for normalization in :math:`M_\odot`. May be ``np.inf``
        when ``alpha_high > 1``. Default is 100.0.

    Returns
    -------
    logp : numpy.ndarray
        Normalized log-prior density (per :math:`M_\odot`, or per
        :math:`M_\odot^2` when ``mgrid2`` is given) with the shape of ``mgrid``.
        Masses outside ``[mass_min, mass_max]`` receive ``LOG_ZERO`` (-1e300).

    Raises
    ------
    ValueError
        If ``mass_min`` or ``mass_break`` is not finite and positive, if
        ``mass_max`` does not exceed ``mass_min``, or if a slope is not
        finite.

    See Also
    --------
    logp_ps1_luminosity_function : Alternative luminosity-based prior
    brutus.analysis.individual.BruteForce : Uses IMF priors for fitting

    Notes
    -----
    The IMF follows the form:

    .. math::
        \xi(M) \propto M^{-\alpha}

    where α = α_low for M ≤ M_break and α = α_high for M > M_break.

    For binary systems, assumes independent sampling from the same IMF
    for both components.

    References
    ----------
    Kroupa, P. (2001), MNRAS, 322, 231
    """
    for name, value in (
        ("mass_min", mass_min),
        ("mass_break", mass_break),
    ):
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"`{name}` must be finite and > 0; got {value}.")
    if not mass_max > mass_min:
        raise ValueError(
            f"`mass_max` must exceed `mass_min`; got mass_min={mass_min}, "
            f"mass_max={mass_max}."
        )
    for name, value in (("alpha_low", alpha_low), ("alpha_high", alpha_high)):
        if not np.isfinite(value):
            raise ValueError(f"`{name}` must be finite; got {value}.")

    logp = _log_imf_unnormalized(
        mgrid, alpha_low, alpha_high, mass_min, mass_break, mass_max
    )

    # Compute normalization factor over [mass_min, mass_max], integrating
    # only over the power-law regime(s) the range actually overlaps. The
    # high-mass piece carries the continuity factor
    # mass_break^(alpha_high - alpha_low) so the density is continuous at
    # mass_break.
    continuity_factor = mass_break ** (alpha_high - alpha_low)
    if mass_min < mass_break <= mass_max:
        # Range straddles the break: both regimes contribute
        norm_low = _powerlaw_mass_integral(mass_min, mass_break, alpha_low)
        norm_high = continuity_factor * _powerlaw_mass_integral(
            mass_break, mass_max, alpha_high
        )
        norm = norm_low + norm_high
    elif mass_min >= mass_break:
        # Entire range in the high-mass regime
        norm = continuity_factor * _powerlaw_mass_integral(
            mass_min, mass_max, alpha_high
        )
    else:
        # Entire range in the low-mass regime (mass_max < mass_break)
        norm = _powerlaw_mass_integral(mass_min, mass_max, alpha_low)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"IMF normalization is non-positive or non-finite (norm={norm}); "
            f"check mass_min/mass_max/mass_break and the power-law slopes."
        )

    if mgrid2 is not None:
        logp = logp + _log_imf_unnormalized(
            mgrid2, alpha_low, alpha_high, mass_min, mass_break, mass_max
        )
        norm = norm**2

    return logp - np.log(norm)


# Global interpolator for PS1 luminosity function (loaded on first use)
_ps1_lf_interpolator = None


def logp_ps1_luminosity_function(Mr):
    """
    Evaluate the Pan-STARRS 1 r-band luminosity function log-prior.

    Interpolates the empirical stellar luminosity function tabulated for
    Pan-STARRS 1 :math:`M_r` in ``[-1, 20]`` mag (Green et al. 2015, 2018) and
    extrapolates linearly outside that interval.

    Parameters
    ----------
    Mr : array_like
        Absolute :math:`r`-band magnitudes in the Pan-STARRS 1 system (mag).

    Returns
    -------
    logp : numpy.ndarray
        Unnormalized log-prior values with the shape of ``Mr`` (``exp(logp)``
        integrates to approximately 0.086 over the tabulated range).

    See Also
    --------
    logp_imf : Mass-based IMF prior
    brutus.analysis.individual.BruteForce : Uses luminosity priors for fitting

    References
    ----------
    Green, G. M., et al. (2015), ApJ, 810, 25
    Green, G. M., et al. (2018), MNRAS, 478, 651
    """
    global _ps1_lf_interpolator

    if _ps1_lf_interpolator is None:
        import os

        from scipy.interpolate import interp1d

        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(module_dir, "PSMrLF_lnprior.dat")
        grid_Mr, grid_lnp = np.loadtxt(data_path).T
        _ps1_lf_interpolator = interp1d(
            grid_Mr, grid_lnp, fill_value="extrapolate", kind="linear"
        )

    return _ps1_lf_interpolator(Mr)
