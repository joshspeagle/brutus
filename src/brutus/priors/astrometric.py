"""
Astrometric parallax priors and scale-factor conversions.

See Also
--------
brutus.priors.galactic : Galactic structure distance priors
brutus.analysis.individual.BruteForce : Uses parallax priors for fitting

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
    Compute the Gaussian log-prior for parallax measurements.

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
    logp : numpy.ndarray
        Log-prior probability density (per mas) with the shape of
        ``parallaxes``. Returns zeros when ``p_meas`` or ``p_err`` is
        non-finite or ``p_err <= 0``.

    Notes
    -----
    For valid measurements, the log-prior is

    .. math::
        \log p(\pi \mid \pi_{\text{obs}}, \sigma_\pi) =
        -\frac{1}{2} \left[ \frac{(\pi - \pi_{\text{obs}})^2}{\sigma_\pi^2} +
        \log(2\pi\sigma_\pi^2) \right]
    """
    parallaxes = np.asarray(parallaxes)

    if np.isfinite(p_meas) and np.isfinite(p_err) and p_err > 0:
        chi2 = (parallaxes - p_meas) ** 2 / p_err**2
        log_norm = np.log(2.0 * np.pi * p_err**2)
        logp = -0.5 * (chi2 + log_norm)
    else:
        logp = np.zeros_like(parallaxes, dtype=float)

    return logp


def logp_parallax_scale(scales, scale_errs, p_meas, p_err, snr_lim=4.0):
    r"""
    Compute the Gaussian log-prior on flux scale factors from a parallax measurement.

    Approximates the distribution of the flux scale factor
    :math:`s = 1 / d_{\text{kpc}}^2 = \pi_{\text{mas}}^2` as a Gaussian when the
    parallax signal-to-noise ratio exceeds ``snr_lim``.

    Parameters
    ----------
    scales : array_like
        Flux density scale factors (:math:`\text{kpc}^{-2}`, equal to
        :math:`\text{mas}^2`).
    scale_errs : array_like
        Uncertainties on ``scales`` (:math:`\text{kpc}^{-2}`), broadcastable
        with ``scales``.
    p_meas : float
        Measured parallax in milliarcseconds (mas).
    p_err : float
        Parallax measurement uncertainty in milliarcseconds (mas).
    snr_lim : float, optional
        Minimum ``|p_meas / p_err|`` required to apply the Gaussian
        approximation. Below this threshold, zeros are returned. Default is 4.0.

    Returns
    -------
    logp : numpy.ndarray
        Log-prior probability density with the shape of ``scales``.

    Notes
    -----
    When ``|p_meas / p_err| > snr_lim``, moments of :math:`s = \pi^2` are
    computed via :func:`convert_parallax_to_scale` and combined in quadrature
    with ``scale_errs``:

    .. math::
        s_{\text{mean}} = \max(0, \pi_{\text{obs}})^2 + \sigma_\pi^2, \quad
        \sigma_s = \sqrt{2\sigma_\pi^4 + 4\max(0, \pi_{\text{obs}})^2\sigma_\pi^2}
    """
    scales = np.asarray(scales)
    scale_errs = np.asarray(scale_errs)

    if (
        np.isfinite(p_meas)
        and np.isfinite(p_err)
        and p_err > 0
        and abs(p_meas / p_err) > snr_lim
    ):
        s_mean, s_std = convert_parallax_to_scale(p_meas, p_err, snr_lim)
        total_var = s_std**2 + scale_errs**2
        chi2 = (scales - s_mean) ** 2 / total_var
        log_norm = np.log(2.0 * np.pi * total_var)
        logp = -0.5 * (chi2 + log_norm)
    else:
        logp = np.zeros_like(scales, dtype=float)

    return logp


def convert_parallax_to_scale(p_meas, p_err, snr_lim=4.0):
    r"""
    Convert a parallax measurement to flux scale-factor moments.

    Transforms a parallax measurement and uncertainty into the mean and
    standard deviation of the scale factor :math:`s = \pi^2` via second-order
    moment propagation.

    Parameters
    ----------
    p_meas : float
        Measured parallax in milliarcseconds (mas).
    p_err : float
        Parallax measurement uncertainty in milliarcseconds (mas).
    snr_lim : float, optional
        Minimum ``|p_meas / p_err|`` required for the Gaussian conversion.
        Below this threshold, uninformative bounds ``(SCALE_MIN, SCALE_MAX)``
        are returned. Default is 4.0.

    Returns
    -------
    s_mean : float
        Mean of the scale factor distribution (:math:`\text{kpc}^{-2}`).
    s_std : float
        Standard deviation of the scale factor distribution
        (:math:`\text{kpc}^{-2}`).

    Notes
    -----
    When ``|p_meas / p_err| > snr_lim``, the moments are

    .. math::
        s_{\text{mean}} = \max(0, \pi_{\text{meas}})^2 + \sigma_\pi^2

        s_{\text{std}} = \sqrt{2\sigma_\pi^4 + 4\max(0, \pi_{\text{meas}})^2\sigma_\pi^2}

    Examples
    --------
    >>> s_mean, s_std = convert_parallax_to_scale(1.0, 0.1)
    >>> print(f"Scale factor: {s_mean:.3f} +/- {s_std:.3f}")
    """
    # Guard p_err > 0 before dividing because p_err == 0 would raise
    # ZeroDivisionError for Python floats or yield inf > snr_lim for numpy
    # scalars instead of returning the uninformative fallback.
    if (
        np.isfinite(p_meas)
        and np.isfinite(p_err)
        and p_err > 0
        and abs(p_meas / p_err) > snr_lim
    ):
        p_positive = max(0.0, p_meas)
        s_mean = p_positive**2 + p_err**2
        s_std = np.sqrt(2 * p_err**4 + 4 * p_positive**2 * p_err**2)
    else:
        s_mean, s_std = SCALE_MIN, SCALE_MAX

    return s_mean, s_std
