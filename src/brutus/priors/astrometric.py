"""
Gaussian likelihood of a measured parallax.

See Also
--------
brutus.priors.galactic : Galactic structure distance priors
brutus.analysis.individual.BruteForce : Uses the parallax likelihood for fitting

Examples
--------
>>> from brutus.priors.astrometric import logp_parallax
>>> import numpy as np
>>>
>>> # Gaia parallax measurement
>>> p_meas = 2.5  # mas
>>> p_err = 0.1   # mas
>>>
>>> # Evaluate the likelihood at model parallaxes
>>> parallaxes = np.linspace(1.0, 4.0, 100)
>>> log_like = logp_parallax(parallaxes, p_meas, p_err)
"""

import numpy as np

__all__ = ["logp_parallax"]


def logp_parallax(parallaxes, p_meas, p_err):
    r"""
    Compute the Gaussian log-likelihood of a measured parallax.

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
        Log-likelihood density (per mas) with the shape of
        ``parallaxes``. Returns zeros when ``p_meas`` or ``p_err`` is
        non-finite or ``p_err <= 0``.

    Notes
    -----
    For valid measurements, the log-likelihood is

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
