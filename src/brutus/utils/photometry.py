#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Photometric utility functions for brutus.

This module contains functions for converting between magnitudes and fluxes,
handling asinh magnitudes ("luptitudes"), and computing photometric likelihoods.
These utilities are fundamental for all photometric analysis in brutus.

Functions
---------
magnitude : Convert flux to magnitudes
    AB magnitude system conversion
inv_magnitude : Convert magnitudes to flux
    Inverse of magnitude conversion
luptitude : Convert flux to asinh magnitudes
    Asinh magnitude system (Lupton et al. 1999)
inv_luptitude : Convert asinh magnitudes to flux
    Inverse of luptitude conversion
add_mag : Combine magnitudes
    Add fluxes in magnitude space
phot_loglike : Photometric log-likelihood
    Core likelihood function for stellar fitting
chisquare_outlier_loglike : Chi-square outlier model
    Outlier likelihood for mixture models
uniform_outlier_loglike : Uniform outlier model
    Alternative outlier likelihood

See Also
--------
brutus.analysis.individual.BruteForce : Uses phot_loglike for fitting
brutus.analysis.populations : Uses outlier models for mixture fitting
brutus.core.sed_utils : SED generation functions

Notes
-----
The photometric likelihood function `phot_loglike` is the core of brutus's
Bayesian inference machinery. It supports:

- Multi-filter photometry with flexible masking
- Optional dimensionality prior (chi-square distribution)
- Degrees of freedom reduction for fitted parameters

The asinh magnitude system (luptitudes) provides better behavior than
standard magnitudes for faint sources with high noise, preventing
negative flux issues.

Examples
--------
Basic magnitude conversion:

>>> import numpy as np
>>> from brutus.utils.photometry import magnitude, inv_magnitude
>>>
>>> # Convert flux to magnitude
>>> flux = np.array([[100, 200]])  # arbitrary flux units
>>> flux_err = np.array([[10, 20]])
>>> mags, mag_errs = magnitude(flux, flux_err)
>>>
>>> # Convert back
>>> flux_recovered, flux_err_recovered = inv_magnitude(mags, mag_errs)

Photometric likelihood:

>>> from brutus.utils.photometry import phot_loglike
>>>
>>> # Observed photometry
>>> obs_flux = np.array([[1.0, 2.0, 3.0]])  # 1 object, 3 filters
>>> obs_err = np.array([[0.1, 0.2, 0.3]])
>>>
>>> # Model predictions for 10 models
>>> model_flux = np.random.uniform(0.5, 4.0, (1, 10, 3))
>>>
>>> # Compute likelihoods
>>> lnl = phot_loglike(obs_flux, obs_err, model_flux)
>>> # lnl.shape is (1, 10) - likelihood for each model
"""

import numpy as np
from scipy.special import gammaln, xlogy

__all__ = [
    "magnitude",
    "inv_magnitude",
    "luptitude",
    "inv_luptitude",
    "add_mag",
    "phot_loglike",
    "chisquare_outlier_loglike",
    "uniform_outlier_loglike",
]


def magnitude(phot, err, zeropoints=1.0):
    """
    Convert photometry to AB magnitudes.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes errors corresponding to input `err`.

    """
    # Compute magnitudes.
    mag = -2.5 * np.log10(phot / zeropoints)

    # Compute errors.
    mag_err = 2.5 / np.log(10.0) * err / phot

    return mag, mag_err


def inv_magnitude(mag, err, zeropoints=1.0):
    """
    Convert AB magnitudes to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitude errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """
    # Compute magnitudes.
    phot = 10 ** (-0.4 * mag) * zeropoints

    # Compute errors.
    phot_err = err * 0.4 * np.log(10.0) * phot

    return phot, phot_err


def luptitude(phot, err, skynoise=1.0, zeropoints=1.0):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). See Lupton et.

    al. (1999) for more details.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    lupt : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes corresponding to input `phot`.

    lupt_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes errors corresponding to input `err`.

    """
    # Normalize photometry.
    f = phot / zeropoints
    df = err / zeropoints
    b = skynoise / zeropoints

    # Compute luptitudes.
    lupt = -2.5 / np.log(10.0) * (np.arcsinh(0.5 * f / b) + np.log(b))

    # Compute errors.
    lupt_err = 2.5 / np.log(10.0) * df / (2.0 * b * np.sqrt(1.0 + (0.5 * f / b) ** 2))

    return lupt, lupt_err


def inv_luptitude(lupt, err, skynoise=1.0, zeropoints=1.0):
    """
    Convert asinh magnitudes (Luptitudes) to photometry.

    See Lupton et al. (1999) for more details.

    Parameters
    ----------
    lupt : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Luptitude errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `lupt`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """
    # Normalize sky noise.
    b = skynoise / zeropoints

    # Compute flux.
    f = 2.0 * b * np.sinh(-0.4 * np.log(10.0) * lupt - np.log(b))

    # Compute errors.
    df = (
        err
        * 0.4
        * np.log(10.0)
        * 2.0
        * b
        * np.cosh(-0.4 * np.log(10.0) * lupt - np.log(b))
    )

    # Convert back to original units.
    phot = f * zeropoints
    phot_err = df * zeropoints

    return phot, phot_err


def add_mag(mag1, mag2):
    """
    Add magnitudes.

    Parameters
    ----------
    mag1 : float or `~numpy.ndarray`
        First set of magnitudes.

    mag2 : float or `~numpy.ndarray`
        Second set of magnitudes.

    Returns
    -------
    mag_combined : float or `~numpy.ndarray`
        Combined magnitudes corresponding to the combined flux from
        `mag1` and `mag2`.

    """
    # Compute combined flux.
    flux_combined = 10 ** (-0.4 * mag1) + 10 ** (-0.4 * mag2)

    # Convert back to magnitudes.
    mag_combined = -2.5 * np.log10(flux_combined)

    return mag_combined


def phot_loglike(flux, err, mfluxes, mask=None, dim_prior=False, dof_reduction=0):
    r"""
    Compute the log-likelihood between observed and model fluxes.

    Parameters
    ----------
    flux : `~numpy.ndarray` with shape (Nobj, Nfilt)
        Observed flux values.

    err : `~numpy.ndarray` with shape (Nobj, Nfilt)
        Associated flux errors.

    mfluxes : `~numpy.ndarray` with shape (Nobj, Nmod, Nfilt)
        Model fluxes (for each model).

    mask : `~numpy.ndarray` with shape (Nobj, Nfilt), optional
        Binary mask indicating whether each observed band can be
        used (1) or should be skipped (0).

    dim_prior : bool, optional
        Whether to apply a dimensionality prior from the
        chi-squared distribution. Default is `False`.

        .. warning::
           When dim_prior=True, perfect model matches (chi2≈0) can cause
           problematic behavior: +inf likelihood for DOF=1, -inf likelihood
           for DOF≥3. Ensure test data has small but non-zero residuals.

    dof_reduction : int, optional
        Number of degrees of freedom to subtract from the effective DOF
        when using dim_prior=True. This accounts for parameters being
        fitted simultaneously (e.g., scale factors, extinction).
        Default is 0.

    Returns
    -------
    lnl : `~numpy.ndarray` with shape (Nobj, Nmod)
        Log-likelihood values.

    See Also
    --------
    chisquare_outlier_loglike : Outlier model for dim_prior=True
    uniform_outlier_loglike : Outlier model for dim_prior=False
    brutus.analysis.individual.BruteForce.loglike_grid : Uses this function

    Notes
    -----
    The log-likelihood without dimensionality prior is:

    .. math::
        \\ln L = -\\frac{1}{2} \\left[ \\chi^2 + N \\ln(2\\pi) + \\ln|\\Sigma| \\right]

    where :math:`\\chi^2 = \\sum_i (f_i - m_i)^2/\\sigma_i^2` and
    :math:`|\\Sigma|` is the determinant of the covariance matrix.

    With dimensionality prior (dim_prior=True), this becomes the log-PDF
    of a chi-square distribution, which weights models by goodness-of-fit
    relative to the number of degrees of freedom.
    """
    # Initialize values.
    Nobj, Nfilt = flux.shape[:2]
    Nmod = mfluxes.shape[1]

    # Ensure proper dimensions.
    if mfluxes.shape != (Nobj, Nmod, Nfilt):
        raise ValueError("Inconsistent dimensions between flux and mfluxes")

    if mask is None:
        mask = np.ones_like(flux)

    # Sanitize NaN/invalid values in masked bands to prevent NaN propagation.
    # In numpy, NaN * 0 = NaN, so masked bands with NaN flux or error would
    # corrupt the entire computation. Replace with safe placeholders.
    flux = np.where(mask > 0, flux, 0.0)
    err = np.where(mask > 0, err, 1.0)

    # Apply mask to get effective dimensionality per object.
    Ndim = np.sum(mask, axis=1)

    # Compute variance (including errors).
    var = err**2

    # Mask fluxes and model fluxes.
    flux_masked = flux[:, None, :] * mask[:, None, :]  # (Nobj, 1, Nfilt)
    mfluxes_masked = mfluxes * mask[:, None, :]  # (Nobj, Nmod, Nfilt)
    var_masked = var[:, None, :] * mask[:, None, :]  # (Nobj, 1, Nfilt)

    # Compute residuals.
    resid = flux_masked - mfluxes_masked  # (Nobj, Nmod, Nfilt)

    # Compute chi-squared.
    chi2 = np.sum(resid**2 / np.where(var_masked > 0, var_masked, np.inf), axis=2)

    # Compute log-likelihood.
    lnl = -0.5 * chi2

    # Add normalization term.
    log_det_term = np.sum(np.log(var) * mask, axis=1)  # (Nobj,)
    lnl += -0.5 * (Ndim[:, None] * np.log(2.0 * np.pi) + log_det_term[:, None])

    # Apply dimensionality prior if requested.
    if dim_prior:
        # Compute log-pdf of chi2 distribution in a vectorized way.
        # NOTE: chi2=0 causes +inf (DOF=1) or -inf (DOF>=3) - see docstring warning
        dof = Ndim - dof_reduction  # effective degrees of freedom
        a = 0.5 * dof[:, None]  # shape (Nobj, 1)
        mask_valid_dof = dof > 0  # shape (Nobj,)

        # Broadcast chi2 to (Nobj, Nmod)
        lnl_dim = np.full_like(lnl, -np.inf)
        valid_idx = np.where(mask_valid_dof)[0]
        if valid_idx.size > 0:
            chi2_valid = chi2[valid_idx]
            a_valid = a[valid_idx]
            lnl_dim[valid_idx] = (
                xlogy(a_valid - 1.0, chi2_valid)
                - (chi2_valid / 2.0)
                - gammaln(a_valid)
                - (np.log(2.0) * a_valid)
            )

        lnl = lnl_dim

    return lnl


def chisquare_outlier_loglike(
    flux, err, stellar_params=None, parallax=None, parallax_err=None, p_value_cut=1e-5
):
    """
    Compute chi-square based outlier model log-likelihood.

    Uses a chi-square distribution with a p-value cut to model outlier
    probabilities. This is the default outlier model for dim_prior=True.

    Parameters
    ----------
    flux : array-like, shape (Nobj, Nfilt)
        Observed flux values
    err : array-like, shape (Nobj, Nfilt)
        Flux errors
    stellar_params : dict, optional
        Stellar parameters (masses, colors, etc.) - not used in current implementation
        but provided for future stellar-dependent outlier models
    parallax : array-like, shape (Nobj,), optional
        Parallax measurements (mas)
    parallax_err : array-like, shape (Nobj,), optional
        Parallax errors (mas)
    p_value_cut : float, optional
        P-value threshold for outlier definition. Default 1e-5.

    Returns
    -------
    lnl_outlier : array-like, shape (Nobj,)
        Log-likelihood for outlier model for each object
    """
    from scipy.stats import chi2 as chisquare

    flux = np.asarray(flux)
    err = np.asarray(err)

    # Get effective dimensionality per object
    mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    ndim = np.sum(mask, axis=1)  # shape (Nobj,)

    # Add parallax contribution to dimensionality
    if parallax is not None and parallax_err is not None:
        parallax = np.asarray(parallax)
        parallax_err = np.asarray(parallax_err)
        parallax_mask = (
            np.isfinite(parallax) & np.isfinite(parallax_err) & (parallax_err > 0)
        )
        ndim = ndim + parallax_mask.astype(int)

    # Compute chi-square threshold and log-probability
    chi2_threshold = chisquare.ppf(1.0 - p_value_cut, ndim)
    lnl_outlier = chisquare.logpdf(chi2_threshold, ndim)

    return lnl_outlier


def uniform_outlier_loglike(
    flux, err, stellar_params=None, parallax=None, parallax_err=None, sigma_clip=3.0
):
    """
    Compute quasi-uniform outlier model log-likelihood.

    Assumes uniform distribution within +/- sigma_clip * error bounds
    around the data. This is the default outlier model for dim_prior=False.

    Parameters
    ----------
    flux : array-like, shape (Nobj, Nfilt)
        Observed flux values
    err : array-like, shape (Nobj, Nfilt)
        Flux errors
    stellar_params : dict, optional
        Stellar parameters - not used in current implementation
    parallax : array-like, shape (Nobj,), optional
        Parallax measurements (mas)
    parallax_err : array-like, shape (Nobj,), optional
        Parallax errors (mas)
    sigma_clip : float, optional
        Number of sigma for uniform bounds. Default 3.0.

    Returns
    -------
    lnl_outlier : array-like, shape (Nobj,)
        Log-likelihood for outlier model for each object
    """
    flux = np.asarray(flux)
    err = np.asarray(err)

    # Create mask for valid data
    mask = np.isfinite(flux) & np.isfinite(err) & (err > 0)

    # Compute uniform bounds for each filter
    flux_max = np.nanmax(flux + sigma_clip * err, axis=0)  # shape (Nfilt,)
    flux_min = np.nanmin(flux - sigma_clip * err, axis=0)  # shape (Nfilt,)

    # Compute side length for uniform distribution in each filter
    side_lengths = (2.0 * sigma_clip * err) / (
        flux_max - flux_min
    )  # shape (Nobj, Nfilt)

    # Set invalid entries to 1 (no contribution)
    side_lengths = np.where(mask, side_lengths, 1.0)

    # Compute volume (product over valid filters only)
    volume = np.prod(side_lengths * mask + (1.0 * ~mask), axis=1)  # shape (Nobj,)

    # Add parallax contribution if present
    if parallax is not None and parallax_err is not None:
        parallax = np.asarray(parallax)
        parallax_err = np.asarray(parallax_err)
        parallax_mask = (
            np.isfinite(parallax) & np.isfinite(parallax_err) & (parallax_err > 0)
        )

        if np.any(parallax_mask):
            p_max = np.nanmax((parallax + sigma_clip * parallax_err)[parallax_mask])
            p_min = np.nanmin((parallax - sigma_clip * parallax_err)[parallax_mask])
            parallax_side = (2.0 * sigma_clip * parallax_err) / (p_max - p_min)
            volume = np.where(parallax_mask, volume * parallax_side, volume)

    # Compute log-likelihood of uniform distribution
    lnl_outlier = np.log(1.0 / volume)

    # Handle edge cases
    lnl_outlier = np.where(np.isfinite(lnl_outlier), lnl_outlier, -np.inf)

    return lnl_outlier
