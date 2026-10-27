#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Individual star analysis using grid-based Bayesian inference.

This module provides the BruteForce class for fitting stellar parameters,
distances, and extinction using pre-computed model grids. It performs
brute-force Bayesian inference over the entire grid to derive posterior
distributions for stellar properties.

The fitting procedure uses gradient-based optimization to find the maximum
likelihood extinction and distance for each grid point, then computes
Bayesian posteriors incorporating priors on stellar parameters, Galactic
structure, dust maps, and astrometry.

See Also
--------
brutus.core.StarGrid : Pre-computed stellar model grids
brutus.priors : Prior probability functions
brutus.utils.photometry : Photometry utilities

Notes
-----
The module uses numba JIT compilation for performance-critical functions.
The fitting algorithm alternates between optimizing in magnitude and flux
space for numerical stability.

Examples
--------
Basic usage with grid-based fitting:

>>> from brutus.data import load_models
>>> from brutus.core import StarGrid
>>> from brutus.analysis import BruteForce
>>>
>>> # Load pre-computed grid
>>> models, labels, params = load_models('grid_mist_v9.h5')
>>> grid = StarGrid(models, labels, params)
>>>
>>> # Initialize fitter
>>> fitter = BruteForce(grid)
>>>
>>> # Fit photometry with parallax
>>> results = fitter.fit(
...     phot_data, phot_err, phot_mask,
...     data_labels, save_file='results.h5',
...     parallax=parallax, parallax_err=parallax_err,
...     data_coords=coords
... )
"""

import numbers
import os
import sys
import time
import warnings
from functools import partial

import h5py
import numpy as np
from numba import jit, prange
from scipy.special import logsumexp

# Import StarGrid and SED utilities
from ..core import StarGrid
from ..core.sed_utils import _get_seds, _model_array, get_seds
from ..priors.astrometric import logp_parallax, logp_parallax_scale
from ..priors.extinction import logp_extinction
from ..priors.galactic import logp_galactic_structure

# Import refactored prior functions
from ..priors.stellar import logp_imf, logp_ps1_luminosity_function

# Import utility functions
from ..utils.math import AVLIM, LOG_ZERO, R_SOLAR, RVLIM, Z_SOLAR
from ..utils.math import inverse3 as _inverse3
from ..utils.photometry import magnitude
from ..utils.sampling import sample_multivariate_normal

__all__ = ["BruteForce"]

#: Most Gauss-Newton steps per model in each fitting stage.
_MAX_STEPS = 100
#: Largest Levenberg-Marquardt damping before a model counts as stalled.
_MAX_DAMPING = 1e6


# ============================================================================
# Numerical constants
# ============================================================================

# Minimum allowed scale factor to prevent division by zero or log(0).
MIN_SCALE = 1e-20


class _GridLabels:
    # Default of `dlabels`: the labels of the loaded model grid. It differs
    # from `dlabels=None`, which evaluates the Galactic prior without labels.
    def __repr__(self):
        return "<model grid labels>"


_GRID_LABELS = _GridLabels()


def _has_parallax(parallax, parallax_err):
    """Return True for a usable parallax, as the parallax priors define it."""
    return (
        parallax is not None
        and np.isfinite(parallax)
        and np.isfinite(parallax_err)
        and parallax_err > 0
    )


def _clean_bands(data, data_err):
    """Return True for bands with a finite flux and a finite, positive error."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.0)


def _check_shape(name, arr, shape):
    """Raise ValueError naming `name` unless `arr` has shape `shape`."""
    if np.shape(arr) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {np.shape(arr)}")


def _check_threshold(name, value):
    """Raise ValueError naming `name` unless `value` is a real number in (0, 1).

    Models within ``log(value)`` of the best one pass the cut, so 1 would
    select none.
    """
    if not (isinstance(value, numbers.Real) and 0.0 < value < 1.0):
        raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")


def _check_max_models(value):
    """Raise ValueError unless `value` is None or a positive integer."""
    if value is not None and not (isinstance(value, numbers.Integral) and value > 0):
        raise ValueError(
            f"max_models must be None or a positive integer, got {value!r}"
        )


def _check_positive(name, value):
    """Raise ValueError naming `name` unless `value` is a positive finite
    real number."""
    if not (isinstance(value, numbers.Real) and 0.0 < value < np.inf):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


# ============================================================================
# Grid-based optimization functions
# ============================================================================


@jit(nopython=True, cache=True)
def _bounded_step(av, rv, g_av, g_rv, h_aa, h_ar, h_rr, damping, avlim, rvlim):
    """Return the Levenberg-Marquardt step from `(av, rv)` within the bounds,
    and the log-posterior gain the quadratic model predicts for the undamped
    step.

    A parameter at a bound whose gradient points out of the bounds stays
    there, and the step solves for the other one alone. The diagonal of the
    precision is scaled by ``1 + damping``.
    """
    avmin, avmax = avlim
    rvmin, rvmax = rvlim
    free_av = not ((av <= avmin and g_av < 0.0) or (av >= avmax and g_av > 0.0))
    free_rv = not ((rv <= rvmin and g_rv < 0.0) or (rv >= rvmax and g_rv > 0.0))
    scale = 1.0 + damping
    if free_av and free_rv:
        det = h_aa * h_rr - h_ar * h_ar
        gain = (
            0.5
            * (h_rr * g_av * g_av - 2.0 * h_ar * g_av * g_rv + h_aa * g_rv * g_rv)
            / det
        )
        det = h_aa * h_rr * scale * scale - h_ar * h_ar
        d_av = (h_rr * scale * g_av - h_ar * g_rv) / det
        d_rv = (h_aa * scale * g_rv - h_ar * g_av) / det
    elif free_av:
        gain, d_av, d_rv = 0.5 * g_av * g_av / h_aa, g_av / (h_aa * scale), 0.0
    elif free_rv:
        gain, d_av, d_rv = 0.5 * g_rv * g_rv / h_rr, 0.0, g_rv / (h_rr * scale)
    else:
        gain, d_av, d_rv = 0.0, 0.0, 0.0
    av_new = min(max(av + d_av, avmin), avmax)
    rv_new = min(max(rv + d_rv, rvmin), rvmax)
    return av_new, rv_new, gain


@jit(nopython=True, cache=True)
def _mag_objective(coeffs, av, rv, mags, inv_var, av_gauss, rv_gauss):
    """Return chi2 plus the prior penalties of one model in magnitudes, with
    the distance offset profiled out, and its gradient and Gauss-Newton
    precision in `(Av, Rv)` (log-posterior units)."""
    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    Nfilt = len(mags)
    # Weighted sums over bands. dm/dAv = rvec, dm/dRv = Av * dr.
    s_w = s_res = s_rvec = s_dr = 0.0
    for j in range(Nfilt):
        s_w += inv_var[j]
        rvec = coeffs[j, 1] + rv * coeffs[j, 2]
        s_res += (mags[j] - coeffs[j, 0] - av * rvec) * inv_var[j]
        s_rvec += rvec * inv_var[j]
        s_dr += coeffs[j, 2] * inv_var[j]
    offset = s_res / s_w
    chi2 = g_av = g_rv = h_aa = h_ar = h_rr = 0.0
    for j in range(Nfilt):
        rvec = coeffs[j, 1] + rv * coeffs[j, 2]
        resid = mags[j] - coeffs[j, 0] - av * rvec - offset
        chi2 += resid * resid * inv_var[j]
        g_av += rvec * resid * inv_var[j]
        g_rv += coeffs[j, 2] * resid * inv_var[j]
        h_aa += rvec * rvec * inv_var[j]
        h_ar += rvec * coeffs[j, 2] * inv_var[j]
        h_rr += coeffs[j, 2] * coeffs[j, 2] * inv_var[j]
    # Profile the offset out: Schur complement of its block.
    h_aa = h_aa - s_rvec * s_rvec / s_w + 1.0 / Av_std**2
    h_ar = av * (h_ar - s_rvec * s_dr / s_w)
    h_rr = av * av * (h_rr - s_dr * s_dr / s_w) + 1.0 / Rv_std**2
    g_av += (Av_mean - av) / Av_std**2
    g_rv = av * g_rv + (Rv_mean - rv) / Rv_std**2
    obj = chi2 + (((av - Av_mean) / Av_std) ** 2 + ((rv - Rv_mean) / Rv_std) ** 2)
    return obj, g_av, g_rv, h_aa, h_ar, h_rr


@jit(nopython=True, cache=True, parallel=True)
def _optimize_fit_mag(
    data,
    tot_var,
    mag_coeffs,
    av_init,
    rv_init,
    mags,
    mags_var,
    avlim=AVLIM,
    av_gauss=(0.0, 1e6),
    rvlim=RVLIM,
    rv_gauss=(3.32, 0.18),
    tol=1e-3,
):
    """
    Fit `(Av, Rv)` of every model in magnitudes, then evaluate it in fluxes.

    In magnitudes a model is ``m0 + Av * (r0 + Rv * dr)`` plus a distance
    offset, so the fit is nearly linear. Each model takes Gauss-Newton steps
    in `(Av, Rv)` with the offset profiled out (`_bounded_step`). A step
    that raises chi2 plus the prior penalties is retried with more
    Levenberg-Marquardt damping: 1e-3 first, then ten times more each time. A
    model stops when the predicted gain in log-posterior falls below `tol`,
    when a step no longer moves it, when the damping exceeds `_MAX_DAMPING`,
    or after `_MAX_STEPS` steps.

    Parameters
    ----------
    data, tot_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Observed fluxes and their variances.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients `(m0, r0, dr)` of each model.

    av_init, rv_init : `~numpy.ndarray` of shape `(Nmodel,)`
        Starting `(Av, Rv)`, clipped to the bounds. They are not modified.

    mags, mags_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Observed magnitudes and their variances. A band with a huge variance
        has no weight.

    avlim, rvlim : 2-tuple, optional
        Bounds on A(V) and R(V). Defaults are `AVLIM` and `RVLIM`.

    av_gauss, rv_gauss : 2-tuple, optional
        Mean and standard deviation of the Gaussian priors on A(V) and R(V).
        Defaults are `(0, 1e6)` (flat) and `(3.32, 0.18)`.

    tol : float, optional
        Convergence tolerance on the predicted log-posterior gain. Default is
        1e-3.

    Returns
    -------
    models, rvecs, drvecs, scale, av, rv, icov_sar, resid
        The outputs of `_get_sed_mle` at the fitted `(Av, Rv)`, in fluxes.

    converged : `~numpy.ndarray` of shape `(Nmodel,)`
        False for a model that stopped on the damping or step limit.
    """
    Nmodel, Nfilt, _ = mag_coeffs.shape
    avmin, avmax = avlim
    rvmin, rvmax = rvlim
    inv_var = 1.0 / mags_var
    av, rv = np.empty(Nmodel), np.empty(Nmodel)
    converged = np.empty(Nmodel, dtype=np.bool_)
    for i in prange(Nmodel):
        coeffs = mag_coeffs[i]
        a = min(max(av_init[i], avmin), avmax)
        r = min(max(rv_init[i], rvmin), rvmax)
        obj, g_av, g_rv, h_aa, h_ar, h_rr = _mag_objective(
            coeffs, a, r, mags, inv_var, av_gauss, rv_gauss
        )
        damping, done = 0.0, False
        for _ in range(_MAX_STEPS):
            a_new, r_new, gain = _bounded_step(
                a, r, g_av, g_rv, h_aa, h_ar, h_rr, damping, avlim, rvlim
            )
            if gain < tol:
                done = True
                break
            trial = _mag_objective(
                coeffs, a_new, r_new, mags, inv_var, av_gauss, rv_gauss
            )
            if trial[0] <= obj:
                if a_new == a and r_new == r:
                    done = True
                    break
                a, r = a_new, r_new
                obj, g_av, g_rv, h_aa, h_ar, h_rr = trial
                damping *= 0.1
            else:
                damping = max(10.0 * damping, 1e-3)
                if damping > _MAX_DAMPING:
                    break
        av[i], rv[i], converged[i] = a, r, done

    models, rvecs, drvecs, scale, icov_sar, resid = _get_sed_mle(
        data,
        tot_var,
        np.empty((Nmodel, Nfilt)),
        mag_coeffs,
        av,
        rv,
        av_gauss=av_gauss,
        rv_gauss=rv_gauss,
    )
    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid, converged


@jit(nopython=True, cache=True, parallel=True)
def _flux_objective(av, rv, rvecs, drvecs, resid, tot_var, av_gauss, rv_gauss):
    """
    Return chi2, chi2 plus the prior penalties on `(Av, Rv)`, and the gradient
    of the log-posterior in `(Av, Rv)`, from the outputs of `_get_sed_mle`.

    `_get_sed_mle` returns `drvecs` as ``(1 / Av) * df/dRv``, so the R(V)
    gradient multiplies it by `Av`.
    """
    Nmodel, Nfilt = resid.shape
    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    chi2, obj = np.empty(Nmodel), np.empty(Nmodel)
    grad = np.empty((Nmodel, 2))
    for i in prange(Nmodel):
        chi2_i = 0.0
        g_av = (Av_mean - av[i]) / Av_std**2
        g_rv = (Rv_mean - rv[i]) / Rv_std**2
        for j in range(Nfilt):
            chi2_i += resid[i, j] * resid[i, j] / tot_var[j]
            weighted = resid[i, j] / tot_var[j]
            g_av += rvecs[i, j] * weighted
            g_rv += av[i] * drvecs[i, j] * weighted
        chi2[i] = chi2_i
        obj[i] = chi2_i + (
            ((av[i] - Av_mean) / Av_std) ** 2 + ((rv[i] - Rv_mean) / Rv_std) ** 2
        )
        grad[i, 0], grad[i, 1] = g_av, g_rv
    return chi2, obj, grad


@jit(nopython=True, cache=True, parallel=True)
def _newton_step_flux(av, rv, grad, icov_sar, damping, avlim, rvlim):
    """
    Return trial `(Av, Rv)` from one damped Gauss-Newton step with the scale
    profiled out, and the predicted log-posterior gain of the undamped step.

    The step is `_bounded_step` on the 2x2 Schur complement of `icov_sar`
    onto `(Av, Rv)` and `grad` (from `_flux_objective`).
    """
    Nmodel = len(av)
    av_new, rv_new, gain = np.empty(Nmodel), np.empty(Nmodel), np.empty(Nmodel)
    for i in prange(Nmodel):
        g_av, g_rv = grad[i, 0], grad[i, 1]
        prec = icov_sar[i]
        h_aa = prec[1, 1] - prec[0, 1] * prec[0, 1] / prec[0, 0]
        h_ar = prec[1, 2] - prec[0, 1] * prec[0, 2] / prec[0, 0]
        h_rr = prec[2, 2] - prec[0, 2] * prec[0, 2] / prec[0, 0]
        av_new[i], rv_new[i], gain[i] = _bounded_step(
            av[i], rv[i], g_av, g_rv, h_aa, h_ar, h_rr, damping[i], avlim, rvlim
        )
    return av_new, rv_new, gain


@jit(nopython=True, cache=True, parallel=True)
def _get_sed_mle(
    data, tot_var, resid, mag_coeffs, av, rv, av_gauss=(0.0, 1e6), rv_gauss=(3.32, 0.18)
):
    """
    Recompute model SEDs, derive the MLE scale factor, compute residuals,
    and build the Fisher information (precision) matrix.

    Given current `(Av, Rv)` values for each model, this function
    regenerates the reddened SEDs from the magnitude polynomial
    coefficients, computes the maximum-likelihood scale factor
    (equivalently distance) for each model, forms the residuals
    between the scaled model and data, and assembles the 3x3 Fisher
    information (precision) matrix in `(scale, Av, Rv)` space.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Associated (Normal) variance on the observed flux values.
        1D since the variance is a property of the data, not the models.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients used to compute reddened photometry for a given
        model.

    av : `~numpy.ndarray` of shape `(Nmodel,)`
        Av values of the models.

    rv : `~numpy.ndarray` of shape `(Nmodel,)`
        Rv values of the models.

    av_gauss : 2-tuple, optional
        The mean and standard deviation of the Gaussian prior that is placed
        on A(V). The default is `(0., 1e6)`, which is designed to be
        essentially flat over `avlim`.

    rv_gauss : 2-tuple, optional
        The mean and standard deviation of the Gaussian prior that is placed
        on R(V). The default is `(3.32, 0.18)` based on the results from
        Schlafly et al. (2016).

    Returns
    -------
    models_new : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        New model predictions. Always returned in flux densities.

    rvecs_new : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        New reddening vectors. Always returned in flux densities.

    drvecs_new : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        New differential reddening vectors. Always returned in flux densities.

    scale : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit scale factor.

    icov_sar : `~numpy.ndarray` of shape `(Nmodel, 3, 3)`, optional
        The precision (inverse covariance) matrices expanded around
        `(s_ML, Av_ML, Rv_ML)`.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.

    """
    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss

    # Recompute models with new Rv.
    models, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=True)
    Nmodel, Nfilt = models.shape

    # Precompute reciprocal of flux variance (1D, same for all models).
    inv_tot_var = 1.0 / tot_var

    # Derive scale-factors (`scale`) between data and models.
    s_num, s_den, scale = np.zeros(Nmodel), np.zeros(Nmodel), np.zeros(Nmodel)
    for i in prange(Nmodel):
        for j in range(Nfilt):
            s_num[i] += models[i][j] * data[j] * inv_tot_var[j]
            s_den[i] += models[i][j] * models[i][j] * inv_tot_var[j]
        scale[i] = s_num[i] / s_den[i]  # MLE scalefactor
        if scale[i] <= MIN_SCALE:
            scale[i] = MIN_SCALE

    # Derive reddening terms.
    sr_mix, sa_mix = np.zeros(Nmodel), np.zeros(Nmodel)
    a_den, r_den = np.zeros(Nmodel), np.zeros(Nmodel)
    ar_mix = np.zeros(Nmodel)
    Av_varinv, Rv_varinv = 1.0 / Av_std**2, 1.0 / Rv_std**2
    for i in prange(Nmodel):
        for j in range(Nfilt):
            # Rescale models.
            models[i][j] = models[i][j] * scale[i]

            # Compute residuals.
            resid[i][j] = data[j] - models[i][j]

            # Derive scale cross-terms.
            # Note: drvecs = dR/dRv * fac * flux, so the true
            # ∂f/∂Rv = Av * drvecs. We must include Av for correct
            # Fisher information in R(V)-related terms.
            sr_mix[i] += models[i][j] * drvecs[i][j] * av[i] * inv_tot_var[j]
            sa_mix[i] += models[i][j] * rvecs[i][j] * inv_tot_var[j]

            # Rescale reddening quantities.
            rvecs[i][j] = rvecs[i][j] * scale[i]
            drvecs[i][j] = drvecs[i][j] * scale[i]

            # Derive reddening (cross-)terms
            # ar_mix: Gauss-Newton cross-term (∂f/∂Av)(∂f/∂Rv) / var
            # where ∂f/∂Av = rvecs (already scaled) and
            # ∂f/∂Rv = Av * drvecs (already scaled).
            ar_mix[i] += rvecs[i][j] * drvecs[i][j] * av[i] * inv_tot_var[j]
            a_den[i] += rvecs[i][j] * rvecs[i][j] * inv_tot_var[j]
            r_den[i] += (drvecs[i][j] * av[i]) ** 2 * inv_tot_var[j]

        # Add in priors.
        a_den[i] += Av_varinv
        r_den[i] += Rv_varinv

    # Construct precision matrices (inverse covariances). All nine entries are
    # assigned below, so allocation can skip zero-initialization.
    icov_sar = np.empty((Nmodel, 3, 3))
    for i in prange(Nmodel):
        icov_sar[i][0][0] = s_den[i]  # scale
        icov_sar[i][1][1] = a_den[i]  # Av
        icov_sar[i][2][2] = r_den[i]  # Rv
        icov_sar[i][0][1] = sa_mix[i]  # scale-Av cross-term
        icov_sar[i][1][0] = sa_mix[i]  # scale-Av cross-term
        icov_sar[i][0][2] = sr_mix[i]  # scale-Rv cross-term
        icov_sar[i][2][0] = sr_mix[i]  # scale-Rv cross-term
        icov_sar[i][1][2] = ar_mix[i]  # Av-Rv cross-term
        icov_sar[i][2][1] = ar_mix[i]  # Av-Rv cross-term

    return models, rvecs, drvecs, scale, icov_sar, resid


def _inclusion_probabilities(weights, size):
    """Return inclusion probabilities ``min(1, c * weights)`` that sum to `size`.

    `weights` must have more than `size` positive entries. The models with the
    largest weights get probability 1 when ``c * weight`` would exceed 1.
    """
    total = weights.sum()
    if size * weights.max() < total:
        # No model is certain, so no sort is needed.
        return weights * (size / total)
    ws = np.sort(weights)[::-1]
    tail = np.cumsum(ws[::-1])[::-1]  # tail[k] = sum(ws[k:])
    k = np.arange(size)
    # The k largest weights are certain when the next one, scaled by
    # c = (size - k) / tail[k], stays below 1.
    ncert = np.argmax((size - k) * ws[:size] < tail[:size])
    return np.minimum(weights * (size - ncert) / tail[ncert], 1.0)


class BruteForce:
    """
    Bayesian parameter estimation for individual stars using grid-based models.

    This class performs brute-force fitting over a pre-computed stellar model
    grid to estimate stellar parameters, distances, and extinction. It uses
    the StarGrid infrastructure for model management and applies Bayesian
    priors for robust inference.

    Parameters
    ----------
    star_grid : StarGrid
        Pre-loaded stellar model grid for SED generation.

    verbose : bool, optional
        Whether to print initialization information. Default is True.

    Attributes
    ----------
    star_grid : StarGrid
        The underlying stellar model grid.

    models : numpy.ndarray
        Magnitude coefficients from the grid.

    models_labels : structured numpy.ndarray
        Labels for each model in the grid.

    labels_mask : dict
        Mask indicating which labels are grid parameters (True)
        vs predictions (False).

    See Also
    --------
    brutus.core.StarGrid : Stellar model grid infrastructure
    brutus.data.load_models : Load pre-computed grids
    loglike_grid : Compute log-likelihoods
    logpost_grid : Compute log-posteriors

    Notes
    -----
    The BruteForce fitter uses a two-stage approach:

    1. **Likelihood computation** (`loglike_grid`): Optimizes distance
       and extinction for each grid point to find maximum likelihood

    2. **Posterior computation** (`logpost_grid`): Integrates over
       distance and extinction uncertainty using Monte Carlo, applying
       priors for Galactic structure, dust maps, and astrometry

    The fitter automatically handles:
    - Grid parameter vs. prediction distinction
    - Age weighting for proper sampling
    - Grid spacing corrections
    - Parallax constraints
    - Galactic structure priors
    - Dust map priors

    Examples
    --------
    Basic usage with a pre-loaded grid:

    >>> from brutus.core import StarGrid
    >>> from brutus.analysis.individual import BruteForce
    >>> from brutus.data import load_models
    >>>
    >>> # Load grid
    >>> models, labels, params = load_models('grid_mist_v9.h5')
    >>> grid = StarGrid(models, labels, params)
    >>>
    >>> # Initialize fitter
    >>> fitter = BruteForce(grid)
    >>>
    >>> # Fit data
    >>> results = fitter.fit(
    ...     data, data_err, data_mask,
    ...     data_labels, save_file='results.h5',
    ...     parallax=parallax_data,
    ...     data_coords=coordinates
    ... )

    """

    def __init__(self, star_grid, verbose=True):
        """Initialize BruteForce with a StarGrid instance."""
        if not isinstance(star_grid, StarGrid):
            raise TypeError("star_grid must be a StarGrid instance")

        self.star_grid = star_grid
        self.models = star_grid.models
        self.models_labels = star_grid.labels

        # Generate labels mask automatically
        self.labels_mask = self._generate_labels_mask()

        if verbose:
            n_grid = sum(1 for m in self.labels_mask.values() if m)
            n_pred = sum(1 for m in self.labels_mask.values() if not m)
            grid_params = [lbl for lbl, m in self.labels_mask.items() if m]
            pred_params = [lbl for lbl, m in self.labels_mask.items() if not m]

            print(f"BruteForce initialized with {self.nmodels:,} models")
            print(f"  Grid parameters ({n_grid}): {', '.join(grid_params)}")
            if n_pred > 0:
                preview = ", ".join(pred_params[:3])
                if len(pred_params) > 3:
                    preview += f", ... ({len(pred_params)-3} more)"
                print(f"  Predictions ({n_pred}): {preview}")

    @property
    def nmodels(self):
        """Number of models in the grid."""
        return self.models.shape[0]

    @property
    def nfilters(self):
        """Number of filters in the grid."""
        return self.models.shape[1]

    def _generate_labels_mask(self):
        """
        Generate labels mask from StarGrid structure.

        Creates a dictionary mapping label names to boolean values indicating
        whether each label is a grid parameter (True) or a derived prediction
        (False). Grid parameters are the dimensions used to construct the grid,
        while predictions are stellar properties interpolated from the grid.

        Returns
        -------
        labels_mask : dict
            Dictionary where keys are label names and values are True for
            grid parameters (e.g., mini, eep, feh) or False for predictions
            (e.g., loga, logt, logg).

        Notes
        -----
        This distinction is important for applying grid spacing corrections
        (only to grid parameters) and for understanding which parameters
        define the grid structure vs. which are interpolated outputs.
        """
        labels_mask = {}

        # Grid parameters (used to compute the grid)
        for label in self.star_grid.label_names:
            labels_mask[label] = True

        # Predictions (derived from grid)
        if self.star_grid.param_names:
            for param in self.star_grid.param_names:
                labels_mask[param] = False

        return labels_mask

    def get_sed_grid(self, indices=None, av=None, rv=None, return_flux=False):
        r"""
        Compute SEDs for multiple grid points simultaneously.

        This is the grid-based batch computation method, distinct from
        StarGrid.get_seds() which handles single star synthesis.

        Parameters
        ----------
        indices : array-like, optional
            Grid indices to compute SEDs for. If None, uses all models.

        av : array-like or float, optional
            A(V) values for each model. If None, defaults to 0.

        rv : array-like or float, optional
            R(V) values for each model. If None, defaults to 3.3.

        return_flux : bool, optional
            If True, return fluxes instead of magnitudes. Default is False.

        Returns
        -------
        seds : numpy.ndarray of shape (Nmodels, Nbands)
            Computed SEDs.

        rvecs : numpy.ndarray of shape (Nmodels, Nbands)
            Reddening vectors.

        drvecs : numpy.ndarray of shape (Nmodels, Nbands)
            Differential reddening vectors with respect to Rv.

        Raises
        ------
        ValueError
            If ``av`` or ``rv`` is an array whose shape is not ``(Nmodels,)``.

        See Also
        --------
        brutus.core.sed_utils.get_seds : Underlying SED computation
        StarGrid.get_seds : Single star SED generation

        Notes
        -----
        This method performs batch SED computation for multiple grid
        points simultaneously, which is more efficient than calling
        `StarGrid.get_seds()` repeatedly.

        The scale factor relates to distance as :math:`s = 1/d^2` where
        d is distance in kpc. The reddening is applied as:

        .. math::
            m(\lambda) = m_0(\lambda) + A_V \cdot [r_0(\lambda) + R_V \cdot dr(\lambda)]

        where :math:`r_0` and :math:`dr` are the reddening vector components.
        """
        mag_coeffs = self.models if indices is None else self.models[indices]
        return get_seds(
            mag_coeffs,
            av=av,
            rv=rv,
            return_flux=return_flux,
            return_rvec=True,
            return_drvec=True,
        )

    def _setup(
        self,
        data,
        data_err,
        data_mask,
        phot_offsets=None,
        lnprior=None,
        apply_agewt=True,
        apply_grad=True,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        data_coords=None,
        mag_max=50.0,
        merr_max=0.25,
        R_solar=R_SOLAR,
        Z_solar=Z_SOLAR,
    ):
        """
        Pre-process data and initialize priors for fitting.

        This internal method prepares photometric data for fitting by
        applying quality cuts, photometric offsets, and initializing
        appropriate prior distributions.

        Parameters
        ----------
        data : numpy.ndarray
            Photometric flux densities
        data_err : numpy.ndarray
            Photometric errors
        data_mask : numpy.ndarray
            Initial data quality mask
        apply_agewt : bool
            Whether to apply age weighting to priors
        apply_grad : bool
            Whether to apply grid spacing corrections
        Other parameters
            See fit() method for full parameter descriptions

        Returns
        -------
        tuple
            Processed (data, data_err, data_mask, lnprior, lngalprior, lndustprior)

        Notes
        -----
        This method performs several important setup tasks:

        1. Applies photometric offsets if provided
        2. Filters data based on magnitude and error limits
        3. Initializes stellar priors (IMF or luminosity function)
        4. Applies age gradient weighting for proper sampling
        5. Applies grid spacing corrections
        6. Sets up Galactic structure and dust priors
        """
        # Apply photometric offsets if provided
        if phot_offsets is not None:
            data = data * phot_offsets
            data_err = data_err * phot_offsets

        # Apply magnitude cuts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mags, merr = magnitude(data, data_err)

        # Mask bad data. With _clean_bands this is the final mask, so fit()
        # can tell which objects have no valid bands.
        mask_update = (mags < mag_max) & (merr < merr_max)
        data_mask = data_mask & mask_update & _clean_bands(data, data_err)

        # Initialize prior if not provided
        if lnprior is None:
            # Check for initial mass prior
            if "mini" in self.models_labels.dtype.names:
                lnprior = logp_imf(self.models_labels["mini"])
            # Check for luminosity function prior
            elif "Mr" in self.models_labels.dtype.names:
                lnprior = logp_ps1_luminosity_function(self.models_labels["Mr"])
            else:
                lnprior = np.zeros(self.nmodels)
        else:
            # Defensive copy: the age-weight and grid-gradient corrections
            # below are applied in place and must never mutate the caller's
            # array (re-using it across fit() calls would compound them).
            lnprior = np.array(lnprior, dtype=float)

        # Apply age weighting if requested
        if apply_agewt:
            try:
                lnprior += np.log(np.abs(self.models_labels["agewt"]))
            except (KeyError, ValueError):
                pass

        # Reweight based on grid spacing
        if apply_grad:
            for lbl in self.models_labels.dtype.names:
                label = self.models_labels[lbl]
                if self.labels_mask[lbl]:  # Only for grid parameters
                    ulabel = np.unique(label)
                    if len(ulabel) > 1:
                        # Compute and add gradient
                        lngrad_label = np.log(np.gradient(ulabel))
                        lnprior += np.interp(label, ulabel, lngrad_label)

        # Initialize Galactic prior
        if lngalprior is None and data_coords is None:
            raise ValueError(
                "`data_coords` must be provided if using the "
                "default Galactic model prior."
            )
        if lngalprior is None:
            lngalprior = partial(
                logp_galactic_structure, R_solar=R_solar, Z_solar=Z_solar
            )

        # Initialize dust prior. The dust map is queried by sky position, so
        # fail fast here rather than per-object inside logpost_grid.
        if dustfile is not None and data_coords is None:
            raise ValueError(
                "`data_coords` must be provided when using a dust-map "
                "extinction prior (`dustfile`)."
            )
        if lndustprior is None and dustfile is not None:
            lndustprior = logp_extinction

        return (data, data_err, data_mask, lnprior, lngalprior, lndustprior)

    def loglike_grid(
        self,
        data,
        data_err,
        data_mask,
        avlim=AVLIM,
        av_gauss=(0.0, 1e6),
        rvlim=RVLIM,
        rv_gauss=(3.32, 0.18),
        av_init=None,
        rv_init=None,
        dim_prior=True,
        ltol=1e-3,
        init_thresh=1e-6,
        parallax=None,
        parallax_err=None,
        return_vals=False,
        indices=None,
    ):
        """
        Compute log-likelihood over the stellar model grid.

        Optimizes (scale, A(V), R(V)) for each model in the instance's grid
        and evaluates the corresponding maximum-likelihood fit to the data.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape `(Nfilt)`
            Measured flux densities.

        data_err : `~numpy.ndarray` of shape `(Nfilt)`
            Measurement errors.

        data_mask : `~numpy.ndarray` of shape `(Nfilt)`
            Boolean mask of valid bands (or 0/1 values, converted to bool).

        indices : array-like, optional
            Subset of model indices to use. If None, uses all models.

        avlim : tuple, optional
            (min, max) bounds on A(V). Default is `AVLIM`, (0, 20).

        av_gauss : tuple, optional
            (mean, std) for Gaussian prior on A(V). Default is (0.0, 1e6).

        rvlim : tuple, optional
            (min, max) bounds on R(V). Default is `RVLIM`, (1, 8).

        rv_gauss : tuple, optional
            (mean, std) for Gaussian prior on R(V). Default is (3.32, 0.18).

        av_init, rv_init : float or `~numpy.ndarray` of shape `(Nmodels,)`, optional
            Starting A(V) and R(V) of the magnitude-space fit, one value for
            every model or one per model. Default is the mean of `av_gauss`
            and of `rv_gauss`.

        dim_prior : bool, optional
            Subtract ``0.5 * (3 - Ndim) * ln(Ndim)`` from every
            log-likelihood, where ``Ndim`` is the number of valid bands. It
            changes only comparisons between objects with different numbers
            of bands. Default is True.

        ltol : float, optional
            Convergence tolerance of both fits: a model stops when the next
            Gauss-Newton step is predicted to raise its log-posterior in
            (A(V), R(V)) by less than `ltol`. Must be positive. Default is
            1e-3.

        init_thresh : float, optional
            Models whose magnitude-space fit, with the parallax term, is below
            ``init_thresh`` times the best are not refined in flux space and
            keep their magnitude-space fit. Default is 1e-6.

        parallax : float, optional
            Parallax measurement in mas.

        parallax_err : float, optional
            Parallax error in mas.

        return_vals : bool, optional
            If True, also return the fitted parameters and their precision
            matrices. Default is False.

        Returns
        -------
        lnl : `~numpy.ndarray` of shape `(Nmodels,)`
            Log-likelihood of each model at its fit, with the Gaussian
            normalization and, if `dim_prior`, the dimensional term. It does
            not include the parallax.
        Ndim : int
            Number of valid bands.
        chi2 : `~numpy.ndarray` of shape `(Nmodels,)`
            Chi-square of each model at its fit, without the parallax.
        scale : `~numpy.ndarray` of shape `(Nmodels,)`
            Fitted scale, ``1 / d**2`` with ``d`` in kpc. Only with
            ``return_vals=True``, as are the arrays below.
        av, rv : `~numpy.ndarray` of shape `(Nmodels,)`
            Fitted A(V) in mag and R(V).
        icov_sar : `~numpy.ndarray` of shape `(Nmodels, 3, 3)`
            Precision matrix of (scale, A(V), R(V)) at the fit.

        ``Nmodels`` is ``len(indices)`` when `indices` is given. A model below
        `init_thresh` keeps its magnitude-space fit; the others are refined in
        flux space.

        Raises
        ------
        ValueError
            If the shapes of ``data``, ``data_err`` and ``data_mask`` disagree,
            if only one of ``parallax`` and ``parallax_err`` is given, if
            ``init_thresh`` is not in (0, 1), if ``ltol`` is not positive, or
            if no band is valid.

        Warns
        -----
        RuntimeWarning
            If fewer than 4 bands are valid, or if a model's fit stopped on
            the step or damping limit before it converged.

        See Also
        --------
        logpost_grid : Compute log-posteriors from likelihoods
        _optimize_fit_mag : Magnitude-space optimization

        Notes
        -----
        This method optimizes (distance, Av, Rv) for each grid point by:

        1. A magnitude-space fit of every model.
        2. A cut on that fit (``init_thresh``), with the parallax term if a
           parallax is given.
        3. A flux-space refinement of the kept models.

        Both fits take Gauss-Newton steps in (A(V), R(V)) with the distance
        profiled out. A step that raises chi2 plus the prior penalties is
        retried with ten times more Levenberg-Marquardt damping. A model
        stops when the predicted gain is below `ltol`, when a step no longer
        moves it, when the damping passes 1e6, or after 100 steps.

        The optimization uses priors on Av and Rv but not on distance/scale
        (distance priors are applied in logpost_grid).
        """
        # Select models
        if indices is not None:
            mag_coeffs = self.models[indices]
        else:
            mag_coeffs = self.models

        # Implementation of grid-based likelihood computation
        Nmodels, Nfilt, Ncoef = mag_coeffs.shape

        if (parallax is None) != (parallax_err is None):
            raise ValueError("parallax and parallax_err must be given together")
        _check_threshold("init_thresh", init_thresh)
        _check_positive("ltol", ltol)

        # Clean data. A new mask, so the caller's array is left unchanged.
        data_mask = np.asarray(data_mask, dtype=bool)
        for name, arr in [
            ("data", data),
            ("data_err", data_err),
            ("data_mask", data_mask),
        ]:
            _check_shape(name, arr, (Nfilt,))
        data_mask = data_mask & _clean_bands(data, data_err)
        Ndim = int(data_mask.sum())

        if Ndim == 0:
            # Fail fast with a clear message. With zero valid bands the
            # optimizer operates on empty arrays and divides by zero (an opaque
            # ZeroDivisionError under numba), so this would otherwise abort a
            # whole batch with an unhelpful error.
            raise ValueError(
                "No valid photometric bands for this object (all bands are "
                "masked or non-finite). Fitting requires at least 4 valid bands "
                "(3 free parameters scale/A(V)/R(V) plus >=1 DOF); filter such "
                "objects out of the input before fitting."
            )
        if Ndim < 4:
            warnings.warn(
                f"Only {Ndim} valid photometric bands. Minimum 4 recommended "
                f"for reliable fits (3 free parameters: scale, Av, Rv).",
                RuntimeWarning,
                stacklevel=2,
            )

        # Subselect only clean observations
        flux, fluxerr = data[data_mask], data_err[data_mask]
        mcoeffs = mag_coeffs[:, data_mask, :]
        tot_var = np.square(fluxerr)  # 1D: (Nfilt,)

        # Get started by fitting in magnitudes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mags = -2.5 * np.log10(flux)
            mags_var = np.square(2.5 / np.log(10.0)) * tot_var / np.square(flux)
            mclean = np.isfinite(mags)
            mags[~mclean], mags_var[~mclean] = 0.0, 1e50

        # Set default Gaussian priors if not provided
        # These defaults are essentially flat over the allowed range.
        # Default flow: fit(av_gauss=None) -> _fit(av_gauss=(0.0, 1e6))
        # -> loglike_grid(av_gauss=(0.0, 1e6)) -> _get_sed_mle internal
        # default (0.0, 1e6). The large std (1e6) makes the prior
        # effectively flat over the allowed A(V) range.
        if av_gauss is None:
            av_gauss = (0.0, 1e6)
        if rv_gauss is None:
            # When the user explicitly passes None, it means "no R(V) prior
            # in the likelihood computation; priors are applied separately
            # in logpost_grid."  A huge sigma effectively removes the prior.
            rv_gauss = (3.32, 1e6)

        av_init = _model_array(av_init, av_gauss[0], Nmodels, "av_init")
        rv_init = _model_array(rv_init, rv_gauss[0], Nmodels, "rv_init")
        _, rvecs, drvecs, scale, av, rv, icov_sar, resid, converged = _optimize_fit_mag(
            flux,
            tot_var,
            mcoeffs,
            av_init,
            rv_init,
            mags,
            mags_var,
            avlim=avlim,
            av_gauss=av_gauss,
            rvlim=rvlim,
            rv_gauss=rv_gauss,
            tol=ltol,
        )

        # Cull initial bad fits before moving on
        chi2, obj, grad = _flux_objective(
            av, rv, rvecs, drvecs, resid, tot_var, av_gauss, rv_gauss
        )
        lnl = -0.5 * chi2

        # Add parallax to log-likelihood
        lnl_p = lnl
        if _has_parallax(parallax, parallax_err):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                par = np.sqrt(scale)  # sqrt(scale) = 1/d_kpc = parallax_mas
                chi2_p = (par - parallax) ** 2 / parallax_err**2
                lnl_p = lnl - 0.5 * chi2_p

        # Subselect models using log-likelihood thresholding
        lnl_sel = lnl_p > np.max(lnl_p) + np.log(init_thresh)
        init_sel = np.where(lnl_sel)[0]

        # Refine the kept models in flux space, with the same stopping rules
        # as the magnitude-space fit.
        mcoeffs = mcoeffs[init_sel]
        scale_new, icov_sar_new = scale[init_sel], icov_sar[init_sel]
        av_new, rv_new = av[init_sel], rv[init_sel]
        chi2_new, obj, grad = chi2[init_sel], obj[init_sel], grad[init_sel]
        converged = converged[init_sel]
        damping = np.zeros(len(init_sel))
        active = np.arange(len(init_sel))
        for _ in range(_MAX_STEPS):
            av_t, rv_t, gain = _newton_step_flux(
                av_new[active],
                rv_new[active],
                grad[active],
                icov_sar_new[active],
                damping[active],
                avlim,
                rvlim,
            )
            unconverged = gain >= ltol
            active, av_t, rv_t = (
                active[unconverged],
                av_t[unconverged],
                rv_t[unconverged],
            )
            if active.size == 0:
                break
            _, rvecs_t, drvecs_t, scale_t, icov_t, resid_t = _get_sed_mle(
                flux,
                tot_var,
                np.empty((active.size, Ndim)),
                mcoeffs[active],
                av_t,
                rv_t,
                av_gauss=av_gauss,
                rv_gauss=rv_gauss,
            )
            chi2_t, obj_t, grad_t = _flux_objective(
                av_t, rv_t, rvecs_t, drvecs_t, resid_t, tot_var, av_gauss, rv_gauss
            )
            improved = obj_t <= obj[active]
            moved = (av_t != av_new[active]) | (rv_t != rv_new[active])
            accepted = active[improved]
            av_new[accepted], rv_new[accepted] = av_t[improved], rv_t[improved]
            chi2_new[accepted], obj[accepted] = chi2_t[improved], obj_t[improved]
            grad[accepted], icov_sar_new[accepted] = grad_t[improved], icov_t[improved]
            scale_new[accepted] = scale_t[improved]
            damping[accepted] *= 0.1
            rejected = active[~improved]
            damping[rejected] = np.maximum(10.0 * damping[rejected], 1e-3)
            # A model stops when a step no longer moves it, or gives up when
            # the damping grows past _MAX_DAMPING.
            gave_up = ~improved & (damping[active] > _MAX_DAMPING)
            converged[active[gave_up]] = False
            active = active[~((improved & ~moved) | gave_up)]
        converged[active] = False
        if not np.all(converged):
            warnings.warn(
                "Some models stopped before their A(V), R(V) fit converged "
                "(step or damping limit).",
                RuntimeWarning,
                stacklevel=2,
            )

        # Insert optimized models into initial array of results. Culled models
        # keep their magnitude-space fit; the Gaussian normalization applies to
        # every model, so that all of lnl shares one constant.
        lnl[init_sel], chi2[init_sel] = -0.5 * chi2_new, chi2_new
        scale[init_sel], av[init_sel], rv[init_sel] = scale_new, av_new, rv_new
        icov_sar[init_sel] = icov_sar_new
        lnl += -0.5 * (Ndim * np.log(2.0 * np.pi) + np.sum(np.log(tot_var)))

        # Apply dimensional prior
        if dim_prior:
            lnl -= 0.5 * (3.0 - Ndim) * np.log(Ndim)

        if return_vals:
            return lnl, Ndim, chi2, scale, av, rv, icov_sar
        else:
            return lnl, Ndim, chi2

    def logpost_grid(
        self,
        results,
        parallax=None,
        parallax_err=None,
        coord=None,
        Nmc_prior=100,
        lnprior=None,
        wt_thresh=1e-3,
        cdf_thresh=2e-3,
        max_models=50000,
        precision_shrinkage=0.0,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        dlabels=_GRID_LABELS,
        avlim=AVLIM,
        rvlim=RVLIM,
        mem_lim=8000.0,
        rstate=None,
        apply_av_prior=True,
        R_solar=R_SOLAR,
        Z_solar=Z_SOLAR,
        return_precision=False,
    ):
        """
        Compute log-posterior over the stellar model grid.

        Applies model priors, parallax constraints, Galactic structure and
        dust-map priors on top of the grid log-likelihoods, integrating over
        (distance, A(V), R(V)) uncertainty via Monte Carlo sampling.

        Parameters
        ----------
        results : tuple
            The 7-tuple of `loglike_grid` with ``return_vals=True``. If it
            was computed with ``indices``, ``lnprior`` and ``dlabels`` must be
            subset with the same indices, and the returned ``sel`` indexes
            that subset.

        parallax, parallax_err : float, optional
            Parallax and its error, in mas. Both or neither.

        coord : tuple of 2 floats, optional
            Galactic (l, b) of the object, in degrees. The Galactic prior
            applies only when it is given, and the dust-map prior needs it.

        Nmc_prior : int, optional
            Number of Monte Carlo draws per model, lowered if needed to fit
            `mem_lim`. Default is 100.

        lnprior : `~numpy.ndarray` of shape `(Nmodels,)`, optional
            Log-prior of each model. Default is 0 for every model.

        wt_thresh : float or None, optional
            Keep the models whose selection weight is above `wt_thresh`
            times the largest. Default is 1e-3.

        cdf_thresh : float or None, optional
            Used only if `wt_thresh` is None: keep the models of highest
            weight that hold ``1 - cdf_thresh`` of the total. If both are
            None, every model is kept. Default is 2e-3.

        max_models : int or None, optional
            Most models kept after the threshold. Above it, models are kept
            by systematic sampling with probability proportional to their
            selection weight, and ``ln_incl`` holds the log-probabilities.
            None sets no limit. Default is 50000.

        precision_shrinkage : float, optional
            Fraction by which the off-diagonal terms of each (scale, A(V),
            R(V)) precision matrix shrink before the draws. Default is 0.0.

        lngalprior : callable, optional
            Galactic prior ``f(dists, coord, labels=...)``, with ``dists`` in
            kpc. Default is `~brutus.priors.logp_galactic_structure` with
            `R_solar` and `Z_solar`.

        lndustprior : callable, optional
            Dust-map prior ``f(avs, dustmap, coord, distance=...)``. Default
            is `~brutus.priors.logp_extinction`, given `avlim` unless it is a
            `functools.partial` that binds its own.

        dustfile : str, path-like or `~brutus.dust.Bayestar`, optional
            3D dust map for the dust-map prior. A path is loaded as a
            `~brutus.dust.Bayestar` map.

        dlabels : structured `~numpy.ndarray` of shape `(Nmodels,)` or None, optional
            Model labels passed to the Galactic prior. By default, the labels
            of the loaded model grid. With None, the prior gets
            ``labels=None``.

        avlim, rvlim : tuple, optional
            (min, max) bounds on A(V) and R(V). A draw outside them gets zero
            weight. Defaults are `AVLIM`, (0, 20), and `RVLIM`, (1, 8).

        mem_lim : float, optional
            Memory limit for the draws, in MB. Default is 8000.0.

        rstate : `~numpy.random.RandomState` or `~numpy.random.Generator`, optional
            Random state for the draws and the `max_models` sampling.

        apply_av_prior : bool, optional
            Whether to apply the dust-map extinction prior when ``dustfile``
            is provided. Default is True.

        R_solar, Z_solar : float, optional
            Galactocentric radius and height of the Sun, in kpc, for the
            default Galactic prior. Defaults are ``R_SOLAR`` and ``Z_SOLAR``
            (8.275 and 0.0208).

        return_precision : bool, optional
            Return the (scale, A(V), R(V)) precision matrices of the selected
            models in place of their covariances, which skips the inversion.
            Default is False.

        Returns
        -------
        sel : `~numpy.ndarray` of shape `(Nsel,)`
            Indices of the selected models in ``results``.
        cov_sar : `~numpy.ndarray` of shape `(Nsel, 3, 3)`
            Covariances of (scale, A(V), R(V)) of the selected models, or
            their precisions with ``return_precision=True``.
        lnp : `~numpy.ndarray` of shape `(Nsel,)`
            Log-posterior of each selected model.
        dist_mc : `~numpy.ndarray` of shape `(Nsel, Nmc)`
            Monte Carlo distance draws, in kpc.
        a_mc, r_mc : `~numpy.ndarray` of shape `(Nsel, Nmc)`
            Monte Carlo A(V) and R(V) draws.
        lnp_mc : `~numpy.ndarray` of shape `(Nsel, Nmc)`
            Log-posterior weight of each draw: the model's log-likelihood and
            ``lnprior``, the log-distance Jacobian, and the Galactic, dust and
            parallax terms at the draw. To pool draws across models, subtract
            ``ln_incl[:, None]``.
        mc_ess : `~numpy.ndarray` of shape `(Nsel,)`
            ``1 / sum(w**2)`` of the normalized weights of each model's draws.
            The draws are antithetic pairs, so this measures how concentrated
            the weights are, not a count of independent draws.
        ln_incl : `~numpy.ndarray` of shape `(Nsel,)`
            Log-probability that each selected model was kept by the
            ``max_models`` subsampling; 0 when there was none. Use
            ``lnp - ln_incl`` as the log-weights to sum (evidence) or to
            resample models.

        Raises
        ------
        ValueError
            If only one of ``parallax`` and ``parallax_err`` is given, if
            ``dustfile`` is given with ``apply_av_prior`` but without
            ``coord``, if ``max_models`` is neither None nor a positive
            integer, or if ``lnprior`` or ``dlabels`` (with ``coord``) does
            not have one entry per model in ``results``.
        """
        # The dust-map prior is a function of sky position; without one it
        # would crash deep inside the map query with an opaque IndexError.
        if dustfile is not None and apply_av_prior and coord is None:
            raise ValueError(
                "`dustfile` (dust-map extinction prior) requires a sky "
                "position: pass `coord` to logpost_grid (or `data_coords` "
                "to fit())."
            )

        if (parallax is None) != (parallax_err is None):
            raise ValueError("parallax and parallax_err must be given together")
        _check_max_models(max_models)
        if dlabels is _GRID_LABELS:
            dlabels = self.models_labels

        # Implementation of grid-based posterior computation
        # Unpack results (using plural names for consistency with original lnpost)
        lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results
        Nmodels = len(lnlike)
        # dlabels is read only by the Galactic prior, which needs coord.
        checked = [("lnprior", lnprior)]
        if coord is not None:
            checked.append(("dlabels", dlabels))
        for name, arr in checked:
            if arr is not None and np.size(arr) != Nmodels:
                raise ValueError(
                    f"{name} has {np.size(arr)} entries but results has {Nmodels} "
                    f"models. Results from loglike_grid(indices=...) need "
                    f"{name} subset with the same indices."
                )

        # Initialize random state
        if rstate is None:
            rstate = np.random.RandomState()

        # Apply prior
        if lnprior is None:
            lnprior = np.zeros(Nmodels)

        # Compute initial posterior (without parallax — parallax is applied
        # exactly on MC samples below, not via the approximate scale-based
        # form which would double-count the constraint).
        lnprob_base = lnlike + lnprior

        # Add parallax prior for MODEL SELECTION only.
        # logp_parallax_scale provides a Gaussian approximation on scale
        # to help select relevant models. This is NOT propagated to MC
        # sample weights where the exact logp_parallax is used instead.
        lnprob_sel = lnprob_base.copy()
        if parallax is not None and parallax_err is not None:
            # Gate each model on "is its fitted scale consistent with the
            # measured parallax?", treating the photometric scale as a
            # Gaussian with std scales_err. Two stds can be read off the
            # 3x3 (scale, A_V, R_V) precision matrix, and the choice
            # matters:
            #
            # * conditional, 1/sqrt(icov[0,0]): scale uncertainty with
            #   (A_V, R_V) HELD FIXED at their best-fit values;
            # * marginal, sqrt(cov[0,0]): scale uncertainty after
            #   propagating the (A_V, R_V) uncertainties and their
            #   correlation with scale. For two correlated parameters,
            #   marginal variance = conditional variance / (1 - rho^2).
            #
            # Distance and extinction both dim a star, so the scale-A(V)
            # correlation reaches rho ~ 0.95-0.998 for reddened stars and
            # the conditional std understates the truth by 3-16x there.
            # Using it here scored models genuinely ~2 sigma from the
            # parallax as ~20+ sigma away and pruned them -- irreversibly,
            # because this gate only selects which models proceed; the
            # exact parallax likelihood applied to the MC samples later
            # can reweight survivors but cannot resurrect pruned models.
            # The marginal std is therefore the correct choice, and being
            # more permissive here costs only compute, never accuracy.
            #
            # cov[0,0] follows from the cofactor identity
            # cov[0,0] = (i11*i22 - i12^2) / det(icov), so no full matrix
            # inverse is needed. Degenerate matrices (det <= 0, or a
            # nonfinite/nonpositive variance) fall back to the conditional
            # estimate -- the pre-fix behavior -- rather than inventing a
            # number; if even icov[0,0] <= 0, scales_err stays huge and
            # the parallax simply does not prune that model.
            scales_err = np.full(Nmodels, 1e10)  # Large error = uninformative
            i00, i01, i02 = icovs_sar[:, 0, 0], icovs_sar[:, 0, 1], icovs_sar[:, 0, 2]
            i11, i12, i22 = icovs_sar[:, 1, 1], icovs_sar[:, 1, 2], icovs_sar[:, 2, 2]
            minor00 = i11 * i22 - i12 * i12
            det = (
                i00 * minor00
                - i01 * (i01 * i22 - i02 * i12)
                + i02 * (i01 * i12 - i02 * i11)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                var00 = minor00 / det  # marginal scale variance
                cond_var00 = 1.0 / i00  # conditional fallback
            marg_ok = (det > 0) & np.isfinite(var00) & (var00 > 0)
            cond_ok = ~marg_ok & (i00 > 0)  # degenerate matrix: old behavior
            scales_err[marg_ok] = np.sqrt(var00[marg_ok])
            scales_err[cond_ok] = np.sqrt(cond_var00[cond_ok])

            lnprob_sel += logp_parallax_scale(
                scales, scales_err, parallax, parallax_err
            )

        # Select models above threshold (using parallax-informed weights)
        if wt_thresh is not None:
            sel = np.where(lnprob_sel > np.max(lnprob_sel) + np.log(wt_thresh))[0]
        elif cdf_thresh is not None:
            idx_sort = np.argsort(lnprob_sel)[::-1]
            cdf = np.cumsum(np.exp(lnprob_sel[idx_sort] - np.max(lnprob_sel)))
            cdf /= cdf[-1]
            Nsel = np.searchsorted(cdf, 1.0 - cdf_thresh) + 1
            sel = idx_sort[:Nsel]
        else:
            sel = np.arange(Nmodels)

        Nsel = len(sel)

        # Above max_models, keep models by systematic sampling with
        # probability proportional to their selection weight
        # (Horvitz-Thompson). Models of zero weight are dropped first.
        ln_incl = np.zeros(Nsel)
        if max_models is not None and Nsel > max_models:
            lnprob_selected = lnprob_sel[sel]
            w = np.exp(lnprob_selected - lnprob_selected.max())
            sel, w = sel[w > 0], w[w > 0]
            ln_incl = np.zeros(len(sel))
            if len(sel) > max_models:
                pi = _inclusion_probabilities(w, max_models)
                edges = np.concatenate(([0.0], np.cumsum(pi))) - rstate.uniform()
                kept = np.diff(np.floor(edges)) > 0
                sel, ln_incl = sel[kept], np.log(pi[kept])
            Nsel = len(sel)

        # Select precision matrices for the chosen models.
        icovs_selected = icovs_sar[sel]  # Shape: (Nsel, 3, 3)

        # Monte Carlo integration over distance and extinction (VECTORIZED).
        # Floor at 1: a very small mem_lim relative to Nsel can drive the
        # memory-derived cap to 0, which would otherwise yield empty MC draws,
        # log(Nmc) = -inf, and a crash in the downstream reduction.
        Nmc = max(1, min(Nmc_prior, int(mem_lim * 1e6 / (8.0 * Nsel * 4))))

        # Transform PRECISION matrix from (scale, Av, Rv) to (ln d, Av, Rv).
        # This reparameterization avoids the Jacobian bias that arises when
        # sampling in scale-space but evaluating priors in distance-space.
        # The transformation uses eta = ln(d) = -0.5*ln(s), so
        # d(eta)/d(s) = -1/(2s), i.e. J = diag(-1/(2s), 1, 1).
        #
        # The PRECISION matrix transforms as:
        #   icov_lnd = J^{-T} * icov_sar * J^{-1}
        # where J^{-1} = diag(-2s, 1, 1). This gives:
        #   icov_lnd[0,0] = 4*s^2 * icov_sar[0,0]
        #   icov_lnd[0,j] = -2*s * icov_sar[0,j]  for j=1,2
        #   icov_lnd[j,0] = -2*s * icov_sar[j,0]  for j=1,2
        #   icov_lnd[j,k] = icov_sar[j,k]          for j,k in {1,2}
        #
        # This is well-behaved: for small s (distant/unconstrained stars),
        # 4*s^2 * icov_sar[0,0] gets SMALLER (less precision in ln d),
        # which is physically correct. The old approach of transforming the
        # covariance had cov_lnd[0,0] = cov_sar[0,0] / (4*s^2) which
        # explodes for small s, causing MC samples to span many orders of
        # magnitude in distance.
        s_sel = scales[sel]  # Shape: (Nsel,)
        s_sel = np.maximum(s_sel, MIN_SCALE)  # Guard against zero/negative
        two_s = 2.0 * s_sel
        four_s2 = 4.0 * s_sel**2

        icov_lnd = icovs_selected.copy()
        icov_lnd[:, 0, 0] = four_s2 * icovs_selected[:, 0, 0]
        icov_lnd[:, 0, 1] = -two_s * icovs_selected[:, 0, 1]
        icov_lnd[:, 1, 0] = -two_s * icovs_selected[:, 1, 0]
        icov_lnd[:, 0, 2] = -two_s * icovs_selected[:, 0, 2]
        icov_lnd[:, 2, 0] = -two_s * icovs_selected[:, 2, 0]
        # (1,1), (1,2), (2,1), (2,2) are unchanged

        # Optional: apply fixed diagonal shrinkage to the precision matrix
        # before inversion for numerical stability. Small alpha (e.g., 0.05)
        # regularizes the off-diagonal cross-terms which carry the most
        # inversion error at high condition numbers. Default: no shrinkage
        # (alpha=0), relying on the eigenvalue regularization in _inverse3.
        if precision_shrinkage > 0:
            off_diag_mask = ~np.eye(3, dtype=bool)
            icov_lnd[:, off_diag_mask] *= 1.0 - precision_shrinkage

        # Invert precision to get covariance in (ln d, Av, Rv)
        cov_lnd = _inverse3(icov_lnd, regularize=True)

        # Prepare means in (ln d, Av, Rv) space
        eta_sel = -0.5 * np.log(s_sel)  # ln(d) = -0.5*ln(s)
        means = np.column_stack([eta_sel, avs[sel], rvs[sel]])  # Shape: (Nsel, 3)

        # BATCH SAMPLING in (ln d, Av, Rv) space.
        # Antithetic pairs (z, -z) reduce the variance of the MC prior integral
        # (and halve the Gaussian draws) while keeping the estimator unbiased.
        # The brutus integrand is dominated along ln(d) by the galactic-density
        # falloff and the +log(d) volume Jacobian, both monotone (odd) in the
        # sampling direction -- exactly the regime where antithetic variates cut
        # variance. Empirically this gives ~3-5x lower MC-integration std and
        # lower finite-Nmc (Jensen) bias vs a high-Nmc reference. The only regime
        # where antithetic can slightly *increase* variance (never bias) is a
        # very precise parallax centered almost exactly on the photometric
        # proposal mode with poorly-constraining photometry; this corner is
        # empirically negligible for real Gaia data (worst observed std ratio
        # ~1.06).
        samples_all = sample_multivariate_normal(
            means, cov_lnd, size=Nmc, rstate=rstate, antithetic=True
        )
        # samples_all shape: (3, Nmc, Nsel)

        # Extract and transform samples (VECTORIZED)
        eta_samples = samples_all[0]  # Shape: (Nmc, Nsel)
        a_mc = samples_all[1]  # Shape: (Nmc, Nsel)
        r_mc = samples_all[2]  # Shape: (Nmc, Nsel)

        # Convert ln(d) to distance. Draws outside the support are truncated:
        # they get zero weight below. Their values are clipped only so that
        # the priors stay finite and no saved draw leaves the support. A scale
        # clamped to MIN_SCALE gives ln(d) draws that overflow to inf here.
        with np.errstate(over="ignore"):
            dist_raw = np.exp(eta_samples)
        dist_mc = np.clip(dist_raw, 0.001, 1e6)
        outside = dist_mc != dist_raw
        a_clip = np.clip(a_mc, avlim[0], avlim[1])
        outside |= a_clip != a_mc
        r_clip = np.clip(r_mc, rvlim[0], rvlim[1])
        outside |= r_clip != r_mc
        a_mc, r_mc = a_clip, r_clip

        # Initialize log-posterior from base (without parallax scale approx).
        # The exact parallax likelihood is added below via logp_parallax.
        lnp_mc = np.tile(lnprob_base[sel], (Nmc, 1))  # Shape: (Nmc, Nsel)

        # Jacobian correction for ln(d) -> d transformation.
        # The prior pi_d(d) is defined in distance-space. In ln(d)-space,
        # the prior becomes pi_eta(eta) = pi_d(d) * |dd/d(eta)| = pi_d(d) * d.
        # Since logp_galactic_structure already includes the d^2 volume element,
        # the net Jacobian is just d (not d^3 as it would be in scale-space).
        lnp_mc += np.log(dist_mc + 1e-300)

        # Prior evaluations - coordinate is fixed, so we can still optimize
        if coord is not None:
            if lngalprior is None:
                lngalprior = partial(
                    logp_galactic_structure, R_solar=R_solar, Z_solar=Z_solar
                )

            # Galactic prior evaluation for every (MC sample, model) pair.
            # dist_mc has shape (Nmc, Nsel); the labels (feh/loga) of a model
            # are the same for its Nmc samples.
            gal_is_default = lngalprior is logp_galactic_structure or (
                getattr(lngalprior, "func", None) is logp_galactic_structure
            )

            if gal_is_default:
                if dlabels is not None and dlabels.dtype.names is None:
                    raise ValueError(
                        "`dlabels` must be a structured array for the default "
                        "Galactic prior, which reads its 'feh' and 'loga' fields."
                    )
                # Per-model float arrays broadcast over the samples, so the
                # prior computes each model's label terms once.
                names = () if dlabels is None else dlabels.dtype.names
                per_model = {
                    name: dlabels[name][sel].astype(np.float64, copy=False)
                    for name in ("feh", "loga")
                    if name in names
                }
                lnp_gal = lngalprior(dist_mc, coord, **per_model)
            else:
                # Custom prior: the structured labels tiled to match the
                # flattened distances.
                labels_flat = None if dlabels is None else np.tile(dlabels[sel], Nmc)
                lnp_gal = lngalprior(
                    dist_mc.ravel(), coord, labels=labels_flat
                ).reshape(Nmc, Nsel)
            lnp_mc += lnp_gal

        if dustfile is not None and apply_av_prior:
            # Load dust map from file path (str or os.PathLike) if needed
            if isinstance(dustfile, (str, os.PathLike)):
                from ..dust import Bayestar

                dustfile = Bayestar(dustfile=dustfile)

            if lndustprior is None:
                lndustprior = logp_extinction

            # Dust prior evaluation (VECTORIZED)
            # For 3D dust maps, pass distances so the prior can interpolate
            # the extinction profile to each star's distance
            av_flat = a_mc.ravel()  # Shape: (Nmc * Nsel,)
            dist_flat = dist_mc.ravel()  # Shape: (Nmc * Nsel,)
            # The built-in prior, bare or a partial that leaves avlim free,
            # normalizes over the fitted A(V) support; custom priors keep the
            # documented 4-argument signature.
            builtin = lndustprior is logp_extinction or (
                getattr(lndustprior, "func", None) is logp_extinction
                and "avlim" not in lndustprior.keywords
            )
            dust_kw = {"avlim": avlim} if builtin else {}
            lnp_dust_flat = lndustprior(
                av_flat, dustfile, coord, distance=dist_flat, **dust_kw
            )
            lnp_dust_reshaped = lnp_dust_flat.reshape(Nmc, Nsel)  # Shape: (Nmc, Nsel)
            lnp_mc += lnp_dust_reshaped

        # Parallax prior (FULLY VECTORIZED)
        if parallax is not None and parallax_err is not None:
            par_mc = 1.0 / dist_mc  # Shape: (Nmc, Nsel)
            lnp_parallax_all = logp_parallax(par_mc, parallax, parallax_err)
            lnp_mc += lnp_parallax_all

        lnp_mc[outside] = LOG_ZERO

        # Compute integrated posterior (VECTORIZED)
        lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc)

        # Safety check - replace non-finite values with log(0) proxy
        lnp_mask = np.where(~np.isfinite(lnp))[0]
        if len(lnp_mask) > 0:
            lnp[lnp_mask] = LOG_ZERO

        # Compute effective sample size (ESS) for each selected model's
        # MC samples. ESS = 1 / sum(w_i^2) where w_i are normalized weights.
        # VECTORIZED across all Nsel models.
        #
        # Note: the MC draws are generated in antithetic pairs (see the
        # `antithetic=True` sampling call above), so the per-sample weights are
        # not strictly independent. This formula therefore measures WEIGHT
        # CONCENTRATION (proposal/target mismatch) rather than a literal count of
        # independent samples, and can read up to ~2x optimistic for well-mixed
        # models. It remains monotone and useful as a mismatch diagnostic
        # (degenerate weights -> low value), which is how brutus uses it.
        lw_max = np.max(lnp_mc, axis=0, keepdims=True)  # (1, Nsel)
        w = np.exp(lnp_mc - lw_max)  # (Nmc, Nsel)
        w_sum = w.sum(axis=0, keepdims=True)  # (1, Nsel)
        w_normed = w / np.maximum(w_sum, 1e-300)  # (Nmc, Nsel)
        mc_ess = 1.0 / np.sum(w_normed**2, axis=0)  # (Nsel,)
        # A model with no draw of nonzero weight (lw_max = LOG_ZERO) gets 0;
        # its weights above would be spuriously uniform.
        mc_ess = np.where(lw_max.ravel() > LOG_ZERO, mc_ess, 0.0)

        if return_precision:
            matrices = icovs_selected
        else:
            matrices = _inverse3(icovs_selected, regularize=True)
        return (
            sel,
            matrices,
            lnp,
            dist_mc.T,
            a_mc.T,
            r_mc.T,
            lnp_mc.T,
            mc_ess,
            ln_incl,
        )

    def fit(
        self,
        data,
        data_err,
        data_mask,
        data_labels,
        save_file,
        phot_offsets=None,
        parallax=None,
        parallax_err=None,
        Nmc_prior=50,
        avlim=AVLIM,
        av_gauss=None,
        rvlim=RVLIM,
        rv_gauss=(3.32, 0.18),
        lnprior=None,
        wt_thresh=1e-3,
        cdf_thresh=2e-3,
        Ndraws=250,
        apply_agewt=True,
        apply_grad=True,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        apply_dlabels=True,
        data_coords=None,
        logl_dim_prior=True,
        ltol=1e-3,
        logl_initthresh=1e-6,
        mag_max=50.0,
        merr_max=0.25,
        rstate=None,
        save_dar_draws=True,
        running_io=True,
        mem_lim=8000.0,
        max_models=50000,
        precision_shrinkage=0.0,
        verbose=True,
        R_solar=R_SOLAR,
        Z_solar=Z_SOLAR,
    ):
        """
        Fit all input models to the input data to compute log-posteriors.

        This is the main interface for fitting stellar parameters using
        grid-based Bayesian inference. Results are saved to an HDF5 file.

        Parameters
        ----------
        data : numpy.ndarray of shape (Ndata, Nfilt)
            Observed flux densities for each object.

        data_err : numpy.ndarray of shape (Ndata, Nfilt)
            Associated errors on the flux densities.

        data_mask : numpy.ndarray of shape (Ndata, Nfilt)
            Boolean mask of valid measurements (or 0/1 values, converted to
            bool).

        data_labels : numpy.ndarray of shape (Ndata, Nlabels)
            Labels for each object to be stored in the output file.

        save_file : str
            Path to the output HDF5 file. The '.h5' extension will be added
            if not present.

        phot_offsets : numpy.ndarray of shape (Nfilt,), optional
            Multiplicative photometric offsets applied to data and errors.

        parallax : numpy.ndarray of shape (Ndata,), optional
            Parallax measurements in mas for each object.

        parallax_err : numpy.ndarray of shape (Ndata,), optional
            Errors on parallax measurements. Required if parallax is provided.

        Nmc_prior : int, optional
            Number of Monte Carlo samples for prior integration. Default is 50.

        avlim : tuple, optional
            (min, max) bounds on A(V). Default is `AVLIM`, (0, 20).

        av_gauss : tuple, optional
            (mean, std) for a Gaussian prior on A(V) applied during the
            per-model likelihood optimization. Default is None, which maps
            to the essentially-flat `(0.0, 1e6)`. Note this is applied in
            *addition* to any dust-map prior from ``dustfile`` (they are
            not exclusive), so passing an informative ``av_gauss`` together
            with ``dustfile`` counts the A(V) constraint twice.

        rvlim : tuple, optional
            (min, max) bounds on R(V). Default is `RVLIM`, (1, 8).

        rv_gauss : tuple, optional
            (mean, std) for Gaussian prior on R(V). Default is (3.32, 0.18)
            based on Schlafly et al. (2016).

        lnprior : numpy.ndarray of shape (Nmodel,), optional
            Log-prior for each model. If not provided, defaults to Kroupa IMF
            prior on initial mass (for MIST models) or PS1 luminosity function
            prior (for Bayestar models).

        wt_thresh : float, optional
            Threshold `wt_thresh * max(weight)` for model selection.
            Default is 1e-3.

        cdf_thresh : float, optional
            CDF threshold for model selection (used if wt_thresh is None).
            Default is 2e-3.

        Ndraws : int, optional
            Number of posterior draws to save per object. Default is 250.

        apply_agewt : bool, optional
            Whether to apply age weighting from MIST models. Default is True.

        apply_grad : bool, optional
            Whether to apply grid spacing corrections. Default is True.

        lngalprior : callable, optional
            Galactic structure prior function with signature
            ``f(dists, coord, labels=None)``. If not provided, uses
            `~brutus.priors.logp_galactic_structure`. Applied only when
            ``data_coords`` is given.

        lndustprior : callable, optional
            Dust prior function with signature
            ``f(avs, dustmap, coord, distance=None)``. If not provided
            and dustfile is given, uses ``logp_extinction``.

        dustfile : str, path-like, or `~brutus.dust.Bayestar`, optional
            3D dust map for extinction priors. Can be a file path (string
            or `os.PathLike`, e.g. `pathlib.Path`) to a Bayestar HDF5
            file, which will be loaded automatically, or a pre-loaded
            ``Bayestar`` object.

        apply_dlabels : bool, optional
            Whether to pass model labels to the Galactic prior. With False,
            the prior gets ``labels=None``. Default is True.

        data_coords : numpy.ndarray of shape (Ndata, 2), optional
            Galactic (l, b) coordinates for each object in degrees.
            Required if using default Galactic prior.

        logl_dim_prior : bool, optional
            Whether to apply dimensional correction to log-likelihood.
            Default is True.

        ltol : float, optional
            Convergence tolerance of the fits, as in `loglike_grid`. Default
            is 1e-3.

        logl_initthresh : float, optional
            Models whose magnitude-space fit is below this fraction of the
            best are not refined (``init_thresh`` of `loglike_grid`). Default
            is 1e-6.

        mag_max : float, optional
            Maximum allowed magnitude for valid data. Default is 50.0.

        merr_max : float, optional
            Maximum allowed magnitude error. Default is 0.25.

        rstate : numpy.random.RandomState, optional
            Random state for reproducibility.

        save_dar_draws : bool, optional
            Whether to save distance, A(V), and R(V) draws. Default is True.

        running_io : bool, optional
            If True, writes results to disk after each object (safer for long
            runs). If False, accumulates results in memory and writes at end
            (faster for slow filesystems). Default is True.

        mem_lim : float, optional
            Memory limit in MB for Monte Carlo sampling. Default is 8000.0.

        verbose : bool, optional
            Whether to print progress to stderr. Default is True.

        max_models : int, optional
            When more models than this are selected for an object, about
            `max_models` of them are kept at random, each with a probability
            proportional to its weight (at most 1), to bound memory and
            runtime. The evidence and the posterior draws divide each kept
            model's weight by that probability, so they stay unbiased. None
            keeps every selected model. Default is 50000.

        precision_shrinkage : float, optional
            Fractional shrinkage applied to the off-diagonal terms of the 3x3
            (scale, A(V), R(V)) precision matrix before Monte Carlo sampling,
            which stabilizes strongly-correlated fits. 0 disables it; ~0.03 is a
            reasonable nonzero value. Default is 0.0.

        R_solar : float, optional
            Solar galactocentric radius in kpc, used by the Galactic structure
            prior. Default is ``R_SOLAR``, 8.275 (GRAVITY Collaboration 2021).

        Z_solar : float, optional
            Solar height above the Galactic midplane in kpc, used by the
            Galactic structure prior.
            Default is ``Z_SOLAR``, 0.0208 (Bennett & Bovy 2019).

        Returns
        -------
        str
            Path to the output HDF5 file.

        Raises
        ------
        ValueError
            If the shapes of ``data``, ``data_err``, ``data_mask``,
            ``data_labels``, ``parallax``, ``parallax_err``, ``data_coords``
            or ``lnprior`` disagree, if
            only one of ``parallax`` and ``parallax_err`` is given, if
            ``logl_initthresh`` is not in (0, 1), if ``ltol`` is not positive,
            or if ``max_models`` is neither None nor a positive integer. These
            checks run before the output file is created.

        Notes
        -----
        The output HDF5 file contains the following datasets:

        - ``labels``: Object labels (Ndata, Nlabels)
        - ``model_idx``: Resampled model indices (Ndata, Ndraws)
        - ``ml_scale``: Maximum likelihood scale factors (Ndata, Ndraws)
        - ``ml_av``: Maximum likelihood A(V) values (Ndata, Ndraws)
        - ``ml_rv``: Maximum likelihood R(V) values (Ndata, Ndraws)
        - ``ml_cov_sar``: Covariance matrices (Ndata, Ndraws, 3, 3)
        - ``obj_log_post``: Log-posteriors (Ndata, Ndraws)
        - ``obj_log_evid``: Log-evidence per object (Ndata,)
        - ``obj_chi2min``: Minimum chi-squared per object (Ndata,)
        - ``obj_Nbands``: Number of bands used per object (Ndata,)
        - ``mc_ess``: Monte Carlo weight-concentration index (Ndata, Ndraws).
          For each posterior draw, ``1 / sum(w_i^2)`` over the MC integration
          weights for that draw's source model. Low values indicate poor
          overlap between the Gaussian proposal and the target posterior.
          Because the MC draws are antithetic (correlated in pairs) this is a
          concentration diagnostic rather than a strict count of independent
          samples, and may read up to ~2x optimistic for well-mixed models.

        An object with no valid bands is not fitted. Its row has
        ``model_idx = -99``, ``obj_Nbands = 0`` and NaN in every float field,
        and ``fit`` warns once with the number of such objects.

        If save_dar_draws=True, also includes:

        - ``samps_dist``: Distance draws in kpc (Ndata, Ndraws)
        - ``samps_red``: A(V) draws (Ndata, Ndraws)
        - ``samps_dred``: R(V) draws (Ndata, Ndraws)
        - ``samps_logp``: Log-weights for draws (Ndata, Ndraws)

        Examples
        --------
        >>> from brutus.analysis import BruteForce
        >>> from brutus.core import StarGrid
        >>> from brutus.data import load_models
        >>>
        >>> # Load model grid
        >>> models, labels, params = load_models('grid.h5')
        >>> grid = StarGrid(models, labels, params)
        >>> fitter = BruteForce(grid)
        >>>
        >>> # Fit photometry
        >>> results_file = fitter.fit(
        ...     phot, phot_err, phot_mask, obj_labels,
        ...     save_file='results.h5',
        ...     parallax=parallax, parallax_err=parallax_err,
        ...     data_coords=coords
        ... )
        """
        data = np.asarray(data, dtype=float)
        Ndata = len(data)
        data_mask = np.asarray(data_mask, dtype=bool)
        for name, arr in [
            ("data", data),
            ("data_err", data_err),
            ("data_mask", data_mask),
        ]:
            _check_shape(name, arr, (Ndata, self.nfilters))
        if (parallax is None) != (parallax_err is None):
            raise ValueError("parallax and parallax_err must be given together")
        for name, arr, shape in [
            ("parallax", parallax, (Ndata,)),
            ("parallax_err", parallax_err, (Ndata,)),
            ("data_coords", data_coords, (Ndata, 2)),
            ("lnprior", lnprior, (self.nmodels,)),
        ]:
            if arr is not None:
                _check_shape(name, arr, shape)
        if len(data_labels) != Ndata:
            raise ValueError(
                f"data_labels must have {Ndata} rows, one per object, "
                f"got {len(data_labels)}"
            )
        _check_threshold("logl_initthresh", logl_initthresh)
        _check_positive("ltol", ltol)
        _check_max_models(max_models)

        # Load dust map from file path (str or os.PathLike) if needed
        # (once for all objects)
        if dustfile is not None and isinstance(dustfile, (str, os.PathLike)):
            from ..dust import Bayestar

            dustfile = Bayestar(dustfile=dustfile)

        # Pre-process data and initialize priors
        setup_results = self._setup(
            data,
            data_err,
            data_mask,
            phot_offsets=phot_offsets,
            lnprior=lnprior,
            apply_agewt=apply_agewt,
            apply_grad=apply_grad,
            lngalprior=lngalprior,
            lndustprior=lndustprior,
            dustfile=dustfile,
            data_coords=data_coords,
            mag_max=mag_max,
            merr_max=merr_max,
            R_solar=R_solar,
            Z_solar=Z_solar,
        )

        (
            data_proc,
            data_err_proc,
            data_mask_proc,
            lnprior_proc,
            lngalprior_proc,
            lndustprior_proc,
        ) = setup_results

        # Initialize random state
        if rstate is None:
            rstate = np.random.RandomState()

        # Ensure save_file has .h5 extension
        if not save_file.endswith(".h5"):
            save_file = f"{save_file}.h5"

        # Get model labels for priors
        dlabels = self.models_labels if apply_dlabels else None

        # Output fields, in the order _fit returns them:
        # name -> (shape per object, dtype, fill value)
        fields = {
            "model_idx": ((Ndraws,), "int32", -99),
            "ml_scale": ((Ndraws,), "float32", 1.0),
            "ml_av": ((Ndraws,), "float32", 0.0),
            "ml_rv": ((Ndraws,), "float32", 0.0),
            "ml_cov_sar": ((Ndraws, 3, 3), "float32", 0.0),
            "obj_Nbands": ((), "int16", 0),
            "obj_log_post": ((Ndraws,), "float32", 0.0),
            "obj_log_evid": ((), "float32", 0.0),
            "obj_chi2min": ((), "float32", 0.0),
        }
        if save_dar_draws:
            for name in ["samps_dist", "samps_red", "samps_dred", "samps_logp"]:
                fields[name] = ((Ndraws,), "float32", 1.0)
        fields["mc_ess"] = ((Ndraws,), "float32", 0.0)

        # An object with no valid bands is written as not fitted.
        unfitted = dict.fromkeys(fields, np.nan) | {"model_idx": -99, "obj_Nbands": 0}
        nbands_valid = data_mask_proc.sum(axis=1)

        with h5py.File(save_file, "w") as out:
            out.create_dataset("labels", data=data_labels)
            if running_io:
                # Streaming mode: create datasets upfront, write as we go
                store = out
                for name, (shape, dtype, fill) in fields.items():
                    out.create_dataset(
                        name, shape=(Ndata,) + shape, dtype=dtype, fillvalue=fill
                    )
            else:
                # Batch mode: accumulate in memory, write at end
                store = {
                    name: np.full((Ndata,) + shape, fill, dtype=dtype)
                    for name, (shape, dtype, fill) in fields.items()
                }

            # Fit data
            if verbose:
                t1 = time.time()
                t = 0.0
                sys.stderr.write(f"\rFitting object 1/{Ndata}  ")
                sys.stderr.flush()

            for i in range(Ndata):
                if nbands_valid[i] == 0:
                    row = unfitted
                else:
                    results = self._fit(
                        data_proc[i],
                        data_err_proc[i],
                        data_mask_proc[i],
                        parallax=None if parallax is None else parallax[i],
                        parallax_err=None if parallax_err is None else parallax_err[i],
                        coord=None if data_coords is None else data_coords[i],
                        Nmc_prior=Nmc_prior,
                        avlim=avlim,
                        av_gauss=av_gauss,
                        rvlim=rvlim,
                        rv_gauss=rv_gauss,
                        lnprior=lnprior_proc,
                        wt_thresh=wt_thresh,
                        cdf_thresh=cdf_thresh,
                        Ndraws=Ndraws,
                        lngalprior=lngalprior_proc,
                        lndustprior=lndustprior_proc,
                        dustfile=dustfile,
                        dlabels=dlabels,
                        logl_dim_prior=logl_dim_prior,
                        ltol=ltol,
                        logl_initthresh=logl_initthresh,
                        mem_lim=mem_lim,
                        precision_shrinkage=precision_shrinkage,
                        max_models=max_models,
                        rstate=rstate,
                        return_distreds=save_dar_draws,
                        R_solar=R_solar,
                        Z_solar=Z_solar,
                    )
                    row = dict(zip(fields, results, strict=True))

                for name, value in row.items():
                    store[name][i] = value

                # Print progress
                if verbose and i < Ndata - 1:
                    t2 = time.time()
                    dt = t2 - t1
                    t1 = t2
                    t += dt
                    t_avg = t / (i + 1)
                    t_est = t_avg * (Ndata - i - 1)
                    sys.stderr.write(
                        f"\rFitting object {i + 2}/{Ndata} "
                        f"[chi2/n: {row['obj_chi2min']:.1f}/{row['obj_Nbands']}] "
                        f"(mean time: {t_avg:.3f} s/obj, "
                        f"est. remaining: {t_est:.3f} s)    "
                    )
                    sys.stderr.flush()

            # Final progress update
            if verbose:
                t2 = time.time()
                dt = t2 - t1
                t += dt
                t_avg = t / Ndata
                sys.stderr.write(
                    f"\rFitting object {Ndata}/{Ndata} "
                    f"[chi2/n: {row['obj_chi2min']:.1f}/{row['obj_Nbands']}] "
                    f"(mean time: {t_avg:.3f} s/obj, "
                    f"total: {t:.3f} s)    \n"
                )
                sys.stderr.flush()

            # Dump results to disk if using batch mode
            if not running_io:
                for name, arr in store.items():
                    out.create_dataset(name, data=arr)

        nskipped = int(np.sum(nbands_valid == 0))
        if nskipped:
            warnings.warn(
                f"{nskipped} of {Ndata} objects have no valid bands and were not "
                "fitted: they are written with model_idx = -99 and NaN values.",
                RuntimeWarning,
                stacklevel=2,
            )

        return save_file

    def _fit(
        self,
        data,
        data_err,
        data_mask,
        parallax=None,
        parallax_err=None,
        coord=None,
        Nmc_prior=100,
        avlim=AVLIM,
        av_gauss=(0.0, 1e6),
        rvlim=RVLIM,
        rv_gauss=(3.32, 0.18),
        lnprior=None,
        wt_thresh=1e-3,
        cdf_thresh=2e-3,
        Ndraws=250,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        dlabels=_GRID_LABELS,
        logl_dim_prior=True,
        ltol=1e-3,
        logl_initthresh=1e-6,
        mem_lim=8000.0,
        max_models=50000,
        precision_shrinkage=0.0,
        rstate=None,
        return_distreds=True,
        R_solar=R_SOLAR,
        Z_solar=Z_SOLAR,
    ):
        """
        Perform internal fitting for a single object.

        Parameters
        ----------
        data : numpy.ndarray
            Photometric flux densities for a single object.
        data_err : numpy.ndarray
            Photometric errors.
        data_mask : numpy.ndarray
            Binary mask for valid data.
        parallax : float, optional
            Parallax measurement in mas.
        parallax_err : float, optional
            Parallax error in mas.
        coord : tuple, optional
            Galactic (l, b) coordinates in degrees.
        Nmc_prior : int, optional
            Number of Monte Carlo samples for prior integration.
        avlim : tuple, optional
            (min, max) bounds on A(V).
        av_gauss : tuple, optional
            (mean, std) for Gaussian prior on A(V).
        rvlim : tuple, optional
            (min, max) bounds on R(V).
        rv_gauss : tuple, optional
            (mean, std) for Gaussian prior on R(V).
        lnprior : numpy.ndarray, optional
            Log-prior for each model.
        wt_thresh : float, optional
            Threshold for weight-based model selection.
        cdf_thresh : float, optional
            CDF threshold for model selection.
        Ndraws : int, optional
            Number of posterior draws to return.
        lngalprior : callable, optional
            Galactic structure prior function.
        lndustprior : callable, optional
            Dust prior function.
        dustfile : str, path-like, or `~brutus.dust.Bayestar`, optional
            Path to a 3D dust map file (loaded automatically), or a
            pre-loaded ``Bayestar`` object.
        dlabels : structured numpy.ndarray or None, optional
            Model labels for the Galactic prior, as in `logpost_grid`.
            Default is the labels of the loaded model grid.
        logl_dim_prior : bool, optional
            Whether to apply dimensional prior correction.
        ltol : float, optional
            Convergence tolerance, as in `loglike_grid`.
        logl_initthresh : float, optional
            ``init_thresh`` of `loglike_grid`.
        mem_lim : float, optional
            Memory limit in MB.
        rstate : numpy.random.RandomState, optional
            Random state for reproducibility.
        return_distreds : bool, optional
            Whether to return distance/reddening draws. Default is True.

        Returns
        -------
        tuple
            If return_distreds=True:
                (idxs, scales, avs, rvs, covs_sar, Ndim, lnprob, levid, chi2min,
                 dists, reds, dreds, logwts, mc_ess)
            If return_distreds=False:
                (idxs, scales, avs, rvs, covs_sar, Ndim, lnprob, levid, chi2min,
                 mc_ess)

            Where:
            - idxs: Resampled model indices (Ndraws,)
            - scales: Scale factors for resampled models (Ndraws,)
            - avs: A(V) values for resampled models (Ndraws,)
            - rvs: R(V) values for resampled models (Ndraws,)
            - covs_sar: Covariance matrices (Ndraws, 3, 3)
            - Ndim: Number of photometric bands used
            - lnprob: Log-posteriors for resampled models (Ndraws,)
            - levid: Log-evidence (scalar)
            - chi2min: Minimum chi-squared (scalar)
            - dists: Distance draws in kpc (Ndraws,)
            - reds: A(V) draws (Ndraws,)
            - dreds: R(V) draws (Ndraws,)
            - logwts: Log-weights for draws (Ndraws,)
            - mc_ess: Effective sample size for each draw's source model (Ndraws,)
        """
        # Initialize random state
        if rstate is None:
            rstate = np.random.RandomState()

        # Compute grid likelihoods
        loglike_results = self.loglike_grid(
            data,
            data_err,
            data_mask,
            avlim=avlim,
            av_gauss=av_gauss,
            rvlim=rvlim,
            rv_gauss=rv_gauss,
            dim_prior=logl_dim_prior,
            ltol=ltol,
            init_thresh=logl_initthresh,
            parallax=parallax,
            parallax_err=parallax_err,
            return_vals=True,
        )

        # Unpack likelihood results
        lnlike, Ndim, chi2, scales_all, avs_all, rvs_all, icovs_sar_all = (
            loglike_results
        )

        # Compute grid posteriors
        logpost_results = self.logpost_grid(
            loglike_results,
            parallax=parallax,
            parallax_err=parallax_err,
            coord=coord,
            Nmc_prior=Nmc_prior,
            lnprior=lnprior,
            wt_thresh=wt_thresh,
            cdf_thresh=cdf_thresh,
            lngalprior=lngalprior,
            lndustprior=lndustprior,
            dustfile=dustfile,
            dlabels=dlabels,
            avlim=avlim,
            rvlim=rvlim,
            mem_lim=mem_lim,
            precision_shrinkage=precision_shrinkage,
            max_models=max_models,
            rstate=rstate,
            R_solar=R_solar,
            Z_solar=Z_solar,
            return_precision=True,
        )

        sel, icov_sar_sel, lnp, dist_mc, a_mc, r_mc, lnp_mc, mc_ess_sel, ln_incl = (
            logpost_results
        )
        Nsel = len(sel)

        # Add parallax contribution to chi2 and Ndim if provided
        Ndim_out = Ndim
        chi2_with_par = chi2.copy()
        if _has_parallax(parallax, parallax_err):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                par_pred = np.sqrt(scales_all)  # 1/d_kpc = parallax_mas
                chi2_with_par += (par_pred - parallax) ** 2 / parallax_err**2
            Ndim_out += 1

        # Compute goodness-of-fit metrics (over ALL models, not just posterior-selected)
        chi2min = np.min(chi2_with_par)
        # Models kept by the max_models subsampling count 1 / pi times.
        lnwt = lnp - ln_incl
        levid = logsumexp(lnwt)

        # Resample from posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt = np.exp(lnwt - levid)
            wt_sum = wt.sum()
            if wt_sum > 0:
                wt /= wt_sum
            else:
                wt = np.ones(Nsel) / Nsel
            idxs_local = rstate.choice(Nsel, size=Ndraws, p=wt)
            sidxs = sel[idxs_local]

        # Extract values for resampled models
        scales = scales_all[sidxs]
        avs = avs_all[sidxs]
        rvs = rvs_all[sidxs]
        # Invert each drawn model once.
        drawn, which = np.unique(idxs_local, return_inverse=True)
        covs_sar = _inverse3(icov_sar_sel[drawn], regularize=True)[which]
        lnprob = lnp[idxs_local]
        mc_ess = mc_ess_sel[idxs_local]

        if return_distreds:
            # Draw distance and reddening samples from MC integration
            # For each resampled model, pick one MC sample weighted by its contribution
            dists = np.zeros(Ndraws, dtype="float32")
            reds = np.zeros(Ndraws, dtype="float32")
            dreds = np.zeros(Ndraws, dtype="float32")
            logwts = np.zeros(Ndraws, dtype="float32")

            # Vectorized MC sample drawing using Gumbel-max trick.
            # For each posterior draw, select one MC sample proportional
            # to its weight — equivalent to rstate.choice(p=weights)
            # but fully vectorized across all Ndraws at once.
            all_mc_logwts = lnp_mc[idxs_local]  # (Ndraws, Nmc)
            Nmc_actual = all_mc_logwts.shape[1]
            gumbel_noise = -np.log(
                -np.log(rstate.uniform(size=(Ndraws, Nmc_actual)) + 1e-300) + 1e-300
            )
            imc_all = np.argmax(all_mc_logwts + gumbel_noise, axis=1)
            dists = dist_mc[idxs_local, imc_all].astype("float32")
            reds = a_mc[idxs_local, imc_all].astype("float32")
            dreds = r_mc[idxs_local, imc_all].astype("float32")
            logwts = all_mc_logwts[np.arange(Ndraws), imc_all].astype("float32")

            return (
                sidxs,
                scales,
                avs,
                rvs,
                covs_sar,
                Ndim_out,
                lnprob,
                levid,
                chi2min,
                dists,
                reds,
                dreds,
                logwts,
                mc_ess,
            )
        else:
            return (
                sidxs,
                scales,
                avs,
                rvs,
                covs_sar,
                Ndim_out,
                lnprob,
                levid,
                chi2min,
                mc_ess,
            )

    def __repr__(self):
        """Return string representation of BruteForce object."""
        return (
            f"BruteForce(nmodels={self.nmodels:,}, "
            f"nfilters={self.nfilters}, "
            f"labels={len(self.labels_mask)})"
        )
