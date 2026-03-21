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

Classes
-------
BruteForce : Grid-based stellar parameter estimation
    Performs Bayesian inference over pre-computed stellar model grids
    to estimate stellar parameters, distances, and extinction for
    individual stars. Provides methods for computing log-likelihoods
    and log-posteriors over the grid.

Functions
---------
_optimize_fit_mag : Optimize extinction in magnitude space
_optimize_fit_flux : Optimize extinction in flux space
_get_sed_mle : Compute maximum likelihood SED parameters

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

import sys
import time
import warnings
from math import log

import h5py
import numpy as np
from numba import jit

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

# Import StarGrid and SED utilities
from ..core import StarGrid
from ..core.sed_utils import _get_seds
from ..priors.astrometric import logp_parallax, logp_parallax_scale
from ..priors.extinction import logp_extinction
from ..priors.galactic import logp_galactic_structure

# Import refactored prior functions
from ..priors.stellar import logp_imf, logp_ps1_luminosity_function

# Import utility functions
from ..utils.math import inverse3 as _inverse3
from ..utils.photometry import magnitude
from ..utils.sampling import sample_multivariate_normal

__all__ = ["BruteForce"]


# ============================================================================
# Numerical constants
# ============================================================================

# Proxy for log(0) or negative infinity in log-probability calculations.
# Used instead of -np.inf to avoid numerical issues with arithmetic operations.
LOG_ZERO = -1e300

# Minimum allowed scale factor to prevent division by zero or log(0).
MIN_SCALE = 1e-20

# Small epsilon for numerical stability in matrix operations and comparisons.
EPS_NUMERIC = 1e-10


# ============================================================================
# Grid-based optimization functions
# ============================================================================


@jit(nopython=True, cache=True)
def _optimize_fit_mag(
    data,
    tot_var,
    models,
    rvecs,
    drvecs,
    av,
    rv,
    mag_coeffs,
    resid,
    stepsize,
    mags,
    mags_var,
    avlim=(0.0, 20.0),
    av_gauss=(0.0, 1e6),
    rvlim=(1.0, 8.0),
    rv_gauss=(3.32, 0.18),
    tol=0.05,
    init_thresh=5e-3,
):
    """
    Optimize the distance and reddening between the models and the data using.

    the gradient in **magnitudes**. This executes multiple `(Av, Rv)` updates.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Associated (Normal) variance on the observed flux values.
        1D since the variance is a property of the data, not the models.

    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Model predictions.

    rvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated model reddening vectors.

    drvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated differential model reddening vectors.

    av : `~numpy.ndarray` of shape `(Nmodel,)`
        Av values of the models.

    rv : `~numpy.ndarray` of shape `(Nmodel,)`
        Rv values of the models.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Initial residuals.

    stepsize : `~numpy.ndarray` of shape `(Nmodel)`
        Gradient stepsize.

    mags : `~numpy.ndarray` of shape `(Nfilt)`
        Data in magnitudes.

    mags_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Data variance in magnitudes.

    avlim : 2-tuple, optional
        Bounds on A(V). Default is `(0., 20.)`.

    av_gauss : 2-tuple, optional
        The mean and standard deviation of a Gaussian prior on A(V).
        Default is `(0., 1e6)` (i.e. flat).

    rvlim : 2-tuple, optional
        Bounds on R(V). Default is `(1., 8.)`.

    rv_gauss : 2-tuple, optional
        The mean and standard deviation of a Gaussian prior on R(V).
        Default is `(3.32, 0.18)`.

    tol : float, optional
        The fractional tolerance to convergence. Default is `0.05`.

    init_thresh : float, optional
        The initial fractional tolerance to convergence. Default is `5e-3`.

    Returns
    -------
    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Model flux densities for each model and filter.

    rvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Reddening vectors at a given Av and Rv for all models.

    drvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Differential reddening vectors at a given Av and Rv for all models.

    scale : `~numpy.ndarray` of shape `(Nmodel)`
        Scale-factors (related to distance as s = 1/d^2).

    av : `~numpy.ndarray` of shape `(Nmodel)`
        Optimized A(V) values.

    rv : `~numpy.ndarray` of shape `(Nmodel)`
        Optimized R(V) values.

    icov_sar : `~numpy.ndarray` of shape `(Nmodel, 3, 3)`
        Inverse covariance matrices over `(scale, av, rv)`.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Final residuals between data and optimized models.

    See Also
    --------
    _optimize_fit_flux : Flux-space optimization (single iteration)
    _get_sed_mle : MLE computation for SED parameters

    Notes
    -----
    This function performs iterative optimization in magnitude space by
    solving a linear system for (scale, Av, Rv) at each iteration. The
    magnitude-space formulation is numerically stable but requires
    multiple iterations to converge.

    Convergence is determined by monitoring the fractional change in
    Av and Rv for all well-fitting models (within init_thresh of the
    best fit).

    The optimization alternately solves for Av (at fixed Rv) and Rv
    (at fixed Av) to incorporate priors and bounds on each parameter
    independently.
    """
    Nmodel, Nfilt = models.shape

    avmin, avmax = avlim
    rvmin, rvmax = rvlim

    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    Av_varinv, Rv_varinv = 1.0 / Av_std**2, 1.0 / Rv_std**2

    log_init_thresh = log(init_thresh)

    # Precompute reciprocal of magnitude variance (1D, same for all models).
    inv_mags_var = 1.0 / mags_var

    # In magnitude space, we can solve a linear system
    # explicitly for `(s_ML, Av_ML, r_ML=Av_ML*Rv_ML)`. We opt to
    # solve for Av and Rv in turn to so we can impose priors and bounds
    # on both quantities as well as additional regularization.

    # Compute constants.
    s_den, rp_den = np.zeros(Nmodel), np.zeros(Nmodel)
    srp_mix = np.zeros(Nmodel)
    for i in range(Nmodel):
        for j in range(Nfilt):
            s_den[i] += inv_mags_var[j]
            rp_den[i] += drvecs[i][j] * drvecs[i][j] * inv_mags_var[j]
            srp_mix[i] += drvecs[i][j] * inv_mags_var[j]

    # Main loop.
    # Uses loop fusion to reduce from 5 to 3 inner loops per model,
    # cutting memory traffic by ~40% (verified numerically identical).
    logwt = np.zeros(Nmodel)
    dav, drv = np.zeros(Nmodel), np.zeros(Nmodel)
    while True:
        for i in range(Nmodel):
            # --- Fused pass 1: Av solve ---
            # Read resid[i][j] and rvecs[i][j] once to compute all Av terms.
            a_den_i = Av_varinv
            sa_mix_i = 0.0
            resid_s_i = 0.0
            resid_a_i = (Av_mean - av[i]) * Av_varinv
            for j in range(Nfilt):
                iv = inv_mags_var[j]
                rv_j = rvecs[i][j]
                res_j = resid[i][j]
                riv = rv_j * iv
                a_den_i += rv_j * riv
                sa_mix_i += riv
                resid_s_i += res_j * iv
                resid_a_i += res_j * riv
            # Compute ML solution for Delta_Av.
            sa_idet = 1.0 / (s_den[i] * a_den_i - sa_mix_i * sa_mix_i)
            dav[i] = sa_idet * (s_den[i] * resid_a_i - sa_mix_i * resid_s_i)
            dav[i] = dav[i] * stepsize[i]

            # Prevent Av from sliding off the provided bounds.
            if dav[i] < avmin - av[i]:
                dav[i] = avmin - av[i]
            if dav[i] > avmax - av[i]:
                dav[i] = avmax - av[i]

            # Increment to new Av.
            av[i] = av[i] + dav[i]

            # --- Fused pass 2: Av residual update + Rv solve ---
            # Update residuals from Av change and immediately compute Rv terms.
            resid_s_i = 0.0
            resid_r_i = 0.0
            for j in range(Nfilt):
                resid[i][j] = resid[i][j] - dav[i] * rvecs[i][j]
                iv = inv_mags_var[j]
                resid_s_i += resid[i][j] * iv
                resid_r_i += resid[i][j] * drvecs[i][j] * iv
            # Derive Rv partial derivatives.
            r_den_i = rp_den[i] * av[i] * av[i] + Rv_varinv
            sr_mix_i = srp_mix[i] * av[i]
            resid_r_i = resid_r_i * av[i] + (Rv_mean - rv[i]) * Rv_varinv
            # Compute ML solution for Delta_Rv.
            sr_idet = 1.0 / (s_den[i] * r_den_i - sr_mix_i * sr_mix_i)
            drv[i] = sr_idet * (s_den[i] * resid_r_i - sr_mix_i * resid_s_i)
            drv[i] = drv[i] * stepsize[i]

            # Prevent Rv from sliding off the provided bounds.
            if drv[i] < rvmin - rv[i]:
                drv[i] = rvmin - rv[i]
            if drv[i] > rvmax - rv[i]:
                drv[i] = rvmax - rv[i]

            # Increment to new Rv.
            rv[i] = rv[i] + drv[i]

            # --- Fused pass 3: Rv residual/rvecs update + chi2 ---
            av_drv = av[i] * drv[i]
            chi2_i = 0.0
            for j in range(Nfilt):
                resid[i][j] = resid[i][j] - av_drv * drvecs[i][j]
                rvecs[i][j] = rvecs[i][j] + drv[i] * drvecs[i][j]
                chi2_i += resid[i][j] * resid[i][j] * inv_mags_var[j]
            logwt[i] = -0.5 * chi2_i

        # Find current best-fit model.
        max_logwt = -1e300
        for i in range(Nmodel):
            if logwt[i] > max_logwt:
                max_logwt = logwt[i]

        # Find relative tolerance (error) to determine convergance.
        err = -1e300
        for i in range(Nmodel):
            # Only include models that are "reasonably good" fits.
            if logwt[i] > max_logwt + log_init_thresh:
                dav_err, drv_err = abs(dav[i]), abs(drv[i])
                if dav_err > err:
                    err = dav_err
                if drv_err > err:
                    err = drv_err

        # Check convergence.
        if err < tol:
            break

    # Get MLE models and associated quantities.
    (models, rvecs, drvecs, scale, icov_sar, resid) = _get_sed_mle(
        data, tot_var, resid, mag_coeffs, av, rv, av_gauss=av_gauss, rv_gauss=rv_gauss
    )

    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid


@jit(nopython=True, cache=True)
def _optimize_fit_flux(
    data,
    tot_var,
    models,
    rvecs,
    drvecs,
    av,
    rv,
    mag_coeffs,
    resid,
    stepsize,
    avlim=(0.0, 20.0),
    av_gauss=(0.0, 1e6),
    rvlim=(1.0, 8.0),
    rv_gauss=(3.32, 0.18),
):
    """
    Optimize distance and reddening using flux densities gradient (single update).

    This executes **only one** `(Av, Rv)` update using the gradient
    in **flux densities**.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nfilt,)`
        Associated (Normal) variance on the observed flux values.
        1D since the variance is a property of the data, not the models.

    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Model predictions.

    rvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated model reddening vectors.

    drvecs : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated differential model reddening vectors.

    av : `~numpy.ndarray` of shape `(Nmodel,)`
        Av values of the models.

    rv : `~numpy.ndarray` of shape `(Nmodel,)`
        Rv values of the models.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients used to compute reddened photometry for a given
        model.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.

    stepsize : `~numpy.ndarray`
        The stepsize (in units of the computed gradient).

    avlim : 2-tuple, optional
        The lower and upper bound where the reddened photometry is reliable.
        Default is `(0., 20.)`.

    av_gauss : 2-tuple, optional
        The mean and standard deviation of the Gaussian prior that is placed
        on A(V). The default is `(0., 1e6)`, which is designed to be
        essentially flat over `avlim`.

    rvlim : 2-tuple, optional
        The lower and upper bound where the reddening vector shape changes
        are reliable. Default is `(1., 8.)`.

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

    Av : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddening.

    Rv : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddening shapes.

    icov_sar : `~numpy.ndarray` of shape `(Nmodel, 3, 3)`, optional
        The precision (inverse covariance) matrices expanded around
        `(s_ML, Av_ML, Rv_ML)`.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.

    See Also
    --------
    _optimize_fit_mag : Magnitude-space optimization (iterative)
    _get_sed_mle : MLE computation for SED parameters

    Notes
    -----
    This function performs a single update step in flux space using
    gradient descent. It is called iteratively by the main fitting
    routine until convergence.

    Unlike magnitude-space fitting, flux-space optimization uses a
    Taylor expansion which can be less numerically stable for large
    extinctions but is faster per iteration.
    """
    Nmodel, Nfilt = models.shape

    avmin, avmax = avlim
    rvmin, rvmax = rvlim

    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    Av_varinv, Rv_varinv = 1.0 / Av_std**2, 1.0 / Rv_std**2

    # Precompute reciprocal of flux variance (1D, same for all models).
    inv_tot_var = 1.0 / tot_var

    # In flux density space, we can solve the linear system
    # implicitly for `(s_ML, Av_ML, Rv_ML)`. However, the solution
    # is not necessarily as numerically stable as one might hope
    # due to the nature of our Taylor expansion in flux.
    # Instead, it is easier to iterate in `(dAv, dRv)` from
    # a good guess for `(s_ML, Av_ML, Rv_ML)`. We opt to solve both
    # independently at fixed `(Av, Rv)` to avoid recomputing models.

    # Fused: compute Av and Rv sums in a single pass over filters,
    # since both read the same resid[i][j], rvecs[i][j], drvecs[i][j].
    for i in range(Nmodel):
        a_num_i = (Av_mean - av[i]) * Av_varinv
        a_den_i = Av_varinv
        r_num_i = (Rv_mean - rv[i]) * Rv_varinv
        r_den_i = Rv_varinv
        for j in range(Nfilt):
            iv = inv_tot_var[j]
            rv_j = rvecs[i][j]
            drv_j = drvecs[i][j]
            res_j = resid[i][j]
            a_num_i += rv_j * res_j * iv
            a_den_i += rv_j * rv_j * iv
            r_num_i += drv_j * res_j * iv
            r_den_i += drv_j * drv_j * iv
        dav_i = a_num_i / a_den_i * stepsize[i]
        drv_i = r_num_i / r_den_i * stepsize[i]

        # Prevent Av from sliding off the provided bounds.
        if dav_i < avmin - av[i]:
            dav_i = avmin - av[i]
        if dav_i > avmax - av[i]:
            dav_i = avmax - av[i]

        # Increment to new Av.
        av[i] += dav_i

        # Prevent Rv from sliding off the provided bounds.
        if drv_i < rvmin - rv[i]:
            drv_i = rvmin - rv[i]
        if drv_i > rvmax - rv[i]:
            drv_i = rvmax - rv[i]

        # Increment to new Rv.
        rv[i] += drv_i

    # Get MLE models and associated quantities.
    (models, rvecs, drvecs, scale, icov_sar, resid) = _get_sed_mle(
        data, tot_var, resid, mag_coeffs, av, rv, av_gauss=av_gauss, rv_gauss=rv_gauss
    )

    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid


@jit(nopython=True, cache=True)
def _get_sed_mle(
    data, tot_var, resid, mag_coeffs, av, rv, av_gauss=(0.0, 1e6), rv_gauss=(3.32, 0.18)
):
    """
    Optimize the distance and reddening between the models and the data using.

    the gradient in **flux densities**. This executes **only one**
    `(Av, Rv)` update.

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
    for i in range(Nmodel):
        for j in range(Nfilt):
            s_num[i] += models[i][j] * data[j] * inv_tot_var[j]
            s_den[i] += models[i][j] * models[i][j] * inv_tot_var[j]
        scale[i] = s_num[i] / s_den[i]  # MLE scalefactor
        if scale[i] <= 1e-20:
            scale[i] = 1e-20

    # Derive reddening terms.
    sr_mix, sa_mix = np.zeros(Nmodel), np.zeros(Nmodel)
    a_den, r_den = np.zeros(Nmodel), np.zeros(Nmodel)
    ar_mix = np.zeros(Nmodel)
    Av_varinv, Rv_varinv = 1.0 / Av_std**2, 1.0 / Rv_std**2
    for i in range(Nmodel):
        for j in range(Nfilt):
            # Compute reddening effect.
            models_int = 10.0 ** (-0.4 * mag_coeffs[i][j][0])
            reddening = models[i][j] - models_int

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
            reddening *= scale[i]

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

    # Construct precision matrices (inverse covariances).
    icov_sar = np.zeros((Nmodel, 3, 3))
    for i in range(Nmodel):
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

        See Also
        --------
        brutus.core.sed_utils._get_seds : Underlying SED computation
        StarGrid.get_seds : Single star SED generation

        Notes
        -----
        This method performs batch SED computation for multiple grid
        points simultaneously, which is more efficient than calling
        `StarGrid.get_seds()` repeatedly.

        The scale factor relates to distance as :math:`s = 1/d^2` where
        d is distance in parsecs. The reddening is applied as:

        .. math::
            m(\\lambda) = m_0(\\lambda) + A_V \\cdot [r_0(\\lambda) + R_V \\cdot dr(\\lambda)]

        where :math:`r_0` and :math:`dr` are the reddening vector components.
        """
        if indices is not None:
            mag_coeffs = self.models[indices]
        else:
            mag_coeffs = self.models

        # Ensure av and rv are arrays
        n_models = len(mag_coeffs)

        if av is None:
            av = np.zeros(n_models)
        elif np.isscalar(av):
            av = np.full(n_models, av)
        else:
            av = np.asarray(av)

        if rv is None:
            rv = np.full(n_models, 3.3)
        elif np.isscalar(rv):
            rv = np.full(n_models, rv)
        else:
            rv = np.asarray(rv)

        return _get_seds(mag_coeffs, av, rv, return_flux=return_flux)

    def _setup(
        self,
        data,
        data_err,
        data_mask,
        data_labels=None,
        phot_offsets=None,
        parallax=None,
        parallax_err=None,
        av_gauss=None,
        lnprior=None,
        wt_thresh=1e-3,
        cdf_thresh=2e-3,
        apply_agewt=True,
        apply_grad=True,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        data_coords=None,
        ltol_subthresh=1e-2,
        logl_initthresh=5e-3,
        mag_max=50.0,
        merr_max=0.25,
        rstate=None,
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

        # Mask bad data
        mask_update = (mags < mag_max) & (merr < merr_max)
        data_mask = data_mask & mask_update

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
            lngalprior = logp_galactic_structure

        # Initialize dust prior
        if lndustprior is None and dustfile is not None:
            lndustprior = logp_extinction

        return (data, data_err, data_mask, lnprior, lngalprior, lndustprior)

    def loglike_grid(
        self,
        data,
        data_err,
        data_mask,
        avlim=(0.0, 20.0),
        av_gauss=(0.0, 1e6),
        rvlim=(1.0, 8.0),
        rv_gauss=(3.32, 0.18),
        av_init=None,
        rv_init=None,
        dim_prior=True,
        ltol=3e-2,
        ltol_subthresh=1e-2,
        init_thresh=5e-3,
        parallax=None,
        parallax_err=None,
        return_vals=False,
        indices=None,
        **kwargs,
    ):
        """
        Compute log-likelihood over the stellar model grid.

        This is a wrapper around the module-level loglike_grid function
        that uses the instance's model grid.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape `(Nfilt)`
            Measured flux densities.

        data_err : `~numpy.ndarray` of shape `(Nfilt)`
            Measurement errors.

        data_mask : `~numpy.ndarray` of shape `(Nfilt)`
            Binary mask for valid data.

        indices : array-like, optional
            Subset of model indices to use. If None, uses all models.

        avlim : tuple, optional
            (min, max) bounds on A(V). Default is (0.0, 20.0).

        av_gauss : tuple, optional
            (mean, std) for Gaussian prior on A(V). Default is (0.0, 1e6).

        rvlim : tuple, optional
            (min, max) bounds on R(V). Default is (1.0, 8.0).

        rv_gauss : tuple, optional
            (mean, std) for Gaussian prior on R(V). Default is (3.32, 0.18).

        parallax : float, optional
            Parallax measurement in mas.

        parallax_err : float, optional
            Parallax error in mas.

        return_vals : bool, optional
            If True, return full results including covariances. Default is False.

        Other Parameters
        ----------------
        **kwargs
            Passed to optimization functions.

        Returns
        -------
        If return_vals=False:
            lnl : numpy.ndarray
                Log-likelihoods for each grid point
            Ndim : int
                Number of dimensions (filters)
            chi2 : numpy.ndarray
                Chi-squared values

        If return_vals=True:
            Also includes scale, av, rv, icov_sar arrays

        See Also
        --------
        logpost_grid : Compute log-posteriors from likelihoods
        _optimize_fit_mag : Magnitude-space optimization
        _optimize_fit_flux : Flux-space optimization

        Notes
        -----
        This method optimizes (distance, Av, Rv) for each grid point by:

        1. Initial magnitude-space fit for numerical stability
        2. Iterative flux-space refinement until convergence
        3. Optional parallax constraint during optimization

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

        # Clean data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.0)
            data_mask[~clean] = False
        Ndim = sum(data_mask)

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
        # These defaults are essentially flat over the allowed range
        if av_gauss is None:
            av_gauss = (0.0, 1e6)
        if rv_gauss is None:
            rv_gauss = (3.32, 1e6)

        # Initialize values
        if av_init is None:
            av_init = np.zeros(Nmodels) + av_gauss[0]
        if rv_init is None:
            rv_init = np.zeros(Nmodels) + rv_gauss[0]

        # Compute unreddened photometry
        models, rvecs, drvecs = _get_seds(mcoeffs, av_init, rv_init, return_flux=False)

        # Compute initial magnitude fit
        mtol = 2.5 * ltol
        resid = mags - models
        stepsize = np.ones(Nmodels)
        results = _optimize_fit_mag(
            flux,
            tot_var,
            models,
            rvecs,
            drvecs,
            av_init,
            rv_init,
            mcoeffs,
            resid,
            stepsize,
            mags,
            mags_var,
            tol=mtol,
            init_thresh=init_thresh,
            avlim=avlim,
            av_gauss=av_gauss,
            rvlim=rvlim,
            rv_gauss=rv_gauss,
        )
        models, rvecs, drvecs, scale, av, rv, icov_sar, resid = results

        if init_thresh is not None:
            # Cull initial bad fits before moving on
            chi2 = np.sum(np.square(resid) / tot_var, axis=1)
            lnl = -0.5 * chi2

            # Add parallax to log-likelihood
            lnl_p = lnl
            if parallax is not None and parallax_err is not None:
                if np.isfinite(parallax) and np.isfinite(parallax_err):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        par = np.sqrt(scale)
                        chi2_p = (par - parallax) ** 2 / parallax_err**2
                        lnl_p = lnl - 0.5 * chi2_p

            # Subselect models using log-likelihood thresholding
            lnl_sel = lnl_p > np.max(lnl_p) + np.log(init_thresh)
            init_sel = np.where(lnl_sel)[0]

            # Subselect models (tot_var is 1D and data-only, no subselection needed)
            models = models[init_sel]
            rvecs = rvecs[init_sel]
            drvecs = drvecs[init_sel]
            av_new = av[init_sel]
            rv_new = rv[init_sel]
            mcoeffs = mcoeffs[init_sel]
            resid = resid[init_sel]
        else:
            # Keep all models
            init_sel = np.arange(Nmodels)
            chi2 = np.ones(Nmodels) - LOG_ZERO  # Large positive value
            lnl = np.ones(Nmodels) + LOG_ZERO  # Large negative value
            av_new = np.array(av, order="F")
            rv_new = np.array(rv, order="F")

        # Iterate until convergence
        lnl_old, lerr = LOG_ZERO, -LOG_ZERO  # -inf, +inf proxies
        stepsize, rescaling = np.ones(Nmodels)[init_sel], 1.2
        ln_ltol_subthresh = np.log(ltol_subthresh)
        while lerr > ltol:
            # Re-compute models
            results = _optimize_fit_flux(
                flux,
                tot_var,
                models,
                rvecs,
                drvecs,
                av_new,
                rv_new,
                mcoeffs,
                resid,
                stepsize,
                avlim=avlim,
                av_gauss=av_gauss,
                rvlim=rvlim,
                rv_gauss=rv_gauss,
            )
            (models, rvecs, drvecs, scale_new, av_new, rv_new, icov_sar_new, resid) = (
                results
            )

            # Compute chi2
            chi2_new = np.sum(np.square(resid) / tot_var, axis=1)

            # Compute multivariate normal logpdf
            lnl_new = -0.5 * chi2_new

            # Compute stopping criterion
            lnl_sel = np.where(lnl_new > np.max(lnl_new) + ln_ltol_subthresh)[0]
            lerr = np.max(np.abs(lnl_new - lnl_old)[lnl_sel])

            # Adjust stepsize
            stepsize[lnl_new < lnl_old] /= rescaling
            lnl_old = lnl_new

        # Insert optimized models into initial array of results
        lnl_new += -0.5 * (Ndim * np.log(2.0 * np.pi) + np.sum(np.log(tot_var)))
        lnl[init_sel], chi2[init_sel] = lnl_new, chi2_new
        scale[init_sel], av[init_sel], rv[init_sel] = scale_new, av_new, rv_new
        icov_sar[init_sel] = icov_sar_new

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
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        dlabels=None,
        avlim=(0.0, 20.0),
        rvlim=(1.0, 8.0),
        mem_lim=8000.0,
        rstate=None,
        apply_av_prior=True,
        **kwargs,
    ):
        """
        Compute log-posterior over the stellar model grid.

        This is a wrapper around the module-level logpost_grid function.

        Parameters
        ----------
        results : tuple
            Results from loglike_grid with return_vals=True.

        Other parameters are passed to logpost_grid.

        Returns
        -------
        Results from logpost_grid.
        """
        # Use instance's labels if not provided
        if dlabels is None:
            dlabels = self.models_labels

        # Implementation of grid-based posterior computation
        # Unpack results (using plural names for consistency with original lnpost)
        lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results
        Nmodels = len(lnlike)

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
            # Convert parallax to scale (VECTORIZED)
            scales_err = np.full(Nmodels, 1e10)  # Large error = uninformative
            valid_mask = icovs_sar[:, 0, 0] > 0
            scales_err[valid_mask] = 1.0 / np.sqrt(icovs_sar[valid_mask, 0, 0])

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

        # Compute covariance from inverse covariance (VECTORIZED)
        icovs_selected = icovs_sar[sel]  # Shape: (Nsel, 3, 3)

        # Batch inversion with diagonal preconditioning for stability.
        # cov_sar is kept in (s, Av, Rv) space for backward compatibility.
        cov_sar = _inverse3(icovs_selected, regularize=True)

        # Monte Carlo integration over distance and extinction (VECTORIZED)
        Nmc = min(Nmc_prior, int(mem_lim * 1e6 / (8.0 * Nsel * 4)))

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
        s_sel = np.maximum(s_sel, 1e-20)  # Guard against zero/negative
        two_s = 2.0 * s_sel
        four_s2 = 4.0 * s_sel**2

        icov_lnd = icovs_selected.copy()
        icov_lnd[:, 0, 0] = four_s2 * icovs_selected[:, 0, 0]
        icov_lnd[:, 0, 1] = -two_s * icovs_selected[:, 0, 1]
        icov_lnd[:, 1, 0] = -two_s * icovs_selected[:, 1, 0]
        icov_lnd[:, 0, 2] = -two_s * icovs_selected[:, 0, 2]
        icov_lnd[:, 2, 0] = -two_s * icovs_selected[:, 2, 0]
        # (1,1), (1,2), (2,1), (2,2) are unchanged

        # Invert the transformed precision to get covariance in (ln d, Av, Rv)
        cov_lnd = _inverse3(icov_lnd, regularize=True)

        # Prepare means in (ln d, Av, Rv) space
        eta_sel = -0.5 * np.log(s_sel)  # ln(d) = -0.5*ln(s)
        means = np.column_stack([eta_sel, avs[sel], rvs[sel]])  # Shape: (Nsel, 3)

        # BATCH SAMPLING in (ln d, Av, Rv) space
        samples_all = sample_multivariate_normal(
            means, cov_lnd, size=Nmc, rstate=rstate
        )
        # samples_all shape: (3, Nmc, Nsel)

        # Extract and transform samples (VECTORIZED)
        eta_samples = samples_all[0]  # Shape: (Nmc, Nsel)
        a_mc = samples_all[1]  # Shape: (Nmc, Nsel)
        r_mc = samples_all[2]  # Shape: (Nmc, Nsel)

        # Convert ln(d) to distance
        dist_mc = np.exp(eta_samples)
        dist_mc = np.clip(dist_mc, 0.001, 1e6)
        a_mc = np.clip(a_mc, avlim[0], avlim[1])
        r_mc = np.clip(r_mc, rvlim[0], rvlim[1])

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
                lngalprior = logp_galactic_structure

            # Galactic prior evaluation (FULLY VECTORIZED)
            # We have dist_mc shape (Nmc, Nsel) and need to evaluate for each model's labels
            # Each model has 1 label, each model has Nmc distances
            # Solution: tile labels to match distances, then evaluate all at once

            # Flatten all distances: shape (Nmc * Nsel,)
            dist_flat = dist_mc.ravel()

            if dlabels is None:
                # No labels - evaluate once for all distances
                lnp_gal_flat = lngalprior(dist_flat, coord, labels=None)
            else:
                # Create labels array that matches flattened distances (VECTORIZED)
                # Extract labels for selected models: shape (Nsel,)
                labels_selected = dlabels[sel]

                # Use np.repeat to repeat each label Nmc times: shape (Nmc * Nsel,)
                # This creates: [label0, label0, ..., label0, label1, label1, ..., label1, ...]
                #               |---- Nmc times ----|  |---- Nmc times ----|
                labels_flat = np.repeat(labels_selected, Nmc)

                # Evaluate prior for all distance-label pairs at once
                lnp_gal_flat = lngalprior(dist_flat, coord, labels=labels_flat)

            # Reshape back to (Nmc, Nsel)
            lnp_gal_reshaped = lnp_gal_flat.reshape(Nmc, Nsel)
            lnp_mc += lnp_gal_reshaped

        if dustfile is not None:
            # Load dust map from file path if needed
            if isinstance(dustfile, str):
                from ..dust import Bayestar

                dustfile = Bayestar(dustfile=dustfile)

            if lndustprior is None:
                lndustprior = logp_extinction

            # Dust prior evaluation (VECTORIZED)
            # For 3D dust maps, pass distances so the prior can interpolate
            # the extinction profile to each star's distance
            av_flat = a_mc.ravel()  # Shape: (Nmc * Nsel,)
            dist_flat = dist_mc.ravel()  # Shape: (Nmc * Nsel,)
            lnp_dust_flat = lndustprior(av_flat, dustfile, coord, distance=dist_flat)
            lnp_dust_reshaped = lnp_dust_flat.reshape(Nmc, Nsel)  # Shape: (Nmc, Nsel)
            lnp_mc += lnp_dust_reshaped

        # Parallax prior (FULLY VECTORIZED)
        if parallax is not None and parallax_err is not None:
            par_mc = 1.0 / dist_mc  # Shape: (Nmc, Nsel)
            lnp_parallax_all = logp_parallax(par_mc, parallax, parallax_err)
            lnp_mc += lnp_parallax_all

        # Compute integrated posterior (VECTORIZED)
        lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc)

        # Safety check - replace non-finite values with log(0) proxy
        lnp_mask = np.where(~np.isfinite(lnp))[0]
        if len(lnp_mask) > 0:
            lnp[lnp_mask] = LOG_ZERO

        # Compute effective sample size (ESS) for each selected model's
        # MC samples. ESS = 1 / sum(w_i^2) where w_i are normalized weights.
        # Low ESS indicates that the MC integration is dominated by a few
        # samples, suggesting poor overlap between the proposal (Gaussian in
        # ln d, Av, Rv) and the target (posterior with priors).
        # lnp_mc shape is (Nmc, Nsel) at this point.
        mc_ess = np.zeros(Nsel)
        for j in range(Nsel):
            lw = lnp_mc[:, j]
            lw_max = np.max(lw)
            w = np.exp(lw - lw_max)
            w_sum = w.sum()
            if w_sum > 0:
                w_normed = w / w_sum
                mc_ess[j] = 1.0 / np.sum(w_normed**2)
            else:
                mc_ess[j] = 0.0

        return sel, cov_sar, lnp, dist_mc.T, a_mc.T, r_mc.T, lnp_mc.T, mc_ess

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
        avlim=(0.0, 20.0),
        av_gauss=None,
        rvlim=(1.0, 8.0),
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
        ltol=3e-2,
        ltol_subthresh=1e-2,
        logl_initthresh=5e-3,
        mag_max=50.0,
        merr_max=0.25,
        rstate=None,
        save_dar_draws=True,
        running_io=True,
        mem_lim=8000.0,
        verbose=True,
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
            Binary mask (0/1) indicating whether each measurement is valid.

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
            (min, max) bounds on A(V). Default is (0.0, 20.0).

        av_gauss : tuple, optional
            (mean, std) for Gaussian prior on A(V). If provided, this is used
            instead of the distance-reddening prior during fitting.

        rvlim : tuple, optional
            (min, max) bounds on R(V). Default is (1.0, 8.0).

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
            Galactic structure prior function. If not provided, uses the
            default Galactic model from Green et al. (2014).

        lndustprior : callable, optional
            Dust prior function with signature
            ``f(avs, dustmap, coord, distance=None)``. If not provided
            and dustfile is given, uses ``logp_extinction``.

        dustfile : str or `~brutus.dust.Bayestar`, optional
            3D dust map for extinction priors. Can be a file path (string)
            to a Bayestar HDF5 file, which will be loaded automatically,
            or a pre-loaded ``Bayestar`` object.

        apply_dlabels : bool, optional
            Whether to pass model labels to Galactic prior. Default is True.

        data_coords : numpy.ndarray of shape (Ndata, 2), optional
            Galactic (l, b) coordinates for each object in degrees.
            Required if using default Galactic prior.

        logl_dim_prior : bool, optional
            Whether to apply dimensional correction to log-likelihood.
            Default is True.

        ltol : float, optional
            Convergence tolerance for likelihood optimization. Default is 3e-2.

        ltol_subthresh : float, optional
            Sub-threshold for convergence. Default is 1e-2.

        logl_initthresh : float, optional
            Initial likelihood threshold for model culling. Default is 5e-3.

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

        Returns
        -------
        str
            Path to the output HDF5 file.

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
        - ``mc_ess``: Monte Carlo effective sample size (Ndata, Ndraws).
          For each posterior draw, the ESS of the MC integration over
          (distance, Av, Rv) for that draw's source model. Low ESS
          indicates poor overlap between the Gaussian proposal and the
          target posterior.

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
        # Load dust map from file path if needed (once for all objects)
        if dustfile is not None and isinstance(dustfile, str):
            from ..dust import Bayestar

            dustfile = Bayestar(dustfile=dustfile)

        # Pre-process data and initialize priors
        setup_results = self._setup(
            data,
            data_err,
            data_mask,
            data_labels=data_labels,
            phot_offsets=phot_offsets,
            parallax=parallax,
            parallax_err=parallax_err,
            av_gauss=av_gauss,
            lnprior=lnprior,
            wt_thresh=wt_thresh,
            cdf_thresh=cdf_thresh,
            apply_agewt=apply_agewt,
            apply_grad=apply_grad,
            lngalprior=lngalprior,
            lndustprior=lndustprior,
            dustfile=dustfile,
            data_coords=data_coords,
            ltol_subthresh=ltol_subthresh,
            logl_initthresh=logl_initthresh,
            mag_max=mag_max,
            merr_max=merr_max,
            rstate=rstate,
        )

        (
            data_proc,
            data_err_proc,
            data_mask_proc,
            lnprior_proc,
            lngalprior_proc,
            lndustprior_proc,
        ) = setup_results

        Ndata, Nfilt = data.shape

        # Initialize random state
        if rstate is None:
            rstate = np.random.RandomState()

        # Ensure save_file has .h5 extension
        if not save_file.endswith(".h5"):
            save_file = f"{save_file}.h5"

        # Get model labels for priors
        dlabels = self.models_labels if apply_dlabels else None

        # Initialize results file
        out = h5py.File(save_file, "w")
        out.create_dataset("labels", data=data_labels)

        if running_io:
            # Streaming mode: create datasets upfront, write as we go
            out.create_dataset(
                "model_idx", data=np.full((Ndata, Ndraws), -99, dtype="int32")
            )
            out.create_dataset(
                "ml_scale", data=np.ones((Ndata, Ndraws), dtype="float32")
            )
            out.create_dataset("ml_av", data=np.zeros((Ndata, Ndraws), dtype="float32"))
            out.create_dataset("ml_rv", data=np.zeros((Ndata, Ndraws), dtype="float32"))
            out.create_dataset(
                "ml_cov_sar", data=np.zeros((Ndata, Ndraws, 3, 3), dtype="float32")
            )
            out.create_dataset(
                "obj_log_post", data=np.zeros((Ndata, Ndraws), dtype="float32")
            )
            out.create_dataset("obj_log_evid", data=np.zeros(Ndata, dtype="float32"))
            out.create_dataset("obj_chi2min", data=np.zeros(Ndata, dtype="float32"))
            out.create_dataset("obj_Nbands", data=np.zeros(Ndata, dtype="int16"))
            out.create_dataset(
                "mc_ess", data=np.zeros((Ndata, Ndraws), dtype="float32")
            )
            if save_dar_draws:
                out.create_dataset(
                    "samps_dist", data=np.ones((Ndata, Ndraws), dtype="float32")
                )
                out.create_dataset(
                    "samps_red", data=np.ones((Ndata, Ndraws), dtype="float32")
                )
                out.create_dataset(
                    "samps_dred", data=np.ones((Ndata, Ndraws), dtype="float32")
                )
                out.create_dataset(
                    "samps_logp", data=np.ones((Ndata, Ndraws), dtype="float32")
                )
        else:
            # Batch mode: accumulate in memory, write at end
            idxs_arr = np.full((Ndata, Ndraws), -99, dtype="int32")
            scale_arr = np.ones((Ndata, Ndraws), dtype="float32")
            av_arr = np.zeros((Ndata, Ndraws), dtype="float32")
            rv_arr = np.zeros((Ndata, Ndraws), dtype="float32")
            cov_arr = np.zeros((Ndata, Ndraws, 3, 3), dtype="float32")
            logpost_arr = np.zeros((Ndata, Ndraws), dtype="float32")
            logevid_arr = np.zeros(Ndata, dtype="float32")
            chi2best_arr = np.zeros(Ndata, dtype="float32")
            nbands_arr = np.zeros(Ndata, dtype="int16")
            mc_ess_arr = np.zeros((Ndata, Ndraws), dtype="float32")
            if save_dar_draws:
                dist_arr = np.ones((Ndata, Ndraws), dtype="float32")
                red_arr = np.ones((Ndata, Ndraws), dtype="float32")
                dred_arr = np.ones((Ndata, Ndraws), dtype="float32")
                logp_arr = np.ones((Ndata, Ndraws), dtype="float32")

        # Fit data
        if verbose:
            t1 = time.time()
            t = 0.0
            sys.stderr.write(f"\rFitting object 1/{Ndata}  ")
            sys.stderr.flush()

        for i in range(Ndata):
            # Get parallax for this object
            par_i = parallax[i] if parallax is not None else None
            par_err_i = parallax_err[i] if parallax_err is not None else None
            coord_i = data_coords[i] if data_coords is not None else None

            # Fit individual object
            results = self._fit(
                data_proc[i],
                data_err_proc[i],
                data_mask_proc[i],
                parallax=par_i,
                parallax_err=par_err_i,
                coord=coord_i,
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
                ltol_subthresh=ltol_subthresh,
                logl_initthresh=logl_initthresh,
                mem_lim=mem_lim,
                rstate=rstate,
                return_distreds=save_dar_draws,
            )

            # Unpack results
            if save_dar_draws:
                (
                    idxs,
                    scales,
                    avs,
                    rvs,
                    covs_sar,
                    Ndim,
                    lpost,
                    levid,
                    chi2min,
                    dists,
                    reds,
                    dreds,
                    logwt,
                    mc_ess_i,
                ) = results
            else:
                (
                    idxs,
                    scales,
                    avs,
                    rvs,
                    covs_sar,
                    Ndim,
                    lpost,
                    levid,
                    chi2min,
                    mc_ess_i,
                ) = results

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
                    f"[chi2/n: {chi2min:.1f}/{Ndim}] "
                    f"(mean time: {t_avg:.3f} s/obj, "
                    f"est. remaining: {t_est:.3f} s)    "
                )
                sys.stderr.flush()

            # Save results
            if running_io:
                out["model_idx"][i] = idxs
                out["ml_scale"][i] = scales
                out["ml_av"][i] = avs
                out["ml_rv"][i] = rvs
                out["ml_cov_sar"][i] = covs_sar
                out["obj_Nbands"][i] = Ndim
                out["obj_log_post"][i] = lpost
                out["obj_log_evid"][i] = levid
                out["obj_chi2min"][i] = chi2min
                out["mc_ess"][i] = mc_ess_i
                if save_dar_draws:
                    out["samps_dist"][i] = dists
                    out["samps_red"][i] = reds
                    out["samps_dred"][i] = dreds
                    out["samps_logp"][i] = logwt
            else:
                idxs_arr[i] = idxs
                scale_arr[i] = scales
                av_arr[i] = avs
                rv_arr[i] = rvs
                cov_arr[i] = covs_sar
                logpost_arr[i] = lpost
                logevid_arr[i] = levid
                chi2best_arr[i] = chi2min
                nbands_arr[i] = Ndim
                mc_ess_arr[i] = mc_ess_i
                if save_dar_draws:
                    dist_arr[i] = dists
                    red_arr[i] = reds
                    dred_arr[i] = dreds
                    logp_arr[i] = logwt

        # Final progress update
        if verbose:
            t2 = time.time()
            dt = t2 - t1
            t += dt
            t_avg = t / Ndata
            sys.stderr.write(
                f"\rFitting object {Ndata}/{Ndata} "
                f"[chi2/n: {chi2min:.1f}/{Ndim}] "
                f"(mean time: {t_avg:.3f} s/obj, "
                f"total: {t:.3f} s)    \n"
            )
            sys.stderr.flush()

        # Dump results to disk if using batch mode
        if not running_io:
            out.create_dataset("model_idx", data=idxs_arr)
            out.create_dataset("ml_scale", data=scale_arr)
            out.create_dataset("ml_av", data=av_arr)
            out.create_dataset("ml_rv", data=rv_arr)
            out.create_dataset("ml_cov_sar", data=cov_arr)
            out.create_dataset("obj_log_post", data=logpost_arr)
            out.create_dataset("obj_log_evid", data=logevid_arr)
            out.create_dataset("obj_chi2min", data=chi2best_arr)
            out.create_dataset("obj_Nbands", data=nbands_arr)
            out.create_dataset("mc_ess", data=mc_ess_arr)
            if save_dar_draws:
                out.create_dataset("samps_dist", data=dist_arr)
                out.create_dataset("samps_red", data=red_arr)
                out.create_dataset("samps_dred", data=dred_arr)
                out.create_dataset("samps_logp", data=logp_arr)

        # Close output file
        out.close()

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
        avlim=(0.0, 20.0),
        av_gauss=(0.0, 1e6),
        rvlim=(1.0, 8.0),
        rv_gauss=(3.32, 0.18),
        lnprior=None,
        wt_thresh=1e-3,
        cdf_thresh=2e-3,
        Ndraws=250,
        lngalprior=None,
        lndustprior=None,
        dustfile=None,
        dlabels=None,
        logl_dim_prior=True,
        ltol=3e-2,
        ltol_subthresh=1e-2,
        logl_initthresh=5e-3,
        mem_lim=8000.0,
        rstate=None,
        return_distreds=True,
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
        dustfile : str, optional
            Path to 3D dust map file.
        dlabels : numpy.ndarray, optional
            Model labels for prior evaluation.
        logl_dim_prior : bool, optional
            Whether to apply dimensional prior correction.
        ltol : float, optional
            Convergence tolerance.
        ltol_subthresh : float, optional
            Sub-threshold for convergence.
        logl_initthresh : float, optional
            Initial likelihood threshold.
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
            ltol_subthresh=ltol_subthresh,
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
            rstate=rstate,
        )

        # Unpack posterior results
        # sel: selected model indices, cov_sar: covariances for selected models
        # lnp: log-posteriors, dist_mc/a_mc/r_mc: MC samples, lnp_mc: MC log-posteriors
        # mc_ess: effective sample size per selected model
        sel, cov_sar_sel, lnp, dist_mc, a_mc, r_mc, lnp_mc, mc_ess_sel = logpost_results
        Nsel = len(sel)

        # Add parallax contribution to chi2 and Ndim if provided
        Ndim_out = Ndim
        chi2_with_par = chi2.copy()
        if parallax is not None and parallax_err is not None:
            if np.isfinite(parallax) and np.isfinite(parallax_err):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    par_pred = np.sqrt(scales_all)
                    chi2_with_par += (par_pred - parallax) ** 2 / parallax_err**2
                    Ndim_out += 1

        # Compute goodness-of-fit metrics
        chi2min = np.min(chi2_with_par[sel])
        levid = logsumexp(lnp)

        # Resample from posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wt = np.exp(lnp - levid)
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
        covs_sar = cov_sar_sel[idxs_local]
        lnprob = lnp[idxs_local]
        mc_ess = mc_ess_sel[idxs_local]

        if return_distreds:
            # Draw distance and reddening samples from MC integration
            # For each resampled model, pick one MC sample weighted by its contribution
            dists = np.zeros(Ndraws, dtype="float32")
            reds = np.zeros(Ndraws, dtype="float32")
            dreds = np.zeros(Ndraws, dtype="float32")
            logwts = np.zeros(Ndraws, dtype="float32")

            for i, idx_local in enumerate(idxs_local):
                # Get MC samples for this model
                mc_logwts = lnp_mc[idx_local]  # Shape: (Nmc,)
                mc_logwts_max = np.max(mc_logwts)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mc_wt = np.exp(mc_logwts - mc_logwts_max)
                    mc_wt_sum = mc_wt.sum()
                    if mc_wt_sum > 0:
                        mc_wt /= mc_wt_sum
                    else:
                        mc_wt = np.ones(len(mc_logwts)) / len(mc_logwts)

                # Draw one MC sample
                imc = rstate.choice(len(mc_logwts), p=mc_wt)
                dists[i] = dist_mc[idx_local, imc]
                reds[i] = a_mc[idx_local, imc]
                dreds[i] = r_mc[idx_local, imc]
                logwts[i] = mc_logwts[imc]

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
