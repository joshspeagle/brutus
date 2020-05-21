#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute force fitter.

"""

from __future__ import (print_function, division)

import sys
import warnings
import numpy as np
import h5py
import time
from numba import jit
from math import log

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .pdf import imf_lnprior, ps1_MrLF_lnprior
from .pdf import parallax_lnprior, scale_parallax_lnprior
from .pdf import gal_lnprior, dust_lnprior
from .utils import (_inverse3, _chisquare_logpdf, _get_seds,
                    magnitude, sample_multivariate_normal)

__all__ = ["_optimize_fit_mag", "_optimize_fit_flux", "_get_sed_mle",
           "loglike", "lnpost", "BruteForce"]


@jit(nopython=True, cache=True)
def _optimize_fit_mag(data, tot_var, models, rvecs, drvecs,
                      av, rv, mag_coeffs, resid, stepsize,
                      mags, mags_var,
                      avlim=(0., 20.), av_gauss=(0., 1e6),
                      rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
                      tol=0.05, init_thresh=5e-3):
    """
    Optimize the distance and reddening between the models and the data using
    the gradient in **magnitudes**. This executes multiple `(Av, Rv)` updates.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated (Normal) errors on the observed values compared to the
        models.

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

    mags : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values in magnitudes.

    mags_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated (Normal) errors on the observed values compared to the
        models in magnitudes.

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

    tol : float, optional
        The maximum tolerance in the computed Av and Rv values used to
        determine convergence. Default is `0.05`.

    init_thresh : bool, optional
        The weight threshold used to mask out fits after the initial
        magnitude-based fit when determining convergence. Default is `5e-3`.

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

    """

    Nmodel, Nfilt = models.shape

    avmin, avmax = avlim
    rvmin, rvmax = rvlim

    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    Av_varinv, Rv_varinv = 1. / Av_std**2, 1. / Rv_std**2

    log_init_thresh = log(init_thresh)

    # In magnitude space, we can solve a linear system
    # explicitly for `(s_ML, Av_ML, r_ML=Av_ML*Rv_ML)`. We opt to
    # solve for Av and Rv in turn to so we can impose priors and bounds
    # on both quantities as well as additional regularization.

    # Compute constants.
    s_den, rp_den = np.zeros(Nmodel), np.zeros(Nmodel)
    srp_mix = np.zeros(Nmodel)
    for i in range(Nmodel):
        for j in range(Nfilt):
            s_den[i] += 1. / mags_var[i][j]
            rp_den[i] += drvecs[i][j] * drvecs[i][j] / mags_var[i][j]
            srp_mix[i] += drvecs[i][j] / mags_var[i][j]

    # Main loop.
    a_den, r_den = np.zeros(Nmodel), np.zeros(Nmodel)
    sa_mix, sr_mix = np.zeros(Nmodel), np.zeros(Nmodel)
    chi2, logwt = np.zeros(Nmodel), np.zeros(Nmodel)
    dav, drv = np.zeros(Nmodel), np.zeros(Nmodel)
    resid_s = np.zeros(Nmodel)
    resid_a, resid_r = np.zeros(Nmodel), np.zeros(Nmodel)
    while True:
        for i in range(Nmodel):
            # Solve for Av.
            a_den[i], sa_mix[i], resid_s[i], resid_a[i] = 0.0, 0.0, 0.0, 0.0
            for j in range(Nfilt):
                # Derive partial derivatives.
                a_den[i] += rvecs[i][j] * rvecs[i][j] / mags_var[i][j]
                sa_mix[i] += rvecs[i][j] / mags_var[i][j]
                # Compute residual terms
                resid_s[i] += resid[i][j] / mags_var[i][j]
                resid_a[i] += resid[i][j] * rvecs[i][j] / mags_var[i][j]
            # Add in Gaussian Av prior.
            resid_a[i] += (Av_mean - av[i]) * Av_varinv
            a_den[i] += Av_varinv
            # Compute determinants (normalization terms).
            sa_idet = 1. / (s_den[i] * a_den[i] - sa_mix[i] * sa_mix[i])
            # Compute ML solution for Delta_Av.
            dav[i] = sa_idet * (s_den[i] * resid_a[i] - sa_mix[i] * resid_s[i])
            # Adjust dAv based on the provided stepsize.
            dav[i] = dav[i] * stepsize[i]

            # Prevent Av from sliding off the provided bounds.
            if dav[i] < avmin - av[i]:
                dav[i] = avmin - av[i]
            if dav[i] > avmax - av[i]:
                dav[i] = avmax - av[i]

            # Increment to new Av.
            av[i] = av[i] + dav[i]
            # Update residuals.
            for j in range(Nfilt):
                resid[i][j] = resid[i][j] - dav[i] * rvecs[i][j]

            # Solve for Rv.
            resid_s[i], resid_r[i] = 0.0, 0.0
            # Derive partial derivatives.
            r_den[i] = rp_den[i] * av[i] * av[i]
            sr_mix[i] = srp_mix[i] * av[i]
            for j in range(Nfilt):
                # Compute residual terms.
                resid_s[i] += resid[i][j] / mags_var[i][j]
                resid_r[i] += resid[i][j] * drvecs[i][j] / mags_var[i][j]
            resid_r[i] = resid_r[i] * av[i]
            # Add in Gaussian Rv prior.
            resid_r[i] += (Rv_mean - rv[i]) * Rv_varinv
            r_den[i] += Rv_varinv
            # Compute determinants (normalization terms).
            sr_idet = 1. / (s_den[i] * r_den[i] - sr_mix[i] * sr_mix[i])
            # Compute ML solution for Delta_Rv.
            drv[i] = sr_idet * (s_den[i] * resid_r[i] - sr_mix[i] * resid_s[i])
            # Adjust dRv based on the provided stepsize.
            drv[i] = drv[i] * stepsize[i]

            # Prevent Rv from sliding off the provided bounds.
            if drv[i] < rvmin - rv[i]:
                drv[i] = rvmin - rv[i]
            if drv[i] > rvmax - rv[i]:
                drv[i] = rvmax - rv[i]

            # Increment to new Rv.
            rv[i] = rv[i] + drv[i]
            # Update residuals and reddening vector.
            for j in range(Nfilt):
                resid[i][j] = resid[i][j] - av[i] * drv[i] * drvecs[i][j]
                rvecs[i][j] = rvecs[i][j] + drv[i] * drvecs[i][j]

            # Compute error based on best-fitting objects.
            chi2[i] = 0.0
            for j in range(Nfilt):
                chi2[i] += resid[i][j] * resid[i][j] / mags_var[i][j]
            logwt[i] = -0.5 * chi2[i]

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
    (models, rvecs, drvecs, scale,
     icov_sar, resid) = _get_sed_mle(data, tot_var, resid, mag_coeffs, av, rv,
                                     av_gauss=av_gauss, rv_gauss=rv_gauss)

    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid


@jit(nopython=True, cache=True)
def _optimize_fit_flux(data, tot_var, models, rvecs, drvecs,
                       av, rv, mag_coeffs, resid, stepsize,
                       avlim=(0., 20.), av_gauss=(0., 1e6),
                       rvlim=(1., 8.), rv_gauss=(3.32, 0.18)):
    """
    Optimize the distance and reddening between the models and the data using
    the gradient in **flux densities**. This executes **only one**
    `(Av, Rv)` update.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated (Normal) errors on the observed values compared to the
        models.

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

    """

    Nmodel, Nfilt = models.shape

    avmin, avmax = avlim
    rvmin, rvmax = rvlim

    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss
    Av_varinv, Rv_varinv = 1. / Av_std**2, 1. / Rv_std**2

    # In flux density space, we can solve the linear system
    # implicitly for `(s_ML, Av_ML, Rv_ML)`. However, the solution
    # is not necessarily as numerically stable as one might hope
    # due to the nature of our Taylor expansion in flux.
    # Instead, it is easier to iterate in `(dAv, dRv)` from
    # a good guess for `(s_ML, Av_ML, Rv_ML)`. We opt to solve both
    # independently at fixed `(Av, Rv)` to avoid recomputing models.

    a_num, a_den, dav = np.zeros(Nmodel), np.zeros(Nmodel), np.zeros(Nmodel)
    r_num, r_den, drv = np.zeros(Nmodel), np.zeros(Nmodel), np.zeros(Nmodel)

    for i in range(Nmodel):
        # Derive ML Delta_Av (`dav`) between data and models.
        for j in range(Nfilt):
            a_num[i] += rvecs[i][j] * resid[i][j] / tot_var[i][j]
            a_den[i] += rvecs[i][j] * rvecs[i][j] / tot_var[i][j]
        a_num[i] += (Av_mean - av[i]) * Av_varinv
        a_den[i] += Av_varinv
        dav[i] = a_num[i] / a_den[i]
        dav[i] *= stepsize[i]

        # Derive ML Delta_Rv (`drv`) between data and models.
        for j in range(Nfilt):
            r_num[i] += drvecs[i][j] * resid[i][j] / tot_var[i][j]
            r_den[i] += drvecs[i][j] * drvecs[i][j] / tot_var[i][j]
        r_num[i] += (Rv_mean - rv[i]) * Rv_varinv
        r_den[i] += Rv_varinv
        drv[i] = r_num[i] / r_den[i]
        drv[i] *= stepsize[i]

        # Prevent Av from sliding off the provided bounds.
        if dav[i] < avmin - av[i]:
            dav[i] = avmin - av[i]
        if dav[i] > avmax - av[i]:
            dav[i] = avmax - av[i]

        # Increment to new Av.
        av[i] += dav[i]

        # Prevent Rv from sliding off the provided bounds.
        if drv[i] < rvmin - rv[i]:
            drv[i] = rvmin - rv[i]
        if drv[i] > rvmax - rv[i]:
            drv[i] = rvmax - rv[i]

        # Increment to new Rv.
        rv[i] += drv[i]

    # Get MLE models and associated quantities.
    (models, rvecs, drvecs, scale,
     icov_sar, resid) = _get_sed_mle(data, tot_var, resid, mag_coeffs, av, rv,
                                     av_gauss=av_gauss, rv_gauss=rv_gauss)

    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid


@jit(nopython=True, cache=True)
def _get_sed_mle(data, tot_var, resid, mag_coeffs, av, rv,
                 av_gauss=(0., 1e6), rv_gauss=(3.32, 0.18),
                 av_reg=0.05, rv_reg=0.1):
    """
    Optimize the distance and reddening between the models and the data using
    the gradient in **flux densities**. This executes **only one**
    `(Av, Rv)` update.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    tot_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Associated (Normal) errors on the observed values compared to the
        models.

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

    av_reg : float, optional
        The regularization applied to A(V), which functions as an effective
        standard deviation. Default is `0.05`.

    rv_reg : float, optional
        The regularization applied to R(V), which functions as an effective
        standard deviation. Default is `0.1`.

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
    models, rvecs, drvecs = _get_seds(mag_coeffs, av, rv,
                                      return_flux=True)
    Nmodel, Nfilt = models.shape

    # Derive scale-factors (`scale`) between data and models.
    s_num, s_den, scale = np.zeros(Nmodel), np.zeros(Nmodel), np.zeros(Nmodel)
    for i in range(Nmodel):
        for j in range(Nfilt):
            s_num[i] += models[i][j] * data[j] / tot_var[i][j]
            s_den[i] += models[i][j] * models[i][j] / tot_var[i][j]
        scale[i] = s_num[i] / s_den[i]  # MLE scalefactor
        if scale[i] <= 1e-20:
            scale[i] = 1e-20

    # Derive reddening terms.
    sr_mix, sa_mix = np.zeros(Nmodel), np.zeros(Nmodel)
    a_den, r_den = np.zeros(Nmodel), np.zeros(Nmodel)
    ar_mix = np.zeros(Nmodel)
    Av_varinv, Rv_varinv = 1. / Av_std**2, 1. / Rv_std**2
    a_den_reg, r_den_reg = 1. / av_reg**2, 1. / rv_reg**2
    for i in range(Nmodel):
        for j in range(Nfilt):
            # Compute reddening effect.
            models_int = 10.**(-0.4 * mag_coeffs[i][j][0])
            reddening = models[i][j] - models_int

            # Rescale models.
            models[i][j] = models[i][j] * scale[i]

            # Compute residuals.
            resid[i][j] = data[j] - models[i][j]

            # Derive scale cross-terms.
            sr_mix[i] += drvecs[i][j] * ((models[i][j] - resid[i][j])
                                         / tot_var[i][j])
            sa_mix[i] += rvecs[i][j] * ((models[i][j] - resid[i][j])
                                        / tot_var[i][j])

            # Rescale reddening quantities.
            rvecs[i][j] = rvecs[i][j] * scale[i]
            drvecs[i][j] = drvecs[i][j] * scale[i]
            reddening *= scale[i]

            # Derive reddening (cross-)terms
            ar_mix[i] += drvecs[i][j] * ((reddening - resid[i][j])
                                         / tot_var[i][j])
            a_den[i] += rvecs[i][j] * rvecs[i][j] / tot_var[i][j]
            r_den[i] += drvecs[i][j] * drvecs[i][j] / tot_var[i][j]

        # Add in priors.
        a_den[i] += Av_varinv
        r_den[i] += Rv_varinv

        # Add in additional regularization to ensure solution well-behaved.
        a_den[i] += a_den_reg
        r_den[i] += r_den_reg

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


def loglike(data, data_err, data_mask, mag_coeffs,
            avlim=(0., 20.), av_gauss=(0., 1e6),
            rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
            av_init=None, rv_init=None,
            dim_prior=True, ltol=3e-2, ltol_subthresh=1e-2, init_thresh=5e-3,
            parallax=None, parallax_err=None,
            return_vals=False, *args, **kwargs):
    """
    Computes the log-likelihood between noisy data and noiseless models
    optimizing over the scale-factor, dust attenuation, and the shape of the
    reddening curve.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Measured flux densities, in units of `10**(-0.4 * mag)`.

    data_err : `~numpy.ndarray` of shape `(Nfilt)`
        Associated (Normal) errors on the observed flux densities.

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`
        Binary mask (0/1) indicating whether the data was observed.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients used to compute reddened photometry for a given
        model. Contains `(mag, r0, dr)` pairs referencing the unreddening
        magnitudes, the reddening vector as a function of A(V),
        and the change in the reddening vector as a function of R(V).

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

    av_init : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The initial A(V) guess. Default is `0.`.

    rv_init : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The initial R(V) guess. Default is `3.3`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 3` degrees of freedom. Default is `True`.

    ltol : float, optional
        The weighted tolerance in the computed log-likelihoods used to
        determine convergence. Default is `3e-2`.

    ltol_subthresh : float, optional
        The threshold used to sub-select the best-fit log-likelihoods used
        to determine convergence. Default is `1e-2`.

    init_thresh : bool, optional
        The weight threshold used to mask out fits after the initial
        magnitude-based fit before transforming the results back to
        flux density (and iterating until convergence). Default is `5e-3`.
        **This must be smaller than or equal to `ltol_subthresh`.**

    parallax : float, optional
        Parallax measurement. If provided and `init_thresh` is not `None`,
        (i.e. initial clipping is turned on), this will be used to clip models
        after the initial magnitude fits.

    parallax_err : float, optional
        Error on the parallax measurements. Must be provided along with
        `parallax`.

    return_vals : bool, optional
        Whether to return the best-fit scale-factor, reddening, and shape
        along with the associated precision matrix (inverse covariance).
        Default is `False`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape `(Nmodel)`
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape `(Nmodel)`
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape `(Nmodel)`
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit scale factors.

    Av : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddenings.

    Rv : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddening shapes.

    icov_sar : `~numpy.ndarray` of shape `(Nmodel, 3, 3)`, optional
        The precision (inverse covariance) matrices expanded around
        `(s_ML, Av_ML, Rv_ML)`.

    """

    if init_thresh > ltol_subthresh:
        raise ValueError("The initial threshold must be smaller than or equal "
                         "to the final threshold applied to be useful!")

    if av_gauss is None:
        av_gauss = (0., 1e6)

    # Initialize values.
    Nmodels, Nfilt, Ncoef = mag_coeffs.shape
    if av_init is None:
        av_init = np.zeros(Nmodels) + av_gauss[0]
    if rv_init is None:
        rv_init = np.zeros(Nmodels) + rv_gauss[0]

    # Clean data (safety checks).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
        data_mask[~clean] = False
    Ndim = sum(data_mask)  # number of dimensions

    # Subselect only clean observations.
    flux, fluxerr = data[data_mask], data_err[data_mask]  # mean, error
    mcoeffs = mag_coeffs[:, data_mask, :]  # model magnitude coefficients
    tot_var = np.square(fluxerr)  # total variance
    tot_var = np.repeat(tot_var[np.newaxis, :], Nmodels, axis=0)

    # Get started by fitting in magnitudes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore bad values
        # Convert to magnitudes.
        mags = -2.5 * np.log10(flux)
        mags_var = np.square(2.5 / np.log(10.)) * tot_var / np.square(flux)
        mclean = np.isfinite(mags)
        mags[~mclean], mags_var[:, ~mclean] = 0., 1e50  # mask negative values

    # Compute unreddened photometry.
    models, rvecs, drvecs = _get_seds(mcoeffs, av_init, rv_init,
                                      return_flux=False)

    # Compute initial magnitude fit.
    mtol = 2.5 * ltol
    resid = mags - models
    stepsize = np.ones(Nmodels)
    results = _optimize_fit_mag(flux, tot_var, models, rvecs, drvecs,
                                av_init, rv_init, mcoeffs,
                                resid, stepsize, mags, mags_var,
                                tol=mtol, init_thresh=init_thresh,
                                avlim=avlim, av_gauss=av_gauss,
                                rvlim=rvlim, rv_gauss=rv_gauss)
    models, rvecs, drvecs, scale, av, rv, icov_sar, resid = results

    if init_thresh is not None:
        # Cull initial bad fits before moving on.
        chi2 = np.sum(np.square(resid) / tot_var, axis=1)
        # Compute multivariate normal logpdf.
        lnl = -0.5 * chi2  # ignore constant
        # Add parallax to log-likelihood.
        lnl_p = lnl
        if parallax is not None and parallax_err is not None:
            if np.isfinite(parallax) and np.isfinite(parallax_err):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ignore bad values
                    par = np.sqrt(scale)
                    chi2_p = (par - parallax)**2 / parallax_err**2
                    lnl_p = lnl - 0.5 * chi2_p
        # Subselect models using log-likelihood thresholding.
        lnl_sel = lnl_p > np.max(lnl_p) + np.log(init_thresh)
        init_sel = np.where(lnl_sel)[0]
        # Subselect data.
        tot_var = tot_var[init_sel]
        models = models[init_sel]
        rvecs = rvecs[init_sel]
        drvecs = drvecs[init_sel]
        av_new = av[init_sel]
        rv_new = rv[init_sel]
        mcoeffs = mcoeffs[init_sel]
        resid = resid[init_sel]
    else:
        # Keep all models.
        init_sel = np.arange(Nmodels)
        chi2 = np.ones(Nmodels) + 1e300
        lnl = np.ones(Nmodels) - 1e300
        av_new = np.array(av, order='F')
        rv_new = np.array(rv, order='F')

    # Iterate until convergence.
    lnl_old, lerr = -1e300, 1e300
    stepsize, rescaling = np.ones(Nmodels)[init_sel], 1.2
    ln_ltol_subthresh = np.log(ltol_subthresh)
    while lerr > ltol:

        # Re-compute models.
        results = _optimize_fit_flux(flux, tot_var, models, rvecs, drvecs,
                                     av_new, rv_new, mcoeffs, resid, stepsize,
                                     avlim=avlim, av_gauss=av_gauss,
                                     rvlim=rvlim, rv_gauss=rv_gauss)
        (models, rvecs, drvecs,
         scale_new, av_new, rv_new, icov_sar_new, resid) = results

        # Compute chi2.
        chi2_new = np.sum(np.square(resid) / tot_var, axis=1)

        # Compute multivariate normal logpdf.
        lnl_new = -0.5 * chi2_new  # ignore constant

        # Compute stopping criterion.
        lnl_sel = np.where(lnl_new > np.max(lnl_new) + ln_ltol_subthresh)[0]
        lerr = np.max(np.abs(lnl_new - lnl_old)[lnl_sel])

        # Adjust stepsize.
        stepsize[lnl_new < lnl_old] /= rescaling
        lnl_old = lnl_new

    # Insert optimized models into initial array of results.
    lnl_new += -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))  # add constant
    lnl[init_sel], chi2[init_sel] = lnl_new, chi2_new
    scale[init_sel], av[init_sel], rv[init_sel] = scale_new, av_new, rv_new
    icov_sar[init_sel] = icov_sar_new

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        lnl = _chisquare_logpdf(chi2, Ndim - 3)

    if return_vals:
        return lnl, Ndim, chi2, scale, av, rv, icov_sar
    else:
        return lnl, Ndim, chi2


def lnpost(results, parallax=None, parallax_err=None, coord=None,
           Nmc_prior=100, lnprior=None, wt_thresh=5e-3, cdf_thresh=2e-3,
           lngalprior=None, lndustprior=None, dustfile=None,
           dlabels=None, avlim=(0., 20.), rvlim=(1., 8.),
           rstate=None, apply_av_prior=True, *args, **kwargs):
    """
    Internal function used to estimate posteriors from fits using the
    provided priors via Monte Carlo integration.

    Parameters
    ----------
    results : tuple of `(lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar)`
        Fits returned from `loglike` with `return_vals=True`. Ndim is
        an integer, while the rest of the results are `~numpy.ndarray`s
        with shape `(Nmodels,)`.

    parallax : float, optional
        Parallax measurement to be used as a prior.

    parallax_err : float, optional
        Errors on the parallax measurement. Must be provided along with
        `parallax`.

    coord : tuple of shape `(2,)`, optional
        The galactic `(l, b)` coordinates for the object being
        fit. These are passed to `lngalprior` when constructing the
        Galactic prior.

    Nmc_prior : int, optional
        The number of Monte Carlo realizations used to estimate the
        integral over the prior. Default is `100`.

    lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
        Log-prior grid to be used. If not provided, will default
        to `0.`.

    wt_thresh : float, optional
        The threshold `wt_thresh * max(y_wt)` used to ignore models
        with (relatively) negligible weights.
        Default is `5e-3`.

    cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
        models with (relatively) negligible weights when resampling.
        This option is only used when `wt_thresh=None`.
        Default is `2e-3`.

    lngalprior : func, optional
        The log-prior function to be applied based on a 3-D Galactic model.
        If not provided, this will default to a modified Galactic model
        from Green et al. (2014) that places priors on distance,
        metallicity, and age.

    lndustprior : func, optional
        The log-dust prior function to be applied. If not provided,
        this will default to the 3-D dust map from Green et al. (2019).

    dustfile : str, optional
        The filepath to the 3-D dust map. Default is `bayestar2017_v1.h5`.

    dlabels : bool, optional
        The model labels to be passed the Galactic prior to
        apply any additional priors on the parameters.

    avlim : 2-tuple, optional
        The bounds where Av predictions are reliable.
        Default is `(0., 20.)`.

    rvlim : 2-tuple, optional
        The lower and upper bound where the reddening vector shape changes
        are reliable. Default is `(1., 8.)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    apply_av_prior : bool, optional
        Whether to apply the 3-D dust prior. Default is `True`.

    return_distreds : bool, optional
        Whether to return weighted distance and reddening draws (in units of
        kpc and Av, respectively). Default is `True`.

    Returns
    -------
    sel : `~numpy.ndarray` of shape `(Nsel,)`
        Array of indices selecting out the subset of models with reasonable
        fits.

    lnpost : `~numpy.ndarray` of shape `(Nsel,)`
        The modified log-posteriors for the subset of models with
        reasonable fits.

    dists : `~numpy.ndarray` of shape `(Nsel, Nmc_prior)`, optional
        The dist draws for each selected model.

    reds : `~numpy.ndarray` of shape `(Nsel, Nmc_prior)`, optional
       The reddening draws (Av) for each selected model.

    dreds : `~numpy.ndarray` of shape `(Nsel, Nmc_prior)`, optional
       The differential reddening draws (Rv) for each selected model.

    logwts : `~numpy.ndarray` of shape `(Nsel, Nmc_prior)`, optional
        The ln(weights) for each selected model.

    """

    # Initialize values.
    if wt_thresh is None and cdf_thresh is None:
        wt_thresh = -np.inf  # default to no clipping/thresholding
    if rstate is None:
        try:
            # Attempt to use intel-specific version.
            rstate = np.random_intel
        except:
            # Fall back to regular np.random if not present.
            rstate = np.random
            pass
    mvn = sample_multivariate_normal  # assign alias
    if parallax is not None and parallax_err is None:
        raise ValueError("Must provide both `parallax` and "
                         "`parallax_err`.")

    # Initialize log(prior).
    if lnprior is None:
        lnprior = 0.

    # Initialize distance log(prior).
    if lngalprior is None and coord is None:
        raise ValueError("`coord` must be provided if using the "
                         "default Galactic model prior.")
    if lndustprior is None and coord is None and apply_av_prior:
        raise ValueError("`coord` must be provided if using the "
                         "default dust prior.")
    if lngalprior is None:
        lngalprior = gal_lnprior
    if lndustprior is None:
        lndustprior = dust_lnprior
    if coord is None:
        coord = np.zeros(2)

    # Grab results.
    lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results

    # Compute initial log-posteriors.
    lnp = lnlike + lnprior

    # Apply rough parallax prior for clipping.
    if parallax is not None and parallax_err is not None:
        ds2 = icovs_sar[:, 0, 0]
        scales_err = 1. / np.sqrt(np.abs(ds2))  # approximate scale errors
        lnprob = lnp + scale_parallax_lnprior(scales, scales_err,
                                              parallax, parallax_err)
    else:
        lnprob = lnp
    lnprob_mask = np.where(~np.isfinite(lnprob))[0]  # safety check
    if len(lnprob_mask) > 0:
        lnprob[lnprob_mask] = -1e300

    # Apply thresholding.
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        lwt_min = np.log(wt_thresh) + np.max(lnprob)
        sel = np.where(lnprob > lwt_min)[0]
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(lnprob)
        prob = np.exp(lnprob - logsumexp(lnprob))
        cdf = np.cumsum(prob[idx_sort])
        sel = idx_sort[cdf <= (1. - cdf_thresh)]
    lnp = lnp[sel]
    scale, av, rv = scales[sel], avs[sel], rvs[sel]
    icov_sar = icovs_sar[sel]
    Nsel = len(sel)

    # Generate covariance matrices for the selected fits.
    cov_sar = _inverse3(icov_sar)

    # Ensure final set of matrices are positive semidefinite.
    not_psd = np.where(~np.all(np.linalg.eigvals(cov_sar) > 0, axis=1))[0]
    count = 1
    while len(not_psd) > 0:
        # Regularize non-PSD matrices by adding Gaussian prior with 5% width.
        # The amount of times this is added increases with each pass.
        icov_sar[not_psd] += np.array([np.diag([count / sfrac**2,
                                                count / 0.05**2,
                                                count / 0.05**2])
                                       for sfrac in scale[not_psd] * 0.05])
        cov_sar[not_psd] = _inverse3(icov_sar[not_psd])
        new_idx = np.where(~np.all(np.linalg.eigvals(cov_sar[not_psd]) > 0,
                                   axis=1))[0]
        not_psd = not_psd[new_idx]
        count += 1

    # Now actually apply priors.
    if Nmc_prior > 0:
        # Use Monte Carlo integration to get an estimate of the
        # overlap integral.
        s_mc, a_mc, r_mc = mvn(np.transpose([scale, av, rv]), cov_sar,
                               size=Nmc_prior, rstate=rstate)
        if dlabels is not None:
            dlabels_mc = np.tile(dlabels[sel], Nmc_prior).reshape(-1, Nsel)
        else:
            dlabels_mc = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            par_mc = np.sqrt(s_mc)
            dist_mc = 1. / par_mc
            # Evaluate Galactic prior.
            lnp_mc = lngalprior(dist_mc, coord, labels=dlabels_mc)
            # Evaluate dust prior.
            if apply_av_prior:
                lnp_mc += lndustprior(dist_mc, coord, a_mc, dustfile=dustfile)
        # Evaluate parallax prior.
        if parallax is not None and parallax_err is not None:
            lnp_mc += parallax_lnprior(par_mc, parallax, parallax_err)
        # Ignore points that are out of bounds.
        inbounds = ((s_mc >= 1e-20) &
                    (a_mc >= avlim[0]) & (a_mc <= avlim[1]) &
                    (r_mc >= rvlim[0]) & (r_mc <= rvlim[1]))
        lnp_mc[~inbounds] = -1e300
        Nmc_prior_eff = np.sum(inbounds, axis=0)
        # Compute integral.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            lnp += logsumexp(lnp_mc, axis=0) - np.log(Nmc_prior_eff)
    else:
        # Just assume the maximum-likelihood estimate.
        lnp = lnprob[sel]

    lnp_mask = np.where(~np.isfinite(lnp))[0]  # safety check
    if len(lnp_mask) > 0:
        lnp[lnp_mask] = -1e300

    return sel, lnp, dist_mc.T, a_mc.T, r_mc.T, lnp_mc.T


class BruteForce():
    """
    Fits data and generates predictions for scale-factors and reddening
    over a grid in stellar parameters.

    """

    def __init__(self, models, models_labels, labels_mask):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
            Magnitude coefficients used to compute reddened photometry over
            the desired bands for all models on the grid.

        models_labels : structured `~numpy.ndarray` of shape `(Nmodel, Nlabel)`
            Labels corresponding to each model on the grid.

        labels_mask : structured `~numpy.ndarray` of shape `(1, Nlabel)`
            Masks corresponding to each label to indicate whether it is
            an ancillary prediction (e.g., `logt`) or was used to compute
            the model grid (e.g., `feh`).

        """

        # Initialize values.
        self.NMODEL, self.NDIM, self.NCOEF = models.shape
        self.models = models
        self.models_labels = models_labels
        self.labels_mask = labels_mask
        self.NLABELS = len(models_labels[0])

    def _setup(self, data, data_err, data_mask, data_labels=None,
               phot_offsets=None, parallax=None, parallax_err=None,
               av_gauss=None, lnprior=None,
               wt_thresh=5e-3, cdf_thresh=2e-3,
               apply_agewt=True, apply_grad=True,
               lngalprior=None, lndustprior=None, dustfile=None,
               data_coords=None, ltol_subthresh=1e-2,
               logl_initthresh=5e-3, mag_max=50., merr_max=0.25,
               rstate=None):
        """
        Pre-process data and perform a bunch of safety checks.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Observed data values.

        data_err : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Binary mask (0/1) indicating whether the data value was observed.

        data_labels : `~numpy.ndarray` of shape `(Ndata, Nlabels)`
            Labels for the data to be stored during runtime.

        phot_offsets : `~numpy.ndarray` of shape `(Nfilt)`, optional
            Multiplicative photometric offsets that will be applied to
            the data (i.e. `data_new = data * phot_offsets`) and errors
            when provided.

        parallax : `~numpy.ndarray` of shape `(Ndata)`, optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape `(Ndata)`, optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        av_gauss : 2-tuple, optional
            The mean and standard deviation of a Gaussian prior on A(V).
            If provided, this will be used in lieu of the default
            distance-reddening prior and incorporated directly into the fits.

        lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
            Log-prior grid to be used. If not provided, this will default
            to [1] a Kroupa IMF prior in initial mass (`'mini'`) and
            uniform priors in age, metallicity, and dust if we are using the
            MIST models and [2] a PanSTARRS r-band luminosity function-based
            prior if we are using the Bayestar models. Unresolved binaries
            with secondary mass fractions of `'smf'` are treated as
            being drawn from a uniform prior from `smf=[0., 1.]`.
            **Be sure to check this behavior you are using custom models.**

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights when resampling.
            Default is `5e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            models with (relatively) negligible weights when resampling.
            This option is only used when `wt_thresh=None`.
            Default is `2e-3`.

        apply_agewt : bool, optional
            Whether to apply the age weights derived from the MIST models
            to reweight from EEP to age. Default is `True`.

        apply_grad : bool, optional
            Whether to account for the grid spacing using `np.gradient`.
            Default is `True`.

        lngalprior : func, optional
            The log-prior function to be applied based on a 3-D Galactic model.
            If not provided, this will default to a modified Galactic model
            from Green et al. (2014) that places priors on distance,
            metallicity, and age.

        lndustprior : func, optional
            The log-dust prior function to be applied. If not provided,
            this will default to the 3-D dust map from the `dustfile` provided
            or just a flat prior on A(V) if no 3-D dust map is provided.

        dustfile : str, optional
            The filepath to the 3-D dust map.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`, optional
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lngalprior` when constructing the
            Galactic prior.

        ltol_subthresh : float, optional
            The threshold used to sub-select the best-fit log-likelihoods used
            to determine convergence. Default is `1e-2`.

        logl_initthresh : float, optional
            The threshold `logl_initthresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights after computing the initial
            set of fits but before optimizing them. Default is `5e-3`.
            **This must be smaller than or equal to `ltol_subthresh`.**

        mag_max : float, optional
            The maximum allowed magnitude (converted from the provided
            fluxes) used for internal masking. Default is `50.`.

        merr_max : float, optional
            The maximum allowed magnitude error (converted from the provided
            fluxes) used for internal masking. Default is `0.25`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        Returns
        -------
        data : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Observed data values.

        data_err : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Binary mask (0/1) indicating whether the data value was observed.

        data_labels : `~numpy.ndarray` of shape `(Ndata, Nlabels)`
            Labels for the data to be stored during runtime.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lngalprior` when constructing the
            Galactic prior.

        lnprior : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
            Log-prior grid to be used.

        lngalprior : func
            The log-prior function to be applied based on a 3-D Galactic model.

        lndustprior : func, optional
            The log-dust prior function to be applied.

        av_gauss : 2-tuple or `None`
            The mean and standard deviation of a Gaussian prior on A(V).

        wt_thresh : float
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights when resampling.

        rstate : `~numpy.random.RandomState`
            `~numpy.random.RandomState` instance.

        """

        # Reorder data to Fortran ordering.
        data = np.asfortranarray(data)
        data_err = np.asfortranarray(data_err)
        data_mask = np.asfortranarray(data_mask)
        if data_labels is not None:
            data_labels = np.asfortranarray(data_labels)
        Ndata, Nfilt = data.shape

        # Check fitting thresholds.
        if logl_initthresh > ltol_subthresh:
            raise ValueError("The initial threshold must be smaller than "
                             "or equal to the convergence threshold in order "
                             "to be useful!")

        # Check clipping thresholds.
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding

        # Check random state.
        if rstate is None:
            try:
                # Attempt to use intel-specific version.
                rstate = np.random_intel
            except:
                # Fall back to regular np.random if not present.
                rstate = np.random
                pass

        # Check parallaxes.
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")

        # Check photometric offsets.
        if phot_offsets is None:
            phot_offsets = np.ones(Nfilt)

        # Initialize log(prior).
        if lnprior is None:
            try:
                # Set IMF prior.
                lnprior = imf_lnprior(self.models_labels['mini'])
            except:
                # Set PS1 r-band LF prior.
                lnprior = ps1_MrLF_lnprior(self.models_labels['Mr'])
                pass

        # Apply age weights to reweight from EEP to age.
        if apply_agewt:
            try:
                lnprior += np.log(np.abs(self.models_labels['agewt']))
            except:
                pass

        # Reweight based on spacing.
        if apply_grad:
            for l in self.models_labels.dtype.names:
                label = self.models_labels[l]
                if self.labels_mask[l][0]:
                    ulabel = np.unique(label)  # grab underlying grid
                    if len(ulabel) > 1:
                        # Compute and add gradient.
                        lngrad_label = np.log(np.gradient(ulabel))
                        lnprior += np.interp(label, ulabel, lngrad_label)

        # Initialize Galactic log(prior).
        if lngalprior is None and data_coords is None:
            raise ValueError("`data_coords` must be provided if using the "
                             "default Galactic model prior.")
        if lngalprior is None:
            lngalprior = gal_lnprior

        # Initialize 3-D dust log(prior).
        if lndustprior is None and dustfile is not None:
            # Use a 3-D dust map as a prior.
            lndustprior = dust_lnprior

            # Check if `data_coords` are provided.
            if data_coords is None:
                raise ValueError("`data_coords` must be provided if using the "
                                 "default dust prior.")

            # Check provided `dustfile` is valid.
            try:
                # Try reading in parallel-friendly way if possible.
                try:
                    h5py.File(dustfile, 'r', libver='latest', swmr=True)
                except:
                    h5py.File(dustfile, 'r')
                    pass
            except:
                raise ValueError("The default dust prior is being used but "
                                 "the relevant data file is not located at "
                                 "the provided `dustpath`.")
            try:
                # Pre-load provided dustfile into default prior.
                lndustprior(np.linspace(0, 100), np.array([180., 90.]),
                            np.linspace(0, 100), dustfile=dustfile)
            except:
                pass
        elif lndustprior is None and av_gauss is None:
            # If no prior is provided, default to flat Av prior.
            av_gauss = (0, 1e6)

        # Fill data coordinates.
        if data_coords is None:
            data_coords = np.zeros((Ndata, 2))

        # Clean data to remove bad photometry user may not have masked.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values
            mag, err = magnitude(data, data_err)
            bad_mag = (mag > mag_max) | (err > merr_max)
            clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
            data_mask *= (clean & ~bad_mag)

        # Check there are enough bands to fit.
        Nbmin = 4  # minimum number of bands needed
        if np.any(np.sum(data_mask, axis=1) < Nbmin):
            raise ValueError("Objects with fewer than {0} bands of "
                             "acceptable photometry are currently included in "
                             "the dataset. These objects give degenerate fits "
                             "and cannot be properly modeled. Please remove "
                             "these objects or modify `mag_max` or `merr_max`."
                             .format(Nbmin))

        return (data * phot_offsets, data_err * phot_offsets, data_mask,
                data_labels, data_coords, lnprior, lngalprior, lndustprior,
                av_gauss, wt_thresh, rstate)

    def fit(self, data, data_err, data_mask, data_labels, save_file,
            phot_offsets=None, parallax=None, parallax_err=None,
            Nmc_prior=50, avlim=(0., 20.), av_gauss=None,
            rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
            lnprior=None, wt_thresh=5e-3, cdf_thresh=2e-3, Ndraws=250,
            apply_agewt=True, apply_grad=True,
            lngalprior=None, lndustprior=None, dustfile=None,
            apply_dlabels=True, data_coords=None, logl_dim_prior=True,
            ltol=3e-2, ltol_subthresh=1e-2, logl_initthresh=5e-3,
            mag_max=50., merr_max=0.25, rstate=None, save_dar_draws=True,
            running_io=True, verbose=True):
        """
        Fit all input models to the input data to compute the associated
        log-posteriors.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Observed data values.

        data_err : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Binary mask (0/1) indicating whether the data value was observed.

        data_labels : `~numpy.ndarray` of shape `(Ndata, Nlabels)`
            Labels for the data to be stored during runtime.

        save_file : str, optional
            File where results will be written out in HDF5 format.

        phot_offsets : `~numpy.ndarray` of shape `(Nfilt)`, optional
            Multiplicative photometric offsets that will be applied to
            the data (i.e. `data_new = data * phot_offsets`) and errors
            when provided.

        parallax : `~numpy.ndarray` of shape `(Ndata)`, optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape `(Ndata)`, optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        Nmc_prior : int, optional
            The number of Monte Carlo realizations used to estimate the
            integral over the prior. Default is `100`.

        avlim : 2-tuple, optional
            The bounds where Av predictions are reliable.
            Default is `(0., 20.)`.

        av_gauss : 2-tuple, optional
            The mean and standard deviation of a Gaussian prior on A(V).
            If provided, this will be used in lieu of the default
            distance-reddening prior and incorporated directly into the fits.

        rvlim : 2-tuple, optional
            The lower and upper bound where the reddening vector shape changes
            are reliable. Default is `(1., 8.)`.

        rv_gauss : 2-tuple, optional
            The mean and standard deviation of the Gaussian prior on R(V).
            The default is `(3.32, 0.18)` based on the results from
            Schlafly et al. (2016).

        lnprior : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`, optional
            Log-prior grid to be used. If not provided, this will default
            to [1] a Kroupa IMF prior in initial mass (`'mini'`) and
            uniform priors in age, metallicity, and dust if we are using the
            MIST models and [2] a PanSTARRS r-band luminosity function-based
            prior if we are using the Bayestar models. Unresolved binaries
            with secondary mass fractions of `'smf'` are treated as
            being drawn from a uniform prior from `smf=[0., 1.]`.
            **Be sure to check this behavior you are using custom models.**

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights when resampling.
            Default is `5e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            models with (relatively) negligible weights when resampling.
            This option is only used when `wt_thresh=None`.
            Default is `2e-3`.

        Ndraws : int, optional
            The number of realizations of the brute-force PDF to save
            to disk. Indices, scales, and scale errors are saved.
            Default is `250`.

        apply_agewt : bool, optional
            Whether to apply the age weights derived from the MIST models
            to reweight from EEP to age. Default is `True`.

        apply_grad : bool, optional
            Whether to account for the grid spacing using `np.gradient`.
            Default is `True`.

        lngalprior : func, optional
            The log-prior function to be applied based on a 3-D Galactic model.
            If not provided, this will default to a modified Galactic model
            from Green et al. (2014) that places priors on distance,
            metallicity, and age.

        lndustprior : func, optional
            The log-dust prior function to be applied. If not provided,
            this will default to the 3-D dust map from the `dustfile` provided
            or just a flat prior on A(V) if no 3-D dust map is provided.

        dustfile : str, optional
            The filepath to the 3-D dust map.

        apply_dlabels : bool, optional
            Whether to pass the model labels to the Galactic prior to
            apply any additional priors on the parameters.
            Default is `True`.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`, optional
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lngalprior` when constructing the
            Galactic prior.

        logl_dim_prior : bool, optional
            Whether to apply a dimensional-based correction (prior) to the
            log-likelihood. Transforms the likelihood to a chi2 distribution
            with `Nfilt - 3` degrees of freedom. Default is `True`.

        ltol : float, optional
            The weighted tolerance in the computed log-likelihoods used to
            determine convergence. Default is `3e-2`.

        ltol_subthresh : float, optional
            The threshold used to sub-select the best-fit log-likelihoods used
            to determine convergence. Default is `1e-2`.

        logl_initthresh : float, optional
            The threshold `logl_initthresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights after computing the initial
            set of fits but before optimizing them. Default is `5e-3`.
            **This must be smaller than or equal to `ltol_subthresh`.**

        mag_max : float, optional
            The maximum allowed magnitude (converted from the provided
            fluxes) used for internal masking. Default is `50.`.

        merr_max : float, optional
            The maximum allowed magnitude error (converted from the provided
            fluxes) used for internal masking. Default is `0.25`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        save_dar_draws : bool, optional
            Whether to save distance (kpc), reddening (Av), and
            dust curve shape (Rv) draws. Default is `True`.

        running_io : bool, optional
            Whether to write out the result from each fit in real time. If
            `True`, then the results from each object will be saved as soon as
            they are complete to prevent losing a batch of objects if something
            goes wrong. If `False`, the results will be stored internally
            until the end of the run, when they will all be written together.
            This may be better if working on a filesystem with slow I/O
            operations. Default is `True`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        # Pre-process data and initialize functions.
        (data, data_err, data_mask, data_labels, data_coords,
         lnprior, lngalprior, lndustprior, av_gauss, wt_thresh,
         rstate) = self._setup(data, data_err, data_mask, data_labels,
                               phot_offsets=phot_offsets, parallax=parallax,
                               parallax_err=parallax_err, av_gauss=av_gauss,
                               lnprior=lnprior, wt_thresh=wt_thresh,
                               cdf_thresh=cdf_thresh, apply_agewt=apply_agewt,
                               apply_grad=apply_grad, lngalprior=lngalprior,
                               lndustprior=lndustprior, dustfile=dustfile,
                               data_coords=data_coords,
                               ltol_subthresh=ltol_subthresh,
                               logl_initthresh=logl_initthresh,
                               mag_max=mag_max, merr_max=merr_max,
                               rstate=rstate)
        Ndata, Nfilt = data.shape

        # Initialize results file.
        out = h5py.File("{0}.h5".format(save_file), "w-")
        out.create_dataset("labels", data=data_labels)
        if running_io:
            out.create_dataset("model_idx", data=np.full((Ndata, Ndraws), -99,
                                                         dtype='int32'))
            out.create_dataset("ml_scale", data=np.ones((Ndata, Ndraws),
                                                        dtype='float32'))
            out.create_dataset("ml_av", data=np.zeros((Ndata, Ndraws),
                                                      dtype='float32'))
            out.create_dataset("ml_rv", data=np.zeros((Ndata, Ndraws),
                                                      dtype='float32'))
            out.create_dataset("ml_cov_sar", data=np.zeros((Ndata, Ndraws,
                                                            3, 3),
                                                           dtype='float32'))
            out.create_dataset("obj_log_post", data=np.zeros((Ndata, Ndraws),
                                                             dtype='float32'))
            out.create_dataset("obj_log_evid", data=np.zeros(Ndata,
                                                             dtype='float32'))
            out.create_dataset("obj_chi2min", data=np.zeros(Ndata,
                                                            dtype='float32'))
            out.create_dataset("obj_Nbands", data=np.zeros(Ndata,
                                                           dtype='int16'))
            if save_dar_draws:
                out.create_dataset("samps_dist", data=np.ones((Ndata, Ndraws),
                                                              dtype='float32'))
                out.create_dataset("samps_red", data=np.ones((Ndata, Ndraws),
                                                             dtype='float32'))
                out.create_dataset("samps_dred", data=np.ones((Ndata, Ndraws),
                                                              dtype='float32'))
                out.create_dataset("samps_logp", data=np.ones((Ndata, Ndraws),
                                                              dtype='float32'))
        else:
            idxs_arr = np.full((Ndata, Ndraws), -99, dtype='int32')
            scale_arr = np.ones((Ndata, Ndraws), dtype='float32')
            av_arr = np.zeros((Ndata, Ndraws), dtype='float32')
            rv_arr = np.zeros((Ndata, Ndraws), dtype='float32')
            cov_arr = np.zeros((Ndata, Ndraws, 3, 3), dtype='float32')
            logpost_arr = np.zeros((Ndata, Ndraws), dtype='float32')
            logevid_arr = np.zeros(Ndata, dtype='float32')
            chi2best_arr = np.zeros(Ndata, dtype='float32')
            nbands_arr = np.ones(Ndata, dtype='int16')
            if save_dar_draws:
                dist_arr = np.ones((Ndata, Ndraws), dtype='float32')
                red_arr = np.ones((Ndata, Ndraws), dtype='float32')
                dred_arr = np.ones((Ndata, Ndraws), dtype='float32')
                logp_arr = np.ones((Ndata, Ndraws), dtype='float32')

        # Fit data.
        if verbose:
            t1 = time.time()
            t = 0.
            sys.stderr.write('\rFitting object {0}/{1}  '.format(1, Ndata))
            sys.stderr.flush()
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              parallax=parallax,
                                              parallax_err=parallax_err,
                                              avlim=avlim, rvlim=rvlim,
                                              av_gauss=av_gauss,
                                              rv_gauss=rv_gauss,
                                              Nmc_prior=Nmc_prior,
                                              lnprior=lnprior,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              Ndraws=Ndraws, rstate=rstate,
                                              lngalprior=lngalprior,
                                              lndustprior=lndustprior,
                                              dustfile=dustfile,
                                              apply_dlabels=apply_dlabels,
                                              data_coords=data_coords,
                                              return_distreds=save_dar_draws,
                                              ltol_subthresh=ltol_subthresh,
                                              logl_dim_prior=logl_dim_prior,
                                              logl_initthresh=logl_initthresh,
                                              ltol=ltol)):

            # Grab results.
            if save_dar_draws:
                (idxs, scales, avs, rvs, covs_sar, Ndim,
                 lpost, levid, chi2min, dists, reds, dreds, logwt) = results
            else:
                (idxs, scales, avs, rvs, covs_sar,
                 Ndim, lpost, levid, chi2min) = results

            # Print progress.
            if verbose and i < Ndata - 1:
                # Compute time stamps.
                t2 = time.time()
                dt = t2 - t1  # time for current object
                t1 = t2
                t += dt  # total time elapsed
                t_avg = t / (i + 1)  # avg time per object
                t_est = t_avg * (Ndata - i - 1)  # estimated remaining time
                sys.stderr.write('\rFitting object {:d}/{:d} '
                                 '[chi2/n: {:2.1f}/{:d}] '
                                 '(mean time: {:2.3f} s/obj, '
                                 'est. remaining: {:10.3f} s)    '
                                 .format(i + 2, Ndata, chi2min, Ndim,
                                         t_avg, t_est))
                sys.stderr.flush()

            # Save results.
            if running_io:
                out['model_idx'][i] = idxs
                out['ml_scale'][i] = scales
                out['ml_av'][i] = avs
                out['ml_rv'][i] = rvs
                out['ml_cov_sar'][i] = covs_sar
                out['obj_Nbands'][i] = Ndim
                out['obj_log_post'][i] = lpost
                out['obj_log_evid'][i] = levid
                out['obj_chi2min'][i] = chi2min
                if save_dar_draws:
                    out['samps_dist'][i] = dists
                    out['samps_red'][i] = reds
                    out['samps_dred'][i] = dreds
                    out['samps_logp'][i] = logwt
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
                if save_dar_draws:
                    dist_arr[i] = dists
                    red_arr[i] = reds
                    dred_arr[i] = dreds
                    logp_arr[i] = logwt

        if verbose:
            # Compute time stamps.
            t2 = time.time()
            dt = t2 - t1  # time for current object
            t1 = t2
            t += dt  # total time elapsed
            t_avg = t / (i + 1)  # avg time per object
            t_est = t_avg * (Ndata - i - 1)  # estimated remaining time
            sys.stderr.write('\rFitting object {:d}/{:d} '
                             '[chi2/n: {:2.1f}/{:d}] '
                             '(mean time: {:2.3f} s/obj, '
                             'est. time remaining: {:10.3f} s)    '
                             .format(i + 1, Ndata, chi2min, Ndim,
                                     t_avg, t_est))
            sys.stderr.flush()
            sys.stderr.write('\n')
            sys.stderr.flush()

        # Dump results to disk if we have disabled running I/O.
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
            if save_dar_draws:
                out.create_dataset("samps_dist", data=dist_arr)
                out.create_dataset("samps_red", data=red_arr)
                out.create_dataset("samps_dred", data=dred_arr)
                out.create_dataset("samps_logp", data=logp_arr)

        # Close the output file.
        out.close()

    def _fit(self, data, data_err, data_mask,
             parallax=None, parallax_err=None, Nmc_prior=100,
             avlim=(0., 20.), av_gauss=None,
             rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
             lnprior=None, wt_thresh=5e-3, cdf_thresh=2e-3, Ndraws=250,
             lngalprior=None, lndustprior=None, dustfile=None,
             apply_dlabels=True, data_coords=None,
             return_distreds=True, logl_dim_prior=True, ltol=3e-2,
             ltol_subthresh=1e-2, logl_initthresh=5e-3, rstate=None):
        """
        Internal generator used to compute fits.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Model values.

        data_err : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape `(Ndata, Nfilt)`
            Binary mask (0/1) indicating whether the data value was observed.

        parallax : `~numpy.ndarray` of shape `(Ndata)`, optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape `(Ndata)`, optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        Nmc_prior : int, optional
            The number of Monte Carlo realizations used to estimate the
            integral over the prior. Default is `100`.

        avlim : 2-tuple, optional
            The bounds where Av predictions are reliable.
            Default is `(0., 20.)`.

        av_gauss : 2-tuple, optional
            The mean and standard deviation of a Gaussian prior on A(V).
            If provided, this will be used in lieu of the default
            distance-reddening prior and incorporated directly into the fits.

        rvlim : 2-tuple, optional
            The lower and upper bound where the reddening vector shape changes
            are reliable. Default is `(1., 8.)`.

        rv_gauss : 2-tuple, optional
            The mean and standard deviation of the Gaussian prior on R(V).
            The default is `(3.32, 0.18)` based on the results from
            Schlafly et al. (2016).

        lnprior : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`, optional
            Log-prior grid to be used. If not provided, will default
            to `0.`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights.
            Default is `5e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            models with (relatively) negligible weights when resampling.
            This option is only used when `wt_thresh=None`.
            Default is `2e-3`.

        Ndraws : int, optional
            The number of realizations of the brute-force PDF to save
            to disk. Indices, scales, and scale errors are saved.
            Default is `250`.

        lngalprior : func, optional
            The log-prior function to be applied based on a 3-D Galactic model.
            If not provided, this will default to a modified Galactic model
            from Green et al. (2014) that places priors on distance,
            metallicity, and age.

        lndustprior : func, optional
            The log-dust prior function to be applied. If not provided,
            this will default to the 3-D dust map from the `dustfile` provided
            or just a flat prior on A(V) if no 3-D dust map is provided.

        dustfile : str, optional
            The filepath to the 3-D dust map.

        apply_dlabels : bool, optional
            Whether to pass the model labels to the Galactic prior to
            apply any additional priors on the parameters.
            Default is `True`.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`, optional
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lngalprior` when constructing the
            Galactic prior.

        return_distreds : bool, optional
            Whether to return distance and reddening draws (in units of kpc
            and Av, respectively). Default is `True`.

        logl_dim_prior : bool, optional
            Whether to apply a dimensional-based correction (prior) to the
            log-likelihood. Transforms the likelihood to a chi2 distribution
            with `Nfilt - 3` degrees of freedom. Default is `True`.

        ltol : float, optional
            The weighted tolerance in the computed log-likelihoods used to
            determine convergence. Default is `3e-2`.

        ltol_subthresh : float, optional
            The threshold used to sub-select the best-fit log-likelihoods used
            to determine convergence. Default is `1e-2`.

        logl_initthresh : float, optional
            The threshold `logl_initthresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights after computing the initial
            set of fits but before optimizing them. Default is `5e-3`.
            **This must be smaller than or equal to `ltol_subthresh`.**

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        Returns
        -------
        results : tuple
            Outputs yielded from the generator.

        """

        # Pre-process data and initialize functions.
        (data, data_err, data_mask, data_labels, data_coords,
         lnprior, lngalprior, lndustprior, av_gauss, wt_thresh,
         rstate) = self._setup(data, data_err, data_mask, data_labels=None,
                               phot_offsets=None, parallax=parallax,
                               parallax_err=parallax_err, av_gauss=av_gauss,
                               lnprior=lnprior, wt_thresh=wt_thresh,
                               cdf_thresh=cdf_thresh, apply_agewt=False,
                               apply_grad=False, lngalprior=lngalprior,
                               lndustprior=lndustprior, dustfile=dustfile,
                               data_coords=data_coords,
                               ltol_subthresh=ltol_subthresh,
                               logl_initthresh=logl_initthresh,
                               mag_max=np.inf, merr_max=np.inf,
                               rstate=rstate)
        models = np.array(self.models, order='F')
        apply_av_prior = av_gauss is None  # check whether to apply A(V) prior
        if apply_dlabels:
            dlabels = self.models_labels
        else:
            dlabels = None

        # Main generator for fitting data.
        Ndata, Nfilt = data.shape
        for i in range(Ndata):

            # Compute "log-likelihoods" optimizing over s, Av, and Rv.
            results = loglike(data[i], data_err[i], data_mask[i], models,
                              avlim=avlim, av_gauss=av_gauss,
                              rvlim=rvlim, rv_gauss=rv_gauss,
                              dim_prior=logl_dim_prior, ltol=ltol,
                              ltol_subthresh=ltol_subthresh,
                              init_thresh=logl_initthresh,
                              parallax=parallax[i],
                              parallax_err=parallax_err[i],
                              return_vals=True)
            lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results

            # Compute posteriors using Monte Carlo sampling and integration.
            presults = lnpost(results, parallax=parallax[i],
                              parallax_err=parallax_err[i],
                              coord=data_coords[i],
                              Nmc_prior=Nmc_prior, lnprior=lnprior,
                              wt_thresh=wt_thresh, cdf_thresh=cdf_thresh,
                              lngalprior=lngalprior, lndustprior=lndustprior,
                              dustfile=dustfile, dlabels=dlabels,
                              avlim=avlim, rvlim=rvlim,
                              rstate=rstate, apply_av_prior=apply_av_prior)
            sel, lnprob, dists, reds, dreds, logwts = presults
            Nsel = len(sel)

            # Add in parallax to chi2 and Ndim if provided.
            if np.isfinite(parallax[i]) and np.isfinite(parallax_err[i]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ignore bad values
                    chi2 += ((np.sqrt(scales) - parallax[i])**2
                             / parallax_err[i]**2)
                    Ndim += 1

            # Compute GOF metrics.
            chi2min = np.min(chi2[sel])
            levid = logsumexp(lnprob)

            # Resample.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore bad values
                wt = np.exp(lnprob - levid)
                wt /= wt.sum()  # deal with numerical precision issues
                idxs = rstate.choice(Nsel, size=Ndraws, p=wt)
                sidxs = sel[idxs]

            # Grab/compute corresponding values.
            scales, avs, rvs = scales[sidxs], avs[sidxs], rvs[sidxs]
            cov_sar = _inverse3(icovs_sar[sidxs])
            lnprob = lnprob[idxs]

            if return_distreds:
                # Draw distances and reddenings.
                imc = np.zeros(Ndraws, dtype='int')
                for i, idx in enumerate(idxs):
                    wt = np.exp(logwts[idx] - logsumexp(logwts[idx]))
                    wt /= wt.sum()  # deal with numerical precision issues
                    imc[i] = rstate.choice(Nmc_prior, p=wt)
                dists = dists[idxs, imc]
                reds = reds[idxs, imc]
                dreds = dreds[idxs, imc]
                logwts = logwts[idxs, imc]
                # Return per-object results + draws.
                yield (sidxs, scales, avs, rvs, cov_sar,
                       Ndim, lnprob, levid, chi2min,
                       dists, reds, dreds, logwts)
            else:
                # Return per-object results.
                yield (sidxs, scales, avs, rvs, cov_sar,
                       Ndim, lnprob, levid, chi2min)
