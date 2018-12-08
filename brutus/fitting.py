#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute force fitter.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import os
import warnings
import math
import numpy as np
import warnings
import h5py
import time
from scipy.special import xlogy, gammaln

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .pdf import *
from .utils import *

__all__ = ["loglike", "_optimize_fit", "BruteForce", "_lnpost"]


def loglike(data, data_err, data_mask, mag_coeffs,
            avlim=(0., 6.), av_gauss=(0., 1e6),
            rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
            av_init=None, rv_init=None,
            dim_prior=True, ltol=0.02, wt_thresh=0.005, init_thresh=1e-4,
            return_vals=False, *args, **kwargs):
    """
    Computes the log-likelihood between noisy data and noiseless models
    optimizing over the scale-factor and dust attenuation.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    data_err : `~numpy.ndarray` of shape `(Nfilt)`
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`
        Binary mask (0/1) indicating whether the data was observed.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, 3)`
        Magnitude coefficients used to compute reddened photometry for a given
        model. Contains `(mag, r0, dr)` pairs referencing the unreddening
        magnitudes, the reddening vector as a function of A(V),
        and the change in the reddening vector as a function of R(V).

    avlim : 2-tuple, optional
        The lower and upper bound where the reddened photometry is reliable.
        Default is `(0., 6.)`.

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
        determine convergence. Default is `0.02`.

    wt_thresh : float, optional
        The threshold used to sub-select the best-fit log-likelihoods used
        to determine convergence. Default is `0.005`.

    init_thresh : bool, optional
        The weight threshold used to mask out fits after the initial
        magnitude-based fit before transforming the results back to
        flux density (and iterating until convergence). Default is `1e-4`.

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

    # Initialize values.
    Nmodels, Nfilt, Ncoef = mag_coeffs.shape
    if av_init is None:
        av_init = np.zeros(Nmodels) + av_gauss[0]
    if rv_init is None:
        rv_init = np.zeros(Nmodels) + rv_gauss[0]

    # Clean data (safety checks).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
        data_mask[~clean] = False
    Ndim = sum(data_mask)  # number of dimensions

    # Subselect only clean observations.
    flux, fluxerr = data[data_mask], data_err[data_mask]  # mean, error
    mcoeffs = mag_coeffs[:, data_mask, :]  # model magnitude coefficients
    tot_var = np.square(fluxerr) + np.zeros((Nmodels, Ndim))  # total variance

    # Get started by fitting in magnitudes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Convert to magnitudes.
        mags = -2.5 * np.log10(flux)
        mags_var = np.square(2.5 / np.log(10.)) * tot_var / np.square(flux)
        mclean = np.isfinite(mags)
        mags[~mclean], mags_var[:, ~mclean] = 0., 1e50  # mask negative values

    # Compute unreddened photometry.
    models, rvecs, drvecs = get_seds(mcoeffs, av=av_init, rv=rv_init,
                                     return_rvec=True, return_drvec=True)

    # Compute initial magnitude fit.
    results = _optimize_fit(flux, tot_var, models, rvecs, drvecs,
                            av_init, rv_init, mcoeffs,
                            tol=2.5*ltol, init_thresh=init_thresh,
                            resid=None, mags=mags, mags_var=mags_var,
                            avlim=avlim, av_gauss=av_gauss,
                            rvlim=rvlim, rv_gauss=rv_gauss)
    models, rvecs, drvecs, scale, av, rv, icov_sar, resid = results

    if init_thresh is not None:
        # Cull initial bad fits before moving on.
        chi2 = np.sum(np.square(resid) / tot_var, axis=1)
        # Compute multivariate normal logpdf.
        lnl = -0.5 * chi2
        lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))
        # Subselect models.
        init_sel = np.where(lnl > np.max(lnl) + np.log(init_thresh))[0]
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
        av_new = np.array(av)
        rv_new = np.array(rv)

    # Iterate until convergence.
    lnl_old, lerr = -1e300, 1e300
    stepsize, rescaling = np.ones(Nmodels)[init_sel], 1.2
    while lerr > ltol:

        # Re-compute models.
        results = _optimize_fit(flux, tot_var, models, rvecs, drvecs,
                                av_new, rv_new, mcoeffs,
                                avlim=avlim, av_gauss=av_gauss,
                                rvlim=rvlim, rv_gauss=rv_gauss,
                                resid=resid, stepsize=stepsize)
        (models, rvecs, drvecs,
         scale_new, av_new, rv_new, icov_sar_new, resid) = results

        # Compute chi2.
        chi2_new = np.sum(np.square(resid) / tot_var, axis=1)

        # Compute multivariate normal logpdf.
        lnl_new = -0.5 * chi2_new
        lnl_new += -0.5 * (Ndim * np.log(2. * np.pi) +
                           np.sum(np.log(tot_var), axis=1))

        # Compute stopping criterion.
        lnl_sel = np.where(lnl_new > np.max(lnl_new) + np.log(wt_thresh))[0]
        lerr = np.max(np.abs(lnl_new - lnl_old)[lnl_sel])

        # Adjust stepsize.
        stepsize[lnl_new < lnl_old] /= rescaling
        lnl_old = lnl_new

    # Insert optimized models into initial array of results.
    lnl[init_sel], chi2[init_sel] = lnl_new, chi2_new
    scale[init_sel], av[init_sel], rv[init_sel] = scale_new, av_new, rv_new
    icov_sar[init_sel] = icov_sar_new

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 3)  # effective dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    if return_vals:
        return lnl, Ndim, chi2, scale, av, rv, icov_sar
    else:
        return lnl, Ndim, chi2


def _optimize_fit(data, tot_var, models, rvecs, drvecs, av, rv, mag_coeffs,
                  avlim=(0., 6.), av_gauss=(0., 1e6),
                  rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
                  resid=None, tol=0.05, init_thresh=1e-4, stepsize=1.,
                  mags=None, mags_var=None):
    """
    Optimize the distance and reddening between the models and the data using
    the gradient.

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

    avlim : 2-tuple, optional
        The lower and upper bound where the reddened photometry is reliable.
        Default is `(0., 6.)`.

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

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.
        If not provided, this will be computed.

    tol : float, optional
        The maximum tolerance in the computed Av and Rv values used to
        determine convergence during the magnitude fits. Default is `0.05`.

    init_thresh : bool, optional
        The weight threshold used to mask out fits after the initial
        magnitude-based fit before transforming the results back to
        flux density (and iterating until convergence). Default is `1e-4`.

    stepsize : float or `~numpy.ndarray`, optional
        The stepsize (in units of the computed gradient). Default is `1.`.

    mags : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Observed data values in magnitudes.

    mags_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`, optional
        Associated (Normal) errors on the observed values compared to the
        models in magnitudes.

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

    # Compute residuals.
    if resid is None:
        if mags is not None and mags_var is not None:
            resid = mags - models
        else:
            resid = data - models

    Av_mean, Av_std = av_gauss
    Rv_mean, Rv_std = rv_gauss

    if mags is not None and mags_var is not None:
        # If magnitudes are provided, we can solve the linear system
        # explicitly for `(s_ML, Av_ML, r_ML=Av_ML*Rv_ML)`. We opt to
        # solve for Av and Rv in turn to so we can impose priors and bounds
        # on both quantities.

        err = 1e300
        while err > tol:
            # Solve for Av.
            # Derive partial derivatives.
            s_den = np.sum(1. / mags_var, axis=1)
            a_den = np.sum(np.square(rvecs) / mags_var, axis=1)
            sa_mix = np.sum(rvecs / mags_var, axis=1)
            # Compute residual terms.
            resid_s = np.sum(resid / mags_var, axis=1)
            resid_a = np.sum(resid * rvecs / mags_var, axis=1)
            # Add in Gaussian Av prior.
            resid_a += (Av_mean - av) / Av_std**2
            a_den += 1. / Av_std**2
            # Compute determinants (normalization terms).
            sa_idet = 1. / (s_den * a_den - sa_mix**2)
            # Compute ML solution for Delta_Av.
            dav = sa_idet * (s_den * resid_a - sa_mix * resid_s)

            # Prevent Av from sliding off the provided bounds.
            dav_low, dav_high = avlim[0] - av, avlim[1] - av
            lsel, hsel = dav < dav_low, dav > dav_high
            dav[lsel] = dav_low[lsel]
            dav[hsel] = dav_high[hsel]
            # Increment to new Av.
            av += dav

            # Update residuals.
            resid -= dav[:, None] * rvecs  # update residuals

            # Solve for Rv.
            # Derive partial derivatives.
            s_den = np.sum(1. / mags_var, axis=1)
            r_den = np.sum(np.square(drvecs) / mags_var, axis=1) * av**2
            sr_mix = np.sum(drvecs / mags_var, axis=1) * av
            # Compute residual terms.
            resid_s = np.sum(resid / mags_var, axis=1)
            resid_r = np.sum(resid * drvecs / mags_var, axis=1) * av
            # Add in Gaussian Rv prior.
            resid_r += (Rv_mean - rv) / Rv_std**2
            r_den += 1. / Rv_std**2
            # Compute determinants (normalization terms).
            sr_idet = 1. / (s_den * r_den - sr_mix**2)
            # Compute ML solution for Delta_Rv.
            drv = sr_idet * (s_den * resid_r - sr_mix * resid_s)

            # Prevent Rv from sliding off the provided bounds.
            drv_low, drv_high = rvlim[0] - rv, rvlim[1] - rv
            lsel, hsel = drv < drv_low, drv > drv_high
            drv[lsel] = drv_low[lsel]
            drv[hsel] = drv_high[hsel]
            # Increment to new Rv.
            rv += drv

            # Update residuals.
            resid -= (av * drv)[:, None] * drvecs

            # Update reddening vector.
            rvecs += drv[:, None] * drvecs

            # Compute error based on best-fitting objects.
            chi2 = np.sum(np.square(resid) / mags_var, axis=1)
            logwt = -0.5 * chi2
            init_sel = np.where(logwt > np.max(logwt) + np.log(init_thresh))[0]
            err = np.max([np.abs(dav[init_sel]), np.abs(drv[init_sel])])
    else:
        # If our data is in flux densities, we can solve the linear system
        # implicitly for `(s_ML, Av_ML, Rv_ML)`. However, the solution
        # is not necessarily as numerically stable as one might hope
        # due to the nature of our Taylor expansion in flux.
        # Instead, it is easier to iterate in `(dAv, dRv)` from
        # a good guess for `(s_ML, Av_ML, Rv_ML)`. We opt to solve both
        # independently at fixed `(Av, Rv)` to avoid recomputing models.

        # Derive ML Delta_Av (`dav`) between data and models.
        a_num = np.sum(rvecs * resid / tot_var, axis=1)
        a_den = np.sum(np.square(rvecs) / tot_var, axis=1)
        a_num += (Av_mean - av) / Av_std**2  # add Av gaussian prior
        a_den += 1. / Av_std**2  # add Av gaussian prior
        dav = a_num / a_den
        # Adjust dAv based on the provided stepsize.
        dav *= stepsize

        # Derive ML Delta_Rv (`drv`) between data and models.
        r_num = np.sum(drvecs * resid / tot_var, axis=1)
        r_den = np.sum(np.square(drvecs) / tot_var, axis=1)
        r_num += (Rv_mean - rv) / Rv_std**2  # add Rv gaussian prior
        r_den += 1. / Rv_std**2  # add Rv gaussian prior
        drv = r_num / r_den
        # Adjust dRv based on the provided stepsize.
        drv *= stepsize

        # Prevent Av from sliding off the provided bounds.
        dav_low, dav_high = avlim[0] - av, avlim[1] - av
        lsel, hsel = dav < dav_low, dav > dav_high
        dav[lsel] = dav_low[lsel]
        dav[hsel] = dav_high[hsel]
        # Increment to new Av.
        av += dav

        # Prevent Rv from sliding off the provided bounds.
        drv_low, drv_high = rvlim[0] - rv, rvlim[1] - rv
        lsel, hsel = drv < drv_low, drv > drv_high
        drv[lsel] = drv_low[lsel]
        drv[hsel] = drv_high[hsel]
        # Increment to new Rv.
        rv += drv

    # Recompute models with new Rv.
    models, rvecs, drvecs = get_seds(mag_coeffs, av=av, rv=rv,
                                     return_flux=True,
                                     return_rvec=True, return_drvec=True)

    # Derive scale-factors (`scale`) between data and models.
    s_num = np.sum(models * data[None, :] / tot_var, axis=1)
    s_den = np.sum(np.square(models) / tot_var, axis=1)
    scale = s_num / s_den  # ML scalefactor
    scale[scale <= 1e-20] = 1e-20  # must be non-negative

    # Compute reddening effect.
    models_int = 10**(-0.4 * mag_coeffs[:, :, 0])
    reddening = models - models_int

    # Rescale models.
    models *= scale[:, None]

    # Compute residuals.
    resid = data - models

    # Derive scale cross-terms.
    sr_mix = np.sum(drvecs * (resid - models) / tot_var, axis=1)
    sa_mix = np.sum(rvecs * (resid - models) / tot_var, axis=1)

    # Rescale reddening quantities.
    rvecs *= scale[:, None]
    drvecs *= scale[:, None]
    reddening *= scale[:, None]

    # Deriving reddening (cross-)terms.
    ar_mix = np.sum(drvecs * (resid - reddening) / tot_var, axis=1)
    a_den = np.sum(np.square(rvecs) / tot_var, axis=1)
    r_den = np.sum(np.square(drvecs) / tot_var, axis=1)
    r_den += 1. / Rv_std**2  # add Rv gaussian prior

    # Construct precision matrices (inverse covariances).
    icov_sar = np.zeros((len(models), 3, 3))
    icov_sar[:, 0, 0] = s_den  # scale
    icov_sar[:, 1, 1] = a_den  # Av
    icov_sar[:, 2, 2] = r_den  # Rv
    icov_sar[:, 0, 1] = sa_mix  # scale-Av cross-term
    icov_sar[:, 1, 0] = sa_mix  # scale-Av cross-term
    icov_sar[:, 0, 2] = sr_mix  # scale-Rv cross-term
    icov_sar[:, 2, 0] = sr_mix  # scale-Rv cross-term
    icov_sar[:, 1, 2] = ar_mix  # Av-Rv cross-term
    icov_sar[:, 2, 1] = ar_mix  # Av-Rv cross-term

    return models, rvecs, drvecs, scale, av, rv, icov_sar, resid


class BruteForce():
    """
    Fits data and generates predictions for scale-factors and reddening
    over a grid in stellar parameters.

    """

    def __init__(self, models, models_labels, labels_mask, pool=None):
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

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel when
            fitting each object.

        """

        # Initialize values.
        self.NMODEL, self.NDIM, self.NCOEF = models.shape
        self.models = models
        self.models_labels = models_labels
        self.labels_mask = labels_mask
        self.NLABELS = len(models_labels[0])
        self.pool = pool
        if pool is None:
            # Single core
            self.M = map
            self.nprocs = 1
        else:
            # Multiple cores
            self.M = pool.map
            self.nprocs = pool.size

    def fit(self, data, data_err, data_mask, data_labels, save_file,
            phot_offsets=None, parallax=None, parallax_err=None,
            Nmc_prior=100, avlim=(0., 6.), av_gauss=None,
            rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
            lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
            apply_agewt=True, apply_grad=True,
            lndistprior=None, lndustprior=None, dustfile='bayestar2017_v1.h5',
            apply_dlabels=True, data_coords=None, logl_dim_prior=True,
            ltol=0.02, ltol_subthresh=0.005, logl_initthresh=1e-4,
            rstate=None, save_dar_draws=True, verbose=True):
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
            Default is `(0., 6.)`.

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

        lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
            Log-prior grid to be used. If not provided, this will default
            to [1] a Kroupa IMF prior in initial mass (`'mini'`) and
            uniform priors in age, metallicity, and dust if we are using the
            MIST models and [2] a PanSTARRS r-band luminosity function-based
            prior if we are using the Bayestar models.
            **Be sure to check this behavior you are using custom models.**

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights when resampling.
            Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            models with (relatively) negligible weights when resampling.
            This option is only used when `wt_thresh=None`.
            Default is `2e-4`.

        Ndraws : int, optional
            The number of realizations of the brute-force PDF to save
            to disk. Indices, scales, and scale errors are saved.
            Default is `2000`.

        apply_agewt : bool, optional
            Whether to apply the age weights derived from the MIST models
            to reweight from EEP to age. Default is `True`.

        apply_grad : bool, optional
            Whether to account for the grid spacing using `np.gradient`.
            Default is `True`.

        lndistprior : func, optional
            The log-distance prior function to be applied. If not provided,
            this will default to the galactic model from Green et al. (2014).

        lndustprior : func, optional
            The log-dust prior function to be applied. If not provided,
            this will default to the 3-D dust map from Green et al. (2018).

        dustfile : str, optional
            The filepath to the 3-D dust map. Default is `bayestar2017_v1.h5`.

        apply_dlabels : bool, optional
            Whether to pass the model labels to the distance prior to
            apply any additional distance-based prior on the parameters.
            Default is `True`.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`, optional
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lndistprior` when constructing the
            distance prior.

        logl_dim_prior : bool, optional
            Whether to apply a dimensional-based correction (prior) to the
            log-likelihood. Transforms the likelihood to a chi2 distribution
            with `Nfilt - 2` degrees of freedom. Default is `True`.

        ltol : float, optional
            The weighted tolerance in the computed log-likelihoods used to
            determine convergence. Default is `0.02`.

        ltol_subthresh : float, optional
            The threshold used to sub-select the best-fit log-likelihoods used
            to determine convergence. Default is `0.005`.

        logl_initthresh : float, optional
            The threshold `logl_initthresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights after computing the initial
            set of fits but before optimizing them. Default is `1e-4`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        save_dar_draws : bool, optional
            Whether to save distance (kpc), reddening (Av), and
            dust curve shape (Rv) draws. Default is `True`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        Ndata, Nfilt = data.shape
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        if rstate is None:
            rstate = np.random
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")
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

        # Apply age weights to reweight from EEP to age.
        if apply_agewt:
            try:
                lnprior += np.log(self.models_labels['agewt'])
            except:
                pass

        # Reweight based on spacing.
        if apply_grad:
            for l in self.models_labels.dtype.names:
                label = self.models_labels[l]
                if self.labels_mask[l][0]:
                    ulabel = np.unique(label)  # grab underlying grid
                    lngrad_label = np.log(np.gradient(ulabel))  # compute grad
                    lnprior += np.interp(label, ulabel, lngrad_label)  # add

        # Initialize distance log(prior).
        if lndistprior is None and data_coords is None:
            raise ValueError("`data_coords` must be provided if using the "
                             "default distance prior.")
        if lndistprior is None:
            lndistprior = gal_lnprior

        # Initialize (distance-)dust log(prior).
        if lndustprior is None and data_coords is None and av_gauss is None:
            raise ValueError("`data_coords` must be provided if using the "
                             "default dust prior.")
        if lndustprior is None:
            lndustprior = dust_lnprior
            # Check provided `dustfile` is valid.
            try:
                # Try reading in parallel-friendly way if possible.
                try:
                    ft = h5py.File(dustfile, 'r', libver='latest', swmr=True)
                except:
                    ft = h5py.File(dustfile, 'r')
                    pass
            except:
                raise ValueError("The default dust prior is being used but "
                                 "the relevant data file is not located at "
                                 "the provided `dustpath`.")
            try:
                # Pre-load provided dustfile into default prior.
                lndustprior(np.linspace(100), (180., 90.), np.linspace(100),
                            dustfile=dustfile)
            except:
                pass

        # Fill data coordinates.
        if data_coords is None:
            data_coords = np.zeros((Ndata, 2))

        # Clean data to remove bad photometry user may not have masked.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
            data_mask *= clean

        # Check there are enough bands to fit.
        Nbmin = 4  # minimum number of bands needed
        if np.any(np.sum(data_mask, axis=1) < Nbmin):
            raise ValueError("Objects with fewer than {0} bands of "
                             "acceptable photometry are currently included in "
                             "the dataset. These objects give degenerate fits "
                             "and cannot be properly modeled. Please remove "
                             "these objects.".format(Nbmin))

        # Initialize results file.
        out = h5py.File("{0}.h5".format(save_file), "w-")
        out.create_dataset("labels", data=data_labels)
        out.create_dataset("idxs", data=np.full((Ndata, Ndraws), -99,
                                                dtype='int32'))
        out.create_dataset("scales", data=np.ones((Ndata, Ndraws),
                                                  dtype='float32'))
        out.create_dataset("avs", data=np.ones((Ndata, Ndraws),
                                               dtype='float32'))
        out.create_dataset("rvs", data=np.ones((Ndata, Ndraws),
                                               dtype='float32'))
        out.create_dataset("cov_sar", data=np.zeros((Ndata, Ndraws, 3, 3),
                                                    dtype='float32'))
        out.create_dataset("log_evidence", data=np.zeros(Ndata,
                                                         dtype='float32'))
        out.create_dataset("best_chi2", data=np.zeros(Ndata, dtype='float32'))
        out.create_dataset("Nbands", data=np.zeros(Ndata, dtype='int16'))
        if save_dar_draws:
            out.create_dataset("dists", data=np.ones((Ndata, Ndraws),
                                                     dtype='float32'))
            out.create_dataset("reds", data=np.ones((Ndata, Ndraws),
                                                    dtype='float32'))
            out.create_dataset("dreds", data=np.ones((Ndata, Ndraws),
                                                     dtype='float32'))

        # Fit data.
        if verbose:
            t1 = time.time()
            t = 0.
            sys.stderr.write('\rFitting object {0}/{1}'.format(1, Ndata))
            sys.stderr.flush()
        for i, results in enumerate(self._fit(data * phot_offsets,
                                              data_err * phot_offsets,
                                              data_mask,
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
                                              lndistprior=lndistprior,
                                              lndustprior=lndustprior,
                                              dustfile=dustfile,
                                              apply_dlabels=apply_dlabels,
                                              data_coords=data_coords,
                                              return_distreds=save_dar_draws,
                                              ltol_subthresh=ltol_subthresh,
                                              logl_dim_prior=logl_dim_prior,
                                              logl_initthresh=logl_initthresh,
                                              ltol=ltol)):
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
                                 '(mean time: {:2.3f} s/obj, '
                                 'est. remaining: {:10.3f} s)'
                                 .format(i+2, Ndata, t_avg, t_est))
                sys.stderr.flush()

            # Save results.
            if save_dar_draws:
                (idxs, scales, avs, rvs, covs_sar, Ndim,
                 levid, chi2min, dists, reds, dreds) = results
            else:
                (idxs, scales, avs, rvs, covs_sar,
                 Ndim, levid, chi2min) = results
            out['idxs'][i] = idxs
            out['scales'][i] = scales
            out['avs'][i] = avs
            out['rvs'][i] = rvs
            out['cov_sar'][i] = covs_sar
            out['Nbands'][i] = Ndim
            out['log_evidence'][i] = levid
            out['best_chi2'][i] = chi2min
            if save_dar_draws:
                out['dists'][i] = dists
                out['reds'][i] = reds
                out['dreds'][i] = dreds

        if verbose:
            # Compute time stamps.
            t2 = time.time()
            dt = t2 - t1  # time for current object
            t1 = t2
            t += dt  # total time elapsed
            t_avg = t / (i + 1)  # avg time per object
            t_est = t_avg * (Ndata - i - 1)  # estimated remaining time
            sys.stderr.write('\rFitting object {:d}/{:d} '
                             '(mean time: {:2.3f} s/obj, '
                             'est. time remaining: {:10.3f} s)'
                             .format(i+1, Ndata, t_avg, t_est))
            sys.stderr.flush()
            sys.stderr.write('\n')
            sys.stderr.flush()

        out.close()  # close output results file

    def _fit(self, data, data_err, data_mask,
             parallax=None, parallax_err=None, Nmc_prior=100,
             avlim=(0., 6.), av_gauss=None,
             rvlim=(1., 8.), rv_gauss=(3.32, 0.18),
             lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
             lndistprior=None, lndustprior=None, dustfile='bayestar2017_v1.h5',
             apply_dlabels=True, data_coords=None,
             return_distreds=True, logl_dim_prior=True, ltol=0.02,
             ltol_subthresh=0.005, logl_initthresh=1e-4, rstate=None):
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
            Default is `(0., 6.)`.

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

        lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
            Log-prior grid to be used. If not provided, will default
            to `0.`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights.
            Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            models with (relatively) negligible weights when resampling.
            This option is only used when `wt_thresh=None`.
            Default is `2e-4`.

        Ndraws : int, optional
            The number of realizations of the brute-force PDF to save
            to disk. Indices, scales, and scale errors are saved.
            Default is `2000`.

        lndistprior : func, optional
            The log-distsance prior function to be applied. If not provided,
            this will default to the galactic model from Green et al. (2014).

        lndustprior : func, optional
            The log-dust prior function to be applied. If not provided,
            this will default to the 3-D dust map from Green et al. (2018).

        dustfile : str, optional
            The filepath to the 3-D dust map. Default is `bayestar2017_v1.h5`.

        apply_dlabels : bool, optional
            Whether to pass the model labels to the distance prior to
            apply any additional distance-based prior on the parameters.
            Default is `True`.

        data_coords : `~numpy.ndarray` of shape `(Ndata, 2)`, optional
            The galactic `(l, b)` coordinates for the objects that are being
            fit. These are passed to `lndistprior` when constructing the
            distance prior.

        return_distreds : bool, optional
            Whether to return distance and reddening draws (in units of kpc
            and Av, respectively). Default is `True`.

        logl_dim_prior : bool, optional
            Whether to apply a dimensional-based correction (prior) to the
            log-likelihood. Transforms the likelihood to a chi2 distribution
            with `Nfilt - 2` degrees of freedom. Default is `True`.

        ltol : float, optional
            The weighted tolerance in the computed log-likelihoods used to
            determine convergence. Default is `0.02`.

        ltol_subthresh : float, optional
            The threshold used to sub-select the best-fit log-likelihoods used
            to determine convergence. Default is `0.005`.

        logl_initthresh : float, optional
            The threshold `logl_initthresh * max(y_wt)` used to ignore models
            with (relatively) negligible weights after computing the initial
            set of fits but before optimizing them. Default is `1e-4`.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        Returns
        -------
        results : tuple
            Outputs yielded from the generator.

        """

        Ndata, Nmodels = len(data), self.NMODEL
        models = np.array(self.models)
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        if rstate is None:
            rstate = np.random
        mvn = rstate.multivariate_normal
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")
        if parallax is None:
            parallax = np.full(Ndata, np.nan)
        if parallax_err is None:
            parallax_err = np.full(Ndata, np.nan)

        # Initialize log(prior).
        if lnprior is None:
            lnprior = 0.
        self.lnprior = lnprior

        # Initialize distance log(prior).
        if lndistprior is None and data_coords is None:
            raise ValueError("`data_coords` must be provided if using the "
                             "default distance prior.")
        if lndustprior is None and data_coords is None and av_gauss is None:
            raise ValueError("`data_coords` must be provided if using the "
                             "default dust prior.")
        if lndistprior is None:
            lndistprior = gal_lnprior
        if lndustprior is None:
            # Check provided `dustfile` is valid.
            try:
                # Try reading in parallel-friendly way if possible.
                try:
                    ft = h5py.File(dustfile, 'r', libver='latest', swmr=True)
                except:
                    ft = h5py.File(dustfile, 'r')
                    pass
            except:
                raise ValueError("The default dust prior is being used but "
                                 "the relevant data file is not located at "
                                 "the provided `dustpath`.")
            lndustprior = dust_lnprior
        if data_coords is None:
            data_coords = np.zeros((Ndata, 2))
        if apply_dlabels:
            dlabels = self.models_labels
        else:
            dlabels = None

        # Modifications to support parallelism.

        # Split up data products into `nprocs` chunks.
        counter = 0
        data_list = []
        data_err_list = []
        data_mask_list = []
        counter_list = []
        while counter < Ndata:
            data_list.append(data[counter:counter+self.nprocs])
            data_err_list.append(data_err[counter:counter+self.nprocs])
            data_mask_list.append(data_mask[counter:counter+self.nprocs])
            counter_list.append(np.arange(counter,
                                          min(counter+self.nprocs, Ndata)))
            counter += self.nprocs

        # Re-define log-likelihood to deal with zipped values.
        def _loglike_zip(z, *args, **kwargs):
            d, e, m = z  # grab data, error, mask
            return loglike(d, e, m, models, *args, **kwargs)

        # Wrap log-likelihood with fixed kwargs.
        loglike_kwargs = {'avlim': avlim, 'ltol': ltol,
                          'rvlim': rvlim, 'rv_gauss': rv_gauss,
                          'logl_dim_prior': logl_dim_prior,
                          'wt_thresh': ltol_subthresh,
                          'init_thresh': logl_initthresh,
                          'return_vals': True}
        if av_gauss is not None:
            loglike_kwargs['av_gauss'] = av_gauss  # only pass if provided
        _loglike = _function_wrapper(_loglike_zip, [], loglike_kwargs,
                                     name='loglike')

        # Re-define log-posterior to deal with zipped values.
        def _logpost_zip(z, *args, **kwargs):
            res, p, pe, c = z  # grab results, parallax/error, coord
            return _lnpost(res, parallax=p, parallax_err=pe, coord=c,
                           *args, **kwargs)

        # Wrap log-posterior with fixed kwargs.
        logpost_kwargs = {'Nmc_prior': Nmc_prior, 'lnprior': lnprior,
                          'lnprior': lnprior, 'wt_thresh': wt_thresh,
                          'cdf_thresh': cdf_thresh, 'rstate': rstate,
                          'lndistprior': lndistprior,
                          'lndustprior': lndustprior, 'dustfile': dustfile,
                          'avlim': avlim, 'rvlim': rvlim, 'dlabels': dlabels,
                          'apply_av_prior': av_gauss is None,
                          'return_distreds': return_distreds}
        _logpost = _function_wrapper(_logpost_zip, [], logpost_kwargs,
                                     name='logpost')

        # Fit data.
        for x, xe, xm, chunk in zip(data_list, data_err_list,
                                    data_mask_list, counter_list):

            # Compute log-likelihoods optimizing over s and Av.
            results_map = list(self.M(_loglike, zip(x, xe, xm)))

            # Compute posteriors.
            lnpost_map = list(self.M(_logpost,
                                     zip(results_map, parallax[chunk],
                                         parallax_err[chunk],
                                         data_coords[chunk])))

            # Extract `map`-ed results.
            for results, blob in zip(results_map, lnpost_map):
                lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results
                if return_distreds:
                    sel, lnpost, dists, reds, dreds, logwts = blob
                else:
                    sel, lnpost = blob

                # Compute GOF metrics.
                chi2min = np.min(chi2[sel])
                levid = logsumexp(lnpost)

                # Resample.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wt = np.exp(lnpost - levid)
                    wt /= wt.sum()
                    idxs = rstate.choice(len(sel), size=Ndraws, p=wt)
                    sidxs = sel[idxs]

                # Grab/compute corresponding values.
                scales, avs, rvs = scales[sidxs], avs[sidxs], rvs[sidxs]
                cov_sar = _inverse3(icovs_sar[sidxs])

                # Draw distances and reddenings.
                if return_distreds:
                    imc = np.zeros(Ndraws, dtype='int')
                    for i, idx in enumerate(idxs):
                        wt = np.exp(logwts[idx] - logsumexp(logwts[idx]))
                        wt /= wt.sum()
                        imc[i] = rstate.choice(Nmc_prior, p=wt)
                    dists = dists[idxs, imc]
                    reds = reds[idxs, imc]
                    dreds = dreds[idxs, imc]
                    yield (sidxs, scales, avs, rvs, cov_sar,
                           Ndim, levid, chi2min,
                           dists, reds, dreds)
                else:
                    yield (sidxs, scales, avs, rvs, cov_sar,
                           Ndim, levid, chi2min)


def _lnpost(results, parallax=None, parallax_err=None, coord=None,
            Nmc_prior=100, lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4,
            lndistprior=None, lndustprior=None, dustfile='bayestar2017_v1.h5',
            dlabels=None, avlim=(0., 6.), rvlim=(1., 8.),
            rstate=None, apply_av_prior=True, return_distreds=True,
            *args, **kwargs):
    """
    Internal function used to estimate posteriors from fits using the
    provided priors via Monte Carlo integration.

    Parameters
    ----------
    results : tuple of `(lnlike, Ndim, chi2, scales, avs, ds2, da2, dsda)`
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
        fit. These are passed to `lndistprior` when constructing the
        distance prior.

    Nmc_prior : int, optional
        The number of Monte Carlo realizations used to estimate the
        integral over the prior. Default is `100`.

    lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
        Log-prior grid to be used. If not provided, will default
        to `0.`.

    wt_thresh : float, optional
        The threshold `wt_thresh * max(y_wt)` used to ignore models
        with (relatively) negligible weights.
        Default is `1e-3`.

    cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
        models with (relatively) negligible weights when resampling.
        This option is only used when `wt_thresh=None`.
        Default is `2e-4`.

    lndistprior : func, optional
        The log-distsance prior function to be applied. If not provided,
        this will default to the galactic model from Green et al. (2014).

    lndustprior : func, optional
        The log-dust prior function to be applied. If not provided,
        this will default to the 3-D dust map from Green et al. (2018).

    dustfile : str, optional
        The filepath to the 3-D dust map. Default is `bayestar2017_v1.h5`.

    dlabels : bool, optional
        The model labels to be passed the distance prior to
        apply any additional distance-based prior on the parameters.

    avlim : 2-tuple, optional
        The bounds where Av predictions are reliable.
        Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The lower and upper bound where the reddening vector shape changes
        are reliable. Default is `(1., 8.)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    apply_av_prior : bool, optional
        Whether to apply the 3-D dust prior

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
        rstate = np.random
    mvn = rstate.multivariate_normal
    if parallax is not None and parallax_err is None:
        raise ValueError("Must provide both `parallax` and "
                         "`parallax_err`.")
    if parallax is None:
        np.nan
    if parallax_err is None:
        np.nan

    # Initialize log(prior).
    if lnprior is None:
        lnprior = 0.

    # Initialize distance log(prior).
    if lndistprior is None and coord is None:
        raise ValueError("`coord` must be provided if using the "
                         "default distance prior.")
    if lndustprior is None and coord is None and apply_av_prior:
        raise ValueError("`coord` must be provided if using the "
                         "default dust prior.")
    if lndistprior is None:
        lndistprior = gal_lnprior
    if lndustprior is None:
        lndustprior = dust_lnprior
    if coord is None:
        coord = np.zeros(2)

    # Grab results.
    lnlike, Ndim, chi2, scales, avs, rvs, icovs_sar = results
    Nmodels = len(lnlike)

    # Compute initial log-posteriors.
    lnpost = lnlike + lnprior

    # Apply rough parallax prior for clipping.
    if parallax is not None and parallax_err is not None:
        ds2 = icovs_sar[:, 0, 0]
        scales_err = 1./np.sqrt(np.abs(ds2))  # approximate scale errors
        lnprob = lnpost + scale_parallax_lnprior(scales, scales_err,
                                                 parallax, parallax_err)
    else:
        lnprob = lnpost

    # Apply thresholding.
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        lwt_min = np.log(wt_thresh) + np.max(lnprob)
        sel = np.arange(Nmodels)[lnprob > lwt_min]
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(lnprob)
        prob = np.exp(lnprob - logsumexp(lnprob))
        cdf = np.cumsum(prob[idx_sort])
        sel = idx_sort[cdf <= (1. - cdf_thresh)]
    lnpost = lnpost[sel]
    Nsel = len(sel)

    # Generate covariance matrices for the selected fits.
    scale, av, rv = scales[sel], avs[sel], rvs[sel]
    cov_sar = _inverse3(icovs_sar[sel])

    # Now actually apply priors.
    if Nmc_prior > 0:
        # Use Monte Carlo integration to get an estimate of the
        # overlap integral.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_mc, a_mc, r_mc = np.array([mvn(np.array([s, a, r]), c,
                                         size=Nmc_prior)
                                         for s, a, r, c in zip(scale, av, rv,
                                                               cov_sar)]).T
        if dlabels is not None:
            dlabels_mc = np.tile(dlabels[sel], Nmc_prior).reshape(-1, Nsel)
        else:
            dlabels_mc = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            par_mc = np.sqrt(s_mc)
            dist_mc = 1. / par_mc
            # Evaluate distance prior.
            lnp_mc = lndistprior(dist_mc, coord, labels=dlabels_mc)
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
            warnings.simplefilter("ignore")
            lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc_prior_eff)
            lnpost += lnp
    else:
        # Just assume the maximum-likelihood estimate.
        lnpost = lnprob[sel]

    if return_distreds:
        return sel, lnpost, dist_mc.T, a_mc.T, r_mc.T, lnp_mc.T
    else:
        return sel, lnpost
