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
from scipy.special import xlogy, gammaln

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .pdf import *
from .utils import *

__all__ = ["get_seds", "loglike", "_optimize_fit", "BruteForce", "_lnpost"]


def get_seds(mag_coeffs, av, return_rvec=False, return_flux=False):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    Parameters
    ----------
    mag_coeffs : `~numpy.ndarray` of shape `(Nmodels, Nbands, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    av : `~numpy.ndarray` of shape `(Nmodels)`
        Array of dust attenuation values photometry should be predicted for.

    return_rvec : bool, optional
        Whether to return the differential reddening vectors at the provided
        `av`. Default is `False`.

    return_flux : bool, optional
        Whether to return SEDs as flux densities instead of magnitudes.
        Default is `False`.

    Returns
    -------
    seds : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddened SEDs.

    rvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`, optional
        Differential reddening vectors.

    """

    Nmodels, Nbands, Ncoef = mag_coeffs.shape

    # Turn provided Av values into polynomial features.
    av_poly = np.array([av**(Ncoef-j-1) if j < Ncoef - 1 else np.ones_like(av)
                        for j in range(Ncoef)]).T

    # Compute SEDs.
    seds = np.sum(mag_coeffs * av_poly[:, None, :], axis=-1)
    if return_flux:
        seds = 10**(-0.4 * seds)

    if return_rvec:
        # Compute reddening vectors.
        rvecs = np.sum(np.arange(1, Ncoef)[::-1] * mag_coeffs[:, :, :-1] *
                       av_poly[:, None, 1:], axis=-1)
        if return_flux:
            rvecs *= -0.4 * np.log(10.) * seds
        return seds, rvecs
    else:
        return seds


def loglike(data, data_err, data_mask, mag_coeffs, avlim=(0., 6.),
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

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
        Magnitude coefficients used to compute reddened photometry for a given
        model.

    avlim : 2-tuple, optional
        The lower and upper bound where the reddened photometry is reliable.
        Default is `(0., 6.)`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 2` degrees of freedom. Default is `True`.

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
        Whether to return the best-fit scale-factor and reddening along with
        the associated precision matrix entries. Default is `False`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape `(Nmodel)`
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape `(Nmodel)`
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape `(Nmodel)`
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit scale factor.

    Av : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddening.

    ds2 : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The second-derivative of the log-likelihood with respect to `s`
        around `s_ML` and `Av_ML`.

    da2 : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The second-derivative of the log-likelihood with respect to `Delta_Av`
        around `s_ML` and `Av_ML`.

    dsda : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The mixed-derivative of the log-likelihood with respect to `s` and
        `Delta_Av` around `s_ML` and `Av_ML`.

    """

    # Initialize values.
    Nmodels, Nfilt, Ncoef = mag_coeffs.shape

    # Clean data (safety checks).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
        data[~clean], data_err[~clean], data_mask[~clean] = 1., 1., False

    # Subselect only clean observations.
    Ndim = sum(data_mask)  # number of dimensions
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
    av = np.zeros(Nmodels)
    models, rvecs = get_seds(mcoeffs, av=av, return_rvec=True)

    # Compute initial magnitude fit.
    results = _optimize_fit(flux, tot_var, models, rvecs, av,
                            mcoeffs, resid=None, mags=mags,
                            mags_var=mags_var, stepsize=1.)
    models, rvecs, scale, av, ds2, da2, dsda = results
    resid = flux - models

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
        models, rvecs = models[init_sel], rvecs[init_sel]
        av_new = av[init_sel]
        mcoeffs = mcoeffs[init_sel]
        resid = resid[init_sel]
    else:
        # Keep all models.
        init_sel = np.arange(Nmodels)
        chi2 = np.ones(Nmodels) + 1e300
        lnl = np.ones(Nmodels) - 1e300
        av_new = np.array(av)

    # Iterate until convergence.
    lnl_old, lerr = -1e300, 1e300
    stepsize, rescaling = np.ones(Nmodels)[init_sel] * 3., 1.2
    while lerr > ltol:

        # Re-compute models.
        results = _optimize_fit(flux, tot_var, models, rvecs,
                                av_new, mcoeffs, avlim=avlim, resid=resid,
                                stepsize=stepsize)
        models, rvecs, scale_new, av_new, ds2_new, da2_new, dsda_new = results

        # Compute chi2.
        resid = flux - models
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
    scale[init_sel], av[init_sel] = scale_new, av_new
    ds2[init_sel], da2[init_sel], dsda[init_sel] = ds2_new, da2_new, dsda_new

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 2)  # effective dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    if return_vals:
        return lnl, Ndim, chi2, scale, av, ds2, da2, dsda
    else:
        return lnl, Ndim, chi2


def _optimize_fit(data, tot_var, models, rvecs, av, mag_coeffs,
                  avlim=(0., 6.), resid=None, mags=None, mags_var=None,
                  stepsize=1.):
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

    av : `~numpy.ndarray` of shape `(Nmodel,)`
        Av values of the models.

    mag_coeffs : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
        Magnitude coefficients used to compute reddened photometry for a given
        model.

    avlim : 2-tuple, optional
        The lower and upper bound where the reddened photometry is reliable.
        Default is `(0., 6.)`.

    resid : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Residuals between the data and models.
        If not provided, this will be computed.

    mags : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Observed data values in magnitudes.

    mags_var : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`, optional
        Associated (Normal) errors on the observed values compared to the
        models in magnitudes.

    stepsize : float or `~numpy.ndarray`, optional
        The stepsize (in units of the computed gradient). Default is `1.`.

    Returns
    -------
    models_new : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        New model predictions. Always returned in flux densities.

    rvecs_new : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        New reddening vectors. Always returned in flux densities.

    scale : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit scale factor.

    Av : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The best-fit reddening.

    ds2 : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The second-derivative of the log-likelihood with respect to `s`
        around `s_ML` and `Av_ML`.

    da2 : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The second-derivative of the log-likelihood with respect to `Delta_Av`
        around `s_ML` and `Av_ML`.

    dsda : `~numpy.ndarray` of shape `(Nmodel)`, optional
        The mixed-derivative of the log-likelihood with respect to `s` and
        `Delta_Av` around `s_ML` and `Av_ML`.

    """

    # Compute residuals.
    if resid is None:
        if mags is not None and mags_var is not None:
            resid = mags - models
        else:
            resid = data - models

    # First fit dAv.
    if mags is not None and mags_var is not None:
        # If our data is in magnitudes, our model is `M + dAv*R + s`.
        # The solution can be solved explicitly as linear system for (s, dAv).

        # Derive partial derivatives.
        s_den = np.sum(1. / mags_var, axis=1)
        a_den = np.sum(np.square(rvecs) / mags_var, axis=1)
        sa_mix = np.sum(rvecs / mags_var, axis=1)

        # Compute residual terms.
        resid_s = np.sum(resid / mags_var, axis=1)
        resid_a = np.sum(resid * rvecs / mags_var, axis=1)

        # Compute determinants (normalization terms).
        sa_idet = 1. / (s_den * a_den - sa_mix**2)

        # Compute ML solution for dAv.
        dav = sa_idet * (s_den * resid_a - sa_mix * resid_s)
    else:
        # If our data is in flux densities, our model is `s*F - dAv*s*R`.
        # The solution can be solved explicitly as linear system for (s, s*dAv)
        # and converted back to dAv from s_ML. However, given a good guess
        # for s and Av it is fine to instead just iterate between the two.

        # Derive ML Delta_Av (`dav`) between data and models.
        a_num = np.sum(rvecs * resid / tot_var, axis=1)
        a_den = np.sum(np.square(rvecs) / tot_var, axis=1)
        dav = a_num / a_den

        # Adjust dAv based on the provided stepsize.
        dav *= stepsize

    # Prevent Av from sliding off the provided bounds.
    dav_low, dav_high = avlim[0] - av, avlim[1] - av
    lsel, hsel = dav < dav_low, dav > dav_high
    dav[lsel] = dav_low[lsel]
    dav[hsel] = dav_high[hsel]

    # Recompute models with new Av.
    av += dav
    models, rvecs = get_seds(mag_coeffs, av=av, return_rvec=True,
                             return_flux=True)

    # Derive scale-factors (`scale`) between data and models.
    s_num = np.sum(models * data[None, :] / tot_var, axis=1)
    s_den = np.sum(np.square(models) / tot_var, axis=1)
    scale = s_num / s_den  # ML scalefactor
    scale[scale <= 0.] = 1e-20  # must be non-negative

    # Rescale models.
    models *= scale[:, None]

    # Derive cross-term.
    sa_mix = np.sum(rvecs * (resid - models) / tot_var, axis=1)

    # Rescale reddening vector.
    rvecs *= scale[:, None]

    return models, rvecs, scale, av, s_den, a_den, sa_mix


class BruteForce():
    """
    Fits data and generates predictions for scale-factors and reddening
    over a grid in initial mass, EEP, and metallicity.

    """

    def __init__(self, models, models_labels, labels_mask, pool=None):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
            Magnitude coefficients used to compute reddened photometry over
            the desired bands for all models on the grid.

        models_labels : structured `~numpy.ndarray` of shape `(Nmodel, Nlabel)`
            Labels corresponding to each model on the grid.

        labels_mask : structured `~numpy.ndarray` of shape `(1, Nlabel)`
            Masks corresponding to each label to indicate whether it is
            an ancillary prediction (e.g., `logt`) or was used to compute
            the model grid (e.g., `mini`).

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
            parallax=None, parallax_err=None, Nmc_prior=250, avlim=(0., 6.),
            lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
            apply_agewt=True, apply_grad=True, lndistprior=None,
            apply_dlabels=True, data_coords=None, logl_dim_prior=True,
            ltol=0.02, ltol_subthresh=0.005, logl_initthresh=1e-4,
            rstate=None, save_dist_draws=True, save_red_draws=True,
            verbose=True):
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

        parallax : `~numpy.ndarray` of shape `(Ndata)`, optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape `(Ndata)`, optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        Nmc_prior : int, optional
            The number of Monte Carlo realizations used to estimate the
            integral over the prior. Default is `250`.

        avlim : 2-tuple, optional
            The bounds where Av predictions are reliable.
            Default is `(0., 6.)`.

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
            The log-distsance prior function to be applied. If not provided,
            this will default to the galactic model from Green et al. (2014).

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

        save_dist_draws : bool, optional
            Whether to save distance draws (in units of kpc).
            Default is `True`.

        save_red_draws : bool, optional
            Whether to save reddening draws (in units of Av).
            Default is `True`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        Ndata = len(data)
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        if rstate is None:
            rstate = np.random
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")
        return_distreds = save_dist_draws or save_red_draws

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
                warnings.warn("No age weights provided in `labels`. ")
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
        if data_coords is None:
            data_coords = np.zeros((Ndata, 2))

        # Initialize results file.
        out = h5py.File("{0}.h5".format(save_file), "w-")
        out.create_dataset("labels", data=data_labels)
        out.create_dataset("idxs", data=np.full((Ndata, Ndraws), -99,
                                                dtype='int32'))
        out.create_dataset("scales", data=np.ones((Ndata, Ndraws),
                                                  dtype='float32'))
        out.create_dataset("avs", data=np.ones((Ndata, Ndraws),
                                               dtype='float32'))
        out.create_dataset("cov_sa", data=np.zeros((Ndata, Ndraws, 2, 2),
                                                   dtype='float32'))
        out.create_dataset("log_evidence", data=np.zeros(Ndata,
                                                         dtype='float32'))
        out.create_dataset("best_chi2", data=np.zeros(Ndata, dtype='float32'))
        out.create_dataset("Nbands", data=np.zeros(Ndata, dtype='int16'))
        if save_dist_draws:
            out.create_dataset("dists", data=np.ones((Ndata, Ndraws),
                                                     dtype='float32'))
        if save_red_draws:
            out.create_dataset("reds", data=np.ones((Ndata, Ndraws),
                                                    dtype='float32'))

        # Fit data.
        if verbose:
            sys.stderr.write('\rFitting object {0}/{1}'.format(1, Ndata))
            sys.stderr.flush()
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              parallax=parallax,
                                              parallax_err=parallax_err,
                                              avlim=avlim,
                                              Nmc_prior=Nmc_prior,
                                              lnprior=lnprior,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              Ndraws=Ndraws, rstate=rstate,
                                              lndistprior=lndistprior,
                                              apply_dlabels=apply_dlabels,
                                              data_coords=data_coords,
                                              return_distreds=return_distreds,
                                              ltol_subthresh=ltol_subthresh,
                                              logl_dim_prior=logl_dim_prior,
                                              logl_initthresh=logl_initthresh,
                                              ltol=ltol)):
            # Print progress.
            if verbose and i < Ndata - 1:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+2, Ndata))
                sys.stderr.flush()

            # Save results.
            if return_distreds:
                (idxs, scales, avs, covs_sa, Ndim,
                 levid, chi2min, dists, reds) = results
            else:
                idxs, scales, avs, covs_sa, Ndim, levid, chi2min = results
            out['idxs'][i] = idxs
            out['scales'][i] = scales
            out['avs'][i] = avs
            out['cov_sa'][i] = covs_sa
            out['Nbands'][i] = Ndim
            out['log_evidence'][i] = levid
            out['best_chi2'][i] = chi2min
            if save_dist_draws:
                out['dists'][i] = dists
            if save_red_draws:
                out['reds'][i] = reds

        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        out.close()  # close output results file

    def _fit(self, data, data_err, data_mask,
             parallax=None, parallax_err=None, Nmc_prior=250, avlim=(0., 6.),
             lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
             lndistprior=None, apply_dlabels=True, data_coords=None,
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
            integral over the prior. Default is `250`.

        avlim : 2-tuple, optional
            The bounds where Av predictions are reliable.
            Default is `(0., 6.)`.

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
        if lndistprior is None:
            lndistprior = gal_lnprior
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
                          'logl_dim_prior': logl_dim_prior,
                          'wt_thresh': ltol_subthresh,
                          'init_thresh': logl_initthresh,
                          'return_vals': True}
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
                          'lndistprior': lndistprior, 'avlim': avlim,
                          'dlabels': dlabels,
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
                lnlike, Ndim, chi2, scales, avs, ds2, da2, dsda = results
                if return_distreds:
                    sel, lnpost, dists, reds, logwts = blob
                else:
                    sel, lnpost = blob

                # Compute GOF metrics.
                chi2min = np.min(chi2[sel])
                levid = logsumexp(lnpost)

                # Resample.
                wt = np.exp(lnpost - levid)
                wt /= wt.sum()
                idxs = rstate.choice(len(sel), size=Ndraws, p=wt)
                sidxs = sel[idxs]

                # Grab/compute corresponding values.
                scales, avs, ds2, da2, dsda = (scales[sidxs], avs[sidxs],
                                               ds2[sidxs], da2[sidxs],
                                               dsda[sidxs])
                cov_sa = np.array([np.linalg.inv(np.array([[s, sa], [sa, a]]))
                                   for s, a, sa in zip(ds2, da2, dsda)])

                # Draw distances and reddenings.
                if return_distreds:
                    imc = np.zeros(Ndraws, dtype='int')
                    for i, idx in enumerate(idxs):
                        wt = np.exp(logwts[idx] - logsumexp(logwts[idx]))
                        wt /= wt.sum()
                        imc[i] = rstate.choice(Nmc_prior, p=wt)
                    dists, reds = dists[idxs, imc], reds[idxs, imc]
                    yield (sidxs, scales, avs, cov_sa, Ndim, levid, chi2min,
                           dists, reds)
                else:
                    yield sidxs, scales, avs, cov_sa, Ndim, levid, chi2min


def _lnpost(results, parallax=None, parallax_err=None, coord=None,
            Nmc_prior=250, lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4,
            lndistprior=None, dlabels=None, avlim=(0., 6.), rstate=None,
            return_distreds=True, *args, **kwargs):
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
        integral over the prior. Default is `250`.

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

    dlabels : bool, optional
        The model labels to be passed the distance prior to
        apply any additional distance-based prior on the parameters.

    avlim : 2-tuple, optional
        The bounds where Av predictions are reliable.
        Default is `(0., 6.)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

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
       The reddening draws for each selected model.

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
    if lndistprior is None:
        lndistprior = gal_lnprior
    if coord is None:
        coord = np.zeros(2)

    # Grab results.
    lnlike, Ndim, chi2, scales, avs, ds2, da2, dsda = results
    Nmodels = len(lnlike)

    # Compute initial log-posteriors.
    lnpost = lnlike + lnprior

    # Apply rough parallax prior for clipping.
    if parallax is not None and parallax_err is not None:
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
    scale, av, sinv2, ainv2, sainv = (scales[sel], avs[sel],
                                      ds2[sel], da2[sel],
                                      dsda[sel])
    cinv_sa = np.array([np.array([[s, sa], [sa, a]])
                       for s, a, sa in zip(sinv2, ainv2, sainv)])
    cov_sa = np.array([np.linalg.inv(ci) for ci in cinv_sa])

    # Now actually apply distance (and parallax) priors.
    if Nmc_prior > 0:
        # Use Monte Carlo integration to get an estimate of the
        # overlap integral.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_mc, a_mc = np.array([mvn(np.array([s, a]), c,
                                   size=Nmc_prior)
                                   for s, a, c in zip(scale, av,
                                                      cov_sa)]).T
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
        if parallax is not None and parallax_err is not None:
            # Evaluate parallax prior.
            lnp_mc += parallax_lnprior(par_mc, parallax, parallax_err)
        # Ignore points that are out of bounds.
        inbounds = ((s_mc >= 0.) & (a_mc >= avlim[0]) &
                    (a_mc <= avlim[1]))
        lnp_mc[~inbounds] = -1e300
        Nmc_prior_eff = np.sum(inbounds, axis=0)
        # Compute integral.
        lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc_prior_eff)
        lnpost += lnp
    else:
        # Just assume the maximum-likelihood estimate.
        lnpost = lnprob[sel]

    if return_distreds:
        return sel, lnpost, dist_mc.T, a_mc.T, lnp_mc.T
    else:
        return sel, lnpost
