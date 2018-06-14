#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute force fitter.

"""

from __future__ import (print_function, division)
import six
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

__all__ = ["get_seds", "loglike", "_optimize_fit", "BruteForce"]


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
            dim_prior=True, ltol=0.02, wt_thresh=0.005, return_vals=False,
            *args, **kwargs):
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
    clean = np.isfinite(data) & np.isfinite(data_err) & (data_err > 0.)
    data[~clean], data_err[~clean], data_mask[~clean] = 1., 1., False

    tot_var = np.square(data_err) + np.zeros((Nmodels, Nfilt))  # variance
    tot_mask = data_mask * np.ones((Nmodels, Nfilt))  # binary mask
    Ndim = sum(data_mask)  # number of dimensions

    # Get started by fitting in magnitudes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Convert to magnitudes.
        mags = -2.5 * np.log10(data)
        mags_var = np.square(2.5 / np.log(10.)) * tot_var / np.square(data)
        mclean = np.isfinite(mags)
        mags[~mclean], mags_var[:, ~mclean] = 0., 1e50  # mask negative values
    # Compute unreddened photometry.
    av = np.zeros(Nmodels)
    models, rvecs = get_seds(mag_coeffs, av=av, return_rvec=True)
    results = _optimize_fit(data, tot_var, tot_mask, models, rvecs, av,
                            mag_coeffs, resid=None, mags=mags,
                            mags_var=mags_var, stepsize=1.)
    models, rvecs, scale, av, ds2, da2, dsda = results
    resid = data - models

    # Iterate until convergence.
    lnl_old, lerr = -1e300, 1e300
    stepsize, rescaling = np.ones(Nmodels) * 3., 1.2
    while lerr > ltol:

        # Re-compute models.
        results = _optimize_fit(data, tot_var, tot_mask, models, rvecs, av,
                                mag_coeffs, avlim=avlim, resid=resid,
                                stepsize=stepsize)
        models, rvecs, scale, av, ds2, da2, dsda = results

        # Compute chi2.
        resid = data - models
        chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)

        # Compute multivariate normal logpdf.
        lnl = -0.5 * chi2
        lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))

        # Compute stopping criterion.
        lnl_sel = lnl > np.max(lnl) + np.log(wt_thresh)
        lerr = np.max(np.abs(lnl - lnl_old)[lnl_sel])

        # Adjust stepsize.
        stepsize[lnl < lnl_old] /= rescaling
        lnl_old = lnl

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 2)  # effective dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    if return_vals:
        return lnl, Ndim, chi2, scale, av, ds2, da2, dsda
    else:
        return lnl, Ndim, chi2


def _optimize_fit(data, tot_var, tot_mask, models, rvecs, av, mag_coeffs,
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

    tot_mask : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Binary mask (0/1) indicating whether the data was observed compared
        to the models.

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
        s_den = np.sum(tot_mask / mags_var, axis=1)
        a_den = np.sum(tot_mask * np.square(rvecs) / mags_var, axis=1)
        sa_mix = np.sum(tot_mask * rvecs / mags_var, axis=1)

        # Compute residual terms.
        resid_s = np.sum(tot_mask * resid / mags_var, axis=1)
        resid_a = np.sum(tot_mask * resid * rvecs / mags_var, axis=1)

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
        a_num = np.sum(tot_mask * rvecs * resid / tot_var, axis=1)
        a_den = np.sum(tot_mask * np.square(rvecs) / tot_var, axis=1)
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
    s_num = np.sum(tot_mask * models * data[None, :] / tot_var, axis=1)
    s_den = np.sum(tot_mask * np.square(models) / tot_var, axis=1)
    scale = s_num / s_den  # ML scalefactor
    scale[scale <= 0.] = 1e-20  # must be non-negative

    # Rescale models.
    models *= scale[:, None]

    # Derive cross-term.
    sa_mix = np.sum(tot_mask * rvecs * (resid - models) / tot_var, axis=1)

    # Rescale reddening vector.
    rvecs *= scale[:, None]

    return models, rvecs, scale, av, s_den, a_den, sa_mix


class BruteForce():
    """
    Fits data and generates predictions for scale-factors and reddening
    over a grid in initial mass, EEP, and metallicity.

    """

    def __init__(self, models, models_labels, models_params=None,
                 avlim=(0., 6.)):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
            Magnitude coefficients used to compute reddened photometry over
            the desired bands for all models on the grid.

        models_labels : `~numpy.ndarray` of shape `(Nmodel, Nlabels)`
            Labels corresponding to each model on the grid.

        models_params : `~numpy.ndarray` of shape `(Nmodel, Nparams)`, optional
            Output parameters for the models. These were output while
            constructing the grid based on the input labels.

        avlim : 2-tuple, optional
            The bounds where Av predictions are reliable.
            Default is `(0., 6.)`.

        """

        # Initialize values.
        self.NMODEL, self.NDIM, self.NCOEF = models.shape
        self.models = models
        self.models_labels = models_labels
        self.NLABELS = len(models_labels[0])
        if models_params is not None:
            self.NPARAMS = len(models_params[0])
        else:
            self.NPARAMS = 0
        self.models_params = models_params
        self.avlim = avlim

    def fit(self, data, data_err, data_mask, data_labels, save_file,
            parallax=None, parallax_err=None, Nmc_prior=150,
            lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
            apply_agewt=True, apply_grad=True, lndistprior=None,
            data_coords=None, logl_dim_prior=True, ltol=0.02,
            ltol_subthresh=0.005, rstate=None, verbose=True):
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
            integral over the prior. Default is `150`.

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

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

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
                lnprior += np.log(self.models_params['agewt'])
            except:
                warnings.warn("No age weights provided in `models_params`. ")
                pass

        # Reweight based on spacing.
        if apply_grad:
            for l in self.models_labels.dtype.names:
                label = self.models_labels[l]
                ulabel = np.unique(label)  # grab underlying grid
                lngrad_label = np.log(np.gradient(ulabel))  # compute gradient
                lnprior += np.interp(label, ulabel, lngrad_label)  # add to lnp

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

        # Fit data.
        if verbose:
            sys.stderr.write('\rFitting object {0}/{1}'.format(1, Ndata))
            sys.stderr.flush()
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              parallax=parallax,
                                              parallax_err=parallax_err,
                                              Nmc_prior=Nmc_prior,
                                              lnprior=lnprior,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              Ndraws=Ndraws, rstate=rstate,
                                              lndistprior=lndistprior,
                                              data_coords=data_coords,
                                              ltol_subthresh=ltol_subthresh,
                                              logl_dim_prior=logl_dim_prior,
                                              ltol=ltol)):
            # Print progress.
            if verbose and i < Ndata - 1:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+2, Ndata))
                sys.stderr.flush()

            # Save results.
            idxs, scales, avs, covs_sa, Ndim, levid, chi2min = results
            out['idxs'][i] = idxs
            out['scales'][i] = scales
            out['avs'][i] = avs
            out['cov_sa'][i] = covs_sa
            out['Nbands'][i] = Ndim
            out['log_evidence'][i] = levid
            out['best_chi2'][i] = chi2min

        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        out.close()  # close output results file

    def _fit(self, data, data_err, data_mask,
             parallax=None, parallax_err=None, Nmc_prior=150,
             lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
             lndistprior=None, data_coords=None,
             logl_dim_prior=True, ltol=0.02, ltol_subthresh=0.005,
             rstate=None):
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
            integral over the prior. Default is `150`.

        lnprior : `~numpy.ndarray` of shape `(Ndata, Nfilt)`, optional
            Log-prior grid to be used. If not provided, will default
            to `0.`.

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

        lndistprior : func, optional
            The log-distsance prior function to be applied. If not provided,
            this will default to the galactic model from Green et al. (2014).

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

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance.

        Returns
        -------
        results : tuple
            Output of `lprob_func` yielded from the generator.

        """

        Ndata, Nmodels = len(data), self.NMODEL
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        if rstate is None:
            rstate = np.random
        mvn = rstate.multivariate_normal
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")

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

        # Fit data.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):
            results = loglike(x, xe, xm, self.models, self.avlim,
                              logl_dim_prior=logl_dim_prior,
                              ltol=ltol, wt_thresh=ltol_subthresh,
                              return_vals=True)

            lnlike, Ndim, chi2, scales, avs, ds2, da2, dsda = results
            lnpost = lnlike + lnprior
            pars = np.sqrt(scales)
            dists = 1. / pars
            coord = data_coords[i]

            # Apply rough distance prior for clipping.
            lnprob = lnpost + lndistprior(dists, coord)

            # Apply rough parallax prior for clipping.
            if parallax is not None:
                p, pe = parallax[i], parallax_err[i]
                lnprob += parallax_lnprior(pars, p, pe)

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
            scale, av, sinv2, ainv2, sainv = (scales[sel], avs[sel], ds2[sel],
                                              da2[sel], dsda[sel])
            cov_sa = np.array([np.linalg.inv(np.array([[s, sa], [sa, a]]))
                               for s, a, sa in zip(sinv2, ainv2, sainv)])

            # Now actually apply distance (and parallax) priors.
            if Nmc_prior > 0:
                # Use Monte Carlo integration to get an estimate of the
                # overlap integral.
                s_mc, a_mc = np.array([mvn(np.array([s, a]), c,
                                       size=Nmc_prior)
                                       for s, a, c in zip(scale, av,
                                                          cov_sa)]).T
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    par_mc = np.sqrt(s_mc)
                    dist_mc = 1. / par_mc
                    lnp_mc = lndistprior(dist_mc, coord)
                if parallax is not None:
                    lnp_mc += parallax_lnprior(par_mc, p, pe)
                # Ignore points that are out of bounds.
                inbounds = ((s_mc >= 0.) & (a_mc >= self.avlim[0]) &
                            (a_mc <= self.avlim[1]))
                lnp_mc[~inbounds] = -1e300
                Nmc_prior_eff = np.sum(inbounds, axis=0)
                # Compute integral.
                lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc_prior_eff)
            else:
                # Just assume the maximum-likelihood estimate.
                par = np.sqrt(scale)
                dist = 1. / par
                lnp = lndistprior(dist, coord)
                if parallax is not None:
                    lnp += parallax_lnprior(par, p, pe)
            lnpost += lnp

            # Compute GOF metrics.
            chi2min = np.min(chi2[sel])
            levid = logsumexp(lnpost)

            # Subsample.
            wt = np.exp(lnpost - levid)
            wt /= wt.sum()
            idxs = rstate.choice(Nsel, size=Ndraws, p=wt)
            sidxs = sel[idxs]
            scale, av, cov_sa = scales[sidxs], avs[sidxs], cov_sa[idxs]

            yield sidxs, scale, av, cov_sa, Ndim, levid, chi2min
