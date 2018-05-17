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
from copy import deepcopy
from itertools import product
import h5py
from scipy.interpolate import RegularGridInterpolator
import h5py
from scipy.stats import truncnorm
from scipy.special import xlogy, gammaln

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["imf_lnprior", "get_seds", "loglike", "BruteForce"]


def imf_lnprior(mgrid):
    """
    Apply Kroupa IMF prior over the provided initial mass grid.

    Parameters
    ----------
    mgrid : `~numpy.ndarray` of shape (Ngrid)
        Grid of initial mass the Kroupa IMF will be evaluated over.

    Returns
    -------
    lnprior : `~numpy.ndarray` of shape (Ngrid)
        The corresponding unnormalized ln(prior).

    """

    # Initialize log-prior.
    lnprior = np.zeros_like(mgrid)

    # Low mass.
    low_mass = mgrid <= 0.08
    lnprior[low_mass] = -0.3 * np.log(mgrid[low_mass])

    # Intermediate mass.
    mid_mass = (mgrid <= 0.5) & (mgrid > 0.08)
    lnprior[mid_mass] = -1.3 * np.log(mgrid[mid_mass]) + np.log(0.08)

    # High mass.
    high_mass = mgrid > 0.5
    lnprior[high_mass] = -2.3 * np.log(mgrid[high_mass]) + np.log(0.5 * 0.08)

    return lnprior


def get_seds(mag_coeffs, av, return_rvec=False, return_flux=False):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    Parameters
    ----------
    mag_coeffs : `~numpy.ndarray` of shape (Nmodels, Nbands, Ncoeffs)
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    av : `~numpy.ndarray` of shape (Nmodels)
        Array of dust attenuation values photometry should be predicted for.

    return_rvec : bool, optional
        Whether to return the differential reddening vectors at the provided
        `av`. Default is `False`.

    return_flux : bool, optional
        Whether to return SEDs as flux densities instead of magnitudes.
        Default is `False`.

    Returns
    -------
    seds : `~numpy.ndarray` of shape (Nmodels, Nbands)
        Reddened SEDs.

    rvecs : `~numpy.ndarray` of shape (Nmodels, Nbands), optional
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


def loglike(data, data_err, data_mask, mag_coeffs,
            models_avgrid=None, avgrid=None, avlim=(0., 6.),
            dim_prior=True, ltol=0.02, wt_thresh=0.005, return_vals=False,
            *args, **kwargs):
    """
    Computes the log-likelihood between noisy data and noiseless models
    optimizing over the scale-factor and dust attenuation.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape (Nfilt)
        Observed data values.

    data_err : `~numpy.ndarray` of shape (Nfilt)
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape (Nfilt)
        Binary mask (0/1) indicating whether the data was observed.

    mag_coeffs : `~numpy.ndarray` of shape (Nmodel, Nfilt, Ncoef)
        Magnitude coefficients used to compute reddened photometry for a given
        model.

    models_avgrid : `~numpy.ndarray` of shape (Nav, Nmodel, Nfilt), optional
        Precomputed SEDs (in flux densities) over a grid in Av. If provided,
        will be used to initialize the initial Av guess.

    avgrid : `~numpy.ndarray` of shape (Nav), optional
        The corresponding Av grid used to compute the SEDs in `models_avgrid`.

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
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape (Nmodel)
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape (Nmodel), optional
        The best-fit scale factor.

    Av : `~numpy.ndarray` of shape (Nmodel), optional
        The best-fit reddening.

    ds2 : `~numpy.ndarray` of shape (Nmodel), optional
        The second-derivative of the log-likelihood with respect to `s`
        around `s_ML` and `Av_ML`.

    da2 : `~numpy.ndarray` of shape (Nmodel), optional
        The second-derivative of the log-likelihood with respect to `Delta_Av`
        around `s_ML` and `Av_ML`.

    dsda : `~numpy.ndarray` of shape (Nmodel), optional
        The mixed-derivative of the log-likelihood with respect to `s` and
        `Delta_Av` around `s_ML` and `Av_ML`.

    """

    # Initialize values.
    Nmodels, Nfilt, Ncoef = mag_coeffs.shape
    tot_var = np.square(data_err) + np.zeros((Nmodels, Nfilt))  # variance
    tot_mask = data_mask * np.ones((Nmodels, Nfilt))  # binary mask
    Ndim = sum(data_mask)  # number of dimensions

    # Get a rough estimate of the Av by fitting pre-generated models over
    # a coarse grid in Av.
    av = np.zeros(Nmodels)
    if models_avgrid is not None and avgrid is not None:
        lnl_old = np.full(Nmodels, -np.inf)
        for i, av_val in enumerate(avgrid):
            models = models_avgrid[i]
            # Derive scale-factors (`scale`) between data and models.
            s_num = np.sum(tot_mask * models * data[None, :] / tot_var, axis=1)
            s_den = np.sum(tot_mask * np.square(models) / tot_var, axis=1)
            scale = s_num / s_den  # ML scalefactor
            # Compute chi2.
            resid = data - scale[:, None] * models
            chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)
            # Compute multivariate normal logpdf.
            lnl = -0.5 * chi2
            lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                           np.sum(np.log(tot_var), axis=1))
            # Assign next best guess.
            sel = lnl > lnl_old
            av[sel] = av_val
            lnl_old[sel] = lnl[sel]

    # Iterate until convergence.
    lnl_old, lerr = -1e300, 1e300
    while lerr > ltol:
        # Recompute models.
        models, rvecs = get_seds(mag_coeffs, av=av, return_rvec=True,
                                 return_flux=True)

        # Derive scale-factors (`scale`) between data and models.
        s_num = np.sum(tot_mask * models * data[None, :] / tot_var, axis=1)
        s_den = np.sum(tot_mask * np.square(models) / tot_var, axis=1)
        scale = s_num / s_den  # ML scalefactor
        scale[scale < 0.] = 0.  # must be non-negative

        # Rescale models and reddening vectors.
        models *= scale[:, None]
        rvecs *= scale[:, None]

        # Derive Delta_Av (`a`) between data and models.
        resid = data - models
        a_num = np.sum(tot_mask * rvecs * resid / tot_var, axis=1)
        a_den = np.sum(tot_mask * np.square(rvecs) / tot_var, axis=1)
        dav = a_num / a_den  # ML Delta_Av
        dav *= 5.  # take more aggressive steps

        # Prevent Av from sliding off the provided bounds.
        dav_low, dav_high = avlim[0] - av, avlim[1] - av
        lsel, hsel = dav < dav_low, dav > dav_high
        dav[lsel] = dav_low[lsel]
        dav[hsel] = dav_high[hsel]

        # Shift models.
        av += dav
        models += dav[:, None] * rvecs

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
        lnl_old = lnl

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 2)  # effective dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    if return_vals:
        rvecs /= scale[:, None]  # remove normalization
        sa_mix = np.sum(tot_mask * rvecs * (resid - models) / tot_var, axis=1)
        return lnl, Ndim, chi2, scale, av, s_den, a_den, sa_mix
    else:
        return lnl, Ndim, chi2


class BruteForce():
    """
    Fits data and generates predictions for scale-factors and reddening
    over a grid in initial mass, EEP, and metallicity.

    """

    def __init__(self, models, models_labels, models_params=None,
                 avgrid=None, avlim=(0., 6.)):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt, Ncoef)
            Magnitude coefficients used to compute reddened photometry over
            the desired bands for all models on the grid.

        models_labels : `~numpy.ndarray` of shape (Nmodel, Nlabels)
            Labels corresponding to each model on the grid.

        models_params : `~numpy.ndarray` of shape (Nmodel, Nparams), optional
            Output parameters for the models. These were output while
            constructing the grid based on the input labels.

        avgrid : `~numpy.ndarray` of shape (Nav), optional
            A grid of Av values. If provided, these will be used to precompute
            a grid of SEDS used to initialize the Av values used when fitting.

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

        # Create rough SED grid in Av.
        if avgrid is not None:
            self.models_avgrid = np.array([get_seds(models,
                                                    np.full(self.NMODEL, a),
                                                    return_rvec=False,
                                                    return_flux=True)
                                           for a in avgrid])
        else:
            self.models_avgrid = None
        self.avgrid = avgrid

    def fit(self, data, data_err, data_mask, data_labels, save_file,
            parallax=None, parallax_err=None, Nmc_parallax=150,
            lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
            apply_agewt=True, apply_grad=True, dim_prior=True, ltol=0.02,
            ltol_subthresh=0.005, rstate=None, verbose=True):
        """
        Fit all input models to the input data to compute the associated
        log-posteriors.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        data_labels : `~numpy.ndarray` of shape (Ndata, Nlabels)
            Labels for the data to be stored during runtime.

        save_file : str, optional
            File where results will be written out in HDF5 format.

        parallax : `~numpy.ndarray` of shape (Ndata), optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape (Ndata), optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        Nmc_parallax : int, optional
            The number of Monte Carlo realizations used to estimate the
            integral of the product of `P(parallax|given) * P(parallax|fix)`.
            Default is `150`.

        lnprior : `~numpy.ndarray` of shape (Ndata, Nfilt), optional
            Log-prior grid to be used. If not provided, will default
            to a Kroupa IMF prior in initial mass (`'mini'`) and
            uniform priors in age, metallicity, and dust.

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

        dim_prior : bool, optional
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
            lnprior = imf_lnprior(self.models_labels['mini'])

        # Apply age weights to reweight from EEP to age.
        if apply_agewt:
            try:
                lnprior += np.log(self.models_params['agewt'])
            except:
                warnings.warn("No age weights provided in `models_params`. "
                              "Unable to apply age weights!")
                pass

        # Reweight based on spacing.
        if apply_grad:
            for l in self.models_labels.dtype.names:
                label = self.models_labels[l]
                ulabel = np.unique(label)  # grab underlying grid
                lngrad_label = np.log(np.gradient(ulabel))  # compute gradient
                lnprior += np.interp(label, ulabel, lngrad_label)  # add to lnp

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
                                              Nmc_parallax=Nmc_parallax,
                                              lnprior=lnprior,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              Ndraws=Ndraws, rstate=rstate,
                                              ltol_subthresh=ltol_subthresh,
                                              dim_prior=dim_prior, ltol=ltol)):
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
             parallax=None, parallax_err=None, Nmc_parallax=150,
             lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
             dim_prior=True, ltol=0.02, ltol_subthresh=0.005, rstate=None):
        """
        Internal generator used to compute fits.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        parallax : `~numpy.ndarray` of shape (Ndata), optional
            Parallax measurements to be used as a prior.

        parallax_err : `~numpy.ndarray` of shape (Ndata), optional
            Errors on the parallax measurements. Must be provided along with
            `parallax`.

        Nmc_parallax : int, optional
            The number of Monte Carlo realizations used to evaluate the
            integral of the product of `P(parallax|given) * P(parallax|fix)`.
            Default is `150`.

        lnprior : `~numpy.ndarray` of shape (Ndata, Nfilt), optional
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

        dim_prior : bool, optional
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

        # Fit data.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):
            results = loglike(x, xe, xm, self.models, self.models_avgrid,
                              self.avgrid, self.avlim, dim_prior=dim_prior,
                              ltol=ltol, wt_thresh=ltol_subthresh,
                              return_vals=True)

            lnlike, Ndim, chi2, scales, avs, ds2, da2, dsda = results
            lnpost = lnlike + lnprior

            # Apply rough parallax prior for clipping.
            if parallax is not None:
                p, pe = parallax[i], parallax_err[i]
                sthresh = np.max(np.c_[scales, np.zeros_like(scales)], axis=1)
                lnprob = -0.5 * (np.sqrt(sthresh) - p)**2 / pe**2
                lnprob += lnpost
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
            scale, av, sinv2, ainv2, sainv = (scales[sel], avs[sel], ds2[sel],
                                              da2[sel], dsda[sel])
            cov_sa = np.array([np.linalg.inv(np.array([[s, sa], [sa, a]]))
                               for s, a, sa in zip(sinv2, ainv2, sainv)])

            # Now apply actual parallax prior.
            if parallax is not None:
                if Nmc_parallax > 0:
                    # Use Monte Carlo integration to get an estimate of the
                    # overlap integral.
                    s_mc, a_mc = np.array([mvn(np.array([s, a]), c,
                                           size=Nmc_parallax)
                                           for s, a, c in zip(scale, av,
                                                              cov_sa)]).T
                    inbounds = ((s_mc >= 0.) & (a_mc >= self.avlim[0]) &
                                (a_mc <= self.avlim[1]))
                    lnp_mc = -0.5 * ((np.sqrt(s_mc) - p)**2 / pe**2 +
                                     np.log(2. * np.pi * pe**2))
                    lnp_mc[~inbounds] = -1e300
                    Nmc_parallax_eff = np.sum(inbounds, axis=0)
                    lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc_parallax_eff)
                    lnpost += lnp
                else:
                    # Just assume the maximum-likelihood estimate.
                    lnp = -0.5 * ((np.sqrt(scale) - p)**2 / pe**2 +
                                  np.log(2. * np.pi * pe**2))
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
