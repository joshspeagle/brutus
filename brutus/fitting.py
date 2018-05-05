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

from frankenz.fitting import BruteForce as BF

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["BruteForce", "imf_lnprior"]


def imf_lnprior(mgrid):
    """
    Apply Kroupa IMF prior.

    """

    lnprior = np.zeros_like(mgrid)
    low_mass = mgrid <= 0.08
    lnprior[low_mass] = -0.3 * np.log(mgrid[low_mass])
    mid_mass = (mgrid <= 0.5) & (mgrid > 0.08)
    lnprior[mid_mass] = -1.3 * np.log(mgrid[mid_mass]) + np.log(0.08)
    high_mass = mgrid > 0.5
    lnprior[high_mass] = -2.3 * np.log(mgrid[high_mass]) + np.log(0.5 * 0.08)

    return lnprior


class BruteForce():
    """
    Fits data and generates predictions using a simple brute-force approach.

    """

    def __init__(self, models, models_labels, models_params=None):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Model values.

        models_labels : `~numpy.ndarray` of shape (Nmodel, Nlabels)
            Labels for the models to be. These were
            used to compute the model grid.

        models_params : `~numpy.ndarray` of shape (Nmodel, Nparams)
            Output parameters for the models. These were output while
            constructing the grid based on the input labels.

        """

        # Initialize values.
        self.BF = BF(models, np.zeros_like(models), np.ones_like(models))
        self.models = self.BF.models
        self.models_err = self.BF.models_err
        self.models_mask = self.BF.models_mask
        self.models_labels = models_labels
        self.models_params = models_params
        self.NMODEL, self.NDIM = models.shape
        self.NLABELS = len(models_labels[0])
        self.NPARAMS = len(models_params[0])

    def fit(self, data, data_err, data_mask, data_labels, save_file,
            parallax=None, parallax_err=None, Nmc_parallax=50,
            lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
            apply_agewt=True, apply_grad=True, dim_prior=True,
            rstate=None, verbose=True):
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
            Default is `50`.

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
            with `Nfilt - 1` degrees of freedom. Default is `True`.

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
        out.create_dataset("scales_err", data=np.zeros((Ndata, Ndraws),
                                                       dtype='float32'))
        out.create_dataset("log_evidence", data=np.zeros(Ndata,
                                                         dtype='float32'))
        out.create_dataset("best_chi2", data=np.zeros(Ndata, dtype='float32'))
        out.create_dataset("Nbands", data=np.zeros(Ndata, dtype='int16'))

        # Fit data.
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              parallax=parallax,
                                              parallax_err=parallax_err,
                                              Nmc_parallax=Nmc_parallax,
                                              lnprior=lnprior,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              Ndraws=Ndraws, rstate=rstate,
                                              dim_prior=dim_prior)):
            # Print progress.
            if verbose:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+1, Ndata))
                sys.stderr.flush()

            # Save results.
            idxs, scales, scales_err, Ndim, levid, chi2min = results
            out['idxs'][i] = idxs
            out['scales'][i] = scales
            out['scales_err'][i] = scales_err
            out['Nbands'][i] = Ndim
            out['log_evidence'][i] = levid
            out['best_chi2'][i] = chi2min

        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        out.close()  # close output results file

    def _fit(self, data, data_err, data_mask,
             parallax=None, parallax_err=None, Nmc_parallax=50,
             lnprior=None, wt_thresh=1e-3, cdf_thresh=2e-4, Ndraws=2000,
             dim_prior=True, rstate=None):
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
            Default is `50`.

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
            with `Nfilt - 1` degrees of freedom. Default is `True`.

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
        if parallax is not None and parallax_err is None:
            raise ValueError("Must provide both `parallax` and "
                             "`parallax_err`.")

        # Initialize log(prior).
        if lnprior is None:
            lnprior = 0.
        self.lnprior = lnprior

        # Fit data.
        lprob_kwargs = {'ignore_model_err': True, 'free_scale': True,
                        'return_scale': True, 'dim_prior': dim_prior}
        for i, res in enumerate(self.BF._fit(data, data_err, data_mask,
                                             save_fits=False, track_scale=True,
                                             lprob_kwargs=lprob_kwargs)):
            _, lnlike, _, Ndim, chi2, scale, scale_err = res
            lnpost = lnlike + lnprior

            # Apply rough parallax prior for clipping.
            if parallax is not None:
                p, pe = parallax[i], parallax_err[i]
                sthresh = np.max(np.c_[scale, np.zeros_like(scale)], axis=1)
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
            s, se = scale[sel], scale_err[sel]
            Nsel = len(sel)

            # Now apply actual parallax prior.
            if parallax is not None:
                if Nmc_parallax > 0:
                    # Use Monte Carlo integration to get an estimate of the
                    # overlap integral.
                    s_mc = truncnorm.rvs((0. - s) / se, np.inf, s, se,
                                         size=(Nmc_parallax, Nsel),
                                         random_state=rstate)
                    lnp_mc = -0.5 * ((np.sqrt(s_mc) - p)**2 / pe**2 +
                                     np.sqrt(2. * np.pi) * pe)
                    lnp = logsumexp(lnp_mc, axis=0) - np.log(Nmc_parallax)
                    lnpost += lnp
                else:
                    # Just assume the maximum-likelihood estimate.
                    lnp = -0.5 * ((np.sqrt(s) - p)**2 / pe**2 +
                                  np.sqrt(2. * np.pi) * pe)
                    lnpost += lnp

            # Compute GOF metrics.
            chi2min = np.min(chi2[sel])
            levid = logsumexp(lnpost)

            # Subsample.
            wt = np.exp(lnpost - levid)
            idxs = rstate.choice(sel, size=Ndraws, p=wt)
            s, se = scale[idxs], scale_err[idxs]

            yield idxs, scale[idxs], scale_err[idxs], Ndim[0], levid, chi2min
