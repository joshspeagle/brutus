#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions.

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

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .filters import FILTERS

__all__ = ["_function_wrapper", "_adjoint3", "_inverse_transpose3",
           "_inverse3", "_dot3", "load_models", "quantile", "draw_sar",
           "magnitude", "inv_magnitude", "luptitude", "inv_luptitude",
           "get_seds", "photometric_offsets"]


class _function_wrapper(object):
    """
    A hack to make functions pickleable when `args` or `kwargs` are
    also included. Based on the implementation in
    `emcee <http://dan.iel.fm/emcee/>`_.

    """

    def __init__(self, func, args, kwargs, name='input'):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self, x):
        try:
            return self.func(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("Exception while calling {0} function:".format(self.name))
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def _adjoint3(A):
    """
    Compute the inverse of a series of 3x3 matrices without division
    by the determinant.

    """

    AI = np.empty_like(A)

    for i in range(3):
        AI[..., i, :] = np.cross(A[..., i-2, :], A[..., i-1, :])

    return AI


def _dot3(A, B):
    """
    Take the dot product of arrays of vectors, contracting over the
    last indices.

    """

    return np.einsum('...i,...i->...', A, B)


def _inverse_transpose3(A):
    """
    Compute the inverse-transpose of a series of 3x3 matrices.

    """

    I = _adjoint3(A)
    det = _dot3(I, A).mean(axis=-1)

    return I / det[..., None, None]


def _inverse3(A):
    """
    Compute the inverse of a series of 3x3 matrices using adjoints.

    """

    return np.swapaxes(_inverse_transpose3(A), -1, -2)


def load_models(filepath, filters=None, labels=None, verbose=True):
    """

    Parameters
    ----------
    filepath : string, optional
        The filepath to where the models are located.

    filters : iterable of strings with length `Nfilt`, optional
        List of filters that will be loaded. If not provided, will default
        to all available filters. See the internally-defined `FILTERS` variable
        for more details on filter names. Any filters that are not available
        will be skipped over.

    labels : iterable of strings with length `Nlabel`, optional
        List of labels associated with the set of imported stellar models.
        Any labels that are not available will be skipped over.
        The default set is
        `['mini', 'feh', 'eep', 'loga', 'logl', 'logt', 'logg', 'Mr', 'agewt']`

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    Returns
    -------
    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
        Array of models comprised of coefficients in each band used to
        describe the photometry as a function of reddening, parameterized
        in terms of Av.

    labels : structured `~numpy.ndarray` with dimensions `(Nmodel, Nlabel)`
        A structured array with the labels corresponding to each model.

    label_mask : structured `~numpy.ndarray` with dimensions `(1, Nlabel)`
        A structured array that masks ancillary labels associated with
        predictions (rather than those used to compute the model grid).

    """

    # Initialize values.
    if filters is None:
        filters = FILTERS
    if labels is None:
        labels = ['mini', 'feh', 'eep',
                  'loga', 'logl', 'logt', 'logg',
                  'Mr', 'agewt']

    # Read in models.
    try:
        f = h5py.File(filepath, 'r', libver='latest', swmr=True)
    except:
        f = h5py.File(filepath, 'r')
        pass
    mag_coeffs = f['mag_coeffs']
    models = np.zeros((len(mag_coeffs), len(filters), len(mag_coeffs[0][0])),
                      dtype='float32')
    for i, filt in enumerate(filters):
        try:
            models[:, i] = mag_coeffs[filt]  # fitted magnitude coefficients
            if verbose:
                sys.stderr.write('\rReading filter {}           '.format(filt))
                sys.stderr.flush()
        except:
            pass
    if verbose:
        sys.stderr.write('\n')

    # Remove extraneous/undefined filters.
    sel = np.all(models == 0., axis=(0, 2))
    models = models[:, ~sel, :]

    # Read in labels.
    combined_labels = np.full(len(models), np.nan,
                              dtype=np.dtype([(n, np.float) for n in labels]))
    label_mask = np.zeros(1, dtype=np.dtype([(n, np.bool) for n in labels]))
    try:
        # Grab "labels" (inputs).
        flabels = f['labels'][:]
        for n in flabels.dtype.names:
            if n in labels:
                combined_labels[n] = flabels[n]
                label_mask[n] = True
    except:
        pass
    try:
        # Grab "parameters" (predictions from labels).
        fparams = f['parameters'][:]
        for n in fparams.dtype.names:
            if n in labels:
                combined_labels[n] = fparams[n]
    except:
        pass

    # Remove extraneous/undefined labels.
    labels2 = [l for i, l in zip(combined_labels[0], labels) if ~np.isnan(i)]
    combined_labels = combined_labels[labels2]
    label_mask = label_mask[labels2]

    return models, combined_labels, label_mask


def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape `(nsamps,)`
        Input samples.

    q : `~numpy.ndarray` with shape `(nquantiles,)`
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape `(nsamps,)`, optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape `(nquantiles,)`
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def draw_sar(scales, avs, rvs, covs_sar, ndraws=500, avlim=(0., 6.),
             rvlim=(1., 8.), rstate=None):
    """
    Generate random draws from the joint scale and Av posterior for a
    given object.

    Parameters
    ----------
    scales : `~numpy.ndarray` of shape `(Nsamps)`
        An array of scale factors `s` derived between the models and the data.

    avs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of reddenings `A(V)` derived for the models.

    rvs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of reddening shapes `R(V)` derived for the models.

    covs_sar : `~numpy.ndarray` of shape `(Nsamps, 3, 3)`
        An array of covariance matrices corresponding to `(scales, avs, rvs)`.

    ndraws : int, optional
        The number of desired random draws. Default is `500`.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

    rvlim : 2-tuple, optional
        The Rv limits used to truncate results. Default is `(1., 8.)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    sdraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Scale-factor samples.

    adraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Reddening (A(V)) samples.

    rdraws : `~numpy.ndarray` of shape `(Nsamps, Ndraws)`
        Reddening shape (R(V)) samples.

    """

    if rstate is None:
        rstate = np.random

    # Generate realizations for each (scale, av, cov_sa) set.
    nsamps = len(scales)
    sdraws, adraws, rdraws = np.zeros((3, nsamps, ndraws))
    for i, (s, a, r, c) in enumerate(zip(scales, avs, rvs, covs_sar)):
        s_temp, a_temp, r_temp = [], [], []
        # Loop in case a significant chunk of draws are out-of-boudns.
        while len(s_temp) < ndraws:
            # Draw samples.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s_mc, a_mc, r_mc = rstate.multivariate_normal([s, a, r],
                                                              c, size=ndraws).T
            # Flag draws that are out of bounds.
            inbounds = ((s_mc >= 0.) &
                        (a_mc >= avlim[0]) & (a_mc <= avlim[1]) &
                        (r_mc >= rvlim[0]) & (r_mc <= rvlim[1]))
            s_mc, a_mc, r_mc = s_mc[inbounds], a_mc[inbounds], r_mc[inbounds]
            # Add to pre-existing samples.
            s_temp = np.append(s_temp, s_mc)
            a_temp = np.append(a_temp, a_mc)
            r_temp = np.append(r_temp, r_mc)
        # Cull any extra points.
        sdraws[i] = s_temp[:ndraws]
        adraws[i] = a_temp[:ndraws]
        rdraws[i] = r_temp[:ndraws]

    return sdraws, adraws, rdraws


def magnitude(phot, err, zeropoints=1.):
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
    mag_err = 2.5 / np.log(10.) * err / phot

    return mag, mag_err


def inv_magnitude(mag, err, zeropoints=1.):
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
    phot = 10**(-0.4 * mag) * zeropoints

    # Compute errors.
    phot_err = err * 0.4 * np.log(10.) * phot

    return phot, phot_err


def luptitude(phot, err, skynoise=1., zeropoints=1.):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). See Lupton et
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
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes errors corresponding to input `err`.

    """

    # Compute asinh magnitudes.
    mag = -2.5 / np.log(10.) * (np.arcsinh(phot / (2. * skynoise)) +
                                np.log(skynoise / zeropoints))

    # Compute errors.
    mag_err = np.sqrt(np.square(2.5 * np.log10(np.e) * err) /
                      (np.square(2. * skynoise) + np.square(phot)))

    return mag, mag_err


def inv_luptitude(mag, err, skynoise=1., zeropoints=1.):
    """
    Convert asinh magnitudes ("Luptitudes") to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitude errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

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

    # Compute photometry.
    phot = (2. * skynoise) * np.sinh(np.log(10.) / -2.5 * mag -
                                     np.log(skynoise / zeropoints))

    # Compute errors.
    phot_err = np.sqrt((np.square(2. * skynoise) + np.square(phot)) *
                       np.square(err)) / (2.5 * np.log10(np.e))

    return phot, phot_err


def get_seds(mag_coeffs, av=None, rv=None, return_flux=False,
             return_rvec=False, return_drvec=False):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    Parameters
    ----------
    mag_coeffs : `~numpy.ndarray` of shape `(Nmodels, Nbands, 3)`
        Array of `(mag, R, dR/dRv)` coefficients used to generate
        reddened photometry in all bands. The first coefficient is the
        unreddened photometry, the second is the A(V) reddening vector for
        R(V)=0, and the third is the change in the reddening vector
        as a function of R(V).

    av : float or `~numpy.ndarray` of shape `(Nmodels)`, optional
        Array of A(V) dust attenuation values.
        If not provided, defaults to `av=0.`.

    rv : float or `~numpy.ndarray` of shape `(Nmodels)`, optional
        Array of R(V) dust attenuation curve "shape" values.
        If not provided, defaults to `rv=3.3`.

    return_flux : bool, optional
        Whether to return SEDs as flux densities instead of magnitudes.
        Default is `False`.

    return_rvec : bool, optional
        Whether to return the reddening vectors at the provided
        `av` and `rv`. Default is `False`.

    return_drvec : bool, optional
        Whether to return the differential reddening vectors at the provided
        `av` and `rv`. Default is `False`.

    Returns
    -------
    seds : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddened SEDs.

    rvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`, optional
        Reddening vectors.

    drvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`, optional
        Differential reddening vectors.

    """

    Nmodels, Nbands, Ncoef = mag_coeffs.shape
    if av is None:
        av = np.zeros(Nmodels)
    if rv is None:
        rv = np.zeros(Nmodels) + 3.3

    # Turn provided Av values into polynomial features.
    mags = mag_coeffs[:, :, 0]
    r0 = mag_coeffs[:, :, 1]
    dr = mag_coeffs[:, :, 2]

    # Compute SEDs.
    drvecs = np.array(dr)
    rvecs = r0 + rv[:, None] * drvecs
    seds = mags + av[:, None] * rvecs

    # Convert to flux.
    if return_flux:
        seds = 10**(-0.4 * seds)
        if return_rvec:
            rvecs *= -0.4 * np.log(10.) * seds
        if return_drvec:
            drvecs *= -0.4 * np.log(10.) * seds

    if return_rvec and return_drvec:
        return seds, rvecs, drvecs
    elif return_rvec:
        return seds, rvecs
    elif return_drvec:
        return seds, drvecs
    else:
        return seds


def photometric_offsets(phot, err, mask, models, idxs, reds, dreds, dists,
                        sel=None, Nmc=500, rstate=None):
    """
    Compute (multiplicative) photometric offsets between data and model.

    Parameters
    ----------
    phot : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The observed fluxes for all our objects.

    err : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The associated flux errors for all our objects.

    mask : `~numpy.ndarray` of shape `(Nobj, Nfilt)`
        The associated band mask for all our objects.

    models : `~numpy.ndarray` of shape `(Nmodels, Nfilt, Ncoeffs)`
        Array of magnitude polynomial coefficients used to generate
        reddened photometry.

    idxs : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Set of models fit to each object.

    reds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of reddenings (Av values) derived for each object.

    dreds : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of reddening curve shapes (Rv values) derived
        for each object.

    dists : `~numpy.ndarray` of shape `(Nobj, Nsamps)`
        Associated set of distances (kpc) derived for each object.

    sel : `~numpy.ndarray` of shape `(Nobj)`, optional
        Boolean selection array of objects that should be used when
        computing offsets. If not provided, all objects will be used.

    Nmc : float, optional
        Number of realizations used to bootstrap the sample and
        average over the model realizations. Default is `500`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    ratios : `~numpy.ndarray` of shape `(Nfilt)`
        Median ratios of model / data.

    nratio : `~numpy.ndarray` of shape `(Nfilt)`
        The number of objects used to compute `ratios`.

    """

    # Initialize values.
    Nobj, Nfilt = phot.shape
    Nmodels = len(models)
    Nsamps = idxs.shape[1]
    if sel is None:
        sel = np.ones(Nobj, dtype='bool')
    if rstate is None:
        rstate = np.random

    # Generate SEDs.
    seds = get_seds(models[idxs.flatten()], av=reds.flatten(),
                    rv=dreds.flatten(), return_flux=True)
    seds /= dists.flatten()[:, None]**2  # scale based on distance
    seds = seds.reshape(Nobj, Nsamps, Nfilt)  # reshape back

    # Compute photometric ratios.
    ratios, nratio = np.ones(Nfilt), np.zeros(Nfilt, dtype='int')
    for i in range(Nfilt):
        # Subselect objects with reliable data.
        s = mask[:, i] & sel
        n = sum(s)
        nratio[i] = n
        if n > 0:
            # Compute photometric ratio.
            ratio = seds[s, :, i] / phot[s, None, i]
            # Bootstrap results.
            offsets = []
            for j in range(Nmc):
                ridx = rstate.choice(n, size=n)  # resample objects
                midx = rstate.choice(Nsamps, size=n)  # select random models
                offsets.append(np.median(ratio[ridx, midx]))  # compute median
            # Compute median (of median).
            ratios[i] = np.median(offsets)

    return ratios / np.median(ratios), nratio
