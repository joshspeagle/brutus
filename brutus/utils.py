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

__all__ = ["load_models", "quantile", "draw_sav",
           "magnitude", "inv_magnitude", "luptitude", "inv_luptitude"]


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
    f = h5py.File(filepath)
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


def draw_sav(scales, avs, covs_sa, ndraws=500, avlim=(0., 6.), rstate=None):
    """
    Generate random draws from the joint scale and Av posterior for a
    given object.

    Parameters
    ----------
    scales : `~numpy.ndarray` of shape `(Nsamps)`
        An array of scale factors derived between the model and the data.

    avs : `~numpy.ndarray` of shape `(Nsamps)`
        An array of reddenings derived between the model and the data.

    covs_sa : `~numpy.ndarray` of shape `(Nsamps, 2, 2)`
        An array of covariance matrices corresponding to `scales` and `avs`.

    ndraws : int, optional
        The number of desired random draws. Default is `500`.

    avlim : 2-tuple, optional
        The Av limits used to truncate results. Default is `(0., 6.)`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    """

    if rstate is None:
        rstate = np.random

    # Generate realizations for each (scale, av, cov_sa) set.
    nsamps = len(scales)
    sdraws, adraws = np.zeros((nsamps, ndraws)), np.zeros((nsamps, ndraws))
    for i, (s, a, c) in enumerate(zip(scales, avs, covs_sa)):
        s_temp, a_temp = [], []
        # Loop is just in case a significant chunk of the distribution
        # falls outside of the bounds.
        while len(s_temp) < ndraws:
            s_mc, a_mc = rstate.multivariate_normal(np.array([s, a]), c,
                                                    size=ndraws*4).T
            inbounds = ((s_mc >= 0.) & (a_mc >= avlim[0]) &
                        (a_mc <= avlim[1]))  # flag out-of-bounds draws
            s_mc, a_mc = s_mc[inbounds], a_mc[inbounds]
            s_temp, a_temp = np.append(s_temp, s_mc), np.append(a_temp, a_mc)
        sdraws[i], adraws[i] = s_temp[:ndraws], a_temp[:ndraws]

    return sdraws, adraws


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
