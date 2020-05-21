#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions.

"""

from __future__ import (print_function, division)
from six.moves import range

import sys
import numpy as np
import h5py
from scipy.special import xlogy, gammaln
from numba import jit

from math import log, gamma, erf, sqrt

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .filters import FILTERS

__all__ = ["_function_wrapper", "_adjoint3", "_inverse_transpose3",
           "_inverse3", "_dot3", "_isPSD", "_chisquare_logpdf",
           "_truncnorm_pdf", "_truncnorm_logpdf", "_get_seds",
           "load_models", "load_offsets",
           "quantile", "draw_sar", "sample_multivariate_normal",
           "magnitude", "inv_magnitude",
           "luptitude", "inv_luptitude", "add_mag", "get_seds",
           "phot_loglike", "photometric_offsets"]


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
        AI[..., i, :] = np.cross(A[..., i - 2, :], A[..., i - 1, :])

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

    Id = _adjoint3(A)
    det = _dot3(Id, A).mean(axis=-1)

    return Id / det[..., None, None]


def _inverse3(A):
    """
    Compute the inverse of a series of 3x3 matrices using adjoints.

    """

    return np.swapaxes(_inverse_transpose3(A), -1, -2)


def _isPSD(A):
    """
    Check if `A` is a positive semidefinite matrix.

    """

    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _chisquare_logpdf(x, df, loc=0, scale=1):
    """
    Compute log-PDF of a chi-square distribution.

    `_chisquare_logpdf(x, df, loc, scale)` is equal to
    `_chisquare_logpdf(y, df) - ln(scale)`, where `y = (x-loc)/scale`.
    NOTE: This function replicates `~scipy.stats.chisquare.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    df : float
        Degrees of freedom.

    loc : float, optional
        Offset of distribution.

    scale : float, optional
        Scaling of distribution.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)`, the natural log of the PDF.

    """

    if isinstance(x, list):
        x = np.asarray(x)

    y = (x - loc) / scale
    is_scalar = isinstance(y, (float, int))
    if is_scalar:
        if y <= 0:
            return -np.inf
    else:
        keys = y <= 0
        y[keys] = 0.1  # placeholder value, will actually return -np.inf

    ans = - log(2 ** (df / 2.) * gamma(df / 2.))
    ans = ans + (df / 2. - 1.) * np.log(y) - y / 2. - log(scale)

    if not is_scalar:
        ans[keys] = -np.inf

    return ans


def _truncnorm_pdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.pdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower cutoff of normal distribution.

    b : float
        Upper cutoff of normal distribution.

    loc : float, optional
        Mean of normal distribution.

    scale : float, optional
        Standard deviation of normal distribution.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)`, the PDF.

    """

    _a = scale * a + loc
    _b = scale * b + loc
    xi = (x - loc) / scale
    alpha = (_a - loc) / scale
    beta = (_b - loc) / scale

    phix = np.exp(-0.5 * xi ** 2) / np.sqrt(2. * np.pi)
    Phia = 0.5 * (1 + erf(alpha / np.sqrt(2)))
    Phib = 0.5 * (1 + erf(beta / np.sqrt(2)))

    ans = phix / (scale * (Phib - Phia))

    if not isinstance(x, (float, int)):
        keys = np.logical_or(x < _a, x > _b)
        ans[keys] = 0
    else:
        if x < _a or x > _b:
            ans = 0

    return ans


def _truncnorm_logpdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute log-PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower cutoff of normal distribution.

    b : float
        Upper cutoff of normal distribution.

    loc : float, optional
        Mean of normal distribution.

    scale : float, optional
        Standard deviation of normal distribution.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)`, the natural log pdf

    """

    _a = scale * a + loc
    _b = scale * b + loc

    xi = (x - loc) / scale
    alpha = (_a - loc) / scale
    beta = (_b - loc) / scale

    lnphi = -log(sqrt(2 * np.pi)) - 0.5 * np.square(xi)
    lndenom = (log(scale / 2.0) + log(erf(beta / np.sqrt(2))
                                      - erf(alpha / sqrt(2))))

    ans = np.subtract(lnphi, lndenom)

    if not isinstance(x, (float, int)):
        keys = np.logical_or(x < _a, x > _b)
        ans[keys] = -np.inf
    else:
        if x < _a or x > _b:
            ans = -np.inf

    return ans


@jit(nopython=True, cache=True)
def _get_seds(mag_coeffs, av, rv, return_flux=False):
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

    av : float or `~numpy.ndarray` of shape `(Nmodels)`
        Array of A(V) dust attenuation values.

    rv : float or `~numpy.ndarray` of shape `(Nmodels)`
        Array of R(V) dust attenuation curve "shape" values.

    return_flux : bool, optional
        Whether to return SEDs as flux densities instead of magnitudes.
        Default is `False`.

    Returns
    -------
    seds : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddened SEDs.

    rvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddening vectors.

    drvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Differential reddening vectors.

    """

    Nmodels, Nbands, Ncoef = mag_coeffs.shape
    seds = np.zeros((Nmodels, Nbands))
    rvecs = np.zeros((Nmodels, Nbands))
    drvecs = np.zeros((Nmodels, Nbands))

    fac = -0.4 * log(10.)

    for i in range(Nmodels):
        for j in range(Nbands):
            mags = mag_coeffs[i, j, 0]
            r0 = mag_coeffs[i, j, 1]
            dr = mag_coeffs[i, j, 2]

            # Compute SEDs.
            drvecs[i][j] = dr
            rvecs[i][j] = r0 + rv[i] * dr
            seds[i][j] = mags + av[i] * rvecs[i][j]

            # Convert to flux.
            if return_flux:
                seds[i][j] = 10. ** (-0.4 * seds[i][j])
                rvecs[i][j] *= fac * seds[i][j]
                drvecs[i][j] *= fac * seds[i][j]

    return seds, rvecs, drvecs


def load_models(filepath, filters=None, labels=None,
                include_ms=True, include_postms=True, include_binaries=False,
                verbose=True):
    """

    Parameters
    ----------
    filepath : string, optional
        The filepath of the models.

    filters : iterable of strings with length `Nfilt`, optional
        List of filters that will be loaded. If not provided, will default
        to all available filters. See the internally-defined `FILTERS` variable
        for more details on filter names. Any filters that are not available
        will be skipped over.

    labels : iterable of strings with length `Nlabel`, optional
        List of labels associated with the set of imported stellar models.
        Any labels that are not available will be skipped over.
        The default set is `['mini', 'feh', 'eep', 'smf'. 'loga', 'logl',`
        `'logt', 'logg', 'Mr', 'agewt']`.

    include_ms : bool, optional
        Whether to include objects on the Main Sequence. Applied as a cut on
        `eep <= 454` when `'eep'` is included. Default is `True`.

    include_postms : bool, optional
        Whether to include objects evolved off the Main Sequence. Applied as a
        cut on `eep > 454` when `'eep'` is included. Default is `True`.

    include_binaries : bool, optional
        Whether to include unresolved binaries. Applied as a cut on
        secondary mass fraction (`'smf'`) when it has been included. Default
        is `True`. If set to `False`, `'smf'` is not returned as a label.

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
        labels = ['mini', 'feh', 'eep', 'smf',
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

    # Apply cuts.
    sel = np.ones(len(combined_labels), dtype='bool')
    if include_ms and include_postms:
        sel = np.ones(len(combined_labels), dtype='bool')
    elif not include_ms and not include_postms:
        raise ValueError("If you don't include the Main Sequence and "
                         "Post-Main Sequence models you have nothing left!")
    elif include_postms:
        try:
            sel = combined_labels['eep'] > 454.
        except:
            pass
    elif include_ms:
        try:
            sel = combined_labels['eep'] <= 454.
        except:
            pass
    else:
        raise RuntimeError("Something has gone horribly wrong!")
    if not include_binaries and 'smf' in labels2:
        try:
            sel *= combined_labels['smf'] == 0.
            labels2 = [x for x in labels2 if x != 'smf']
        except:
            pass

    # Compile results.
    combined_labels = combined_labels[labels2]
    label_mask = label_mask[labels2]

    return models[sel], combined_labels[sel], label_mask


def load_offsets(filepath, filters=None, verbose=True):
    """

    Parameters
    ----------
    filepath : string, optional
        The filepath of the photometric offsets.

    filters : iterable of strings with length `Nfilt`, optional
        List of filters that will be loaded. If not provided, will default
        to all available filters. See the internally-defined `FILTERS` variable
        for more details on filter names. Any filters that are not available
        will be skipped over.

    verbose : bool, optional
        Whether to print a summary of the offsets. Default is `True`.

    Returns
    -------
    offsets : `~numpy.ndarray` of shape `(Nfilt)`
        Array of constants that will be *multiplied* to the *data* to account
        for offsets (i.e. multiplicative flux offsets).

    """

    # Initialize values.
    if filters is None:
        filters = FILTERS
    Nfilters = len(filters)

    # Read in offsets.
    filts, vals = np.loadtxt(filepath, dtype='str').T
    vals = vals.astype(float)

    # Fill in offsets where appropriate.
    offsets = np.full(Nfilters, np.nan)
    for i, filt in enumerate(filters):
        filt_idx = np.where(filts == filt)[0]  # get filter location
        if len(filt_idx) == 1:
            offsets[i] = vals[filt_idx[0]]  # insert offset
        elif len(filt_idx) == 0:
            offsets[i] = 1.  # assume no offset if not calibrated
        else:
            raise ValueError("Something went wrong when extracting "
                             "offsets for filter {}.".format(filt))

    if verbose:
        for filt, zp in zip(filters, offsets):
            sys.stderr.write('{0} ({1:3.2}%)\n'.format(filt, 100 * (zp - 1.)))

    return offsets


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
        try:
            # Attempt to use intel-specific version.
            rstate = np.random_intel
        except:
            # Fall back to default if not present.
            rstate = np.random

    # Generate realizations for each (scale, av, cov_sa) set.
    nsamps = len(scales)
    sdraws, adraws, rdraws = np.zeros((3, nsamps, ndraws))
    for i, (s, a, r, c) in enumerate(zip(scales, avs, rvs, covs_sar)):
        s_temp, a_temp, r_temp = [], [], []
        # Loop in case a significant chunk of draws are out-of-boudns.
        while len(s_temp) < ndraws:
            # Draw samples.
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


def sample_multivariate_normal(mean, cov, size=1, eps=1e-6, rstate=None):
    """
    Draw samples from many multivariate normal distributions.

    Returns samples from an arbitrary number of multivariate distributions.
    The multivariate distributions must all have the same dimension.
    NOTE: Provided covariances must be positive semi-definite
    (use `_isPSD` to check individual matrices if unsure).

    Parameters
    ----------
    mean : `~numpy.ndarray` of shape `(Ndist, dim)` or `(dim)`
        Means of the various multivariate distributions, where
        `Ndist` is the number of desired distributions and
        `dim` is the dimension of the distributions.

    cov : `~numpy.ndarray` of shape `(Ndist, dim, dim)`
        Covariances of the various multivariate distributions, where
        `Ndist` is the number of desired distributions and
        `dim` is the dimension of the distributions.

    size : float, optional
        Number of samples to draw from each distribution. Default is `1`.

    eps : float, optional
        Small factor added to covariances prior to Cholesky decomposition.
        Helps ensure numerical stability and should have no effect on the
        outcome. Default is `1e-6`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    samples : `~numpy.ndarray` of shape `(dim, size, Ndist)`
        Sampled values.

    """

    # If we have a single distribution, just revert to `numpy.random` version.
    if len(np.shape(mean)) == 1:
        return rstate.multivariate_normal(mean, cov, size=size)

    # Otherwise, proceed with the Cholesky decomposition.
    N, d = np.shape(mean)
    K = cov + eps * np.full((N, d, d), np.identity(d))
    L = np.linalg.cholesky(K)

    # Generate random samples from a standard iid normal distributions.
    z = rstate.normal(loc=0, scale=1, size=d * size * N).reshape(N, d, size)

    # Transform these samples to the appropriate correlated distributions.
    # In matrix form, this is just `ans = mean + L * u`.
    ans = np.repeat(mean[:, :, np.newaxis], size, axis=2) + np.matmul(L, z)
    ans = np.swapaxes(ans, 0, 1)
    ans = np.swapaxes(ans, 1, 2)

    return ans


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


def add_mag(mag1, mag2, f1=1., f2=1.):
    """
    Return combined magnitude from adding together individual components
    with corresponding weights.

    Parameters
    ----------
    mag1 : float or `~numpy.ndarray`
        Magnitude(s) of the first ("primary") component.

    mag2 : float or `~numpy.ndarray`
        Magnitude(s) of the second ("secondary") component.

    f1 : float or `~numpy.ndarray`
        Fraction of contribution from the first component. Default is `1.`.

    f2 : float or `~numpy.ndarray`
        Fraction of contribution from the second component. Default is `1.`.

    """

    flux1, flux2 = 10**(-0.4 * mag1), 10**(-0.4 * mag2)
    flux_tot = f1 * flux1 + f2 * flux2
    mag_tot = -2.5 * np.log10(flux_tot)

    return mag_tot


def get_seds(mag_coeffs, av=None, rv=None, return_flux=False,
             return_rvec=False, return_drvec=False):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    NOTE: This is a thin wrapper around `_get_seds` to preserve
    old functionality.

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
    elif isinstance(av, (int, float)):
        av = np.full(Nmodels, av)
    if rv is None:
        rv = np.full(Nmodels, 3.3)
    elif isinstance(rv, (int, float)):
        rv = np.full(Nmodels, rv)

    seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv,
                                    return_flux=return_flux)

    if return_rvec and return_drvec:
        return seds, rvecs, drvecs
    elif return_rvec:
        return seds, rvecs
    elif return_drvec:
        return seds, drvecs
    else:
        return seds


def phot_loglike(data, data_err, data_mask, models, dim_prior=True):
    """
    Computes the log-likelihood between noisy data and noiseless models.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape `(Nfilt)`
        Observed data values.

    data_err : `~numpy.ndarray` of shape `(Nfilt)`
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape `(Nfilt)`
        Binary mask (0/1) indicating whether the data was observed.

    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt)`
        Models predictions.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 3` degrees of freedom. Default is `True`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape `(Nmodel)`
        Log-likelihood values.

    """

    # Subselect only clean observations.
    Ndim = sum(data_mask)  # number of dimensions
    flux, fluxerr = data[data_mask], data_err[data_mask]  # mean, error
    mfluxes = models[:, data_mask]  # model predictions
    tot_var = np.square(fluxerr) + np.zeros_like(mfluxes)  # total variance

    # Compute residuals.
    resid = flux - mfluxes

    # Compute chi2.
    chi2 = np.sum(np.square(resid) / tot_var, axis=1)

    # Compute multivariate normal logpdf.
    lnl = -0.5 * chi2
    lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                   np.sum(np.log(tot_var), axis=1))

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 3)  # effective dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    return lnl


def photometric_offsets(phot, err, mask, models, idxs, reds, dreds, dists,
                        sel=None, weights=None, mask_fit=None, Nmc=150,
                        old_offsets=None, dim_prior=True,
                        prior_mean=None, prior_std=None, verbose=True,
                        rstate=None):
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

    weights : `~numpy.ndarray` of shape `(Nobj, Nsamps)`, optional
        Associated set of weights for each sample.

    mask_fit : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Boolean selection array of indicating the filters that were used
        in the fit. If a filter was used, the models will be re-weighted
        ignoring that band when computing the photometric offsets. If a filter
        was not used, then no additional re-weighting will be applied.
        If not provided, by default all bands will be assumed to have been
        used.

    Nmc : float, optional
        Number of realizations used to bootstrap the sample and
        average over the model realizations. Default is `150`.

    old_offsets : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Multiplicative photometric offsets that were applied to
        the data (i.e. `data_new = data * phot_offsets`) and errors
        when computing the fits.

    prior_mean : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Mean of Gaussian prior on the photometric offsets. Must be provided
        with `prior_std`.

    prior_std : `~numpy.ndarray` of shape `(Nfilt)`, optional
        Standard deviation of Gaussian prior on the photometric offsets.
        Must be provided with `prior_mean`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood when reweighting the data while cycling through each
        band. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 3` degrees of freedom. Default is `True`.

    verbose : bool, optional
        Whether to print progress to `~sys.stderr`. Default is `True`.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance.

    Returns
    -------
    ratios : `~numpy.ndarray` of shape `(Nfilt)`
        Median ratios of model / data.

    ratios_err : `~numpy.ndarray` of shape `(Nfilt)`
        Errors (bootstrapped) on ratios of model / data.

    nratio : `~numpy.ndarray` of shape `(Nfilt)`
        The number of objects used to compute `ratios`.

    """

    # Initialize values.
    Nobj, Nfilt = phot.shape
    Nsamps = idxs.shape[1]
    if sel is None:
        sel = np.ones(Nobj, dtype='bool')
    if weights is None:
        weights = np.ones((Nobj, Nsamps), dtype='float')
    if mask_fit is None:
        mask_fit = np.ones(Nfilt, dtype='bool')
    if old_offsets is None:
        old_offsets = np.ones(Nfilt)
    if rstate is None:
        try:
            # Attempt to use intel-specific version.
            rstate = np.random_intel
        except:
            # Fall back to default if not present.
            rstate = np.random

    # Generate SEDs.
    seds = get_seds(models[idxs.flatten()], av=reds.flatten(),
                    rv=dreds.flatten(), return_flux=True)
    seds /= dists.flatten()[:, None]**2  # scale based on distance
    seds = seds.reshape(Nobj, Nsamps, Nfilt)  # reshape back

    # Compute photometric ratios.
    ratios, nratio = np.ones(Nfilt), np.zeros(Nfilt, dtype='int')
    ratios_err = np.zeros(Nfilt)
    for i in range(Nfilt):
        # Subselect objects with reliable data.
        if mask_fit[i]:
            # Band was used in the fit. So we want to select objects that are:
            # 1. observed in the band,
            # 2. selected by user argument, and
            # 3. with >3 bands of photometry *excluding* the current band.
            s = np.where(mask[:, i] & sel & (np.sum(mask, axis=1) > 3 + 1) &
                         (np.sum(weights, axis=1) > 0))[0]
        else:
            # Band was not used in the fit. We will use the same criteria as
            # above, but do not need to impose an additional band restriction.
            s = np.where(mask[:, i] & sel & (np.sum(mask, axis=1) > 3) &
                         (np.sum(weights, axis=1) > 0))[0]
        n = len(s)
        nratio[i] = n
        if n > 0:
            # Compute photometric ratio.
            ratio = seds[s, :, i] / phot[s, None, i]
            if mask_fit[i]:
                # Compute weights from ignoring current band.
                mtemp = np.array(mask)
                mtemp[:, i] = False
                lnl = np.array([phot_loglike(p * old_offsets, e * old_offsets,
                                             mt, sed, dim_prior=dim_prior)
                                for p, e, mt, sed in zip(phot[s], err[s],
                                                         mtemp[s], seds[s])])
                levid = logsumexp(lnl, axis=1)
                logwt = lnl - levid[:, None]
                wt = np.exp(logwt)
            else:
                # Weights are uniform.
                wt = np.ones((n, Nsamps))
            wt *= weights[s]
            wt /= wt.sum(axis=1)[:, None]
            wt_obj = np.array(np.sum(weights[s], axis=1) > 0, dtype='float')
            wt_obj /= sum(wt_obj)
            # Bootstrap results.
            offsets = []
            for j in range(Nmc):
                if verbose:
                    sys.stderr.write('\rBand {0} ({1}/{2})     '
                                     .format(i + 1, j + 1, Nmc))
                    sys.stderr.flush()
                # Resample objects.
                ridx = rstate.choice(n, size=n, p=wt_obj)
                # Resample models based on computed weights (ignoring band).
                midx = [rstate.choice(Nsamps, p=w) for w in wt[ridx]]
                # Compute median.
                offsets.append(np.median(ratio[ridx, midx]))
            # Compute median (of median).
            ratios[i], ratios_err[i] = np.median(offsets), np.std(offsets)
    if verbose:
        sys.stderr.write('\n')

    # Apply prior.
    if prior_mean is not None and prior_std is not None:
        var_tot = ratios_err**2 + prior_std**2
        ratios = (ratios * prior_std**2 + prior_mean * ratios_err**2) / var_tot
        ratios_err = ratios_err * prior_std / np.sqrt(var_tot)

    return ratios, ratios_err, nratio
