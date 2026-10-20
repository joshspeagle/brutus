#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SED computation utilities for brutus.

This module contains functions for computing reddened spectral energy
distributions (SEDs) from magnitude coefficients and dust parameters.
"""

from math import exp, log

import numpy as np
from numba import jit, prange

__all__ = ["_get_seds", "get_seds"]


@jit(nopython=True, cache=True, parallel=True)
def _get_seds(mag_coeffs, av, rv, return_flux=False):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    This is the core function for computing reddened SEDs, optimized with numba
    for performance.

    Parameters
    ----------
    mag_coeffs : `~numpy.ndarray` of shape `(Nmodels, Nbands, 3)`
        Array of `(mag, R, dR/dRv)` coefficients used to generate
        reddened photometry in all bands. The first coefficient is the
        unreddened photometry, the second is the A(V) reddening vector for
        R(V)=0, and the third is the change in the reddening vector
        as a function of R(V).

    av : `~numpy.ndarray` of shape `(Nmodels,)`
        Array of A(V) dust attenuation values.

    rv : `~numpy.ndarray` of shape `(Nmodels,)`
        Array of R(V) dust attenuation curve "shape" values.

    return_flux : bool, optional
        Whether to return SEDs as flux densities instead of magnitudes.
        Default is `False`.

    Returns
    -------
    seds : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddened SEDs, in magnitudes, or in maggies if `return_flux`.

    rvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Reddening vectors. With `return_flux=False`, dm/dA(V) =
        R + R(V) * dR/dRv, in mag per mag of A(V). With `return_flux=True`,
        dF/dA(V), in maggies per mag of A(V).

    drvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`
        Differential reddening vectors. With `return_flux=False`,
        (1 / A(V)) dm/dR(V) = dR/dRv, in mag per mag of A(V). With
        `return_flux=True`, (1 / A(V)) dF/dR(V), in maggies per mag of
        A(V).

    """
    Nmodels, Nbands, Ncoef = mag_coeffs.shape
    # Every (i, j) entry is written below, so skip zero-initialization.
    seds = np.empty((Nmodels, Nbands))
    rvecs = np.empty((Nmodels, Nbands))
    drvecs = np.empty((Nmodels, Nbands))

    fac = -0.4 * log(10.0)

    for i in prange(Nmodels):
        for j in range(Nbands):
            mags = mag_coeffs[i, j, 0]
            r0 = mag_coeffs[i, j, 1]
            dr = mag_coeffs[i, j, 2]

            # Compute SEDs.
            drvecs[i][j] = dr
            rvecs[i][j] = r0 + rv[i] * dr
            seds[i][j] = mags + av[i] * rvecs[i][j]

            # Convert to flux. Use exp(fac * mag) instead of 10**(-0.4*mag):
            # mathematically identical (fac = -0.4*ln10), agrees to ~3e-15, but
            # a single exp is ~1.6x faster than a generic pow in this hot loop.
            if return_flux:
                seds[i][j] = exp(fac * seds[i][j])
                rvecs[i][j] *= fac * seds[i][j]
                drvecs[i][j] *= fac * seds[i][j]

    return seds, rvecs, drvecs


def _model_array(value, default, nmodels, name):
    """Return `value` as a float64 array of shape (nmodels,).

    None gives `default` for every model, and a scalar is broadcast. The numba
    kernels index ``value[i]`` for ``i < nmodels`` with no bounds check, so a
    wrong length raises here instead of reading out-of-bounds memory.
    """
    if value is None:
        return np.full(nmodels, float(default))
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        return np.full(nmodels, float(value))
    if value.shape != (nmodels,):
        raise ValueError(
            f"`{name}` must be a scalar or have shape ({nmodels},), one value "
            f"per model; got shape {value.shape}."
        )
    return np.ascontiguousarray(value)


def get_seds(
    mag_coeffs,
    av=None,
    rv=None,
    return_flux=False,
    return_rvec=False,
    return_drvec=False,
):
    """
    Compute reddened SEDs from the provided magnitude coefficients.

    This is a convenience wrapper around `_get_seds` that provides parameter
    handling and optional return values.

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
        Reddening vectors. Only returned if `return_rvec=True`.

    drvecs : `~numpy.ndarray` of shape `(Nmodels, Nbands)`, optional
        Differential reddening vectors. Only returned if `return_drvec=True`.

    Raises
    ------
    ValueError
        If `mag_coeffs` does not have shape `(Nmodels, Nbands, 3)`, or if
        `av` or `rv` is an array whose shape is not `(Nmodels,)`.

    Examples
    --------
    >>> import numpy as np
    >>> from brutus.core.sed_utils import get_seds
    >>>
    >>> # Create mock magnitude coefficients for 2 models, 3 bands
    >>> mag_coeffs = np.random.random((2, 3, 3))
    >>> mag_coeffs[:, :, 0] = 15.0  # Base magnitudes
    >>> mag_coeffs[:, :, 1] = 1.0   # R(V)=0 reddening
    >>> mag_coeffs[:, :, 2] = 0.1   # dR/dR(V)
    >>>
    >>> # Compute reddened SEDs
    >>> seds = get_seds(mag_coeffs, av=0.1, rv=3.1)
    >>> print(f"SED shape: {seds.shape}")

    >>> # Get SEDs and reddening vectors
    >>> seds, rvecs = get_seds(mag_coeffs, av=0.1, rv=3.1, return_rvec=True)

    >>> # Get all outputs
    >>> seds, rvecs, drvecs = get_seds(mag_coeffs, av=0.1, rv=3.1,
    ...                                return_rvec=True, return_drvec=True)

    Notes
    -----
    This function evaluates per-band reddening from the grid's precomputed
    coefficients, which parameterize the R(V)-dependent extinction curve of
    Fitzpatrick & Massa (2009) used throughout brutus (see the stellar
    models documentation). The reddening is applied as:

    A(λ) = A(V) * [R(V)=0 + R(V) * dR/dR(V)]

    where A(λ) is the extinction in a given band.
    """
    if mag_coeffs.ndim != 3 or mag_coeffs.shape[2] != 3:
        raise ValueError(
            "`mag_coeffs` must have shape (Nmodels, Nbands, 3); "
            f"got {mag_coeffs.shape}."
        )
    Nmodels = mag_coeffs.shape[0]

    av = _model_array(av, 0.0, Nmodels, "av")
    rv = _model_array(rv, 3.3, Nmodels, "rv")

    # Compute SEDs using the core function
    seds, rvecs, drvecs = _get_seds(mag_coeffs, av, rv, return_flux=return_flux)

    # Return requested outputs
    if return_rvec and return_drvec:
        return seds, rvecs, drvecs
    elif return_rvec:
        return seds, rvecs
    elif return_drvec:
        return seds, drvecs
    else:
        return seds
