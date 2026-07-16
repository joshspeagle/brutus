#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extinction and coordinate utilities for dust mapping.

This module provides utilities for coordinate transformations and
extinction calculations used in 3D dust mapping.
"""

import healpy as hp  # type: ignore
import numpy as np

__all__ = ["lb2pix"]


def lb2pix(nside, l, b, nest=True):  # noqa: E741
    """
    Convert Galactic (l, b) coordinates to HEALPix pixel indices.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter. Must be a power of 2.
    l : float or array_like
        Galactic longitude in degrees.
    b : float or array_like
        Galactic latitude in degrees.
    nest : bool, optional
        Whether to use nested pixel ordering instead of ring ordering.
        Default is True.

    Returns
    -------
    pix_ids : int or ndarray
        HEALPix pixel indices corresponding to the input (l, b) coordinates.
        Invalid coordinates (absolute b > 90°) return -1.

    Examples
    --------
    >>> # Single coordinate
    >>> pix = lb2pix(nside=64, l=0.0, b=0.0)
    >>> isinstance(pix, int)
    True

    >>> # Multiple coordinates
    >>> l_arr = np.array([0.0, 90.0, 180.0])
    >>> b_arr = np.array([0.0, 30.0, -30.0])
    >>> pix_arr = lb2pix(nside=64, l=l_arr, b=b_arr)
    >>> len(pix_arr) == 3
    True

    >>> # Invalid coordinate
    >>> invalid_pix = lb2pix(nside=64, l=0.0, b=95.0)
    >>> invalid_pix == -1
    True
    """
    # Rename to avoid E741 (ambiguous variable 'l')
    gal_l = l

    # Convert angles to spherical coordinates
    theta = np.radians(90.0 - b)
    phi = np.radians(gal_l)

    # Handle scalar inputs
    if not hasattr(gal_l, "__len__"):
        # Check for valid coordinate. The chained comparison is False for
        # NaN, matching the array branch's mask (which treats NaN as
        # invalid) instead of falling through to a healpy ValueError.
        if not (-90.0 <= b <= 90.0):
            return -1

        # Query HEALPix pixel
        pix_idx = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)
        return int(pix_idx)

    # Handle array inputs - use broadcasting
    l_arr = np.asarray(gal_l)
    b_arr = np.asarray(b)

    # Use numpy broadcasting to handle different shapes
    try:
        l_broadcast, b_broadcast = np.broadcast_arrays(l_arr, b_arr)
        theta_broadcast, phi_broadcast = np.broadcast_arrays(theta, phi)
    except ValueError as e:
        raise ValueError(
            f"Coordinate arrays have incompatible shapes: l.shape={l_arr.shape}, b.shape={b_arr.shape}"
        ) from e

    # Initialize output array with broadcast shape
    pix_idx = np.empty(l_broadcast.shape, dtype="i8")

    # Mask for valid coordinates
    valid_idx = (b_broadcast >= -90.0) & (b_broadcast <= 90.0)

    # Compute pixels for valid coordinates
    pix_idx[valid_idx] = hp.pixelfunc.ang2pix(
        nside, theta_broadcast[valid_idx], phi_broadcast[valid_idx], nest=nest
    )

    # Set invalid coordinates to -1
    pix_idx[~valid_idx] = -1

    return pix_idx
