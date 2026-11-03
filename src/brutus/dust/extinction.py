"""
Coordinate-to-HEALPix pixel index conversion for 3D dust maps.
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
        HEALPix ``nside`` resolution parameter (must be a power of 2).
    l : float or array_like
        Galactic longitude in degrees.
    b : float or array_like
        Galactic latitude in degrees.
    nest : bool, optional
        If True (default), use NESTED pixel ordering; if False, use RING
        ordering.

    Returns
    -------
    pix_ids : int or numpy.ndarray
        HEALPix pixel indices with the broadcast shape of ``l`` and ``b``
        (``int`` for scalar inputs, ``int64`` array otherwise). Coordinates
        with non-finite ``l`` or ``b`` outside ``[-90, 90]`` degrees return -1.

    Raises
    ------
    ValueError
        If ``l`` and ``b`` cannot be broadcast to a common shape.

    Examples
    --------
    >>> pix = lb2pix(nside=64, l=0.0, b=0.0)
    >>> isinstance(pix, int)
    True

    >>> l_arr = np.array([0.0, 90.0, 180.0])
    >>> b_arr = np.array([0.0, 30.0, -30.0])
    >>> pix_arr = lb2pix(nside=64, l=l_arr, b=b_arr)
    >>> pix_arr.shape
    (3,)

    >>> lb2pix(nside=64, l=0.0, b=95.0)
    -1
    """
    gal_l = l
    theta = np.radians(90.0 - b)
    phi = np.radians(gal_l)

    if not hasattr(gal_l, "__len__"):
        # The chained comparison evaluates to False when b is NaN.
        if not (np.isfinite(gal_l) and -90.0 <= b <= 90.0):
            return -1
        return int(hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest))

    l_arr = np.asarray(gal_l)
    b_arr = np.asarray(b)

    try:
        l_broadcast, b_broadcast = np.broadcast_arrays(l_arr, b_arr)
        theta_broadcast, phi_broadcast = np.broadcast_arrays(theta, phi)
    except ValueError as e:
        raise ValueError(
            f"Coordinate arrays have incompatible shapes: "
            f"l.shape={l_arr.shape}, b.shape={b_arr.shape}"
        ) from e

    pix_idx = np.empty(l_broadcast.shape, dtype="i8")
    valid_idx = (
        np.isfinite(l_broadcast) & (b_broadcast >= -90.0) & (b_broadcast <= 90.0)
    )
    pix_idx[valid_idx] = hp.pixelfunc.ang2pix(
        nside, theta_broadcast[valid_idx], phi_broadcast[valid_idx], nest=nest
    )
    pix_idx[~valid_idx] = -1

    return pix_idx
