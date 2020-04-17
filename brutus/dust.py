#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for manipulating 3-D dust maps.
Based on the `dustmaps` package by Greg Green (Green et al. 2018).

"""

from __future__ import (print_function, division)

import numpy as np
import h5py

import astropy.coordinates as coordinates
import astropy.units as units
import healpy as hp

__all__ = ["lb2pix", "DustMap", "Bayestar"]


def lb2pix(nside, l, b, nest=True):
    """
    Converts Galactic `(l, b)` coordinates to HEALPix pixel index.

    Parameters
    ----------
    nside : int
        The HEALPix `nside` parameter.

    l : float or `~numpy.ndarray`
        Galactic longitude (degrees).

    b : float or `~numpy.ndarray`
        Galactic latitude (degrees).

    nest : bool, optional
        Whether nested pixel ordering should be used instead of ring ordering.
        Default is `True` (nested pixel ordering).

    Returns
    -------
    pix_ids : int or `~numpy.ndarray`
        HEALPix pixel indices corresponding to the input `(l, b)` coordinates.

    """

    # Convert angles.
    theta = np.radians(90. - b)
    phi = np.radians(l)

    if not hasattr(l, '__len__'):
        # If we have a single `(l, b)`, check if we have a valid coordinate.
        if (b < -90.) or (b > 90.):
            return -1

        # Query our HEALPix pixels.
        pix_idx = hp.pixelfunc.ang2pix(nside, theta, phi, nest=nest)

        return pix_idx

    # Query our HEALPix pixels for all `(l, b)` values.
    pix_idx = np.empty(l.shape, dtype='i8')
    idx = (b >= -90.) & (b <= 90.)  # mask out bad coordinates
    pix_idx[idx] = hp.pixelfunc.ang2pix(nside, theta[idx], phi[idx], nest=nest)
    pix_idx[~idx] = -1

    return pix_idx


class DustMap(object):
    """
    Base class for querying dust maps.

    """

    def __init__(self):
        pass

    def __call__(self, coords, **kwargs):
        """
        An alias for `query`.

        """

        return self.query(coords, **kwargs)

    def query(self, coords, **kwargs):
        """
        Query the map at a set of coordinates.

        """

        raise NotImplementedError("`DustMap.query` must be implemented by "
                                  "subclasses.")

    def query_gal(self, ell, b, d=None, **kwargs):
        """
        Query using Galactic coordinates.

        Parameters
        ----------
        ell : float
            Galactic longitude in degrees or an `~astropy.unit.Quantity`.

        b : float
            Galactic latitude in degrees or an `~astropy.unit.Quantity`.

        d : float, optional
            Distance from the Solar System in kpc or an
            `~astropy.unit.Quantity`.

        Returns
        -------
        The results of the query, which must be implemented by derived
        classes.

        """

        if not isinstance(ell, units.Quantity):
            ell = ell * units.deg
        if not isinstance(b, units.Quantity):
            b = b * units.deg

        if d is None:
            coords = coordinates.SkyCoord(ell, b, frame='galactic')
        else:
            if not isinstance(d, units.Quantity):
                d = d * units.kpc
            coords = coordinates.SkyCoord(ell, b, distance=d, frame='galactic')

        return self.query(coords, **kwargs)

    def query_equ(self, ra, dec, d=None, frame='icrs', **kwargs):
        """
        Query using Equatorial coordinates. By default, the ICRS frame is used,
        although other frames implemented by `~astropy.coordinates` may also be
        specified.

        Parameters
        ----------
        ra : float
            Right ascension in degrees or an `~astropy.unit.Quantity`.

        dec : float
            Declination in degrees or an `~astropy.unit.Quantity`.

        d : float, optional
            Distance from the Solar System in kpc or an
            `~astropy.unit.Quantity`.

        frame : str, optional
            The coordinate system to be used. Options include `'icrs'`,
            `'fk4'`, `'fk5'`, and `'fk4noeterms'`. Default is `'icrs'`.

        Returns
        -------
        The results of the query, which must be implemented by derived
        classes.

        """

        valid_frames = ['icrs', 'fk4', 'fk5', 'fk4noeterms']

        if frame not in valid_frames:
            raise ValueError("`frame` {} not understood. Must be one of "
                             "valid frames {}.".format(frame, valid_frames))

        if not isinstance(ra, units.Quantity):
            ra = ra * units.deg
        if not isinstance(dec, units.Quantity):
            dec = dec * units.deg

        if d is None:
            coords = coordinates.SkyCoord(ra, dec, frame='icrs')
        else:
            if not isinstance(d, units.Quantity):
                d = d * units.kpc
            coords = coordinates.SkyCoord(ra, dec, distance=d, frame='icrs')

        return self.query(coords, **kwargs)


class Bayestar(DustMap):
    """
    Class to interpret outputs generated from the Bayestar 3D dust maps
    (Green et al. 2015, 2018), which cover the Pan-STARRS 1
    footprint (dec > -30 deg) over ~3/4 of the sky.

    """

    def __init__(self, dustfile='bayestar2019_v1.h5'):
        """
        Initialize the map from `dustfile`.

        """

        # Read in data.
        try:
            f = h5py.File(dustfile, 'r', libver='latest', swmr=True)
        except:
            f = h5py.File(dustfile, 'r')
            pass

        # Load pixel information.
        self._pixel_info = f['pixel_info'][:]
        self._n_pix = self._pixel_info.size

        # Load reddening information.
        self._distances = f['dists'][:]
        self._av_mean = f['av_mean'][:]
        self._av_std = f['av_std'][:]
        self._n_distances = len(self._distances)

        # Get healpix indices at each nside level.
        sort_idx = np.argsort(self._pixel_info,
                              order=['nside', 'healpix_index'])
        self._nside_levels = np.unique(self._pixel_info['nside'])
        self._hp_idx_sorted = []
        self._data_idx = []

        start_idx = 0
        for nside in self._nside_levels:
            end_idx = np.searchsorted(self._pixel_info['nside'], nside,
                                      side='right', sorter=sort_idx)
            idx = sort_idx[start_idx:end_idx]
            self._hp_idx_sorted.append(self._pixel_info['healpix_index'][idx])
            self._data_idx.append(idx)
            start_idx = end_idx

    def _find_data_idx(self, l, b):
        """
        Located index corresponding to given `(l, b)`.

        """

        pix_idx = np.empty(l.shape, dtype='i8')
        pix_idx[:] = -1

        # Search at each nside.
        for k, nside in enumerate(self._nside_levels):
            # Get pixels.
            ipix = lb2pix(nside, l, b, nest=True)

            # Find the insertion points of the query pixels in the large,
            # ordered pixel list.
            idx = np.searchsorted(self._hp_idx_sorted[k], ipix, side='left')

            # Determine which insertion points are beyond the edge
            # of the pixel list.
            in_bounds = (idx < self._hp_idx_sorted[k].size)

            if not np.any(in_bounds):
                continue

            # Determine which query pixels are correctly placed.
            idx[~in_bounds] = -1
            match_idx = (self._hp_idx_sorted[k][idx] == ipix)
            match_idx[~in_bounds] = False
            idx = idx[match_idx]

            if np.any(match_idx):
                pix_idx[match_idx] = self._data_idx[k][idx]

        return pix_idx

    def get_query_size(self, coords):
        """
        Check the total size of the query.

        """

        n_coords = np.prod(coords.shape, dtype=int)

        return n_coords * self._n_distances

    def query(self, coords):
        """
        Returns distances, mean(Av), and std(Av) at the requested coordinates.

        """

        # Extract the correct angular pixel(s).
        try:
            pix_idx = self._find_data_idx(coords.l.deg, coords.b.deg)
        except:
            # Execute 2-D query.
            cds = np.atleast_2d(coords)
            pix_idx = self._find_data_idx(cds[:, 0], cds[:, 1])
        in_bounds_idx = (pix_idx != -1)
        avmean, avstd = self._av_mean[pix_idx], self._av_std[pix_idx]
        avmean[~in_bounds_idx] = np.nan
        avstd[~in_bounds_idx] = np.nan

        # Convert back to 1-D if needed.
        if avmean.shape[0] == 1:
            avmean, avstd = avmean[0], avstd[0]

        return self._distances, avmean, avstd
