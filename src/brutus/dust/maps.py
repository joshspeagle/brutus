#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D dust map implementations.

This module provides classes for querying 3D dust maps, particularly the
Bayestar maps from Green et al. (2019).
"""

import astropy.coordinates as coordinates
import astropy.units as units
import h5py
import numpy as np

from .extinction import lb2pix

__all__ = ["DustMap", "Bayestar"]


class DustMap:
    """
    Base class for querying 3D dust maps.

    This abstract base class defines the interface that all dust map
    implementations should follow.
    """

    def __init__(self):
        """Initialize the dust map."""
        pass

    def __call__(self, coords, **kwargs):
        """
        Convenience method for querying the map.

        This is an alias for the `query` method.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Coordinates to query.
        **kwargs
            Additional keyword arguments passed to query.

        Returns
        -------
        Query results as implemented by subclasses.
        """
        return self.query(coords, **kwargs)

    def query(self, coords, **kwargs):
        """
        Query the map at a set of coordinates.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Coordinates to query.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Query results as implemented by subclasses.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("DustMap.query must be implemented by subclasses.")

    def query_gal(self, ell, b, d=None, **kwargs):
        """
        Query the map using Galactic coordinates.

        Parameters
        ----------
        ell : float or astropy.units.Quantity
            Galactic longitude in degrees.
        b : float or astropy.units.Quantity
            Galactic latitude in degrees.
        d : float or astropy.units.Quantity, optional
            Distance from the Solar System in kpc. Not used by HEALPix-based
            maps but accepted for API compatibility.
        **kwargs
            Additional keyword arguments passed to query.

        Returns
        -------
        Query results as implemented by subclasses.
        """
        # Extract numeric values from astropy Quantities if needed
        if isinstance(ell, units.Quantity):
            ell = ell.to(units.deg).value
        if isinstance(b, units.Quantity):
            b = b.to(units.deg).value

        # Pass (l, b) arrays directly to query, avoiding SkyCoord overhead
        ell = np.atleast_1d(np.asarray(ell, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        coords = np.column_stack([ell, b])

        return self.query(coords, **kwargs)

    def query_equ(self, ra, dec, d=None, frame="icrs", **kwargs):
        """
        Query the map using Equatorial coordinates.

        Parameters
        ----------
        ra : float or astropy.units.Quantity
            Right ascension in degrees.
        dec : float or astropy.units.Quantity
            Declination in degrees.
        d : float or astropy.units.Quantity, optional
            Distance from the Solar System in kpc.
        frame : str, optional
            Coordinate frame. Options: 'icrs', 'fk4', 'fk5', 'fk4noeterms'.
            Default is 'icrs'.
        **kwargs
            Additional keyword arguments passed to query.

        Returns
        -------
        Query results as implemented by subclasses.

        Raises
        ------
        ValueError
            If frame is not one of the supported coordinate frames.
        """
        valid_frames = ["icrs", "fk4", "fk5", "fk4noeterms"]

        if frame not in valid_frames:
            raise ValueError(
                f"Frame '{frame}' not supported. Must be one of {valid_frames}."
            )

        # Handle units
        if not isinstance(ra, units.Quantity):
            ra = ra * units.deg
        if not isinstance(dec, units.Quantity):
            dec = dec * units.deg

        # Create coordinate object
        if d is None:
            coords = coordinates.SkyCoord(ra, dec, frame=frame)
        else:
            if not isinstance(d, units.Quantity):
                d = d * units.kpc
            coords = coordinates.SkyCoord(ra, dec, distance=d, frame=frame)

        return self.query(coords, **kwargs)


class Bayestar(DustMap):
    """
    Query the Bayestar 3D dust maps from Green et al. (2019).

    The Bayestar maps cover the Pan-STARRS 1 footprint (dec > -30°) over
    approximately 3/4 of the sky, providing 3D extinction information.

    Parameters
    ----------
    dustfile : str, optional
        Path to the Bayestar HDF5 data file. Default is 'bayestar2019_v1.h5'.

    Attributes
    ----------
    _distances : ndarray
        Distance grid points (kpc).
    _av_mean : ndarray
        Mean A(V) extinction values.
    _av_std : ndarray
        Standard deviation of A(V) extinction values.
    """

    def __init__(self, dustfile="bayestar2019_v1.h5"):
        """
        Initialize the Bayestar dust map.

        Parameters
        ----------
        dustfile : str, optional
            Path to the Bayestar HDF5 data file.
        """
        super().__init__()

        # Open the HDF5 file
        try:
            # Try SWMR mode first (for concurrent access)
            f = h5py.File(dustfile, "r", libver="latest", swmr=True)
        except (OSError, ValueError):
            # Fall back to regular mode
            f = h5py.File(dustfile, "r")

        try:
            # Load pixel information
            self._pixel_info = f["pixel_info"][:]
            self._n_pix = self._pixel_info.size

            # Load extinction data
            self._distances = f["dists"][:]
            self._av_mean = f["av_mean"][:]
            self._av_std = f["av_std"][:]
            self._n_distances = len(self._distances)

            # Prepare HEALPix index lookup structures
            self._prepare_index_structures()

        finally:
            f.close()

    def _prepare_index_structures(self):
        """Prepare optimized lookup structures for HEALPix indices."""
        # Sort pixels by nside and healpix_index for efficient searching
        sort_idx = np.argsort(self._pixel_info, order=["nside", "healpix_index"])

        self._nside_levels = np.unique(self._pixel_info["nside"])
        self._hp_idx_sorted = []
        self._data_idx = []

        start_idx = 0
        for nside in self._nside_levels:
            # Find pixels at this nside level
            end_idx = np.searchsorted(
                self._pixel_info["nside"], nside, side="right", sorter=sort_idx
            )

            idx = sort_idx[start_idx:end_idx]

            # Store sorted HEALPix indices and corresponding data indices
            self._hp_idx_sorted.append(self._pixel_info["healpix_index"][idx])
            self._data_idx.append(idx)

            start_idx = end_idx

    def _find_data_idx(self, gal_l, b):
        """
        Find data indices corresponding to Galactic coordinates.

        Parameters
        ----------
        gal_l : array_like
            Galactic longitude(s) in degrees.
        b : array_like
            Galactic latitude(s) in degrees.

        Returns
        -------
        pix_idx : ndarray
            Data indices for each coordinate. Invalid coordinates return -1.
        """
        # Ensure arrays and get shape
        l_arr = np.asarray(gal_l)
        b_arr = np.asarray(b)
        pix_idx = np.full(l_arr.shape, -1, dtype="i8")

        # Search at each nside level (coarse to fine resolution)
        for k, nside in enumerate(self._nside_levels):
            # Convert coordinates to HEALPix pixel indices
            ipix = lb2pix(nside, l_arr, b_arr, nest=True)

            # Find insertion points in the sorted pixel list
            idx = np.searchsorted(self._hp_idx_sorted[k], ipix, side="left")

            # Handle scalar case
            if np.isscalar(idx):
                if (
                    idx < len(self._hp_idx_sorted[k])
                    and self._hp_idx_sorted[k][idx] == ipix
                ):
                    pix_idx[...] = self._data_idx[k][idx]
            else:
                # Check bounds for array case
                in_bounds = idx < len(self._hp_idx_sorted[k])

                if not np.any(in_bounds):
                    continue

                # Check for exact matches
                idx = np.where(in_bounds, idx, -1)
                match_idx = in_bounds & (self._hp_idx_sorted[k][idx] == ipix)

                if np.any(match_idx):
                    valid_idx = idx[match_idx]
                    pix_idx[match_idx] = self._data_idx[k][valid_idx]

        return pix_idx

    def get_query_size(self, coords):
        """
        Estimate the total size of a query result.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Coordinates that would be queried.

        Returns
        -------
        int
            Estimated total number of data points that would be returned.
        """
        n_coords = np.prod(coords.shape, dtype=int)
        return n_coords * self._n_distances

    def query(self, coords):
        """
        Query extinction at the specified coordinates.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Coordinates to query. Can be single coordinate or array.

        Returns
        -------
        distances : ndarray
            Distance grid points (kpc).
        av_mean : ndarray
            Mean A(V) extinction values along each line of sight.
        av_std : ndarray
            Standard deviation of A(V) extinction values.

        Notes
        -----
        For coordinates outside the map coverage, NaN values are returned.
        """
        try:
            # Try to access as SkyCoord object - convert to Galactic if needed
            if hasattr(coords, "galactic"):
                gal_coords = coords.galactic
            else:
                gal_coords = coords
            l_deg = gal_coords.l.deg
            b_deg = gal_coords.b.deg
        except AttributeError:
            # Handle as array of coordinates [l, b] in degrees
            coords_arr = np.atleast_2d(coords)
            l_deg = coords_arr[:, 0]
            b_deg = coords_arr[:, 1]

        # Find corresponding data indices
        pix_idx = self._find_data_idx(l_deg, b_deg)

        # Extract extinction data
        in_bounds = pix_idx != -1
        av_mean = self._av_mean[pix_idx].copy()
        av_std = self._av_std[pix_idx].copy()

        # Set out-of-bounds values to NaN
        av_mean[~in_bounds] = np.nan
        av_std[~in_bounds] = np.nan

        # Handle scalar case - check if input was scalar
        scalar_input = (hasattr(coords, "isscalar") and coords.isscalar) or (
            not hasattr(coords, "__len__") and np.isscalar(l_deg)
        )

        if scalar_input and av_mean.shape[0] == 1:
            av_mean = av_mean[0]
            av_std = av_std[0]

        return self._distances, av_mean, av_std
