#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D dust map implementations.

This module provides classes for querying 3D dust maps, particularly the
Bayestar maps from Green et al. (2019).
"""

import warnings

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
    apply_reliability_mask : bool, optional
        Whether queries mask (set to NaN) extinction values that Green et
        al. (2019) flag as unreliable: distance bins outside a pixel's
        reliable distance-modulus range (``DM_reliable_min``/``DM_reliable_max``)
        and pixels whose fits did not converge (``converged == 0``). NaN
        values degrade to a uniform prior in
        `~brutus.priors.extinction.logp_extinction`. Can be overridden per
        query. If the data file lacks the reliability fields, masking is
        skipped with a warning. Default is True.

    Attributes
    ----------
    _distances : ndarray
        Distance grid points (kpc).
    _av_mean : ndarray
        Mean A(V) extinction values.
    _av_std : ndarray
        Standard deviation of A(V) extinction values.
    _d_reliable_min : ndarray or None
        Per-pixel minimum reliable distance (kpc), or None if the file
        lacks reliability metadata.
    _d_reliable_max : ndarray or None
        Per-pixel maximum reliable distance (kpc), or None if the file
        lacks reliability metadata.
    _converged : ndarray or None
        Per-pixel convergence flags, or None if the file lacks
        reliability metadata.
    """

    def __init__(self, dustfile="bayestar2019_v1.h5", apply_reliability_mask=True):
        """
        Initialize the Bayestar dust map.

        Parameters
        ----------
        dustfile : str, optional
            Path to the Bayestar HDF5 data file.
        apply_reliability_mask : bool, optional
            Default masking behavior for queries (see class docstring).
            Default is True.
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

            # Load per-pixel reliability metadata (Green et al. 2019):
            # convergence flags and the distance-modulus range over which
            # each profile is constrained by data. Older or repackaged
            # files may lack these fields; degrade gracefully.
            self._load_reliability_info(dustfile, apply_reliability_mask)

            # Prepare HEALPix index lookup structures
            self._prepare_index_structures()

        finally:
            f.close()

    def _load_reliability_info(self, dustfile, apply_reliability_mask):
        """Extract reliability metadata from pixel_info, if present."""
        required = {"converged", "DM_reliable_min", "DM_reliable_max"}
        names = set(self._pixel_info.dtype.names or ())

        if required <= names:
            # DM -> distance (kpc): d = 10**(DM/5 - 2). Non-finite DM
            # bounds (unconstrained pixels) map to 0/inf and thus mask
            # the entire profile, which is the intended semantics.
            with np.errstate(over="ignore"):
                self._d_reliable_min = 10.0 ** (
                    self._pixel_info["DM_reliable_min"].astype(np.float64) / 5.0 - 2.0
                )
                self._d_reliable_max = 10.0 ** (
                    self._pixel_info["DM_reliable_max"].astype(np.float64) / 5.0 - 2.0
                )
            self._converged = self._pixel_info["converged"].astype(bool)
        else:
            self._d_reliable_min = None
            self._d_reliable_max = None
            self._converged = None
            if apply_reliability_mask:
                warnings.warn(
                    f"Dust map file '{dustfile}' lacks reliability metadata "
                    "(converged, DM_reliable_min, DM_reliable_max); queries "
                    "will return unmasked extinction profiles.",
                    RuntimeWarning,
                )

        self._apply_reliability_mask = bool(apply_reliability_mask)

    def _prepare_index_structures(self):
        """Prepare optimized lookup structures for HEALPix indices."""
        # Sort pixels by nside and healpix_index for efficient searching.
        # lexsort (last key is primary) is equivalent to
        # np.argsort(order=["nside", "healpix_index"]) but avoids the slow
        # structured-dtype record comparison (~85x faster on the full map);
        # (nside, healpix_index) pairs are unique, so tie-breaking cannot
        # change the result.
        sort_idx = np.lexsort(
            (self._pixel_info["healpix_index"], self._pixel_info["nside"])
        )

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

        # In NESTED ordering the pixel index at a coarser nside is the
        # finest-level index right-shifted by 2 bits per resolution level,
        # so a single ang2pix pass at the finest nside serves every level.
        # The -1 sentinel for invalid coordinates survives the arithmetic
        # shift (-1 >> k == -1).
        nside_max = int(self._nside_levels[-1])
        order_max = nside_max.bit_length() - 1
        ipix_max = lb2pix(nside_max, l_arr, b_arr, nest=True)

        # Search at each nside level (coarse to fine resolution)
        for k, nside in enumerate(self._nside_levels):
            # Derive HEALPix pixel indices at this resolution
            ipix = ipix_max >> (2 * (order_max - (int(nside).bit_length() - 1)))

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
                safe_idx = np.clip(idx, 0, None)
                match_idx = in_bounds & (self._hp_idx_sorted[k][safe_idx] == ipix)

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

    def query(self, coords, apply_reliability_mask=None):
        """
        Query extinction at the specified coordinates.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord or array_like
            Coordinates to query. Can be a (scalar or array) SkyCoord, a
            single ``[l, b]`` pair in degrees, or an ``(Ncoords, 2)``
            array of ``[l, b]`` pairs in degrees.
        apply_reliability_mask : bool, optional
            Whether to set extinction values flagged as unreliable by the
            map (distance bins outside a pixel's reliable range, and
            non-converged pixels) to NaN. If None (default), uses the
            value set at construction (which defaults to True).

        Returns
        -------
        distances : ndarray
            Distance grid points (kpc).
        av_mean : ndarray
            Mean A(V) extinction values along each line of sight. Shape is
            ``(n_dist,)`` for a scalar SkyCoord or a single ``[l, b]``
            pair, and ``(Ncoords, n_dist)`` for array input.
        av_std : ndarray
            Standard deviation of A(V) extinction values (same shape as
            ``av_mean``).

        Notes
        -----
        For coordinates outside the map coverage, NaN values are returned.
        With reliability masking active (the default), NaN is also
        returned for distance bins outside a pixel's reliable
        distance-modulus range and for pixels whose fits did not converge;
        `~brutus.priors.extinction.logp_extinction` treats NaN as "no
        coverage" and falls back to a uniform prior there.
        """
        if apply_reliability_mask is None:
            apply_reliability_mask = self._apply_reliability_mask

        single_pair = False
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
            single_pair = np.ndim(coords) == 1
            coords_arr = np.atleast_2d(coords)
            l_deg = coords_arr[:, 0]
            b_deg = coords_arr[:, 1]

        # Find corresponding data indices
        pix_idx = self._find_data_idx(l_deg, b_deg)

        # Promote 0-d (scalar SkyCoord) indices to 1-d so the advanced
        # indexing below always yields a fresh array (a 0-d index would
        # return a mutable view into the loaded map).
        pix_idx = np.atleast_1d(np.asarray(pix_idx))

        # Extract extinction data
        in_bounds = pix_idx != -1
        safe_idx = np.clip(pix_idx, 0, None)
        av_mean = self._av_mean[safe_idx]
        av_std = self._av_std[safe_idx]

        # Set out-of-bounds values to NaN
        av_mean[~in_bounds] = np.nan
        av_std[~in_bounds] = np.nan

        # Mask distance bins the map flags as unreliable: outside the
        # per-pixel reliable distance range or in non-converged pixels.
        if apply_reliability_mask and self._converged is not None:
            dists = self._distances
            unreliable = (
                (dists < self._d_reliable_min[safe_idx][..., None])
                | (dists > self._d_reliable_max[safe_idx][..., None])
                | ~self._converged[safe_idx][..., None]
            )
            av_mean[unreliable] = np.nan
            av_std[unreliable] = np.nan

        # Squeeze the leading axis for scalar SkyCoord input or a single
        # [l, b] pair so both forms return (n_dist,) profiles.
        scalar_input = single_pair or (
            (hasattr(coords, "isscalar") and coords.isscalar)
            or (not hasattr(coords, "__len__") and np.isscalar(l_deg))
        )

        if scalar_input and av_mean.shape[0] == 1:
            av_mean = av_mean[0]
            av_std = av_std[0]

        # Return a copy of the distance grid: handing out the live
        # internal array would let caller mutation corrupt later queries.
        return self._distances.copy(), av_mean, av_std
