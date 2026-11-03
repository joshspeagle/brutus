"""
Base class and Bayestar implementation for querying 3D dust extinction maps.
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

        # Pass (l, b) arrays directly to query, avoiding SkyCoord overhead.
        # A scalar pair gives a (2,) array, so query returns (n_dist,).
        coords = np.stack(np.broadcast_arrays(ell, b), axis=-1).astype(float)

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
            Distance from the Solar System in kpc. Accepted for API
            compatibility because the query always returns the full distance
            profile.
        frame : str, optional
            Coordinate frame (``'icrs'``, ``'fk4'``, ``'fk5'``, or
            ``'fk4noeterms'``). Default is ``'icrs'``.
        **kwargs
            Additional keyword arguments passed to :meth:`query`.

        Returns
        -------
        Query results as implemented by subclasses.

        Raises
        ------
        ValueError
            If ``frame`` is not one of the supported coordinate frames.
        """
        valid_frames = ["icrs", "fk4", "fk5", "fk4noeterms"]

        if frame not in valid_frames:
            raise ValueError(
                f"Frame '{frame}' not supported. Must be one of {valid_frames}."
            )

        if not isinstance(ra, units.Quantity):
            ra = ra * units.deg
        if not isinstance(dec, units.Quantity):
            dec = dec * units.deg

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
            # Convert distance modulus to distance in kpc via d = 10**(DM/5 - 2).
            # A DM bound of -inf (min) or +inf (max) means no cut on that side
            # and maps to 0 or +inf. A NaN bound means the pixel has no
            # determined reliable range and must mask the entire profile.
            # Because comparisons against NaN evaluate to False and would leave
            # the pixel unmasked, NaN minimums are mapped to +inf and NaN
            # maximums to -inf so every distance bin fails the range test.
            with np.errstate(over="ignore"):
                d_min = 10.0 ** (
                    self._pixel_info["DM_reliable_min"].astype(np.float64) / 5.0 - 2.0
                )
                d_max = 10.0 ** (
                    self._pixel_info["DM_reliable_max"].astype(np.float64) / 5.0 - 2.0
                )
            self._d_reliable_min = np.where(np.isnan(d_min), np.inf, d_min)
            self._d_reliable_max = np.where(np.isnan(d_max), -np.inf, d_max)
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
            Data indices for each coordinate, at least 1-D. Invalid
            coordinates and coordinates outside the map return -1.
        """
        l_arr = np.atleast_1d(gal_l)
        b_arr = np.atleast_1d(b)
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

            # Find each pixel in the sorted list; a pixel absent from the map
            # (or the -1 sentinel) matches no entry.
            hp_sorted = self._hp_idx_sorted[k]
            idx = np.minimum(np.searchsorted(hp_sorted, ipix), len(hp_sorted) - 1)
            match = hp_sorted[idx] == ipix
            pix_idx[match] = self._data_idx[k][idx[match]]

        return pix_idx

    def query(self, coords, apply_reliability_mask=None):
        """
        Query extinction at the specified coordinates.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord or array_like
            Coordinates to query as a scalar or array ``SkyCoord``, a single
            ``[l, b]`` pair in degrees of shape ``(2,)``, or an array of
            ``[l, b]`` pairs in degrees of shape ``(Ncoords, 2)``. Any distance
            attached to ``coords`` is ignored because the full distance profile
            is returned.
        apply_reliability_mask : bool, optional
            Whether to set extinction values in unreliable distance bins and
            non-converged pixels to NaN. If None (default), uses the value set
            at construction (default True).

        Returns
        -------
        distances : numpy.ndarray
            Distance grid points in kpc of shape ``(n_dist,)``.
        av_mean : numpy.ndarray
            Mean :math:`A(V)` extinction in magnitudes along each sightline,
            with shape ``(n_dist,)`` for a single sightline or
            ``(Ncoords, n_dist)`` for array input. Preserves the storage dtype
            of the file (``float16`` for ``bayestar2019_v1.h5``), so cast to
            ``float64`` before arithmetic requiring more than 3 significant
            digits.
        av_std : numpy.ndarray
            Standard deviation of :math:`A(V)` in magnitudes, with the same
            shape and dtype as ``av_mean``.

        Raises
        ------
        ValueError
            If an array ``coords`` does not have shape ``(2,)`` or
            ``(Ncoords, 2)``. A ``(2, 2)`` array is interpreted as two
            ``[l, b]`` pairs; pass separate longitude and latitude arrays to
            :meth:`query_gal`.

        Notes
        -----
        Coordinates outside the map footprint return NaN. When reliability
        masking is active (the default), distance bins outside a pixel's
        reliable distance-modulus range and pixels whose fits did not converge
        also return NaN. :func:`~brutus.priors.extinction.logp_extinction`
        treats NaN bins as uninformative and falls back to the normalized
        uniform prior over ``avlim``.
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
            coords_arr = np.asarray(coords, dtype=float)
            if coords_arr.ndim not in (1, 2) or coords_arr.shape[-1] != 2:
                raise ValueError(
                    "coords must be a SkyCoord, an [l, b] pair or an "
                    f"(Ncoords, 2) array of [l, b]; got shape {coords_arr.shape}."
                ) from None
            single_pair = coords_arr.ndim == 1
            coords_arr = np.atleast_2d(coords_arr)
            l_deg = coords_arr[:, 0]
            b_deg = coords_arr[:, 1]

        # The indices are at least 1-D, so the indexing below copies the map.
        pix_idx = self._find_data_idx(l_deg, b_deg)

        in_bounds = pix_idx != -1
        safe_idx = np.clip(pix_idx, 0, None)
        av_mean = self._av_mean[safe_idx]
        av_std = self._av_std[safe_idx]

        av_mean[~in_bounds] = np.nan
        av_std[~in_bounds] = np.nan

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

        # Return a copy of the distance grid because mutating the internal
        # array would corrupt subsequent queries.
        return self._distances.copy(), av_mean, av_std
