#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Individual stellar modeling and synthetic photometry generation.

This module provides classes for modeling individual stars using MIST
(MESA Isochrones and Stellar Tracks) evolutionary tracks and generating
synthetic photometry with neural network-based bolometric corrections.

The module follows a clean separation of concerns:
- EEPTracks: Stellar parameter predictions for individual stars
- StarEvolTrack: SED/photometry generation for individual stars

This design mirrors the stellar population modeling pattern:
- Isochrone: Stellar parameter predictions for populations
- StellarPop: SED/photometry generation for populations

Examples
--------
Basic individual star modeling:

>>> from brutus.core.individual import EEPTracks, StarEvolTrack
>>>
>>> # Create tracks for parameter predictions
>>> tracks = EEPTracks()
>>> params = tracks.get_predictions([1.0, 350, 0.0, 0.0])  # mini, eep, feh, afe
>>>
>>> # Create star track for photometry
>>> star_track = StarEvolTrack(tracks=tracks)
>>> sed, params, params2 = star_track.get_seds(
>>>     mini=1.0, eep=350, feh=0.0, afe=0.0,
>>>     av=0.1, dist=1000.0
>>> )

Advanced usage with binary stars:

>>> # Model binary system with mass ratio 0.7
>>> sed, params, params2 = star_track.get_seds(
>>>     mini=1.2, eep=400, feh=-0.2, afe=0.0,
>>>     smf=0.7, av=0.15, dist=1500.0
>>> )

Notes
-----
This design provides several advantages:

1. **Separation of Concerns**: Parameter prediction vs. photometry synthesis
2. **Consistency**: Mirrors the stellar population modeling pattern
3. **Flexibility**: Can use different SED generators with same tracks
4. **Generality**: EEPTracks name allows for future non-MIST implementations
5. **Maintainability**: Cleaner, more focused class responsibilities

The StarEvolTrack class uses dependency injection, accepting an EEPTracks
instance rather than inheriting from it. This makes the code more modular
and allows for different track implementations.

This implementation is based on the MIST stellar evolution framework
(Choi et al. 2016; Dotter 2016).

References
----------
- Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
  for the Construction of Stellar Isochrones", ApJ, 823, 102
- Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
  Models", ApJS, 222, 8
"""

import os
import pickle
import sys
import uuid
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from ..data.download import _find_data_file

# Import filter definitions
from ..data.filters import FILTERS

# Import utilities
from ..utils.photometry import add_mag

# Import neural network predictor
from .neural_nets import FastNNPredictor

__all__ = ["EEPTracks", "StarEvolTrack", "StarGrid"]


# EEPTracks pickle-cache format version. Bump whenever the set of cached
# attributes (or their semantics) changes so that stale caches written by
# older versions are ignored rather than mis-loaded.
_EEPTRACKS_CACHE_VERSION = 3

# Cached attributes. The interpolator and its padded axes are rebuilt from
# `gridpoints` and the unpadded `ygrid` on load: pickling them would store a
# padded single-afe grid at twice the size.
_EEPTRACKS_CACHE_ATTRS = (
    "labels",
    "predictions",
    "ndim",
    "npred",
    "mini_idx",
    "eep_idx",
    "feh_idx",
    "logt_idx",
    "logl_idx",
    "logg_idx",
    "_ageidx",
    "mistfile",
    "gridpoints",
    "mini_bound",
    "ygrid",
)


# Rename parameters from MIST HDF5 file for easier use as keyword arguments
rename = {
    "mini": "initial_mass",  # input parameters
    "eep": "EEP",
    "feh": "initial_[Fe/H]",
    "afe": "initial_[a/Fe]",
    "mass": "star_mass",  # outputs
    "feh_surf": "[Fe/H]",
    "afe_surf": "[a/Fe]",
    "loga": "log_age",
    "logt": "log_Teff",
    "logg": "log_g",
    "logl": "log_L",
    "logr": "log_R",
}


def _mass_corrections(mini, eep, feh, corr_params=None):
    """
    Return the empirical (dlogt, dlogr) corrections of low-mass stars.

    `EEPTracks.get_corrections` documents the formula. `corr_params` is
    (dtdm, drdm, msto_smooth, feh_scale), by default (0.09, -0.09, 30.0, 0.5).
    """
    if corr_params is None:
        corr_params = (0.09, -0.09, 30.0, 0.5)
    if len(corr_params) != 4:
        raise ValueError(
            "corr_params must be (dtdm, drdm, msto_smooth, feh_scale); "
            f"got {corr_params}"
        )
    dtdm, drdm, msto_smooth, feh_scale = corr_params

    # Clip the log arguments away from zero for extreme masses
    mass_offset = mini - 1.0
    eps = 1e-10
    dlogt = np.log10(np.maximum(1.0 + mass_offset * dtdm, eps))
    dlogr = np.log10(np.maximum(1.0 + mass_offset * drdm, eps))

    # EEP and metallicity dependence
    ecorr = 1.0 - 1.0 / (1.0 + np.exp(-(eep - 454.0) / msto_smooth))
    fcorr = np.exp(feh_scale * feh)
    dlogt = dlogt * (ecorr * fcorr)
    dlogr = dlogr * (ecorr * fcorr)

    # No corrections at solar mass and above
    solar = mini >= 1.0
    return np.where(solar, 0.0, dlogt), np.where(solar, 0.0, dlogr)


class EEPTracks:
    """
    Stellar parameter predictions for individual stars using evolutionary tracks.

    This class provides interpolation of stellar parameters along evolutionary
    tracks as a function of initial mass, equivalent evolutionary point (EEP),
    metallicity, and alpha enhancement. It focuses solely on stellar parameter
    prediction without photometry generation.

    The class name "EEPTracks" is intentionally general to allow for future
    implementations beyond MIST, while the current implementation uses MIST
    (MESA Isochrones and Stellar Tracks) evolutionary models.

    For photometry generation, use StarEvolTrack with this class.

    Parameters
    ----------
    mistfile : str, optional
        Path to the HDF5 file containing the evolutionary tracks. If not
        provided, `MIST_1.2_EEPtrk.h5` is looked up in the brutus download
        directory (the Pooch cache, or ``$ASTRO_BRUTUS_DATA_DIR`` when set),
        then in ``data/DATAFILES`` in the repository.

    predictions : iterable of str, optional
        The names of stellar parameters to predict, from `'mass'`,
        `'feh_surf'`, `'afe_surf'`, `'loga'`, `'logt'`, `'logg'`, `'logl'`
        and `'logr'`. It must include `'loga'`, `'logl'`, `'logt'` and
        `'logg'`. Default is:
        `["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"]`.

    ageweight : bool, optional
        Whether to compute age weights d(age)/d(EEP) for age priors.
        Default is `True`.

    verbose : bool, optional
        Whether to output progress messages during initialization. Default is `True`.

    use_cache : bool, optional
        Whether to use pickle caching to speed up loading. If True, will save
        processed EEPTracks to a .pkl file for faster subsequent loads, and
        load from cache if available and newer than the original file.
        Default is `True`.

    Raises
    ------
    ValueError
        If `predictions` has an unknown name or lacks a required one.
    FileNotFoundError
        If the track file does not exist. `brutus.data.fetch_tracks`
        downloads it.

    Attributes
    ----------
    labels : list of str
        Input parameter names: ['mini', 'eep', 'feh', 'afe']

    predictions : list of str
        Output column names of `get_predictions`: the requested
        predictions, then ``"agewt"`` when ``ageweight=True``.

    ndim, npred : int
        Number of input dimensions and of requested predictions. `npred`
        does not count the appended ``"agewt"`` column.

    interpolator : scipy.interpolate.RegularGridInterpolator
        The main interpolation object for stellar parameter prediction

    gridpoints : dict
        Unique grid points for each input parameter

    Examples
    --------
    Predict parameters for a solar-mass star on the main sequence:

    >>> tracks = EEPTracks()
    >>> params = tracks.get_predictions([1.0, 350, 0.0, 0.0])  # mini, eep, feh, afe
    >>> log_age = params[0]  # log10(age in years)
    >>> log_teff = params[2]  # log10(effective temperature)
    >>> print(f"Age: {10**log_age:.1e} yr, Teff: {10**log_teff:.0f} K")

    Batch prediction for multiple stars:

    >>> import numpy as np
    >>> labels = np.array([[0.8, 350, -0.5, 0.2],   # Metal-poor dwarf
    ...                    [1.2, 454, 0.0, 0.0],    # Solar at turnoff
    ...                    [2.0, 500, 0.3, 0.0]])   # Massive metal-rich
    >>> preds = tracks.get_predictions(labels)
    >>> ages = 10**preds[:, 0]  # Convert to linear ages
    """

    def __init__(
        self,
        mistfile=None,
        predictions=["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"],
        ageweight=True,
        verbose=True,
        use_cache=True,
    ):

        # Define input parameter labels
        labels = ["mini", "eep", "feh", "afe"]
        known = [name for name in rename if name not in labels]
        unknown = [p for p in predictions if p not in known]
        missing = [p for p in ("loga", "logl", "logt", "logg") if p not in predictions]
        if unknown or missing:
            raise ValueError(
                f"predictions must be names from {known} that include 'loga', "
                f"'logl', 'logt' and 'logg'; got {list(predictions)}."
            )

        # Initialize values
        self.labels = list(np.array(labels))
        self.predictions = list(np.array(predictions))
        self.ndim, self.npred = len(self.labels), len(self.predictions)

        # Set label references for fast indexing
        self.mini_idx = np.where(np.array(self.labels) == "mini")[0][0]
        self.eep_idx = np.where(np.array(self.labels) == "eep")[0][0]
        self.feh_idx = np.where(np.array(self.labels) == "feh")[0][0]
        self.logt_idx = np.where(np.array(self.predictions) == "logt")[0][0]
        self.logl_idx = np.where(np.array(self.predictions) == "logl")[0][0]
        self.logg_idx = np.where(np.array(self.predictions) == "logg")[0][0]

        if mistfile is None:
            mistfile = _find_data_file(("MIST_1.2_EEPtrk.h5",), "fetch_tracks")
        self.mistfile = Path(mistfile)
        if not self.mistfile.is_file():
            raise FileNotFoundError(
                f"mistfile '{self.mistfile}' does not exist. Run "
                "brutus.data.fetch_tracks() to download the MIST tracks."
            )

        # Generate cache file path based on original file and configuration.
        # The cache format version is part of the filename so caches written
        # by incompatible older versions are ignored, not mis-loaded.
        cache_key = (
            f"{self.mistfile.stem}_ageweight{ageweight}_pred{''.join(predictions)}"
            f"_cachev{_EEPTRACKS_CACHE_VERSION}"
        )
        cache_file = self.mistfile.parent / f"{cache_key}.pkl"

        # Try to load from cache first
        if use_cache and cache_file.exists():
            state = dict(vars(self))
            try:
                # Check if cache is newer than original file
                cache_mtime = cache_file.stat().st_mtime
                orig_mtime = self.mistfile.stat().st_mtime

                if cache_mtime > orig_mtime:
                    if verbose:
                        sys.stderr.write(
                            f"Loading cached EEPTracks from {cache_file}...\n"
                        )

                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Restore all cached attributes
                    for attr, value in cached_data.items():
                        setattr(self, attr, value)
                    self._make_interpolator()

                    if verbose:
                        sys.stderr.write("Cached EEPTracks loaded successfully!\n")
                    return
                else:
                    if verbose:
                        sys.stderr.write(
                            "Cache is older than data file, regenerating...\n"
                        )
            except Exception as e:
                # A cache that cannot be loaded (unreadable, truncated,
                # corrupt, or written by another numpy) is rebuilt from the
                # data file, which gives the same result. A cache that loaded
                # but held bad values must not leave them behind.
                vars(self).clear()
                vars(self).update(state)
                if verbose:
                    sys.stderr.write(
                        f"Cache loading failed ({e}), loading from original file...\n"
                    )

        # Load from original file
        if verbose:
            sys.stderr.write(f"Loading evolutionary tracks from {mistfile}...\n")

        # Load and process track data
        with h5py.File(self.mistfile, "r") as misth5:
            self._make_lib(misth5, verbose=verbose)
        self._lib_as_grid()

        # Construct age weights if requested
        self._ageidx = self.predictions.index("loga")
        if ageweight:
            self._add_age_weights(verbose=verbose)

        # Build interpolation grid
        self._build_interpolator()

        # Free construction-only intermediates: nothing reads them after
        # the interpolator exists, and retaining them roughly doubles the
        # memory footprint.
        for attr in ("libparams", "output", "X"):
            if hasattr(self, attr):
                delattr(self, attr)

        # Save to cache if enabled
        if use_cache:
            try:
                # Cache only the runtime attribute whitelist
                cache_data = {
                    attr: getattr(self, attr)
                    for attr in _EEPTRACKS_CACHE_ATTRS
                    if hasattr(self, attr)
                }

                # Write a temporary file and rename it, so a failed write
                # leaves no partial cache for the next load to rebuild. A
                # random name keeps concurrent writers apart, also on other
                # nodes of a shared file system.
                tmp_file = cache_file.with_name(
                    f"{cache_file.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with open(tmp_file, "wb") as f:
                        pickle.dump(cache_data, f)
                    os.replace(tmp_file, cache_file)
                finally:
                    tmp_file.unlink(missing_ok=True)

                if verbose:
                    sys.stderr.write(f"EEPTracks cached to {cache_file}\n")

            except OSError as e:
                # A read-only data directory only costs a rebuild next time
                if verbose:
                    sys.stderr.write(f"Warning: Failed to cache EEPTracks ({e})\n")

        if verbose:
            sys.stderr.write("done!\n")

    def _make_lib(self, misth5, verbose=True):
        """
        Convert HDF5 input to numpy arrays for labels and outputs.

        Reads evolutionary track data from MIST HDF5 file and organizes it
        into structured arrays for input parameters (mini, eep, feh, afe) and
        predicted outputs (loga, logl, logt, logg, etc.).

        Parameters
        ----------
        misth5 : h5py.File
            Open HDF5 file containing MIST evolutionary track data.
        verbose : bool, optional
            Whether to print progress messages. Default is True.

        Notes
        -----
        This method handles the case where alpha enhancement data ([α/Fe])
        is not available in the file by setting it to zero.

        Each track dataset is read from disk exactly once; label and output
        fields are then sliced from the in-memory record array. Compound HDF5
        datasets are stored row-wise, so per-field reads would re-scan (and
        re-decompress) every record once per column.
        """
        if verbose:
            sys.stderr.write("  Constructing track library...\n")

        # Input parameter (mini, eep, feh, afe) and output columns
        label_cols = [rename[p] for p in self.labels]
        cols = [rename[p] for p in self.predictions]

        # Handle alpha enhancement availability
        afe_col = rename["afe_surf"]
        afe_available = True
        afe_surf_idx = None

        try:
            # Test alpha enhancement column availability
            first_z = list(misth5["index"])[0]
            _ = misth5[first_z].dtype[afe_col]
        except (KeyError, ValueError):
            afe_available = False
            for i, pred in enumerate(self.predictions):
                if pred == "afe_surf":
                    afe_surf_idx = i
                    break
            if verbose:
                sys.stderr.write("    [alpha/Fe] column not found, will set to zero\n")

        cols_to_read = []
        read_to_pred_mapping = []

        for pred_idx, col in enumerate(cols):
            if not afe_available and col == afe_col:
                continue
            else:
                cols_to_read.append(col)
                read_to_pred_mapping.append(pred_idx)

        if verbose:
            sys.stderr.write(f"    Reading {len(cols_to_read)} parameter columns\n")

        # Single pass over the file: read each track group once, then copy
        # the needed fields out of the in-memory record array.
        label_dtype = None
        label_chunks = []
        output_chunks = []
        for z in misth5["index"]:
            arr = np.asarray(misth5[z])
            if label_dtype is None:
                label_dtype = np.dtype(
                    [(lbl, arr.dtype[col]) for lbl, col in zip(self.labels, label_cols)]
                )
            lp = np.empty(len(arr), dtype=label_dtype)
            for lbl, col in zip(self.labels, label_cols):
                lp[lbl] = arr[col]
            label_chunks.append(lp)

            out = np.empty((len(arr), len(cols_to_read)), dtype="f8")
            for j, col in enumerate(cols_to_read):
                out[:, j] = arr[col]
            output_chunks.append(out)

        self.libparams = np.concatenate(label_chunks)
        output_data = np.concatenate(output_chunks)

        # Create and fill output array
        self.output = np.empty((len(output_data), len(self.predictions)), dtype="f8")

        for read_idx, pred_idx in enumerate(read_to_pred_mapping):
            self.output[:, pred_idx] = output_data[:, read_idx]

        # Handle missing alpha enhancement
        if not afe_available and afe_surf_idx is not None:
            self.output[:, afe_surf_idx] = 0.0

    def _lib_as_grid(self):
        """
        Convert library parameters to pixel indices for interpolation.

        Determines the unique grid points in each parameter dimension
        (mini, eep, feh, afe) and creates mappings from continuous parameter
        values to discrete grid indices for efficient interpolation.

        Notes
        -----
        This method populates the following attributes:
        - gridpoints: dict of unique values for each parameter
        - X: array of grid indices for each library point
        - mini_bound: minimum initial mass in the grid
        """
        # Get unique grid points in each dimension
        self.gridpoints = {}
        for p in self.labels:
            self.gridpoints[p] = np.unique(self.libparams[p])

        # Digitize library parameters to grid indices
        X = np.array(
            [
                np.digitize(self.libparams[p], bins=self.gridpoints[p], right=True)
                for p in self.labels
            ]
        )
        self.X = X.T

        # Store minimum mass bound
        self.mini_bound = self.gridpoints["mini"].min()

    def _add_age_weights(self, verbose=True):
        """
        Compute age gradient d(age)/d(EEP) for age priors.

        Calculates the derivative of age with respect to evolutionary point
        (EEP) for each track. This is used to properly weight age priors when
        sampling in EEP space, accounting for the non-uniform mapping between
        EEP and age.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.

        Notes
        -----
        The age weights are appended to the predictions array and "agewt"
        is added to the predictions list. A track (fixed mini, feh, afe)
        with one point gets weight 0.

        The gradient is computed as d(age)/d(EEP) where age is in linear
        (not logarithmic) units, even though ages are stored as log(age).
        The EEP values are passed to ``np.gradient`` as the coordinate array
        (after sorting each track by EEP), so the Jacobian remains correct
        for track libraries with non-unit or non-uniform EEP sampling. For
        unit-spaced tracks (e.g. the shipped MIST file) this is identical to
        the index-spacing gradient.
        """
        if verbose:
            sys.stderr.write("  Computing age weights...\n")

        lib = self.libparams
        eeps = lib["eep"].astype(float)
        # Sort by track, then by EEP; lexsort is stable
        order = np.lexsort((eeps, lib["afe"], lib["feh"], lib["mini"]))
        keys = np.column_stack([lib[name][order] for name in ("mini", "feh", "afe")])
        new_track = np.any(keys[1:] != keys[:-1], axis=1)
        bounds = np.concatenate(([0], np.flatnonzero(new_track) + 1, [len(order)]))

        linear_ages = 10 ** self.output[order, self._ageidx]
        ageweights = np.zeros(len(lib))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop - start > 1:
                track = order[start:stop]
                ageweights[track] = np.gradient(linear_ages[start:stop], eeps[track])

        # Append to outputs
        self.output = np.hstack([self.output, ageweights[:, None]])
        self.predictions += ["agewt"]

    def _build_interpolator(self):
        """
        Fill the output grid `ygrid` from the library and build the interpolator.

        `ygrid` has shape (Nmini, Neep, Nfeh, Nafe, Npred + agewt), with NaN
        where the library has no model.
        """
        grid_dims = [len(self.gridpoints[p]) for p in self.labels]
        self.ygrid = np.full(grid_dims + [self.output.shape[-1]], np.nan)
        if len(self.X) > 0:
            self.ygrid[tuple(self.X.T)] = self.output
        self._make_interpolator()

    def _make_interpolator(self):
        """
        Build the RegularGridInterpolator for fast predictions.

        Creates a scipy RegularGridInterpolator over (mini, eep, feh, afe)
        that predicts all stellar parameters at once, with linear
        interpolation and NaN outside the grid. It sets `xgrid` and
        `grid_dims`, the axes and shape of the interpolated grid.

        Notes
        -----
        A single afe value is padded to two nodes, 1e-5 on either side,
        that share its values. The padding is a broadcast view of `ygrid`,
        not a copy.
        """
        xgrid = [self.gridpoints[lbl] for lbl in self.labels]
        ygrid = self.ygrid
        if len(xgrid[-1]) == 1:
            afe_val = xgrid[-1][0]
            xgrid[-1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            shape = list(ygrid.shape)
            shape[-2] = 2
            ygrid = np.broadcast_to(ygrid, shape)
        self.xgrid = tuple(xgrid)
        self.grid_dims = np.array(ygrid.shape)

        self.interpolator = RegularGridInterpolator(
            self.xgrid,
            ygrid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def get_predictions(self, labels, apply_corr=True, corr_params=None):
        """
        Generate stellar parameter predictions for given input parameters.

        Parameters
        ----------
        labels : array-like of shape (4,) or (Nobj, 4)
            Input parameters [mini, eep, feh, afe] where:
            - mini: Initial mass in solar masses
            - eep: Equivalent evolutionary point
            - feh: Metallicity [Fe/H] in logarithmic solar units
            - afe: Alpha enhancement [alpha/Fe] in logarithmic solar units

        apply_corr : bool, optional
            Whether to apply empirical corrections. Default is True.

        corr_params : tuple, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale).

        Returns
        -------
        preds : numpy.ndarray of shape (Npred,) or (Nobj, Npred)
            Predicted stellar parameters in the order specified by
            `self.predictions` attribute.

        See Also
        --------
        get_corrections : Computes empirical corrections applied when apply_corr=True
        StarEvolTrack.get_seds : Uses these predictions to generate photometry

        Notes
        -----
        When ``apply_corr=True`` the empirical corrections are applied as
        ``logt += dlogt``, ``logl += 2*dlogr``, ``logg -= 2*dlogr``. This is
        an effective map calibrated as-is; see `get_corrections` for why the
        luminosity shift intentionally omits the Stefan-Boltzmann ``4*dlogt``
        term.

        Examples
        --------
        Single star prediction:

        >>> tracks = EEPTracks()
        >>> params = tracks.get_predictions([1.0, 350, 0.0, 0.0])
        >>> log_age, log_L, log_Teff, log_g = params[:4]

        Multiple star prediction:

        >>> import numpy as np
        >>> labels = np.array([[0.8, 350, -0.5, 0.2], [1.2, 454, 0.0, 0.0]])
        >>> params = tracks.get_predictions(labels)
        """
        labels = np.array(labels)
        ndim = labels.ndim

        # Perform interpolation
        if ndim == 1:
            preds = self.interpolator(labels)[0]
        elif ndim == 2:
            preds = self.interpolator(labels)
        else:
            raise ValueError("Input `labels` must be 1-D or 2-D array.")

        # Apply empirical corrections if requested
        if apply_corr:
            corrs = self.get_corrections(labels, corr_params=corr_params)
            # NOTE: logl gets +2*dlogr only (no +4*dlogt): an effective map
            # calibrated as-is -- see get_corrections Notes before changing.
            if ndim == 1:
                dlogt, dlogr = corrs
                preds[self.logt_idx] += dlogt
                preds[self.logl_idx] += 2.0 * dlogr  # L ∝ R^2 (effective)
                preds[self.logg_idx] -= 2.0 * dlogr  # g ∝ M/R^2
            elif ndim == 2:
                dlogt, dlogr = corrs.T
                preds[:, self.logt_idx] += dlogt
                preds[:, self.logl_idx] += 2.0 * dlogr
                preds[:, self.logg_idx] -= 2.0 * dlogr

        return preds

    def get_corrections(self, labels, corr_params=None):
        r"""
        Compute empirical corrections to stellar parameters.

        Applies empirical corrections to effective temperature and radius
        based on stellar mass, evolutionary phase (EEP), and metallicity.
        These corrections account for systematic offsets between MIST models
        and observations, particularly for low-mass stars.

        Parameters
        ----------
        labels : array-like of shape (4,) or (Nobj, 4)
            Input parameters [mini, eep, feh, afe] where:
            - mini: Initial mass in solar masses
            - eep: Equivalent evolutionary point
            - feh: Metallicity [Fe/H]
            - afe: Alpha enhancement [α/Fe]

        corr_params : tuple of float, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale) where:
            - dtdm: Temperature correction slope with mass
            - drdm: Radius correction slope with mass
            - msto_smooth: Smoothing scale for main sequence turnoff transition
            - feh_scale: Metallicity scaling factor
            Default is (0.09, -0.09, 30.0, 0.5).

        Returns
        -------
        corrs : numpy.ndarray of shape (2,) or (Nobj, 2)
            Corrections to [log(Teff), log(R)]. These are added to the
            base predictions from the interpolator.

        Raises
        ------
        ValueError
            If `labels` is not 1-D or 2-D, or `corr_params` does not have 4
            entries.

        See Also
        --------
        get_predictions : Applies these corrections to stellar parameters

        Notes
        -----
        Corrections are applied as:

        .. math::
            \Delta \log T_{\rm eff} = f_{\rm EEP} \cdot f_{\rm [Fe/H]} \cdot \log(1 + \Delta M \cdot \alpha_T)
            \Delta \log R = f_{\rm EEP} \cdot f_{\rm [Fe/H]} \cdot \log(1 + \Delta M \cdot \alpha_R)

        where :math:`\Delta M = M_{\rm ini} - 1.0` and the EEP factor smoothly
        transitions from 1 (main sequence) to 0 (post-turnoff): the empirical
        low-mass corrections apply on the main sequence and are switched off
        after the main-sequence turnoff (EEP 454).

        Corrections are set to zero for stars with :math:`M_{\rm ini} \geq 1.0 M_\odot`.

        The corrections are applied by `get_predictions` as an *effective*,
        calibrated-as-is map: the temperature shift is ``dlogt``, while the
        luminosity/gravity shifts are ``+2*dlogr``/``-2*dlogr``. The
        luminosity shift deliberately omits the ``4*dlogt`` term that strict
        Stefan-Boltzmann consistency (:math:`L = 4\pi R^2 \sigma T^4`) would
        require, so the corrected (logl, logt, logg) do not describe a
        perfectly self-consistent star. The default `corr_params` were
        calibrated against data in exactly this form (inherited from the
        original brutus v0.8 implementation), so do not "fix" the propagation
        without recalibrating them.
        """
        labels = np.array(labels)
        ndim = labels.ndim

        # Extract parameters
        if ndim == 1:
            mini = labels[self.mini_idx]
            eep = labels[self.eep_idx]
            feh = labels[self.feh_idx]
        elif ndim == 2:
            mini = labels[:, self.mini_idx]
            eep = labels[:, self.eep_idx]
            feh = labels[:, self.feh_idx]
        else:
            raise ValueError("Input `labels` must be 1-D or 2-D array.")

        return np.stack(_mass_corrections(mini, eep, feh, corr_params), axis=-1)


class StarEvolTrack:
    """
    Synthetic photometry generation for individual stars.

    This class generates synthetic SEDs and photometry for individual stars
    using neural network-based bolometric corrections. It provides modeling
    of binary stars, dust extinction, and observational effects.

    This class mirrors StellarPop but for individual stars:
    - StarEvolTrack: Individual star photometry
    - StellarPop: Stellar population photometry

    The class uses dependency injection, accepting an EEPTracks instance for
    stellar parameter predictions rather than inheriting from it.

    Parameters
    ----------
    tracks : EEPTracks
        EEPTracks instance for stellar parameter predictions. Its
        `predictions` must include loga, logl, logt, logg, feh_surf and
        afe_surf.

    filters : list of str, optional
        Filter names for photometry computation. If None, uses all available.

    nnfile : str, optional
        Path to neural network file for bolometric corrections. If None,
        `FastNNPredictor` finds the default file.

    verbose : bool, optional
        Whether to output progress messages. Default is True.

    Raises
    ------
    ValueError
        If `tracks.predictions` lacks a label listed above, or if a filter
        is not in the neural network file.

    Attributes
    ----------
    tracks : EEPTracks
        The evolutionary track model for stellar parameter predictions

    filters : numpy.ndarray
        Array of filter names

    predictor : FastNNPredictor
        Neural network predictor for photometry

    See Also
    --------
    EEPTracks : Stellar parameter predictions used by this class
    StarGrid : Alternative grid-based approach for photometry
    brutus.core.neural_nets.FastNNPredictor : Neural network used for SEDs

    Examples
    --------
    Basic individual star photometry:

    >>> tracks = EEPTracks()
    >>> star_track = StarEvolTrack(tracks=tracks)
    >>>
    >>> # Generate SED for a solar-mass main sequence star
    >>> sed, params, params2 = star_track.get_seds(
    ...     mini=1.0, eep=350, feh=0.0, afe=0.0,
    ...     av=0.1, rv=3.1, dist=1000.0
    ... )

    Binary star modeling:

    >>> # Model binary with 70% mass ratio secondary
    >>> sed, params, params2 = star_track.get_seds(
    ...     mini=1.2, eep=400, feh=-0.2, afe=0.0,
    ...     smf=0.7, av=0.15, dist=1500.0
    ... )
    """

    def __init__(self, tracks, filters=None, nnfile=None, verbose=True):

        needed = ("loga", "logl", "logt", "logg", "feh_surf", "afe_surf")
        missing = [p for p in needed if p not in tracks.predictions]
        if missing:
            raise ValueError(
                f"tracks.predictions lacks {missing}, which StarEvolTrack needs"
            )
        self.tracks = tracks

        # Set up filters
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters

        self.predictor = FastNNPredictor(
            filters=filters, nnfile=nnfile, verbose=verbose
        )

    def get_seds(
        self,
        mini=1.0,
        eep=350,
        feh=0.0,
        afe=0.0,
        av=0.0,
        rv=3.3,
        smf=0.0,
        dist=1000.0,
        loga_max=10.15,
        eep2=None,
        mini_bound=0.08,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
        return_eep2=False,
        return_dict=True,
        combine_seds=True,
        tol=1e-2,
    ):
        r"""
        Generate synthetic SED for an individual star.

        Parameters
        ----------
        mini : float, optional
            Initial stellar mass in solar masses. Default is 1.0.

        eep : float, optional
            Equivalent evolutionary point. Default is 350.

        feh : float, optional
            Metallicity [Fe/H]. Default is 0.0.

        afe : float, optional
            Alpha enhancement [α/Fe]. Default is 0.0.

        av : float, optional
            V-band extinction A(V). Default is 0.0.

        rv : float, optional
            Extinction law parameter R(V). Default is 3.3.

        smf : float, optional
            Secondary mass fraction for binary. Default is 0.0 (single star).

        dist : float, optional
            Distance in parsecs. Default is 1000.0.

        loga_max : float, optional
            Maximum log(age) for SED computation. Default is 10.15.

        eep2 : float, optional
            EEP of secondary component for binaries.

        mini_bound : float, optional
            Minimum mass for SED computation. Default is 0.08.

        eep_binary_max : float, optional
            Maximum EEP for binary modeling. Default is 480.0.

        apply_corr : bool, optional
            Apply empirical corrections. Default is True.

        corr_params : tuple, optional
            Correction parameters.

        return_eep2 : bool, optional
            Return secondary EEP. Default is False.

        return_dict : bool, optional
            Return parameters as dictionary. Default is True.

        combine_seds : bool, optional
            If True and binary star requested, combine primary and secondary SEDs by adding
            their magnitudes. If False, separate SEDs for primary and secondary will be returned.
            Default is True.

        tol : float, optional
            Convergence tolerance (in dex) on the log10(age) match when solving
            for the secondary's EEP in a binary. A companion whose age cannot be
            matched to within ``tol`` is treated as non-existent and a
            primary-only SED is returned (rather than discarding the model).
            Default is 1e-2.

        Returns
        -------
        sed : numpy.ndarray of shape (Nfilters,) or tuple
            Synthetic SED in magnitudes.

        params : dict or numpy.ndarray
            Primary component stellar parameters.

        params2 : dict or numpy.ndarray
            Secondary component parameters.

        eep2 : float, optional
            Secondary EEP (if return_eep2=True).

        See Also
        --------
        EEPTracks.get_predictions : Stellar parameter predictions
        FastNNPredictor.sed : Neural network SED generation
        _get_eep_for_secondary : Binary companion EEP calculation

        Notes
        -----
        Distance modulus is applied as:

        .. math::
            m = M + 5 \log_{10}(d/10\,{\rm pc})

        where d is the distance in parsecs.

        Binary SEDs are combined using magnitude addition (flux summing):

        .. math::
            m_{\rm combined} = -2.5 \log_{10}(10^{-0.4 m_1} + 10^{-0.4 m_2})

        Examples
        --------
        Single star:

        >>> sed, params, params2 = star_track.get_seds(
        ...     mini=1.0, eep=350, feh=0.0, afe=0.0
        ... )

        Binary system:

        >>> sed, params, params2 = star_track.get_seds(
        ...     mini=1.2, eep=400, feh=-0.2, afe=0.0, smf=0.7
        ... )
        """
        # Grab input labels
        labels = {"mini": mini, "eep": eep, "feh": feh, "afe": afe}
        labels = np.array([labels[lbl] for lbl in self.tracks.labels])

        # Generate primary component predictions
        params_arr = self.tracks.get_predictions(
            labels, apply_corr=apply_corr, corr_params=corr_params
        )

        # Convert to dictionary format
        params = dict(zip(self.tracks.predictions, params_arr))
        sed = np.full(self.predictor.NFILT, np.nan)
        sed2 = np.full(self.predictor.NFILT, np.nan)

        # Initialize secondary parameters
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.tracks.predictions, params_arr2))

        # Generate primary SED
        loga = params["loga"]

        if loga <= loga_max:
            sed = self.predictor.sed(
                logl=params["logl"],
                logt=params["logt"],
                logg=params["logg"],
                feh_surf=params["feh_surf"],
                afe=params["afe_surf"],
                av=av,
                rv=rv,
                dist=dist,
            )

            # Add binary companion if requested
            if self._tries_companion(mini, eep, smf, mini_bound, eep_binary_max):
                # Generate secondary parameters
                if eep2 is None:
                    eep2 = self._get_eep_for_secondary(
                        loga, mini, eep, feh, afe, smf, tol
                    )

                if not np.isfinite(eep2):
                    # The companion's age could not be matched to the primary
                    # within tolerance. Keep the (valid) primary SED rather than
                    # poisoning it with a NaN secondary via add_mag, which would
                    # turn the whole combined SED into NaN and silently discard
                    # an otherwise-valid model during grid generation.
                    warnings.warn(
                        f"Secondary EEP did not converge for binary "
                        f"(mini={mini}, smf={smf}); returning primary-only SED.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    labels2 = {
                        "mini": mini * smf,
                        "eep": eep2,
                        "feh": feh,
                        "afe": afe,
                    }
                    labels2 = np.array([labels2[lbl] for lbl in self.tracks.labels])

                    params_arr2 = self.tracks.get_predictions(
                        labels2, apply_corr=apply_corr, corr_params=corr_params
                    )
                    params2 = dict(zip(self.tracks.predictions, params_arr2))

                    # Generate secondary SED
                    sed2 = self.predictor.sed(
                        logl=params2["logl"],
                        logt=params2["logt"],
                        logg=params2["logg"],
                        feh_surf=params2["feh_surf"],
                        afe=params2["afe_surf"],
                        av=av,
                        rv=rv,
                        dist=dist,
                    )
                    if combine_seds:
                        # Combine SEDs (magnitude addition)
                        sed = add_mag(sed, sed2)

        # Format output
        if not return_dict:
            params = params_arr
            params2 = params_arr2
        if not combine_seds and smf > 0.0:
            # Return separate SEDs for primary and secondary as a list
            sed = [sed, sed2]
        if return_eep2:
            return sed, params, params2, eep2
        else:
            return sed, params, params2

    def _tries_companion(self, mini, eep, smf, mini_bound, eep_binary_max):
        """Return whether `get_seds` looks for a companion for these labels."""
        mini_min = max(getattr(self.tracks, "mini_bound", 0.08), mini_bound)
        return smf > 0.0 and eep <= eep_binary_max and mini * smf >= mini_min

    def _get_eep_for_secondary(self, loga, mini, eep, feh, afe, smf, tol):
        r"""
        Calculate EEP for secondary component that matches the age of the primary.

        This method solves the inverse problem: given a target age (from the primary),
        find the EEP that produces that age for the secondary star with mass mini*smf.
        Age is monotonically increasing along a track, so the interpolated
        age(EEP) curve is inverted directly with a single vectorized
        interpolator call over the EEP grid axis. A scalar Nelder-Mead solve
        is kept as a fallback for track objects that do not expose the EEP
        grid axis (``tracks.gridpoints``) or whose age curve is not monotone.

        Parameters
        ----------
        loga : float
            Target log10(age in years) to match from primary star
        mini : float
            Primary star initial mass in solar masses
        eep : float
            Primary star EEP (used as initial guess for the Nelder-Mead
            fallback)
        feh : float
            Metallicity [Fe/H] in logarithmic solar units
        afe : float
            Alpha enhancement [α/Fe] in logarithmic solar units
        smf : float
            Secondary mass fraction (secondary mass = mini * smf)
        tol : float
            Convergence tolerance (in dex) on the log10(age) match. A match
            is accepted when ``|dloga| < tol``; otherwise NaN is returned.

        Returns
        -------
        eep2 : float
            EEP for secondary star that produces the target age.
            Returns NaN if the inversion fails or doesn't converge within
            tolerance.

        See Also
        --------
        get_seds : Uses this method for binary star modeling
        EEPTracks.get_predictions : Called to evaluate ages at different EEPs

        Notes
        -----
        The secondary uses the primary's [α/Fe], as `get_seds` does for the
        secondary's parameters and SED.

        Empirical corrections never touch ``loga``, so the age lookups skip
        them (``apply_corr=False``).
        """
        # Get age index from tracks
        aidx = self.tracks.predictions.index("loga")
        mini2 = mini * smf

        # Fast path: invert the monotone age(EEP) curve directly.
        gridpoints = getattr(self.tracks, "gridpoints", None)
        if isinstance(gridpoints, dict) and "eep" in gridpoints:
            eep2 = self._invert_age_curve(loga, mini2, feh, afe, aidx, tol)
            if eep2 is not None:
                return eep2

        # Fallback: scalar Nelder-Mead solve on the squared log-age
        # difference (duck-typed tracks without an EEP grid axis, or a
        # non-monotone age curve).
        def loss(x):
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x[0]
            # Get predicted age for secondary star with mass mini*smf at EEP x
            loga_pred = self.tracks.get_predictions(
                [mini2, x, feh, afe], apply_corr=False
            )[aidx]
            return (loga_pred - loga) ** 2

        # Find best-fit EEP that minimizes age difference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore bad values during optimization
            res = minimize(loss, eep, method="Nelder-Mead")

        # Check if solution meets tolerance. loss is the squared log-age
        # difference, so this accepts matches with |dloga| < tol (in dex).
        return res.x[0] if res.fun < tol**2 else np.nan

    def _invert_age_curve(self, loga, mini2, feh, afe, aidx, tol):
        """
        Invert the interpolated age(EEP) curve for a star of mass `mini2`.

        Evaluates log10(age) along the full EEP grid axis at fixed
        (mini2, feh, afe) with one vectorized interpolator call, then
        inverts the piecewise-linear curve with ``np.interp``. Because the
        underlying interpolant is linear in EEP with knots at the grid
        values, this inversion is exact (up to the acceptance check).

        Parameters
        ----------
        loga : float
            Target log10(age in years).
        mini2 : float
            Secondary initial mass in solar masses.
        feh : float
            Metallicity [Fe/H].
        afe : float
            Alpha enhancement [α/Fe] of the primary.
        aidx : int
            Index of 'loga' in the tracks' predictions.
        tol : float
            Acceptance tolerance in dex on |loga(eep2) - loga|.

        Returns
        -------
        eep2 : float or None
            The matching EEP; NaN if no EEP matches the target age within
            `tol`; None if the age curve is not monotone (the caller then
            falls back to the scalar Nelder-Mead solve).
        """
        eep_axis = np.asarray(self.tracks.gridpoints["eep"], dtype=float)
        values = {"mini": mini2, "eep": eep_axis, "feh": feh, "afe": afe}
        labels = np.column_stack(
            [np.broadcast_to(values[lbl], eep_axis.shape) for lbl in self.tracks.labels]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.tracks.get_predictions(labels, apply_corr=False)
        loga_curve = np.asarray(preds, dtype=float)[:, aidx]

        # Restrict to the EEP range where the track exists (NaN = off-track)
        finite = np.isfinite(loga_curve)
        if not np.any(finite):
            return np.nan
        eeps_f = eep_axis[finite]
        loga_f = loga_curve[finite]

        # Age must be (weakly) monotone increasing for a direct inversion
        if np.any(np.diff(loga_f) < 0):
            return None

        eep2 = float(np.interp(loga, loga_f, eeps_f))

        # Acceptance check against the actual interpolant (also rejects
        # targets outside the track's age range, which np.interp clamps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check = self.tracks.get_predictions(
                [values[lbl] if lbl != "eep" else eep2 for lbl in self.tracks.labels],
                apply_corr=False,
            )[aidx]

        if np.isfinite(check) and abs(check - loga) < tol:
            return eep2
        return np.nan


class StarGrid:
    """
    Grid-based stellar modeling and synthetic photometry generation.

    This class provides an interface for working with pre-computed stellar
    model grids, enabling parameter interpolation and SED generation similar
    to StarEvolTrack but using grid-based models rather than evolutionary tracks.

    The grid structure allows for irregular spacing in each dimension, with
    models indexed by their array location. Multi-linear interpolation is used
    to compute stellar parameters and photometry between grid points.

    Parameters
    ----------
    models : numpy.ndarray of shape (Nmodel, Nfilt, Ncoef) or dict
        Pre-computed model grid containing photometric coefficients.
        A dict must contain the 'mag_coeffs' key.
        Each model contains 3 coefficients per filter:
        - Unreddened magnitude
        - Reddening vector for R_V = 0
        - Change in reddening vector as function of R_V

    models_labels : structured numpy.ndarray of shape (Nmodel,)
        Labels for each model in the grid (e.g., mini, eep, feh, afe, smf).

    models_params : structured numpy.ndarray of shape (Nmodel,), optional
        Additional parameters for each model (e.g., loga, logl, logt, logg).
        If not provided, these won't be available in predictions.

    filters : list of str, optional
        Filter names for photometry. For a structured `models` array, the
        filters to keep, in this order; names not in `models` are skipped
        with a warning. Otherwise, one name per filter of `models`. If
        None, uses all available filters from the models.

    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Attributes
    ----------
    nmodels : int
        Number of models in the grid

    nfilters : int
        Number of filters

    filters : numpy.ndarray
        Array of filter names

    labels : structured numpy.ndarray
        Grid labels (mini, eep, feh, etc.)

    params : structured numpy.ndarray or None
        Additional parameters if provided

    label_names : list
        Names of available labels

    param_names : list
        Names of available parameters

    Raises
    ------
    ValueError
        If no requested filter is in a structured `models` array, if
        `filters` does not give one name per filter of `models`, or if a grid
        label has no finite value or no model has finite values for all of
        them.

    Notes
    -----
    Binary star support is currently limited. The `smf` parameter is accepted
    for API compatibility with StarEvolTrack, but full binary modeling requires
    a dedicated binary grid with pre-computed combined photometry. The current
    implementation returns empty placeholders for secondary parameters.

    Examples
    --------
    Load a pre-computed grid and generate photometry:

    >>> from brutus.data import load_models
    >>> models, labels, label_mask = load_models('grid_mist_v9.h5')
    >>> grid = StarGrid(models, labels)
    >>>
    >>> # Get predictions for specific stellar parameters
    >>> predictions = grid.get_predictions(mini=1.0, eep=350, feh=0.0)
    >>>
    >>> # Generate SED with extinction
    >>> sed, params, params2 = grid.get_seds(
    ...     mini=1.0, eep=350, feh=0.0,
    ...     av=0.1, rv=3.3, dist=1000.0
    ... )
    """

    def __init__(
        self, models, models_labels, models_params=None, filters=None, verbose=True
    ):
        """Initialize the StarGrid with model data."""
        # Handle different input formats
        if isinstance(models, dict):
            # Dictionary input
            if "mag_coeffs" in models:
                mag_coeffs = models["mag_coeffs"]
            else:
                raise ValueError("models dict must contain 'mag_coeffs' key")

            if "labels" in models and models_labels is None:
                models_labels = models["labels"]

            if "parameters" in models and models_params is None:
                models_params = models["parameters"]

            models = mag_coeffs

        # Store model data
        self.models = np.asarray(models)
        self.labels = np.asarray(models_labels)
        self.params = np.asarray(models_params) if models_params is not None else None

        # Get dimensions
        if self.models.ndim == 2:
            # Models are (Nmodel, Nfilt*3) - reshape to (Nmodel, Nfilt, 3)
            self.nmodels = self.models.shape[0]
            self.nfilters = self.models.shape[1] // 3
            self.models = self.models.reshape(self.nmodels, self.nfilters, 3)
        elif self.models.ndim == 3:
            # Models are already (Nmodel, Nfilt, 3)
            self.nmodels, self.nfilters, ncoef = self.models.shape
            if ncoef != 3:
                raise ValueError(f"Expected 3 coefficients per filter, got {ncoef}")
        else:
            # Handle structured array from actual data files
            if models.dtype.names is not None:
                # Columns follow the requested order, as in `load_models`
                filter_names = list(models.dtype.names)
                if filters is not None:
                    missing = [f for f in filters if f not in filter_names]
                    filter_names = [f for f in filters if f in filter_names]
                    if not filter_names:
                        raise ValueError(
                            f"None of the requested filters {list(filters)} "
                            f"exist in models. Available filters: "
                            f"{list(models.dtype.names)}."
                        )
                    if missing:
                        warnings.warn(
                            f"Requested filters not found in models and "
                            f"skipped: {missing}. The model columns correspond "
                            f"(in order) to: {filter_names}.",
                            UserWarning,
                            stacklevel=2,
                        )

                self.filters = np.array(filter_names)
                self.nfilters = len(self.filters)
                self.nmodels = len(models)

                # Extract coefficients for each filter
                model_array = np.zeros((self.nmodels, self.nfilters, 3))
                for i, filt in enumerate(self.filters):
                    model_array[:, i, :] = models[filt]
                self.models = model_array
            else:
                raise ValueError(f"Unexpected models shape: {self.models.shape}")

        # Set filter names if not already set
        if not hasattr(self, "filters"):
            if filters is not None:
                if len(filters) != self.nfilters:
                    raise ValueError(
                        f"filters has {len(filters)} names, but models has "
                        f"{self.nfilters} filters."
                    )
                self.filters = np.array(filters)
            else:
                # Generate default filter names
                self.filters = np.array([f"filter_{i}" for i in range(self.nfilters)])

        # Store label and parameter names
        if hasattr(self.labels, "dtype") and self.labels.dtype.names:
            self.label_names = list(self.labels.dtype.names)
        else:
            self.label_names = []

        if (
            self.params is not None
            and hasattr(self.params, "dtype")
            and self.params.dtype.names
        ):
            self.param_names = list(self.params.dtype.names)
        else:
            self.param_names = []

        # Build lookup indices for efficient interpolation
        self._build_grid_indices()

        # The KD-tree is built on first use
        self.kdtree = None

        if verbose:
            print(
                f"Loaded StarGrid with {self.nmodels:,} models, "
                f"{self.nfilters} filters, {len(self.label_names)} labels"
            )
            if self.param_names:
                print(f"Additional parameters: {', '.join(self.param_names)}")

    def _build_grid_indices(self):
        """
        Build indices for efficient grid lookup and interpolation.

        Populates ``grid_axes`` (unique values per label), ``grid_shape``,
        and a corner-index lookup mapping per-axis indices to the model row
        (-1 / absent = no model at that grid combination; grids need not be
        complete Cartesian products, e.g. tracks truncate at different EEPs).
        The lookup is a dense int64 array when the full Cartesian size is
        modest, otherwise a dict keyed by index tuples.
        """
        # Create unique value arrays for each label dimension
        self.grid_axes = {}
        self.grid_shape = []

        # A non-finite label is not a grid value, so the axes and both
        # neighbour searches use only the models with finite labels
        valid = np.ones(self.nmodels, dtype=bool)
        for label in self.label_names:
            if label in ["mini", "eep", "feh", "afe", "smf"]:
                vals = self.labels[label]
                finite = np.isfinite(vals)
                unique_vals = np.unique(vals[finite])
                if unique_vals.size == 0:
                    raise ValueError(f"Grid label {label} has no finite value.")
                self.grid_axes[label] = unique_vals
                self.grid_shape.append(len(unique_vals))
                valid &= finite
        self._finite_rows = np.where(valid)[0]
        # (low, high) of each label a query may give: the grid axes, and the
        # constant labels that load_models left out (labels.dtype.metadata)
        self._label_range = {
            name: (vals[0], vals[-1]) for name, vals in self.grid_axes.items()
        }
        dtype_metadata = getattr(self.labels.dtype, "metadata", None) or {}
        for name, value in dtype_metadata.get("constants", {}).items():
            self._label_range.setdefault(name, (value, value))

        # Corner-index lookup: axis indices -> model row (first match wins)
        self._axis_names = list(self.grid_axes.keys())
        self._grid_idx_map = None
        if self._axis_names:
            if self._finite_rows.size == 0:
                raise ValueError("No model has finite values for every grid label.")
            rows = self._finite_rows

            # Per-model axis indices (exact: axes come from np.unique of
            # these same label values)
            coords = tuple(
                np.searchsorted(self.grid_axes[name], self.labels[name][rows])
                for name in self._axis_names
            )
            shape = tuple(len(self.grid_axes[name]) for name in self._axis_names)
            if np.prod(shape, dtype=np.int64) <= 20_000_000:
                # The FIRST model at duplicated coordinates must win; fancy
                # assignment with repeated indices has no write-order
                # guarantee, so resolve duplicates explicitly via np.unique
                # (returns first occurrences).
                flat = np.ravel_multi_index(coords, shape)
                uniq, first = np.unique(flat, return_index=True)
                idx_map = np.full(shape, -1, dtype=np.int64)
                idx_map.ravel()[uniq] = rows[first]
                self._grid_idx_map = idx_map
            else:
                idx_map = {}
                for i in range(len(rows) - 1, -1, -1):
                    idx_map[tuple(int(c[i]) for c in coords)] = int(rows[i])
                self._grid_idx_map = idx_map

    def _lookup_corner(self, axis_indices):
        """Return the model row at the given per-axis indices, or -1."""
        if isinstance(self._grid_idx_map, dict):
            return self._grid_idx_map.get(axis_indices, -1)
        return int(self._grid_idx_map[axis_indices])

    def _check_query(self, **query):
        """Return the given grid-axis values of a query, or raise ValueError.

        Every multi-valued axis must be given. Each given value must be
        finite and inside its axis range, with a relative tolerance of 1e-6
        at the edges: stored edges are not round (the MIST mini maximum is
        2.0000000000000013). A constant label that `load_models` left out
        is checked against its value. Any other label not in the grid warns
        and is ignored.
        """
        if not self._axis_names:
            raise ValueError(
                "StarGrid needs at least one grid label (mini, eep, feh, afe or "
                f"smf) to answer a query; models_labels has {self.label_names}."
            )
        given = {name: value for name, value in query.items() if value is not None}
        ignored = sorted(given.keys() - self._label_range.keys())
        if ignored:
            warnings.warn(
                f"{ignored} are not labels of this grid and are ignored.",
                UserWarning,
                stacklevel=4,
            )
        missing = [
            name
            for name in self._axis_names
            if name not in given and len(self.grid_axes[name]) > 1
        ]
        if missing:
            ranges = {name: self._label_range[name] for name in missing}
            raise ValueError(
                f"Grid parameter(s) {missing} not specified. These axes span "
                f"{ranges}. Specify them explicitly."
            )
        for name, value in given.items():
            if name not in self._label_range:
                continue
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
            low, high = self._label_range[name]
            tol = 1e-6 * max(abs(low), abs(high), 1.0)
            if not low - tol <= value <= high + tol:
                raise ValueError(
                    f"{name}={value} is outside the grid range [{low}, {high}]."
                )
        return {name: v for name, v in given.items() if name in self.grid_axes}

    def _build_kdtree(self):
        """Build the KD-tree over all grid axes, each scaled to [0, 1]."""
        from scipy.spatial import cKDTree

        # A fixed axis order keeps the distances, summed over axes, independent
        # of the column order of models_labels
        # Per-axis (minimum, span), kept for scaling the queries
        self._kdtree_scale = {}
        coords = []
        for label in ["mini", "eep", "feh", "afe", "smf"]:
            if label not in self.grid_axes:
                continue
            vals = self.labels[label][self._finite_rows]
            val_min, val_max = self._label_range[label]
            span = val_max - val_min
            self._kdtree_scale[label] = (val_min, span)
            if span > 0:
                coords.append((vals - val_min) / span)
            else:
                coords.append(np.zeros_like(vals))
        self.kdtree = cKDTree(np.column_stack(coords))

    def _find_neighbors_multilinear(self, **req_params):
        """
        Find bracketing grid points and compute multi-linear interpolation weights.

        For each dimension, finds the two bracketing values and computes
        interpolation weights. This gives us 2^N neighbors for N dimensions.

        Parameters
        ----------
        **req_params : keyword arguments
            Grid-axis values checked by `_check_query`.

        Returns
        -------
        indices : numpy.ndarray
            Indices of neighboring grid points

        weights : numpy.ndarray
            Interpolation weights for each neighbor

        Notes
        -----
        Performs multi-linear interpolation by:
        1. Finding bracketing grid points in each dimension
        2. Computing linear interpolation weights
        3. Generating all 2^N corner points for N dimensions
        4. Weighting each corner by product of dimension weights

        Corners are resolved via the precomputed corner-index lookup (O(1)
        per corner) rather than boolean scans over all models.

        Falls back to the KD-tree method when the bracketing corners are
        missing from the (incomplete) grid: entirely, or when the surviving
        corners carry less than half of the total interpolation weight.
        Renormalizing low-weight surviving corners would silently return a
        far-away model (e.g. beyond a truncated track's last EEP).
        """
        import itertools

        if not req_params:
            # Degenerate grid (every axis single-valued): return the first
            # model with weight 1
            return np.array([0]), np.array([1.0])

        # For each grid axis, find bracketing (axis index, weight) pairs;
        # unspecified single-valued axes take their only value
        axis_brackets = []
        for name in self._axis_names:
            if name not in req_params:
                axis_brackets.append([(0, 1.0)])
                continue

            value = req_params[name]
            axis_values = self.grid_axes[name]

            # Find bracketing indices using searchsorted
            idx = np.searchsorted(axis_values, value)

            if idx == 0:
                # Before first point
                axis_brackets.append([(0, 1.0)])
            elif idx >= len(axis_values):
                # After last point
                axis_brackets.append([(len(axis_values) - 1, 1.0)])
            else:
                # Between points
                idx_low, idx_high = idx - 1, idx
                val_low = axis_values[idx_low]
                val_high = axis_values[idx_high]
                if val_high > val_low:
                    weight_high = (value - val_low) / (val_high - val_low)
                else:
                    weight_high = 0.5
                axis_brackets.append(
                    [(idx_low, 1.0 - weight_high), (idx_high, weight_high)]
                )

        # Resolve each corner combination with an O(1) index lookup
        indices = []
        weights = []
        n_missing = 0

        for combo in itertools.product(*axis_brackets):
            # Weight is product of all dimension weights; a query on a grid
            # node gives its neighbours weight 0, which need no model
            weight = float(np.prod([c[1] for c in combo]))
            if weight == 0.0:
                continue
            model_idx = self._lookup_corner(tuple(c[0] for c in combo))
            if model_idx >= 0:
                indices.append(model_idx)
                weights.append(weight)
            else:
                n_missing += 1

        wsum = float(np.sum(weights))
        if wsum < 0.5:
            # All bracketing corners are missing, or the surviving corners
            # carry a minority of the interpolation weight (the query's
            # dominant corner is absent from the grid). Renormalizing the
            # survivors would silently snap to a far-away model, so use the
            # nearest-neighbor fallback instead.
            return self._find_neighbors_kdtree(**req_params)

        if n_missing > 0:
            warnings.warn(
                f"{n_missing} bracketing grid corner(s) missing from the "
                f"model grid for query {req_params}; renormalizing over the "
                f"surviving corners (total weight {wsum:.3f}).",
                UserWarning,
                stacklevel=3,
            )

        # Normalize weights
        indices = np.array(indices)
        weights = np.array(weights)
        weights /= wsum

        return indices, weights

    def _find_neighbors_kdtree(self, **kwargs):
        """
        Find nearest neighbors using KD-tree (fallback method).

        Parameters
        ----------
        **kwargs : keyword arguments
            Grid-axis values checked by `_check_query`.

        Returns
        -------
        indices : numpy.ndarray
            Indices of neighboring grid points

        weights : numpy.ndarray
            Interpolation weights for each neighbor

        Notes
        -----
        Uses inverse distance weighting with up to k=8 neighbors.
        Distances are computed in normalized parameter space where
        each dimension is scaled to [0, 1] for balanced metrics.

        The KD-tree is built lazily on first call and cached for
        subsequent queries.
        """
        if self.kdtree is None:
            self._build_kdtree()

        # A single-valued axis scales to 0 and may be omitted
        query_point = [
            (kwargs[label] - val_min) / span if span > 0 else 0.0
            for label, (val_min, span) in self._kdtree_scale.items()
        ]

        k = min(8, len(self._finite_rows))  # Use up to 8 neighbors
        distances, indices = self.kdtree.query(query_point, k=k)
        # A 1-model grid returns scalars
        distances, indices = np.atleast_1d(distances, indices)
        indices = self._finite_rows[indices]

        # Convert distances to weights (inverse distance weighting)
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights /= weights.sum()

        return indices, weights

    def _neighbors(self, use_multilinear, **labels):
        """Return the checked query and the indices and weights of its neighbours."""
        query = self._check_query(**labels)
        if use_multilinear:
            return (query, *self._find_neighbors_multilinear(**query))
        return (query, *self._find_neighbors_kdtree(**query))

    def _interpolate_params(self, query, indices, weights):
        """Return the query labels and the weighted parameters as a dict.

        Without parameters, return the weighted labels instead.
        """
        if self.params is not None and self.param_names:
            predictions = dict(query)
            neighbors = self.params[indices]
            for param in self.param_names:
                predictions[param] = np.sum(neighbors[param] * weights)
            return predictions
        return {
            label: np.sum(self.labels[label][indices] * weights)
            for label in self.label_names
        }

    def get_predictions(
        self,
        mini=None,
        eep=None,
        feh=None,
        afe=None,
        smf=None,
        use_multilinear=True,
    ):
        """
        Get stellar parameter predictions from the grid.

        Interpolates grid models to estimate stellar parameters at the
        requested input values using multi-linear interpolation.

        Parameters
        ----------
        mini : float, optional
            Initial mass in solar masses

        eep : float, optional
            Equivalent evolutionary phase

        feh : float, optional
            Metallicity [Fe/H]

        afe : float, optional
            Alpha enhancement [α/Fe]

        smf : float, optional
            Secondary mass fraction for binaries

        use_multilinear : bool, optional
            Use multi-linear interpolation (True) or KD-tree nearest neighbor (False).
            Default is True.

        Returns
        -------
        predictions : dict or numpy.ndarray
            Predicted stellar parameters. Returns dict with parameter names
            as keys if parameters are available, otherwise returns array.

        Raises
        ------
        ValueError
            If a grid dimension with more than one value is omitted, or if a
            given grid label is not finite or lies outside its axis range
            (with a relative tolerance of 1e-6 at the edges). Single-valued
            dimensions may be omitted and take their only value. Also if a
            constant label that `load_models` left out (such as ``afe`` on a
            single-afe grid) is given another value.

        Warns
        -----
        UserWarning
            If a label is given that the grid does not have and that
            `load_models` did not record as a constant. It is ignored.

        See Also
        --------
        get_seds : Generate photometry along with parameter predictions
        _find_neighbors_multilinear : Multi-linear interpolation method
        _find_neighbors_kdtree : KD-tree nearest neighbor method

        Examples
        --------
        >>> grid = StarGrid(models, labels, params)
        >>> preds = grid.get_predictions(mini=1.0, eep=350, feh=0.0)
        >>> print(f"log(age) = {preds['loga']:.2f}")
        >>> print(f"log(L) = {preds['logl']:.2f}")
        """
        query, indices, weights = self._neighbors(
            use_multilinear, mini=mini, eep=eep, feh=feh, afe=afe, smf=smf
        )

        return self._interpolate_params(query, indices, weights)

    def get_seds(
        self,
        mini=None,
        eep=None,
        feh=None,
        afe=None,
        av=0.0,
        rv=3.3,
        smf=None,
        dist=1000.0,
        return_dict=True,
        return_flux=False,
        return_predictions=True,
        use_multilinear=True,
    ):
        r"""
        Generate synthetic SED from the grid.

        Interpolates grid models and applies extinction to generate
        synthetic photometry using multi-linear interpolation.

        Parameters
        ----------
        mini : float, optional
            Initial mass in solar masses

        eep : float, optional
            Equivalent evolutionary phase

        feh : float, optional
            Metallicity [Fe/H]

        afe : float, optional
            Alpha enhancement [α/Fe]

        av : float, optional
            V-band extinction in magnitudes. Default is 0.0.

        rv : float, optional
            Reddening law parameter. Default is 3.3.

        smf : float, optional
            Secondary mass fraction for binaries. NOTE: Currently returns
            empty placeholder for params2. Full binary support requires
            a dedicated binary grid with pre-computed combined photometry.

        dist : float, optional
            Distance in parsecs. Default is 1000.0 (1 kpc), which corresponds
            to parallax = 1 mas for consistency with Gaia units.

        return_dict : bool, optional
            If True, return parameters as dict. Default is True.

        return_flux : bool, optional
            If True, return fluxes instead of magnitudes. Default is False.

        return_predictions : bool, optional
            If True, compute and return stellar parameters. Set to False
            to only get photometry (more efficient). Default is True.

        use_multilinear : bool, optional
            Use multi-linear interpolation (True) or KD-tree nearest neighbor (False).
            Default is True.

        Returns
        -------
        sed : numpy.ndarray of shape (Nfilt,)
            Synthetic photometry (magnitudes or fluxes)

        params : dict or numpy.ndarray or None
            Primary star parameters (None if return_predictions=False)

        params2 : dict or numpy.ndarray or None
            Secondary star parameters (empty placeholder - full binary
            implementation requires dedicated binary grid)

        Raises
        ------
        ValueError
            As in `get_predictions`, or if `dist` is not a finite distance
            greater than 0.

        Warns
        -----
        UserWarning
            As in `get_predictions`.

        See Also
        --------
        get_predictions : Get stellar parameters without photometry
        StarEvolTrack.get_seds : Alternative track-based SED generation
        brutus.analysis.BruteForce : Fitting with StarGrid

        Notes
        -----
        The SED is computed by interpolating magnitude coefficients and
        applying the extinction law:

        .. math::
            m(\lambda) = m_0(\lambda) + A_V \cdot [r_0(\lambda) + R_V \cdot dr(\lambda)]

        where :math:`m_0` is the unreddened magnitude, :math:`r_0` and :math:`dr`
        are the reddening vector coefficients from the grid.

        Examples
        --------
        >>> grid = StarGrid(models, labels)
        >>> sed, params, _ = grid.get_seds(
        ...     mini=1.0, eep=350, feh=0.0,
        ...     av=0.1, dist=500.0
        ... )
        >>> print(f"G magnitude: {sed[0]:.2f}")
        """
        if dist is None or not np.isfinite(dist) or dist <= 0:
            raise ValueError(f"dist must be a finite distance > 0 in pc; got {dist}.")
        # Warning for binary support limitations (a grid without smf ignores it)
        if smf is not None and smf > 0 and "smf" in self.grid_axes:
            warnings.warn(
                "Binary star support in StarGrid is limited. Secondary parameters "
                "(params2) will be empty. For full binary modeling, use StarEvolTrack "
                "or wait for binary grid implementation.",
                UserWarning,
                stacklevel=2,
            )

        query, indices, weights = self._neighbors(
            use_multilinear, mini=mini, eep=eep, feh=feh, afe=afe, smf=smf
        )

        # Weighted average of the magnitude coefficients (np.sum adds the rows
        # in order; np.tensordot can change the last bit), reddened as in
        # `_get_seds`. Grid magnitudes are at 1 kpc.
        coeffs = np.sum(self.models[indices] * weights[:, None, None], axis=0)
        mags = (
            coeffs[:, 0]
            + av * (coeffs[:, 1] + rv * coeffs[:, 2])
            + 5.0 * np.log10(dist / 1000.0)
        )
        sed = 10 ** (-0.4 * mags) if return_flux else mags

        if return_predictions:
            params = self._interpolate_params(query, indices, weights)
        else:
            params = None

        # Binary placeholder (full implementation requires dedicated binary grid)
        params2 = None

        # Format output for compatibility
        if not return_dict:
            if isinstance(params, dict):
                params = np.array(list(params.values())) if params else np.array([])
            if params2 is None:
                params2 = np.array([])

        elif params2 is None:
            params2 = {}

        return sed, params, params2

    def __repr__(self):
        """Return string representation of the StarGrid object."""
        rep = f"StarGrid(nmodels={self.nmodels:,}, nfilters={self.nfilters}"
        if self.label_names:
            rep += f", labels={self.label_names}"
        rep += ")"
        return rep

    def __len__(self):
        """Return the number of models in the grid."""
        return self.nmodels
