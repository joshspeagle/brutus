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

Classes
-------
EEPTracks : Individual stellar parameter predictions
    Interpolates MIST evolutionary tracks to predict stellar parameters
    (age, luminosity, temperature, etc.) as a function of initial mass,
    evolutionary phase (EEP), metallicity, and alpha enhancement.

StarEvolTrack : Individual stellar photometry synthesis
    Generates synthetic photometry for individual stars using neural
    network bolometric corrections, with support for binary companions,
    dust extinction, and observational effects.

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
>>>     mini=1.2, eep=400, feh=-0.2, afe=0.1,
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

This implementation is based on the MIST stellar evolution framework [1]_ [2]_.

References
----------
.. [1] Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
       for the Construction of Stellar Isochrones", ApJ, 823, 102
.. [2] Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
       Models", ApJS, 222, 8
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

# Import filter definitions
from ..data.filters import FILTERS

# Import utilities
from ..utils.photometry import add_mag

# Import neural network predictor
from .neural_nets import FastNNPredictor

__all__ = ["EEPTracks", "StarEvolTrack"]


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
        provided, defaults to the standard MIST v1.2 EEP tracks file.

    predictions : iterable of str, optional
        The names of stellar parameters to predict. Default is:
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

    Attributes
    ----------
    labels : list of str
        Input parameter names: ['mini', 'eep', 'feh', 'afe']

    predictions : list of str
        Output parameter names as specified in initialization

    ndim, npred : int
        Number of input dimensions and predicted parameters

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

        # Initialize values
        self.labels = list(np.array(labels))
        self.predictions = list(np.array(predictions))
        self.ndim, self.npred = len(self.labels), len(self.predictions)
        self.null = np.zeros(self.npred) + np.nan

        # Set label references for fast indexing
        self.mini_idx = np.where(np.array(self.labels) == "mini")[0][0]
        self.eep_idx = np.where(np.array(self.labels) == "eep")[0][0]
        self.feh_idx = np.where(np.array(self.labels) == "feh")[0][0]
        self.logt_idx = np.where(np.array(self.predictions) == "logt")[0][0]
        self.logl_idx = np.where(np.array(self.predictions) == "logl")[0][0]
        self.logg_idx = np.where(np.array(self.predictions) == "logg")[0][0]

        # Set default file path
        if mistfile is None:
            package_root = Path(__file__).parent.parent.parent.parent
            mistfile = package_root / "data" / "DATAFILES" / "MIST_1.2_EEPtrk.h5"

            # If default path doesn't exist, try pooch cache directory
            # Convert to string for os.path.exists() - helps with mock compatibility
            if not os.path.exists(str(mistfile)):
                import pooch

                cache_dir = Path(pooch.os_cache("astro-brutus"))
                cache_path = cache_dir / "MIST_1.2_EEPtrk.h5"
                if os.path.exists(str(cache_path)):
                    mistfile = cache_path

        self.mistfile = Path(mistfile)

        # Generate cache file path based on original file and configuration
        cache_key = (
            f"{self.mistfile.stem}_ageweight{ageweight}_pred{''.join(predictions)}"
        )
        cache_file = self.mistfile.parent / f"{cache_key}.pkl"

        # Try to load from cache first
        if use_cache and cache_file.exists():
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

                    if verbose:
                        sys.stderr.write("Cached EEPTracks loaded successfully!\n")
                    return
                else:
                    if verbose:
                        sys.stderr.write(
                            "Cache is older than data file, regenerating...\n"
                        )
            except Exception as e:
                if verbose:
                    sys.stderr.write(
                        f"Cache loading failed ({e}), loading from original file...\n"
                    )

        # Load from original file
        if verbose:
            sys.stderr.write(f"Loading evolutionary tracks from {mistfile}...\n")

        # Load and process track data
        try:
            with h5py.File(self.mistfile, "r") as misth5:
                self._make_lib(misth5, verbose=verbose)
            self._lib_as_grid()

            # Construct age weights if requested
            self._ageidx = self.predictions.index("loga")
            if ageweight:
                self._add_age_weights(verbose=verbose)

            # Build interpolation grid
            self._build_interpolator()

            # Save to cache if enabled
            if use_cache:
                try:
                    # Collect all relevant attributes for caching
                    cache_data = {}
                    cache_attrs = [
                        "labels",
                        "predictions",
                        "ndim",
                        "npred",
                        "null",
                        "mini_idx",
                        "eep_idx",
                        "feh_idx",
                        "logt_idx",
                        "logl_idx",
                        "logg_idx",
                        "_ageidx",
                        "mistfile",
                        "grid_dims",
                        "interpolator",
                    ]

                    # Add dynamic attributes that get created during processing
                    for attr in dir(self):
                        if not attr.startswith("__") and attr not in cache_attrs:
                            if hasattr(self, attr):
                                value = getattr(self, attr)
                                # Only cache serializable objects
                                if not callable(value):
                                    cache_attrs.append(attr)

                    # Cache all attributes
                    for attr in cache_attrs:
                        if hasattr(self, attr):
                            cache_data[attr] = getattr(self, attr)

                    with open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f)

                    if verbose:
                        sys.stderr.write(f"EEPTracks cached to {cache_file}\n")

                except Exception as e:
                    if verbose:
                        sys.stderr.write(f"Warning: Failed to cache EEPTracks ({e})\n")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize EEPTracks: {e}")

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
        """
        if verbose:
            sys.stderr.write("  Constructing track library...\n")

        # Extract input parameters (mini, eep, feh, afe)
        cols = [rename[p] for p in self.labels]
        self.libparams = np.concatenate(
            [np.array(misth5[z])[cols] for z in misth5["index"]]
        )
        self.libparams.dtype.names = tuple(self.labels)

        # Handle alpha enhancement availability
        cols = [rename[p] for p in self.predictions]
        afe_col = rename["afe_surf"]
        afe_available = True
        afe_surf_idx = None

        try:
            # Test alpha enhancement column availability
            first_z = list(misth5["index"])[0]
            _ = misth5[first_z][afe_col]
        except (KeyError, ValueError):
            afe_available = False
            for i, pred in enumerate(self.predictions):
                if pred == "afe_surf":
                    afe_surf_idx = i
                    break
            if verbose:
                sys.stderr.write("    [alpha/Fe] column not found, will set to zero\n")

        # Read output parameters efficiently
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

        output_data = [
            np.concatenate([misth5[z][p] for z in misth5["index"]])
            for p in cols_to_read
        ]

        # Create and fill output array
        self.output = np.empty((len(output_data[0]), len(self.predictions)), dtype="f8")

        for read_idx, pred_idx in enumerate(read_to_pred_mapping):
            self.output[:, pred_idx] = output_data[read_idx]

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
        - binwidths: dict of grid spacings for each parameter
        - X: array of grid indices for each library point
        - mini_bound: minimum initial mass in the grid
        """
        # Get unique grid points in each dimension
        self.gridpoints = {}
        self.binwidths = {}
        for p in self.labels:
            self.gridpoints[p] = np.unique(self.libparams[p])
            self.binwidths[p] = np.diff(self.gridpoints[p])

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
        Uses pandas for vectorized computation if available, otherwise falls
        back to a slower loop-based implementation. The age weights are
        appended to the predictions array and "agewt" is added to the
        predictions list.

        The gradient is computed as d(age)/d(EEP) where age is in linear
        (not logarithmic) units, even though ages are stored as log(age).
        """
        if verbose:
            sys.stderr.write("  Computing age weights...\n")

        # Use vectorized approach with pandas if available
        try:
            import pandas as pd

            age_ind = self._ageidx
            df_data = {
                "mini": self.libparams["mini"],
                "feh": self.libparams["feh"],
                "afe": self.libparams["afe"],
                "loga": self.output[:, age_ind],
                "index": np.arange(len(self.libparams)),
            }
            df = pd.DataFrame(df_data)

            ageweights = np.zeros(len(self.libparams))

            for (m, z, a), group in df.groupby(["mini", "feh", "afe"]):
                indices = group["index"].values
                log_ages = group["loga"].values

                if len(log_ages) > 1:
                    linear_ages = 10**log_ages
                    age_gradients = np.gradient(linear_ages)
                    ageweights[indices] = age_gradients

        except ImportError:
            # Fallback to original method
            if verbose:
                sys.stderr.write("    Using fallback method (pandas not available)\n")

            age_ind = self._ageidx
            ageweights = np.zeros(len(self.libparams))

            for i, m in enumerate(self.gridpoints["mini"]):
                for j, z in enumerate(self.gridpoints["feh"]):
                    for k, a in enumerate(self.gridpoints["afe"]):
                        inds = (
                            (self.libparams["mini"] == m)
                            & (self.libparams["feh"] == z)
                            & (self.libparams["afe"] == a)
                        )
                        try:
                            agewts = np.gradient(10 ** self.output[inds, age_ind])
                            ageweights[inds] = agewts
                        except (ValueError, IndexError):
                            pass

        # Append to outputs
        self.output = np.hstack([self.output, ageweights[:, None]])
        self.predictions += ["agewt"]

    def _build_interpolator(self):
        """
        Build the RegularGridInterpolator for fast predictions.

        Creates a scipy RegularGridInterpolator object that enables fast
        multi-linear interpolation of stellar parameters across the
        4-dimensional grid (mini, eep, feh, afe).

        Notes
        -----
        Handles the special case where alpha enhancement has only one value
        by padding the grid dimension to enable interpolation. Uses linear
        interpolation with NaN fill values for out-of-bounds queries.

        The interpolator maps from (mini, eep, feh, afe) input coordinates
        to all predicted stellar parameters simultaneously.
        """
        # Set up grid dimensions
        self.grid_dims = np.append(
            [len(self.gridpoints[p]) for p in self.labels], self.output.shape[-1]
        )
        self.xgrid = tuple([self.gridpoints[lbl] for lbl in self.labels])

        # Initialize output grid
        self.ygrid = np.zeros(self.grid_dims) + np.nan

        # Fill grid using optimized indexing
        if len(self.X) > 0:
            indices = tuple(self.X.T)
            self.ygrid[indices] = self.output

        # Handle singular alpha enhancement dimension
        if self.grid_dims[-2] == 1:
            afe_val = self.xgrid[-1][0]
            xgrid = list(self.xgrid)
            xgrid[-1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)

            # Duplicate values in padded dimension
            self.grid_dims[-2] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, :, :, 0, :] = self.ygrid[:, :, :, 0, :]
            ygrid[:, :, :, 1, :] = self.ygrid[:, :, :, 0, :]
            self.ygrid = ygrid

        # Initialize interpolator
        self.interpolator = RegularGridInterpolator(
            self.xgrid,
            self.ygrid,
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
            if ndim == 1:
                dlogt, dlogr = corrs
                preds[self.logt_idx] += dlogt
                preds[self.logl_idx] += 2.0 * dlogr  # L ∝ R^2
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

        See Also
        --------
        get_predictions : Applies these corrections to stellar parameters

        Notes
        -----
        Corrections are applied as:

        .. math::
            \\Delta \\log T_{\\rm eff} = f_{\\rm EEP} \\cdot f_{\\rm [Fe/H]} \\cdot \\log(1 + \\Delta M \\cdot \\alpha_T)
            \\Delta \\log R = f_{\\rm EEP} \\cdot f_{\\rm [Fe/H]} \\cdot \\log(1 + \\Delta M \\cdot \\alpha_R)

        where :math:`\\Delta M = M_{\\rm ini} - 1.0` and the EEP factor smoothly
        transitions from 0 (pre-main sequence) to 1 (post-turnoff).

        Corrections are set to zero for stars with :math:`M_{\\rm ini} \\geq 1.0 M_\\odot`.
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

        # Set correction parameters
        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30.0, 0.5

        # Compute corrections with safeguards
        mass_offset = mini - 1.0
        eps = 1e-10
        temp_arg = np.maximum(1.0 + mass_offset * dtdm, eps)
        radius_arg = np.maximum(1.0 + mass_offset * drdm, eps)

        dlogt = np.log10(temp_arg)
        dlogr = np.log10(radius_arg)

        # EEP and metallicity dependence
        ecorr = 1.0 - 1.0 / (1.0 + np.exp(-(eep - 454.0) / msto_smooth))
        fcorr = np.exp(feh_scale * feh)

        dlogt *= ecorr * fcorr
        dlogr *= ecorr * fcorr

        # Zero corrections for solar mass and above
        if ndim == 1:
            if mini >= 1.0:
                dlogt, dlogr = 0.0, 0.0
        elif ndim == 2:
            mask = mini >= 1.0
            dlogt[mask] = 0.0
            dlogr[mask] = 0.0

        # Format output
        if ndim == 1:
            corrs = np.array([dlogt, dlogr])
        elif ndim == 2:
            corrs = np.c_[dlogt, dlogr]

        return corrs


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
        EEPTracks instance for stellar parameter predictions.

    filters : list of str, optional
        Filter names for photometry computation. If None, uses all available.

    nnfile : str, optional
        Path to neural network file for bolometric corrections.

    verbose : bool, optional
        Whether to output progress messages. Default is True.

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
    ...     mini=1.2, eep=400, feh=-0.2, afe=0.1,
    ...     smf=0.7, av=0.15, dist=1500.0
    ... )
    """

    def __init__(self, tracks, filters=None, nnfile=None, verbose=True):

        # Store tracks reference
        self.tracks = tracks

        # Set up filters
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters

        # Set default neural network file
        if nnfile is None:
            from ..data.loader import find_nn_file

            nnfile = find_nn_file()

        # Initialize neural network predictor
        try:
            self.predictor = FastNNPredictor(
                filters=filters, nnfile=nnfile, verbose=verbose
            )
        except Exception as e:
            if verbose:
                sys.stderr.write(
                    f"Warning: Neural network initialization failed: {e}\n"
                )
            self.predictor = None

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
        **kwargs,
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
            Default is 1e-2. The previous default (1e-6) was far tighter than
            the age emulator's own precision and silently dropped most
            otherwise-valid binary models.

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
            m = M + 5 \\log_{10}(d/10\\,{\\rm pc})

        where d is the distance in parsecs.

        Binary SEDs are combined using magnitude addition (flux summing):

        .. math::
            m_{\\rm combined} = -2.5 \\log_{10}(10^{-0.4 m_1} + 10^{-0.4 m_2})

        Examples
        --------
        Single star:

        >>> sed, params, params2 = star_track.get_seds(
        ...     mini=1.0, eep=350, feh=0.0, afe=0.0
        ... )

        Binary system:

        >>> sed, params, params2 = star_track.get_seds(
        ...     mini=1.2, eep=400, feh=-0.2, afe=0.1, smf=0.7
        ... )
        """
        if self.predictor is None:
            raise RuntimeError("Neural network predictor not available")

        # Grab input labels
        labels = {"mini": mini, "eep": eep, "feh": feh, "afe": afe}
        labels = np.array([labels[lbl] for lbl in self.tracks.labels])

        # Generate primary component predictions
        try:
            params_arr = self.tracks.get_predictions(
                labels, apply_corr=apply_corr, corr_params=corr_params
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate stellar parameters: {e}")

        # Convert to dictionary format
        params = dict(zip(self.tracks.predictions, params_arr))
        sed = np.full(self.predictor.NFILT, np.nan)
        sed2 = np.full(self.predictor.NFILT, np.nan)

        # Initialize secondary parameters
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.tracks.predictions, params_arr2))

        # Generate primary SED
        mini_min = max(getattr(self.tracks, "mini_bound", 0.08), mini_bound)
        loga = params["loga"]

        if loga <= loga_max:
            try:
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
            except Exception as e:
                warnings.warn(
                    f"Primary SED generation failed for (mini={mini}, eep={eep}): {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Add binary companion if requested
            if smf > 0.0 and eep <= eep_binary_max and mini * smf >= mini_min:
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

                    try:
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

                    except Exception as e:
                        warnings.warn(
                            f"Secondary SED generation failed for binary "
                            f"(mini={mini}, smf={smf}, eep2={eep2}): {e}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

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

    def _get_eep_for_secondary(self, loga, mini, eep, feh, afe, smf, tol):
        r"""
        Calculate EEP for secondary component that matches the age of the primary.

        This method solves the inverse problem: given a target age (from the primary),
        find the EEP that produces that age for the secondary star with mass mini*smf.
        Uses scipy.optimize.minimize with Nelder-Mead to find the best-fit EEP.

        Parameters
        ----------
        loga : float
            Target log10(age in years) to match from primary star
        mini : float
            Primary star initial mass in solar masses
        eep : float
            Primary star EEP (used as initial guess for optimization)
        feh : float
            Metallicity [Fe/H] in logarithmic solar units
        afe : float
            Alpha enhancement [α/Fe] in logarithmic solar units
        smf : float
            Secondary mass fraction (secondary mass = mini * smf)
        tol : float
            Convergence tolerance (in dex) on the log10(age) match. The
            squared-age-difference loss is accepted when ``sqrt(loss) < tol``
            (equivalently ``loss < tol**2``); otherwise NaN is returned.

        Returns
        -------
        eep2 : float
            EEP for secondary star that produces the target age.
            Returns NaN if optimization fails or doesn't converge within tolerance.

        See Also
        --------
        get_seds : Uses this method for binary star modeling
        EEPTracks.get_predictions : Called to evaluate ages at different EEPs

        Notes
        -----
        The optimization minimizes the squared age difference:

        .. math::
            L(EEP_2) = (\\log_{10} age(M_2, EEP_2) - \\log_{10} age_{\\rm target})^2

        where :math:`M_2 = M_1 \\times smf` is the secondary mass.

        The alpha enhancement parameter is currently fixed to solar ([α/Fe] = 0.0)
        for secondary stars, regardless of the primary's value. This is intentional
        because the current MIST model grids do not include α-enhanced tracks.
        The `afe` parameter is accepted for API consistency but not used.
        """
        # Get age index from tracks
        aidx = self.tracks.predictions.index("loga")

        # Define loss function: minimize difference between predicted and target age
        def loss(x):
            if isinstance(x, np.ndarray) and x.size == 1:
                x = x[0]
            # Get predicted age for secondary star with mass mini*smf at EEP x
            try:
                loga_pred = self.tracks.get_predictions([mini * smf, x, feh, 0.0])[aidx]
                return (loga_pred - loga) ** 2
            except Exception:
                # Return large loss if prediction fails
                return 1e6

        # Find best-fit EEP that minimizes age difference
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore bad values during optimization
                res = minimize(loss, eep, method="Nelder-Mead")

            # Check if solution meets tolerance. loss is the squared log-age
            # difference, so this accepts matches with |dloga| < tol (in dex).
            if res.fun < tol**2:
                eep2 = res.x[0]
            else:
                eep2 = np.nan

        except Exception as e:
            warnings.warn(
                f"EEP optimization failed for secondary star "
                f"(mini={mini}, smf={smf}): {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            eep2 = np.nan

        return eep2


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
    models : numpy.ndarray of shape (Nmodel, Nfilt, Ncoef) or dict/h5py file
        Pre-computed model grid containing photometric coefficients.
        If dict or h5py file, should contain 'mag_coeffs' key.
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
        Filter names for photometry. If None, uses all available filters
        from the models.

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
            # Dictionary input (e.g., from h5py file)
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
                # Extract filter names and reshape
                filter_names = list(models.dtype.names)
                if filters is not None:
                    # Filter to requested filters
                    filter_names = [f for f in filter_names if f in filters]

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

        # Initialize KD-tree placeholder (built on first use)
        self.kdtree = None
        self._kdtree_labels = None  # Track which labels are in KD-tree

        if verbose:
            print(
                f"Loaded StarGrid with {self.nmodels:,} models, "
                f"{self.nfilters} filters, {len(self.label_names)} labels"
            )
            if self.param_names:
                print(f"Additional parameters: {', '.join(self.param_names)}")

    def _build_grid_indices(self):
        """Build indices for efficient grid lookup and interpolation."""
        # Create unique value arrays for each label dimension
        self.grid_axes = {}
        self.grid_shape = []

        for label in self.label_names:
            if label in ["mini", "eep", "feh", "afe", "smf"]:
                unique_vals = np.unique(self.labels[label])
                self.grid_axes[label] = unique_vals
                self.grid_shape.append(len(unique_vals))

        # Create mapping from label values to grid indices
        self.label_to_idx = {}
        for label, values in self.grid_axes.items():
            self.label_to_idx[label] = {val: idx for idx, val in enumerate(values)}

    def _build_kdtree(self, **kwargs):
        """
        Build KD-tree for efficient nearest neighbor queries.

        Only built on first use to avoid overhead if only using via BruteForce.
        """
        if self.kdtree is not None:
            return  # Already built

        from scipy.spatial import cKDTree

        # Determine which labels to use for KD-tree
        active_labels = []
        for label in ["mini", "eep", "feh", "afe", "smf"]:
            if (
                label in self.label_names
                and label in kwargs
                and kwargs[label] is not None
            ):
                active_labels.append(label)

        if not active_labels:
            # Use all available labels
            active_labels = [
                lbl
                for lbl in ["mini", "eep", "feh", "afe", "smf"]
                if lbl in self.label_names
            ]

        # Build normalized coordinates for KD-tree
        coords = []
        for label in active_labels:
            vals = self.labels[label]
            # Normalize to [0, 1] for balanced distance metrics
            val_min, val_max = vals.min(), vals.max()
            if val_max > val_min:
                normalized = (vals - val_min) / (val_max - val_min)
            else:
                normalized = np.zeros_like(vals)
            coords.append(normalized)

        if coords:
            self.kdtree = cKDTree(np.column_stack(coords))
            self._kdtree_labels = active_labels

    def _find_neighbors_multilinear(self, **kwargs):
        """
        Find bracketing grid points and compute multi-linear interpolation weights.

        For each dimension, finds the two bracketing values and computes
        interpolation weights. This gives us 2^N neighbors for N dimensions.

        Parameters
        ----------
        **kwargs : keyword arguments
            Stellar parameters (mini, eep, feh, afe, smf)

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

        Falls back to KD-tree method if grid structure is irregular or
        interpolation fails.
        """
        import itertools

        # Get requested parameters
        req_params = {}
        for key in ["mini", "eep", "feh", "afe", "smf"]:
            if key in kwargs and kwargs[key] is not None:
                req_params[key] = kwargs[key]

        if not req_params:
            # No parameters specified, return first model with weight 1
            return np.array([0]), np.array([1.0])

        # For each parameter, find bracketing indices and weights
        bracket_info = {}
        for param, value in req_params.items():
            if param in self.grid_axes:
                axis_values = self.grid_axes[param]

                # Find bracketing indices using searchsorted
                idx = np.searchsorted(axis_values, value)

                if idx == 0:
                    # Before first point
                    idx_low = idx_high = 0
                    weight_high = 1.0
                elif idx >= len(axis_values):
                    # After last point
                    idx_low = idx_high = len(axis_values) - 1
                    weight_high = 1.0
                else:
                    # Between points
                    idx_low = idx - 1
                    idx_high = idx
                    # Linear interpolation weight
                    val_low = axis_values[idx_low]
                    val_high = axis_values[idx_high]
                    if val_high > val_low:
                        weight_high = (value - val_low) / (val_high - val_low)
                    else:
                        weight_high = 0.5

                bracket_info[param] = {
                    "indices": (
                        [idx_low, idx_high] if idx_low != idx_high else [idx_low]
                    ),
                    "weights": (
                        [1.0 - weight_high, weight_high]
                        if idx_low != idx_high
                        else [1.0]
                    ),
                    "values": (
                        axis_values[[idx_low, idx_high]]
                        if idx_low != idx_high
                        else axis_values[[idx_low]]
                    ),
                }

        # Generate all combinations of bracket points
        param_names = list(bracket_info.keys())
        index_combinations = itertools.product(
            *[bracket_info[p]["indices"] for p in param_names]
        )
        weight_combinations = itertools.product(
            *[bracket_info[p]["weights"] for p in param_names]
        )

        # Find actual grid indices for each combination
        indices = []
        weights = []

        for idx_combo, wt_combo in zip(index_combinations, weight_combinations):
            # Build selection criteria
            sel = np.ones(self.nmodels, dtype=bool)
            for param_name, param_idx in zip(param_names, idx_combo):
                param_val = bracket_info[param_name]["values"][
                    bracket_info[param_name]["indices"].index(param_idx)
                ]
                sel &= self.labels[param_name] == param_val

            # Handle other parameters not being interpolated
            for param in self.label_names:
                if param not in req_params and param in [
                    "mini",
                    "eep",
                    "feh",
                    "afe",
                    "smf",
                ]:
                    # Use first available value for unspecified parameters
                    if param in self.grid_axes:
                        sel &= self.labels[param] == self.grid_axes[param][0]

            # Find matching grid point
            grid_idx = np.where(sel)[0]
            if len(grid_idx) > 0:
                indices.append(grid_idx[0])
                # Weight is product of all dimension weights
                weights.append(np.prod(wt_combo))

        if not indices:
            # Fallback to KD-tree nearest neighbor if multi-linear fails
            return self._find_neighbors_kdtree(**kwargs)

        # Normalize weights
        indices = np.array(indices)
        weights = np.array(weights)
        weights /= weights.sum()

        return indices, weights

    def _find_neighbors_kdtree(self, **kwargs):
        """
        Find nearest neighbors using KD-tree (fallback method).

        Parameters
        ----------
        **kwargs : keyword arguments
            Stellar parameters (mini, eep, feh, afe, smf)

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
        # Build KD-tree on first use
        self._build_kdtree(**kwargs)

        if self.kdtree is None:
            # KD-tree couldn't be built, use simple nearest neighbor
            distances = np.zeros(self.nmodels)
            for label in ["mini", "eep", "feh", "afe", "smf"]:
                if (
                    label in kwargs
                    and kwargs[label] is not None
                    and label in self.label_names
                ):
                    label_vals = self.labels[label]
                    val_range = label_vals.max() - label_vals.min()
                    if val_range > 0:
                        distances += ((label_vals - kwargs[label]) / val_range) ** 2

            nearest_idx = np.argmin(distances)
            return np.array([nearest_idx]), np.array([1.0])

        # Build query point
        query_point = []
        for label in self._kdtree_labels:
            if label in kwargs and kwargs[label] is not None:
                val = kwargs[label]
            else:
                val = self.grid_axes[label][0] if label in self.grid_axes else 0

            # Normalize
            vals = self.labels[label]
            val_min, val_max = vals.min(), vals.max()
            if val_max > val_min:
                normalized = (val - val_min) / (val_max - val_min)
            else:
                normalized = 0.0
            query_point.append(normalized)

        # Query KD-tree for nearest neighbors
        k = min(8, self.nmodels)  # Use up to 8 neighbors
        distances, indices = self.kdtree.query(query_point, k=k)

        # Convert distances to weights (inverse distance weighting)
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights /= weights.sum()

        return indices, weights

    def get_predictions(
        self,
        mini=None,
        eep=None,
        feh=None,
        afe=None,
        smf=None,
        use_multilinear=True,
        **kwargs,
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

        **kwargs : additional parameters
            Any additional selection criteria

        Returns
        -------
        predictions : dict or numpy.ndarray
            Predicted stellar parameters. Returns dict with parameter names
            as keys if parameters are available, otherwise returns array.

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
        # Find neighboring grid points
        if use_multilinear:
            indices, weights = self._find_neighbors_multilinear(
                mini=mini, eep=eep, feh=feh, afe=afe, smf=smf, **kwargs
            )
        else:
            indices, weights = self._find_neighbors_kdtree(
                mini=mini, eep=eep, feh=feh, afe=afe, smf=smf, **kwargs
            )

        # Interpolate parameters if available
        if self.params is not None and self.param_names:
            predictions = {}

            # Add input labels to predictions
            for label in ["mini", "eep", "feh", "afe", "smf"]:
                value = locals()[label]
                if label in self.label_names and value is not None:
                    predictions[label] = value

            # Interpolate each parameter
            for param in self.param_names:
                param_vals = self.params[param][indices]
                predictions[param] = np.sum(param_vals * weights)

            return predictions

        else:
            # Return weighted average of labels only
            predictions = {}
            for label in self.label_names:
                label_vals = self.labels[label][indices]
                predictions[label] = np.sum(label_vals * weights)

            return predictions

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
        **kwargs,
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

        **kwargs : additional parameters
            Passed to interpolation methods

        Returns
        -------
        sed : numpy.ndarray of shape (Nfilt,)
            Synthetic photometry (magnitudes or fluxes)

        params : dict or numpy.ndarray or None
            Primary star parameters (None if return_predictions=False)

        params2 : dict or numpy.ndarray or None
            Secondary star parameters (empty placeholder - full binary
            implementation requires dedicated binary grid)

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
            m(\\lambda) = m_0(\\lambda) + A_V \\cdot [r_0(\\lambda) + R_V \\cdot dr(\\lambda)]

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
        # Warning for binary support limitations
        if smf is not None and smf > 0:
            import warnings

            warnings.warn(
                "Binary star support in StarGrid is limited. Secondary parameters "
                "(params2) will be empty. For full binary modeling, use StarEvolTrack "
                "or wait for binary grid implementation.",
                UserWarning,
                stacklevel=2,
            )

        # Find neighboring grid points using chosen interpolation method
        if use_multilinear:
            indices, weights = self._find_neighbors_multilinear(
                mini=mini, eep=eep, feh=feh, afe=afe, smf=smf, **kwargs
            )
        else:
            indices, weights = self._find_neighbors_kdtree(
                mini=mini, eep=eep, feh=feh, afe=afe, smf=smf, **kwargs
            )

        # Get weighted average of magnitude coefficients
        weighted_coeffs = np.zeros((self.nfilters, 3))
        for idx, weight in zip(indices, weights):
            weighted_coeffs += self.models[idx] * weight

        # Apply extinction and distance modulus using sed_utils
        from .sed_utils import _get_seds

        # Get reddened magnitudes
        # _get_seds expects (Nmodels, Nbands, Ncoef) and arrays for av/rv
        weighted_coeffs_3d = weighted_coeffs[np.newaxis, :, :]
        av_array = np.array([av])
        rv_array = np.array([rv])
        seds_array, _, _ = _get_seds(
            weighted_coeffs_3d, av_array, rv_array, return_flux=False
        )
        mags = seds_array[0]  # Extract the single model result

        # Apply distance modulus (grid magnitudes stored at 1 kpc = 1000 pc reference)
        # Note: Grid files are generated at 1 kpc distance for consistency with
        # Gaia parallax measurements (1 mas = 1 kpc)
        if dist is not None and dist != 1000.0:  # Grid reference is 1 kpc
            dist_mod = 5.0 * np.log10(dist / 1000.0)
            mags += dist_mod

        # Convert to flux if requested
        if return_flux:
            from ..utils.photometry import inv_magnitude

            # Use zero errors for now (could be improved with proper error propagation)
            sed, _ = inv_magnitude(mags, np.zeros_like(mags))
        else:
            sed = mags

        # Get parameter predictions using same neighbors (avoid redundant computation)
        if return_predictions:
            # Interpolate parameters using already-found neighbors
            if self.params is not None and self.param_names:
                params = {}

                # Add input labels
                for label in ["mini", "eep", "feh", "afe", "smf"]:
                    value = locals()[label]
                    if label in self.label_names and value is not None:
                        params[label] = value

                # Interpolate parameters
                for param in self.param_names:
                    param_vals = self.params[param][indices]
                    params[param] = np.sum(param_vals * weights)
            else:
                # Interpolate labels only
                params = {}
                for label in self.label_names:
                    label_vals = self.labels[label][indices]
                    params[label] = np.sum(label_vals * weights)
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
