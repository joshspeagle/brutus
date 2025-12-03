#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stellar population modeling and synthetic photometry generation.

This module provides classes for modeling stellar populations using MIST
(MESA Isochrones and Stellar Tracks) isochrones and generating synthetic
photometry with neural network-based bolometric corrections.

The module follows a clean separation of concerns:
- Isochrone: Stellar parameter predictions for populations
- StellarPop: SED/photometry generation for populations

This design is consistent with the individual star modeling pattern:
- EEPtracks: Stellar parameter predictions for individuals
- StarEvolTrack: SED/photometry generation for individuals

Classes
-------
Isochrone : Stellar population parameter predictions
    Interpolates MIST isochrones to predict stellar parameters (mass, age,
    temperature, etc.) as a function of metallicity, alpha enhancement,
    age, and evolutionary phase.

StellarPop : Stellar population photometry synthesis
    Generates synthetic photometry for stellar populations using neural
    network bolometric corrections, with support for binary stars, dust
    extinction, and complex evolutionary modeling.

Examples
--------
Basic stellar population modeling:

>>> from brutus.core.populations import Isochrone, StellarPop
>>>
>>> # Create isochrone for parameter predictions
>>> iso = Isochrone()
>>> params = iso.get_predictions(feh=0.0, afe=0.0, loga=9.0)
>>>
>>> # Create population synthesizer for photometry
>>> pop_synth = StellarPop(isochrone=iso)
>>> seds, params, params2 = pop_synth.synthesize(
...     feh=0.0, afe=0.0, loga=9.0,
...     av=0.1, dist=1000.0
... )

Advanced usage with binary populations:

>>> # Model binary stellar populations
>>> seds, params, params2 = pop_synth.synthesize(
...     feh=-0.5, afe=0.3, loga=10.0,
...     binary_fraction=0.4,  # 40% mass ratio binaries
...     av=0.2, dist=2000.0
... )

Notes
-----
This design provides several advantages over the original combined approach:

1. **Separation of Concerns**: Parameter prediction vs. photometry synthesis
2. **Flexibility**: Can use different SED generators with same isochrone
3. **Consistency**: Matches the individual star modeling pattern
4. **Maintainability**: Cleaner, more focused class responsibilities
5. **Testability**: Each component can be tested independently

The StellarPop class uses dependency injection, accepting an
Isochrone instance rather than inheriting from it. This makes the code
more modular and allows for different isochrone implementations.

This implementation is based on the MIST stellar evolution framework [1]_ [2]_.

References
----------
.. [1] Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
       for the Construction of Stellar Isochrones", ApJ, 823, 102
.. [2] Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
       Models", ApJS, 222, 8
"""

import sys
import warnings
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Import filter definitions
from ..data.filters import FILTERS

# Import from utils (to be reorganized)
from ..utils.photometry import add_mag

# Import neural network predictor
from .neural_nets import FastNNPredictor

__all__ = ["Isochrone", "StellarPop"]


class Isochrone(object):
    """
    Stellar parameter predictions for isochrones using MIST evolutionary models.

    This class provides interpolation of stellar parameters along isochrones of
    fixed metallicity, alpha enhancement, and age. It focuses solely on stellar
    parameter prediction (masses, temperatures, luminosities, etc.) without
    photometry generation.

    This class is analogous to MISTtracks but for stellar populations:
    - MISTtracks: Individual star parameters (mini, EEP, feh, afe) → parameters
    - Isochrone: Population parameters (feh, afe, loga, EEP) → parameters

    For photometry generation, use StellarPop with this class.

    Parameters
    ----------
    mistfile : str, optional
        Path to the HDF5 file containing the MIST isochrone grid. If not
        provided, defaults to the standard MIST v1.2 isochrone file.

    predictions : list of str, optional
        The names of stellar parameters to predict. Default is:
        `["mini", "mass", "logl", "logt", "logr", "logg", "feh_surf", "afe_surf"]`.

    verbose : bool, optional
        Whether to output progress messages during initialization. Default is `True`.

    Attributes
    ----------
    predictions : list of str
        List of stellar parameter names for prediction

    feh_grid, afe_grid, loga_grid, eep_grid : numpy.ndarray
        Input parameter grids from MIST isochrone file

    pred_grid : numpy.ndarray
        Stellar parameter predictions organized by grid coordinates

    interpolator : scipy.interpolate.RegularGridInterpolator
        Main interpolation object for stellar parameter prediction

    Examples
    --------
    Generate stellar parameters for a solar metallicity, 1 Gyr population:

    >>> iso = Isochrone()
    >>> params = iso.get_predictions(feh=0.0, afe=0.0, loga=9.0)
    >>> masses = params[:, 0]  # Initial masses
    >>> log_teff = params[:, 3]  # log(effective temperatures)
    >>> print(f"Mass range: {masses.min():.2f} - {masses.max():.2f} M_sun")

    Generate parameters for specific evolutionary phases:

    >>> import numpy as np
    >>> eep_range = np.arange(200, 500, 50)  # Main sequence to turnoff
    >>> params = iso.get_predictions(feh=-0.5, afe=0.3, loga=10.0, eep=eep_range)
    """

    def __init__(self, mistfile=None, predictions=None, verbose=True):

        # Set default file path
        if mistfile is None:
            package_root = Path(__file__).parent.parent.parent.parent
            mistfile = package_root / "data" / "DATAFILES" / "MIST_1.2_iso_vvcrit0.0.h5"

        # Set default predictions
        if predictions is None:
            predictions = [
                "mini",
                "mass",
                "logl",
                "logt",
                "logr",
                "logg",
                "feh_surf",
                "afe_surf",
            ]
        self.predictions = predictions

        if verbose:
            sys.stderr.write("Constructing MIST isochrones...\n")

        # Load isochrone data
        try:
            with h5py.File(mistfile, "r") as f:
                self.feh_grid = f["feh"][:]
                self.afe_grid = f["afe"][:]
                self.loga_grid = f["loga"][:]
                self.eep_grid = f["eep"][:]
                self.pred_grid = f["predictions"][:]
                self.pred_labels = f["predictions"].attrs["labels"]
        except (OSError, KeyError) as e:
            raise RuntimeError(f"Failed to load isochrone data from {mistfile}: {e}")

        # Initialize interpolator
        self._build_interpolator()

        if verbose:
            sys.stderr.write("done!\n")

    def _build_interpolator(self):
        """
        Construct the RegularGridInterpolator for stellar parameter prediction.

        This method processes the MIST isochrone grid and creates the interpolation
        machinery for fast multi-dimensional interpolation. It includes gap filling
        along the EEP dimension and handles singular alpha enhancement dimensions.

        Notes
        -----
        The interpolator operates on a 4D grid (feh, afe, loga, eep) and
        returns predictions for all stellar parameters simultaneously.

        Gap filling is performed by linear interpolation along the EEP
        dimension for each (feh, afe, loga) combination where data exists.

        If alpha enhancement has only one value, the grid is padded with
        duplicate values to enable scipy interpolation.
        """
        # Set up coordinate grids
        self.feh_u = np.unique(self.feh_grid)
        self.afe_u = np.unique(self.afe_grid)
        self.loga_u = np.unique(self.loga_grid)
        self.eep_u = np.unique(self.eep_grid)
        self.xgrid = (self.feh_u, self.afe_u, self.loga_u, self.eep_u)

        # Determine grid dimensions
        self.grid_dims = np.array(
            [
                len(self.xgrid[0]),  # feh
                len(self.xgrid[1]),  # afe
                len(self.xgrid[2]),  # loga
                len(self.xgrid[3]),  # eep
                len(self.pred_labels),  # predictions
            ],
            dtype="int",
        )

        # Fill gaps in the grid where possible
        for i in range(len(self.feh_u)):
            for j in range(len(self.afe_u)):
                for k in range(len(self.loga_u)):
                    # Select EEP values where predictions exist
                    sel = np.all(np.isfinite(self.pred_grid[i, j, k]), axis=1)

                    if np.sum(sel) > 1:  # Need at least 2 points
                        try:
                            # Linearly interpolate over EEP grid
                            pnew = [
                                np.interp(
                                    self.eep_u,
                                    self.eep_u[sel],
                                    par,
                                    left=np.nan,
                                    right=np.nan,
                                )
                                for par in self.pred_grid[i, j, k, sel].T
                            ]
                            self.pred_grid[i, j, k] = np.array(pnew).T
                        except Exception as e:
                            warnings.warn(
                                f"Gap filling failed for grid cell "
                                f"(feh={self.feh_u[i]}, afe={self.afe_u[j]}, "
                                f"loga={self.loga_u[k]}): {e}",
                                RuntimeWarning,
                                stacklevel=2,
                            )

        # Handle singular alpha enhancement dimension
        if self.grid_dims[1] == 1:
            afe_val = self.xgrid[1][0]
            xgrid = list(self.xgrid)
            xgrid[1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)

            # Duplicate values in padded dimension
            self.grid_dims[1] += 1
            ygrid = np.empty(self.grid_dims)
            ygrid[:, 0, :, :, :] = self.pred_grid[:, 0, :, :, :]
            ygrid[:, 1, :, :, :] = self.pred_grid[:, 0, :, :, :]
            self.pred_grid = ygrid

        # Initialize interpolator
        self.interpolator = RegularGridInterpolator(
            self.xgrid,
            self.pred_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def get_predictions(
        self, feh=0.0, afe=0.0, loga=8.5, eep=None, apply_corr=True, corr_params=None
    ):
        """
        Generate stellar parameter predictions for an isochrone.

        Parameters
        ----------
        feh : float, optional
            Metallicity [Fe/H] relative to solar. Default is 0.0.

        afe : float, optional
            Alpha enhancement [alpha/Fe] relative to solar. Default is 0.0.

        loga : float, optional
            Log10(age in years). Default is 8.5 (≈316 Myr).

        eep : array-like, optional
            Equivalent evolutionary points. If None, uses default grid.

        apply_corr : bool, optional
            Whether to apply empirical corrections. Default is True.

        corr_params : tuple, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale).

        Returns
        -------
        preds : numpy.ndarray of shape (Neep, Npred)
            Stellar parameter predictions for each EEP value. The columns
            correspond to the parameters listed in `self.pred_labels`.

        See Also
        --------
        StellarPop.synthesize : Uses these predictions for photometry generation
        brutus.core.EEPTracks.get_predictions : Similar function for individual stars

        Notes
        -----
        The method interpolates across the pre-computed MIST isochrone grid.
        Returns NaN for parameters outside the grid bounds or where data is
        not available.

        Common EEP ranges:
        - Pre-main sequence: EEP < 202
        - Zero-age main sequence: EEP = 202
        - Terminal-age main sequence: EEP = 454
        - Subgiant branch: EEP 454-605
        - Red giant branch: EEP 605-1409

        Examples
        --------
        >>> iso = Isochrone()
        >>> # Get parameters for a 1 Gyr solar metallicity population
        >>> params = iso.get_predictions(feh=0.0, afe=0.0, loga=9.0)
        >>> masses = params[:, 0]  # Initial masses
        >>> temps = 10**params[:, 3]  # Effective temperatures
        """
        # Set default EEP grid
        if eep is None:
            eep = self.eep_u
        eep = np.array(eep, dtype=float)

        # Create input array for interpolation
        feh_arr = np.full_like(eep, feh)
        afe_arr = np.full_like(eep, afe)
        loga_arr = np.full_like(eep, loga)
        labels = np.c_[feh_arr, afe_arr, loga_arr, eep]

        # Generate predictions
        try:
            preds = self.interpolator(labels)
        except Exception as e:
            raise RuntimeError(f"Interpolation failed: {e}")

        # Apply empirical corrections if requested
        if apply_corr and hasattr(self, "_apply_corrections"):
            try:
                # Extract parameters needed for corrections
                mini_idx = list(self.pred_labels).index("mini")
                logt_idx = list(self.pred_labels).index("logt")
                logl_idx = list(self.pred_labels).index("logl")
                logg_idx = list(self.pred_labels).index("logg")

                mini = preds[:, mini_idx]

                # Apply corrections
                corrs = self._apply_corrections(
                    mini=mini, feh=feh_arr, eep=eep, corr_params=corr_params
                )

                dlogt, dlogr = corrs.T
                preds[:, logt_idx] += dlogt
                preds[:, logl_idx] += 2.0 * dlogr  # L ∝ R^2
                preds[:, logg_idx] -= 2.0 * dlogr  # g ∝ M/R^2

            except Exception as e:
                warnings.warn(f"Correction application failed: {e}")

        return preds

    def _apply_corrections(self, mini, feh, eep, corr_params=None):
        """
        Apply empirical corrections to stellar parameters.

        Computes corrections to effective temperature and radius based on
        stellar mass, evolutionary phase, and metallicity. This method
        parallels the corrections in EEPTracks but adapted for isochrone data.

        Parameters
        ----------
        mini : array_like
            Initial stellar masses in solar masses
        feh : array_like
            Metallicities [Fe/H]
        eep : array_like
            Equivalent evolutionary points
        corr_params : tuple, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale).
            Default is (0.09, -0.09, 30.0, 0.5).

        Returns
        -------
        corrs : numpy.ndarray of shape (N, 2)
            Corrections to [log(Teff), log(R)]

        See Also
        --------
        brutus.core.EEPTracks.get_corrections : Individual star corrections
        get_predictions : Applies these corrections to predictions
        """
        if corr_params is not None:
            dtdm, drdm, msto_smooth, feh_scale = corr_params
        else:
            dtdm, drdm, msto_smooth, feh_scale = 0.09, -0.09, 30.0, 0.5

        # Safeguarded corrections computation
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
        mask = mini >= 1.0
        dlogt[mask] = 0.0
        dlogr[mask] = 0.0

        return np.c_[dlogt, dlogr]


class StellarPop(object):
    """
    Synthetic photometry generation for stellar populations.

    This class generates synthetic SEDs and photometry for stellar populations
    using neural network-based bolometric corrections. It provides sophisticated
    modeling of binary stars, dust extinction, and observational effects.

    This class is analogous to StarEvolTrack but for stellar populations:
    - StarEvolTrack: Individual star photometry
    - StellarPop: Stellar population photometry

    The class uses dependency injection, accepting an Isochrone instance for
    stellar parameter predictions rather than inheriting from it.

    Parameters
    ----------
    isochrone : Isochrone
        Isochrone instance for stellar parameter predictions.

    filters : list of str, optional
        Filter names for photometry computation. If None, uses all available.

    nnfile : str, optional
        Path to neural network file for bolometric corrections.

    verbose : bool, optional
        Whether to output progress messages. Default is True.

    Attributes
    ----------
    isochrone : Isochrone
        The isochrone model used for stellar parameter predictions

    filters : numpy.ndarray
        Array of filter names

    predictor : FastNNPredictor
        Neural network predictor for photometry

    Examples
    --------
    Basic stellar population photometry:

    >>> iso = Isochrone()
    >>> ssp = StellarPop(isochrone=iso)
    >>>
    >>> # Generate SEDs for solar metallicity, 1 Gyr population
    >>> seds, params, params2 = ssp.get_seds(
    ...     feh=0.0, afe=0.0, loga=9.0,
    ...     av=0.1, rv=3.1, dist=1000.0
    ... )

    Binary population modeling:

    >>> # Model population with 40% mass ratio binaries
    >>> seds, params, params2 = ssp.get_seds(
    ...     feh=-0.5, afe=0.3, loga=10.0,
    ...     binary_fraction=0.4,
    ...     av=0.2, dist=2000.0
    ... )

    See Also
    --------
    Isochrone : Stellar parameter predictions used by this class
    StarEvolTrack : Individual star analog
    brutus.core.neural_nets.FastNNPredictor : Neural network SED generation

    Notes
    -----
    The synthetic photometry generation includes:

    1. **Stellar Parameter Prediction**: Uses the injected Isochrone instance
    2. **Neural Network Photometry**: Fast bolometric correction computation
    3. **Binary Star Modeling**: Sophisticated binary population synthesis
    4. **Dust Extinction**: Parameterized extinction laws
    5. **Distance Effects**: Distance modulus calculations

    Binary modeling includes mass ratio constraints, evolutionary phase matching,
    and realistic limits on binary evolution (typically restricted to main
    sequence phases).
    """

    def __init__(self, isochrone, filters=None, nnfile=None, verbose=True):

        # Store isochrone reference
        self.isochrone = isochrone

        # Set up filters
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters

        # Set default neural network file
        if nnfile is None:
            import os

            package_root = Path(__file__).parent.parent.parent.parent

            # Try multiple possible names (nn_c3k.h5 is downloaded by pooch,
            # nnMIST_BC.h5 is legacy name - they are the same file)
            possible_names = ["nn_c3k.h5", "nnMIST_BC.h5"]

            for nn_name in possible_names:
                # Check local data directory first
                nnfile = package_root / "data" / "DATAFILES" / nn_name
                if os.path.exists(str(nnfile)):
                    break

                # If not found locally, try pooch cache directory
                import pooch

                cache_dir = Path(pooch.os_cache("astro-brutus"))
                cache_path = cache_dir / nn_name
                if os.path.exists(str(cache_path)):
                    nnfile = cache_path
                    break

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
        feh=0.0,
        afe=0.0,
        loga=8.5,
        eep=None,
        av=0.0,
        rv=3.3,
        binary_fraction=0.0,
        dist=1000.0,
        mini_bound=0.5,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
        return_dict=True,
        **kwargs,
    ):
        """
        Generate synthetic photometry for a stellar population.

        Parameters
        ----------
        feh : float, optional
            Metallicity [Fe/H]. Default is 0.0.

        afe : float, optional
            Alpha enhancement [alpha/Fe]. Default is 0.0.

        loga : float, optional
            Log10(age in years). Default is 8.5.

        eep : array-like, optional
            Equivalent evolutionary points. If None, uses isochrone default.

        av : float, optional
            V-band extinction A(V). Default is 0.0.

        rv : float, optional
            Extinction law parameter R(V). Default is 3.3.

        binary_fraction : float, optional
            Secondary mass fraction for binaries. Default is 0.0 (no binaries).
            - 0.0: No binaries
            - 0 < binary_fraction < 1: Mass ratio binaries
            - 1.0: Equal mass binaries

        dist : float, optional
            Distance in parsecs. Default is 1000.0.

        mini_bound : float, optional
            Minimum mass for SED computation. Default is 0.5.

        eep_binary_max : float, optional
            Maximum EEP for binary modeling. Default is 480.0.

        apply_corr : bool, optional
            Apply empirical corrections. Default is True.

        corr_params : tuple, optional
            Correction parameters.

        return_dict : bool, optional
            Return parameters as dictionaries. Default is True.

        Returns
        -------
        seds : numpy.ndarray of shape (Neep, Nfilters)
            Synthetic SEDs in magnitudes.

        params : dict or numpy.ndarray
            Primary component stellar parameters.

        params2 : dict or numpy.ndarray
            Secondary component parameters (for binaries).

        See Also
        --------
        Isochrone.get_predictions : Stellar parameter predictions
        FastNNPredictor.sed : Neural network photometry generation
        StarEvolTrack.get_seds : Individual star analog

        Notes
        -----
        The photometry generation workflow:

        1. Predict stellar parameters from isochrone
        2. Generate primary SEDs using neural network
        3. For binaries: generate secondary SEDs and combine
        4. Apply dust extinction with specified R(V)
        5. Apply distance modulus

        Binary stars are modeled with the same age and metallicity as
        primaries, with masses determined by the binary_fraction parameter.
        Binary companions are only added for stars below eep_binary_max
        (typically restricted to main sequence).

        Examples
        --------
        Simple stellar population:

        >>> seds, params, params2 = pop_synth.synthesize(
        ...     feh=0.0, loga=9.0, av=0.1, dist=1000.0
        ... )

        Binary population:

        >>> seds, params, params2 = pop_synth.synthesize(
        ...     feh=-0.5, loga=10.0, binary_fraction=0.4,
        ...     av=0.2, dist=2000.0
        ... )
        """
        if self.predictor is None:
            raise RuntimeError("Neural network predictor not available")

        # Generate stellar parameters using isochrone
        try:
            params_arr = self.isochrone.get_predictions(
                feh=feh,
                afe=afe,
                loga=loga,
                eep=eep,
                apply_corr=apply_corr,
                corr_params=corr_params,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate stellar parameters: {e}")

        # Convert to dictionary format
        params = dict(zip(self.isochrone.predictions, params_arr.T))
        Neep = len(params_arr)

        # Initialize SED array
        seds = np.full((Neep, self.predictor.NFILT), np.nan)

        # Generate primary component SEDs
        for i in range(Neep):
            if params["mini"][i] >= mini_bound:
                try:
                    seds[i] = self.predictor.sed(
                        logl=params["logl"][i],
                        logt=params["logt"][i],
                        logg=params["logg"][i],
                        feh_surf=params["feh_surf"][i],
                        afe=params["afe_surf"][i],
                        av=av,
                        rv=rv,
                        dist=dist,
                    )
                except Exception as e:
                    warnings.warn(
                        f"Primary SED generation failed for EEP index {i} "
                        f"(mini={params['mini'][i]:.3f}): {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Initialize secondary parameters
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.isochrone.predictions, params_arr2.T))

        # Handle binary stars
        if 0.0 < binary_fraction <= 1.0:
            self._add_binary_components(
                seds,
                params,
                params2,
                params_arr,
                params_arr2,
                binary_fraction,
                feh,
                afe,
                loga,
                mini_bound,
                eep_binary_max,
                av,
                rv,
                dist,
                apply_corr,
                corr_params,
            )
        elif binary_fraction == 1.0:
            # Equal mass binaries
            seds[self.isochrone.eep_u <= eep_binary_max] -= 2.5 * np.log10(2.0)
            params2 = deepcopy(params)
            params_arr2 = deepcopy(params_arr.T)

        # Format output
        if not return_dict:
            params = params_arr.T
            params2 = params_arr2

        return seds, params, params2

    def _add_binary_components(
        self,
        seds,
        params,
        params2,
        params_arr,
        params_arr2,
        binary_fraction,
        feh,
        afe,
        loga,
        mini_bound,
        eep_binary_max,
        av,
        rv,
        dist,
        apply_corr,
        corr_params,
    ):
        """Add binary star components to the population synthesis."""
        # Calculate secondary masses and EEPs
        mini = params["mini"]
        mini2 = mini * binary_fraction
        eep = self.isochrone.eep_u

        # Interpolate secondary EEPs
        mini_mask = np.where(np.isfinite(mini))[0]
        if len(mini_mask) > 0:
            try:
                eep2 = np.interp(
                    mini2, mini[mini_mask], eep[mini_mask], left=np.nan, right=np.nan
                )
            except Exception:
                eep2 = np.full_like(eep, np.nan)
        else:
            eep2 = np.full_like(eep, np.nan)

        # Restrict binaries to main sequence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eep2[(eep2 > eep_binary_max) | (eep > eep_binary_max)] = np.nan

        # Generate secondary parameters
        try:
            params_arr2[:] = self.isochrone.get_predictions(
                feh=feh,
                afe=afe,
                loga=loga,
                eep=eep2,
                apply_corr=apply_corr,
                corr_params=corr_params,
            ).T
            params2.update(dict(zip(self.isochrone.predictions, params_arr2)))
        except Exception as e:
            warnings.warn(
                f"Secondary parameter generation failed for binary population: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Generate secondary SEDs
        seds2 = np.full_like(seds, np.nan)
        for i in range(len(seds)):
            if params2["mini"][i] >= mini_bound and np.isfinite(params2["mini"][i]):
                try:
                    seds2[i] = self.predictor.sed(
                        logl=params2["logl"][i],
                        logt=params2["logt"][i],
                        logg=params2["logg"][i],
                        feh_surf=params2["feh_surf"][i],
                        afe=params2["afe_surf"][i],
                        av=av,
                        rv=rv,
                        dist=dist,
                    )
                except Exception as e:
                    warnings.warn(
                        f"Secondary SED generation failed for EEP index {i} "
                        f"(mini2={params2['mini'][i]:.3f}): {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Combine primary and secondary SEDs
        seds[:] = add_mag(seds, seds2)
