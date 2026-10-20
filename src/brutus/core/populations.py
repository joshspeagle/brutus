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
- EEPTracks: Stellar parameter predictions for individuals
- StarEvolTrack: SED/photometry generation for individuals

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
>>> seds, params, params2 = pop_synth.get_seds(
...     feh=0.0, afe=0.0, loga=9.0,
...     av=0.1, dist=1000.0
... )

Advanced usage with binary populations:

>>> # Model binary stellar populations
>>> seds, params, params2 = pop_synth.get_seds(
...     feh=-0.5, afe=0.0, loga=10.0,
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

This implementation is based on the MIST stellar evolution framework
(Choi et al. 2016; Dotter 2016).

References
----------
- Choi et al. 2016, "MESA Isochrones and Stellar Tracks (MIST) 0. Methods
  for the Construction of Stellar Isochrones", ApJ, 823, 102
- Dotter 2016, "MESA Isochrones and Stellar Tracks (MIST) I. Solar-scaled
  Models", ApJS, 222, 8
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..data.download import _find_data_file

# Import filter definitions
from ..data.filters import FILTERS

# Import from utils (to be reorganized)
from ..utils.photometry import add_mag
from .individual import _mass_corrections

# Import neural network predictor
from .neural_nets import FastNNPredictor

__all__ = ["Isochrone", "StellarPop"]

# Isochrone predictions that StellarPop reads to compute SEDs.
_SED_LABELS = ("mini", "logl", "logt", "logg", "feh_surf", "afe_surf")


def _check_binary_fractions(name, values):
    """Raise `ValueError` unless every value is finite and in [0, 1]."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
        raise ValueError(f"{name} must be finite and in [0, 1]; got {values}")


class Isochrone:
    """
    Stellar parameter predictions for isochrones using MIST evolutionary models.

    This class provides interpolation of stellar parameters along isochrones of
    fixed metallicity, alpha enhancement, and age. It focuses solely on stellar
    parameter prediction (masses, temperatures, luminosities, etc.) without
    photometry generation.

    This class is analogous to EEPTracks but for stellar populations:
    - EEPTracks: Individual star parameters (mini, EEP, feh, afe) → parameters
    - Isochrone: Population parameters (feh, afe, loga, EEP) → parameters

    For photometry generation, use StellarPop with this class.

    Parameters
    ----------
    mistfile : str, optional
        Path to the HDF5 file containing the MIST isochrone grid. If not
        provided, `MIST_1.2_iso_vvcrit0.0.h5` is looked up in the brutus
        download directory (the Pooch cache, or ``$ASTRO_BRUTUS_DATA_DIR``
        when set), then in ``data/DATAFILES`` in the repository; a
        `FileNotFoundError` names `brutus.data.fetch_isos` when it is absent.
        The file must hold the mini, logt, logl and logg labels, which the
        empirical corrections need; a missing one raises a `ValueError`.

    predictions : list of str, optional
        The names of stellar parameters to predict. Must be a subset of the
        parameters available in the isochrone file (`pred_labels`); the
        columns returned by `get_predictions` follow this list's order.
        Unknown names raise a `ValueError`. Default is:
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
    >>> params = iso.get_predictions(feh=-0.5, afe=0.0, loga=10.0, eep=eep_range)
    """

    def __init__(self, mistfile=None, predictions=None, verbose=True):

        if mistfile is None:
            mistfile = _find_data_file(("MIST_1.2_iso_vvcrit0.0.h5",), "fetch_isos")
        if not Path(mistfile).is_file():
            raise FileNotFoundError(
                f"mistfile '{mistfile}' does not exist. Run "
                "brutus.data.fetch_isos() to download the MIST isochrones."
            )

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
                raw_labels = f["predictions"].attrs["labels"]
                # Decode byte strings from HDF5 to regular strings
                self.pred_labels = [
                    s.decode() if isinstance(s, bytes) else s for s in raw_labels
                ]
        except (OSError, KeyError) as e:
            raise RuntimeError(f"Failed to load isochrone data from {mistfile}: {e}")

        # Validate the requested predictions against the file's labels and
        # build the column map used to subset/reorder interpolator output.
        # Without this, `get_predictions` columns would silently follow the
        # file's `pred_labels` order regardless of `self.predictions`.
        missing = [p for p in self.predictions if p not in self.pred_labels]
        if missing:
            raise ValueError(
                f"Requested predictions {missing} are not available in "
                f"{mistfile}; available predictions are {self.pred_labels}."
            )
        self._pred_idxs = np.array(
            [self.pred_labels.index(p) for p in self.predictions], dtype=int
        )
        self._pred_identity = list(self.predictions) == list(self.pred_labels)

        # Columns that the empirical corrections read and shift
        corr_labels = ["mini", "logt", "logl", "logg"]
        missing = [p for p in corr_labels if p not in self.pred_labels]
        if missing:
            raise ValueError(
                f"{mistfile} lacks the labels {missing} that the empirical "
                "corrections need"
            )
        self._corr_idxs = [self.pred_labels.index(p) for p in corr_labels]

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

        # Handle singular alpha enhancement dimension
        if self.grid_dims[1] == 1:
            afe_val = self.xgrid[1][0]
            xgrid = list(self.xgrid)
            xgrid[1] = np.array([afe_val - 1e-5, afe_val + 1e-5])
            self.xgrid = tuple(xgrid)

            # Duplicate values in the padded dimension via a read-only
            # broadcast view: both afe layers share the original grid's
            # memory, so this costs nothing (materializing a copy would
            # double the ~176 MB grid and transiently hold both arrays).
            self.grid_dims[1] += 1
            self.pred_grid = np.broadcast_to(self.pred_grid, tuple(self.grid_dims))

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
            A value outside the grid's `afe` range returns NaN; for the
            shipped MIST file, the only valid value is 0.0.

        loga : float, optional
            Log10(age in years). Default is 8.5 (≈316 Myr).

        eep : array-like, optional
            Equivalent evolutionary points. If None, uses default grid.

        apply_corr : bool, optional
            Whether to apply empirical corrections. Default is True.

        corr_params : tuple, optional
            Correction parameters (dtdm, drdm, msto_smooth, feh_scale).
            Default is (0.09, -0.09, 30.0, 0.5), as in
            `EEPTracks.get_corrections`.

        Returns
        -------
        preds : numpy.ndarray of shape (Neep, Npred)
            Stellar parameter predictions for each EEP value. The columns
            correspond to the parameters listed in `self.predictions`
            (a subset/reordering of the file's `self.pred_labels`).

        Raises
        ------
        ValueError
            If `corr_params` does not have 4 entries.

        See Also
        --------
        StellarPop.get_seds : Uses these predictions for photometry generation
        brutus.core.EEPTracks.get_predictions : Similar function for individual stars

        Notes
        -----
        The method interpolates across the pre-computed MIST isochrone grid.
        Returns NaN for parameters outside the grid bounds or where data is
        not available.

        MIST primary EEPs (Dotter 2016):
        - Pre-main sequence: EEP < 202
        - Zero-age main sequence: EEP = 202
        - Terminal-age main sequence: EEP = 454
        - Subgiant and red giant branches: EEP 454-605 (RGB tip at 605)
        - Core helium burning: EEP 631-707
        - Early asymptotic giant branch: EEP 707-808
        - Thermally pulsing AGB: EEP 808
        - Post-AGB: EEP 1409

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
        preds = self.interpolator(labels)

        # Apply empirical corrections if requested (the same effective map
        # as EEPTracks.get_predictions; see EEPTracks.get_corrections)
        if apply_corr:
            mini_idx, logt_idx, logl_idx, logg_idx = self._corr_idxs
            dlogt, dlogr = _mass_corrections(
                preds[:, mini_idx], eep, feh_arr, corr_params
            )
            preds[:, logt_idx] += dlogt
            preds[:, logl_idx] += 2.0 * dlogr  # L ∝ R^2
            preds[:, logg_idx] -= 2.0 * dlogr  # g ∝ M/R^2

        # Subset/reorder the interpolator output (file `pred_labels` order)
        # into the order the user requested via `self.predictions`. The
        # corrections above index columns by `pred_labels`, so they must be
        # applied before this remapping.
        if not self._pred_identity:
            preds = preds[:, self._pred_idxs]

        return preds


class StellarPop:
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
        Isochrone instance for stellar parameter predictions. Its
        `predictions` must include mini, logl, logt, logg, feh_surf and
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
        If `isochrone.predictions` lacks a label listed above, or if a
        filter is not in the neural network file.

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
    ...     feh=-0.5, afe=0.0, loga=10.0,
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

        missing = [p for p in _SED_LABELS if p not in isochrone.predictions]
        if missing:
            raise ValueError(
                f"isochrone.predictions lacks {missing}, which StellarPop needs"
            )
        self.isochrone = isochrone

        # Set up filters
        if filters is None:
            filters = np.array(FILTERS)
        self.filters = filters

        self.predictor = FastNNPredictor(
            filters=filters, nnfile=nnfile, verbose=verbose
        )

    def _evaluate_seds(self, params, valid_mask, av, rv, dist):
        """
        Evaluate SEDs with ``sed_batch`` on the rows in `valid_mask`.

        Parameters
        ----------
        params : dict
            Dictionary of stellar parameter arrays, keyed by parameter name.
            Must contain ``"logt"``, ``"logg"``, ``"feh_surf"``, ``"logl"``,
            ``"afe_surf"``, and ``"mini"``.

        valid_mask : numpy.ndarray of bool
            Boolean mask indicating which stars should have SEDs evaluated.

        av : float
            V-band extinction A(V) in magnitudes.

        rv : float
            Extinction law parameter R(V).

        dist : float
            Distance in parsecs.

        Returns
        -------
        seds : numpy.ndarray of shape (N, Nfilt)
            SED array with predictions for valid stars and NaN elsewhere.
        """
        seds = np.full((len(params["mini"]), self.predictor.NFILT), np.nan)
        if np.any(valid_mask):
            seds[valid_mask] = self.predictor.sed_batch(
                logt=params["logt"][valid_mask],
                logg=params["logg"][valid_mask],
                feh_surf=params["feh_surf"][valid_mask],
                logl=params["logl"][valid_mask],
                afe=params["afe_surf"][valid_mask],
                av=av,
                rv=rv,
                dist=dist,
            )
        return seds

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
    ):
        """
        Generate synthetic photometry for a stellar population.

        Parameters
        ----------
        feh : float, optional
            Metallicity [Fe/H]. Default is 0.0.

        afe : float, optional
            Alpha enhancement [alpha/Fe]. Default is 0.0. A value outside the
            isochrone grid's `afe` range gives NaN SEDs; for the shipped MIST
            file, the only valid value is 0.0.

        loga : float, optional
            Log10(age in years). Default is 8.5.

        eep : float or array-like, optional
            Equivalent evolutionary points, a scalar or a 1-D array.
            If None, uses the isochrone's EEP grid.

        av : float, optional
            V-band extinction A(V). Default is 0.0.

        rv : float, optional
            Extinction law parameter R(V). Default is 3.3.

        binary_fraction : float, optional
            Secondary mass fraction for binaries, in [0, 1]. Default is 0.0
            (no binaries).
            - 0.0: No binaries
            - 0 < binary_fraction < 1: Mass ratio binaries
            - 1.0: Equal mass binaries

            Binary companions are only added for primaries with
            ``eep <= eep_binary_max``; other stars (and stars whose companion
            falls below ``mini_bound`` or has no age-consistent EEP) keep
            their primary-only SED. The companion EEP inverts mini(EEP) on
            the isochrone's own EEP grid, so it does not depend on `eep`.

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
            Primary component stellar parameters: a dict of (Neep,) arrays
            keyed by `isochrone.predictions`, or with ``return_dict=False``
            an array of shape (Neep, Npred) with columns in that order.

        params2 : dict or numpy.ndarray
            Secondary component parameters, in the same form as `params`.
            NaN for every star whose SED has no companion.

        Raises
        ------
        ValueError
            If `binary_fraction` is not finite or lies outside [0, 1].

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

        >>> seds, params, params2 = pop_synth.get_seds(
        ...     feh=0.0, loga=9.0, av=0.1, dist=1000.0
        ... )

        Binary population:

        >>> seds, params, params2 = pop_synth.get_seds(
        ...     feh=-0.5, loga=10.0, binary_fraction=0.4,
        ...     av=0.2, dist=2000.0
        ... )
        """
        _check_binary_fractions("binary_fraction", binary_fraction)
        eep = self._eep_array(eep)
        seds, params_arr, params = self._compute_primary(
            feh, afe, loga, eep, av, rv, dist, mini_bound, apply_corr, corr_params
        )
        mini_eep = (
            self._mini_eep_relation(feh, afe, loga)
            if 0.0 < binary_fraction < 1.0
            else None
        )

        return self._apply_binaries(
            seds,
            params_arr,
            params,
            binary_fraction,
            mini_eep,
            feh,
            afe,
            loga,
            eep,
            mini_bound,
            eep_binary_max,
            av,
            rv,
            dist,
            apply_corr,
            corr_params,
            return_dict,
        )

    def get_seds_smf_grid(
        self,
        smf_values,
        feh=0.0,
        afe=0.0,
        loga=8.5,
        eep=None,
        av=0.0,
        rv=3.3,
        dist=1000.0,
        mini_bound=0.5,
        eep_binary_max=480.0,
        apply_corr=True,
        corr_params=None,
        return_dict=True,
    ):
        """
        Generate synthetic photometry for several binary mass fractions.

        Equivalent to calling :meth:`get_seds` once per entry of
        ``smf_values`` (with ``binary_fraction`` set to that entry), but the
        primary isochrone predictions and primary SEDs — which do not depend
        on the binary mass fraction — are computed only once and reused
        across all values. Only the (per-SMF) secondary components are
        recomputed, roughly halving the cost of building a full
        (mass, SMF) population grid.

        Parameters
        ----------
        smf_values : array-like of float
            Binary mass fractions (secondary mass fractions) to evaluate, each
            in [0, 1]. Each entry is interpreted exactly like
            ``binary_fraction`` in :meth:`get_seds`.

        feh, afe, loga, eep, av, rv, dist, mini_bound, eep_binary_max, \
apply_corr, corr_params, return_dict
            Same as in :meth:`get_seds`; shared by every SMF value.

        Returns
        -------
        results : list of (seds, params, params2) tuples
            One :meth:`get_seds`-style result per entry of ``smf_values``,
            in order. The primary-parameter outputs (``params`` arrays)
            share underlying arrays across entries and must not be modified
            in place.

        Raises
        ------
        ValueError
            If any value of `smf_values` is not finite or lies outside [0, 1].

        See Also
        --------
        get_seds : Single binary-mass-fraction evaluation.
        brutus.analysis.populations.generate_isochrone_population_grid :
            Main consumer of this method.
        """
        smf_values = np.atleast_1d(np.asarray(smf_values, dtype=float))
        _check_binary_fractions("smf_values", smf_values)
        eep = self._eep_array(eep)
        seds0, params_arr, params = self._compute_primary(
            feh, afe, loga, eep, av, rv, dist, mini_bound, apply_corr, corr_params
        )
        mini_eep = (
            self._mini_eep_relation(feh, afe, loga)
            if np.any((smf_values > 0.0) & (smf_values < 1.0))
            else None
        )

        results = []
        for smf in smf_values:
            # Binary handling mutates the SED array in place, so each SMF
            # slice starts from a fresh copy of the shared primary SEDs.
            results.append(
                self._apply_binaries(
                    seds0.copy(),
                    params_arr,
                    dict(params),
                    smf,
                    mini_eep,
                    feh,
                    afe,
                    loga,
                    eep,
                    mini_bound,
                    eep_binary_max,
                    av,
                    rv,
                    dist,
                    apply_corr,
                    corr_params,
                    return_dict,
                )
            )
        return results

    def _eep_array(self, eep):
        """Return `eep` as a 1-D float array, or the isochrone's EEP grid."""
        if eep is None:
            return self.isochrone.eep_u
        eep = np.atleast_1d(np.asarray(eep, dtype=float))
        if eep.ndim != 1:
            raise ValueError(f"eep must be a scalar or a 1-D array; got {eep.shape}")
        return eep

    def _mini_eep_relation(self, feh, afe, loga):
        """
        Return the isochrone's finite (mini, eep) pairs, sorted by mini.

        Interpolating EEP in mini on these pairs inverts mini(EEP) on the
        isochrone's own EEP grid, independent of the caller's `eep`.
        """
        iso = self.isochrone
        mini = iso.get_predictions(
            feh=feh, afe=afe, loga=loga, eep=iso.eep_u, apply_corr=False
        )[:, list(iso.predictions).index("mini")]
        finite = np.isfinite(mini)
        order = np.argsort(mini[finite])
        return mini[finite][order], iso.eep_u[finite][order]

    def _compute_primary(
        self, feh, afe, loga, eep, av, rv, dist, mini_bound, apply_corr, corr_params
    ):
        """
        Compute primary-component stellar parameters and SEDs.

        These are independent of the binary mass fraction, so callers that
        evaluate several binary fractions (see :meth:`get_seds_smf_grid`)
        compute them once and reuse them.

        Returns
        -------
        seds : numpy.ndarray of shape (Neep, Nfilt)
            Primary-only SEDs (NaN where the primary is invalid).
        params_arr : numpy.ndarray of shape (Neep, Npred)
            Primary stellar parameters (columns follow
            ``self.isochrone.predictions``).
        params : dict
            ``params_arr`` keyed by prediction name (column views).
        """
        # Generate stellar parameters using isochrone
        params_arr = self.isochrone.get_predictions(
            feh=feh,
            afe=afe,
            loga=loga,
            eep=eep,
            apply_corr=apply_corr,
            corr_params=corr_params,
        )

        # Convert to dictionary format
        params = dict(zip(self.isochrone.predictions, params_arr.T))

        # Generate primary component SEDs using vectorized evaluation
        mass_valid = params["mini"] >= mini_bound
        seds = self._evaluate_seds(params, mass_valid, av, rv, dist)

        return seds, params_arr, params

    def _apply_binaries(
        self,
        seds,
        params_arr,
        params,
        binary_fraction,
        mini_eep,
        feh,
        afe,
        loga,
        eep,
        mini_bound,
        eep_binary_max,
        av,
        rv,
        dist,
        apply_corr,
        corr_params,
        return_dict,
    ):
        """
        Add the binary-fraction-dependent secondary component to primary SEDs.

        Mutates ``seds`` in place (primary + secondary combined) and returns
        the final ``(seds, params, params2)`` tuple of :meth:`get_seds`.
        ``params_arr``/``params`` are read, never modified. ``mini_eep`` is
        the output of :meth:`_mini_eep_relation`, needed only when
        ``0 < binary_fraction < 1``.
        """
        # Initialize secondary parameters
        params_arr2 = np.full_like(params_arr, np.nan)
        params2 = dict(zip(self.isochrone.predictions, params_arr2.T))

        # Handle binary stars
        if 0.0 < binary_fraction < 1.0:
            self._add_binary_components(
                seds,
                params,
                params2,
                params_arr,
                params_arr2,
                binary_fraction,
                mini_eep,
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
                eep,
            )
        elif binary_fraction == 1.0:
            # Equal-mass binary: flux doubles -> magnitude decreases by
            # 2.5*log10(2). Binaries are only modeled on the main sequence
            # (eep <= eep_binary_max), matching _add_binary_components:
            # post-MS stars keep their primary-only SED and get no secondary.
            is_binary = (eep <= eep_binary_max) & ~np.isnan(seds[:, 0])
            seds[is_binary] -= 2.5 * np.log10(2.0)
            # np.where already returns fresh arrays, so no copies are needed;
            # keep params_arr2 in the same (Neep, Npred) orientation as the
            # 0 < binary_fraction < 1 path so return_dict=False is consistent.
            params2 = {p: np.where(is_binary, arr, np.nan) for p, arr in params.items()}
            params_arr2 = np.array(params_arr, copy=True)
            params_arr2[~is_binary, :] = np.nan

        # Format output
        if not return_dict:
            params = params_arr
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
        mini_eep,
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
        eep,
    ):
        """Add binary star components to the population synthesis."""
        # Secondary EEPs from the isochrone's mini(EEP) relation, so that
        # they do not depend on how the caller samples `eep`.
        mini2 = params["mini"] * binary_fraction
        mini_sorted, eep_sorted = mini_eep
        if len(mini_sorted) > 0:
            eep2 = np.interp(mini2, mini_sorted, eep_sorted, left=np.nan, right=np.nan)
        else:
            eep2 = np.full(np.shape(eep), np.nan)

        # Restrict binaries to main sequence
        eep2[(eep2 > eep_binary_max) | (eep > eep_binary_max)] = np.nan

        # Generate secondary parameters
        params_arr2[:] = self.isochrone.get_predictions(
            feh=feh,
            afe=afe,
            loga=loga,
            eep=eep2,
            apply_corr=apply_corr,
            corr_params=corr_params,
        )

        # Generate secondary SEDs using vectorized evaluation
        sec_valid = (params2["mini"] >= mini_bound) & np.isfinite(params2["mini"])
        seds2 = self._evaluate_seds(params2, sec_valid, av, rv, dist)

        # Combine primary and secondary SEDs only where a valid secondary
        # exists. A star without a resolvable companion (post-MS primary,
        # secondary below the mass bound, or failed EEP match) keeps its
        # primary-only SED — combining with a NaN secondary via add_mag would
        # silently discard the (valid) primary as well.
        # A companion counts only where the primary has an SED too: adding
        # it to a NaN primary leaves the combined SED NaN.
        has_secondary = np.all(np.isfinite(seds2), axis=1) & np.all(
            np.isfinite(seds), axis=1
        )
        params_arr2[~has_secondary] = np.nan
        if np.any(has_secondary):
            seds[has_secondary] = add_mag(seds[has_secondary], seds2[has_secondary])
