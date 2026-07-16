#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid generation utilities for pre-computed stellar model grids.

This module provides functionality to generate the HDF5 model grid files
used by StarGrid for fast photometric predictions. Grids store pre-computed
photometry at a reference distance (1 kpc) along with polynomial coefficients
for reddening corrections as a function of A_V and R_V.

The grid generation process:
1. Creates a 5D grid in (mini, eep, feh, afe, smf) parameter space
2. Computes base photometry at A_V=0, R_V=3.3, distance=1 kpc
3. Fits polynomial coefficients for reddening dependence
4. Saves in HDF5 format compatible with StarGrid and load_models()

Classes
-------
GridGenerator : Main grid generation class
    Generates pre-computed model grids from evolutionary tracks

Functions
---------
None (all functionality in GridGenerator class)

Examples
--------
Generate a grid from evolutionary tracks:

>>> from brutus.core.individual import EEPTracks
>>> from brutus.core.grid_generation import GridGenerator
>>>
>>> # Initialize tracks and generator
>>> tracks = EEPTracks()
>>> generator = GridGenerator(tracks, filters=['g', 'r', 'i', 'z'])
>>>
>>> # Generate grid with default spacing
>>> generator.make_grid(output_file='my_grid.h5')

Generate a custom sparse grid:

>>> import numpy as np
>>> # Define custom grids
>>> mini_grid = np.arange(0.7, 1.5, 0.1)
>>> eep_grid = np.arange(300, 500, 10)
>>> feh_grid = np.array([-0.5, 0.0, 0.5])
>>>
>>> generator.make_grid(
...     mini_grid=mini_grid,
...     eep_grid=eep_grid,
...     feh_grid=feh_grid,
...     output_file='custom_grid.h5',
...     verbose=True
... )

Notes
-----
**Grid Reference Distance**: All grids are generated at 1 kpc (1000 pc)
reference distance. This choice provides consistency with Gaia parallax
measurements (1 mas = 1 kpc) and is documented in the grid attributes.

**Reddening Parameterization**: Photometry is parameterized as:
    m(A_V, R_V) = m_0 + A_V · (a + b · R_V)

where m_0 is the unreddened magnitude and (a, b) are fitted coefficients:
`a` is the A_V reddening vector extrapolated to R_V=0 (the linear-fit
intercept) and `b` is its change per unit R_V. This matches how the
coefficients are consumed in `brutus.core.sed_utils._get_seds`
(`rvec = a + b * R_V`) and allows fast computation of reddened photometry
for arbitrary (A_V, R_V).

**Grid Format**: HDF5 files follow the format established in Speagle et al. (2025)
and contain three datasets:
- `mag_coeffs`: Structured array (Nmodel, Nfilter, 3) with coefficients
- `labels`: Structured array (Nmodel,) with input parameters
- `parameters`: Structured array (Nmodel,) with predicted stellar parameters

**Memory and Performance**: Large grids can require substantial memory
and computation time. A full grid with default spacing contains ~300k models
and takes ~2-4 hours to generate. Use sparse grids for testing or when
covering a limited parameter space.

References
----------
- Grid format follows the structure established in Speagle et al. (2025),
  optimized for StarGrid interpolation performance.
"""

import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import h5py
import numpy as np

# Import brutus components
from ..data.filters import FILTERS
from ..utils.photometry import add_mag
from .individual import StarEvolTrack

__all__ = ["GridGenerator"]


class GridGenerator:
    """
    Generate pre-computed stellar model grids with reddening coefficients.

    This class creates HDF5 grid files used by StarGrid for fast photometric
    predictions. Grids are computed at 1 kpc reference distance and include
    polynomial coefficients for A_V and R_V reddening corrections.

    The generator uses evolutionary tracks (EEPTracks) combined with neural
    network bolometric corrections to compute photometry across a
    multi-dimensional parameter grid, then fits reddening vectors to enable
    fast interpolation.

    Parameters
    ----------
    tracks : EEPTracks
        Stellar evolutionary track model providing parameter predictions.
        Must expose the EEPTracks interface: `labels`, `predictions`, and a
        `get_predictions(label_array, ...)` method. `Isochrone` objects are
        NOT supported (they lack `labels` and use a keyword-based
        `get_predictions` signature).

    filters : list of str, optional
        Names of photometric filters for which to generate models.
        If None, uses all available filters from FILTERS.

    nnfile : str or Path, optional
        Path to neural network file for bolometric corrections.
        If None, uses default neural network file.

    verbose : bool, optional
        Whether to print progress messages during initialization.
        Default is True.

    Attributes
    ----------
    tracks : EEPTracks
        Evolutionary track model

    star_track : StarEvolTrack
        SED generator combining tracks with neural networks

    filters : numpy.ndarray
        Array of filter names

    predictor : FastNNPredictor
        Neural network predictor for photometry

    Examples
    --------
    Create grid generator and generate default grid:

    >>> from brutus.core.individual import EEPTracks
    >>> from brutus.core.grid_generation import GridGenerator
    >>>
    >>> tracks = EEPTracks(verbose=False)
    >>> gen = GridGenerator(tracks)
    >>> gen.make_grid(output_file='grid.h5', verbose=True)

    Generate grid for specific filters only:

    >>> gen_gaia = GridGenerator(tracks, filters=['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW'])
    >>> gen_gaia.make_grid(output_file='grid_gaia.h5')

    Notes
    -----
    The GridGenerator uses dependency injection rather than inheritance,
    accepting an EEPTracks instance. This design:
    - Maintains consistency with StarEvolTrack architecture
    - Allows flexibility in track implementations
    - Enables easier testing with mock tracks
    - Separates grid generation from track interpolation logic

    See Also
    --------
    brutus.core.individual.StarGrid : Uses grids generated by this class
    brutus.data.loader.load_models : Loads grids generated by this class
    """

    def __init__(self, tracks, filters=None, nnfile=None, verbose=True):
        """Initialize grid generator with evolutionary tracks."""
        # Fail fast with a clear error for incompatible track objects
        # (e.g. Isochrone, which lacks `labels` and takes keyword scalars in
        # get_predictions) instead of an opaque AttributeError mid-grid.
        missing = [
            attr
            for attr in ("labels", "predictions", "get_predictions")
            if not hasattr(tracks, attr)
        ]
        if missing:
            raise TypeError(
                "`tracks` must provide the EEPTracks interface; missing "
                f"attributes: {missing}. Note that Isochrone objects are "
                "not supported by GridGenerator."
            )

        # Store tracks
        self.tracks = tracks

        # Initialize filters
        if filters is None:
            filters = FILTERS
        self.filters = np.array(filters)

        if verbose:
            sys.stderr.write(f"Grid filters: {list(self.filters)}\n")

        # Create StarEvolTrack for SED generation
        self.star_track = StarEvolTrack(
            tracks=tracks, filters=filters, nnfile=nnfile, verbose=verbose
        )

        # Store neural network predictor
        self.predictor = self.star_track.predictor

    def make_grid(
        self,
        mini_grid=None,
        eep_grid=None,
        feh_grid=None,
        afe_grid=None,
        smf_grid=None,
        av_grid=None,
        av_wt=None,
        rv_grid=None,
        rv_wt=None,
        dist=1000.0,
        loga_max=10.14,
        eep_binary_max=480.0,
        mini_bound=0.5,
        apply_corr=True,
        corr_params=None,
        output_file=None,
        verbose=True,
    ):
        """
        Generate model grid with reddening coefficients.

        Creates a grid of stellar models across the specified parameter space,
        computing photometry at 1 kpc reference distance and fitting polynomial
        coefficients for reddening corrections. Results are saved to HDF5 format
        compatible with StarGrid.

        Parameters
        ----------
        mini_grid : numpy.ndarray, optional
            Grid of initial masses in solar masses. If None, defaults to
            np.arange(0.5, 2.0, 0.025) covering low to intermediate mass stars.

        eep_grid : numpy.ndarray, optional
            Grid of equivalent evolutionary points. If None, defaults to
            adaptive grid: resolution 6 from 202-454 (MS), resolution 2 from
            454-808 (post-MS).

        feh_grid : numpy.ndarray, optional
            Grid of [Fe/H] metallicity values. If None, defaults to adaptive
            grid: resolution 0.1 from -3.0 to -2.0, resolution 0.05 from
            -2.0 to +0.5.

        afe_grid : numpy.ndarray, optional
            Grid of [α/Fe] alpha enhancement values. If None, defaults to
            np.arange(-0.2, 0.6, 0.2).

        smf_grid : numpy.ndarray, optional
            Grid of secondary mass fractions for binaries. If None, defaults
            to [0.] (single stars only).

        av_grid : numpy.ndarray, optional
            Grid of A_V extinction values used for fitting reddening vector.
            If None, defaults to np.arange(0., 1.5, 0.3).

        av_wt : numpy.ndarray, optional
            Weights for A_V grid points when fitting. If None, defaults to
            (1e-5 + av_grid)**-1 which forces fit through A_V=0.

        rv_grid : numpy.ndarray, optional
            Grid of R_V values used for fitting differential reddening.
            If None, defaults to np.arange(2.4, 4.2, 0.3).

        rv_wt : numpy.ndarray, optional
            Weights for R_V grid points when fitting. If None, defaults to
            exp(-abs(R_V - 3.3) / 0.5) favoring R_V=3.3.

        dist : float, optional
            Reference distance in parsecs. Default is 1000 (1 kpc).
            **This should not be changed** as it affects StarGrid calibration.

        loga_max : float, optional
            Maximum log10(age) in years. Models older than this are masked.
            Default is 10.14 (13.8 Gyr).

        eep_binary_max : float, optional
            Maximum EEP for binary models. Above this, binaries are not
            computed (typically giant phase). Default is 480.

        mini_bound : float, optional
            Minimum initial mass threshold. Models below this are masked.
            Default is 0.5 solar masses.

        apply_corr : bool, optional
            Whether to apply empirical corrections to Teff and radius.
            Default is True.

        corr_params : tuple, optional
            Parameters for empirical corrections (dtdm, drdm, msto_smooth,
            feh_scale). If None, uses default values from tracks.

        output_file : str or Path, optional
            Path to output HDF5 file. If None, results are stored as
            attributes but not saved to disk.

        verbose : bool, optional
            Whether to print progress messages. Default is True.

        Returns
        -------
        None
            Results are saved to output_file if provided, and stored as
            class attributes: grid_labels, grid_seds, grid_params, grid_sel

        Notes
        -----
        **Grid Size**: Default parameters create ~300,000 models. Computation
        time scales linearly with grid size and quadratically with number of
        (av_grid × rv_grid) points for reddening fits.

        **Memory Usage**: Full grid with 20 filters requires ~2 GB for storage.
        Peak memory during generation can be 3-4× larger.

        **Reddening Fits**: For each valid model, the code:
        1. Computes SEDs across (av_grid × rv_grid) combinations
        2. Fits linear dependence on A_V at each R_V value
        3. Fits linear dependence of A_V slope on R_V
        4. Stores [m_0, a, b] where m = m_0 + A_V·(a + b·R_V), i.e. `a` is
           the A_V reddening vector extrapolated to R_V=0

        Examples
        --------
        Generate default grid:

        >>> gen.make_grid(output_file='grid_default.h5')

        Generate sparse testing grid:

        >>> gen.make_grid(
        ...     mini_grid=np.linspace(0.8, 1.2, 5),
        ...     eep_grid=np.linspace(300, 450, 10),
        ...     feh_grid=np.array([0.0]),
        ...     afe_grid=np.array([0.0]),
        ...     smf_grid=np.array([0.0]),
        ...     output_file='grid_test.h5',
        ...     verbose=True
        ... )
        """
        # Initialize parameter grids with defaults
        if mini_grid is None:
            mini_grid = np.arange(0.5, 2.0 + 1e-5, 0.025)
        if eep_grid is None:
            eep_grid = np.concatenate(
                [np.arange(202.0, 454.0, 6.0), np.arange(454.0, 808.0 + 1e-5, 2.0)]
            )
        if feh_grid is None:
            feh_grid = np.concatenate(
                [np.arange(-3.0, -2.0, 0.1), np.arange(-2.0, 0.5 + 1e-5, 0.05)]
            )
        if afe_grid is None:
            afe_grid = np.arange(-0.2, 0.6 + 1e-5, 0.2)
        if smf_grid is None:
            smf_grid = np.array([0.0])

        # Initialize reddening grids and weights
        if av_grid is None:
            av_grid = np.arange(0.0, 1.5 + 1e-5, 0.3)
            av_grid[-1] -= 1e-5  # Ensure we don't exceed 1.5
        if av_wt is None:
            # Inverse weighting favoring A_V=0
            av_wt = (1e-5 + av_grid) ** -1.0
        if rv_grid is None:
            rv_grid = np.arange(2.4, 4.2 + 1e-5, 0.3)
        if rv_wt is None:
            # Gaussian weighting favoring R_V=3.3
            rv_wt = np.exp(-np.abs(rv_grid - 3.3) / 0.5)

        # Create grid labels
        label_names = ["mini", "eep", "feh", "afe", "smf"]
        ltype = np.dtype([(n, "f8") for n in label_names])
        self.grid_labels = np.array(
            list(product(mini_grid, eep_grid, feh_grid, afe_grid, smf_grid)),
            dtype=ltype,
        )
        Ngrid = len(self.grid_labels)

        if verbose:
            sys.stderr.write(f"\nGenerating grid with {Ngrid:,} models\n")
            sys.stderr.write(
                f"  mini: {len(mini_grid)} points "
                f"[{mini_grid.min():.2f}, {mini_grid.max():.2f}]\n"
            )
            sys.stderr.write(
                f"  eep:  {len(eep_grid)} points "
                f"[{eep_grid.min():.0f}, {eep_grid.max():.0f}]\n"
            )
            sys.stderr.write(
                f"  feh:  {len(feh_grid)} points "
                f"[{feh_grid.min():.2f}, {feh_grid.max():.2f}]\n"
            )
            sys.stderr.write(
                f"  afe:  {len(afe_grid)} points "
                f"[{afe_grid.min():.2f}, {afe_grid.max():.2f}]\n"
            )
            sys.stderr.write(
                f"  smf:  {len(smf_grid)} points "
                f"[{smf_grid.min():.2f}, {smf_grid.max():.2f}]\n"
            )
            sys.stderr.write(
                f"\nReddening grid: {len(av_grid)} A_V × "
                f"{len(rv_grid)} R_V = "
                f"{len(av_grid)*len(rv_grid)} evaluations per model\n\n"
            )

        # Initialize storage arrays
        param_names = self.tracks.predictions
        ptype = np.dtype([(n, "f8") for n in param_names])
        stype = np.dtype([(filt, "f4", 3) for filt in self.filters])

        self.grid_seds = np.full(Ngrid, np.nan, dtype=stype)
        self.grid_params = np.full(Ngrid, np.nan, dtype=ptype)
        self.grid_sel = np.ones(Ngrid, dtype=bool)

        # Generate models
        percentage = -99
        ttot, t1 = 0.0, time.time()

        for i, (mini, eep, feh, afe, smf) in enumerate(self.grid_labels):
            if mini < mini_bound:
                # Documented contract: primaries below `mini_bound` are
                # masked (previously the bound only gated binary secondaries,
                # silently keeping sub-threshold primaries as valid).
                self.grid_sel[i] = False
                sed = params = None  # storage rows stay NaN-filled
            else:
                # Compute model and parameters at base reddening
                sed, params, params2, eep2 = self.star_track.get_seds(
                    mini=mini,
                    eep=eep,
                    feh=feh,
                    afe=afe,
                    smf=smf,
                    av=0.0,
                    rv=3.3,
                    dist=dist,
                    loga_max=loga_max,
                    eep_binary_max=eep_binary_max,
                    mini_bound=mini_bound,
                    apply_corr=apply_corr,
                    corr_params=corr_params,
                    return_dict=False,
                    return_eep2=True,
                )

                # Save parameters for primary
                self.grid_params[i] = tuple(params)

            # Check if SED is valid
            if sed is None or np.any(np.isnan(sed)) or np.any(np.isnan(params)):
                # Flag as invalid; storage rows are pre-filled with NaN
                self.grid_sel[i] = False
                self.grid_seds[i] = tuple(np.full((len(self.filters), 3), np.nan))
            else:
                # Fit reddening coefficients. Stellar parameters are
                # independent of (A_V, R_V), so pass the already-computed
                # primary/secondary predictions instead of re-interpolating
                # the tracks for every reddening-grid point.
                coeffs = self._fit_reddening_coefficients(
                    params=params,
                    params2=params2,
                    sed_base=sed,
                    av_grid=av_grid,
                    av_wt=av_wt,
                    rv_grid=rv_grid,
                    rv_wt=rv_wt,
                    dist=dist,
                )
                if np.any(np.isnan(coeffs)):
                    # A reddening-grid point outside the NN training bounds
                    # yields NaN photometry which polyfit silently propagates;
                    # such models must not be flagged as valid.
                    self.grid_sel[i] = False
                    self.grid_seds[i] = tuple(np.full((len(self.filters), 3), np.nan))
                else:
                    self.grid_seds[i] = tuple(coeffs)

            # Update timing
            t2 = time.time()
            dt = t2 - t1
            ttot += dt
            tavg = ttot / (i + 1)
            test = tavg * (Ngrid - i - 1)
            t1 = t2

            # Print progress
            new_percentage = int((i + 1) / Ngrid * 1e5)
            if verbose and new_percentage != percentage:
                percentage = new_percentage
                sys.stderr.write(
                    f"\rConstructing grid {percentage/1e3:6.3f}% "
                    f"({i+1:,}/{Ngrid:,}) "
                    f"[mini={mini:6.3f}, eep={eep:6.1f}, feh={feh:6.2f}, "
                    f"afe={afe:5.2f}, smf={smf:4.2f}] "
                    f"(t/obj: {tavg*1e3:5.2f} ms, "
                    f"est. remaining: {test:8.1f} s)          "
                )
                sys.stderr.flush()

        if verbose:
            sys.stderr.write("\n\n")
            valid_models = self.grid_sel.sum()
            sys.stderr.write(
                f"Grid generation complete: "
                f"{valid_models:,}/{Ngrid:,} valid models "
                f"({100*valid_models/Ngrid:.1f}%)\n"
            )

        # Save to file if requested
        if output_file is not None:
            self._save_grid(output_file, dist, verbose=verbose)

    def _fit_reddening_coefficients(
        self,
        params,
        params2,
        sed_base,
        av_grid,
        av_wt,
        rv_grid,
        rv_wt,
        dist,
    ):
        """
        Fit polynomial coefficients for reddening dependence.

        Computes SEDs across a grid of (A_V, R_V) values and fits a bilinear
        model: m(A_V, R_V) = m_0 + A_V · (a + b · R_V), where `a` is the A_V
        reddening vector extrapolated to R_V=0 (linear-fit intercept) and `b`
        is its change per unit R_V, matching the consumer
        `brutus.core.sed_utils._get_seds` (`rvec = a + b * R_V`).

        Stellar parameters do not depend on (A_V, R_V), so the neural network
        is evaluated in a single batched call over the reddening lattice using
        the track predictions already computed for the base model (avoiding
        Nav × Nrv redundant track interpolations per grid point).

        Parameters
        ----------
        params : numpy.ndarray
            Primary-component track predictions (ordered as
            ``self.tracks.predictions``) from the base A_V=0 evaluation.
        params2 : numpy.ndarray
            Secondary-component predictions (same ordering); all-NaN when the
            model has no contributing binary companion.
        sed_base : numpy.ndarray
            Base SED at A_V=0, R_V=3.3.
        av_grid, av_wt, rv_grid, rv_wt : numpy.ndarray
            Reddening lattice points and fit weights (see `make_grid`).
        dist : float
            Reference distance in parsecs.

        Returns
        -------
        coeffs : numpy.ndarray of shape (Nfilters, 3)
            Coefficients [m_0, a, b] for each filter
        """
        # Reddening lattice, flattened in (Nrv, Nav) order.
        rv_mesh, av_mesh = np.meshgrid(rv_grid, av_grid, indexing="ij")
        av_pts = av_mesh.ravel()
        rv_pts = rv_mesh.ravel()

        # Primary SEDs across the reddening lattice (single batched NN call).
        p1 = dict(zip(self.tracks.predictions, params))
        seds = self._sed_reddening_grid(p1, av_pts, rv_pts, dist)

        # Add binary companion if it contributed to the base SED. The
        # secondary predictions are all-NaN unless the companion existed and
        # converged during the base evaluation (its presence conditions are
        # independent of A_V/R_V).
        p2 = dict(zip(self.tracks.predictions, params2))
        sed_fields = ("logl", "logt", "logg", "feh_surf", "afe_surf")
        if np.all(np.isfinite([p2[k] for k in sed_fields])):
            seds2 = self._sed_reddening_grid(p2, av_pts, rv_pts, dist)
            seds = add_mag(seds, seds2)

        seds = seds.reshape(len(rv_grid), len(av_grid), -1)  # Shape: (Nrv, Nav, Nfilt)

        # Fit A_V dependence at each R_V
        # For each (rv, filter): fit m vs A_V
        sfits = np.array(
            [np.polyfit(av_grid, s, 1, w=av_wt).T for s in seds]
        )  # Shape: (Nrv, Nfilt, 2) where [..., 0] = slope, [..., 1] = intercept

        # Fit R_V dependence of the A_V slope
        # For each filter: fit A_V_slope vs R_V
        sedr, seda = np.polyfit(
            rv_grid, sfits[:, :, 0], 1, w=rv_wt
        )  # sedr: Rv dependence, seda: Av vector extrapolated to Rv=0

        # Combine: [base_magnitude, av_vector, rv_vector]
        coeffs = np.c_[sed_base, seda, sedr]

        return coeffs

    def _sed_reddening_grid(self, pdict, av_pts, rv_pts, dist):
        """
        Evaluate NN photometry on an (A_V, R_V) lattice at fixed stellar
        parameters.

        Batched equivalent of ``FastNNPredictor.sed`` for one star and many
        reddening points: the stellar inputs are constant across columns and
        only the (av, rv) columns vary. Out-of-bounds points return NaN rows,
        mirroring the scalar path.

        Parameters
        ----------
        pdict : dict
            Track predictions for one component; must contain 'logl', 'logt',
            'logg', 'feh_surf', and 'afe_surf'.
        av_pts, rv_pts : numpy.ndarray of shape (Npts,)
            Flattened reddening lattice coordinates.
        dist : float
            Distance in parsecs.

        Returns
        -------
        seds : numpy.ndarray of shape (Npts, Nfilters)
            Apparent magnitudes at each lattice point (NaN where the NN
            inputs fall outside its training bounds).
        """
        predictor = self.predictor
        mu = 5.0 * np.log10(dist) - 5.0

        npts = len(av_pts)
        x = np.empty((6, npts))
        x[0] = 10.0 ** pdict["logt"]
        x[1] = pdict["logg"]
        x[2] = pdict["feh_surf"]
        x[3] = pdict["afe_surf"]
        x[4] = av_pts
        x[5] = rv_pts

        valid = (
            np.all(np.isfinite(x), axis=0)
            & np.all(x >= predictor.xmin[:, None], axis=0)
            & np.all(x <= predictor.xmax[:, None], axis=0)
        )

        seds = np.full((npts, predictor.NFILT), np.nan)
        if valid.any():
            # nneval squeezes trailing singleton axes; restore (NFILT, n).
            BC = predictor.nneval(x[:, valid]).reshape(predictor.NFILT, -1)
            seds[valid] = -2.5 * pdict["logl"] + 4.74 - BC.T + mu

        return seds

    def _save_grid(self, output_file, dist, verbose=True):
        """
        Save grid to HDF5 file in StarGrid-compatible format.

        Only valid models (``grid_sel == True``) are written: invalid grid
        points carry all-NaN coefficients/parameters that would otherwise
        poison every downstream likelihood evaluation in BruteForce. This
        matches the published grid files, which contain only valid models.

        Parameters
        ----------
        output_file : str or Path
            Path to output HDF5 file
        dist : float
            Reference distance in parsecs
        verbose : bool, optional
            Whether to print save confirmation
        """
        output_path = Path(output_file)

        if verbose:
            sys.stderr.write(f"Saving grid to {output_path}...\n")

        # Persist only valid models; invalid rows are NaN-filled and would
        # crash BruteForce fitting if written (NaN lnl poisons np.max).
        sel = self.grid_sel

        with h5py.File(output_path, "w") as f:
            # Save magnitude coefficients
            f.create_dataset(
                "mag_coeffs",
                data=self.grid_seds[sel],
                compression="gzip",
                compression_opts=4,
            )

            # Save labels (inputs)
            f.create_dataset(
                "labels",
                data=self.grid_labels[sel],
                compression="gzip",
                compression_opts=4,
            )

            # Save parameters (predictions)
            f.create_dataset(
                "parameters",
                data=self.grid_params[sel],
                compression="gzip",
                compression_opts=4,
            )

            # Save metadata
            f.attrs["reference_distance_pc"] = dist
            f.attrs["reference_distance_note"] = (
                "All magnitudes computed at this distance (default 1 kpc = 1000 pc)"
            )
            try:
                import brutus

                f.attrs["brutus_version"] = brutus.__version__
            except (ImportError, AttributeError):
                f.attrs["brutus_version"] = "unknown"
            f.attrs["creation_date"] = datetime.now().isoformat()
            f.attrs["n_models_total"] = len(self.grid_labels)
            f.attrs["n_models_valid"] = self.grid_sel.sum()
            # Convert filters to ASCII strings for HDF5 compatibility
            f.attrs["filters"] = [filt.encode("ascii") for filt in self.filters]

        if verbose:
            file_size_mb = output_path.stat().st_size / 1024**2
            sys.stderr.write(f"Grid saved successfully " f"({file_size_mb:.1f} MB)\n")
