#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading utilities for brutus.

This module contains functions for loading stellar evolution models,
photometric offsets, and other data files into memory for use in
stellar fitting and analysis.
"""

import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

# Import filter definitions
from .filters import FILTERS

__all__ = ["find_nn_file", "load_models", "load_offsets"]


def find_nn_file(possible_names=("nn_c3k.h5", "nnMIST_BC.h5")):
    """
    Find the neural network model file in standard locations.

    Searches for the neural network HDF5 file used for bolometric correction
    predictions, checking the local ``data/DATAFILES`` directory first, then
    the pooch cache directory.

    Parameters
    ----------
    possible_names : tuple of str, optional
        Filenames to search for, tried in order. Default is
        ``("nn_c3k.h5", "nnMIST_BC.h5")``. The first file found is returned.

    Returns
    -------
    path : pathlib.Path
        Absolute path to the neural network file.

    Raises
    ------
    FileNotFoundError
        If none of the candidate files can be found in any searched location.
    """
    # Package root: src/brutus/data/loader.py -> four levels up to repo root
    package_root = Path(__file__).parent.parent.parent.parent

    for nn_name in possible_names:
        # Check local data directory first
        local_path = package_root / "data" / "DATAFILES" / nn_name
        if os.path.exists(str(local_path)):
            return local_path

        # If not found locally, try pooch cache directory
        import pooch

        cache_dir = Path(pooch.os_cache("astro-brutus"))
        cache_path = cache_dir / nn_name
        if os.path.exists(str(cache_path)):
            return cache_path

    raise FileNotFoundError(
        f"Could not find neural network file. "
        f"Searched for {possible_names} in "
        f"{package_root / 'data' / 'DATAFILES'} and pooch cache. "
        f"Run brutus.data.fetch_nns() to download the file."
    )


def load_models(
    filepath,
    filters=None,
    labels=None,
    include_ms=True,
    include_postms=True,
    include_binaries=False,
    verbose=True,
):
    """
    Loads pre-computed stellar model grids with photometric coefficients
    for multiple filters and stellar parameters. Models can be filtered
    by evolutionary phase and binary status.

    Parameters
    ----------
    filepath : str
        The filepath of the stellar model file (typically .h5 format).

    filters : iterable of str with length `Nfilt`, optional
        List of filters that will be loaded. If not provided, will default
        to all available filters. See the internally-defined `FILTERS` variable
        for more details on filter names. Any filters that are not available
        will be skipped over (a `UserWarning` listing the skipped names and
        the surviving column order is emitted).

    labels : iterable of str with length `Nlabel`, optional
        List of labels associated with the set of imported stellar models.
        Any labels that are not available will be skipped over.
        The default set is `['mini', 'feh', 'eep', 'smf', 'afe', 'loga',
        'logl', 'logt', 'logg', 'Mr', 'agewt']`.

    include_ms : bool, optional
        Whether to include objects on the Main Sequence. Applied as a cut on
        `eep <= 454` when `'eep'` is included. Default is `True`.

    include_postms : bool, optional
        Whether to include objects evolved off the Main Sequence. Applied as a
        cut on `eep > 454` when `'eep'` is included. Default is `True`.

    include_binaries : bool, optional
        Whether to include unresolved binaries. Applied as a cut on
        secondary mass fraction (`'smf'`) when it has been included. Default
        is `False`. If set to `False`, `'smf'` is not returned as a label.

    verbose : bool, optional
        Whether to print progress messages. Default is `True`.

    Returns
    -------
    models : `~numpy.ndarray` of shape `(Nmodel, Nfilt, Ncoef)`
        Array of models comprised of coefficients in each band used to
        describe the photometry as a function of reddening, parameterized
        in terms of A_V. Each model contains coefficients for:
        - Unreddened magnitude
        - Reddening vector for R_V = 0
        - Change in reddening vector as function of R_V

    labels : structured `~numpy.ndarray` with dimensions `(Nmodel, Nlabel)`
        A structured array with the labels corresponding to each model.
        Contains stellar parameters like initial mass, metallicity, age, etc.

    label_mask : structured `~numpy.ndarray` with dimensions `(1, Nlabel)`
        A structured array that masks ancillary labels associated with
        predictions (rather than those used to compute the model grid).

    Raises
    ------
    ValueError
        If neither main sequence nor post-main sequence models are included.

    Notes
    -----
    Models with non-finite photometric coefficients (invalid grid points
    written by older `GridGenerator` versions) are dropped so the returned
    grid is always safe to fit with `BruteForce`.

    The `label_mask` return value is a boolean structured array indicating which
    labels are ancillary (derived from the grid) vs. those used to generate
    the grid. For example, if luminosity is predicted from mass/age/metallicity,
    this mask would be False for luminosity. Used internally by StarGrid to
    determine which parameters to marginalize over during fitting.

    Examples
    --------
    Basic usage with default settings:

    >>> from brutus.data import load_models, fetch_grids
    >>> fetch_grids()  # Download data (first time only)
    >>> models, labels, label_mask = load_models('grid_mist_v9.h5')
    >>> print(f"Loaded {len(labels)} stellar models")
    >>> print(f"Available labels: {labels.dtype.names}")

    Loading specific filters only:

    >>> models, labels, mask = load_models(
    ...     'grid_mist_v9.h5',
    ...     filters=['g', 'r', 'i', 'z', 'y']
    ... )

    Loading only main sequence stars:

    >>> models, labels, mask = load_models(
    ...     'grid_mist_v9.h5',
    ...     include_ms=True,
    ...     include_postms=False
    ... )

    Using with StarGrid for fitting:

    >>> from brutus.core import StarGrid
    >>> from brutus.analysis import BruteForce
    >>> models, labels, mask = load_models('grid_mist_v9.h5')
    >>> grid = StarGrid(models, labels, mask)
    >>> fitter = BruteForce(grid)
    """
    # Initialize values.
    if filters is None:
        filters = FILTERS
    if labels is None:
        # 'afe' is a grid input (grids are generated on a 5D
        # (mini, eep, feh, afe, smf) lattice); omitting it makes multi-afe
        # grids ambiguous. For single-afe grids the constant column is
        # harmless, and files without it drop the all-NaN column below.
        labels = [
            "mini",
            "feh",
            "eep",
            "smf",
            "afe",
            "loga",
            "logl",
            "logt",
            "logg",
            "Mr",
            "agewt",
        ]

    # Read in models.
    try:
        f = h5py.File(filepath, "r", libver="latest", swmr=True)
    except (OSError, ValueError):
        f = h5py.File(filepath, "r")
    with f:
        mag_coeffs_dataset = f["mag_coeffs"]

        # Find which requested filters actually exist in the file
        available_filters = list(mag_coeffs_dataset.dtype.names)
        valid_filters = [filt for filt in filters if filt in available_filters]
        missing_filters = [filt for filt in filters if filt not in available_filters]
        if missing_filters:
            # The returned model columns follow `valid_filters`; callers who
            # zip their original filter list against the columns would
            # silently mislabel every band, so report the survivors.
            warnings.warn(
                f"Requested filters not found in '{filepath}' and skipped: "
                f"{missing_filters}. The returned model columns correspond "
                f"(in order) to: {valid_filters}.",
                UserWarning,
                stacklevel=2,
            )

        if verbose:
            sys.stderr.write(
                f"Reading entire dataset ({len(available_filters)} filters) once...\n"
            )

        # Read the ENTIRE dataset once into memory (this is the key optimization!)
        mag_coeffs = mag_coeffs_dataset[:]

        if verbose:
            sys.stderr.write(
                f"Extracting {len(valid_filters)} requested filters "
                f"from memory: {valid_filters}\n"
            )

        # Pre-allocate array for only the valid filters
        models = np.zeros((len(mag_coeffs), len(valid_filters), 3), dtype="float32")

        # Extract each valid filter from the in-memory data (no more H5 I/O!)
        for i, filt in enumerate(valid_filters):
            try:
                models[:, i] = mag_coeffs[filt]  # Extract from memory, not H5!
            except KeyError:
                pass

        # Update filters list to only include the ones we actually loaded
        filters = valid_filters

        # Read in labels.
        combined_labels = np.full(
            len(models), np.nan, dtype=np.dtype([(n, np.float64) for n in labels])
        )
        label_mask = np.zeros(1, dtype=np.dtype([(n, np.bool_) for n in labels]))
        try:
            # Grab "labels" (inputs).
            flabels = f["labels"][:]
            for n in flabels.dtype.names:
                if n in labels:
                    combined_labels[n] = flabels[n]
                    label_mask[n] = True
        except KeyError:
            pass
        try:
            # Grab "parameters" (predictions from labels).
            fparams = f["parameters"][:]
            for n in fparams.dtype.names:
                if n in labels:
                    combined_labels[n] = fparams[n]
        except KeyError:
            pass

    # Remove extraneous/undefined labels. A label is available if ANY model
    # defines it (all-NaN column = absent from the file); checking only the
    # first row would silently drop every predicted-parameter column when
    # row 0 happens to be an invalid model.
    labels2 = [
        label for label in labels if not np.all(np.isnan(combined_labels[label]))
    ]

    # Apply cuts. Start from a finiteness cut: user-generated grids written
    # by older GridGenerator versions may contain all-NaN rows for invalid
    # grid points, which crash BruteForce fitting if they survive loading.
    sel = np.isfinite(models).all(axis=(1, 2))
    n_nonfinite = len(models) - int(sel.sum())
    if n_nonfinite > 0 and verbose:
        sys.stderr.write(
            f"Dropping {n_nonfinite} models with non-finite "
            f"photometric coefficients...\n"
        )
    if not include_ms and not include_postms:
        raise ValueError(
            "If you don't include the Main Sequence and "
            "Post-Main Sequence models you have nothing left!"
        )
    elif include_postms and not include_ms:
        # Gate on actual availability: when the file lacks 'eep' the column
        # is all-NaN and `NaN > 454.` would silently select ZERO models.
        if "eep" in labels2:
            sel &= combined_labels["eep"] > 454.0
    elif include_ms and not include_postms:
        if "eep" in labels2:
            sel &= combined_labels["eep"] <= 454.0
    # else: include_ms and include_postms — no evolutionary-phase cut

    if not include_binaries and "smf" in labels2:
        sel &= combined_labels["smf"] == 0.0
        labels2 = [x for x in labels2 if x != "smf"]

    # Compile results.
    combined_labels = combined_labels[labels2]
    label_mask = label_mask[labels2]

    return models[sel], combined_labels[sel], label_mask


def load_offsets(filepath, filters=None, verbose=True):
    """
    Loads multiplicative photometric offsets used to calibrate
    systematic differences between observed and synthetic photometry.

    Parameters
    ----------
    filepath : str
        The filepath of the photometric offsets file (typically .txt format).

    filters : iterable of str with length `Nfilt`, optional
        List of filters that will be loaded. If not provided, will default
        to all available filters. See the internally-defined `FILTERS` variable
        for more details on filter names. Any filters that are not available
        will be skipped over.

    verbose : bool, optional
        Whether to print a summary of the offsets. Default is `True`.

    Returns
    -------
    offsets : `~numpy.ndarray` of shape `(Nfilt)`
        Array of constants that will be *multiplied* to the *data* to account
        for offsets (i.e. multiplicative flux offsets). Values are typically
        close to 1.0, with deviations indicating systematic differences.

    Notes
    -----
    The offset file should contain two columns: filter names and offset values.
    Filters not found in the file will be assigned an offset of 1.0 (no correction).

    Examples
    --------
    >>> from brutus.data import load_offsets
    >>> offsets = load_offsets('./data/DATAFILES/offsets_mist_v9.txt')
    >>> print(f"Loaded offsets for {len(offsets)} filters")

    >>> # Load specific filters
    >>> gri_offsets = load_offsets('./data/DATAFILES/offsets_mist_v9.txt',
    ...                            filters=['g', 'r', 'i'])

    >>> # Check which filters have significant offsets
    >>> significant = np.abs(offsets - 1.0) > 0.01
    >>> print(f"Filters with >1% offsets: {np.sum(significant)}")
    """
    # Initialize values.
    if filters is None:
        filters = FILTERS
    Nfilters = len(filters)

    # Read in offsets. numpy.loadtxt may return a 2D array (rows x cols)
    # where transposing gives columns, or tests may mock it to return a
    # tuple of (filts, vals). Handle both cases robustly.
    _tmp = np.loadtxt(filepath, dtype="str")
    if isinstance(_tmp, tuple):
        filts, vals = _tmp
    else:
        # np.loadtxt collapses a single data row to shape (2,); atleast_2d
        # promotes it back to (Nrows, 2) so a single-filter offsets file does
        # not crash (the old arr.T unpack yielded 0-d scalars and a later
        # "nonzero on 0d arrays" ValueError).
        arr = np.atleast_2d(np.asarray(_tmp))
        filts, vals = arr[:, 0], arr[:, 1]
    vals = vals.astype(float)

    # Fill in offsets where appropriate.
    offsets = np.full(Nfilters, np.nan)
    for i, filt in enumerate(filters):
        filt_idx = np.where(filts == filt)[0]  # get filter location
        if len(filt_idx) == 1:
            offsets[i] = vals[filt_idx[0]]  # insert offset
        elif len(filt_idx) == 0:
            offsets[i] = 1.0  # assume no offset if not calibrated
        else:
            raise ValueError(
                "Something went wrong when extracting "
                "offsets for filter {}.".format(filt)
            )

    if verbose:
        for filt, zp in zip(filters, offsets):
            sys.stderr.write("{0} ({1:3.2}%)\n".format(filt, 100 * (zp - 1.0)))

    return offsets
