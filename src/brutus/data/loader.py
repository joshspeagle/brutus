#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading utilities for brutus.

This module contains functions for loading stellar evolution models,
photometric offsets, and other data files into memory for use in
stellar fitting and analysis.
"""

import sys
import warnings

import h5py
import numpy as np

# Import filter definitions
from .download import _find_data_file
from .filters import FILTERS

__all__ = ["find_nn_file", "load_models", "load_offsets"]


def find_nn_file(possible_names=("nn_c3k.h5", "nnMIST_BC.h5")):
    """
    Find the neural network model file in the brutus data directories.

    Searches for the neural network HDF5 file used for bolometric correction
    predictions in the brutus download directory (the Pooch cache, or
    ``$ASTRO_BRUTUS_DATA_DIR`` when set), then in ``data/DATAFILES`` in the
    repository.

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
    TypeError
        If `possible_names` is a single string.
    """
    if isinstance(possible_names, str):
        raise TypeError(
            "possible_names must be a tuple of file names, not a string; "
            f"use ({possible_names!r},)."
        )
    return _find_data_file(possible_names, "fetch_nns")


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
        Labels that are not in the file are skipped, with a `UserWarning`
        that lists them; the default set skips its absent labels silently.
        The default set is `['mini', 'feh', 'eep', 'smf', 'afe', 'loga',
        'logl', 'logt', 'logg', 'Mr', 'agewt']`.

    include_ms : bool, optional
        Whether to include objects on the Main Sequence (`eep <= 454`).
        The cut reads the file's `'eep'` column whatever `labels` holds.
        Default is `True`.

    include_postms : bool, optional
        Whether to include objects evolved off the Main Sequence
        (`eep > 454`). The cut reads the file's `'eep'` column whatever
        `labels` holds. Default is `True`.

    include_binaries : bool, optional
        Whether to include unresolved binaries. If `False`, only models with
        secondary mass fraction `smf == 0` are kept, read from the file's
        `'smf'` column whatever `labels` holds, and `'smf'` is not returned
        as a label. A file without an `'smf'` column is taken to hold no
        binaries. Default is `False`.

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
        If `include_ms` and `include_postms` are both `False`, or if
        `include_ms` or `include_postms` is `False` and the file has no
        `'eep'` column.

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
    ...     filters=['PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']
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
    if not include_ms and not include_postms:
        raise ValueError(
            "include_ms and include_postms are both False, so no models "
            "would remain."
        )

    # Initialize values.
    if filters is None:
        filters = FILTERS
    labels_defaulted = labels is None
    if labels_defaulted:
        # 'afe' is a grid input (grids are generated on a 5D
        # (mini, eep, feh, afe, smf) lattice); omitting it makes multi-afe
        # grids ambiguous. For single-afe grids the constant column carries
        # no information and is dropped below (matching how a constant 'smf'
        # is dropped when include_binaries=False); pass `labels` explicitly
        # including 'afe' to keep the constant column.
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
        if not valid_filters:
            # Fail fast: a zero-filter model array (Nmodel, 0, 3) would only
            # produce a confusing shape error later in StarGrid/BruteForce.
            raise ValueError(
                f"None of the requested filters {list(filters)} exist in "
                f"'{filepath}'. Available filters: {available_filters}."
            )
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
            models[:, i] = mag_coeffs[filt]  # Extract from memory, not H5!

        # Update filters list to only include the ones we actually loaded
        filters = valid_filters

        # Read in labels.
        combined_labels = np.full(
            len(models), np.nan, dtype=np.dtype([(n, np.float64) for n in labels])
        )
        label_mask = np.zeros(1, dtype=np.dtype([(n, np.bool_) for n in labels]))
        # The phase and binary cuts read the file's columns, so they apply
        # whatever labels the caller requested.
        cut_columns = {}
        try:
            # Grab "labels" (inputs).
            flabels = f["labels"][:]
            for n in flabels.dtype.names:
                if n in labels:
                    combined_labels[n] = flabels[n]
                    label_mask[n] = True
                if n in ("eep", "smf"):
                    cut_columns[n] = flabels[n]
        except KeyError:
            pass
        try:
            # Grab "parameters" (predictions from labels).
            fparams = f["parameters"][:]
            for n in fparams.dtype.names:
                if n in labels:
                    combined_labels[n] = fparams[n]
                if n in ("eep", "smf"):
                    cut_columns.setdefault(n, fparams[n])
        except KeyError:
            pass

    # Remove extraneous/undefined labels. A label is available if ANY model
    # defines it (all-NaN column = absent from the file); checking only the
    # first row would silently drop every predicted-parameter column when
    # row 0 happens to be an invalid model.
    labels2 = [
        label for label in labels if not np.all(np.isnan(combined_labels[label]))
    ]
    missing_labels = [label for label in labels if label not in labels2]
    if missing_labels and not labels_defaulted:
        warnings.warn(
            f"Requested labels not found in '{filepath}' and skipped: "
            f"{missing_labels}.",
            UserWarning,
            stacklevel=2,
        )

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
    eep = cut_columns.get("eep")
    if not (include_ms and include_postms) and eep is None:
        raise ValueError(
            "include_ms=False or include_postms=False needs an 'eep' "
            f"column, which '{filepath}' does not have."
        )
    if not include_ms:
        sel &= eep > 454.0
    elif not include_postms:
        sel &= eep <= 454.0

    if not include_binaries:
        if "smf" in cut_columns:
            sel &= cut_columns["smf"] == 0.0
        labels2 = [x for x in labels2 if x != "smf"]

    # The default label set includes 'afe' only to disambiguate multi-afe
    # grids. When the file is single-afe the column is constant and carries
    # no information, so tracking it would just change the output schema
    # (e.g. a 4th "grid parameter" in BruteForce); drop it, mirroring the
    # constant-'smf' convention above. Explicitly requested labels are
    # always honored.
    if labels_defaulted and "afe" in labels2:
        afe_vals = combined_labels["afe"]
        afe_finite = afe_vals[np.isfinite(afe_vals)]
        if afe_finite.size > 0 and np.all(afe_finite == afe_finite[0]):
            labels2 = [x for x in labels2 if x != "afe"]
            if verbose:
                sys.stderr.write(
                    "Dropping constant 'afe' label column "
                    f"(single-afe grid, afe={afe_finite[0]:g})...\n"
                )

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
        for more details on filter names. A filter that is not in the file
        gets an offset of 1.0, with a `UserWarning` when the name is not in
        `FILTERS` either.

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

    Examples
    --------
    >>> from brutus.data import load_offsets
    >>> offsets = load_offsets('./data/DATAFILES/offsets_mist_v9.txt')
    >>> print(f"Loaded offsets for {len(offsets)} filters")

    >>> # Load specific filters
    >>> gri_offsets = load_offsets('./data/DATAFILES/offsets_mist_v9.txt',
    ...                            filters=['PS_g', 'PS_r', 'PS_i'])

    >>> # Check which filters have significant offsets
    >>> significant = np.abs(offsets - 1.0) > 0.01
    >>> print(f"Filters with >1% offsets: {np.sum(significant)}")
    """
    # Initialize values.
    if filters is None:
        filters = FILTERS
    Nfilters = len(filters)

    # np.loadtxt collapses a single data row to shape (2,); atleast_2d
    # promotes it back to (Nrows, 2).
    arr = np.atleast_2d(np.loadtxt(filepath, dtype="str"))
    filts, vals = arr[:, 0], arr[:, 1].astype(float)

    unknown_filters = [f for f in filters if f not in FILTERS and f not in filts]
    if unknown_filters:
        warnings.warn(
            f"Filter names {unknown_filters} are neither in brutus.FILTERS nor "
            f"in '{filepath}', so they get an offset of 1.0.",
            UserWarning,
            stacklevel=2,
        )

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
            sys.stderr.write("{0} ({1:+.2f}%)\n".format(filt, 100 * (zp - 1.0)))

    return offsets
