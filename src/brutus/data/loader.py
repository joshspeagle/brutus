#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading utilities for brutus.

This module contains functions for loading stellar evolution models,
photometric offsets, and other data files into memory for use in
stellar fitting and analysis.
"""

import sys

import h5py
import numpy as np

# Import filter definitions
from .filters import FILTERS

__all__ = ["load_models", "load_offsets"]


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
        will be skipped over.

    labels : iterable of str with length `Nlabel`, optional
        List of labels associated with the set of imported stellar models.
        Any labels that are not available will be skipped over.
        The default set is `['mini', 'feh', 'eep', 'smf', 'loga', 'logl',
        'logt', 'logg', 'Mr', 'agewt']`.

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
        labels = [
            "mini",
            "feh",
            "eep",
            "smf",
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
    mag_coeffs_dataset = f["mag_coeffs"]

    # Find which requested filters actually exist in the file
    available_filters = list(mag_coeffs_dataset.dtype.names)
    valid_filters = [filt for filt in filters if filt in available_filters]

    if verbose:
        sys.stderr.write(
            f"Reading entire dataset ({len(available_filters)} filters) once...\n"
        )

    # Read the ENTIRE dataset once into memory (this is the key optimization!)
    mag_coeffs = mag_coeffs_dataset[:]

    if verbose:
        sys.stderr.write(
            f"Extracting {len(valid_filters)} requested filters from memory...\n"
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

    # Remove extraneous/undefined labels.
    labels2 = [l for i, l in zip(combined_labels[0], labels) if ~np.isnan(i)]

    # Apply cuts.
    sel = np.ones(len(combined_labels), dtype="bool")
    if include_ms and include_postms:
        sel = np.ones(len(combined_labels), dtype="bool")
    elif not include_ms and not include_postms:
        raise ValueError(
            "If you don't include the Main Sequence and "
            "Post-Main Sequence models you have nothing left!"
        )
    elif include_postms:
        try:
            sel = combined_labels["eep"] > 454.0
        except KeyError:
            pass
    elif include_ms:
        try:
            sel = combined_labels["eep"] <= 454.0
        except KeyError:
            pass
    else:
        raise RuntimeError("Something has gone horribly wrong!")

    if not include_binaries and "smf" in labels2:
        try:
            sel *= combined_labels["smf"] == 0.0
            labels2 = [x for x in labels2 if x != "smf"]
        except KeyError:
            pass

    # Compile results.
    combined_labels = combined_labels[labels2]
    label_mask = label_mask[labels2]

    # Close file
    f.close()

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
        arr = np.asarray(_tmp)
        # Expecting shape (Nrows, 2) -> transpose to get two columns
        filts, vals = arr.T
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
