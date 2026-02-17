#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared utilities for brutus tutorials.

Provides consistent plotting styles, data loading helpers, output
management, assertion helpers, and standardized setup for all
tutorial scripts and notebooks.
"""

import os
from pathlib import Path

import numpy as np
from matplotlib import rcParams


def set_plot_style():
    """
    Set consistent matplotlib style for all tutorial plots.

    Uses larger fonts and plot elements suitable for documentation
    and presentation.
    """
    rcParams.update(
        {
            "xtick.major.pad": "7.0",
            "xtick.major.size": "7.5",
            "xtick.major.width": "1.5",
            "xtick.minor.pad": "7.0",
            "xtick.minor.size": "3.5",
            "xtick.minor.width": "1.0",
            "ytick.major.pad": "7.0",
            "ytick.major.size": "7.5",
            "ytick.major.width": "1.5",
            "ytick.minor.pad": "7.0",
            "ytick.minor.size": "3.5",
            "ytick.minor.width": "1.0",
            "axes.titlepad": "15.0",
            "axes.labelpad": "15.0",
            "font.size": 14,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "grid.alpha": 0.3,
        }
    )


def get_plot_dir(tutorial_num):
    """
    Get (and create if needed) the plot output directory for a tutorial.

    Parameters
    ----------
    tutorial_num : int
        Tutorial number (0-8).

    Returns
    -------
    plot_dir : Path
        Path to the plot directory for this tutorial.
    """
    tutorials_dir = Path(__file__).parent
    plot_dir = tutorials_dir / "plots" / f"tutorial_{tutorial_num:02d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def save_figure(fig, tutorial_num, fig_name, tight=True, **kwargs):
    """
    Save a figure to the appropriate tutorial plot directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    tutorial_num : int
        Tutorial number (0-8).
    fig_name : str
        Base name for the figure file (without extension).
    tight : bool, optional
        Whether to use tight_layout before saving. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to fig.savefig().

    Returns
    -------
    save_path : Path
        Path where the figure was saved.
    """
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass  # tight_layout sometimes fails, ignore

    plot_dir = get_plot_dir(tutorial_num)
    save_path = plot_dir / f"{fig_name}.png"

    # Set default kwargs
    save_kwargs = {"dpi": 150, "bbox_inches": "tight", "facecolor": "white"}
    save_kwargs.update(kwargs)

    fig.savefig(save_path, **save_kwargs)
    print(f"  Saved: {save_path}")

    return save_path


def print_section(title, char="="):
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title.
    char : str, optional
        Character to use for the underline. Default is '='.
    """
    print(f"\n{title}")
    print(char * len(title))


def find_brutus_data_file(filename, search_paths=None):
    """
    Find a brutus data file, checking multiple possible locations.

    This handles cross-platform paths, different mount points
    (e.g., /mnt/c/ vs /mnt/d/ in WSL), the Pooch cache directory,
    and the ``BRUTUS_DATA_DIR`` environment variable.

    Parameters
    ----------
    filename : str
        Name of the file to find (can include subdirectories).
    search_paths : list of str, optional
        Additional search paths to check. If None, uses default paths.

    Returns
    -------
    filepath : str
        Full path to the found file.

    Raises
    ------
    FileNotFoundError
        If file is not found in any search location.
    """
    if search_paths is None:
        # Default search paths
        tutorials_dir = Path(__file__).parent
        search_paths = [
            tutorials_dir,  # tutorials directory itself
            tutorials_dir / "data",
        ]

        # Pooch cache directory (used by fetch_* functions and CI)
        try:
            import pooch

            search_paths.append(Path(pooch.os_cache("astro-brutus")))
        except ImportError:
            pass

        # BRUTUS_DATA_DIR environment variable
        env_dir = os.environ.get("BRUTUS_DATA_DIR")
        if env_dir:
            search_paths.append(Path(env_dir))

        # Standard project data directories
        search_paths.extend(
            [
                tutorials_dir.parent / "data" / "DATAFILES",
                tutorials_dir.parent / "data" / "benchmarks",
                tutorials_dir.parent / "data" / "VICs",
                Path("/mnt/d/Dropbox/GitHub/brutus/data/DATAFILES"),
                Path("/mnt/d/Dropbox/GitHub/brutus/data/benchmarks"),
                Path("/mnt/d/Dropbox/GitHub/brutus/data/VICs"),
                Path("/mnt/d/Dropbox/GitHub/brutus/tutorials"),
                Path("/mnt/c/Dropbox/GitHub/brutus/data/DATAFILES"),
                Path("/mnt/c/Dropbox/GitHub/brutus/data/benchmarks"),
                Path("/mnt/c/Dropbox/GitHub/brutus/data/VICs"),
                Path("../data/DATAFILES"),
                Path("../data/benchmarks"),
                Path("../data/VICs"),
                Path("./data"),
            ]
        )

    for base_path in search_paths:
        base_path = Path(base_path)
        full_path = base_path / filename

        if full_path.exists():
            return str(full_path)

    # File not found
    raise FileNotFoundError(
        f"Could not find '{filename}' in any of the following locations:\n"
        + "\n".join([f"  - {p}" for p in search_paths])
    )


def load_tutorial_data(dataset_name, **kwargs):
    """
    Load common tutorial datasets with consistent handling.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Options:
        - 'orion': Orion field data from Tutorial 3
        - 'm67' or 'ngc2682': M67 cluster data from Tutorial 6
    **kwargs : dict
        Additional keyword arguments for dataset-specific loading.

    Returns
    -------
    data : dict
        Dictionary containing loaded data arrays and metadata.
    """
    import h5py

    from brutus.utils import inv_magnitude

    if dataset_name.lower() == "orion":
        # Load Orion field data
        try:
            filename = find_brutus_data_file("Orion_l204.7_b-19.2.h5")
        except FileNotFoundError:
            # Try tutorial directory
            filename = find_brutus_data_file(
                "Orion_l204.7_b-19.2.h5", [Path(__file__).parent]
            )

        with h5py.File(filename, "r") as f:
            fpix = f["photometry"]["pixel 0-0"]
            mag, magerr = fpix["mag"][:], fpix["err"][:]
            mask = np.isfinite(magerr)
            phot, err = inv_magnitude(mag, magerr)
            parallax = fpix["parallax"][:] * 1e3  # to mas
            parallax_err = fpix["parallax_error"][:] * 1e3

            # Screen bad parallaxes
            psel = (
                np.isclose(parallax_err, 0.0)
                | np.isclose(parallax, 0.0)
                | (parallax_err > 1e6)
            )
            parallax[psel], parallax_err[psel] = np.nan, np.nan

            coords = np.c_[fpix["l"][:], fpix["b"][:]]

            return {
                "phot": phot,
                "err": err,
                "mask": mask,
                "parallax": parallax,
                "parallax_err": parallax_err,
                "coords": coords,
                "obj_id": fpix["obj_id"][:],
                "filename": filename,
            }

    elif dataset_name.lower() in ["m67", "ngc2682"]:
        # Load M67 cluster data
        try:
            filename = find_brutus_data_file("NGC_2682.fits")
        except FileNotFoundError:
            filename = find_brutus_data_file("NGC_2682.fits", [Path(__file__).parent])

        from astropy.io import fits

        with fits.open(filename) as f:
            fdata = f[1].data

        return {
            "data": fdata,
            "filename": filename,
        }

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Options are: 'orion', 'm67', 'ngc2682'"
        )


def check_data_requirements(tutorial_num, verbose=True):
    """
    Check if required data files are available for a tutorial.

    Parameters
    ----------
    tutorial_num : int
        Tutorial number (0-11).
    verbose : bool, optional
        Whether to print detailed status. Default is True.

    Returns
    -------
    available : bool
        True if all required files are available.
    missing : list of str
        List of missing files.
    """
    # Define requirements per tutorial
    requirements = {
        0: [],
        1: ["nn_c3k.h5", "MIST_1.2_EEPtrk.h5"],
        2: ["nn_c3k.h5", "MIST_1.2_iso_vvcrit0.0.h5"],
        3: [
            "nn_c3k.h5",
            "MIST_1.2_EEPtrk.h5",
            "grid_mist_v9.h5",
            "grid_bayestar_v5.h5",
        ],
        4: ["nn_c3k.h5", "MIST_1.2_iso_vvcrit0.0.h5", "bayestar2019_v1.h5"],
        5: [
            "grid_mist_v9.h5",
            "grid_bayestar_v5.h5",
            "offsets_mist_v8.txt",
            "offsets_bs_v5.txt",
            "bayestar2019_v1.h5",
            "Orion_l204.7_b-19.2.h5",
        ],
        6: [
            "nn_c3k.h5",
            "MIST_1.2_iso_vvcrit0.0.h5",
            "offsets_mist_v8.txt",
            "NGC_2682.fits",
        ],
        7: ["Orion_l204.7_b-19.2_mist.h5"],
        8: [
            "grid_mist_v9.h5",
            "grid_bayestar_v5.h5",
            "offsets_mist_v9.txt",
            "offsets_bs_v5.txt",
        ],
        9: [],
        10: ["Orion_l204.7_b-19.2_mist.h5"],
        11: ["Orion_l204.7_b-19.2_mist.h5"],
    }

    required_files = requirements.get(tutorial_num, [])
    missing = []

    if verbose:
        print_section(f"Checking data requirements for Tutorial {tutorial_num}")

    for filename in required_files:
        try:
            find_brutus_data_file(filename)  # Just check if file exists
            if verbose:
                print(f"  Found: {filename}")
        except FileNotFoundError:
            missing.append(filename)
            if verbose:
                print(f"  Missing: {filename}")

    if verbose:
        if missing:
            print(f"\n  Missing {len(missing)} required file(s)")
        else:
            print("\n  All required files available")

    return len(missing) == 0, missing


def format_runtime(seconds):
    """
    Format runtime in human-readable format.

    Parameters
    ----------
    seconds : float
        Runtime in seconds.

    Returns
    -------
    formatted : str
        Formatted runtime string.
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


# Convenience functions for specific datasets
def load_orion_data(**kwargs):
    """Load Orion field data. Convenience wrapper for load_tutorial_data."""
    return load_tutorial_data("orion", **kwargs)


def load_m67_data(**kwargs):
    """Load M67 cluster data. Convenience wrapper for load_tutorial_data."""
    return load_tutorial_data("m67", **kwargs)


def setup_tutorial(tutorial_num, title=None):
    """
    Standardized bootstrap for every tutorial notebook.

    Sets plot style, creates output directory, checks data availability,
    and prints a status header.

    Parameters
    ----------
    tutorial_num : int
        Tutorial number (0-11).
    title : str, optional
        Tutorial title for the header. If None, uses a generic title.

    Returns
    -------
    info : dict
        Dictionary with keys:
        - ``plot_dir``: Path to the plot output directory.
        - ``data_available``: bool, whether all required data are available.
        - ``missing_files``: list of missing data file names.
    """
    import matplotlib

    # Set headless backend if not interactive
    if "MPLBACKEND" in os.environ:
        matplotlib.use(os.environ["MPLBACKEND"])

    set_plot_style()

    plot_dir = get_plot_dir(tutorial_num)

    if title is None:
        title = f"Tutorial {tutorial_num:02d}"
    print_section(title)

    data_ok, missing = check_data_requirements(tutorial_num, verbose=True)

    return {
        "plot_dir": plot_dir,
        "data_available": data_ok,
        "missing_files": missing,
    }


def load_example_results(filename="Orion_l204.7_b-19.2_mist.h5"):
    """
    Load pre-computed BruteForce results for plotting/results tutorials.

    Parameters
    ----------
    filename : str, optional
        Results filename to load.

    Returns
    -------
    results : dict
        Dictionary with result arrays loaded from the HDF5 file.
    """
    import h5py

    filepath = find_brutus_data_file(filename)

    results = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                results[key] = f[key][:]
            elif isinstance(f[key], h5py.Group):
                results[key] = {}
                for subkey in f[key].keys():
                    if isinstance(f[key][subkey], h5py.Dataset):
                        results[key][subkey] = f[key][subkey][:]

    results["filename"] = filepath
    return results


# ---------------------------------------------------------------------------
# Assertion helpers for programmatic output verification
# ---------------------------------------------------------------------------


def assert_figure_valid(fig, min_axes=1):
    """
    Assert that a matplotlib figure is valid and contains content.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to validate.
    min_axes : int, optional
        Minimum number of axes expected. Default is 1.

    Raises
    ------
    AssertionError
        If the figure is not valid.
    """
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure), "Not a matplotlib Figure"
    axes = fig.get_axes()
    assert len(axes) >= min_axes, f"Expected at least {min_axes} axes, got {len(axes)}"


def assert_array_properties(
    arr, name="array", ndim=None, shape=None, min_val=None, max_val=None, finite=False
):
    """
    Assert properties of a numpy array for output verification.

    Parameters
    ----------
    arr : array_like
        Array to check.
    name : str, optional
        Name for error messages.
    ndim : int, optional
        Expected number of dimensions.
    shape : tuple, optional
        Expected shape (use -1 for any size along a dimension).
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.
    finite : bool, optional
        Whether all values must be finite.

    Raises
    ------
    AssertionError
        If any property check fails.
    """
    arr = np.asarray(arr)
    if ndim is not None:
        assert arr.ndim == ndim, f"{name}: expected {ndim}D, got {arr.ndim}D"
    if shape is not None:
        for i, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected != -1:
                assert (
                    expected == actual
                ), f"{name}: dimension {i} expected {expected}, got {actual}"
    if min_val is not None:
        valid = arr[np.isfinite(arr)] if not finite else arr
        if len(valid) > 0:
            assert np.all(valid >= min_val), (
                f"{name}: values below minimum {min_val} "
                f"(min found: {np.nanmin(arr)})"
            )
    if max_val is not None:
        valid = arr[np.isfinite(arr)] if not finite else arr
        if len(valid) > 0:
            assert np.all(valid <= max_val), (
                f"{name}: values above maximum {max_val} "
                f"(max found: {np.nanmax(arr)})"
            )
    if finite:
        assert np.all(np.isfinite(arr)), f"{name}: contains non-finite values"
