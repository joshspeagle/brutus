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

TUTORIALS_DIR = Path(__file__).resolve().parent

#: Data files of each tutorial, as (required, optional)
DATA_REQUIREMENTS = {
    0: ([], ["grid_mist_v9.h5", "grid_bayestar_v5.h5", "offsets_mist_v9.txt"]),
    1: (["nn_c3k.h5", "MIST_1.2_EEPtrk.h5"], []),
    2: (["nn_c3k.h5", "MIST_1.2_iso_vvcrit0.0.h5"], []),
    3: (
        ["nn_c3k.h5", "MIST_1.2_EEPtrk.h5"],
        ["grid_mist_v9.h5", "grid_bayestar_v5.h5"],
    ),
    4: ([], ["bayestar2019_v1.h5"]),
    5: (
        ["grid_mist_v9.h5", "Orion_l209.1_b-19.9.h5"],
        ["offsets_mist_v9.txt", "bayestar2019_v1.h5"],
    ),
    6: (["nn_c3k.h5", "MIST_1.2_iso_vvcrit0.0.h5", "NGC_2682.fits"], []),
    7: ([], ["Orion_l209.1_b-19.9_mist_nodust.h5"]),
    8: (["grid_mist_v9.h5"], []),
    9: ([], []),
    10: (
        ["Orion_l209.1_b-19.9_mist.h5"],
        ["Orion_l209.1_b-19.9.h5", "grid_mist_v9.h5"],
    ),
    11: (["Orion_l209.1_b-19.9_mist.h5"], ["Orion_l209.1_b-19.9.h5"]),
}

#: Data files that a tutorial writes rather than a fetch_* function
WRITTEN_BY = {
    "Orion_l209.1_b-19.9_mist.h5": "Tutorial 5",
    "Orion_l209.1_b-19.9_mist_nodust.h5": "Tutorial 5, with the optional dust map",
}

# Files recorded by record_file in this session, in order
_saved_files = {}


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
    plot_dir = TUTORIALS_DIR / "plots" / f"tutorial_{tutorial_num:02d}"
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
    record_file(save_path)
    print(f"  Saved: {os.path.relpath(save_path, TUTORIALS_DIR)}")

    return save_path


def record_file(path):
    """Record a file written by this run, for `print_saved_files`."""
    _saved_files[Path(path)] = None


def print_saved_files():
    """Print the files this run wrote, relative to the tutorials directory."""
    print("\nGenerated files:")
    for path in _saved_files:
        print(f"  - {os.path.relpath(path, TUTORIALS_DIR)}")


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
        search_paths = [
            TUTORIALS_DIR,
            TUTORIALS_DIR / "data",
        ]

        # Pooch cache directory (used by fetch_* functions and CI)
        try:
            import pooch

            search_paths.append(Path(pooch.os_cache("astro-brutus")))
        except ImportError:
            pass

        # Data-directory environment variables. BRUTUS_DATA_DIR is the tutorial
        # convention; ASTRO_BRUTUS_DATA_DIR is what Pooch itself honors, so we
        # check both to stay consistent with the download cache override.
        for _env_name in ("BRUTUS_DATA_DIR", "ASTRO_BRUTUS_DATA_DIR"):
            env_dir = os.environ.get(_env_name)
            if env_dir:
                search_paths.append(Path(env_dir))

        # Standard project data directories (relative to the repo layout).
        search_paths.extend(
            [
                TUTORIALS_DIR.parent / "data" / "DATAFILES",
                TUTORIALS_DIR.parent / "data" / "benchmarks",
                TUTORIALS_DIR.parent / "data" / "VICs",
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
            filename = find_brutus_data_file("Orion_l209.1_b-19.9.h5")
        except FileNotFoundError:
            # Try tutorial directory
            filename = find_brutus_data_file(
                "Orion_l209.1_b-19.9.h5", [Path(__file__).parent]
            )

        with h5py.File(filename, "r") as f:
            fpix = f["photometry"]["pixel 0-0"]
            mag, magerr = fpix["mag"][:], fpix["err"][:]
            mask = np.isfinite(magerr)
            phot, err = inv_magnitude(mag, magerr)
            parallax = fpix["parallax"][:]  # already in mas
            parallax_err = fpix["parallax_error"][:]

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


def missing_note(filename):
    """Return a note on where a missing data file comes from, or ''."""
    source = WRITTEN_BY.get(filename)
    return "" if source is None else f" (written by {source})"


def check_data_requirements(tutorial_num, verbose=True):
    """
    Check if the data files of a tutorial are available.

    The files are listed in `DATA_REQUIREMENTS`. A missing optional file
    is reported but does not count as missing.

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
        List of missing required files.
    """
    if tutorial_num not in DATA_REQUIREMENTS:
        raise ValueError(
            f"tutorial_num must be one of {sorted(DATA_REQUIREMENTS)}; "
            f"got {tutorial_num}."
        )
    required, optional = DATA_REQUIREMENTS[tutorial_num]
    missing = []

    if verbose:
        print_section(f"Checking data requirements for Tutorial {tutorial_num}")

    for filename in required + optional:
        is_required = filename in required
        tag = "" if is_required else " (optional)"
        try:
            find_brutus_data_file(filename)
            if verbose:
                print(f"  Found: {filename}{tag}")
        except FileNotFoundError:
            if is_required:
                missing.append(filename)
            if verbose:
                print(f"  Missing: {filename}{tag}{missing_note(filename)}")

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


def load_example_results(filename="Orion_l209.1_b-19.9_mist.h5"):
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
