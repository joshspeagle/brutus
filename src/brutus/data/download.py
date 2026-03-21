#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data downloading utilities for brutus.

This module contains functions for downloading stellar evolution models,
isochrones, dust maps, and other data files required by brutus.
All downloads use the Pooch library for robust data management.
"""

import pathlib

__all__ = [
    "fetch_isos",
    "fetch_tracks",
    "fetch_dustmaps",
    "fetch_grids",
    "fetch_offsets",
    "fetch_nns",
]

import pooch

# Mapping of data file names to their corresponding DOIs for citation and download
_dois = {
    "MIST_1.2_iso_vvcrit0.0.h5": "10.7910/DVN/FZMFQY/BKAG41",
    "MIST_1.2_iso_vvcrit0.4.h5": "10.7910/DVN/FZMFQY/PRGJIP",
    "MIST_1.2_EEPtrk.h5": "10.7910/DVN/JV866N/FJ5NNO",
    "bayestar2019_v1.h5": "10.7910/DVN/G49MEI/Y9UZPG",
    "grid_mist_v9.h5": "10.7910/DVN/7BA4ZG/Z7MGA7",
    "grid_mist_v8.h5": "10.7910/DVN/7BA4ZG/NKVZFT",
    "grid_bayestar_v5.h5": "10.7910/DVN/7BA4ZG/LLZP0B",
    "offsets_mist_v9.txt": "10.7910/DVN/L7D1FY/ASPSUR",
    "offsets_mist_v8.txt": "10.7910/DVN/L7D1FY/QTNKKN",
    "offsets_bs_v9.txt": "10.7910/DVN/L7D1FY/W4O6NJ",
    "nn_c3k.h5": "10.7910/DVN/MSCY2O/XHU1VJ",
}

# Create a Pooch data manager for robust file downloading and caching
strato = pooch.create(
    path=pooch.os_cache("astro-brutus"),  # Cache directory for downloaded files
    base_url="https://dataverse.harvard.edu/api/access/datafile/",  # Base URL for file downloads
    registry={
        # Mapping of file names to their SHA256 hashes for integrity checking
        "MIST_1.2_iso_vvcrit0.0.h5": "ac46048acb9c9c1c10f02ac1bd958a8c4dd80498923297907fd64c5f3d82cb57",
        "MIST_1.2_iso_vvcrit0.4.h5": "25d97db9760df5e4e3b65c686a04d5247cae5027c55683e892acb7d1a05c30f7",
        "MIST_1.2_EEPtrk.h5": "001558c1b32f4a85ea9acca3ad3f7332a565167da3f6164a565c3f3f05afc11b",
        "bayestar2019_v1.h5": "73064ab18f4d1d57b356f7bd8cbcc77be836f090f660cca6727da85ed973d1e6",
        "grid_mist_v9.h5": "7d128a5caded78ca9d1788a8e6551b4329aeed9ca74e7a265e531352ecb75288",
        "grid_mist_v8.h5": "b07d9c19e7ff5e475b1b061af6d1bb4ebd13e0e894fd0703160206964f1084e0",
        "grid_bayestar_v5.h5": "c5d195430393ebd6c8865a9352c8b0906b2c43ec56d3645bb9d5b80e6739fd0c",
        "offsets_mist_v9.txt": "e2c33ee4669ff4da67bcdd6e2d7551fbe4d4b3a6f95d29bb220da622304af468",
        "offsets_mist_v8.txt": "35425281b5d828431ca5ef93262cb7c6f406814b649d7e7ca4866b8203408e5f",
        "offsets_bs_v9.txt": "b5449c08eb7b894b6d9aa1449a351851ca800ef4ed461c987434a0c250cba386",
        "nn_c3k.h5": "bc86d4bf55b2173b97435d24337579a2f337e80ed050c73f1e31abcd04163259",
    },
    env="ASTRO_BRUTUS_DATA_DIR",  # Environment variable to override cache path
    retry_if_failed=3,  # Retry downloads up to 3 times if they fail
)

# Customize the URLs for each file using their DOIs, since Pooch cannot build them automatically
strato.urls = {
    k: f"{strato.base_url}:persistentId?persistentId=doi:{v}" for k, v in _dois.items()
}


def _fetch(name, symlink_dir):
    """
    Fetch file using Pooch, creating a symlink at symlink_dir.

    This internal helper function downloads the requested file using the
    Pooch registry and creates a symbolic link in the target directory.

    Parameters
    ----------
    name : str
        Name of the file to fetch from the registry.
    symlink_dir : str or pathlib.Path
        Directory where the symlink should be created.

    Returns
    -------
    target_path : pathlib.Path
        Path to the symlinked file in the target directory.
    """
    import os

    # In CI, try to use cached file directly if it exists to avoid re-downloads
    if os.environ.get("CI") == "true":
        cache_path = pathlib.Path(strato.path) / name
        if cache_path.exists():
            # File exists in cache, use it directly without SHA verification
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(
                f"    Using cached {name} ({size_mb:.1f} MB) - skipping SHA verification"
            )
            fpath = cache_path
        else:
            # File doesn't exist, download it
            print(f"    {name} not in cache - downloading...")
            fpath = strato.fetch(name, progressbar=True)
            fpath = pathlib.Path(fpath)
    else:
        # Normal behavior with SHA256 verification
        fpath = strato.fetch(name, progressbar=True)
        fpath = pathlib.Path(fpath)

    target_path = pathlib.Path(symlink_dir).resolve() / name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        target_path.symlink_to(fpath)

    return target_path


def fetch_isos(target_dir=".", iso="MIST_1.2_vvcrit0.0"):
    """
    Download isochrone files to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    iso : str, optional
        The desired isochrone file. Available options:

        - 'MIST_1.2_vvcrit0.0' (default) : Non-rotating MIST v1.2 isochrones
        - 'MIST_1.2_vvcrit0.4' : Rotating MIST v1.2 isochrones

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded isochrone file.

    Raises
    ------
    ValueError
        If the specified isochrone file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_isos
    >>> iso_path = fetch_isos(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded isochrones to: {iso_path}")

    >>> # Download rotating models
    >>> rotating_path = fetch_isos(target_dir='./data/DATAFILES/', iso='MIST_1.2_vvcrit0.4')
    """
    if iso == "MIST_1.2_vvcrit0.0":
        name = "MIST_1.2_iso_vvcrit0.0.h5"
    elif iso == "MIST_1.2_vvcrit0.4":
        name = "MIST_1.2_iso_vvcrit0.4.h5"
    else:
        raise ValueError("The specified isochrone file does not exist!")

    return _fetch(name, target_dir)


def fetch_tracks(target_dir=".", track="MIST_1.2_vvcrit0.0"):
    """
    Download EEP (Equivalent Evolutionary Point) track files to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    track : str, optional
        The desired track file. Available options:

        - 'MIST_1.2_vvcrit0.0' (default) : Non-rotating MIST v1.2 tracks

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded evolutionary track file.

    Raises
    ------
    ValueError
        If the specified track file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_tracks
    >>> track_path = fetch_tracks(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded tracks to: {track_path}")
    """
    if track == "MIST_1.2_vvcrit0.0":
        name = "MIST_1.2_EEPtrk.h5"
    else:
        raise ValueError("The specified track file does not exist!")

    return _fetch(name, target_dir)


def fetch_dustmaps(target_dir=".", dustmap="bayestar19"):
    """
    Download 3D dust extinction map files to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    dustmap : str, optional
        The desired dust map file. Available options:

        - 'bayestar19' (default) : Bayestar dust map from Green et al. (2019)

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded dust map file.

    Raises
    ------
    ValueError
        If the specified dust map file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_dustmaps
    >>> dust_path = fetch_dustmaps(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded dust map to: {dust_path}")
    """
    if dustmap == "bayestar19":
        name = "bayestar2019_v1.h5"
    else:
        raise ValueError("The specified dustmap file does not exist!")

    return _fetch(name, target_dir)


def fetch_grids(grid="mist_v9", target_dir="."):
    """
    Downloads pre-computed stellar model grids (used for fast stellar
    parameter inference and photometric fitting) to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    grid : str, optional
        The desired grid file. Available options:

        - 'mist_v9' (default) : MIST v1.2 with empirical corrections (v9)
        - 'mist_v8' : MIST v1.2 with empirical corrections (v8)
        - 'bayestar_v5' : Bayestar models (v5)

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded stellar model grid file.

    Raises
    ------
    ValueError
        If the specified grid file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_grids
    >>> grid_path = fetch_grids(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded model grid to: {grid_path}")

    >>> # Download older version
    >>> old_grid = fetch_grids(target_dir='./data/DATAFILES/', grid='mist_v8')
    """
    if grid == "mist_v9":
        name = "grid_mist_v9.h5"
    elif grid == "mist_v8":
        name = "grid_mist_v8.h5"
    elif grid == "bayestar_v5":
        name = "grid_bayestar_v5.h5"
    else:
        raise ValueError("The specified grid file does not exist!")

    return _fetch(name, target_dir)


def fetch_offsets(target_dir=".", grid="mist_v9"):
    """
    Downloads photometric offset files (used to calibrate systematic
    differences between observed and model photometry) to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    grid : str, optional
        The associated model grid for the offsets. Available options:

        - 'mist_v9' (default) : Offsets for MIST v1.2 with corrections (v9)
        - 'mist_v8' : Offsets for MIST v1.2 with corrections (v8)
        - 'bayestar_v5' : Offsets for Bayestar models (v5)

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded photometric offset file.

    Raises
    ------
    ValueError
        If the specified offset file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_offsets
    >>> offset_path = fetch_offsets(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded offsets to: {offset_path}")
    """
    if grid == "mist_v9":
        name = "offsets_mist_v9.txt"
    elif grid == "mist_v8":
        name = "offsets_mist_v8.txt"
    elif grid == "bayestar_v5":
        name = "offsets_bs_v9.txt"
    else:
        raise ValueError("The specified grid file does not exist!")

    return _fetch(name, target_dir)


def fetch_nns(target_dir=".", model="c3k"):
    """
    Download pre-trained neural network model files (used for fast spectral energy
    distribution prediction) to target directory.

    Parameters
    ----------
    target_dir : str, optional
        The target directory where the file should be downloaded. If not
        specified, files will be downloaded to the current directory.
        Default is ".".

    model : str, optional
        The desired neural network model file. Available options:

        - 'c3k' (default) : Network trained on C3K spectral models

    Returns
    -------
    file_path : pathlib.Path
        Path to the downloaded neural network file.

    Raises
    ------
    ValueError
        If the specified neural network file does not exist in the registry.

    Examples
    --------
    >>> from brutus.data import fetch_nns
    >>> nn_path = fetch_nns(target_dir='./data/DATAFILES/')
    >>> print(f"Downloaded neural network to: {nn_path}")
    """
    if model == "c3k":
        name = "nn_c3k.h5"
    else:
        raise ValueError("The specified neural network file does not exist!")

    return _fetch(name, target_dir)
