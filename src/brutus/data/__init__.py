#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus data module: Data downloading and loading utilities.

This module provides functions for downloading and loading stellar evolution
models, isochrones, dust maps, and other data files required by brutus.
"""

# Import data downloading functions
from .download import (
    fetch_dustmaps,
    fetch_grids,
    fetch_isos,
    fetch_nns,
    fetch_offsets,
    fetch_tracks,
)

# Import data loading functions
from .loader import find_nn_file, load_models, load_offsets

__all__ = [
    # Data downloading
    "fetch_isos",
    "fetch_tracks",
    "fetch_dustmaps",
    "fetch_grids",
    "fetch_offsets",
    "fetch_nns",
    # Data loading
    "find_nn_file",
    "load_models",
    "load_offsets",
]
