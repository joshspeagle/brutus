#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus: Brute-force Bayesian inference for stellar photometry

A Pure Python package for deriving distances, reddenings, and stellar
properties from photometry using "brute force" Bayesian inference.

The package is designed to be highly modular, with modules for:
- Individual star modeling and fitting
- Star cluster analysis
- 3D dust mapping
- Stellar evolution model management

Usage
-----
For individual star modeling::

    from brutus.core import EEPTracks, StarEvolTrack
    tracks = EEPTracks()
    star = StarEvolTrack(tracks=tracks)
    sed, params, params2 = star.get_seds(mini=1.0, eep=350, feh=0.0)

For stellar population modeling::

    from brutus.core import Isochrone, StellarPop
    iso = Isochrone()
    pop = StellarPop(isochrone=iso)
    seds, params, params2 = pop.get_seds(feh=0.0, afe=0.0, loga=9.0)

For data management::

    from brutus import fetch_grids, load_models
    fetch_grids(target_dir='./data/')
    models, labels, label_mask = load_models('./data/grid_mist_v9.h5')
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("astro-brutus")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "unknown"

# _numba_cache comes first: it must install its cache locators before any
# kernel is decorated.
from . import _numba_cache, analysis, core, data, dust, plotting, priors, utils
from .analysis import BruteForce
from .core import EEPTracks, Isochrone, StarEvolTrack, StarGrid, StellarPop
from .data import (
    fetch_dustmaps,
    fetch_grids,
    fetch_isos,
    fetch_nns,
    fetch_offsets,
    fetch_tracks,
    load_models,
    load_offsets,
)
from .utils import inv_magnitude, magnitude

__all__ = [
    "__version__",
    # Submodules
    "core",
    "analysis",
    "data",
    "utils",
    "priors",
    "plotting",
    "dust",
    # Core classes
    "Isochrone",
    "EEPTracks",
    "StarGrid",
    "StarEvolTrack",
    "StellarPop",
    # Analysis classes
    "BruteForce",
    # Data utilities
    "fetch_dustmaps",
    "fetch_grids",
    "fetch_isos",
    "fetch_nns",
    "fetch_offsets",
    "fetch_tracks",
    "load_models",
    "load_offsets",
    # Photometry utilities
    "magnitude",
    "inv_magnitude",
]

del _numba_cache
