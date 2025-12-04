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
    models = load_models('./data/grid_mist_v9.h5')
"""

from __future__ import division, print_function

# Version management
__version__ = "1.0.0"

# Core functionality imports
try:
    # Import submodules to make them accessible as brutus.plotting, etc.
    from . import analysis, core, data, dust, plotting, priors, utils  # noqa: F401

    # Core stellar evolution models (refactored)
    # Analysis and fitting
    from .analysis import BruteForce  # noqa: F401
    from .core import (  # noqa: F401
        EEPTracks,
        Isochrone,
        StarEvolTrack,
        StarGrid,
        StellarPop,
    )

    # Data management (refactored)
    from .data import fetch_dustmaps, fetch_grids, fetch_isos, load_models  # noqa: F401

    # Essential utilities (refactored)
    from .utils import inv_magnitude, magnitude  # noqa: F401

    # Dust mapping (not yet refactored)
    # from .dust import Bayestar
    # Make key classes and submodules easily accessible
    __all__ = [
        # Version
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
        # Data utilities (refactored)
        "fetch_grids",
        "fetch_isos",
        "fetch_dustmaps",
        "load_models",
        # Photometry utilities (refactored)
        "magnitude",
        "inv_magnitude",
    ]

except ImportError as e:
    # During the transition period, some imports might fail
    # Provide graceful fallback
    import warnings

    warnings.warn(
        f"Some brutus modules are not yet available during reorganization: {e}. "
        "Please use the original module imports temporarily.",
        ImportWarning,
    )

    # Minimal fallback
    __all__ = ["__version__"]
