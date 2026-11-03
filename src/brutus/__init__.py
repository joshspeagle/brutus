"""
brutus: Brute-force Bayesian inference for stellar photometry.

Package for deriving distances, reddenings, and stellar properties from
photometry and astrometry by evaluating Bayesian posteriors over precomputed
stellar model grids:
- Individual star modeling and fitting (`brutus.core`, `brutus.analysis`)
- Star cluster isochrone analysis (`brutus.core`, `brutus.analysis`)
- 3D dust mapping (`brutus.dust`, `brutus.analysis`)
- Stellar evolution model and grid management (`brutus.data`)

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

# _numba_cache comes first because it must install its cache locators before
# any kernel is decorated.
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
