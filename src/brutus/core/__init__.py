"""
Predict stellar parameters and synthetic photometry from evolutionary models.

Provides evolutionary tracks (`EEPTracks`, `StarEvolTrack`), pre-computed model
grids (`StarGrid`, `GridGenerator`), isochrones (`Isochrone`, `StellarPop`),
neural network bolometric corrections (`FastNN`, `FastNNPredictor`), and
reddened SED evaluation (`get_seds`).
"""

from .grid_generation import GridGenerator
from .individual import EEPTracks, StarEvolTrack, StarGrid
from .neural_nets import FastNN, FastNNPredictor
from .populations import Isochrone, StellarPop
from .sed_utils import get_seds

__all__ = [
    "get_seds",
    "FastNN",
    "FastNNPredictor",
    "EEPTracks",
    "StarGrid",
    "StarEvolTrack",
    "Isochrone",
    "StellarPop",
    "GridGenerator",
]
