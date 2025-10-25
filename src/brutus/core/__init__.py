#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus core module: Fundamental stellar modeling utilities.

This module contains the core functionality for stellar evolution modeling,
including isochrones, tracks, SED computation, neural network utilities,
and grid generation.
"""

from .grid_generation import GridGenerator
from .individual import EEPTracks, StarEvolTrack, StarGrid
from .neural_nets import FastNN, FastNNPredictor
from .populations import Isochrone, StellarPop

# Import core components
from .sed_utils import _get_seds, get_seds

__all__ = [
    "_get_seds",
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
