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
