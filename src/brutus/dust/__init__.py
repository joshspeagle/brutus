#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus dust module: 3D dust mapping utilities.

This module provides tools for working with 3D dust maps, including coordinate
transformations and dust extinction queries. It's based on the `dustmaps`
package by Greg Green (Green et al. 2018) and implements interfaces for the
Bayestar 3D dust maps.

Classes
-------
DustMap : Abstract base class for dust maps
Bayestar : Implementation for Bayestar 3D dust maps

Functions
---------
lb2pix : Convert Galactic coordinates to HEALPix indices
"""

# Import from submodules
from .extinction import lb2pix
from .maps import Bayestar, DustMap

__all__ = ["lb2pix", "DustMap", "Bayestar"]
