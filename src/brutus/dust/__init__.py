"""
Three-dimensional dust extinction map interfaces and HEALPix utilities.

Adapted from the ``dustmaps`` package (Green 2018, JOSS, 3, 695) to query
Bayestar 3D dust maps (Green et al. 2019).
"""

from .extinction import lb2pix
from .maps import Bayestar, DustMap

__all__ = ["lb2pix", "DustMap", "Bayestar"]
