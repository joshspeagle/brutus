#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Available filters.

"""

from __future__ import (print_function, division)
import six
from six.moves import range

# Define the set of filters available for the provided MIST models.
# The Bayestar models are only defined for PanSTARRS `grizy` and 2MASS.
gaia = ["Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev"]
sdss = ["SDSS_{}".format(b) for b in "ugriz"]
ps = ["PS_{}".format(b) for b in ["g", "r", "i", "z", "y", "w", "open"]]
decam = ["DECam_{}".format(b) for b in "ugrizY"]
tycho = ["Tycho_B", "Tycho_V"]
bessell = ["Bessell_{}".format(b) for b in "UBVRI"]
tmass = ["2MASS_{}".format(b) for b in ["J", "H", "Ks"]]
ukidss = ["UKIDSS_{}".format(b) for b in "ZYJHK"]
wise = ["WISE_W{}".format(b) for b in "1234"]
hipp = ["Hipparcos_Hp"]
kepler = ["Kepler_D51", "Kepler_Kp"]
tess = ["TESS"]

FILTERS = (gaia + sdss + ps + decam + tycho + bessell + tmass + ukidss + wise +
           hipp + kepler + tess)

__all__ = []
