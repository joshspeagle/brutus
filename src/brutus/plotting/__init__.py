#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
brutus plotting module: Visualization utilities for stellar photometry analysis.

This module provides plotting functions for visualizing posterior distributions,
SEDs, photometric offsets, and other analysis results from brutus.
"""

# Import available functions
from .binning import bin_pdfs_distred
from .corner import cornerplot
from .distance import dist_vs_red
from .offsets import photometric_offsets, photometric_offsets_2d
from .sed import posterior_predictive
from .summary import summary_plot
from .utils import hist2d

__all__ = [
    # Data preparation utilities
    "bin_pdfs_distred",
    # Low-level utilities
    "hist2d",
    # Main plotting functions
    "dist_vs_red",
    "cornerplot",
    "posterior_predictive",
    "summary_plot",
    "photometric_offsets",
    "photometric_offsets_2d",
]
