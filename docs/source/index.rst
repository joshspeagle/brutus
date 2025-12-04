brutus Documentation
====================

*Et tu, Brute?*

**brutus** is a Pure Python package for **"brute force" Bayesian inference** to derive distances, reddenings, and stellar properties from photometry. The package is designed to be highly modular and user-friendly, with comprehensive support for modeling individual stars, star clusters, and 3-D dust mapping.

.. image:: https://github.com/joshspeagle/brutus/blob/master/brutus_logo.png?raw=true
   :alt: brutus logo
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials
   glossary

.. toctree::
   :maxdepth: 2
   :caption: Scientific Background

   scientific_background
   stellar_models
   grid_generation
   priors
   cluster_modeling
   photometric_offsets

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   understanding_results
   choosing_options
   faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Key Features
============

🌟 **Individual Star Modeling**: Fit distances, reddenings, and stellar properties for individual stars using Bayesian inference

🌟 **Cluster Analysis**: Model stellar clusters with consistent ages, metallicities, and distances

🌟 **3D Dust Mapping**: Integrate with 3D dust maps and model extinction along lines of sight

🌟 **Modern Stellar Models**: Built-in support for MIST isochrones and evolutionary tracks

🌟 **Flexible & Fast**: Optimized algorithms with numba acceleration and modular design

🌟 **Publication Ready**: Designed for ease of use in research workflows

Quick Start
===========

Install brutus from PyPI and download required data:

.. code-block:: bash

   pip install astro-brutus

.. code-block:: python

   from brutus import fetch_grids, fetch_isos
   fetch_grids()  # Download stellar model grids (~1-5 GB)
   fetch_isos()   # Download isochrone data (~100 MB)

For individual star modeling:

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   # Initialize stellar evolutionary tracks
   tracks = EEPTracks()

   # Create photometry generator
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i'])

   # Generate SED for a star
   seds, params1, params2 = star.get_seds(
       mini=1.0, eep=400, feh=0.0, av=0.5, dist=1000.0
   )

For stellar population modeling:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Initialize isochrone
   iso = Isochrone()

   # Create population generator
   pop = StellarPop(isochrone=iso, filters=['g', 'r', 'i'])

   # Generate population photometry
   seds, params1, params2 = pop.get_seds(
       feh=0.0, afe=0.0, loga=9.0, av=0.5, dist=2000.0
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
