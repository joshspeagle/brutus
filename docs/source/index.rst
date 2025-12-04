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

Choose Your Workflow
====================

Not sure where to start? Use this guide to find the right approach for your use case.

**I want to fit stellar parameters to photometry:**

- **Large sample (>100 stars) with standard filters** → Use ``BruteForce`` + ``StarGrid``

  *Fastest approach. Pre-computed grids enable fitting thousands of stars quickly.* See :doc:`quickstart` § "Fitting with BruteForce".

- **Small sample or custom filters** → Use ``StarEvolTrack``

  *More flexible but slower. Computes photometry on-the-fly for any filter set.* See :doc:`quickstart` § "Individual Star Modeling".

**I want to model a stellar cluster:**

- Use ``Isochrone`` + ``StellarPop`` with MCMC

  *Fits shared age, metallicity, distance, and extinction for coeval populations.* See :doc:`cluster_modeling`.

**I want to generate synthetic photometry:**

- For individual stars → Use ``StarEvolTrack.get_seds()``
- For populations → Use ``StellarPop.get_seds()``

  *Useful for simulations, testing, or understanding model predictions.* See :doc:`stellar_models`.

**Still unsure?** Start with the :doc:`quickstart` guide or check the :doc:`faq`.

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
