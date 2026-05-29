.. raw:: html

   <div class="hero-section">

brutus
======

*Et tu, Brute?*

**"Brute-force" Bayesian inference for stellar photometry**

.. image:: _static/brutus_logo.png
   :alt: brutus logo
   :align: center
   :width: 300px

.. raw:: html

   <p class="tagline">Derive distances, reddenings, and stellar properties from photometry</p>
   <p><span class="version-badge">v1.0.0</span></p>
   </div>

----

.. grid:: 1 2 2 4
   :gutter: 4
   :class-container: sd-text-center

   .. grid-item-card::
      :link: installation
      :link-type: doc
      :class-card: sd-shadow-sm

      :octicon:`download;3em;sd-text-primary`

      **Installation**
      ^^^^^^^^^^^^^^^^

      Get started quickly with pip install.
      Download required model grids and data.

   .. grid-item-card::
      :link: quickstart
      :link-type: doc
      :class-card: sd-shadow-sm

      :octicon:`rocket;3em;sd-text-primary`

      **Quick Start Guide**
      ^^^^^^^^^^^^^^^^^^^^^

      Learn the basic workflow in 5 minutes.
      Fit your first star with BruteForce.

   .. grid-item-card::
      :link: glossary
      :link-type: doc
      :class-card: sd-shadow-sm

      :octicon:`book;3em;sd-text-primary`

      **Glossary**
      ^^^^^^^^^^^^

      Key terms and concepts including
      EEP, isochrones, and extinction.

   .. grid-item-card::
      :link: changelog
      :link-type: doc
      :class-card: sd-shadow-sm

      :octicon:`megaphone;3em;sd-text-warning`

      **What's New**
      ^^^^^^^^^^^^^^

      See the latest changes in v\ |version|.
      Release notes and migration guides.

----

Key Features
------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-border-0

      :octicon:`star;2em;sd-text-warning` **Individual Star Fitting**

      Fit distances, reddenings, and stellar properties for individual
      stars using Bayesian inference with pre-computed model grids.

      .. dropdown:: How do I fit stellar parameters?
         :color: primary

         - **Large sample (>100 stars) with standard filters**: Use ``BruteForce`` + ``StarGrid``

           *Fastest approach. Pre-computed grids enable fitting thousands of stars quickly.*
           See :doc:`quickstart` - "Fitting with BruteForce".

         - **Small sample or non-standard setup**: Use ``StarEvolTrack``

           *More flexible but slower. Interpolates along stellar evolution tracks.*
           See :doc:`quickstart` - "Individual Star Modeling".

   .. grid-item-card::
      :class-card: sd-border-0

      :octicon:`sun;2em;sd-text-info` **Cluster Analysis**

      Model stellar clusters with consistent ages, metallicities,
      and distances using isochrone-based population models.

      .. dropdown:: How do I model a stellar cluster?
         :color: info

         Use ``Isochrone`` + ``StellarPop`` with your preferred sampler or optimizer.

         *Fits shared age, metallicity, distance, and extinction for coeval populations.
         Handles cluster membership probabilities, binary stars, and outlier rejection.*
         See :doc:`quickstart`.

   .. grid-item-card::
      :class-card: sd-border-0

      :octicon:`cloud;2em;sd-text-secondary` **3D Dust Mapping**

      Integrate with 3D dust maps (Bayestar) and model extinction
      along lines of sight for improved distance estimates.

      .. dropdown:: How do I use dust maps?
         :color: secondary

         - **Extinction priors**: Use ``Bayestar`` to query 3D dust maps and constrain
           extinction in your fitting. See :doc:`priors`.

         - **Dust estimation**: Derive extinction estimates from fitted stellar samples
           along lines of sight.

         - **Line-of-sight modeling**: Fit and visualize dust distributions using
           the ``los_clouds_*`` functions. See :doc:`quickstart`.

   .. grid-item-card::
      :class-card: sd-border-0

      :octicon:`pulse;2em;sd-text-success` **Synthetic Photometry**

      Generate synthetic SEDs for individual stars or full stellar
      populations using MIST isochrones and neural network models.

      .. dropdown:: How do I generate synthetic photometry?
         :color: success

         - For individual stars: Use ``StarEvolTrack.get_seds()``
         - For populations: Use ``StellarPop.get_seds()``

         *Useful for simulations, testing, or understanding model predictions.*
         See :doc:`stellar_models`.

----

Code Examples
-------------

.. tab-set::

   .. tab-item:: Install

      Install the latest stable release from `PyPI <https://pypi.org/project/astro-brutus/>`_:

      .. code-block:: bash

         pip install astro-brutus

      Or install the development version (v\ |version|) directly from GitHub:

      .. code-block:: bash

         pip install git+https://github.com/joshspeagle/brutus.git

      .. note::

         **Windows users**: brutus requires WSL (Windows Subsystem for Linux)
         due to the healpy dependency. See :doc:`installation` for details.

      Download required data files:

      .. code-block:: python

         from brutus.data import fetch_grids, fetch_isos, fetch_offsets

         # Download stellar model grid (~750 MB)
         grid_path = fetch_grids(grid='mist_v9')

         # Download isochrone tables (~250 MB)
         iso_path = fetch_isos(iso='MIST_1.2_vvcrit0.0')

         # Download photometric offsets (recommended for fitting)
         offset_path = fetch_offsets(grid='mist_v9')

   .. tab-item:: Fit Stars

      Fit stellar parameters with BruteForce:

      .. code-block:: python

         from brutus.data import load_models, load_offsets
         from brutus.core import StarGrid
         from brutus.analysis import BruteForce
         from brutus.utils import inv_magnitude

         # Load model grid with specific filters
         filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
                    'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',
                    '2MASS_J', '2MASS_H', '2MASS_Ks']
         models, labels, label_mask = load_models(
             'grid_mist_v9.h5', filters=filters
         )
         grid = StarGrid(models, labels)

         # Load photometric offsets
         offsets = load_offsets('offsets_mist_v9.txt', filters=filters)

         # Convert magnitudes to linear flux densities
         flux, flux_err = inv_magnitude(mag, mag_err)

         # Fit photometry
         fitter = BruteForce(grid)
         results = fitter.fit(
             data=flux,              # (N, Nfilt) linear flux densities
             data_err=flux_err,      # (N, Nfilt) flux errors
             data_mask=flux_mask,    # (N, Nfilt) boolean mask
             data_labels=obj_ids,    # (N,) object identifiers
             phot_offsets=offsets,   # Photometric calibration
             parallax=plx,           # (N,) parallax in mas
             parallax_err=plx_err,   # (N,) parallax error
             data_coords=coords,     # (N, 2) Galactic (l, b) in degrees
             save_file='results.h5'
         )

   .. tab-item:: Synthetic SEDs

      Generate synthetic photometry for stellar populations:

      .. code-block:: python

         from brutus.core import Isochrone, StellarPop

         # Create population generator with specific filters
         filters = ['Gaia_G_MAW', 'PS_g', 'PS_r', 'PS_i', '2MASS_J']
         iso = Isochrone()
         pop = StellarPop(isochrone=iso, filters=filters)

         # Generate photometry for a 1 Gyr solar-metallicity population
         seds, params, params2 = pop.get_seds(
             feh=0.0,        # Solar metallicity
             loga=9.0,       # log10(age/yr) = 1 Gyr
             av=0.5,         # V-band extinction
             rv=3.3,         # Extinction law R(V)
             dist=2000.0     # Distance in parsecs
         )

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   glossary

.. toctree::
   :maxdepth: 2
   :caption: Scientific Background
   :hidden:

   scientific_background
   stellar_models
   priors
   grid_generation
   population_modeling
   photometric_offsets

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   understanding_results

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   changelog
   contributing
