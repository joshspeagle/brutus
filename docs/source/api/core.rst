Core Module (``brutus.core``)
==============================

The core module provides the fundamental stellar modeling infrastructure for brutus. It contains classes and functions for generating synthetic photometry from stellar evolution models, both for individual stars and stellar populations.

**Key Components:**

- **Individual Stars**: ``EEPTracks`` predicts stellar parameters along evolutionary tracks, ``StarGrid`` provides fast grid-based photometry
- **Populations**: ``Isochrone`` models coeval stellar populations, ``StellarPop`` generates population photometry
- **Neural Networks**: ``FastNN`` and ``FastNNPredictor`` compute bolometric corrections efficiently
- **Grid Generation**: ``GridGenerator`` creates pre-computed model grids for large-scale fitting

**Design Philosophy:**

The module follows a clean separation of concerns:

- **Parameter Prediction** (``EEPTracks``, ``Isochrone``): Maps intrinsic stellar parameters (mass, age, metallicity) to observable parameters (temperature, luminosity, radius, surface gravity)
- **Photometry Generation** (``StarEvolTrack``, ``StellarPop``): Converts stellar parameters to synthetic photometry using neural network bolometric corrections and extinction modeling

This separation allows flexible combinations: you can use different photometry generators with the same parameter predictors, or vice versa.

**Typical Usage Patterns:**

For **individual field stars** with unknown evolutionary state:

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   # Parameter prediction
   tracks = EEPTracks()

   # Photometry generation (on-the-fly)
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i', 'z'])
   sed, params1, params2 = star.get_seds(
       mini=1.0, eep=400, feh=0.0, afe=0.0,
       av=0.1, dist=1000.0
   )

For **large samples** requiring speed:

.. code-block:: python

   from brutus.core import StarGrid
   from brutus.data import load_models

   # Load pre-computed grid
   models, labels, label_mask = load_models('grid_file.h5')
   grid = StarGrid(models, labels, label_mask)

   # Fast photometry lookup (no re-computation)
   # Use with BruteForce fitter for thousands to millions of stars

For **stellar clusters** with shared age and metallicity:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Population parameter prediction
   iso = Isochrone()

   # Population photometry
   pop = StellarPop(isochrone=iso)
   seds, params1, params2 = pop.get_seds(
       feh=0.0, loga=9.0, av=0.5, dist=2000.0
   )

**See Also:**

- :doc:`/stellar_models` - Conceptual overview of MIST models, EEP, and isochrones
- :doc:`/grid_generation` - Guide to creating and using pre-computed grids
- :doc:`/cluster_modeling` - Using populations for cluster fitting

.. currentmodule:: brutus.core

Individual Star Models
----------------------

.. autoclass:: EEPTracks
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StarGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StarEvolTrack
   :members:
   :undoc-members:
   :show-inheritance:

Population Models
-----------------

.. autoclass:: Isochrone
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StellarPop
   :members:
   :undoc-members:
   :show-inheritance:

Neural Networks
---------------

.. autoclass:: FastNN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FastNNPredictor
   :members:
   :undoc-members:
   :show-inheritance:

Grid Generation
---------------

.. autoclass:: GridGenerator
   :members:
   :undoc-members:
   :show-inheritance:

SED Utilities
-------------

.. autofunction:: get_seds

.. autofunction:: _get_seds

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.core.individual
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.populations
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.neural_nets
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.sed_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.core.grid_generation
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
