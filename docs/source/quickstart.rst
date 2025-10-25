Quick Start Guide
=================

This guide provides a quick introduction to using brutus for common workflows.

Individual Star Modeling
-------------------------

Generate photometry for individual stars using evolutionary tracks:

.. code-block:: python

   import numpy as np
   from brutus.core import EEPTracks, StarEvolTrack

   # Initialize stellar evolutionary tracks
   tracks = EEPTracks()

   # Create photometry generator for specific filters
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i'])

   # Generate SED for a 1 solar mass star
   seds, params1, params2 = star.get_seds(
       mini=1.0,     # Initial mass (solar masses)
       eep=400,      # Equivalent evolutionary point
       feh=0.0,      # Metallicity [Fe/H]
       afe=0.0,      # Alpha enhancement [α/Fe]
       av=0.5,       # Visual extinction (mag)
       dist=1000.0   # Distance (pc)
   )

For large-scale fitting with pre-computed grids, see :doc:`tutorials` and :doc:`api/analysis`.

Isochrone Generation
--------------------

Generate stellar parameters for stellar populations:

.. code-block:: python

   from brutus import Isochrone

   # Create isochrone generator
   iso = Isochrone()

   # Generate stellar parameters for an isochrone
   params = iso.get_predictions(
       feh=0.0,        # Solar metallicity [Fe/H]
       afe=0.0,        # Solar alpha enhancement [alpha/Fe]
       loga=9.0        # 1 Gyr age (log10(age/yr))
   )

Data Management
---------------

Download and manage stellar evolution data:

.. code-block:: python

   from brutus import fetch_grids, fetch_isos, fetch_dustmaps

   # Download stellar evolution grids
   fetch_grids()

   # Download isochrone data
   fetch_isos()

   # Download 3D dust maps
   fetch_dustmaps()

Working with Results
--------------------

Brutus provides comprehensive posterior distributions for all fitted parameters:

.. code-block:: python

   # Access posterior samples
   distances = results['dist_samples']
   extinctions = results['av_samples']
   stellar_params = results['stellar_params']

   # Plot results
   from brutus.plotting import cornerplot
   cornerplot(results, show_titles=True)

Common Workflows
----------------

For typical research workflows:

1. **Download data** using ``fetch_*`` functions
2. **Load models** using ``load_models``
3. **Create fitting objects** (``BruteForce``, ``Isochrone``)
4. **Fit your data** and analyze results
5. **Visualize results** using plotting utilities

Next Steps
----------

- See the :doc:`tutorials` for detailed examples
- Check the :doc:`api/index` for complete function documentation
- View the `tutorials/` directory for Jupyter notebook examples
