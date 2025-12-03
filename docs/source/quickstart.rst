Quick Start Guide
=================

This guide provides a quick introduction to using brutus for common workflows.

.. note::
   Before using brutus, download the required data files using ``fetch_grids()``,
   ``fetch_isos()``, and/or ``fetch_dustmaps()``. See :ref:`data-management` below.

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

.. _data-management:

Data Management
---------------

Download and manage stellar evolution data (grids can be 1-5 GB):

.. code-block:: python

   from brutus import fetch_grids, fetch_isos, fetch_dustmaps

   # Download stellar evolution grids
   fetch_grids()

   # Download isochrone data
   fetch_isos()

   # Download 3D dust maps
   fetch_dustmaps()

Fitting Stars with BruteForce
-----------------------------

The main workflow for fitting stellar parameters:

.. code-block:: python

   import numpy as np
   import h5py
   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   # Load pre-computed model grid
   models, labels, label_mask = load_models('grid_mist_v9.h5')
   grid = StarGrid(models, labels, label_mask)

   # Create fitter
   fitter = BruteForce(grid)

   # Fit data (saves results to HDF5 file)
   output_file = fitter.fit(
       data=flux,              # (Nstars, Nfilters) flux densities
       data_err=flux_err,      # (Nstars, Nfilters) errors
       data_mask=mask,         # (Nstars, Nfilters) validity mask
       data_labels=obj_ids,    # (Nstars, Nlabels) object identifiers
       save_file='results.h5',
       parallax=plx,           # (Nstars,) parallax in mas
       parallax_err=plx_err,
       data_coords=coords,     # (Nstars, 2) galactic (l, b) in degrees
   )

Working with Results
--------------------

Results are saved to an HDF5 file. Access posterior samples directly:

.. code-block:: python

   import h5py
   import numpy as np

   # Read results from HDF5
   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][:]    # (Nstars, Ndraws) in kpc
       av_values = f['samps_red'][:]     # (Nstars, Ndraws) A_V
       rv_values = f['samps_dred'][:]    # (Nstars, Ndraws) R_V
       log_weights = f['samps_logp'][:]  # (Nstars, Ndraws) log-weights
       model_idx = f['model_idx'][:]     # (Nstars, Ndraws) model indices
       log_evidence = f['obj_log_evid'][:] # (Nstars,) log-evidence

   # Compute summary statistics
   dist_median = np.median(distances, axis=1)
   dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)

   print(f"Distance: {dist_median[0]*1000:.0f} pc "
         f"(+{(dist_84[0]-dist_median[0])*1000:.0f} "
         f"/-{(dist_median[0]-dist_16[0])*1000:.0f})")

Common Workflows
----------------

1. **Download data**: ``fetch_grids()``, ``fetch_isos()``, ``fetch_dustmaps()``
2. **Load models**: ``load_models('grid_file.h5')``
3. **Create fitter**: ``BruteForce(StarGrid(models, labels, label_mask))``
4. **Fit data**: ``fitter.fit(data, data_err, data_mask, labels, save_file, ...)``
5. **Analyze**: Read HDF5 output, compute statistics, visualize

Next Steps
----------

- See the :doc:`tutorials` for detailed examples
- Check the :doc:`api/index` for complete function documentation
- View the `tutorials/` directory for Jupyter notebook examples
