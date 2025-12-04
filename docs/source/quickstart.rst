Quick Start Guide
=================

This guide walks you through the most common brutus workflows.

.. tip::

   **Which approach should I use?**

   - **Fitting stellar parameters** (distances, extinctions, masses) → Start with :ref:`bruteforce-fitting` below
   - **Generating synthetic photometry** → See :ref:`photometry-generation`
   - **Modeling star clusters** → See :doc:`cluster_modeling`

.. _data-setup:

Step 1: Download Data
---------------------

Before using brutus, download the required data files:

.. code-block:: python

   from brutus import fetch_grids, fetch_isos

   fetch_grids()   # Required: stellar model grids (~1-5 GB)
   fetch_isos()    # Required: isochrone tables (~100 MB)

Optional: download 3D dust maps for extinction priors:

.. code-block:: python

   from brutus import fetch_dustmaps
   fetch_dustmaps()  # Optional: ~1 GB

Files are cached and only downloaded once. See :doc:`installation` for details.

.. _bruteforce-fitting:

Step 2: Fit Stars with BruteForce
---------------------------------

This is the main workflow for deriving stellar parameters from photometry.

.. code-block:: python

   import numpy as np
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

**Expected runtime**: 1-10 seconds per star with good data.

Step 3: Working with Results
----------------------------

Results are saved to an HDF5 file. Access posterior samples:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][:]    # (Nstars, Ndraws) in kpc
       av_values = f['samps_red'][:]     # (Nstars, Ndraws) A_V extinction
       rv_values = f['samps_dred'][:]    # (Nstars, Ndraws) R_V
       model_idx = f['model_idx'][:]     # (Nstars, Ndraws) model indices
       log_evidence = f['obj_log_evid'][:] # (Nstars,) log-evidence

   # Compute summary statistics
   dist_median = np.median(distances, axis=1)
   dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)

   print(f"Distance: {dist_median[0]*1000:.0f} pc "
         f"(+{(dist_84[0]-dist_median[0])*1000:.0f} "
         f"/-{(dist_median[0]-dist_16[0])*1000:.0f})")

See :doc:`understanding_results` for interpreting output and assessing reliability.

.. _photometry-generation:

Alternative: Generate Synthetic Photometry
------------------------------------------

If you need to generate photometry (for simulations or testing) rather than fit it:

**For individual stars** (on-the-fly computation):

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   # Initialize stellar evolutionary tracks
   tracks = EEPTracks()

   # Create photometry generator for specific filters
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i'])

   # Generate SED for a 1 solar mass star
   seds, params1, params2 = star.get_seds(
       mini=1.0,     # Initial mass (solar masses)
       eep=400,      # Equivalent evolutionary point (see glossary)
       feh=0.0,      # Metallicity [Fe/H]
       afe=0.0,      # Alpha enhancement [α/Fe]
       av=0.5,       # Visual extinction (mag)
       dist=1000.0   # Distance (pc)
   )

**For stellar populations** (isochrone-based):

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Create isochrone and population generator
   iso = Isochrone()
   pop = StellarPop(isochrone=iso, filters=['g', 'r', 'i'])

   # Generate photometry for a 1 Gyr population
   seds, params1, params2 = pop.get_seds(
       feh=0.0,      # Metallicity [Fe/H]
       afe=0.0,      # Alpha enhancement [α/Fe]
       loga=9.0,     # log10(age/yr) = 1 Gyr
       av=0.5,       # Visual extinction
       dist=2000.0   # Distance (pc)
   )

Workflow Summary
----------------

1. **Download data**: ``fetch_grids()``, ``fetch_isos()``
2. **Load models**: ``load_models('grid_file.h5')``
3. **Create fitter**: ``BruteForce(StarGrid(models, labels, label_mask))``
4. **Fit data**: ``fitter.fit(data, data_err, ..., save_file='results.h5')``
5. **Analyze**: Read HDF5, compute statistics, visualize

Next Steps
----------

- :doc:`tutorials` - Detailed worked examples
- :doc:`understanding_results` - Interpret output and diagnostics
- :doc:`choosing_options` - Configure fitting parameters
- :doc:`api/index` - Complete API reference
