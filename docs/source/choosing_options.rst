Configuration Guide: Choosing Options
======================================

This page provides guidance on selecting configuration options for brutus fitting.

Model Selection
---------------

**Pre-computed Grids** (``StarGrid`` + ``BruteForce``): Best for large samples (>1000 stars), standard filter sets, and when speed matters.

**On-the-Fly Models** (``StarEvolTrack`` / ``StellarPop``): Best for custom filters, small samples, or cluster modeling with MCMC.

.. code-block:: python

   # Grid-based fitting (fast)
   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   models, labels, label_mask = load_models('grid_mist_v9.h5')
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

   # On-the-fly (flexible)
   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i', 'z'])

Filter Selection
^^^^^^^^^^^^^^^^

**Minimum**: 3-4 bands spanning optical to near-IR.

**Recommended combinations**:

- **Gaia + 2MASS**: G, BP, RP, J, H, Ks (6 bands)
- **Pan-STARRS**: g, r, i, z, y (5 bands, optical-only)
- **Full coverage**: Gaia + 2MASS + WISE (8 bands)

Optical bands constrain temperature; near-IR breaks distance-extinction degeneracy.

Grid Resolution
---------------

Trade-off between precision and speed:

- **High** (5-10M models): Smooth posteriors, large files (5-10 GB)
- **Medium** (1-3M models): Good balance, manageable sizes (1-3 GB) - **recommended default**
- **Low** (200k-500k): Fast, small files, discretization artifacts

To create custom grids with specific parameter arrays:

.. code-block:: python

   import numpy as np
   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   generator = GridGenerator(tracks, filters=['bp', 'g', 'rp', 'j', 'h', 'ks'])

   # Define parameter grids as arrays
   mini_grid = np.linspace(0.08, 10.0, 300)
   eep_grid = np.linspace(202, 808, 200)
   feh_grid = np.linspace(-2.0, 0.5, 30)

   generator.make_grid(
       output_file='custom_grid.h5',
       mini_grid=mini_grid,
       eep_grid=eep_grid,
       feh_grid=feh_grid,
   )

Prior Configuration
-------------------

Priors are controlled via ``fit()`` parameters:

.. code-block:: python

   # With default priors (Galactic structure + dust map)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       data_coords=coords,         # Required for Galactic prior
       dustfile='bayestar19.h5',   # Enables dust map prior
   )

   # With uniform priors (diagnostic mode)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results_uniform.h5',
       lngalprior=lambda *args: 0.0,  # Disable Galactic prior
       lndustprior=lambda *args: 0.0, # Disable dust prior
   )

**When to disable priors**: Diagnostic testing, extra-galactic objects, unusual populations.

.. warning::
   Disabling priors can lead to highly degenerate results.

Custom Prior Functions
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from brutus.priors import logp_galactic_structure

   def custom_galactic_prior(dist, gal_l, gal_b, dlabels=None):
       """Uniform within 100 pc, default otherwise."""
       if dist < 0.1:  # kpc
           return 0.0
       return logp_galactic_structure(dist, gal_l, gal_b, dlabels)

   fitter.fit(..., lngalprior=custom_galactic_prior)

Fitting Parameters
------------------

Distance and Extinction Bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   output_file = fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       parallax=parallax, parallax_err=parallax_err,
       data_coords=coords,
       avlim=(0.0, 5.0),     # A_V bounds (default: 0.0 to 20.0)
       rvlim=(2.0, 6.0),     # R_V bounds (default: 1.0 to 8.0)
   )

Distance bounds are implicit in the model grid parameter coverage.

Likelihood and Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   output_file = fitter.fit(
       ...,
       logl_dim_prior=True,     # Chi-square formulation (default)
       ltol=3e-2,               # Convergence tolerance
       ltol_subthresh=1e-2,     # Sub-threshold tolerance
   )

Posterior Sampling
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   output_file = fitter.fit(
       ...,
       Ndraws=250  # Number of posterior draws (default: 250)
   )

More draws (500-1000) improve posterior characterization but increase file size.

Performance
-----------

Parallelization
^^^^^^^^^^^^^^^

For large samples, parallelize across stars:

.. code-block:: python

   from multiprocessing import Pool

   models, labels, label_mask = load_models('grid.h5')
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

   def fit_one_star(star_data):
       phot, phot_err, plx, plx_err = star_data
       return fitter.fit(phot, phot_err, parallax=plx, parallax_err=plx_err, ...)

   with Pool(processes=32) as pool:
       results = pool.map(fit_one_star, star_data_list)

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   batch_size = 1000
   for i in range(0, len(catalog), batch_size):
       batch = catalog[i:i+batch_size]
       # Process and save batch results
       # Clear memory between batches

Caching
^^^^^^^

EEPTracks supports caching for faster repeated loads:

.. code-block:: python

   tracks = EEPTracks(use_cache=True)  # Creates .pkl cache

Cluster Modeling
----------------

For cluster fitting, see :doc:`cluster_modeling`. Key options:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop
   from brutus.analysis.populations import isochrone_population_loglike

   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # theta = [feh, loga, av, rv, dist]
   lnl = isochrone_population_loglike(
       theta, pop, flux, flux_err,
       dim_prior=True,
       cluster_prob=0.95,
   )

MCMC with emcee: Use 2-4× ndim walkers, check acceptance fraction (target 0.2-0.5).

Empirical Calibration
---------------------

Include empirical corrections when generating grids:

.. code-block:: python

   corr_params = [dtdm, drdm, msto_smooth, feh_scale]
   generator.make_grid('grid_corrected.h5', corr_params=corr_params)

Apply corrections for main-sequence stars; may not apply to giants or very metal-poor stars. See :doc:`photometric_offsets`.

Quick Reference
---------------

**Individual field stars**:

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   models, labels, label_mask = load_models('grid_mist_v9.h5')
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

   output_file = fitter.fit(
       data=flux, data_err=flux_err,
       data_mask=mask, data_labels=obj_ids,
       save_file='results.h5',
       parallax=parallax, parallax_err=parallax_err,
       data_coords=coords,
       Ndraws=250,
   )

**Large surveys**: Use coarser grids and multiprocessing (see Performance section).

Summary
-------

Key decisions:

1. **Model type**: Grid (fast) vs on-the-fly (flexible)
2. **Grid resolution**: High (precise) vs medium (balanced) vs low (fast)
3. **Priors**: Full Galactic model vs custom vs disabled
4. **Sampling**: Ndraws=250 (default) vs more (smoother posteriors)

Defaults are sensible for most applications.

Next Steps
----------

- Understand results: :doc:`understanding_results`
- Learn about priors: :doc:`priors`
- See FAQ: :doc:`faq`

References
----------

- Speagle et al. (2025), arXiv:2503.02227
