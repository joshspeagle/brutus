Grid-Based Modeling
===================

This page explains the **grid-based "brute force" approach** that gives brutus its name.

The Brute-Force Philosophy
---------------------------

Unlike traditional MCMC methods, brutus evaluates likelihood at all points on a pre-computed model grid, then marginalizes to get the posterior. This approach:

- Has no convergence issues (no burn-in, local minima)
- Guarantees systematic parameter space coverage
- Is embarrassingly parallelizable
- Reuses the same grid for millions of stars

The trade-off is computational cost, addressed through pre-computation, efficient multi-stage optimization, and adaptive grid resolution.

.. note::
   **No interpolation**: brutus evaluates likelihoods at discrete grid points only. It does not interpolate between grid models. This means:

   - Posterior distributions have inherent discreteness matching grid resolution
   - Finer grids yield smoother posteriors but larger files and slower fitting
   - For most applications, 1-3M model grids provide adequate resolution

Grid Structure
--------------

A brutus **model grid** (``StarGrid``) maps stellar parameters to photometry:

.. math::

   (M_{\rm init}, {\rm EEP}, [{\rm Fe/H}], [\alpha/{\rm Fe}]) \rightarrow \{\mathbf{M}_{\rm ref}, \mathbf{R}, \mathbf{R}'\}

where :math:`\mathbf{M}_{\rm ref}` are absolute magnitudes at a **reference distance of 1 kpc**, and :math:`\mathbf{R}`, :math:`\mathbf{R}'` are reddening coefficients.

Reddened magnitudes are computed as:

.. math::

   m_{\rm band}(A_V, R_V) = M_{\rm ref,band} + \mu + A_V \times (R_{\rm band} + R_V \times R'_{\rm band})

This allows modeling arbitrary extinction without storing separate grids for each :math:`A_V` value.

Typical Grid Coverage
^^^^^^^^^^^^^^^^^^^^^

- **Initial mass**: 0.08 to 150 :math:`M_\odot` (~200-500 points)
- **EEP**: 202 to 808 (~100-300 points)
- **Metallicity** [Fe/H]: -4.0 to +0.5 dex (~20-40 points)
- **Alpha enhancement** [α/Fe]: -0.2 to +0.6 dex (~5-10 points)

A comprehensive grid contains :math:`\sim 10^6` to :math:`10^7` models (several GB).

Available Pre-Computed Grids
-----------------------------

Download grids with ``fetch_grids()``:

- **grid_mist_v9.h5**: Default MIST v1.2 grid (~1-2 GB) - recommended for most applications
- **grid_gaiadr3_2mass.h5**: Gaia DR3 + 2MASS (G, BP, RP, J, H, Ks)
- **grid_gaiadr3_2mass_wise.h5**: Gaia DR3 + 2MASS + WISE (8 bands)

.. code-block:: python

   from brutus.data import fetch_grids, load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   fetch_grids()  # Download (first time only)

   models, labels, label_mask = load_models('grid_gaiadr3_2mass.h5')
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

Creating Custom Grids
---------------------

Use ``GridGenerator`` for custom filter combinations:

.. code-block:: python

   import numpy as np
   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   generator = GridGenerator(tracks, filters=['g', 'r', 'i', 'z', 'y'])

   # Define parameter grids as arrays
   mini_grid = np.linspace(0.08, 150.0, 300)
   eep_grid = np.linspace(202, 808, 200)
   feh_grid = np.linspace(-2.0, 0.5, 30)

   generator.make_grid(
       output_file='my_grid.h5',
       mini_grid=mini_grid,
       eep_grid=eep_grid,
       feh_grid=feh_grid,
   )

For non-uniform spacing (e.g., finer resolution for low-mass stars):

.. code-block:: python

   mini_grid = np.concatenate([
       np.linspace(0.08, 1.0, 200),   # Fine spacing
       np.linspace(1.0, 10.0, 100),   # Medium spacing
       np.linspace(10.0, 150.0, 50)   # Coarse spacing
   ])
   generator.make_grid(output_file='custom_grid.h5', mini_grid=mini_grid)

The Fitting Algorithm
----------------------

``BruteForce`` implements a multi-stage optimization strategy. For details, see Speagle et al. (2025) §3.

**Stage 1: Magnitude Space Approximation**
   Quickly eliminates poor fits using analytical least-squares in magnitude space. Keeps only grid points with reasonable :math:`\chi^2`.

**Stage 2: Flux Space Optimization**
   Refines (distance, :math:`A_V`, :math:`R_V`) in statistically correct flux space where errors are Gaussian.

**Stage 3: Bayesian Posterior**
   Adds prior contributions (IMF, Galactic structure, metallicity, age, dust maps) and parallax likelihood.

**Stage 4: Marginalization and Sampling**
   Importance-samples the posterior to produce Monte Carlo draws for distance, extinction, and stellar parameters.

Complete Example
----------------

.. code-block:: python

   import h5py
   import numpy as np
   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   # Load grid
   models, labels, label_mask = load_models('grid_gaiadr3_2mass_wise.h5')
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

   # Observed data (Gaia G, BP, RP + 2MASS J, H, Ks + WISE W1, W2)
   flux = np.array([...])  # Flux densities (Nstars, Nfilters)
   flux_err = np.array([...])
   mask = np.ones_like(flux, dtype=bool)
   obj_labels = np.array([[1], [2], ...])  # Object identifiers

   # Fit
   output_file = fitter.fit(
       data=flux, data_err=flux_err,
       data_mask=mask, data_labels=obj_labels,
       save_file='results.h5',
       parallax=parallax_array,
       parallax_err=parallax_err_array,
       data_coords=coords_array,  # Galactic (l, b)
       Ndraws=250,
   )

   # Read results from HDF5
   with h5py.File(output_file, 'r') as f:
       distances = f['samps_dist'][:]  # (Nstars, Ndraws) in kpc
       av_values = f['samps_red'][:]   # A_V
       model_idx = f['model_idx'][:]   # Grid indices

   dist_median = np.median(distances, axis=1) * 1000  # Convert to pc
   print(f"Distance: {dist_median[0]:.1f} pc")

Performance Tips
----------------

**Grid Resolution Trade-offs**:
Fine grids (~5-10M points) give smooth posteriors but are slower and larger. Coarse grids (~1M points) are faster but may show discretization artifacts.

**Parallelization**: Grid evaluation is embarrassingly parallel. Parallelize across stars for large samples:

.. code-block:: python

   from multiprocessing import Pool

   with Pool(processes=32) as pool:
       results = pool.map(fit_star, star_data_list)

**Memory**: For large grids, process stars in batches to limit memory usage.

Troubleshooting
---------------

- **Grid doesn't cover stars**: Check parameter ranges match your targets
- **Fitting slow (>10s/star)**: Use coarser grid, limit distance range, or parallelize
- **Discrete posterior jumps**: Grid resolution too coarse
- **Out of memory**: Reduce grid size or process in batches

Summary
-------

- brutus uses **pre-computed model grids** for fast, systematic parameter exploration
- Grids store **photometry at 1 kpc** plus **reddening coefficients**
- **Multi-stage optimization** balances speed and accuracy
- Use ``GridGenerator`` for **custom grids** with specific filters

Next Steps
----------

- Understand priors: :doc:`priors`
- Cluster fitting: :doc:`cluster_modeling`
- Configuration options: :doc:`choosing_options`
- Interpret results: :doc:`understanding_results`

References
----------

- Speagle et al. (2025), arXiv:2503.02227
