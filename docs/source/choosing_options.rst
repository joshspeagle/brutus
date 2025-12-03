Configuration Guide: Choosing Options
======================================

This page provides guidance on selecting appropriate configuration options for brutus fitting, including model choices, prior settings, optimization parameters, and performance tuning.

Model Selection
---------------

Grid vs On-the-Fly Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use Pre-computed Grids** (``StarGrid`` + ``BruteForce``):

✓ Large samples (> 1000 stars)
✓ Standard filter combinations (Gaia, 2MASS, WISE, Pan-STARRS, SDSS)
✓ Speed is critical
✓ Publication-quality uncertainties needed

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   models, labels, params = load_models('grid_gaiadr3_2mass.h5')
   grid = StarGrid(models, labels, params)
   fitter = BruteForce(grid)

**Use On-the-Fly Models** (``StarEvolTrack`` / ``StellarPop``):

✓ Custom filter combinations
✓ Exploratory analysis
✓ Small samples (< 100 stars)
✓ Cluster modeling with MCMC

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i', 'z'])

Filter Selection
^^^^^^^^^^^^^^^^

**Minimum recommended**: 3-4 photometric bands spanning optical to near-IR

**Optimal combinations**:

- **Gaia + 2MASS**: G, BP, RP, J, H, Ks (6 bands, excellent for most stars)
- **Pan-STARRS**: g, r, i, z, y (5 bands, optical-only)
- **Full coverage**: Gaia + 2MASS + WISE (G, BP, RP, J, H, Ks, W1, W2 = 8 bands)

**Why multi-wavelength matters**:

- **Optical**: Sensitive to temperature
- **Near-IR**: Breaks distance-extinction degeneracy
- **Mid-IR**: Constrains cool stars and circumstellar material

.. code-block:: python

   # Create custom grid with specific filters
   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   generator = GridGenerator(tracks, filters=['bp', 'g', 'rp', 'j', 'h', 'ks'])
   generator.make_grid('my_grid.h5')

Grid Parameters
---------------

Resolution Trade-offs
^^^^^^^^^^^^^^^^^^^^^^

**High resolution** (fine grid spacing):

- Mass: 500+ points
- EEP: 300+ points
- [Fe/H]: 40+ points
- Total: 5-10 million models

✓ Smooth posteriors
✓ Accurate parameter estimates
✗ Large files (5-10 GB)
✗ Slower fitting (more models to evaluate)

**Medium resolution** (default):

- Mass: 200-300 points
- EEP: 150-200 points
- [Fe/H]: 20-30 points
- Total: 1-3 million models

✓ Good balance
✓ Manageable file sizes (1-3 GB)
✓ Reasonable speed

**Low resolution** (coarse grid):

- Mass: 100-150 points
- EEP: 80-100 points
- [Fe/H]: 10-15 points
- Total: 200k-500k models

✓ Fast fitting
✓ Small files (< 500 MB)
✗ Discretization artifacts
✗ Less precise parameters

**Recommendation**: Start with medium resolution. Upgrade to high resolution for publication-quality results if artifacts are visible.

Parameter Coverage
^^^^^^^^^^^^^^^^^^

Ensure grid spans your targets:

.. code-block:: python

   generator.make_grid(
       output_file='custom_grid.h5',
       mini_range=(0.08, 150.0),    # Mass range (Msun)
       eep_range=(202, 808),         # Full evolutionary range
       feh_range=(-4.0, 0.5),        # Metallicity range (dex)
       afe_range=(-0.2, 0.6)         # Alpha enhancement range (dex)
   )

**Tips**:

- **Metal-poor stars**: Extend [Fe/H] to -4.0
- **Young stars**: Include pre-main-sequence (EEP < 353)
- **Giants**: Ensure coverage beyond EEP 454 (TAMS)
- **Low-mass**: Extend down to 0.08 Msun for M dwarfs

Prior Configuration
-------------------

Enabling/Disabling Priors
^^^^^^^^^^^^^^^^^^^^^^^^^^

Priors are controlled via parameters to ``fit()``, not constructor parameters:

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Fit with default priors (Galactic structure + dust map)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       data_coords=coords,         # Required for Galactic prior
       dustfile='bayestar19.h5',   # Enables dust map prior
   )

   # Fit with uniform priors (disable Galactic and dust priors)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results_uniform.h5',
       lngalprior=lambda *args: 0.0,  # Uniform Galactic prior
       lndustprior=lambda *args: 0.0, # Uniform dust prior
   )

**When to disable priors**:

- **Diagnostic purposes**: Test prior sensitivity
- **Non-Galactic objects**: Extra-galactic stars, satellite galaxies
- **Known unusual populations**: Very young clusters, special stellar types

.. warning::
   Disabling priors can lead to highly degenerate results. Only disable when you understand the implications.

Custom Prior Functions
^^^^^^^^^^^^^^^^^^^^^^

Provide custom prior functions via ``lngalprior`` and ``lndustprior``:

.. code-block:: python

   import numpy as np
   from brutus.priors import logp_galactic_structure

   def custom_galactic_prior(dist, gal_l, gal_b, dlabels=None):
       """Custom distance prior for specific region."""
       # Example: Uniform in distance for Local Bubble
       if dist < 0.1:  # Within 100 pc (dist is in kpc)
           return 0.0  # Log-prior (uniform)
       else:
           # Fall back to default Galactic prior
           return logp_galactic_structure(dist, gal_l, gal_b, dlabels)

   # Apply custom prior
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       data_coords=coords,
       lngalprior=custom_galactic_prior,
   )

Dust Map Selection
^^^^^^^^^^^^^^^^^^

Choose which 3-D dust map to use:

.. code-block:: python

   from brutus.dust.maps import use_dust_map

   # Use Bayestar19 (default)
   use_dust_map('bayestar19')

   # Alternatives (if available):
   # use_dust_map('bayestar17')
   # use_dust_map('3d_dust_map_custom')

**Considerations**:

- **Bayestar19**: Best for \|b\| > 5°, distances < 5 kpc
- **High latitudes**: Dust priors less important (low extinction)
- **Galactic plane**: Dust priors critical (high, variable extinction)

Optimization Settings
---------------------

Distance and Extinction Bounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set bounds for extinction and R_V:

.. code-block:: python

   fitter = BruteForce(grid)

   output_file = fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       parallax=parallax, parallax_err=parallax_err,
       data_coords=coords,
       avlim=(0.0, 5.0),     # A_V bounds (default: 0.0 to 20.0)
       rvlim=(2.0, 6.0),     # R_V bounds (default: 1.0 to 8.0)
   )

**Guidelines**:

- **avlim**: Set based on expected extinction range. Use dust map maximum + margin
- **rvlim**: Standard range is (2.0, 6.0). Default (1.0, 8.0) is very broad

Note: Distance bounds are implicit in the model grid - the grid covers specific
parameter ranges, and distances are derived from the flux scale factor.

Likelihood Formulation
^^^^^^^^^^^^^^^^^^^^^^

Choose between different likelihood models:

.. code-block:: python

   output_file = fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       logl_dim_prior=True    # Use chi-square formulation (default: True)
   )

**logl_dim_prior=True** (chi-square with implicit distance prior):
Appropriate for most individual star fitting. Includes geometric volume factor.

**logl_dim_prior=False** (pure Gaussian):
Use when distance prior is explicitly included elsewhere.

Convergence Tolerances
^^^^^^^^^^^^^^^^^^^^^^^

Control likelihood convergence with ``ltol``:

.. code-block:: python

   output_file = fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       ltol=3e-2,           # Convergence tolerance (default: 3e-2)
       ltol_subthresh=1e-2, # Sub-threshold tolerance (default: 1e-2)
   )

Sampling Parameters
-------------------

Number of Posterior Draws
^^^^^^^^^^^^^^^^^^^^^^^^^^

Control the number of posterior draws saved per object:

.. code-block:: python

   output_file = fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       Ndraws=250  # Number of posterior draws (default: 250)
   )

**Trade-offs**:

- **More draws** (500-1000): Better posterior characterization, larger output files
- **Fewer draws** (100-250): Faster I/O, smaller files, sufficient for summary statistics

The draws are importance-sampled from the posterior, weighted by probability.
No additional user configuration is needed for the sampling scheme.

Performance Tuning
------------------

Parallelization
^^^^^^^^^^^^^^^

**Multi-star parallelization** (recommended for large samples):

.. code-block:: python

   from multiprocessing import Pool
   from brutus.analysis import BruteForce

   # Initialize fitter
   models, labels, params = load_models('grid.h5')
   grid = StarGrid(models, labels, params)
   fitter = BruteForce(grid)

   def fit_one_star(star_data):
       """Fit function for one star."""
       phot, phot_err, parallax, parallax_err = star_data
       return fitter.fit(phot, phot_err, parallax=parallax,
                        parallax_err=parallax_err)

   # Parallel execution
   with Pool(processes=32) as pool:
       results_list = pool.map(fit_one_star, star_data_list)

**Within-star parallelization** (not yet implemented):

Future versions may support multi-threading for grid evaluation within a single star fit.

Memory Management
^^^^^^^^^^^^^^^^^

For very large grids:

.. code-block:: python

   # Use memory-mapped HDF5 files (doesn't load full grid into RAM)
   models, labels, params = load_models('huge_grid.h5', memmap=True)
   grid = StarGrid(models, labels, params)

**Batch processing**:

.. code-block:: python

   # Process stars in batches to limit memory usage
   batch_size = 1000
   for i in range(0, len(star_catalog), batch_size):
       batch = star_catalog[i:i+batch_size]
       results_batch = [fitter.fit(s['phot'], s['phot_err']) for s in batch]
       # Save results_batch to disk
       # Clear memory

Caching
^^^^^^^

EEPTracks and Isochrone objects support caching:

.. code-block:: python

   from brutus.core import EEPTracks, Isochrone

   # Enable pickle caching (speeds up repeated loads)
   tracks = EEPTracks(use_cache=True)  # Creates .pkl cache file
   iso = Isochrone(use_cache=True)

   # Subsequent loads are much faster
   tracks2 = EEPTracks(use_cache=True)  # Loads from cache

**When useful**:

- Repeatedly loading same models in scripts
- Interactive sessions with multiple runs

Cluster Modeling Options
-------------------------

Grid Configuration for Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cluster fitting uses different grid parameters:

.. code-block:: python

   from brutus.analysis.populations import generate_isochrone_population_grid

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       smf_grid=None,              # Binary mass fraction grid (default: adaptive)
       eep_grid=None,              # EEP grid (default: 2000 points)
       mini_bound=0.08,            # Minimum mass (Msun)
       eep_binary_max=480.0,       # Max EEP for binaries (MS only)
       corr_params=None            # Empirical corrections
   )

**Custom SMF grid** (if you have knowledge about binary fraction):

.. code-block:: python

   import numpy as np

   # Fine sampling near equal-mass binaries
   smf_grid = np.concatenate([
       np.array([0.0]),           # Single stars
       np.linspace(0.2, 0.9, 8),  # Unequal mass
       np.linspace(0.9, 1.0, 5)   # Near equal mass (fine sampling)
   ])

Outlier Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^

Choose outlier model for field contamination:

.. code-block:: python

   from brutus.analysis.populations import isochrone_population_loglike

   lnl = isochrone_population_loglike(
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0, field_fraction=0.1,
       stellarpop=pop,
       obs_flux=flux, obs_err=flux_err,
       dim_prior=True,             # Chi-square cluster likelihood
       outlier_model='chisquare'  # or 'uniform' or custom function
   )

**Chi-square outlier** (default):
   Assumes outliers follow cluster model with extra scatter. Good for photometric binaries or cluster members with variable extinction.

**Uniform outlier**:
   Assigns constant low likelihood. More aggressive at excluding outliers. Good for clean clusters with known field contamination.

**Custom outlier**:
   Provide your own function based on known contaminant properties (e.g., field star color distribution).

MCMC Configuration
^^^^^^^^^^^^^^^^^^

When using emcee for cluster fitting:

.. code-block:: python

   import emcee

   ndim = 6  # [Fe/H], log(age), A_V, R_V, dist, field_frac
   nwalkers = 32  # Recommended: 2-4 × ndim
   nsteps = 5000  # Burn-in + production

   # Initialize walkers in small ball around guess
   initial = np.array([0.0, 9.0, 0.3, 3.1, 2000.0, 0.1])
   pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)

   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(pos, nsteps, progress=True)

   # Diagnostics
   print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
   # Target: 0.2-0.5

**Common issues**:

- **Low acceptance (<0.1)**: Step size too large or bad initial conditions
- **High acceptance (>0.7)**: Step size too small (slow convergence)
- **Check convergence**: Use ``emcee.autocorr`` to estimate autocorrelation time

Empirical Calibration Options
------------------------------

Applying Corrections
^^^^^^^^^^^^^^^^^^^^

Include empirical corrections:

.. code-block:: python

   # Define correction parameters
   corr_params = [
       dtdm,        # Temperature correction (K/Msun)
       drdm,        # Radius correction (Rsun/Msun)
       msto_smooth, # Smoothing parameter (Msun)
       feh_scale    # Metallicity scaling factor
   ]

   # Apply in grid generation
   generator.make_grid('grid_corrected.h5', corr_params=corr_params)

   # Apply in cluster modeling
   grid = generate_isochrone_population_grid(
       stellarpop=pop, feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       corr_params=corr_params
   )

**When to apply**:

✓ Main-sequence stars with well-calibrated cluster corrections
✓ Publication-quality distance estimates
✗ Giants or post-MS stars (corrections may not apply)
✗ Very metal-poor stars (outside calibration range)

Photometric Offsets
^^^^^^^^^^^^^^^^^^^

Apply filter-specific photometric offsets:

.. code-block:: python

   # After fitting, apply offsets to model magnitudes
   model_mags_corrected = model_mags + offsets

   # Or include in likelihood (modify residuals)
   residuals_corrected = (obs_mags - model_mags) - offsets

See :doc:`photometric_offsets` for deriving survey-specific offsets.

Decision Tree: Configuration Quick Reference
---------------------------------------------

**For individual field stars**:

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   models, labels, mask = load_models('grid_mist_v9.h5')
   grid = StarGrid(models, labels, mask)
   fitter = BruteForce(grid)

   output_file = fitter.fit(
       data=flux, data_err=flux_err,
       data_mask=mask, data_labels=obj_ids,
       save_file='results.h5',
       parallax=parallax, parallax_err=parallax_err,
       data_coords=coords,          # Galactic (l, b) for prior
       dustfile='bayestar19.h5',    # 3D dust map
       Ndraws=250,
   )

**For stellar clusters**:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop
   from brutus.analysis.populations import isochrone_population_loglike
   import emcee

   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   def lnprob(theta):
       # theta = [feh, loga, av, rv, dist]
       return isochrone_population_loglike(
           theta, pop, flux, flux_err,
           parallax=plx, parallax_err=plx_err,
           cluster_prob=0.9, dim_prior=True,
       )

   ndim, nwalkers = 5, 32
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(initial_pos, 5000, progress=True)

**For large surveys** (millions of stars):

.. code-block:: python

   # Use coarse grid for speed
   generator.make_grid(
       'fast_grid.h5',
       mini_range=(0.1, 10.0),  # Limit to dwarfs/subgiants
       eep_range=(300, 500),    # Main sequence only
       feh_range=(-1.0, 0.5)    # Solar neighborhood
   )

   # Parallelize across stars
   with Pool(processes=64) as pool:
       results = pool.map(fit_one_star, star_list)

Summary
-------

Key configuration decisions:

1. **Model type**: Grid (fast, fixed filters) vs on-the-fly (flexible, slower)
2. **Grid resolution**: High (precise, slow) vs medium (balanced) vs low (fast, artifacts)
3. **Priors**: Full Galactic model (default) vs custom vs disabled
4. **Likelihood**: Chi-square (dim_prior=True, default) vs Gaussian
5. **Sampling**: 10k samples (default) vs more (smooth) vs fewer (fast)
6. **Calibration**: Empirical corrections (recommended) vs raw models

For most applications, the **defaults are sensible**. Customize when you understand the trade-offs.

Next Steps
----------

- Understand your results: :doc:`understanding_results`
- Review common questions: :doc:`faq`
- See complete API reference: :doc:`api/index`

References
----------

- Speagle et al. (2025), arXiv:2503.02227
