Grid-Based Modeling
===================

This page explains the **grid-based "brute force" approach** that gives brutus its name. Understanding the grid structure and optimization strategy is essential for choosing appropriate settings and interpreting results.

The Brute-Force Philosophy
---------------------------

brutus takes a fundamentally different approach than traditional stellar parameter estimation methods:

**Traditional MCMC/Optimization**:
   Start with initial guess → Iteratively propose new parameters → Accept/reject based on likelihood → Converge to posterior

**brutus Grid Approach**:
   Pre-compute models on dense grid → Evaluate likelihood at all grid points → Marginalize to get posterior

This "brute-force" strategy has several advantages:

✓ **No convergence issues**: No need to worry about MCMC mixing, burn-in, or local minima

✓ **Guaranteed coverage**: Explores the full parameter space systematically

✓ **Parallelizable**: Grid evaluations are independent and can run in parallel

✓ **Reusable**: Same grid can fit millions of stars with different data

The trade-off is computational cost—evaluating every grid point is expensive. brutus addresses this through:

1. **Pre-computation**: Models are generated once and reused
2. **Efficient optimization**: Multi-stage fitting reduces wasted evaluations
3. **Smart grids**: Adaptive resolution focuses grid points where needed

Grid Structure and Parameters
------------------------------

A brutus **model grid** (``StarGrid``) is a pre-computed table of stellar photometry spanning the intrinsic stellar parameter space:

.. math::

   (M_{\rm init}, {\rm EEP}, [{\rm Fe/H}]_{\rm init}, [\alpha/{\rm Fe}]_{\rm init}) \rightarrow \{\mathbf{M}_{\rm ref}, \mathbf{R}, \mathbf{R}'\}

where:

- :math:`\mathbf{M}_{\rm ref}` are absolute magnitudes at a **reference distance** of 1 kpc
- :math:`\mathbf{R}` and :math:`\mathbf{R}'` are **reddening coefficient vectors** for each photometric band

Grid Dimensions
^^^^^^^^^^^^^^^

Typical grid coverage:

- **Initial mass**: 0.08 to 150 :math:`M_\odot` (logarithmically spaced, ~200-500 points)
- **EEP**: 202 to 808 (linearly spaced, ~100-300 points)
- **Metallicity** [Fe/H]: -4.0 to +0.5 dex (~20-40 points)
- **Alpha enhancement** [α/Fe]: -0.2 to +0.6 dex (~5-10 points)

A comprehensive grid might contain :math:`\sim 10^6` to :math:`10^7` model points, resulting in several GB file sizes.

Reference Distance: Why 1 kpc?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All pre-computed grids store photometry at a reference distance of **1 kpc (1000 pc)**. This choice is not arbitrary:

1. **Gaia parallax**: 1 mas parallax = 1 kpc distance (convenient conversion)
2. **Distance scaling**: Photometry at any distance is: :math:`m(d) = M_{\rm ref} + 5 \log_{10}(d / 1000\,{\rm pc})`
3. **Numerical stability**: Avoids very large or very small magnitude values

When fitting, brutus scales models from the 1 kpc reference to the actual distance being tested.

Reddening Coefficients
^^^^^^^^^^^^^^^^^^^^^^^

Rather than storing photometry for every possible extinction value, brutus pre-computes **reddening derivatives**:

.. math::

   R_{\rm band} &= \frac{\partial m_{\rm band}}{\partial A_V}\bigg|_{R_V=3.1} \\
   R'_{\rm band} &= \frac{\partial^2 m_{\rm band}}{\partial A_V \, \partial R_V}\bigg|_{R_V=3.1}

The reddened magnitude for any (:math:`A_V`, :math:`R_V`) is then:

.. math::

   m_{\rm band}(A_V, R_V) = M_{\rm ref,band} + \mu + A_V \times (R_{\rm band} + R_V \times R'_{\rm band})

This parameterization allows brutus to model arbitrary extinction without storing separate grids for each :math:`A_V` value, reducing storage by factors of 100-1000.

Available Pre-Computed Grids
-----------------------------

brutus provides several pre-computed grids for common filter combinations. These can be downloaded using ``fetch_grids()`` and are stored in ``~/.brutus/data/`` by default.

Standard Grids
^^^^^^^^^^^^^^

The following grids are available via ``fetch_grids()``:

- **grid_mist_v9.h5**: Default MIST v1.2 grid with standard filter set
   - Filters: Comprehensive optical and near-IR coverage
   - Size: ~1-2 GB
   - Recommended for most applications

- **grid_gaiadr3_2mass.h5**: Gaia DR3 + 2MASS
   - Filters: G, BP, RP, J, H, Ks (6 bands)
   - Size: ~500 MB - 1 GB
   - Optimal for Gaia + 2MASS photometry

- **grid_gaiadr3_2mass_wise.h5**: Gaia DR3 + 2MASS + WISE
   - Filters: G, BP, RP, J, H, Ks, W1, W2 (8 bands)
   - Size: ~1-1.5 GB
   - Full optical to mid-IR coverage

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from brutus.data import fetch_grids, load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   # Download grids (first time only)
   fetch_grids()

   # Load a specific grid
   models, labels, label_mask = load_models('grid_gaiadr3_2mass.h5')

   # Create StarGrid for fitting
   grid = StarGrid(models, labels, label_mask)
   fitter = BruteForce(grid)

   # Ready to fit stars with Gaia + 2MASS photometry

Grid Naming Convention
^^^^^^^^^^^^^^^^^^^^^^

Pre-computed grid files follow the naming pattern:

   ``grid_<source>_<version>.h5``

Where:
   - ``<source>`` indicates the photometric system(s) (e.g., ``gaiadr3_2mass``)
   - ``<version>`` indicates the stellar model version (e.g., ``v9`` for MIST v1.2)

Custom Grids
^^^^^^^^^^^^

If the pre-computed grids don't match your filter combination, you can create custom grids (see next section).

Creating Model Grids
---------------------

brutus provides the ``GridGenerator`` class for creating custom grids with specific filter combinations and parameter coverage.

Basic Grid Generation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from brutus.core import GridGenerator, EEPTracks

   # Initialize with stellar tracks
   tracks = EEPTracks()

   # Create grid generator with desired filters
   generator = GridGenerator(
       tracks,
       filters=['g', 'r', 'i', 'z', 'y']  # Pan-STARRS filters
   )

   # Generate and save grid
   generator.make_grid(
       output_file='my_panstarrs_grid.h5',
       mini_range=(0.08, 150.0),  # Mass range in solar masses
       eep_range=(202, 808),       # Full evolutionary range
       verbose=True
   )

This creates an HDF5 file containing the pre-computed photometry and reddening coefficients.

Custom Grid Spacing
^^^^^^^^^^^^^^^^^^^

For specific applications, you may want custom grid resolution:

.. code-block:: python

   import numpy as np

   # Finer spacing for low-mass stars
   mini_grid = np.concatenate([
       np.linspace(0.08, 1.0, 200),   # 0.08-1.0 Msun: fine spacing
       np.linspace(1.0, 10.0, 100),   # 1-10 Msun: medium spacing
       np.linspace(10.0, 150.0, 50)   # 10-150 Msun: coarse spacing
   ])

   generator.make_grid(
       output_file='custom_spacing_grid.h5',
       mini_grid=mini_grid,
       verbose=True
   )

Grid File Format
^^^^^^^^^^^^^^^^

Grid files are stored in HDF5 format with the following structure:

.. code-block:: text

   grid_file.h5
   ├── attributes
   │   ├── reference_distance: 1000.0 (pc)
   │   ├── filters: ['g', 'r', 'i', 'z', 'y']
   │   ├── n_models: 5000000
   │   └── mist_version: 'v1.2'
   ├── labels (N_models, 4): [mini, eep, feh, afe]
   ├── models (N_models, N_filters): absolute mags at 1 kpc
   ├── reddening (N_models, N_filters): R coefficients
   └── reddening_deriv (N_models, N_filters): R' coefficients

Loading and Using Grids
^^^^^^^^^^^^^^^^^^^^^^^^

Once created, grids are loaded with the data utilities:

.. code-block:: python

   from brutus.data import load_models
   from brutus.core import StarGrid

   # Load the grid
   models, labels, params = load_models('my_panstarrs_grid.h5')

   # Create StarGrid object
   grid = StarGrid(models, labels, params)

   # Now ready for fitting with BruteForce
   from brutus.analysis import BruteForce
   fitter = BruteForce(grid)

The Fitting Algorithm
----------------------

The ``BruteForce`` class implements a multi-stage optimization strategy to efficiently evaluate the grid while maintaining accuracy.

Stage 1: Magnitude Space Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Quickly eliminate obviously poor fits

**Method**:

1. For each grid point with fixed (:math:`M_{\rm init}`, EEP, [Fe/H], [α/Fe]):
2. Solve for best-fit (:math:`\mu`, :math:`A_V`) in magnitude space using analytical approximation
3. Compute approximate :math:`\chi^2` goodness-of-fit
4. Keep only grid points with :math:`\chi^2 < \chi^2_{\rm max}` (default: top 10% of models)

**Why magnitude space?** In magnitude space, the model is linear in :math:`\mu` and :math:`A_V \times R`, allowing fast least-squares solutions. However, magnitude space is not statistically correct because measurement errors are not Gaussian in magnitude space.

Stage 2: Flux Space Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Refine distance and extinction estimates in statistically correct flux space

**Method**:

1. For surviving grid points from Stage 1:
2. Convert observed magnitudes and errors to flux densities
3. Optimize (:math:`d`, :math:`A_V`, :math:`R_V`) using gradient-based minimizer (scipy.optimize)
4. Compute proper log-likelihood in flux space: :math:`\ln \mathcal{L} = -0.5 \sum_{\rm bands} (F_{\rm obs} - F_{\rm model})^2 / \sigma_F^2`
5. Keep grid points within likelihood threshold of best fit

**Why flux space?** Flux errors are approximately Gaussian (unlike magnitude errors), making the likelihood statistically correct.

Stage 3: Bayesian Posterior with Priors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Incorporate prior information to get full posterior distribution

**Method**:

1. For grid points passing Stage 2:
2. Add prior probability contributions:

   - :math:`\ln \pi(M_{\rm init})` from IMF
   - :math:`\ln \pi(d, \ell, b)` from Galactic structure
   - :math:`\ln \pi([{\rm Fe/H}], d, \ell, b)` from metallicity distribution
   - :math:`\ln \pi(t_{\rm age}, d, \ell, b)` from age distribution (via EEP-age mapping)
   - :math:`\ln \pi(A_V, R_V, d, \ell, b)` from dust maps

3. Add parallax likelihood if available: :math:`\ln \mathcal{L}_{\varpi} = -0.5 (\varpi_{\rm obs} - 1000/d)^2 / \sigma_\varpi^2`
4. Compute full log-posterior: :math:`\ln P = \ln \mathcal{L}_{\rm phot} + \ln \mathcal{L}_{\varpi} + \ln \pi`

Stage 4: Marginalization and Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Produce posterior samples for derived quantities

**Method**:

1. Normalize posterior across all grid points: :math:`P_i = \exp(\ln P_i - \ln P_{\rm max})`
2. Use importance sampling to draw samples weighted by posterior probability
3. For each sample, compute derived quantities (distance, extinction, stellar parameters)
4. Return ensemble of posterior samples

The result is a Monte Carlo representation of the full posterior distribution, properly accounting for:

- Measurement uncertainties (photometry, parallax)
- Parameter degeneracies (distance-extinction, mass-age-metallicity)
- Prior information (Galactic models, IMF, dust maps)

Optimization Details
^^^^^^^^^^^^^^^^^^^^

**Distance bounds**: brutus searches distances from 10 pc to 100 kpc by default

**Extinction bounds**: :math:`A_V \in [0, A_{V,{\rm max}}]` where :math:`A_{V,{\rm max}}` comes from dust map priors

**:math:`R_V` bounds**: Typically :math:`R_V \in [2.0, 6.0]` based on empirical dust studies

**Gradient-based optimization**: Uses scipy's L-BFGS-B algorithm for bounded optimization in flux space

**Numerical stability**: Automatically handles cases where optimization fails (e.g., impossible parameter combinations) by assigning very low likelihood

Performance Considerations
--------------------------

Grid Resolution Trade-offs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Finer grids** (more points):

✓ Better parameter resolution
✓ Smoother posterior distributions
✗ Larger file sizes (10+ GB)
✗ Slower fitting (more grid points to evaluate)

**Coarser grids** (fewer points):

✓ Faster fitting
✓ Smaller file sizes (1-2 GB)
✗ Discretization artifacts in posteriors
✗ May miss narrow features

**Recommendation**: Use fine grids (~5-10 million points) for publication-quality results, coarse grids (~1 million points) for exploratory analysis.

Parallelization
^^^^^^^^^^^^^^^

Grid evaluation is embarrassingly parallel. brutus supports:

- **Multi-threading**: Within-star parallelization (evaluating grid points in parallel)
- **Multi-processing**: Across-star parallelization (fitting multiple stars in parallel)

For large samples, parallelize at the star level:

.. code-block:: python

   from multiprocessing import Pool
   from brutus.analysis import BruteForce

   def fit_star(star_data):
       phot, phot_err, parallax, parallax_err = star_data
       return fitter.fit(phot, phot_err, parallax=parallax,
                        parallax_err=parallax_err)

   with Pool(processes=32) as pool:
       results = pool.map(fit_star, star_data_list)

Memory Usage
^^^^^^^^^^^^

**Grid storage**: HDF5 files with compression typically ~0.1-1 KB per grid point

**Runtime memory**: Depends on number of filters and grid points kept after Stage 1. Typical usage: 1-4 GB per fitting process.

**Recommendation**: For very large grids (>10M points), use memory-mapped HDF5 files to avoid loading entire grid into RAM.

When to Use Grids vs On-the-Fly Models
---------------------------------------

Use **Pre-computed Grids** (``StarGrid`` + ``BruteForce``) when:

✓ Fitting large samples (>1000 stars) with same filter set
✓ Speed is critical
✓ Filters are standard survey combinations (Gaia, 2MASS, WISE, Pan-STARRS, etc.)
✓ Publication-quality uncertainties needed

Use **On-the-Fly Models** (``StarEvolTrack`` + ``StellarPop``) when:

✓ Exploring different filter combinations
✓ Prototyping or testing
✓ Custom stellar models or modifications
✓ Memory is limited
✓ Fitting small samples (<100 stars)

.. seealso::
   See :doc:`choosing_options` for detailed guidance on selecting fitting strategies.

Common Issues and Solutions
---------------------------

**Grid does not cover observed stars**
   Symptoms: Warnings about extrapolation, poor fits

   Solution: Check grid parameter ranges cover your data. For very metal-poor stars, extend [Fe/H] grid. For very young/old populations, check EEP/age coverage.

**Fitting is very slow**
   Symptoms: >10 seconds per star

   Solutions: (1) Use coarser grid, (2) Reduce Stage 1 threshold to keep fewer grid points, (3) Limit distance range, (4) Parallelize across stars

**Posterior has discrete jumps**
   Symptoms: Step-like features in distance or extinction posteriors

   Solution: Grid resolution too coarse. Use finer grid spacing in problematic parameter range.

**Out of memory errors**
   Symptoms: Python crashes during grid loading

   Solution: Use memory-mapped HDF5 files (``memmap=True`` in load_models), or reduce grid size by limiting parameter ranges.

Examples
--------

Fitting with a Pre-computed Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from brutus.data import load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   # Load grid
   models, labels, params = load_models('grid_gaiadr3_2mass_wise.h5')
   grid = StarGrid(models, labels, params)

   # Initialize fitter
   fitter = BruteForce(grid)

   # Observed data (Gaia G, BP, RP + 2MASS J, H, Ks + WISE W1, W2)
   phot = np.array([16.5, 17.2, 15.8, 14.1, 13.5, 13.3, 13.1, 13.0])
   phot_err = np.array([0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.05, 0.05])
   parallax = 2.5  # mas
   parallax_err = 0.1  # mas

   # Fit
   results = fitter.fit(
       phot, phot_err,
       parallax=parallax, parallax_err=parallax_err,
       n_samples=10000  # Number of posterior samples
   )

   # Extract results
   dist_median = np.median(results['dist_samples'])  # Median distance
   av_median = np.median(results['av_samples'])  # Median extinction
   print(f"Distance: {dist_median:.1f} ± {np.std(results['dist_samples']):.1f} pc")
   print(f"Extinction: {av_median:.3f} ± {np.std(results['av_samples']):.3f} mag")

Creating a Custom Grid
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from brutus.core import GridGenerator, EEPTracks
   import numpy as np

   # Initialize tracks
   tracks = EEPTracks()

   # Create generator for specific science case:
   # Gaia + ground-based photometry for young stars
   generator = GridGenerator(
       tracks,
       filters=['bp', 'g', 'rp', 'j', 'h', 'ks']  # Gaia + 2MASS
   )

   # Custom parameter ranges for young stars
   mini_grid = np.logspace(np.log10(0.1), np.log10(10.0), 300)  # 0.1-10 Msun
   eep_grid = np.linspace(202, 454, 150)  # Pre-MS through main sequence
   feh_grid = np.linspace(-1.0, 0.5, 20)  # Solar neighborhood metallicities
   afe_grid = np.array([0.0, 0.2])  # Solar and alpha-enhanced

   # Generate grid
   generator.make_grid(
       output_file='young_stars_grid.h5',
       mini_grid=mini_grid,
       eep_grid=eep_grid,
       feh_grid=feh_grid,
       afe_grid=afe_grid,
       verbose=True
   )

Summary
-------

- brutus uses **pre-computed model grids** to enable fast, systematic parameter space exploration
- Grids store **photometry at 1 kpc reference distance** plus **reddening coefficients**
- **Multi-stage optimization** (magnitude → flux → Bayesian posterior) balances speed and accuracy
- **Grid resolution** trades file size and speed against parameter precision
- Use ``GridGenerator`` to create **custom grids** for specific filter combinations and science cases

Next Steps
----------

- Understand the prior distributions: :doc:`priors`
- Learn about cluster fitting: :doc:`cluster_modeling`
- Choose appropriate options: :doc:`choosing_options`
- Interpret fitting results: :doc:`understanding_results`

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", arXiv:2503.02227
