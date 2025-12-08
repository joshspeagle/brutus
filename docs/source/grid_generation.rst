Grid-Based Fitting
==================

This page explains the **grid-based fitting approach** used by ``brutus`` and how to work with pre-computed stellar model grids.

.. tip::
   For the list of supported photometric filters, see :ref:`available-filters` in the Stellar Models page.

The Hybrid Approach
-------------------

``brutus`` uses a **hybrid approach** that combines grid-based evaluation with linear regression and Monte Carlo sampling. This design exploits the strengths of each method while mitigating their weaknesses.

From the paper (Speagle et al. 2025, §3 and Appendix B):

   *"brutus uses a combination of linear regression, Monte Carlo sampling, and brute force methods to generate fast but robust approximations to the underlying posterior."*

**Why grids?**

Stellar parameter inference from photometry has characteristics that make grid-based evaluation attractive:

- **Multi-modal posteriors**: A single SED can match multiple stellar solutions (e.g., nearby dwarf vs distant giant). Grid evaluation naturally captures all modes without getting trapped.

- **Amortized computation**: The same stellar models apply to millions of stars. Pre-computing the grid once spreads the cost across all targets.

- **Systematic coverage**: Grids guarantee exploration of the full parameter space, avoiding gaps that sampling might miss.

**Why also sampling?**

Pure grid-based approaches have limitations:

- Grids over 5+ parameters become prohibitively large
- Distance, :math:`A_V`, and :math:`R_V` are continuous parameters poorly suited to coarse discretization
- Prior integration requires numerical methods

``brutus`` addresses these by:

1. Using grids only for **intrinsic stellar parameters** (mass, EEP, metallicity)
2. Solving for **extrinsic parameters** (distance, extinction) via linear regression at each grid point
3. Using **importance sampling** to incorporate priors and produce posterior draws

.. note::
   Because ``brutus`` uses Monte Carlo sampling in its final stage, results have small stochastic variations between runs. For reproducibility, set a random seed before fitting.

Grid Structure
--------------

A ``brutus`` model grid (``StarGrid``) maps intrinsic stellar parameters to photometric coefficients:

.. math::

   (M_{\rm init}, {\rm EEP}, [{\rm Fe/H}]) \rightarrow \{\mathbf{M}_{\rm ref}, \mathbf{R}, \mathbf{R}'\}

where:

- :math:`\mathbf{M}_{\rm ref}` are absolute magnitudes at a **reference distance of 1 kpc**
- :math:`\mathbf{R}`, :math:`\mathbf{R}'` are reddening coefficients

Apparent magnitudes for any distance and extinction are computed as:

.. math::

   m_{\rm band}(d, A_V, R_V) = M_{\rm ref,band} + \mu(d) + A_V \times (R_{\rm band} + R_V \times R'_{\rm band})

This parameterization allows modeling arbitrary extinction without storing separate grids for each :math:`(A_V, R_V)` combination.

Typical Grid Coverage
^^^^^^^^^^^^^^^^^^^^^

Pre-computed grids typically span:

- **Initial mass**: 0.1 to 10 M☉ (finer spacing at low masses)
- **EEP**: 202 to 808 (pre-MS through AGB)
- **Metallicity [Fe/H]**: -2.0 to +0.5 dex

Grids contain :math:`\sim 10^5` to :math:`10^6` models and are several hundred MB to a few GB.

Available Pre-Computed Grids
----------------------------

Download grids with :func:`~brutus.data.fetch_grids`:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Grid
     - Description
   * - ``mist_v9`` (default)
     - MIST v1.2 with empirical corrections. Recommended for most applications.
   * - ``mist_v8``
     - Earlier version of MIST v1.2 corrections.
   * - ``bayestar_v5``
     - Bayestar-compatible grid (Pan-STARRS + 2MASS filters only).

All grids include photometry for the full set of :ref:`available-filters`. When loading, specify which filters you need:

.. code-block:: python

   from brutus.data import fetch_grids, load_models
   from brutus.core import StarGrid

   # Download grid (first time only)
   fetch_grids(grid='mist_v9')

   # Load with specific filters
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)
   grid = StarGrid(models, labels, label_mask)

Creating Custom Grids
---------------------

Use :class:`~brutus.core.GridGenerator` for custom filter combinations or parameter ranges:

.. code-block:: python

   import numpy as np
   from brutus.core import GridGenerator, EEPTracks

   # Initialize with evolutionary tracks and desired filters
   tracks = EEPTracks()
   generator = GridGenerator(
       tracks,
       filters=['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
                'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']
   )

   # Define parameter grids
   mini_grid = np.arange(0.5, 2.0, 0.025)
   eep_grid = np.arange(202, 808, 5)
   feh_grid = np.arange(-2.0, 0.5, 0.1)

   # Generate and save
   generator.make_grid(
       output_file='my_grid.h5',
       mini_grid=mini_grid,
       eep_grid=eep_grid,
       feh_grid=feh_grid,
       verbose=True
   )

For non-uniform spacing (e.g., finer resolution for low-mass stars):

.. code-block:: python

   mini_grid = np.concatenate([
       np.arange(0.1, 1.0, 0.02),    # Fine spacing for M/K dwarfs
       np.arange(1.0, 3.0, 0.05),    # Medium spacing
       np.arange(3.0, 10.0, 0.2)     # Coarse spacing for massive stars
   ])

.. warning::
   Grid generation is computationally expensive. A grid with ~100k models takes minutes; larger grids can take hours. Start with coarse grids for testing.

The Fitting Algorithm
---------------------

:class:`~brutus.analysis.BruteForce` implements a multi-stage fitting strategy designed to be both fast and robust. For full details, see Speagle et al. (2025) §3 and Appendix B.

**Stage 1: Magnitude-Space Screening**

For each grid point, solve for best-fit (distance, :math:`A_V`, :math:`R_V`) using fast linear regression in magnitude space. Discard grid points with poor :math:`\chi^2`.

**Stage 2: Flux-Space Refinement**

Transform remaining solutions to flux space (where errors are Gaussian) and refine the extrinsic parameters. Apply stricter :math:`\chi^2` cuts.

**Stage 3: Prior Integration via Importance Sampling**

At each surviving grid point, draw samples from the approximate likelihood and compute importance weights that incorporate:

- Parallax likelihood (if parallax provided)
- Initial mass function (IMF)
- Galactic structure (3-D stellar density)
- Metallicity and age distributions
- 3-D dust extinction (if dust map provided)

This step dominates the computation time.

**Stage 4: Posterior Resampling**

Use bootstrap resampling from the importance-weighted samples to produce Monte Carlo draws of stellar parameters (mass, EEP, metallicity) and extrinsic parameters (distance, :math:`A_V`, :math:`R_V`).

.. note::
   The screening stages (1-2) can occasionally clip rare but valid solutions. If you suspect this is happening (e.g., for unusual stars), try relaxing the ``chi2_threshold`` parameter.

Complete Example
----------------

.. code-block:: python

   import h5py
   import numpy as np
   from brutus.data import fetch_grids, load_models
   from brutus.core import StarGrid
   from brutus.analysis import BruteForce

   # Download and load grid
   fetch_grids(grid='mist_v9')
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)
   grid = StarGrid(models, labels, label_mask)

   # Initialize fitter
   fitter = BruteForce(grid)

   # Prepare observed data (flux densities, not magnitudes!)
   # Shape: (Nstars, Nfilters)
   flux = np.array([[...]])       # Linear flux densities
   flux_err = np.array([[...]])   # Flux uncertainties
   mask = np.array([[True, True, True, True, True, True]])
   obj_labels = np.array([[1]])   # Object identifier

   # Optional: parallax and coordinates for priors
   parallax = np.array([2.5])       # mas
   parallax_err = np.array([0.1])   # mas
   coords = np.array([[45.0, 10.0]])  # Galactic (l, b) in degrees

   # Fit
   output_file = fitter.fit(
       data=flux,
       data_err=flux_err,
       data_mask=mask,
       data_labels=obj_labels,
       save_file='results.h5',
       parallax=parallax,
       parallax_err=parallax_err,
       data_coords=coords,
       Ndraws=250
   )

   # Read results
   with h5py.File(output_file, 'r') as f:
       distances = f['samps_dist'][:]  # (Nstars, Ndraws) in kpc
       av_values = f['samps_red'][:]   # A_V samples
       rv_values = f['samps_rv'][:]    # R_V samples

   # Summarize
   dist_pc = np.median(distances[0]) * 1000
   dist_err = np.std(distances[0]) * 1000
   print(f"Distance: {dist_pc:.0f} +/- {dist_err:.0f} pc")
   print(f"A_V: {np.median(av_values[0]):.2f} +/- {np.std(av_values[0]):.2f} mag")

Performance Considerations
--------------------------

The :meth:`~brutus.analysis.BruteForce.fit` method provides several parameters for tuning performance and resource usage.

**I/O Mode** (``running_io``):

- ``running_io=True`` (default): Writes results to disk after each star. Safer for long runs—if interrupted, completed stars are saved.
- ``running_io=False``: Accumulates all results in memory, writes once at end. Faster when disk I/O is slow, but loses all progress if interrupted.

**Memory Management** (``mem_lim``):

The ``mem_lim`` parameter (default 8000 MB) controls memory allocation for Monte Carlo sampling. The fitter automatically scales the number of samples to stay within this budget. Reduce this on memory-constrained systems:

.. code-block:: python

   fitter.fit(..., mem_lim=4000.0)  # Limit to ~4 GB for MC sampling

**Reducing Grid Memory**:

Grids are loaded entirely into memory. To reduce footprint, load only the filters you need:

.. code-block:: python

   # Load only Gaia + 2MASS (6 filters) instead of all ~40
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   models, labels, mask = load_models('grid_mist_v9.h5', filters=filters)

**Sampling Resolution** (``Nmc_prior``, ``Ndraws``):

- ``Nmc_prior`` (default 50): Monte Carlo samples per grid point for prior integration. The dominant computational cost.
- ``Ndraws`` (default 250): Number of posterior samples saved per star.

For faster exploratory fits, reduce these:

.. code-block:: python

   fitter.fit(..., Nmc_prior=25, Ndraws=100)  # Faster but lower resolution

**Reproducibility** (``rstate``):

Pass a ``numpy.random.RandomState`` for reproducible results:

.. code-block:: python

   import numpy as np
   rstate = np.random.RandomState(42)
   fitter.fit(..., rstate=rstate)

See Also
--------

- :doc:`stellar_models` - MIST models and :ref:`available filters <available-filters>`
- :doc:`priors` - Prior probability distributions
- :doc:`understanding_results` - Interpreting output files
- :doc:`faq` - Troubleshooting and common questions

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_
