Population-Based Modeling
=========================

This page explains the **isochrone-based population modeling** approach used in ``brutus`` for fitting stellar clusters and coeval populations. For individual field star fitting, see :doc:`grid_generation`.

Overview
--------

Stellar clusters offer powerful constraints because all member stars share the same:

- **Age** (coeval formation)
- **Metallicity** (common birth cloud)
- **Distance** (spatial coherence)
- **Extinction** (similar sightline)

``brutus`` exploits these constraints by fitting a single :term:`isochrone` to all cluster members simultaneously, rather than fitting each star independently. This dramatically reduces degeneracies and improves parameter precision.

The key challenge is **field contamination**: not all stars in the field of view are cluster members. ``brutus`` handles this using a **mixture model** that probabilistically separates cluster members from field interlopers.

For full details on the cluster model, see Speagle et al. (2025) Appendix D.

The Likelihood Function
-----------------------

Population Parameters
^^^^^^^^^^^^^^^^^^^^^

The population is described by six parameters:

.. math::

   \boldsymbol{\theta} = \left[ \text{[Fe/H]}, \log(\text{age}), A_V, R_V, d, f_{\rm field} \right]

where:

- **[Fe/H]**: Metallicity (dex)
- **log(age)**: Logarithm of age in years (e.g., 9.0 = 1 Gyr)
- **A_V**: V-band extinction (magnitudes)
- **R_V**: Extinction curve shape parameter
- **d**: Distance (parsecs)
- **f_field**: Field contamination fraction (0 to 1)

Cluster Likelihood
^^^^^^^^^^^^^^^^^^

For each star, the cluster membership likelihood compares observed photometry to isochrone predictions:

.. math::

   \ln \mathcal{L}_{\rm cluster}(\hat{F}_i | M, q, \boldsymbol{\theta}) = \ln \mathcal{L}_{\rm phot} + \ln \mathcal{L}_{\rm parallax}

The **photometric likelihood** uses a chi-square formulation:

.. math::

   \ln \mathcal{L}_{\rm phot} = -\frac{1}{2} \sum_{\rm bands} \frac{(\hat{F}_{i,j} - F_{j})^2}{\sigma_{F,i,j}^2}

The **parallax likelihood** (if available) constrains distance:

.. math::

   \ln \mathcal{L}_{\rm parallax} = -\frac{1}{2} \frac{(\hat{\varpi}_i - 1000/d)^2}{\sigma_{\varpi,i}^2}

.. note::
   With the default ``dim_prior=True``, the photometric and parallax chi-squares are combined — the parallax counts as one extra degree of freedom, matching the ``BruteForce`` convention — and evaluated under a chi-square *distribution* log-PDF rather than as the Gaussian log-densities shown above. Set ``dim_prior=False`` to use the Gaussian formulation.

Outlier Likelihood
^^^^^^^^^^^^^^^^^^

Field contaminants are modeled with an adaptive outlier distribution. By default, ``brutus`` uses a chi-square outlier model that assigns probability based on how poorly the data fits any reasonable model:

.. math::

   \mathcal{L}_{\rm outlier}(\hat{F}_i) = \mathcal{L}_{\rm cluster}(\chi^2_{\rm max}(k_i), k_i)

where :math:`\chi^2_{\rm max}` is the chi-square value at a cumulative probability threshold (99.999% by default, i.e. ``p_value_cut=1e-5``), and :math:`k_i` is the number of photometric bands plus parallax if available. An alternative uniform outlier model (used when ``dim_prior=False``) instead treats outliers as uniformly distributed over the observed flux range in each band, giving a proper flux density directly comparable to the Gaussian inlier likelihood.

Mixture Model
^^^^^^^^^^^^^

The mixture model combines cluster and outlier probabilities at each grid point:

.. math::

   P(\hat{F}_i | M, q, \boldsymbol{\theta}) = w_c \cdot P_{\rm cluster} + w_o \cdot P_{\rm outlier}

where the weights are:

- :math:`w_c = P_{\rm mem} \times (1 - f_{\rm field})` — probability of being a true cluster member
- :math:`w_o = 1 - w_c` — probability of being a field star or outlier

Here :math:`P_{\rm mem}` is the **external membership probability** (e.g., from proper motion analysis) and :math:`f_{\rm field}` is the **fitted field fraction**.

Marginalization
^^^^^^^^^^^^^^^

After applying the mixture model, ``brutus`` marginalizes over stellar parameters (mass :math:`M` and secondary mass fraction :math:`q`):

.. math::

   P(\hat{F}_i | \boldsymbol{\theta}) = \int \int P(\hat{F}_i | M, q, \boldsymbol{\theta}) \, \frac{dM}{dEEP} \, dEEP \, dq

This integral is computed numerically over a grid of (EEP, SMF) points, with Jacobian corrections for the non-uniform mass spacing along the isochrone. The integration measure is normalized over the valid grid points, making the flat (mass, SMF) measure a proper uniform prior — without this, the :math:`\boldsymbol{\theta}`-dependent grid volume would multiply every mixture component (including the :math:`\boldsymbol{\theta}`-independent outlier model), biasing the population parameters and the field fraction.

Total Likelihood
^^^^^^^^^^^^^^^^

The total population likelihood is the product over all stars:

.. math::

   \ln \mathcal{L}_{\rm total}(\boldsymbol{\theta}) = \sum_{i=1}^{N_{\rm stars}} \ln P(\hat{F}_i | \boldsymbol{\theta})

Binary Star Modeling
--------------------

``brutus`` includes binary stars through the **secondary mass fraction** (SMF or :math:`q`) parameter:

.. math::

   q = \frac{M_{\rm secondary}}{M_{\rm primary}}

where :math:`q = 0` is a single star and :math:`q = 1` is an equal-mass binary.

Binary photometry is computed by adding the fluxes of both components:

.. math::

   F_{\rm binary} = F_{\rm primary} + F_{\rm secondary}

The default SMF grid uses 21 uniformly-spaced values from 0.0 to 1.0.

.. note::
   Binary modeling is restricted to main-sequence stars (EEP ≤ 480, the ``eep_binary_max`` default) to avoid unphysical configurations like two red giants in a close binary. Models above this cutoff are SMF-independent and are stored once, carrying the full SMF integration measure so they are weighted consistently with the main-sequence models.

Basic Usage
-----------

The :func:`~brutus.analysis.isochrone_population_loglike` function computes the log-likelihood for a set of population parameters:

.. code-block:: python

   import numpy as np
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis import isochrone_population_loglike

   # Set up population model with specific filters
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   iso = Isochrone()
   pop = StellarPop(iso, filters=filters)

   # Example observed data (N_stars=100, N_filters=6)
   # Flux densities in units of 10**(-0.4 * mag)
   flux = np.random.rand(100, 6) * 1e-3      # Replace with real data
   flux_err = flux * 0.02                     # 2% errors
   parallax = np.full(100, 2.0)               # 2 mas (500 pc)
   parallax_err = np.full(100, 0.1)           # 0.1 mas errors

   # Population parameters: [feh, loga, av, rv, dist, field_frac]
   theta = np.array([0.0, 9.0, 0.3, 3.3, 500.0, 0.05])

   # Compute log-likelihood
   lnl = isochrone_population_loglike(
       theta,
       stellarpop=pop,
       obs_phot=flux,
       obs_err=flux_err,
       parallax=parallax,
       parallax_err=parallax_err,
   )
   print(f"Log-likelihood: {lnl:.2f}")

Using with Samplers
-------------------

The likelihood function is designed for use with external MCMC or optimization codes.

Optimization (Point Estimate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis import isochrone_population_loglike

   # Set up model (as above)
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   iso = Isochrone()
   pop = StellarPop(iso, filters=filters)

   # Your observed data
   # flux, flux_err, parallax, parallax_err = load_your_data()

   def neg_lnlike(theta):
       return -isochrone_population_loglike(
           theta, pop, flux, flux_err,
           parallax=parallax,
           parallax_err=parallax_err
       )

   # Initial guess: [feh, loga, av, rv, dist, field_frac]
   theta0 = np.array([0.0, 9.0, 0.3, 3.3, 1000.0, 0.05])

   result = minimize(neg_lnlike, theta0, method='Nelder-Mead')
   print(f"Best-fit parameters: {result.x}")

MCMC Sampling (Full Posterior)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import emcee
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis import isochrone_population_loglike

   # Set up model
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   iso = Isochrone()
   pop = StellarPop(iso, filters=filters)

   # Your observed data
   # flux, flux_err, parallax, parallax_err = load_your_data()

   def lnprior(theta):
       feh, loga, av, rv, dist, f_field = theta
       # Uniform priors with bounds
       if not (-2.5 < feh < 0.5):
           return -np.inf
       if not (6.0 < loga < 10.5):
           return -np.inf
       if not (0.0 <= av < 5.0):
           return -np.inf
       if not (2.0 < rv < 5.0):
           return -np.inf
       if not (100 < dist < 10000):
           return -np.inf
       if not (0.0 <= f_field < 0.5):
           return -np.inf
       return 0.0

   def lnprob(theta):
       lp = lnprior(theta)
       if not np.isfinite(lp):
           return -np.inf
       return lp + isochrone_population_loglike(
           theta, pop, flux, flux_err,
           parallax=parallax,
           parallax_err=parallax_err
       )

   # Initialize walkers (use many more walkers than parameters)
   ndim = 6
   nwalkers = 128  # At least 2*ndim, but more is better
   theta0 = np.array([0.0, 9.0, 0.3, 3.3, 1000.0, 0.05])
   p0 = theta0 + 1e-3 * np.random.randn(nwalkers, ndim)

   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(p0, 5000, progress=True)

   # Extract samples (discard burn-in)
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

Nested Sampling
^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import dynesty
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis import isochrone_population_loglike

   # Set up model
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   iso = Isochrone()
   pop = StellarPop(iso, filters=filters)

   # Your observed data
   # flux, flux_err, parallax, parallax_err = load_your_data()

   def prior_transform(u):
       """Transform unit cube to parameter space."""
       theta = np.zeros(6)
       theta[0] = -2.5 + 3.0 * u[0]      # feh: [-2.5, 0.5]
       theta[1] = 6.0 + 4.5 * u[1]       # loga: [6.0, 10.5]
       theta[2] = 5.0 * u[2]             # av: [0, 5]
       theta[3] = 2.0 + 3.0 * u[3]       # rv: [2, 5]
       theta[4] = 100 + 9900 * u[4]      # dist: [100, 10000] pc
       theta[5] = 0.5 * u[5]             # f_field: [0, 0.5]
       return theta

   def lnlike(theta):
       return isochrone_population_loglike(
           theta, pop, flux, flux_err,
           parallax=parallax,
           parallax_err=parallax_err
       )

   sampler = dynesty.NestedSampler(lnlike, prior_transform, ndim=6)
   sampler.run_nested()
   results = sampler.results

Per-Object Membership Probabilities
-----------------------------------

The ``cluster_prob`` parameter specifies the prior probability that each star is a cluster member (before considering photometry). This can be:

- A **scalar** (same for all stars): ``cluster_prob=0.95``
- A **per-object array** from external analysis (e.g., proper motions)

.. code-block:: python

   # From proper motion / radial velocity analysis
   # membership_prob = compute_kinematic_membership(proper_motions, radial_velocities)
   membership_prob = np.random.uniform(0.8, 1.0, size=100)  # Example

   lnl = isochrone_population_loglike(
       theta, pop, flux, flux_err,
       cluster_prob=membership_prob,  # shape (N_stars,)
       parallax=parallax,
       parallax_err=parallax_err,
   )

This allows incorporating kinematic membership information while still fitting for additional photometric field contamination via ``f_field``.

Diagnostics
-----------

Use ``return_components=True`` to inspect intermediate results:

.. code-block:: python

   lnl, components = isochrone_population_loglike(
       theta, pop, flux, flux_err,
       return_components=True
   )

   # Available diagnostic outputs
   print(f"Total log-likelihood: {components['lnl_total']:.2f}")

   # Per-object likelihoods (identify problem stars)
   lnl_per_star = components['lnl_per_object']  # shape (N_stars,)
   worst_stars = np.argsort(lnl_per_star)[:5]
   print(f"Worst-fit stars: {worst_stars}")

   # Cluster vs outlier likelihoods (check mixture)
   # These have shape (N_grid_points, N_objects)
   lnl_cluster = components['lnl_cluster']
   lnl_outlier = components['lnl_outlier']
   lnl_mixture = components['lnl_mixture']

   # The isochrone grid used
   grid = components['isochrone_grid']

Performance Considerations
--------------------------

Timing Benchmarks
^^^^^^^^^^^^^^^^^

The following benchmarks were measured with version 1.2.0 on a 4-core Linux host using 3 Gaia bands and a synthetic 100-star cluster at 1 kpc (min-of-N wall-clock; the script is ``bench/bench_populations.py`` in the repository). All times are per evaluation of the full log-likelihood. Version 1.2.0 made binary modeling functional (previously every SMF slice silently evaluated to the same single-star isochrone), so grid generation now does real per-SMF work and the convergence behavior below reflects a genuinely binary-aware likelihood.

**Pipeline stage breakdown** (100 stars, default grid):

.. list-table::
   :widths: 40 20 20
   :header-rows: 1

   * - Stage
     - Time (ms)
     - Fraction
   * - Grid generation (fixed cost)
     - 47
     - 29%
   * - Cluster loglike
     - 33
     - 21%
   * - Mixture model
     - 27
     - 17%
   * - Marginalization
     - 23
     - 14%
   * - Outlier loglike
     - <1
     - <1%
   * - **Total (end-to-end)**
     - **~160**
     -

The individual stages sum to ~130 ms; the remainder of the end-to-end time is array bookkeeping in the wrapper.

**Scaling with number of stars:**

.. list-table::
   :widths: 30 30 30
   :header-rows: 1

   * - N_stars
     - Time per eval (ms)
     - Per-star cost (ms)
   * - 10
     - 58
     - 5.8
   * - 50
     - 90
     - 1.8
   * - 100
     - 163
     - 1.6
   * - 200
     - 337
     - 1.7
   * - 500
     - 952
     - 1.9

Grid generation is a fixed cost (~47 ms), so per-star cost flattens to ~1.6-1.9 ms for larger samples. Adding more photometric bands increases cost modestly.

Grid Resolution and Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default grid uses 1000 EEP points :math:`\times` 21 SMF values. After applying mass bounds and binary constraints (EEP ≤ 480 for binaries; post-main-sequence models are stored once, not per SMF slice), the effective grid size is typically ~9,500-10,200 points per evaluation.

**EEP convergence** (measured as :math:`\Delta \ln \mathcal{L}` vs 5000-point reference, 100 stars):

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - N_EEP
     - Grid size
     - :math:`\Delta \ln \mathcal{L}`
     - Time (ms)
   * - 200
     - 2,040
     - -25.6
     - 38
   * - 500
     - 5,080
     - -5.7
     - 70
   * - **1000 (default)**
     - **10,180**
     - **+7.0**
     - **152**
   * - 2000
     - 20,360
     - +2.3
     - 338
   * - 5000 (reference)
     - 50,880
     - 0
     - 1064

With binary modeling active the likelihood surface is richer than in earlier releases, so EEP convergence is somewhat slower: the default 1000 points sit within :math:`|\Delta \ln \mathcal{L}| \approx 7` of the reference per 100 stars (:math:`\lesssim 0.1` per star), and 2000 points within :math:`\approx 2`.

**SMF convergence** (measured vs 31-point uniform reference, 100 stars):

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - SMF config
     - Grid size
     - :math:`\Delta \ln \mathcal{L}`
     - Time (ms)
   * - Singles only (N=1)
     - 1,000
     - -184.4
     - 10
   * - 7 uniform
     - 3,754
     - +4.0
     - 43
   * - 15 uniform
     - 7,426
     - -1.1
     - 112
   * - **21 uniform (default)**
     - **10,180**
     - **+0.4**
     - **166**
   * - 31 uniform (reference)
     - 14,770
     - 0
     - 273

The singles-only configuration is catastrophically wrong for a population containing unresolved binaries (:math:`\Delta \ln \mathcal{L} \approx -184` per 100 stars) — binary modeling matters. Among binary-aware grids, convergence is quick: 7 uniform points are within :math:`|\Delta \ln \mathcal{L}| \approx 4` per 100 stars and the default 21 points within :math:`\approx 0.4`.

Custom grids can be specified for faster iteration during development:

.. code-block:: python

   # Coarser grid for testing (~5x faster)
   coarse_eep = np.linspace(202, 808, 200)
   coarse_smf = np.linspace(0, 1, 7)

   lnl = isochrone_population_loglike(
       theta, pop, flux, flux_err,
       eep_grid=coarse_eep,
       smf_grid=coarse_smf,
   )

**Parallelization**

The likelihood function itself is not internally parallelized. For MCMC, parallelize at the sampler level:

.. code-block:: python

   from multiprocessing import Pool

   # Using variables from MCMC example above
   with Pool(processes=8) as pool:
       sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
       sampler.run_mcmc(p0, 5000, progress=True)

Limitations
-----------

- **Single population**: Assumes one coeval population plus field. Multiple populations (e.g., two clusters) require extensions.
- **No proper motion modeling**: Cluster membership from kinematics must be provided externally via ``cluster_prob``.
- **Binary simplifications**: Only photometric binaries; no orbital dynamics or mass transfer.
- **Extinction uniformity**: All cluster members share the same :math:`A_V`. Differential reddening requires extensions.

Technical Notes
---------------

**Mixture-before-marginalization**: ``brutus`` applies the mixture model (cluster vs outlier) at each grid point *before* marginalizing over stellar parameters. This is the mathematically correct approach for contaminated populations:

.. math::

   P(\hat{F}_i | \boldsymbol{\theta}) = \int \left[ w_c P_c(\hat{F}_i|M) + w_o P_o(\hat{F}_i) \right] dM

The alternative—marginalizing first, then mixing—can produce biased results because it compares integrated cluster probabilities against point outlier probabilities. See Appendix D of Speagle et al. (2025) for details.

See Also
--------

- :doc:`grid_generation` - Individual star fitting with ``BruteForce``
- :doc:`stellar_models` - MIST models and :ref:`available filters <available-filters>`
- :doc:`priors` - Prior probability distributions

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_ (see Appendix D for cluster model details)
