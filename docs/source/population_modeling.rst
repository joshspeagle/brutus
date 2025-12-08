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

Outlier Likelihood
^^^^^^^^^^^^^^^^^^

Field contaminants are modeled with an adaptive outlier distribution. By default, ``brutus`` uses a chi-square outlier model that assigns probability based on how poorly the data fits any reasonable model:

.. math::

   \mathcal{L}_{\rm outlier}(\hat{F}_i) = \mathcal{L}_{\rm cluster}(\chi^2_{\rm max}(k_i), k_i)

where :math:`\chi^2_{\rm max}` is the chi-square value at a cumulative probability threshold (default 99.9%), and :math:`k_i` is the number of photometric bands plus parallax if available. This adaptive threshold is more conservative than a uniform outlier model, retaining borderline members.

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

This integral is computed numerically over a grid of (EEP, SMF) points, with Jacobian corrections for the non-uniform mass spacing along the isochrone.

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

The default SMF grid has 15 values from 0.0 to 1.0, with finer sampling near equal-mass binaries where the photometric effect is largest.

.. note::
   Binary modeling is restricted to main-sequence stars (EEP < 480) to avoid unphysical configurations like two red giants in a close binary.

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

**Grid Size**

The default grid uses 2000 EEP points × 15 SMF values. After applying mass bounds and binary constraints (EEP < 480 for binaries), the effective grid size is typically 10,000-20,000 points per likelihood evaluation.

Custom grids can be specified for faster iteration:

.. code-block:: python

   # Coarser grid for development/testing
   coarse_eep = np.linspace(202, 808, 500)
   coarse_smf = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

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
- :doc:`faq` - Troubleshooting and common questions

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_ (see Appendix D for cluster model details)
