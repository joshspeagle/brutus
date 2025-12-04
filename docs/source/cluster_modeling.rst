Stellar Cluster Modeling
=========================

This page explains how brutus models **coeval stellar populations** such as open clusters and globular clusters using the **mixture-before-marginalization** approach.

Coeval Population Assumptions
------------------------------

Stellar clusters provide powerful constraints because all member stars share:

- **Common age**: All stars formed at the same time
- **Common metallicity**: All stars formed from the same gas cloud
- **Common distance**: Cluster size is negligible compared to distance
- **Common extinction**: Foreground dust affects all members similarly

These shared properties allow fitting a **single isochrone** to the entire population.

The Mixture-Before-Marginalization Approach
--------------------------------------------

A common but **incorrect** approach is to marginalize over stellar mass first, then handle field contamination. This biases results because outliers influence the marginalization.

The correct approach applies the mixture model **before** marginalizing:

.. math::

   \mathcal{L}(\Theta) = \prod_{i=1}^N \left[ \int \left( w_{\rm mem} \mathcal{L}_{\rm cluster} + w_{\rm field} \mathcal{L}_{\rm outlier} \right) \pi(M) \, dM \right]

where :math:`\Theta = ([{\rm Fe/H}], \log_{10}({\rm age}), A_V, R_V, d)` are population parameters, :math:`w_{\rm mem} = 1 - f_{\rm field}` is the membership probability, and :math:`\pi(M)` is the IMF prior.

This properly down-weights field contaminants during mass marginalization, preventing biases in age, metallicity, and distance estimates.

For detailed derivation, see Speagle et al. (2025) §5.

Core Functions
--------------

Generating Isochrone Population Grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fixed population parameters, generate a grid over stellar mass and binary fraction:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop
   from brutus.analysis.populations import generate_isochrone_population_grid

   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0,      # Solar metallicity
       loga=9.0,     # 1 Gyr (log10(age in years))
       av=0.5,       # A_V extinction
       rv=3.1,       # R_V
       dist=2000.0   # Distance in pc
   )

Population Log-Likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^

The main function for cluster fitting:

.. code-block:: python

   from brutus.analysis.populations import isochrone_population_loglike

   # theta = [feh, loga, av, rv, dist]
   theta = [0.0, 9.0, 0.5, 3.1, 2000.0]

   lnl = isochrone_population_loglike(
       theta,
       stellarpop=pop,
       obs_phot=obs_flux,      # (N_stars, N_filters)
       obs_err=obs_err,        # (N_stars, N_filters)
       parallax=parallax,      # Optional (N_stars,)
       parallax_err=parallax_err,
       cluster_prob=0.9,       # Prior: 90% likely members
       dim_prior=True,
   )

This function:

1. Generates an isochrone grid for the given population parameters
2. Computes cluster likelihood for each (grid_point, star) pair
3. Computes outlier likelihood for field contamination
4. Applies mixture model before marginalization
5. Returns total log-likelihood across all stars

Binary Stars
------------

Binaries are modeled via **Secondary Mass Fraction (SMF)**, also called ``binary_fraction``:

- **SMF = 0**: Single star (default)
- **SMF = 0.5**: Companion with half the primary mass
- **SMF = 1**: Equal-mass binary

**Key assumptions**:

1. Binary companions share the same age and metallicity as primaries
2. Combined photometry is the sum of fluxes from both components
3. Binaries are only modeled for main-sequence stars (EEP ≤ ``eep_binary_max``, default 480)
4. Post-MS stars are treated as single regardless of SMF setting

**Usage in StellarPop**:

.. code-block:: python

   # Single stars
   seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.0)

   # Binary population with 40% mass ratio
   seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=0.4)

   # Equal-mass binaries
   seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, binary_fraction=1.0)

**Interpreting params2**: When ``binary_fraction > 0``, the ``params2`` return value contains secondary component parameters. For single stars, these are NaN.

Complete Example: MCMC Fitting
------------------------------

.. code-block:: python

   import numpy as np
   import emcee
   from brutus.core import Isochrone, StellarPop
   from brutus.analysis.populations import isochrone_population_loglike

   # Initialize models
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Observed cluster data
   obs_flux = np.array([...])       # (N_stars, N_filters)
   obs_err = np.array([...])
   parallax = np.array([...])       # Optional
   parallax_err = np.array([...])

   def lnprior(theta):
       """Log-prior for population parameters."""
       feh, loga, av, rv, dist = theta
       if not (-2.0 < feh < 0.5): return -np.inf
       if not (6.0 < loga < 10.2): return -np.inf  # 1 Myr to 16 Gyr
       if not (0.0 < av < 5.0): return -np.inf
       if not (2.0 < rv < 6.0): return -np.inf
       if not (100.0 < dist < 10000.0): return -np.inf
       return 0.0

   def lnprob(theta):
       """Log-posterior for MCMC."""
       lp = lnprior(theta)
       if not np.isfinite(lp):
           return -np.inf
       lnl = isochrone_population_loglike(
           theta, pop, obs_flux, obs_err,
           parallax=parallax, parallax_err=parallax_err,
           cluster_prob=0.9, dim_prior=True,
       )
       return lp + lnl

   # Run MCMC
   ndim, nwalkers = 5, 32
   initial = np.array([0.0, 9.0, 0.3, 3.1, 2000.0])
   pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)

   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(pos, 5000, progress=True)

   # Extract results
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)
   feh_median = np.median(samples[:, 0])
   age_median = 10**np.median(samples[:, 1]) / 1e9  # Gyr
   dist_median = np.median(samples[:, 4])

   print(f"[Fe/H]: {feh_median:.2f}, Age: {age_median:.2f} Gyr, Distance: {dist_median:.0f} pc")

Advanced Topics
---------------

**Differential Extinction**: For clusters with cloud-to-cloud variation, treat :math:`A_V` as a per-star parameter rather than a population parameter.

**Empirical Corrections**: Apply calibration corrections during grid generation:

.. code-block:: python

   grid = generate_isochrone_population_grid(
       stellarpop=pop,
       feh=0.0, loga=9.0, av=0.5, rv=3.1, dist=2000.0,
       corr_params=corr_params  # From calibration
   )

**Non-Coeval Populations**: For star-forming regions with age spread, grid over multiple ages and marginalize.

**Performance**: Use coarser EEP/SMF grids for faster likelihood evaluation. Cache grids when varying only distance/extinction.

Summary
-------

brutus cluster modeling implements **mixture-before-marginalization**:

1. Generate (mass, SMF) grid for fixed population parameters
2. Apply mixture model (cluster + outlier) at each grid point
3. Marginalize over mass with proper jacobians
4. Sum log-likelihoods across stars

This avoids field contamination biases and properly propagates mass/binary uncertainties.

Next Steps
----------

- Photometric calibration: :doc:`photometric_offsets`
- Interpret results: :doc:`understanding_results`
- Configuration options: :doc:`choosing_options`

References
----------

- Speagle et al. (2025), arXiv:2503.02227 - brutus methods (§3)
- Hogg et al. (2010), arXiv:1008.4686 - Mixture model fundamentals
