Analysis Module (``brutus.analysis``)
======================================

The analysis module implements the statistical inference machinery for brutus. It provides high-level fitting functions that combine stellar models with Bayesian inference to estimate parameters from photometric and astrometric data.

**Key Components:**

- **Individual Star Fitting**: ``BruteForce`` class performs grid-based Bayesian inference for field stars
- **Population Analysis**: Functions for fitting coeval stellar populations (clusters) with mixture models
- **Photometric Offsets**: Tools for empirical calibration of systematic photometric errors
- **Line-of-Sight Dust**: Functions for 3-D dust mapping from stellar ensembles

**Design Philosophy:**

The module implements the complete Bayesian inference workflow:

1. **Model Evaluation**: Compute likelihoods over stellar model grids
2. **Prior Application**: Incorporate Galactic priors on stellar properties and dust
3. **Optimization**: Find best-fit parameters in distance-extinction space
4. **Marginalization**: Integrate over nuisance parameters
5. **Sampling**: Generate posterior samples for uncertainty quantification

**Typical Usage Patterns:**

For **individual field stars** with photometry and parallax:

.. code-block:: python

   from brutus.analysis import BruteForce
   from brutus.core import StarGrid
   from brutus.data import load_models
   import numpy as np
   import h5py

   # Load pre-computed grid
   models, labels, label_mask = load_models('grid_file.h5')
   grid = StarGrid(models, labels, label_mask)

   # Initialize fitter
   fitter = BruteForce(grid)

   # Observed data (Nstars=1 in this example)
   flux = np.array([[16.5, 15.2, 14.8, 13.5, 13.1]])  # shape (1, 5)
   flux_err = np.array([[0.01, 0.01, 0.02, 0.03, 0.03]])
   mask = np.array([[True, True, True, True, True]])
   obj_ids = np.array([[1]])
   parallax = np.array([2.5])  # mas
   parallax_err = np.array([0.1])  # mas

   # Fit and save results to HDF5
   output_file = fitter.fit(
       data=flux, data_err=flux_err, data_mask=mask,
       data_labels=obj_ids, save_file='results.h5',
       parallax=parallax, parallax_err=parallax_err,
       Ndraws=250
   )

   # Read posterior samples from HDF5
   with h5py.File(output_file, 'r') as f:
       dist = f['samps_dist'][0]  # (Ndraws,) distances in kpc
       av = f['samps_red'][0]     # (Ndraws,) A_V values
   print(f"Distance: {np.median(dist)*1000:.0f} pc")
   print(f"Extinction: {np.median(av):.2f} mag")

For **stellar clusters** with MCMC:

.. code-block:: python

   from brutus.analysis import isochrone_population_loglike
   from brutus.core import Isochrone, StellarPop
   import emcee
   import numpy as np

   # Initialize population model
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Observed cluster data (N stars, M filters)
   obs_flux = np.array([...])  # shape (N, M)
   obs_err = np.array([...])   # shape (N, M)

   # Define log-likelihood for MCMC
   # theta = [feh, loga, av, rv, dist]
   def lnprob(theta):
       return isochrone_population_loglike(
           theta,
           stellarpop=pop,
           obs_phot=obs_flux,
           obs_err=obs_err,
           cluster_prob=0.9  # External membership prior
       )

   # Run MCMC
   ndim, nwalkers = 5, 32
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   sampler.run_mcmc(initial_pos, 5000, progress=True)

   # Extract posterior samples
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

**Key Methodological Features:**

- **Brute-force grid evaluation**: Systematically explores parameter space without convergence issues
- **Multi-stage optimization**: Fast magnitude-space → accurate flux-space → Bayesian posterior
- **Mixture-before-marginalization**: Mathematically correct treatment of field contamination in clusters
- **Importance sampling**: Efficient posterior sampling weighted by probability

**See Also:**

- :doc:`/grid_generation` - Understanding the brute-force fitting algorithm
- :doc:`/cluster_modeling` - Detailed guide to cluster population fitting
- :doc:`/photometric_offsets` - Empirical calibration procedures
- :doc:`/understanding_results` - Interpreting posterior distributions

.. currentmodule:: brutus.analysis

Individual Star Fitting
------------------------

.. autoclass:: BruteForce
   :members:
   :undoc-members:
   :show-inheritance:

Population Analysis
-------------------

.. autofunction:: isochrone_population_loglike

.. autofunction:: generate_isochrone_population_grid

.. autofunction:: compute_isochrone_cluster_loglike

.. autofunction:: compute_isochrone_outlier_loglike

.. autofunction:: apply_isochrone_mixture_model

.. autofunction:: marginalize_isochrone_grid

Photometric Offsets
-------------------

.. autoclass:: PhotometricOffsetsConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: photometric_offsets

Line-of-Sight Dust
-------------------

.. autofunction:: los_clouds_priortransform

.. autofunction:: los_clouds_loglike_samples

.. autofunction:: kernel_tophat

.. autofunction:: kernel_gauss

.. autofunction:: kernel_lorentz
