Understanding and Interpreting Results
========================================

This page explains how to interpret brutus output and assess reliability.

Output Structure
----------------

``BruteForce.fit()`` saves results to an **HDF5 file** and returns the file path:

.. code-block:: python

   import h5py
   import numpy as np

   output_file = fitter.fit(data, data_err, data_mask, labels, save_file='results.h5', ...)

   with h5py.File(output_file, 'r') as f:
       # Posterior draws (Nstars, Ndraws) - default Ndraws=250
       distances = f['samps_dist'][:]    # Distance in kpc
       av_values = f['samps_red'][:]     # A_V extinction (mag)
       rv_values = f['samps_dred'][:]    # R_V values
       log_weights = f['samps_logp'][:]  # Log-weights

       # Model information
       model_idx = f['model_idx'][:]     # Grid model indices (Nstars, Ndraws)
       ml_cov = f['ml_cov_sar'][:]       # Covariance matrices (Nstars, Ndraws, 3, 3)

       # Diagnostics
       log_evidence = f['obj_log_evid'][:]  # Log-evidence (Nstars,)
       chi2_min = f['obj_chi2min'][:]       # Minimum chi-squared (Nstars,)
       n_bands = f['obj_Nbands'][:]         # Number of bands used (Nstars,)

   # Compute summary statistics
   dist_median = np.median(distances, axis=1)
   dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)

To access stellar parameters (mass, age, Teff), use ``model_idx`` to look up values in the model grid labels.

Cluster Fitting Output
^^^^^^^^^^^^^^^^^^^^^^

Cluster fitting uses MCMC with ``isochrone_population_loglike()``. Results are in the sampler:

.. code-block:: python

   # theta = [feh, loga, av, rv, dist]
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)
   feh_samples, loga_samples = samples[:, 0], samples[:, 1]

Visualizing Posteriors
----------------------

.. code-block:: python

   import corner
   import h5py

   with h5py.File('results.h5', 'r') as f:
       dist = f['samps_dist'][0] * 1000  # pc
       av = f['samps_red'][0]
       rv = f['samps_dred'][0]

   samples = np.column_stack([dist, av, rv])
   fig = corner.corner(samples, labels=['Distance (pc)', r'$A_V$', r'$R_V$'],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)

Interpreting Posterior Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Gaussian-like**: Well-constrained parameter
- **Skewed**: Parameter near boundary (e.g., A_V near 0)
- **Bimodal**: Degeneracy between solutions (e.g., dwarf vs giant)
- **Flat**: Parameter unconstrained by data (expected for R_V when A_V~0)

Common Degeneracies
-------------------

**Distance-Extinction**: A faint reddened star looks like a nearby red star. Check correlation:

.. code-block:: python

   with h5py.File('results.h5', 'r') as f:
       dist, av = f['samps_dist'][0], f['samps_red'][0]
   corr = np.corrcoef(dist, av)[0, 1]
   if abs(corr) > 0.7:
       print("Strong distance-extinction degeneracy")

**Solutions**: Add parallax, use multi-band photometry (esp. near-IR), enable dust priors.

**Mass-Age-Metallicity**: Different (M, age, [Fe/H]) can produce similar Teff/L. **Solutions**: Add spectroscopy or asteroseismology; Galactic priors help.

**Binaries**: Unresolved companions add flux, biasing parameters. Affects ~50% of stars.

Diagnostic Checks
-----------------

χ² Goodness-of-Fit
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   with h5py.File('results.h5', 'r') as f:
       chi2_min = f['obj_chi2min'][0]
       n_bands = f['obj_Nbands'][0]

   chi2_reduced = chi2_min / (n_bands - 3)  # 3 free params: dist, A_V, R_V

**Interpreting χ² values:**

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - χ² (reduced)
     - Assessment
     - Likely Cause / Action
   * - < 0.5
     - Errors overestimated
     - Photometric uncertainties may be too large; consider recalibrating
   * - 0.8 - 1.2
     - **Good fit**
     - Model matches data within uncertainties
   * - 1.2 - 2.0
     - Acceptable
     - Minor tension; check individual band residuals
   * - 2.0 - 3.0
     - Marginal
     - Investigate: possible binary, variability, or calibration issue
   * - > 3.0
     - Poor fit
     - Check data quality, verify star is within model coverage, or consider unmodeled physics (binary, rotation)

Parallax Consistency
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   parallax = 2.5  # mas
   parallax_dist = 1000.0 / parallax  # pc

   with h5py.File('results.h5', 'r') as f:
       brutus_dist = np.median(f['samps_dist'][0]) * 1000  # pc

   # Compare - should agree within uncertainties

Prior Sensitivity
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Fit with and without priors
   fitter.fit(..., save_file='with.h5', data_coords=coords)
   fitter.fit(..., save_file='without.h5', lngalprior=lambda *args: 0.0)

   # >30% change indicates prior-dominated results

Reliability Indicators
----------------------

✅ **High confidence**: χ²~1, unimodal posteriors, parallax agrees, residuals < 0.1 mag

⚠ **Caution**: χ² > 2, asymmetric posteriors, some degeneracies

❌ **Unreliable**: Bimodal posteriors, parallax disagrees >3σ, residuals > 0.3 mag

Uncertainty Quantification
--------------------------

brutus provides **Bayesian credible intervals**:

.. code-block:: python

   with h5py.File('results.h5', 'r') as f:
       dist = f['samps_dist'][0] * 1000  # pc

   dist_median = np.median(dist)
   dist_16, dist_84 = np.percentile(dist, [16, 84])
   print(f"Distance: {dist_median:.0f} (+{dist_84-dist_median:.0f} / -{dist_median-dist_16:.0f}) pc")

**Note**: Uncertainties are statistical only. Consider adding ~10% systematic floor for distances.

Derived Quantities
------------------

Stellar Parameters from Model Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from brutus.data import load_models

   models, grid_labels, mask = load_models('grid_mist_v9.h5')

   with h5py.File('results.h5', 'r') as f:
       model_idx = f['model_idx'][0]

   masses = grid_labels['mini'][model_idx]
   log_ages = grid_labels['loga'][model_idx]
   log_teffs = grid_labels['logt'][model_idx]

   print(f"Mass: {np.median(masses):.2f} Msun")
   print(f"Age: {10**np.median(log_ages)/1e9:.2f} Gyr")
   print(f"Teff: {10**np.median(log_teffs):.0f} K")

Absolute Magnitude
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   with h5py.File('results.h5', 'r') as f:
       dist = f['samps_dist'][0]  # kpc
       av = f['samps_red'][0]

   app_mag = 16.5  # Observed magnitude
   R_g = 3.518     # Extinction coefficient
   dist_mod = 5.0 * np.log10(dist * 1000) - 5.0
   abs_mag = app_mag - dist_mod - av * R_g

Troubleshooting
---------------

**Large uncertainties**: Check S/N, verify star is within grid coverage, add parallax

**Flat extinction posterior**: Normal for low-extinction stars; add near-IR bands if A_V needed

**Distance at boundary**: Check parallax, consider exotic objects outside grid

**χ² >> 1**: Bad photometry, underestimated errors, or object outside model coverage

Summary
-------

Key diagnostics: χ² (should be ~1), posterior shapes (should be unimodal), parameter correlations (watch for degeneracies), prior sensitivity (<30% change), parallax consistency.

Next Steps
----------

- Configure options: :doc:`choosing_options`
- Learn about priors: :doc:`priors`
- See FAQ: :doc:`faq`

References
----------

- Speagle et al. (2025), arXiv:2503.02227
- Hogg & Foreman-Mackey (2018), ApJS, 236, 11 - MCMC methods
