Understanding Results
=====================

This page explains how to work with brutus output files.

Output Structure
----------------

``BruteForce.fit()`` saves results to an HDF5 file:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       # Posterior draws - shape (Nstars, Ndraws), default Ndraws=250
       distances = f['samps_dist'][:]    # Distance in kpc
       av_values = f['samps_red'][:]     # A_V extinction (mag)
       rv_values = f['samps_dred'][:]    # R_V values
       log_posts = f['samps_logp'][:]    # Log-posterior values at the draws,
                                         # for diagnostics only -- the draws
                                         # are already resampled, so treat
                                         # them as equally weighted (do NOT
                                         # reweight by samps_logp)

       # Model indices into the grid
       model_idx = f['model_idx'][:]     # Shape (Nstars, Ndraws)

       # Per-object diagnostics
       chi2_min = f['obj_chi2min'][:]    # Best-fit chi-squared (Nstars,)
       n_bands = f['obj_Nbands'][:]      # Number of bands used (Nstars,)

Summary Statistics
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Posterior summaries for first star
   dist = distances[0]  # kpc
   print(f"Distance: {np.median(dist):.2f} kpc")
   print(f"68% interval: [{np.percentile(dist, 16):.2f}, {np.percentile(dist, 84):.2f}] kpc")

Stellar Parameters
^^^^^^^^^^^^^^^^^^

Stellar parameters are accessed via the model grid using ``model_idx``:

.. code-block:: python

   from brutus.data import load_models

   models, grid_labels, label_mask = load_models('grid_mist_v9.h5')

   with h5py.File('results.h5', 'r') as f:
       idx = f['model_idx'][0]  # First star

   # Available labels: mini, feh, eep, loga, logl, logt, logg, agewt
   masses = grid_labels['mini'][idx]      # Initial mass (Msun)
   fehs = grid_labels['feh'][idx]         # [Fe/H]
   log_ages = grid_labels['loga'][idx]    # log10(age/yr)
   log_teffs = grid_labels['logt'][idx]   # log10(Teff/K)
   log_Ls = grid_labels['logl'][idx]      # log10(L/Lsun)

   print(f"Mass: {np.median(masses):.2f} Msun")
   print(f"Teff: {10**np.median(log_teffs):.0f} K")

Cluster Fitting
^^^^^^^^^^^^^^^

Results from MCMC with ``isochrone_population_loglike()`` are stored in the sampler:

.. code-block:: python

   # theta = [feh, loga, av, rv, dist, field_frac]
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

Visualization
-------------

brutus provides plotting utilities in :mod:`brutus.plotting`:

- :func:`~brutus.plotting.cornerplot` - Corner plots from fitting outputs
- :func:`~brutus.plotting.dist_vs_red` - Distance vs extinction posterior
- :func:`~brutus.plotting.posterior_predictive` - Model vs data SED comparison

Example using the SED posterior predictive plot:

.. code-block:: python

   from brutus.plotting import posterior_predictive
   from brutus.data import load_models

   models, labels, label_mask = load_models('grid_mist_v9.h5')

   with h5py.File('results.h5', 'r') as f:
       idxs = f['model_idx'][0]
       reds = f['samps_red'][0]
       dreds = f['samps_dred'][0]
       dists = f['samps_dist'][0]

   fig = posterior_predictive(
       models, idxs, reds, dreds, dists,
       data=observed_flux, data_err=flux_err, data_mask=mask
   )

For quick visualization with external libraries:

.. code-block:: python

   import corner

   with h5py.File('results.h5', 'r') as f:
       dist = f['samps_dist'][0]
       av = f['samps_red'][0]

   fig = corner.corner(
       np.column_stack([dist, av]),
       labels=['Distance (kpc)', r'$A_V$']
   )

Diagnostics
-----------

Goodness of Fit
^^^^^^^^^^^^^^^

Use a **p-value** to assess fit quality, not ``chi2/Nbands``. The raw ratio is not
variance-stabilizing across different numbers of bands and conflates the degrees of
freedom with the band count. Instead, compare ``obj_chi2min`` to the chi-squared
distribution with the proper degrees of freedom.

The fit has 3 free parameters (scale, A_V, R_V), so the degrees of freedom are
``dof = obj_Nbands - 3``. When a parallax is provided, ``obj_chi2min`` includes the
parallax contribution and ``obj_Nbands`` counts the parallax as an extra band (it adds
1 band and 0 free parameters), so ``dof = obj_Nbands - 3`` in both cases.

.. code-block:: python

   from scipy import stats

   with h5py.File('results.h5', 'r') as f:
       chi2min = f['obj_chi2min'][:]    # (Nstars,)
       n_bands = f['obj_Nbands'][:]     # (Nstars,)

   dof = np.maximum(n_bands - 3, 1)
   pvalue = 1.0 - stats.chi2.cdf(chi2min, dof)

   # Classify fits
   good = pvalue > 0.001
   marginal = (pvalue > 1e-6) & (pvalue <= 0.001)
   poor = pvalue <= 1e-6

Thresholds:

- ``p > 0.001``: Good fit
- ``1e-6 < p <= 0.001``: Marginal
- ``p <= 1e-6``: Poor fit (model inadequacy)

Poor fits typically arise from:

- Underestimated photometric errors
- Bad photometry (outliers, saturation, blending)
- Object outside model coverage (white dwarf, brown dwarf, unresolved binary, YSO)

.. warning::
   The p-value depends on your error estimates being accurate. Underestimated errors
   inflate ``obj_chi2min`` and drive the p-value toward zero even for a correct model.

Parallax Consistency
^^^^^^^^^^^^^^^^^^^^

If parallax was provided, check that the inferred distance is consistent:

.. code-block:: python

   parallax_mas = 2.5
   parallax_dist_kpc = 1.0 / parallax_mas

   with h5py.File('results.h5', 'r') as f:
       phot_dist_kpc = np.median(f['samps_dist'][0])

   print(f"Parallax implies: {parallax_dist_kpc:.2f} kpc")
   print(f"Photometry gives: {phot_dist_kpc:.2f} kpc")

Significant disagreement may indicate binary contamination, parallax issues (check Gaia RUWE), or model mismatch.

Prior Sensitivity
^^^^^^^^^^^^^^^^^

To check if results are prior-dominated, compare fits with and without priors:

.. code-block:: python

   # With Galactic prior
   fitter.fit(..., save_file='with_prior.h5', data_coords=coords)

   # Without Galactic prior. A custom prior receives an array of distances
   # (kpc) and must return an array of the same shape.
   fitter.fit(..., save_file='no_prior.h5',
              lngalprior=lambda dist, coord, **kw: np.zeros_like(dist))

Large differences (>30%) indicate the prior strongly influences results, which is expected for faint stars with poor photometry.

See Also
--------

- :mod:`brutus.plotting` - Visualization utilities
