Understanding and Interpreting Results
========================================

This page explains how to interpret brutus output, diagnose potential issues, and assess the reliability of stellar parameter estimates.

Output Structure
----------------

``BruteForce.fit()`` saves results to an **HDF5 file** and returns the file path.
Results are accessed by reading the HDF5 file directly.

Individual Star Fitting (``BruteForce.fit()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import h5py
   import numpy as np

   # fit() returns the path to the output file
   output_file = fitter.fit(data, data_err, data_mask, labels, save_file='results.h5', ...)

   # Read results from HDF5
   with h5py.File(output_file, 'r') as f:
       # Posterior draws (Nstars, Ndraws) - default Ndraws=250
       distances = f['samps_dist'][:]    # Distance in kpc
       av_values = f['samps_red'][:]     # A_V extinction (mag)
       rv_values = f['samps_dred'][:]    # R_V values
       log_weights = f['samps_logp'][:]  # Log-weights for each draw

       # Model information
       model_idx = f['model_idx'][:]     # Grid model indices (Nstars, Ndraws)
       ml_scale = f['ml_scale'][:]       # ML scale factors
       ml_av = f['ml_av'][:]             # ML A_V values
       ml_rv = f['ml_rv'][:]             # ML R_V values
       ml_cov = f['ml_cov_sar'][:]       # Covariance matrices (Nstars, Ndraws, 3, 3)

       # Per-object diagnostics
       log_evidence = f['obj_log_evid'][:]  # Log-evidence (Nstars,)
       chi2_min = f['obj_chi2min'][:]       # Minimum chi-squared (Nstars,)
       n_bands = f['obj_Nbands'][:]         # Number of bands used (Nstars,)
       log_post = f['obj_log_post'][:]      # Log-posteriors (Nstars, Ndraws)

       # Object labels
       obj_labels = f['labels'][:]          # Your input labels (Nstars, Nlabels)

   # Compute summary statistics yourself
   dist_median = np.median(distances, axis=1)
   dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)
   av_median = np.median(av_values, axis=1)

To access stellar parameters (mass, age, Teff, etc.), use the ``model_idx`` to look up
values in the model grid labels that were loaded with ``load_models()``.

Cluster Fitting (``isochrone_population_loglike()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cluster fitting uses MCMC (e.g., ``emcee``) with the likelihood function.
The sampler directly contains the population parameter chains:

.. code-block:: python

   # theta = [feh, loga, av, rv, dist] for isochrone_population_loglike
   samples = sampler.get_chain(discard=1000, thin=10, flat=True)

   feh_samples = samples[:, 0]
   loga_samples = samples[:, 1]  # log10(age in years)
   av_samples = samples[:, 2]
   rv_samples = samples[:, 3]
   dist_samples = samples[:, 4]

Posterior Distributions
-----------------------

The fundamental output from brutus is **posterior samples** representing the probability distribution over parameters given the data.

Visualizing Posteriors
^^^^^^^^^^^^^^^^^^^^^^^

**Histograms** show 1-D marginal distributions:

.. code-block:: python

   import h5py
   import matplotlib.pyplot as plt
   import numpy as np

   # Load results
   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0]  # First star, all draws (in kpc)

   # Compute summary statistics
   dist_median = np.median(distances)
   dist_16, dist_84 = np.percentile(distances, [16, 84])

   # Plot distance posterior (convert kpc to pc)
   plt.figure(figsize=(8, 5))
   plt.hist(distances * 1000, bins=50, density=True, alpha=0.7)
   plt.axvline(dist_median * 1000, color='r', linestyle='--',
               label=f"Median: {dist_median*1000:.0f} pc")
   plt.axvline(dist_16 * 1000, color='orange', linestyle=':')
   plt.axvline(dist_84 * 1000, color='orange', linestyle=':',
               label=f"16-84%: [{dist_16*1000:.0f}, {dist_84*1000:.0f}]")
   plt.xlabel('Distance (pc)')
   plt.ylabel('Probability Density')
   plt.legend()
   plt.show()

**Corner plots** for distance, extinction, and R_V:

.. code-block:: python

   import corner

   with h5py.File('results.h5', 'r') as f:
       dist = f['samps_dist'][0] * 1000  # Convert to pc
       av = f['samps_red'][0]
       rv = f['samps_dred'][0]

   samples = np.column_stack([dist, av, rv])
   labels = ['Distance (pc)', r'$A_V$ (mag)', r'$R_V$']

   fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                       show_titles=True)
   plt.show()

Interpreting Posterior Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Gaussian-like (symmetric)**:
   Well-constrained parameter with good data. Example: Distance for bright star with accurate parallax.

   .. code-block:: none

      |     ****
      |    *    *
      |  **      **
      |**          **
      +---------------
        d_median

**Skewed (asymmetric)**:
   Parameter hitting a boundary or degeneracy. Example: Extinction near A_V = 0.

   .. code-block:: none

      |**
      | ****
      |    ***
      |      ****
      +------------
       0    A_V

**Bimodal (multiple peaks)**:
   Degeneracy between solutions. Example: Faint red star could be nearby M dwarf or distant K giant.

   .. code-block:: none

      |  **      **
      | *  *    *  *
      |**  **  **  **
      +--------------
       d1    d2

   **Action**: Check if parallax helps resolve degeneracy. If not, data may be insufficient.

**Flat/uniform**:
   Parameter unconstrained by data. Example: R_V when extinction is negligible.

   .. code-block:: none

      |************
      |************
      |************
      +-----------
         R_V

   **Action**: This is expected when parameter doesn't affect the data. Not a problem.

Common Degeneracies
-------------------

Distance-Extinction Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: A faint reddened star can look similar to a nearby intrinsically red star.

**Symptoms**:
   - Strong correlation between ``dist_samples`` and ``av_samples`` in corner plot
   - Elongated posterior contours along distance-extinction diagonal

**Solutions**:
   - **Parallax**: Breaks degeneracy by independently constraining distance
   - **Multi-band photometry**: Different wavelength dependence helps separate intrinsic color from reddening
   - **Dust priors**: 3-D dust maps constrain expected extinction at different distances

**Example diagnostic**:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0]  # First star
       av_values = f['samps_red'][0]

   correlation = np.corrcoef(distances, av_values)[0, 1]
   print(f"Distance-extinction correlation: {correlation:.2f}")

   # Strong positive correlation (r > 0.7) indicates degeneracy
   if abs(correlation) > 0.7:
       print("WARNING: Strong distance-extinction degeneracy detected")
       print("Consider adding parallax or improving photometric coverage")

Mass-Age-Metallicity Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Different combinations of (mass, age, metallicity) can produce similar temperatures and luminosities.

**Symptoms**:
   - Broad, multi-modal posteriors for ``mass_samples``, ``age_samples``, ``feh_samples``
   - Multiple solutions in parameter space

**Solutions**:
   - **Asteroseismology**: Directly constrains surface gravity and mass
   - **Spectroscopy**: Breaks metallicity degeneracy
   - **Galactic priors**: Age and metallicity priors from Galactic position help
   - **Multi-epoch data**: Variability or astrometry can constrain mass

**Example**: An old, metal-poor, massive star can look like a young, metal-rich, low-mass star.

Binary Degeneracy
^^^^^^^^^^^^^^^^^

**Problem**: Unresolved binary companions add flux, mimicking a brighter single star.

**Symptoms**:
   - Inferred parameters inconsistent with spectroscopic measurements
   - Residuals suggesting excess flux in some bands

**Solutions**:
   - **Check for binarity**: Radial velocity variations, eclipses, imaging
   - **Model binaries explicitly**: Use SMF parameter in cluster fitting
   - **Be cautious**: Binary contamination affects ~50% of stars

Diagnostic Checks
-----------------

χ² and Goodness-of-Fit
^^^^^^^^^^^^^^^^^^^^^^^

Check the quality of the best-fit model:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       chi2_min = f['obj_chi2min'][0]  # Minimum chi-squared for first star
       n_bands = f['obj_Nbands'][0]    # Number of bands used

   # Compute reduced chi-square
   n_params = 3  # Distance, A_V, R_V (model parameters are fixed from grid)
   dof = n_bands - n_params

   chi2_reduced = chi2_min / dof

   print(f"Reduced χ²: {chi2_reduced:.2f}")

   if chi2_reduced < 0.5:
       print("WARNING: χ² too low - may indicate overestimated errors")
   elif chi2_reduced > 3.0:
       print("WARNING: χ² too high - poor fit or underestimated errors")
   else:
       print("Good fit quality")

Parallax Consistency
^^^^^^^^^^^^^^^^^^^^

If parallax was used, check consistency between photometric and parallax-based distance:

.. code-block:: python

   import h5py
   import numpy as np

   # Parallax-implied distance
   parallax = 2.5  # mas (your measured parallax)
   parallax_err = 0.1  # mas
   parallax_dist = 1000.0 / parallax  # pc
   parallax_dist_err = 1000.0 * parallax_err / parallax**2

   # Brutus distance estimate
   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0] * 1000  # Convert kpc to pc

   brutus_dist = np.median(distances)
   dist_16, dist_84 = np.percentile(distances, [16, 84])
   brutus_dist_err = (dist_84 - dist_16) / 2.0

   # Consistency check
   diff = abs(brutus_dist - parallax_dist)
   combined_err = np.sqrt(brutus_dist_err**2 + parallax_dist_err**2)

   print(f"Parallax distance: {parallax_dist:.1f} ± {parallax_dist_err:.1f} pc")
   print(f"Brutus distance: {brutus_dist:.1f} ± {brutus_dist_err:.1f} pc")
   print(f"Difference: {diff:.1f} pc ({diff/combined_err:.1f} sigma)")

   if diff / combined_err > 2.0:
       print("WARNING: Parallax and photometric distances inconsistent")

Prior Sensitivity
^^^^^^^^^^^^^^^^^

Test how results change with different prior settings by passing different
``lngalprior`` and ``lndustprior`` functions to ``fit()``:

.. code-block:: python

   import h5py
   import numpy as np
   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Fit with default priors (Galactic structure + dust map)
   fitter.fit(data, data_err, data_mask, labels, save_file='with_priors.h5',
              parallax=plx, parallax_err=plx_err,
              data_coords=coords, dustfile='bayestar19.h5')

   # Fit with uniform priors (pass None to disable)
   fitter.fit(data, data_err, data_mask, labels, save_file='no_priors.h5',
              parallax=plx, parallax_err=plx_err,
              lngalprior=lambda *args: 0.0,  # Uniform Galactic prior
              lndustprior=lambda *args: 0.0) # Uniform dust prior

   # Compare results
   with h5py.File('with_priors.h5', 'r') as f:
       dist_with = np.median(f['samps_dist'][0]) * 1000  # pc
   with h5py.File('no_priors.h5', 'r') as f:
       dist_without = np.median(f['samps_dist'][0]) * 1000  # pc

   fractional_change = abs(dist_with - dist_without) / dist_with
   if fractional_change > 0.3:
       print("WARNING: Results strongly prior-dependent (>30% change)")

Reliability Indicators
----------------------

When to Trust Results
^^^^^^^^^^^^^^^^^^^^^

✅ **High confidence**:
   - χ²_reduced ~ 1
   - Narrow, Gaussian-like posteriors
   - Parallax and photometric distances agree (if parallax available)
   - Residuals < 0.1 mag across all bands
   - Results stable with/without priors

✅ **Moderate confidence**:
   - χ²_reduced between 0.5 and 2
   - Asymmetric but unimodal posteriors
   - Some degeneracies but broken by parallax or priors
   - Residuals < 0.2 mag

⚠ **Low confidence**:
   - χ²_reduced > 3 or < 0.3
   - Bimodal or very broad posteriors
   - Strong parameter correlations unbroken by data
   - Large residuals (> 0.3 mag) in multiple bands
   - Results change dramatically without priors

❌ **Unreliable**:
   - Optimization failed to converge
   - Posteriors hit parameter boundaries
   - Parallax and photometric distances disagree by > 3σ
   - Systematic residual patterns (e.g., all blue bands too bright)

Uncertainty Quantification
---------------------------

Credible Intervals
^^^^^^^^^^^^^^^^^^

brutus provides Bayesian **credible intervals** (not frequentist confidence intervals):

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0] * 1000  # First star, convert kpc to pc

   # 68% credible interval (analogous to 1-sigma)
   dist_median = np.median(distances)
   dist_16, dist_84 = np.percentile(distances, [16, 84])
   print(f"Distance: {dist_median:.0f} (+{dist_84-dist_median:.0f} / -{dist_median-dist_16:.0f}) pc")

   # 95% credible interval
   dist_025, dist_975 = np.percentile(distances, [2.5, 97.5])
   print(f"95% interval: [{dist_025:.0f}, {dist_975:.0f}] pc")

**Interpretation**: "There is a 68% probability that the true distance lies in [dist_16, dist_84] given the data and priors."

Systematic vs Statistical Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus uncertainties are primarily **statistical** (measurement errors + parameter degeneracies). They do **not** include:

- Model systematics (errors in stellar evolution models)
- Photometric zero-point uncertainties
- Extinction curve uncertainties
- Prior misspecification

**Recommended practice**: Add systematic error floor (~10% for distances, ~0.05 mag for extinction) in quadrature:

.. code-block:: python

   # Statistical uncertainty from posterior samples
   dist_err_stat = (dist_84 - dist_16) / 2.0

   # Add 10% systematic floor
   dist_err_sys = 0.10 * dist_median

   # Total uncertainty
   dist_err_total = np.sqrt(dist_err_stat**2 + dist_err_sys**2)

   print(f"Distance: {dist_median:.0f} ± {dist_err_total:.0f} pc")

Derived Quantities
------------------

Use the distance and extinction samples to propagate uncertainties to derived quantities.

Absolute Magnitude
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0]  # kpc
       av_values = f['samps_red'][0]

   # Absolute magnitude from distance and apparent magnitude
   app_mag_g = 16.5  # Observed g-band magnitude
   R_g = 3.518  # Extinction coefficient for g-band (approximate)

   dist_modulus = 5.0 * np.log10(distances * 1000) - 5.0  # Convert kpc to pc
   abs_mag_g = app_mag_g - dist_modulus - av_values * R_g

   print(f"M_g = {np.median(abs_mag_g):.2f} ± {np.std(abs_mag_g):.2f} mag")

Stellar Parameters from Model Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get stellar parameters (mass, age, Teff, luminosity, etc.), use the model
indices to look up values in the grid:

.. code-block:: python

   from brutus.data import load_models

   # Load the same grid used for fitting
   models, grid_labels, mask = load_models('grid_mist_v9.h5')

   with h5py.File('results.h5', 'r') as f:
       model_idx = f['model_idx'][0]  # Model indices for first star

   # Get stellar parameters for each posterior draw
   masses = grid_labels['mini'][model_idx]
   log_ages = grid_labels['loga'][model_idx]
   log_teffs = grid_labels['logt'][model_idx]

   print(f"Mass: {np.median(masses):.2f} Msun")
   print(f"Age: {10**np.median(log_ages)/1e9:.2f} Gyr")
   print(f"Teff: {10**np.median(log_teffs):.0f} K")

Galactic Coordinates
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from astropy.coordinates import SkyCoord
   import astropy.units as u

   with h5py.File('results.h5', 'r') as f:
       distances = f['samps_dist'][0]  # kpc

   # Create 3D coordinates for each posterior sample
   coords_3d = SkyCoord(
       ra=ra*u.deg, dec=dec*u.deg,
       distance=distances*u.kpc,
       frame='icrs'
   )

   # Transform to Galactocentric coordinates
   coords_gal = coords_3d.galactocentric

   X_samples = coords_gal.x.to(u.kpc).value
   Y_samples = coords_gal.y.to(u.kpc).value
   Z_samples = coords_gal.z.to(u.kpc).value

   print(f"Galactic X: {np.median(X_samples):.2f} ± {np.std(X_samples):.2f} kpc")
   print(f"Galactic Z: {np.median(Z_samples):.2f} ± {np.std(Z_samples):.2f} kpc")

Troubleshooting Common Issues
------------------------------

"Optimization did not converge"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Gradient-based optimizer failed to find minimum in flux space.

**Solutions**:
   - Check for bad photometry (negative fluxes, very large errors)
   - Increase distance or extinction bounds
   - Try different starting values
   - If persistent, data may be incompatible with models

"Distance hits lower bound"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Best-fit distance is at minimum allowed value (typically 10 pc).

**Solutions**:
   - Check parallax—is star truly very nearby?
   - Inspect residuals—may indicate very bright intrinsic source
   - Consider exotic objects (white dwarfs, brown dwarfs) outside model grid

"Extinction posterior is flat"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Data insufficient to constrain reddening.

**This is often OK**: For bright, blue stars with minimal extinction, A_V is genuinely unconstrained. Not a problem unless you need precise reddening.

**Solutions if you need A_V**:
   - Add redder photometric bands (near-IR helps constrain reddening)
   - Use dust map priors more aggressively
   - Check for instrumental systematics

"All parameters have huge uncertainties"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Data quality poor or star outside model coverage.

**Solutions**:
   - Check photometric S/N—are errors realistic?
   - Verify star is within grid boundaries (mass, metallicity, age)
   - Add parallax if available
   - Consider if object is non-stellar (galaxy, QSO) or exotic (WD, CV)

Summary
-------

Interpreting brutus results requires:

✓ Visualizing posterior distributions (histograms, corner plots)
✓ Checking goodness-of-fit (χ², residuals)
✓ Assessing degeneracies (correlations between parameters)
✓ Testing prior sensitivity
✓ Validating against independent measurements (parallax, spectroscopy)

Results are most reliable when posteriors are unimodal, χ²~1, and independent checks agree.

Next Steps
----------

- Configure fitting options: :doc:`choosing_options`
- Review common questions: :doc:`faq`
- Learn about priors: :doc:`priors`

References
----------

Bayesian Inference and Uncertainty Quantification:

- Hogg & Foreman-Mackey (2018), "Data Analysis Recipes: Using Markov Chain Monte Carlo", ApJS, 236, 11
- Gelman et al. (2013), "Bayesian Data Analysis" (3rd ed.), CRC Press

Stellar Parameter Degeneracies:

- Jørgensen & Lindegren (2005), "Systemic Biases in Star Formation History Studies", A&A, 436, 127
- Bovy (2016), "The Stellar Spectroscopic Surveys In The Gaia Era", in Astrophysical Applications of Gravitational Lensing, IAU Symposium 319

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227
