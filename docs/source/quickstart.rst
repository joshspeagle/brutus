Quick Start Guide
=================

This guide walks you through the core brutus workflows.

.. important::

   **Before you begin**: Ensure you have downloaded the required data files.
   See :doc:`installation` for complete setup instructions.

.. admonition:: Supported Photometric Filters
   :class: tip

   brutus supports many common photometric systems out of the box:

   - **Gaia**: ``Gaia_G_MAW``, ``Gaia_BP_MAWf``, ``Gaia_RP_MAW``
   - **Pan-STARRS**: ``PS_g``, ``PS_r``, ``PS_i``, ``PS_z``, ``PS_y``
   - **2MASS**: ``2MASS_J``, ``2MASS_H``, ``2MASS_Ks``
   - **WISE**: ``WISE_W1``, ``WISE_W2``

   See :ref:`available-filters` for the complete list of supported filters
   (SDSS, DECam, VISTA, UKIDSS, Tycho, Hipparcos, Kepler, TESS, Johnson-Cousins, etc.)
   and recommended combinations.

----

Generating Synthetic Photometry
-------------------------------

brutus can generate synthetic spectral energy distributions (SEDs) for
individual stars or stellar populations. This is useful for creating mock
catalogs, understanding model predictions, and forward-modeling observations.

Using Stellar Evolution Tracks (Individual Stars)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``EEPTracks`` and ``StarEvolTrack`` to generate photometry for individual
stars at specific points along their evolutionary tracks.

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   # Load stellar evolutionary tracks
   tracks = EEPTracks()

   # Create photometry generator with desired filters
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   star = StarEvolTrack(tracks=tracks, filters=filters)

   # Generate SED for a star with specific parameters
   sed, params, params2 = star.get_seds(
       mini=1.0,       # Initial mass in solar masses
       eep=350,        # Equivalent evolutionary point (main sequence ~300-450)
       feh=0.0,        # Metallicity [Fe/H]
       av=0.5,         # V-band extinction in magnitudes
       rv=3.3,         # Extinction law R_V
       dist=1000.0     # Distance in parsecs
   )

   # get_seds returns a 3-tuple:
   # - sed:     synthetic SED in magnitudes (one value per filter)
   # - params:  primary component stellar parameters (dict by default)
   # - params2: secondary component parameters (for binaries; dict by default)

**What is EEP?** The Equivalent Evolutionary Point is an index along a stellar
evolution track. Key values: ~200 (pre-main sequence), ~300-450 (main sequence),
~450-600 (subgiant/red giant). See :doc:`glossary` for details.

Using Isochrones (Stellar Populations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``Isochrone`` and ``StellarPop`` to generate photometry for an entire
stellar population at a fixed age and metallicity.

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Load isochrones and create population generator
   filters = ['Gaia_G_MAW', 'PS_g', 'PS_r', 'PS_i', '2MASS_J']
   iso = Isochrone()
   pop = StellarPop(isochrone=iso, filters=filters)

   # Generate photometry for a 1 Gyr, solar-metallicity population
   seds, params, params2 = pop.get_seds(
       feh=0.0,        # Metallicity [Fe/H]
       loga=9.0,       # log10(age/yr) = 9.0 corresponds to 1 Gyr
       av=0.5,         # V-band extinction in magnitudes
       rv=3.3,         # Extinction law R_V
       dist=2000.0     # Distance in parsecs
   )

   # get_seds returns a 3-tuple:
   # - seds:    (N_stars, N_filters) array of SEDs in magnitudes
   # - params:  (N_stars,) stellar parameters (mini, eep, feh, etc.)
   # - params2: (N_stars,) secondary-component parameters (for binaries)

   print(f"Generated {len(seds)} stars along the isochrone")

----

Fitting Individual Stars
------------------------

The ``BruteForce`` fitter derives stellar parameters (distance, extinction,
mass, age) by evaluating observed photometry against a pre-computed grid of
stellar models.

Loading Models and Preparing Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from brutus.data import load_models, load_offsets
   from brutus.core import StarGrid

   # Define filters matching your photometry
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',
              '2MASS_J', '2MASS_H', '2MASS_Ks']

   # Load pre-computed model grid
   # The third return value is a mask of available ancillary labels;
   # it is not a StarGrid argument.
   models, labels, label_mask = load_models(
       'grid_mist_v9.h5',
       filters=filters
   )
   grid = StarGrid(models, labels)

   # Load photometric calibration offsets (recommended)
   offsets = load_offsets('offsets_mist_v9.txt', filters=filters)

Using Dust Map Priors (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Bayestar`` class provides access to 3D dust maps that can constrain
extinction priors based on Galactic coordinates and distance:

.. code-block:: python

   from brutus.dust import Bayestar
   from brutus.data import fetch_dustmaps

   # Download dust map data (first time only, ~2 GB)
   # Returns path to the downloaded file
   dustmap_path = fetch_dustmaps(dustmap='bayestar19')

   # Load the Bayestar 3D dust map from the downloaded file
   dustmap = Bayestar(dustfile=dustmap_path)

   # Query extinction at a specific sightline using Galactic (l, b)
   l, b = 120.0, 30.0  # Galactic coordinates in degrees
   distances, av_mean, av_std = dustmap.query([l, b])

   # Or equivalently, using an astropy SkyCoord
   from astropy.coordinates import SkyCoord
   import astropy.units as u
   coord = SkyCoord(l=120.0*u.deg, b=30.0*u.deg, frame='galactic')
   distances, av_mean, av_std = dustmap.query(coord)

   # distances: array of distance grid points in kpc
   # av_mean: cumulative A_V extinction at each distance (magnitudes)
   # av_std: uncertainty on A_V at each distance (magnitudes)

   print(f"Distance grid: {distances.min():.2f} - {distances.max():.2f} kpc")
   print(f"A_V at 1 kpc: {av_mean[distances < 1][-1]:.2f} mag")

.. note::

   The Bayestar 3D dust map is based on Pan-STARRS 1 photometry and is only
   available north of declination -30 degrees (the Pan-STARRS 1 footprint). For
   sightlines south of this limit, the dust map prior will not be applied and
   fitting will rely on the other priors instead.

   By default, queries also honor the map's per-pixel reliability metadata:
   distance bins outside a sightline's reliable distance range, and pixels
   whose fits did not converge, are returned as NaN (so the above example can
   print NaN for some sightlines). NaN values degrade to an uninformative
   extinction prior during fitting. Pass
   ``Bayestar(dustfile=..., apply_reliability_mask=False)`` (or the same
   keyword to ``query``) to disable this masking.

Running the Fitter
^^^^^^^^^^^^^^^^^^

brutus expects photometry as **linear flux densities** (not magnitudes) with
uncertainties and a validity mask.

.. important::

   **Expected units:**

   - **Flux**: Linear flux densities where ``mag = -2.5 * log10(flux)``.
     Convert from magnitudes: ``flux = 10**(-0.4 * mag)``.
     These are sometimes called "maggies" (1 maggie = flux of a 0th magnitude source).

   - **Parallax**: Milliarcseconds (mas)

   - **Coordinates**: Galactic longitude and latitude in degrees

.. note::

   **Parallax is helpful but not required.** If parallax measurements are
   available, they significantly improve distance and extinction estimates by
   breaking the distance-extinction degeneracy. When parallax is not provided,
   brutus relies on photometry and priors alone, which still yields useful
   results but with broader posteriors.

.. important::

   **Minimum 4 photometric bands recommended.** The fitter has 3 free
   parameters per star (scale/distance, A_V, R_V), so at least 4 bands are
   needed for a constrained fit. The code will emit a ``RuntimeWarning`` if
   fewer than 4 valid bands are provided but will not hard-fail; an object
   with **zero** valid bands (all bands masked or non-finite) raises a
   ``ValueError``, so filter such objects out before fitting. Optical +
   near-IR coverage (e.g., Gaia + 2MASS) is recommended to break color-extinction
   degeneracies.

.. code-block:: python

   from brutus.analysis import BruteForce
   from brutus.utils import inv_magnitude
   import numpy as np

   # Convert magnitudes to linear flux densities
   flux, flux_err = inv_magnitude(mag, mag_err)

   # Create fitter instance
   fitter = BruteForce(grid)

   # Run fitting (results saved to HDF5 file)
   fitter.fit(
       data=flux,                # (N_stars, N_filters) linear flux densities
       data_err=flux_err,        # (N_stars, N_filters) flux uncertainties
       data_mask=flux_mask,      # (N_stars, N_filters) boolean validity mask
       data_labels=obj_ids,      # (N_stars,) object identifiers
       phot_offsets=offsets,     # Photometric calibration offsets
       parallax=parallax,        # (N_stars,) parallax in mas
       parallax_err=parallax_err,# (N_stars,) parallax error in mas
       data_coords=coords,       # (N_stars, 2) Galactic (l, b) in degrees
       dustfile=dustmap_path,    # 3D dust map for extinction prior (optional)
       avlim=(0.0, 6.0),         # A_V bounds (magnitudes), default (0, 20)
       rvlim=(2.5, 4.5),         # R_V bounds, default (1, 8)
       rv_gauss=(3.32, 0.18),    # Gaussian prior on R_V (mean, std), default
       save_file='results.h5'
   )

Accessing Results
^^^^^^^^^^^^^^^^^

Results are stored in an HDF5 file containing posterior samples and diagnostics:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'r') as f:
       # Posterior samples: shape (N_stars, N_draws)
       distances = f['samps_dist'][:]    # Distance in kpc
       av_values = f['samps_red'][:]     # A_V extinction in magnitudes
       rv_values = f['samps_dred'][:]    # R_V (dimensionless)

       # Model indices for stellar parameters
       model_idx = f['model_idx'][:]     # Indices into grid labels

       # Fit quality diagnostics
       log_evidence = f['obj_log_evid'][:]  # Log-evidence per star
       chi2_min = f['obj_chi2min'][:]       # Minimum chi-squared
       n_bands = f['obj_Nbands'][:]         # Number of bands used

   # Compute median and 68% credible intervals for distance
   dist_median = np.median(distances, axis=1)
   dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)

   print(f"Distance: {dist_median[0]*1000:.0f} pc "
         f"(+{(dist_84[0]-dist_median[0])*1000:.0f}/"
         f"-{(dist_median[0]-dist_16[0])*1000:.0f})")

Deriving Stellar Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``model_idx`` to look up stellar parameters from the model grid:

.. code-block:: python

   # Load the same grid used for fitting
   models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)

   # Extract stellar parameters for each posterior draw
   # labels contains: mini, feh, eep, loga, logl, logt, logg, agewt
   star_idx = 0  # First star

   # Get initial mass and surface gravity for all draws
   mini_draws = labels['mini'][model_idx[star_idx]]   # Initial mass (solar masses)
   logg_draws = labels['logg'][model_idx[star_idx]]   # log(g) in cgs
   logt_draws = labels['logt'][model_idx[star_idx]]   # log(Teff/K)

   # Compute summary statistics
   mass_median = np.median(mini_draws)
   mass_16, mass_84 = np.percentile(mini_draws, [16, 84])
   logg_median = np.median(logg_draws)
   teff_median = 10**np.median(logt_draws)

   print(f"Mass: {mass_median:.2f} (+{mass_84-mass_median:.2f}/-{mass_median-mass_16:.2f}) Msun")
   print(f"log(g): {logg_median:.2f} dex")
   print(f"Teff: {teff_median:.0f} K")

**Output datasets:**

- ``samps_dist``: Distance draws in kpc (N_stars, N_draws)
- ``samps_red``: A_V extinction draws in magnitudes (N_stars, N_draws)
- ``samps_dred``: R_V draws, dimensionless (N_stars, N_draws)
- ``model_idx``: Model grid indices (N_stars, N_draws)
- ``obj_log_evid``: Log-evidence per star (N_stars,)
- ``obj_chi2min``: Minimum chi-squared per star (N_stars,)
- ``obj_Nbands``: Number of photometric bands used (N_stars,)

**Grid labels** (accessible via ``model_idx``):

- ``mini``: Initial mass in solar masses
- ``logg``: Surface gravity, log(g/[cm/s²])
- ``logt``: Effective temperature, log(Teff/K)
- ``logl``: Luminosity, log(L/L☉)
- ``loga``: Age, log(age/yr)
- ``feh``: Metallicity [Fe/H]
- ``eep``: Equivalent evolutionary point

See :doc:`understanding_results` for detailed guidance on interpreting output.

----

Modeling Stellar Clusters
-------------------------

For stellar clusters with shared age, metallicity, and distance, use
``isochrone_population_loglike`` which handles membership probabilities,
binary stars, and field star contamination.

.. code-block:: python

   from brutus.core import Isochrone, StellarPop
   from brutus.analysis import isochrone_population_loglike
   from brutus.utils import inv_magnitude
   import numpy as np

   # Set up isochrone and population generator
   filters = ['Gaia_G_MAW', 'PS_g', 'PS_r', 'PS_i', '2MASS_J']
   iso = Isochrone()
   pop = StellarPop(isochrone=iso, filters=filters)

   # Convert observed magnitudes to flux densities
   flux, flux_err = inv_magnitude(mag, mag_err)

   # Per-object membership probabilities from kinematic analysis
   # e.g., from proper motion/radial velocity modeling
   membership_prob = kinematic_membership  # (N_stars,) array, values 0-1

   # Define negative log-likelihood for optimization
   def neg_loglike(theta):
       """
       Negative log-likelihood for cluster parameters.

       Parameters theta = [feh, loga, av, rv, dist, field_frac]
         - feh: metallicity [Fe/H]
         - loga: log10(age/yr)
         - av: V-band extinction (magnitudes)
         - rv: extinction law R_V
         - dist: distance (parsecs)
         - field_frac: field contamination fraction (0 to 1)
       """
       return -isochrone_population_loglike(
           theta,
           stellarpop=pop,
           obs_phot=flux,                # (N_stars, N_filters) flux densities
           obs_err=flux_err,             # (N_stars, N_filters) flux errors
           parallax=parallax,            # (N_stars,) parallax in mas (optional)
           parallax_err=parallax_err,    # (N_stars,) parallax error in mas
           cluster_prob=membership_prob, # (N_stars,) per-object membership probability
           mask=flux_mask,               # (N_stars, N_filters) validity mask
       )

Find the best-fit parameters using optimization:

.. code-block:: python

   from scipy.optimize import minimize

   # Initial guess: [feh, loga, av, rv, dist, field_frac]
   theta0 = [0.0, 9.0, 0.3, 3.1, 1000.0, 0.05]

   result = minimize(neg_loglike, theta0, method='Nelder-Mead')

   feh, loga, av, rv, dist, field_frac = result.x
   print(f"Age: {10**loga / 1e9:.2f} Gyr")
   print(f"[Fe/H]: {feh:.2f}")
   print(f"Distance: {dist:.0f} pc")
   print(f"A_V: {av:.2f} mag")
   print(f"Field fraction: {field_frac:.2f}")

.. note::

   For full posterior distributions rather than point estimates, use MCMC
   (e.g., ``emcee``) or nested sampling (e.g., ``dynesty``) with appropriate
   priors.

----

Line-of-Sight Dust Modeling
---------------------------

brutus can model 3D dust distributions along lines of sight by fitting
multi-cloud models to stellar posterior samples from ``BruteForce`` fitting.
This is useful for mapping dust structure in specific directions.

.. note::

   This example uses `dynesty <https://dynesty.readthedocs.io/>`_ for nested
   sampling. Install it with ``pip install dynesty``. Other nested sampling
   packages (e.g., ``nestle``, ``ultranest``) can be used with similar interfaces.

.. code-block:: python

   from dynesty import NestedSampler
   from brutus.analysis import (
       los_clouds_priortransform,
       los_clouds_loglike_samples
   )

   # Stellar posteriors from BruteForce fitting:
   # dsamps: (N_stars, N_samples) distance modulus samples
   # rsamps: (N_stars, N_samples) A_V extinction samples

   # 2-cloud model has 8 parameters
   n_clouds = 2
   ndim = 2 * n_clouds + 4

   def prior_transform(u):
       return los_clouds_priortransform(
           u,
           rlims=(0, 6),      # A_V range
           dlims=(4, 19)      # Distance modulus range
       )

   def loglike(theta):
       return los_clouds_loglike_samples(
           theta,
           dsamps,
           rsamps,
           kernel='gauss'     # Smoothing kernel: 'gauss', 'lorentz', 'tophat'
       )

   sampler = NestedSampler(loglike, prior_transform, ndim)
   sampler.run_nested()
   results = sampler.results
