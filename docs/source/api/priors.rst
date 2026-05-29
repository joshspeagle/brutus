Priors Module (``brutus.priors``)
==================================

The priors module implements prior probability distributions for Bayesian inference in brutus. These priors encode astrophysical knowledge about the Galaxy and are critical for breaking parameter degeneracies in stellar parameter estimation.

**Why Priors Matter:**

Photometry alone cannot uniquely determine stellar properties—a faint red star could be either a nearby cool dwarf or a distant reddened giant. Priors resolve these ambiguities by incorporating knowledge about:

- Where different stellar types are located in the Galaxy (Galactic structure)
- How dust extinction varies with distance and direction (3-D dust maps)
- The relative abundance of different stellar masses (IMF)
- The metallicity and age structure of different Galactic populations

**Prior Categories:**

1. **Stellar Priors**: Initial Mass Function (IMF) and luminosity functions
2. **Astrometric Priors**: Parallax-based distance constraints
3. **Galactic Structure Priors**: 3-D spatial distribution of stars (thin/thick disk, halo)
4. **Extinction Priors**: 3-D dust maps and R_V variation

**Typical Usage:**

Priors are automatically applied by the ``BruteForce`` fitter, but can also be evaluated directly:

.. code-block:: python

   from brutus.priors.stellar import logp_imf
   from brutus.priors.galactic import logp_galactic_structure
   from brutus.priors.extinction import logp_extinction
   import numpy as np

   # Evaluate IMF prior for range of masses
   masses = np.array([0.5, 1.0, 2.0, 5.0])
   log_prior_imf = logp_imf(masses)

   # Evaluate Galactic structure prior
   distances = np.array([0.1, 0.5, 1.0, 5.0])  # kpc
   gal_l, gal_b = 45.0, 10.0  # Galactic coordinates (deg)
   log_prior_gal = logp_galactic_structure(distances, (gal_l, gal_b))

   # Evaluate extinction prior from 3-D dust map
   from brutus.dust import Bayestar
   dustmap = Bayestar()  # load a 3-D dust map object
   av_values = np.array([0.1, 0.5, 1.0, 2.0])  # mag
   log_prior_dust = logp_extinction(av_values, dustmap, (gal_l, gal_b),
                                    distance=distances[0])

**Customization:**

Advanced users can customize or disable priors by passing callables to ``fit()``.
The Galactic structure prior is supplied via ``lngalprior``, which is called as
``lngalprior(dist, coord, labels=...)`` where ``coord`` is a single ``(l, b)`` tuple:

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Disable the Galactic structure prior with a flat (zero) prior
   fitter.fit(
       data=flux, data_err=flux_err, data_mask=mask,
       data_labels=obj_ids, save_file='results.h5',
       data_coords=coords,
       lngalprior=lambda dist, coord, labels=None: np.zeros(len(np.atleast_1d(dist))),
   )

   # Or provide a custom Galactic prior function with the same signature

**See Also:**

- :doc:`/priors` - Detailed conceptual guide to all prior distributions
- :doc:`/scientific_background` - How priors fit into the Bayesian framework

.. currentmodule:: brutus.priors

Stellar Priors
--------------

.. autofunction:: logp_imf

.. autofunction:: logp_ps1_luminosity_function

Astrometric Priors
------------------

.. autofunction:: logp_parallax

.. autofunction:: logp_parallax_scale

.. autofunction:: convert_parallax_to_scale

Galactic Structure Priors
--------------------------

.. autofunction:: logp_galactic_structure

.. autofunction:: logn_disk

.. autofunction:: logn_halo

.. autofunction:: logp_feh

.. autofunction:: logp_age_from_feh

Extinction Priors
-----------------

.. autofunction:: logp_extinction

Submodules
----------

For advanced users who need access to internal implementations:

.. automodule:: brutus.priors.stellar
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.astrometric
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.galactic
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: brutus.priors.extinction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
