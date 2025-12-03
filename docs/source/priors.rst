Prior Probability Distributions
=================================

This page describes the **prior probability distributions** used in brutus. Priors encode astrophysical knowledge about the Galaxy and help break parameter degeneracies inherent in photometric fitting.

For detailed mathematical derivations, see `Speagle et al. (2025) <https://arxiv.org/abs/2503.02227>`_, §2.4 and Appendix A.

Why Priors Matter
------------------

Photometry alone cannot uniquely determine stellar properties. A nearby cool M dwarf and a distant reddened K giant can produce identical colors and magnitudes. **Priors resolve this ambiguity** by incorporating knowledge about Galactic structure, stellar mass distributions, and dust extinction.

Prior impact depends on data quality: with good data (bright stars, accurate parallax), the likelihood dominates and results are insensitive to priors. With poor data (faint stars, no parallax), priors strongly influence results.

The Galactic Model
-------------------

brutus uses a 3-D Galactic model with factorized priors:

.. math::

   \pi(\theta, \phi) \propto \pi(M_{\rm init}) \times \pi(d\,|\,\ell,b) \times \pi([{\rm Fe/H}]\,|\,d,\ell,b) \times \pi(t_{\rm age}\,|\,d,\ell,b) \times \pi(A_V\,|\,d,\ell,b) \times \pi(R_V)

Prior Components
-----------------

**Initial Mass Function (IMF)**

The Kroupa (2001) two-part power law describes stellar masses at formation:

- :math:`\pi(M) \propto M^{-1.3}` for :math:`0.08 < M < 0.5\,M_\odot`
- :math:`\pi(M) \propto M^{-2.3}` for :math:`0.5 < M < 150\,M_\odot`

Low-mass stars dominate; high-mass stars are rare.

**Implementation**: :func:`brutus.priors.logp_imf`

**3-D Stellar Density**

The spatial distribution combines three Galactic components:

- **Thin disk**: Scale height ~300 pc, young to intermediate-age stars ([Fe/H] ~ -0.2)
- **Thick disk**: Scale height ~900 pc, older stars ([Fe/H] ~ -0.7)
- **Halo**: Power-law profile, ancient metal-poor stars ([Fe/H] ~ -1.6)

Each component has characteristic metallicity and age distributions. The combined prior is weighted by stellar density at each 3-D position.

**Implementation**: :func:`brutus.priors.logp_galactic_structure`, :func:`brutus.priors.logp_feh`, :func:`brutus.priors.logp_age_from_feh`

**3-D Dust Extinction**

brutus uses **Bayestar19** 3-D dust maps (Green et al. 2019) providing distance-dependent extinction priors. For a given sky position :math:`(\ell, b)` and distance :math:`d`:

.. math::

   \pi(A_V\,|\,d,\ell,b) \sim \mathcal{N}(\mu_{A_V}, \sigma_{A_V}^2)

where the mean and uncertainty come from the dust map.

**Implementation**: Enabled via ``dustfile`` parameter in ``fit()``

**R_V Variation**

The extinction curve shape :math:`R_V \equiv A_V / E(B-V)` has a truncated Gaussian prior:

.. math::

   \pi(R_V) \sim \mathcal{N}(3.32, 0.18^2) \quad {\rm for} \quad 2.0 < R_V < 6.0

**Implementation**: Controlled via ``rv_gauss`` and ``rvlim`` parameters in ``fit()``

Customizing Priors
------------------

Disabling Priors
^^^^^^^^^^^^^^^^

For diagnostic purposes, disable priors by passing uniform functions to ``fit()``:

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Fit without Galactic structure prior
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       lngalprior=lambda *args: 0.0,  # Uniform prior
   )

   # Fit without dust map prior
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       lndustprior=lambda *args: 0.0,  # Uniform prior
   )

.. warning::
   Disabling priors can lead to highly degenerate parameter estimates. Only disable when you understand the implications.

Custom Prior Functions
^^^^^^^^^^^^^^^^^^^^^^

Pass custom functions via ``lngalprior`` and ``lndustprior``:

.. code-block:: python

   from brutus.priors import logp_galactic_structure

   def custom_galactic_prior(dist, gal_l, gal_b, dlabels=None):
       """Custom prior: uniform within 100 pc, default otherwise."""
       if dist < 0.1:  # kpc
           return 0.0
       return logp_galactic_structure(dist, gal_l, gal_b, dlabels)

   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       data_coords=coords,
       lngalprior=custom_galactic_prior,
   )

When to Customize
^^^^^^^^^^^^^^^^^

Consider customizing priors for:

- **Extragalactic objects**: LMC/SMC stars need different Galactic priors
- **Special regions**: Galactic bulge, Local Bubble, or spiral arms
- **Known populations**: If you have independent age/metallicity constraints

For cluster modeling with fixed age/metallicity/distance, see :doc:`cluster_modeling`.

Testing Prior Sensitivity
--------------------------

Compare results with and without priors to assess prior influence:

.. code-block:: python

   import h5py
   import numpy as np

   # Fit with and without Galactic prior
   fitter.fit(data, data_err, data_mask, labels, save_file='with.h5',
              data_coords=coords)
   fitter.fit(data, data_err, data_mask, labels, save_file='without.h5',
              lngalprior=lambda *args: 0.0)

   # Compare
   with h5py.File('with.h5', 'r') as f:
       d1 = np.median(f['samps_dist'][0])
   with h5py.File('without.h5', 'r') as f:
       d2 = np.median(f['samps_dist'][0])

   change = abs(d1 - d2) / d1
   print(f"Fractional change: {change:.1%}")
   # >30% change suggests prior-dominated results

Available Prior Functions
--------------------------

.. currentmodule:: brutus.priors

- :func:`logp_imf` - Initial mass function (Kroupa)
- :func:`logp_galactic_structure` - 3-D stellar density
- :func:`logp_feh` - Metallicity distribution
- :func:`logp_age_from_feh` - Age distribution
- :func:`logp_parallax` - Parallax prior with scale conversion

Next Steps
----------

- Configure fitting options: :doc:`choosing_options`
- Understand results: :doc:`understanding_results`
- Cluster modeling: :doc:`cluster_modeling`

References
----------

- Speagle et al. (2025), arXiv:2503.02227 - brutus methods (§2.4, Appendix A)
- Kroupa (2001), MNRAS, 322, 231 - Initial Mass Function
- Green et al. (2019), ApJ, 887, 93 - Bayestar19 dust maps
- Jurić et al. (2008), ApJ, 673, 864 - Galactic structure
