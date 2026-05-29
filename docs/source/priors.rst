Prior Distributions
===================

This page describes the **prior probability distributions** used in ``brutus``. Priors encode astrophysical knowledge about the Galaxy and help break parameter degeneracies inherent in photometric fitting.

For detailed mathematical derivations, see `Speagle et al. (2025) <https://arxiv.org/abs/2503.02227>`_ §2.4 and Appendix A.

Why Priors Matter
-----------------

Photometry alone cannot uniquely determine stellar properties. A nearby cool M dwarf and a distant reddened K giant can produce similar colors and magnitudes. **Priors help resolve this ambiguity** by incorporating knowledge about Galactic structure, stellar mass distributions, and dust :term:`extinction`.

Prior impact depends on data quality:

- **Well-measured stars** (bright, accurate parallax): Likelihood dominates; results are insensitive to priors
- **Poorly-measured stars** (faint, no parallax): Priors strongly influence results

Always check prior sensitivity for your science case (see `Testing Prior Sensitivity`_ below).

.. note::

   **Parallax is not required but helps significantly.** When parallax
   measurements are available, they break the distance-extinction degeneracy that
   is inherent in photometric fitting, leading to much tighter posteriors. Without
   parallax, brutus relies on photometry and Galactic priors alone, which still
   produces useful results but with broader uncertainties, especially for faint
   stars.

The Galactic Model
------------------

``brutus`` uses a 3-D Galactic model with factorized :term:`priors <prior>`:

.. math::

   \pi(\theta, \phi) \propto \pi(M_{\rm init}) \times \pi({\rm EEP}) \times \pi(d\,|\,\ell,b) \times \pi([{\rm Fe/H}]\,|\,d,\ell,b) \times \pi(t_{\rm age}\,|\,d,\ell,b) \times \pi(A_V\,|\,d,\ell,b) \times \pi(R_V)

Prior Components
----------------

Initial Mass Function
^^^^^^^^^^^^^^^^^^^^^

The Kroupa (2001) :term:`IMF` describes stellar masses at formation as a two-part power law:

.. math::

   \pi(M) \propto \begin{cases} M^{-1.3} & 0.08 < M/M_\odot < 0.5 \\ M^{-2.3} & 0.5 < M/M_\odot < 150 \end{cases}

Low-mass stars strongly dominate; a randomly drawn star is ~10× more likely to be 0.5 M☉ than 1.0 M☉.

**Implementation**: :func:`brutus.priors.logp_imf`

Evolutionary State (EEP)
^^^^^^^^^^^^^^^^^^^^^^^^

The :term:`EEP` prior is uniform, which accounts for varying evolutionary timescales. Stars spend most of their lives on the main sequence (EEP ~300-450), so the uniform EEP prior naturally weights toward main sequence stars when combined with the age prior.

3-D Stellar Density
^^^^^^^^^^^^^^^^^^^

The spatial distribution combines three Galactic components, each with characteristic structure and stellar populations:

.. list-table::
   :widths: 20 40 20 20
   :header-rows: 1

   * - Component
     - Spatial Profile
     - Mean [Fe/H]
     - Mean Age
   * - Thin disk
     - Exponential (h_R = 2.6 kpc, h_Z = 300 pc)
     - -0.2 dex
     - ~5 Gyr
   * - Thick disk
     - Exponential (h_R = 2.0 kpc, h_Z = 900 pc)
     - -0.7 dex
     - ~8 Gyr
   * - Halo
     - Flattened power-law (η = 4.2)
     - -1.6 dex
     - ~12 Gyr

The **halo** follows a flattened (oblate) power-law profile with radius-dependent oblateness: highly flattened near the Galactic center (q = 0.2) and more spherical at large radii (q = 0.8), transitioning over a scale of ~6 kpc.

The combined prior weights each component by its stellar density at the 3-D position :math:`(d, \ell, b)`. Near the Sun, the thin disk dominates; at high Galactic latitudes or large distances, the thick disk and halo contribute more.

**Implementation**: :func:`brutus.priors.logp_galactic_structure`, :func:`brutus.priors.logp_feh`, :func:`brutus.priors.logp_age_from_feh`, :func:`brutus.priors.logn_disk`, :func:`brutus.priors.logn_halo`

3-D Dust Extinction
^^^^^^^^^^^^^^^^^^^

``brutus`` can use 3-D dust maps to provide distance-dependent :term:`extinction` priors. The default dust map file distributed with ``brutus`` is derived from **Bayestar19** (Green et al. 2019). For a given sky position :math:`(\ell, b)` and distance :math:`d`:

.. math::

   \pi(A_V\,|\,d,\ell,b) \sim \mathcal{N}(\mu_{A_V}, \sigma_{A_V}^2)

where the mean :math:`\mu_{A_V}` and uncertainty :math:`\sigma_{A_V}` come from the dust map at that sightline and distance.

.. note::

   The Bayestar 3D dust map is based on Pan-STARRS 1 photometry and is only
   available north of declination -30 degrees (the Pan-STARRS 1 footprint). For
   sightlines south of this limit, the dust map prior will not be applied.

**Implementation**: Enabled via ``dustfile`` parameter in ``fit()``

R_V Variation
^^^^^^^^^^^^^

The extinction curve shape :term:`R_V` :math:`\equiv A_V / E(B-V)` has a truncated Gaussian prior:

.. math::

   \pi(R_V) \sim \mathcal{N}(3.32, 0.18^2) \quad {\rm for} \quad 1.0 < R_V < 8.0

The default mean (3.32) and standard deviation (0.18) are based on Schlafly et al. (2016), who measured R_V variations toward tens of thousands of APOGEE stars across the Milky Way.

**Implementation**: Controlled via ``rv_gauss`` and ``rvlim`` parameters in ``fit()``

Customizing Priors
------------------

Disabling Priors
^^^^^^^^^^^^^^^^

For diagnostic purposes, priors can be disabled or made uninformative:

**Galactic structure prior:**

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Fit without Galactic structure prior
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       lngalprior=lambda *args, **kwargs: 0.0,  # Uniform prior
   )

**Extinction priors:**

.. code-block:: python

   # Fit with uninformative (wide) priors on A_V and R_V
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       avlim=(0.0, 10.0),            # Wide A_V range
       av_gauss=(0.0, 1e6),          # Effectively uniform A_V prior
       rvlim=(1.0, 8.0),             # Wide R_V range
       rv_gauss=(3.32, 1e6),         # Effectively uniform R_V prior
   )

.. warning::
   Disabling priors can lead to highly degenerate parameter estimates. Only disable when you understand the implications.

Custom Prior Functions
^^^^^^^^^^^^^^^^^^^^^^

Pass custom Galactic structure prior functions via ``lngalprior``:

.. code-block:: python

   from brutus.priors import logp_galactic_structure

   def custom_galactic_prior(dist, coord, labels=None):
       """Custom prior: uniform within 100 pc, default otherwise.

       ``dist`` is in kpc and ``coord`` is a single ``(l, b)`` tuple in
       degrees, matching how ``fit()`` calls ``lngalprior``.
       """
       if dist < 0.1:  # kpc
           # Return value at boundary to ensure continuity
           return logp_galactic_structure(0.1, coord, labels=labels)
       return logp_galactic_structure(dist, coord, labels=labels)

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

For cluster modeling with fixed age/metallicity/distance, see :doc:`population_modeling`.

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
              lngalprior=lambda *args, **kwargs: 0.0)

   # Compare
   with h5py.File('with.h5', 'r') as f:
       d1 = np.median(f['samps_dist'][0])
   with h5py.File('without.h5', 'r') as f:
       d2 = np.median(f['samps_dist'][0])

   change = abs(d1 - d2) / d1
   print(f"Fractional change: {change:.1%}")

Available Prior Functions
--------------------------

.. currentmodule:: brutus.priors

**Stellar Priors**:

- :func:`logp_imf` - Initial mass function (Kroupa two-part power law)
- :func:`logp_ps1_luminosity_function` - Pan-STARRS r-band luminosity function

**Galactic Priors**:

- :func:`logp_galactic_structure` - Combined 3-D stellar density (thin disk + thick disk + halo)
- :func:`logn_disk` - Exponential disk number density
- :func:`logn_halo` - Flattened power-law halo number density
- :func:`logp_feh` - Metallicity distribution by Galactic component
- :func:`logp_age_from_feh` - Age distribution conditioned on metallicity

**Astrometric Priors**:

- :func:`logp_parallax` - Parallax prior with distance-to-parallax conversion
- :func:`logp_parallax_scale` - Parallax constraint on flux density scale factors
- :func:`convert_parallax_to_scale` - Convert parallax to flux density scale factor

**Extinction Priors**:

- :func:`logp_extinction` - 3-D dust extinction from Bayestar maps

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_ (§2.4, Appendix A)

Kroupa (2001), "On the variation of the initial mass function", `MNRAS, 322, 231 <https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K>`_

Green et al. (2019), "A 3D Dust Map Based on Gaia, Pan-STARRS 1, and 2MASS", `ApJ, 887, 93 <https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G>`_

Jurić et al. (2008), "The Milky Way Tomography with SDSS. I. Stellar Number Density Distribution", `ApJ, 673, 864 <https://ui.adsabs.harvard.edu/abs/2008ApJ...673..864J>`_

Schlafly et al. (2016), "The Optical-Infrared Extinction Curve and its Variation in the Milky Way", `ApJ, 821, 78 <https://ui.adsabs.harvard.edu/abs/2016ApJ...821...78S>`_
