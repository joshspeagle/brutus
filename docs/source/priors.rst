Prior Probability Distributions
=================================

This page describes the **prior probability distributions** used in brutus for Bayesian stellar parameter estimation. Priors encode astrophysical knowledge about the Galaxy and are critical for breaking parameter degeneracies.

Why Priors Matter
------------------

Photometry alone cannot uniquely determine stellar properties. Consider two stars with identical photometry:

- **Scenario A**: Nearby (500 pc), cool M dwarf, no extinction
- **Scenario B**: Distant (5 kpc), warm K giant, moderate extinction (A_V ~ 2 mag)

Both scenarios can produce the same observed colors and magnitudes. **Priors resolve this ambiguity** by incorporating knowledge about:

1. Where different stellar types are located in the Galaxy
2. How dust extinction varies with distance and direction
3. The relative abundance of different stellar masses (IMF)
4. The metallicity and age structure of Galactic populations

Without priors, parameter estimates are highly uncertain and potentially biased. With appropriate priors, brutus can robustly estimate distances, extinctions, and stellar properties even with limited data.

The Galactic Model
-------------------

brutus uses a comprehensive 3-D Galactic model that factorizes into independent components:

.. math::

   \pi(\theta, \phi) \propto \pi(M_{\rm init}) \times \pi(d\,|\,\ell,b) \times \pi([{\rm Fe/H}]\,|\,d,\ell,b) \times \pi(t_{\rm age}\,|\,d,\ell,b) \times \pi(A_V\,|\,d,\ell,b) \times \pi(R_V)

where :math:`\theta = (M_{\rm init}, [{\rm Fe/H}]_{\rm init}, t_{\rm age})` are intrinsic stellar parameters, :math:`\phi = (d, A_V, R_V)` are extrinsic parameters, and :math:`(\ell, b)` are Galactic coordinates.

This factorization assumes that stellar properties and dust extinction are independent given the 3-D location. While this is a simplification (e.g., metallicity and dust are correlated through spiral structure), it provides a tractable and physically motivated prior framework.

Initial Mass Function
----------------------

The **Initial Mass Function (IMF)** describes the distribution of stellar masses at formation. brutus uses the **Kroupa (2001) IMF**:

.. math::

   \pi(M_{\rm init}) \propto \begin{cases}
   M_{\rm init}^{-1.3} & 0.08 < M_{\rm init} < 0.5\,M_\odot \\
   M_{\rm init}^{-2.3} & 0.5 < M_{\rm init} < 150\,M_\odot
   \end{cases}

This two-part power law reflects the observed mass distribution in stellar populations:

- **Low masses dominate**: The IMF rises steeply toward lower masses
- **Turnover at ~0.5** :math:`M_\odot`: The slope flattens for brown dwarfs and very low-mass stars
- **Exponential cutoff**: Extremely massive stars (>150 :math:`M_\odot`) are vanishingly rare

The IMF is **independent of location** in the Galaxy—it appears to be universal across different environments (clusters, field, nearby galaxies).

**Implementation**: See :func:`brutus.priors.logp_imf`

.. code-block:: python

   from brutus.priors import logp_imf
   import numpy as np

   # Compute IMF prior for range of masses
   masses = np.array([0.1, 0.5, 1.0, 5.0, 20.0])
   log_prior = logp_imf(masses)

   # IMF strongly favors low masses
   print(log_prior)  # More negative = lower probability

3-D Stellar Number Density
---------------------------

The **spatial distribution** of stars in the Milky Way has three main components:

Thin Disk
^^^^^^^^^

The thin disk contains young to intermediate-age stars with low velocity dispersion:

- **Scale height**: :math:`h_z \approx 300` pc
- **Scale radius**: :math:`R_d \approx 2.6` kpc
- **Density profile**: Exponential in radius and height

.. math::

   \rho_{\rm thin}(R, z) \propto \exp\left(-\frac{R - R_\odot}{R_d}\right) \exp\left(-\frac{|z|}{h_z}\right)

where :math:`R` is galactocentric radius, :math:`z` is height above the plane, and :math:`R_\odot = 8.2` kpc is the Solar radius.

Thick Disk
^^^^^^^^^^

The thick disk contains older stars with higher velocity dispersion:

- **Scale height**: :math:`h_z \approx 900` pc
- **Scale radius**: :math:`R_d \approx 2.0` kpc
- **Local normalization**: ~10-15% of thin disk density

.. math::

   \rho_{\rm thick}(R, z) \propto \exp\left(-\frac{R - R_\odot}{R_d}\right) \exp\left(-\frac{|z|}{h_z}\right)

Halo
^^^^

The stellar halo contains ancient, metal-poor stars in a roughly spherical distribution:

- **Density profile**: Power law :math:`\rho \propto r^{-\alpha}` with :math:`\alpha \approx 2.5`
- **Flattening**: Slightly oblate with axis ratio :math:`q \approx 0.6`
- **Local normalization**: ~0.1% of thin disk density

.. math::

   \rho_{\rm halo}(r) \propto r^{-2.5}

**Total stellar density**: The combined prior is the sum of all components:

.. math::

   \pi(d\,|\,\ell, b) \propto \rho_{\rm thin}(R, z) + \rho_{\rm thick}(R, z) + \rho_{\rm halo}(r)

where the 3-D position :math:`(R, z, r)` is computed from distance :math:`d` and sky position :math:`(\ell, b)`.

**Implementation**: See :func:`brutus.priors.logp_galactic_structure`

3-D Metallicity Distribution
-----------------------------

Different Galactic components have distinct metallicity distributions:

Thin Disk Metallicity
^^^^^^^^^^^^^^^^^^^^^^

- **Mean**: :math:`\langle [{\rm Fe/H}] \rangle \approx -0.2` dex
- **Dispersion**: :math:`\sigma_{[{\rm Fe/H}]} \approx 0.2` dex
- **Radial gradient**: :math:`d[{\rm Fe/H}]/dR \approx -0.06` dex/kpc

.. math::

   \pi([{\rm Fe/H}]\,|\,{\rm thin\,disk}) \sim \mathcal{N}\left(-0.2 - 0.06 \times \frac{R - R_\odot}{1\,{\rm kpc}}, 0.2^2\right)

Thick Disk Metallicity
^^^^^^^^^^^^^^^^^^^^^^^

- **Mean**: :math:`\langle [{\rm Fe/H}] \rangle \approx -0.7` dex
- **Dispersion**: :math:`\sigma_{[{\rm Fe/H}]} \approx 0.3` dex
- **Weak radial gradient**

.. math::

   \pi([{\rm Fe/H}]\,|\,{\rm thick\,disk}) \sim \mathcal{N}(-0.7, 0.3^2)

Halo Metallicity
^^^^^^^^^^^^^^^^

- **Mean**: :math:`\langle [{\rm Fe/H}] \rangle \approx -1.6` dex
- **Dispersion**: :math:`\sigma_{[{\rm Fe/H}]} \approx 0.5` dex
- **Extended tail** to very metal-poor ([Fe/H] < -3)

.. math::

   \pi([{\rm Fe/H}]\,|\,{\rm halo}) \sim \mathcal{N}(-1.6, 0.5^2)

**Combined metallicity prior**: Weighted by stellar density of each component:

.. math::

   \pi([{\rm Fe/H}]\,|\,d,\ell,b) = \sum_{i} w_i(d,\ell,b) \times \pi([{\rm Fe/H}]\,|\,{\rm component}_i)

where :math:`w_i = \rho_i / \sum_j \rho_j` are the fractional densities.

**Implementation**: See :func:`brutus.priors.logp_feh`

3-D Age Distribution
---------------------

Stellar age correlates strongly with Galactic component:

Thin Disk Age
^^^^^^^^^^^^^

- **Mean age**: ~5 Gyr
- **Range**: 0 to ~10 Gyr
- **Distribution**: Roughly uniform with slight increase at young ages (ongoing star formation)

Thick Disk Age
^^^^^^^^^^^^^^

- **Mean age**: ~8-9 Gyr
- **Range**: 8 to 12 Gyr
- **Distribution**: Peaked around formation epoch

Halo Age
^^^^^^^^

- **Mean age**: ~12 Gyr
- **Range**: 10 to 13.8 Gyr
- **Distribution**: Very old, formed during Galaxy assembly

The age prior is complicated by the fact that age is not directly an input parameter in brutus. Instead, age is encoded through the **EEP-mass-metallicity mapping**. The prior is effectively:

.. math::

   \pi({\rm EEP}\,|\,M_{\rm init}, [{\rm Fe/H}], d, \ell, b) \propto \pi(t_{\rm age}({\rm EEP}, M_{\rm init}, [{\rm Fe/H}])\,|\,d,\ell,b) \times \left|\frac{dt}{d{\rm EEP}}\right|

where :math:`dt/d{\rm EEP}` is the age-weight Jacobian.

**Implementation**: See :func:`brutus.priors.logp_age_from_feh`

3-D Dust Extinction
-------------------

Interstellar dust extinction varies dramatically with position and distance. brutus uses **3-D dust maps** to provide location-dependent extinction priors.

Bayestar Dust Maps
^^^^^^^^^^^^^^^^^^

The default dust map is **Bayestar19** (Green et al. 2019), which provides:

- **Distance-resolved** extinction estimates
- **Coverage**: Full sky (except \|b\| < 5°)
- **Distance resolution**: ~25% in distance
- **Spatial resolution**: HEALPix nside=2048 (~1.7 arcmin)

For a given sky position :math:`(\ell, b)` and distance :math:`d`, Bayestar provides:

.. math::

   \pi(A_V\,|\,d,\ell,b) \sim \mathcal{N}(\mu_{A_V}(d,\ell,b), \sigma_{A_V}^2(d,\ell,b))

where :math:`\mu_{A_V}` and :math:`\sigma_{A_V}` are the mean and uncertainty from the dust map.

**Near vs Far Stars**:

- **Nearby** (d < 100 pc): Typically low extinction, narrow priors
- **Intermediate** (100 pc < d < 5 kpc): Extinction increases with distance through disk
- **Distant** (d > 5 kpc): Saturates at cumulative Galactic extinction

**Implementation**: See :func:`brutus.priors.logp_extinction` and :mod:`brutus.dust`

.. code-block:: python

   from brutus.priors import logp_extinction
   from brutus.dust.maps import get_dust_prior

   # Get extinction prior for specific sight line and distance
   gal_l, gal_b = 45.0, 10.0  # Galactic coordinates (degrees)
   distance = 2000.0  # pc

   log_prior_av = logp_extinction(
       av=0.5,  # Test A_V value
       dist=distance,
       gal_l=gal_l,
       gal_b=gal_b
   )

Dust Curve Variation: R_V Prior
--------------------------------

The **extinction curve shape** parameter :math:`R_V \equiv A_V / E(B-V)` varies with dust properties:

- **Diffuse ISM**: :math:`R_V \approx 3.1` (standard Milky Way)
- **Dense clouds**: :math:`R_V \approx 5-6` (larger grains)
- **Specific sight lines**: Can vary from :math:`R_V \approx 2` to :math:`R_V \approx 6`

brutus uses a **truncated Gaussian prior**:

.. math::

   \pi(R_V) \sim \mathcal{N}(3.32, 0.18^2) \quad {\rm for} \quad 2.0 < R_V < 6.0

This reflects empirical measurements of :math:`R_V` variation in the Galaxy while preventing unphysical values.

**Implementation**: R_V prior is built into the ``rv_gauss`` parameter of ``BruteForce.fit()``

Customizing Priors
------------------

brutus allows users to customize priors for specific science cases:

Using Custom Priors
^^^^^^^^^^^^^^^^^^^

All prior functions accept arrays and return log-probabilities:

.. code-block:: python

   from brutus.priors import logp_imf

   def custom_imf(masses, alpha=-2.0):
       """Custom single power-law IMF."""
       import numpy as np
       logp = alpha * np.log(masses)
       # Normalize (compute separately for your mass range)
       logp -= np.log(normalization_constant)
       return logp

   # Use in fitting by modifying BruteForce prior function
   fitter.prior_func = custom_imf

Turning Off Priors
^^^^^^^^^^^^^^^^^^

For diagnostic purposes, you can disable specific priors by passing uniform
(constant) prior functions to ``fit()``:

.. code-block:: python

   from brutus.analysis import BruteForce

   fitter = BruteForce(grid)

   # Fit without Galactic structure prior (uniform in distance)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       lngalprior=lambda *args: 0.0,  # Uniform prior
   )

   # Fit without dust map prior (uniform in A_V)
   fitter.fit(
       data, data_err, data_mask, labels, save_file='results.h5',
       lndustprior=lambda *args: 0.0,  # Uniform prior
   )

**Warning**: Disabling priors can lead to highly degenerate parameter estimates. Use with caution and only when you understand the implications.

Cluster-Specific Priors
^^^^^^^^^^^^^^^^^^^^^^^

For cluster modeling, some priors are modified:

- **Distance**: Tight Gaussian around cluster distance (1-10% width)
- **Age**: Fixed to cluster age (single isochrone)
- **Metallicity**: Fixed to cluster [Fe/H] (or narrow Gaussian for dispersion)
- **Extinction**: May be uniform across cluster or individually variable

See :doc:`cluster_modeling` for details on cluster-specific prior choices.

Prior Sensitivity and Validation
---------------------------------

How Sensitive Are Results to Priors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prior impact depends on data quality:

**Good data** (bright stars, low errors, parallax):
   Likelihood dominates → Results weakly sensitive to priors

**Poor data** (faint stars, high errors, no parallax):
   Priors strongly influence results → Prior choice matters

**Best practice**: Check prior sensitivity by fitting with and without specific priors:

.. code-block:: python

   import h5py
   import numpy as np
   import matplotlib.pyplot as plt

   fitter = BruteForce(grid)

   # Fit with full priors
   fitter.fit(data, data_err, data_mask, labels, save_file='with_prior.h5',
              data_coords=coords, dustfile='bayestar19.h5')

   # Fit without Galactic prior
   fitter.fit(data, data_err, data_mask, labels, save_file='no_prior.h5',
              lngalprior=lambda *args: 0.0)

   # Compare posteriors
   with h5py.File('with_prior.h5', 'r') as f:
       dist_with = f['samps_dist'][0] * 1000  # pc
   with h5py.File('no_prior.h5', 'r') as f:
       dist_without = f['samps_dist'][0] * 1000  # pc

   plt.hist(dist_with, alpha=0.5, label='With Gal prior')
   plt.hist(dist_without, alpha=0.5, label='No Gal prior')
   plt.legend()
   plt.show()

If results change dramatically, the data is prior-dominated and caution is needed.

Validating Priors
^^^^^^^^^^^^^^^^^

brutus priors are based on empirical Galactic studies, but they may not be appropriate for all science cases:

**Check prior validity**:

1. Compare assumed Galactic structure to observations in your field
2. Verify dust map predictions against spectroscopic reddening measurements
3. Test IMF assumption against cluster mass functions

**When to customize**:

- **Extragalactic studies**: Need different stellar density and metallicity priors
- **Specific regions**: Spiral arms, bulge, Local Bubble may violate default assumptions
- **Special populations**: White dwarfs, specific age cohorts need tailored IMF/age priors

Common Prior Pitfalls
---------------------

**Over-constraining with tight priors**
   Using very narrow priors can bias results if the priors are wrong. Example: Assuming all stars are thin disk can bias age-metallicity estimates for thick disk/halo stars.

**Ignoring prior volume effects**
   Distance priors have a :math:`d^2` volume factor. Forgetting this can lead to incorrect posterior normalization.

**Applying cluster priors to field stars**
   Field stars have broad age/metallicity distributions. Using isochrone-like priors for field stars produces biased results.

**Mismatched dust maps**
   Bayestar is calibrated for certain tracers and distances. Extrapolating beyond map limits or to very nearby stars can introduce errors.

Summary
-------

brutus uses physically motivated priors based on:

- **Kroupa IMF**: Mass distribution at formation
- **3-D Galactic structure**: Thin disk, thick disk, halo spatial densities
- **Metallicity gradients**: Component-dependent [Fe/H] distributions
- **Age structure**: Component-dependent age distributions
- **3-D dust maps**: Bayestar distance-dependent extinction
- **R_V variation**: Gaussian around R_V ~ 3.3

Priors can be **customized or disabled** for specific science cases, but default priors are validated against Galactic observations and appropriate for most field star applications.

Next Steps
----------

- Learn about cluster modeling: :doc:`cluster_modeling`
- Understand fitting results: :doc:`understanding_results`
- Configure fitting options: :doc:`choosing_options`

References
----------

Priors and Galactic Structure:

- Kroupa (2001), "The Initial Mass Function of Simple and Composite Populations", MNRAS, 322, 231
- Green et al. (2019), "A 3D Dust Map Based on Gaia, Pan-STARRS 1, and 2MASS", ApJ, 887, 93
- Jurić et al. (2008), "The Milky Way Tomography with SDSS", ApJ, 673, 864

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227
