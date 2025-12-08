Bayesian Inference Framework
============================

This page provides an overview of the statistical framework underlying ``brutus``. For detailed mathematical derivations, see `Speagle et al. (2025) <https://arxiv.org/abs/2503.02227>`_.

.. tip::
   For quick definitions of key terms, see the :doc:`glossary`.

Overview
--------

``brutus`` performs :term:`Bayesian inference <posterior>` to derive stellar properties from photometric and astrometric observations. Given multi-band photometry and (optionally) parallax measurements, ``brutus`` estimates:

- **Distance** and **extinction** (:math:`A_V`, :math:`R_V`)
- **Stellar parameters**: mass, age, metallicity, temperature, luminosity, surface gravity
- **Full posterior distributions** with properly quantified uncertainties

The name "brutus" reflects the :term:`brute force` computational approach: rather than using MCMC sampling, ``brutus`` systematically evaluates the posterior over a pre-computed grid of stellar models, then marginalizes to obtain parameter estimates. This guarantees complete parameter space coverage and avoids convergence issues. See :doc:`grid_generation` for details on the computational approach.

The Forward Model
-----------------

``brutus`` uses a forward modeling approach: predict what observations *should* look like given stellar parameters, then compare to actual data.

Stellar Parameters
^^^^^^^^^^^^^^^^^^

The model separates **intrinsic parameters** (properties of the star itself) from **extrinsic parameters** (environmental effects):

**Intrinsic parameters** :math:`\theta`:

- :math:`M_{\rm init}`: :term:`Initial mass <initial mass>` (solar masses)
- :math:`{\rm EEP}`: :term:`Equivalent Evolutionary Point <EEP>` (evolutionary state)
- :math:`[{\rm Fe/H}]`: :term:`Metallicity <metallicity>` (dex)

**Extrinsic parameters** :math:`\phi`:

- :math:`d`: Distance (parsecs)
- :math:`A_V`: V-band :term:`extinction` (magnitudes)
- :math:`R_V`: Extinction curve shape parameter

.. note::
   ``brutus`` parameterizes stellar evolution using :term:`EEP` rather than age. EEP provides a mass-independent coordinate that increases monotonically along evolutionary tracks, avoiding degeneracies where multiple ages produce similar observables. Age is a *derived* quantity computed from (mass, EEP, metallicity). See :doc:`stellar_models` for details.

Predicted Magnitudes
^^^^^^^^^^^^^^^^^^^^

The predicted apparent magnitude in each photometric band is:

.. math::

   m_{\rm band} = M_{\rm band}(\theta) + \mu(d) + A_V \times \left[ R_{\rm band}(\theta) + R_V \times R'_{\rm band}(\theta) \right]

where:

- :math:`M_{\rm band}(\theta)` is the absolute magnitude from stellar models
- :math:`\mu(d) = 5 \log_{10}(d/10\,{\rm pc})` is the :term:`distance modulus`
- :math:`R_{\rm band}` and :math:`R'_{\rm band}` are :term:`reddening vectors <reddening vector>` encoding wavelength-dependent extinction

The reddening vectors depend on both the dust extinction law and the stellar spectrum, allowing accurate extinction modeling across different stellar types.

Model Assumptions
^^^^^^^^^^^^^^^^^

The forward model assumes:

- **Single stars**: No unresolved companions (binaries bias distances and masses)
- **Non-rotating**: MIST models do not include rotational effects
- **Solar-scaled abundances**: Detailed abundance patterns beyond [Fe/H] are not modeled
- **Fitzpatrick & Massa (2009) extinction law**: R_V-dependent parameterization

These assumptions are appropriate for most field stars but may break down for rapid rotators, chemically peculiar stars, or close binaries.

Statistical Framework
---------------------

Observation Model
^^^^^^^^^^^^^^^^^

Real observations include measurement uncertainties. ``brutus`` models:

**Photometry** as Gaussian in flux space:

.. math::

   \hat{F}_{\rm band} \sim \mathcal{N}\left[ F_{\rm band}, \sigma_{F,{\rm band}}^2 \right]

where :math:`F = 10^{-0.4\,m}` converts magnitudes to linear :term:`flux density` (in "maggies", where 1 maggie = flux of a 0th magnitude source). Working in flux space ensures Gaussian errors—magnitude uncertainties are asymmetric and non-Gaussian.

**Parallax** as Gaussian:

.. math::

   \hat{\varpi} \sim \mathcal{N}\left[ 1000/d, \sigma_\varpi^2 \right]

where :math:`\varpi` is in milliarcseconds and :math:`d` in parsecs. Parallax provides a direct distance constraint that helps break the distance-extinction degeneracy.

Posterior Distribution
^^^^^^^^^^^^^^^^^^^^^^

Bayes' theorem combines the likelihood with prior information:

.. math::

   P(\theta, \phi \,|\, \hat{\mathbf{F}}, \hat{\varpi}) \propto \underbrace{\mathcal{L}_{\rm phot}(\theta, \phi)}_{\text{photometric likelihood}} \times \underbrace{\mathcal{L}_{\rm plx}(\phi)}_{\text{parallax likelihood}} \times \underbrace{\pi(\theta, \phi)}_{\text{prior}}

The :term:`posterior` represents our knowledge of stellar parameters given the observed data and astrophysical priors.

Prior Distributions
^^^^^^^^^^^^^^^^^^^

``brutus`` uses informative :term:`priors <prior>` based on Galactic structure:

.. math::

   \pi(\theta, \phi) = \pi(M_{\rm init}) \times \pi({\rm EEP}) \times \pi([{\rm Fe/H}], t_{\rm age} \,|\, d, \ell, b) \times \pi(A_V \,|\, d, \ell, b) \times \pi(R_V)

The components are:

1. **Initial Mass Function**: Kroupa :term:`IMF` favoring low-mass stars
2. **EEP prior**: Uniform in EEP (accounts for varying evolutionary timescales)
3. **Galactic structure**: 3-D density model (thin disk + thick disk + halo) with position-dependent age and metallicity distributions
4. **3-D dust**: Extinction constraints from Bayestar dust maps
5. **R_V variation**: Gaussian prior centered on :math:`R_V = 3.32` with :math:`\sigma = 0.18`

These priors encode that: low-mass stars are more common than high-mass stars; nearby stars are likely disk members with near-solar metallicity; distant stars may be older halo members; and extinction increases with distance along dusty sightlines.

.. seealso::
   See :doc:`priors` for detailed descriptions and customization options.

Outputs
-------

For each fitted star, ``brutus`` produces:

- **Posterior samples**: Draws from P(distance, :math:`A_V`, :math:`R_V` | data)
- **Model indices**: Which grid models contribute to the posterior (for deriving mass, age, Teff, etc.)
- **Diagnostics**: Minimum χ², log-evidence, number of bands used

See :doc:`understanding_results` for interpretation guidance.

Limitations
-----------

**Model Systematics**

Stellar models have known systematic uncertainties:

- *Temperature scale*: MIST temperatures may be offset by 50–200 K depending on mass and evolutionary state
- *Radius inflation*: Magnetic activity inflates radii of low-mass stars by 5–15%
- *Bolometric corrections*: Missing opacities cause ~0.02–0.05 mag errors in some bands

See :doc:`photometric_offsets` for empirical calibration procedures.

**Prior Sensitivity**

For well-measured stars (bright, accurate parallax), priors have minimal impact. For poorly-measured stars (faint, no parallax), results can be prior-dominated. Always check prior sensitivity for your science case.

**Binary Contamination**

Unresolved binaries appear overluminous, biasing distances (too near) and masses (too high). Gaia RUWE > 1.4 often indicates binarity.

**Computational Cost**

Fitting scales with grid size and number of photometric bands. Typical performance: ~0.1–1 second per star with pre-computed grids. Large surveys benefit from parallelization.

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_

Choi et al. (2016), "Mesa Isochrones and Stellar Tracks (MIST). I. Solar-scaled Models", `ApJ, 823, 102 <https://ui.adsabs.harvard.edu/abs/2016ApJ...823..102C>`_

Dotter (2016), "MESA Isochrones and Stellar Tracks (MIST) 0: Methods for the Construction of Stellar Isochrones", `ApJS, 222, 8 <https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D>`_

Green et al. (2019), "A 3D Dust Map Based on Gaia, Pan-STARRS 1, and 2MASS", `ApJ, 887, 93 <https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G>`_

Fitzpatrick & Massa (2009), "An Analysis of the Shapes of Interstellar Extinction Curves. VI. The Near-IR Extinction Law", `ApJ, 699, 1209 <https://ui.adsabs.harvard.edu/abs/2009ApJ...699.1209F>`_
