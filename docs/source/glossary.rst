Glossary
========

Quick reference for key terms used throughout the brutus documentation.

.. glossary::
   :sorted:

Stellar Evolution
-----------------

.. glossary::

   EEP
   Equivalent Evolutionary Point
      A dimensionless index (typically 202-808) that parameterizes a star's evolutionary state independent of mass. Key values: 202 (pre-main sequence), 353 (:term:`ZAMS`), 454 (:term:`TAMS`), 605 (base of RGB). See :doc:`stellar_models` for details.

   ZAMS
   Zero-Age Main Sequence
      The point when a star begins hydrogen fusion in its core (EEP ~ 353). Marks the start of the main sequence phase.

   TAMS
   Terminal-Age Main Sequence
      The point when core hydrogen is exhausted (EEP ~ 454). The star then evolves off the main sequence toward the subgiant and red giant phases.

   isochrone
      A curve in the HR diagram connecting stars of the *same age* but different masses. Used for modeling stellar clusters where stars formed together. Compare with :term:`evolutionary track`.

   evolutionary track
      The path a star of fixed initial mass traces through the HR diagram as it evolves. Tracks are parameterized by :term:`EEP`. Perpendicular to :term:`isochrones <isochrone>`.

   initial mass
      The mass of a star at formation, in solar masses (M☉). Combined with age and metallicity, this determines all other stellar properties.

   MIST
      MESA Isochrones and Stellar Tracks - the stellar evolution models used by brutus. Covers 0.1-300 M☉, [Fe/H] from -4.0 to +0.5, all evolutionary phases from pre-MS through AGB.

   IMF
   Initial Mass Function
      The probability distribution of stellar masses at birth. brutus uses the Kroupa IMF: P(M) ∝ M⁻¹·³ for 0.08-0.5 M☉, P(M) ∝ M⁻²·³ for 0.5-150 M☉.

Chemical Abundances
-------------------

.. glossary::

   metallicity
   [Fe/H]
      The logarithmic iron abundance relative to solar: [Fe/H] = log₁₀(N_Fe/N_H) - log₁₀(N_Fe/N_H)☉. Solar is [Fe/H] = 0; metal-poor stars have [Fe/H] < 0.

   alpha enhancement
   [α/Fe]
      The abundance of alpha-process elements (O, Mg, Si, Ca, Ti) relative to iron, compared to solar. Old halo and thick disk stars typically have [α/Fe] ~ +0.3. *Note: Alpha enhancement is not currently exposed as a user-facing parameter in brutus; models assume solar-scaled abundances.*

Photometry
----------

.. glossary::

   magnitude
      Logarithmic brightness scale: m = -2.5 log₁₀(F/F_ref). Fainter objects have larger (more positive) magnitudes. A difference of 5 magnitudes corresponds to a factor of 100 in brightness.

   flux density
      Linear brightness measurement. brutus uses "maggies" internally, where 1 maggie is the flux of a 0th magnitude source. Convert from magnitudes: flux = 10^(-0.4 × mag).

   SED
   Spectral Energy Distribution
      The distribution of flux across wavelengths or photometric bands. brutus predicts SEDs from stellar models and compares them to observed photometry.

   parallax
      The apparent angular shift of a star due to Earth's orbital motion, measured in milliarcseconds (mas). Distance in parsecs = 1000 / parallax_mas.

   distance modulus
      The difference between apparent and absolute magnitude: μ = m - M = 5 log₁₀(d / 10 pc). A star at 1 kpc has μ = 10 mag.

Extinction & Dust
-----------------

.. glossary::

   extinction
      Wavelength-dependent attenuation of starlight by interstellar dust, in magnitudes. A_V denotes extinction in the V-band (~550 nm). Extinction is stronger at shorter wavelengths.

   reddening
      The color change caused by wavelength-dependent extinction. Quantified as color excess E(B-V) = (B-V)_observed - (B-V)_intrinsic.

   R_V
      The ratio of total-to-selective extinction: R_V = A_V / E(B-V). Characterizes the dust grain size distribution. Typical values: ~3.1 (diffuse ISM), ~2 (dense molecular clouds), ~5 (diffuse high-latitude).

   reddening vector
      The direction a star moves in color-color or color-magnitude space due to dust extinction. Depends on the extinction law and stellar spectrum.

Statistical Inference
---------------------

.. glossary::

   posterior
      The probability distribution of model parameters given the observed data: P(θ|data) ∝ P(data|θ) × P(θ). This is what brutus computes for each star.

   prior
      Probability distribution encoding knowledge before seeing the data: P(θ). In brutus: :term:`IMF`, Galactic structure, dust maps. See :doc:`priors`.

   likelihood
      The probability of observing the data given model parameters: P(data|θ). Measures how well a model matches observations.

   marginalization
      Integrating over nuisance parameters to obtain distributions for parameters of interest. For example, marginalizing over stellar mass to get the distance posterior.

   brute force
      The grid-based inference approach that gives brutus its name: systematically evaluate the likelihood at all pre-computed grid points, apply priors, then marginalize. Avoids MCMC convergence issues and guarantees complete parameter space coverage.

brutus Classes
--------------

.. glossary::

   StarGrid
      A pre-computed grid of stellar models optimized for fast fitting. Stores absolute magnitudes at a 1 kpc reference distance plus reddening coefficients for each model.

   BruteForce
      The main fitting class for individual stars. Takes a :term:`StarGrid` and observed photometry, returns posterior samples for distance, extinction, and stellar parameters.

   EEPTracks
      Class for interpolating stellar parameters along evolutionary tracks. Provides predictions for any combination of mass, :term:`EEP`, and metallicity.

   Isochrone
      Class for generating :term:`isochrones <isochrone>` at specified age and metallicity. Used with :term:`StellarPop` for cluster modeling.

   StellarPop
      Class for generating synthetic photometry for stellar populations. Combines :term:`Isochrone` predictions with bolometric corrections and extinction.

.. note::
   **Model Coverage**

   MIST models in brutus cover: main sequence stars (0.1-300 M☉), subgiants, red giants, horizontal branch, and asymptotic giant branch stars.

   **Not covered**: white dwarfs, brown dwarfs (< 0.08 M☉), neutron stars, black holes, or very young pre-main-sequence stars (< 1 Myr).

See Also
--------

- :doc:`stellar_models` - Detailed explanation of MIST models and EEP
- :doc:`scientific_background` - Statistical framework
- :doc:`priors` - Prior probability distributions
- :ref:`available-filters` - Supported photometric filters
