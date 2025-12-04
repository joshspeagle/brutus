Glossary
========

Quick reference for key terms used throughout the brutus documentation.

Stellar Parameters
------------------

**EEP** (Equivalent Evolutionary Point)
   A dimensionless index (typically 202-808) that parameterizes stellar evolution. EEP 202 is the zero-age main sequence (ZAMS); EEP 454 is the terminal-age main sequence (TAMS); higher values trace post-main-sequence evolution.

**Initial Mass** (M_init)
   The mass of a star at birth, in solar masses (M_sun). This is an *intrinsic* parameter that, combined with age and metallicity, determines a star's current properties.

**Isochrone**
   A curve in the HR diagram representing stars of the *same age* but different masses. Used for modeling stellar clusters where all stars formed together.

**Evolutionary Track**
   The path a star of fixed initial mass traces through the HR diagram as it ages. Perpendicular to isochrones.

**[Fe/H]** (Metallicity)
   The logarithmic iron abundance relative to solar: [Fe/H] = log(N_Fe/N_H) - log(N_Fe/N_H)_sun. Solar metallicity is [Fe/H] = 0; metal-poor stars have [Fe/H] < 0.

**[α/Fe]** (Alpha Enhancement)
   The abundance of alpha-process elements (O, Mg, Si, Ca, Ti) relative to iron. Halo and thick disk stars typically have [α/Fe] > 0.

**MIST**
   The MESA Isochrones and Stellar Tracks library, the default stellar models in brutus. Covers 0.1-300 M_sun, [Fe/H] -4 to +0.5.

Extinction and Reddening
------------------------

**Extinction** (A_λ)
   Wavelength-dependent attenuation of starlight by interstellar dust, measured in magnitudes. A_V is extinction in the V-band.

**Reddening** (E(B-V))
   The *color excess* caused by dust: E(B-V) = (B-V)_observed - (B-V)_intrinsic. Related to extinction by R_V.

**R_V**
   The ratio of total-to-selective extinction: R_V = A_V / E(B-V). Typical value ~3.1 for diffuse ISM; varies from ~2 (dense clouds) to ~5 (diffuse high-latitude).

**Reddening Vector**
   The direction and magnitude of color changes caused by dust extinction in multi-band photometry. Depends on dust properties and stellar spectrum.

**Distance Modulus** (μ)
   The difference between apparent and absolute magnitude: μ = m - M = 5 log(d/10 pc). A star at 1 kpc has μ = 10 mag.

Data and Observations
---------------------

**Magnitude**
   Logarithmic brightness scale: m = -2.5 log(F/F_ref). Fainter objects have larger magnitudes. brutus uses the AB system internally.

**Flux Density**
   Linear brightness measurement, typically in "maggies" (flux relative to a reference source). brutus converts magnitudes to flux internally for likelihood calculations.

**Parallax** (ϖ)
   The apparent angular shift of a star due to Earth's orbital motion, measured in milliarcseconds (mas). Distance in parsecs = 1000/parallax_mas.

**Photometry**
   Brightness measurements through bandpass filters. In brutus, typically refers to apparent magnitudes or flux densities.

**SED** (Spectral Energy Distribution)
   The distribution of flux across wavelength or photometric bands. brutus predicts SEDs from stellar models.

Methods and Algorithms
----------------------

**Brute Force**
   The grid-based inference approach that gives brutus its name: systematically evaluate likelihood at all grid points, then marginalize. Avoids MCMC convergence issues.

**Posterior**
   The probability distribution of parameters given the data: P(θ|data) ∝ P(data|θ) × P(θ). What brutus computes for each star.

**Prior**
   Background knowledge encoded as probability: P(θ). In brutus: IMF, Galactic structure, age-metallicity relation, dust distribution.

**Likelihood**
   The probability of observing the data given parameters: P(data|θ). Measures how well a model matches observations.

**Marginalization**
   Integrating out nuisance parameters to get distributions over parameters of interest. E.g., marginalize over stellar mass to get distance posterior.

**StarGrid**
   Pre-computed grid of stellar models for fast fitting. Contains absolute magnitudes and reddening coefficients at 1 kpc reference distance.

**BruteForce**
   The main fitting class in brutus. Takes a StarGrid and observed data, returns posterior samples for distance, extinction, and stellar parameters.

Model Coverage
--------------

**Covered by MIST**:
   Main sequence (0.1-300 M_sun), subgiants, red giants, horizontal branch, asymptotic giant branch

**Not Covered**:
   White dwarfs, brown dwarfs (<0.08 M_sun), neutron stars, black holes, pre-main-sequence (<1 Myr)

See Also
--------

- :doc:`scientific_background` - Statistical framework details
- :doc:`stellar_models` - MIST model physics
- :doc:`priors` - Prior distributions explained
- :doc:`faq` - Common questions answered
