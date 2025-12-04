Stellar Evolution Models
========================

This page explains the stellar evolution models used in **brutus** and how they connect intrinsic stellar properties to observable photometry. Understanding these models is essential for interpreting results and choosing appropriate modeling strategies.

MIST: MESA Isochrones and Stellar Tracks
-----------------------------------------

brutus uses **MIST (MESA Isochrones and Stellar Tracks)** v1.2 stellar evolution models as its foundation. MIST models are computed using the MESA (Modules for Experiments in Stellar Astrophysics) stellar evolution code and provide:

- **Evolutionary tracks**: How individual stars evolve over time
- **Isochrones**: Snapshots of coeval stellar populations at fixed age
- **Stellar parameters**: Mass, radius, temperature, luminosity, surface gravity
- **Atmospheric spectra**: Photospheric flux distributions across wavelengths

MIST models span a wide range of stellar parameters:

- **Initial mass**: 0.1 to 300 solar masses
- **Metallicity** [Fe/H]: -4.0 to +0.5 dex
- **Alpha enhancement** [α/Fe]: -0.2 to +0.6 dex
- **Ages**: ~1 Myr to 14 Gyr (depending on mass)

The models include full evolutionary phases from the pre-main-sequence through post-main-sequence evolution (red giants, horizontal branch, asymptotic giant branch for low-mass stars; blue loops and core collapse for massive stars).

.. seealso::
   MIST models are described in detail in:

   - Choi et al. (2016), "MIST 0. Methods for the Construction of Stellar Isochrones", ApJ, 823, 102
   - Dotter (2016), "MIST I. Solar-scaled Models", ApJS, 222, 8

EEP: Equivalent Evolutionary Point
-----------------------------------

A key innovation of MIST is the **EEP (Equivalent Evolutionary Point)** parameterization of stellar evolution. EEP is an integer index that tracks a star's evolutionary phase in a mass-independent way.

Why EEP?
^^^^^^^^

Traditional stellar evolution tracks are parameterized by age, but age is not a good coordinate for several reasons:

1. **Non-monotonic evolution**: Stars don't evolve smoothly in the H-R diagram—they can loop back and forth during complex evolutionary phases (e.g., blue loops in intermediate-mass stars).

2. **Mass-dependent timescales**: A 0.8 solar mass star spends ~15 Gyr on the main sequence, while a 5 solar mass star exhausts its hydrogen in ~100 Myr. Age is not a useful coordinate for comparing stars of different masses.

3. **Degeneracy**: Multiple evolutionary phases can have similar temperatures and luminosities at different ages, making age determination ambiguous from photometry alone.

EEP solves these problems by defining evolutionary phase relative to **primary equivalent points** that mark significant transitions in stellar structure:

- **EEP = 202**: Pre-main-sequence
- **EEP = 353**: Zero-age main sequence (ZAMS)
- **EEP = 454**: Terminal-age main sequence (TAMS) - hydrogen exhaustion in core
- **EEP = 605**: Base of red giant branch
- **EEP = 631**: Tip of red giant branch
- **EEP = 707**: Zero-age horizontal branch (low-mass stars)
- **EEP = 808**: Beginning of thermal pulses (AGB phase)

Between these primary points, EEP varies smoothly and monotonically, providing a well-defined coordinate system for stellar evolution.

Using EEP in brutus
^^^^^^^^^^^^^^^^^^^

In brutus, individual star models (``EEPTracks``) are parameterized by:

.. math::

   (M_{\rm init}, {\rm EEP}, [{\rm Fe/H}]_{\rm init}, [\alpha/{\rm Fe}]_{\rm init}) \rightarrow ({\rm age}, T_{\rm eff}, L, R, \log g, \ldots)

This allows brutus to:

- **Predict stellar parameters** for any combination of mass, evolutionary phase, and composition
- **Handle all evolutionary stages** uniformly without age-based degeneracies
- **Interpolate smoothly** across the full parameter space

For example, to get predictions for a 1.0 solar mass star at the terminal-age main sequence with solar metallicity:

.. code-block:: python

   from brutus.core import EEPTracks

   tracks = EEPTracks()
   params = tracks.get_predictions([1.0, 454, 0.0, 0.0])  # mass, EEP, [Fe/H], [α/Fe]

   log_age = params[0]    # log10(age in years)
   log_L = params[1]      # log10(luminosity in solar units)
   log_Teff = params[2]   # log10(effective temperature in K)
   log_g = params[3]      # log10(surface gravity in cm/s^2)

.. note::
   While EEP is the fundamental parameter in evolutionary tracks, users rarely need to work with EEP directly. The ``BruteForce`` fitter and population models handle EEP internally when fitting photometric data.

Isochrones vs Evolutionary Tracks
----------------------------------

brutus uses two complementary representations of stellar evolution:

Evolutionary Tracks (``EEPTracks``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Evolutionary tracks** follow individual stars as they evolve over time. They are parameterized by:

- Fixed initial mass :math:`M_{\rm init}`
- Fixed composition ([Fe/H], [α/Fe])
- Variable EEP (evolutionary phase)

Tracks answer the question: *"How does a star of given mass and composition evolve?"*

Use evolutionary tracks (via ``StarEvolTrack``) when:

- Modeling individual stars with unknown masses
- Generating custom photometry on-the-fly
- Exploring specific stellar evolutionary phases

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks)

   # Generate photometry for a 1.2 solar mass star
   sed, params, params2 = star.get_seds(
       mini=1.2, eep=400, feh=0.0, afe=0.0,
       av=0.1, dist=1000.0
   )

Isochrones (``Isochrone``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Isochrones** represent coeval stellar populations of fixed age and composition. They are parameterized by:

- Fixed age :math:`t_{\rm age}`
- Fixed composition ([Fe/H], [α/Fe])
- Variable mass :math:`M_{\rm init}` (or equivalently EEP)

Isochrones answer the question: *"What is the mass-luminosity-temperature distribution for a stellar population of given age and composition?"*

Use isochrones (via ``StellarPop``) when:

- Modeling stellar clusters or coeval populations
- Fitting ensemble photometry with shared age/metallicity
- Comparing observed color-magnitude diagrams to models

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Generate photometry for a 1 Gyr solar metallicity population
   seds, params1, params2 = pop.get_seds(
       feh=0.0, afe=0.0, loga=9.0,  # log10(1 Gyr) = 9.0
       av=0.1, dist=2000.0
   )

**Key Distinction**: Evolutionary tracks vary evolutionary phase at fixed mass, while isochrones vary mass at fixed age. Both representations are mathematically equivalent (related by the age-EEP-mass mapping) but serve different modeling purposes.

From Stellar Parameters to Photometry
--------------------------------------

Once stellar parameters (temperature, luminosity, surface gravity, composition) are known, brutus must convert them to observable photometry. This involves two steps:

1. Atmospheric Models: Bolometric Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Stellar atmosphere models** compute the emergent spectral flux distribution :math:`F_\lambda(T_{\rm eff}, \log g, [{\rm Fe/H}], [\alpha/{\rm Fe}])` as a function of wavelength. MIST uses a combination of:

- **ATLAS12**: LTE plane-parallel model atmospheres for most stars
- **SYNTHE**: Spectral synthesis code for computing detailed line lists

These models predict the photospheric spectrum, which can be integrated through filter transmission curves to get synthetic magnitudes:

.. math::

   M_{\rm band} = -2.5 \log_{10} \left( \frac{\int F_\lambda \, T_{\rm band}(\lambda) \, \lambda \, d\lambda}{\int F_{\lambda,{\rm ref}} \, T_{\rm band}(\lambda) \, \lambda \, d\lambda} \right)

where :math:`T_{\rm band}(\lambda)` is the filter transmission function and :math:`F_{\lambda,{\rm ref}}` is the reference spectrum (Vega for Vega magnitudes, constant :math:`F_\nu` for AB magnitudes).

The difference between synthetic magnitudes and bolometric magnitude is the **bolometric correction**:

.. math::

   {\rm BC}_{\rm band} = M_{\rm bol} - M_{\rm band}

2. Neural Network Approach: FastNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computing full spectral synthesis for every model evaluation would be prohibitively expensive. Instead, brutus uses **neural networks** (``FastNN``) to learn the mapping:

.. math::

   (T_{\rm eff}, \log g, [{\rm Fe/H}], [\alpha/{\rm Fe}]) \rightarrow \{{\rm BC}_{\rm band}\}

The neural networks are trained on a grid of MIST atmosphere models and provide:

- **Speed**: Orders of magnitude faster than full spectral synthesis
- **Accuracy**: Typical errors of a few millimagnitudes in bolometric corrections
- **Flexibility**: Support arbitrary filter combinations

The ``FastNNPredictor`` class handles loading trained networks and predicting bolometric corrections:

.. code-block:: python

   from brutus.core import FastNNPredictor

   # Initialize with specific filters
   nn_predictor = FastNNPredictor(filters=['g', 'r', 'i', 'z', 'y'])

   # Predict bolometric corrections
   stellar_params = [log_Teff, log_g, feh, afe]
   bc_values = nn_predictor.predict(stellar_params)

Extinction and Reddening
^^^^^^^^^^^^^^^^^^^^^^^^^

Interstellar dust modifies the observed photometry through wavelength-dependent extinction. brutus models this using **reddening vectors** :math:`\mathbf{R}` and :math:`\mathbf{R}'` that describe how extinction affects each photometric band:

.. math::

   m_{\rm band} = M_{\rm band} + \mu + A_V \times (R_{\rm band} + R_V \times R'_{\rm band})

The reddening vectors are pre-computed for each stellar model by:

1. Computing a fiducial extinction curve :math:`A_\lambda / A_V` (e.g., Cardelli, Clayton, & Mathis 1989)
2. Integrating the reddened spectrum through filter transmission curves
3. Computing the derivatives with respect to :math:`A_V` and :math:`R_V`

This allows brutus to model dust extinction efficiently without re-computing atmospheric spectra for different extinction values.

.. seealso::
   See :doc:`scientific_background` for the full extinction model including :math:`R_V` variation.

Grid Pre-computation vs On-the-Fly Models
------------------------------------------

brutus offers two strategies for generating model photometry:

Pre-computed Grids (``StarGrid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Advantages**:

- Extremely fast fitting (pre-computed reddening coefficients)
- Optimized for large samples (thousands to millions of stars)
- Includes distance scaling and extinction at reference distance (1 kpc)

**Disadvantages**:

- Fixed filter set (must regenerate grid for new filters)
- Large file sizes (several GB for comprehensive grids)
- Less flexible for custom stellar models

**When to use**: Large surveys with standardized filter sets (e.g., Gaia + 2MASS + WISE, Pan-STARRS, SDSS)

.. code-block:: python

   from brutus.core import StarGrid
   from brutus.data import load_models

   # Load pre-computed grid
   models, labels, label_mask = load_models('grid_mist_v9.h5')
   grid = StarGrid(models, labels, label_mask)

On-the-Fly Models (``StarEvolTrack``, ``StellarPop``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Advantages**:

- Flexible filter combinations (any set of available filters)
- Smaller memory footprint (compute as needed)
- Easy to modify or extend models

**Disadvantages**:

- Slower than pre-computed grids (neural network evaluations)
- Not optimized for large-scale fitting

**When to use**: Exploratory analysis, custom filter sets, cluster modeling, prototyping

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['g', 'r', 'i'])

.. seealso::
   See :doc:`grid_generation` for details on creating custom pre-computed grids.

Empirical Calibration
----------------------

Theoretical stellar models have known systematic errors, particularly:

- **Effective temperatures**: Models predict slightly wrong temperatures for M dwarfs
- **Radii**: Discrepancies between model and interferometric radii
- **Photometry**: Systematic offsets in specific filters due to incomplete line lists

brutus includes **empirical calibration** corrections derived from:

1. **Open clusters**: Fitting cluster sequences to derive temperature/radius corrections
2. **Field stars**: Comparing nearby low-reddening stars to models to estimate photometric offsets

These corrections are applied as optional parameters (``corr_params``) when generating models. See :doc:`photometric_offsets` for detailed discussion of calibration methodology and when to apply corrections.

Summary
-------

- **MIST models** provide comprehensive stellar evolution predictions across mass, age, and composition
- **EEP parameterization** enables smooth interpolation and handles all evolutionary phases uniformly
- **Evolutionary tracks** (variable EEP, fixed mass) are used for individual stars
- **Isochrones** (variable mass, fixed age) are used for stellar populations
- **Neural networks** provide fast bolometric corrections from stellar parameters to photometry
- **Pre-computed grids** optimize fitting speed; **on-the-fly models** provide flexibility

Next Steps
----------

- Learn about grid-based fitting: :doc:`grid_generation`
- Understand cluster modeling with isochrones: :doc:`cluster_modeling`
- Explore empirical calibration: :doc:`photometric_offsets`

References
----------

MIST Stellar Models:

- Choi et al. (2016), ApJ, 823, 102 - MIST construction methodology
- Dotter (2016), ApJS, 222, 8 - Solar-scaled MIST models
- Paxton et al. (2011, 2013, 2015, 2018, 2019) - MESA stellar evolution code

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227 - brutus methods and validation
