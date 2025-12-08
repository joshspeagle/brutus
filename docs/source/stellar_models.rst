Stellar Models and Photometry
=============================

This page explains the stellar evolution models used in ``brutus`` and how they connect intrinsic stellar properties to observable photometry.

.. tip::
   For quick definitions of key terms (EEP, isochrone, etc.), see the :doc:`glossary`.

MIST Stellar Evolution Models
-----------------------------

``brutus`` uses **MIST** (:term:`MESA Isochrones and Stellar Tracks <MIST>`) v1.2 as its foundation (`MIST homepage <https://waps.cfa.harvard.edu/MIST/>`_). MIST models are computed using the MESA stellar evolution code and provide:

- **Evolutionary tracks**: How individual stars evolve over time
- **Isochrones**: Snapshots of coeval stellar populations at fixed age
- **Stellar parameters**: Mass, radius, temperature, luminosity, surface gravity
- **Bolometric corrections**: Synthetic photometry in many filter systems

Parameter Coverage
^^^^^^^^^^^^^^^^^^

MIST models span:

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Initial mass**
     - 0.1 to 300 M☉
   * - **Metallicity [Fe/H]**
     - -4.0 to +0.5 dex
   * - **Ages**
     - ~1 Myr to 14 Gyr (mass-dependent)

The models include all evolutionary phases: pre-main-sequence, main sequence, subgiant, red giant branch, horizontal branch, and asymptotic giant branch (for low/intermediate mass stars).

EEP: Equivalent Evolutionary Point
----------------------------------

A key feature of MIST is the :term:`EEP` (Equivalent Evolutionary Point) parameterization. EEP is an integer index that tracks evolutionary phase in a mass-independent way.

Why Not Use Age Directly?
^^^^^^^^^^^^^^^^^^^^^^^^^

Age is problematic as a stellar evolution coordinate:

1. **Non-monotonic evolution**: Stars loop back and forth in the H-R diagram during complex phases
2. **Mass-dependent timescales**: A 0.8 M☉ star lives ~15 Gyr on the main sequence; a 5 M☉ star exhausts hydrogen in ~100 Myr
3. **Degeneracies**: Multiple evolutionary phases produce similar temperatures and luminosities

EEP solves these problems by defining phase relative to physical transitions in stellar structure.

Key EEP Values
^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - EEP
     - Phase
     - Description
   * - 202
     - Pre-main-sequence
     - Contracting toward main sequence
   * - 353
     - :term:`ZAMS`
     - Zero-age main sequence (hydrogen fusion begins)
   * - 454
     - :term:`TAMS`
     - Terminal-age main sequence (core hydrogen exhausted)
   * - 605
     - Base RGB
     - Beginning of red giant branch ascent
   * - 631
     - Tip RGB
     - Maximum RGB luminosity before helium flash
   * - 707
     - ZAHB
     - Zero-age horizontal branch (helium core burning)
   * - 808
     - TP-AGB
     - Thermal pulses on asymptotic giant branch

EEP varies smoothly between these primary points, providing a well-defined coordinate for interpolation.

Using EEP in ``brutus``
^^^^^^^^^^^^^^^^^^^^^^^

``brutus`` parameterizes stellar models as:

.. math::

   (M_{\rm init}, {\rm EEP}, [{\rm Fe/H}]) \rightarrow (T_{\rm eff}, L, R, \log g, {\rm age}, \ldots)

This allows prediction of stellar parameters for any combination of mass, evolutionary phase, and metallicity:

.. code-block:: python

   from brutus.core import EEPTracks

   tracks = EEPTracks()

   # Get predictions for a 1.0 solar mass star at TAMS with solar metallicity
   params = tracks.get_predictions([1.0, 454, 0.0])
   # Returns: [log_age, log_L, log_Teff, log_g, ...]

   print(f"Age: {10**params[0] / 1e9:.2f} Gyr")
   print(f"Teff: {10**params[2]:.0f} K")
   print(f"log(g): {params[3]:.2f}")

.. note::
   Users rarely need to work with EEP directly. The ``BruteForce`` fitter handles EEP internally when fitting photometric data. EEP becomes important when generating custom models or interpreting detailed results.

Evolutionary Tracks vs Isochrones
---------------------------------

``brutus`` provides two complementary representations:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * -
     - **Evolutionary Tracks**
     - **Isochrones**
   * - **Fixed**
     - Mass, composition
     - Age, composition
   * - **Variable**
     - EEP (evolutionary phase)
     - Mass
   * - **Question answered**
     - How does a star of given mass evolve?
     - What masses exist at a given age?
   * - **Use case**
     - Individual field stars
     - Stellar clusters

Evolutionary Tracks (``EEPTracks``, ``StarEvolTrack``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use tracks when modeling individual stars with unknown masses:

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW'])

   # Generate photometry for a specific stellar model
   result = star.get_seds(
       mini=1.2,       # Initial mass (solar masses)
       eep=400,        # Main sequence
       feh=0.0,        # Solar metallicity
       av=0.1,         # V-band extinction
       dist=1000.0     # Distance in parsecs
   )

   seds, params1, params2 = result
   # seds: flux densities in each filter
   # params1: input parameters (mini, eep, feh)
   # params2: derived parameters (Teff, logg, age, etc.)

Isochrones (``Isochrone``, ``StellarPop``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :term:`isochrones <isochrone>` when modeling coeval populations (clusters):

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   iso = Isochrone()
   pop = StellarPop(isochrone=iso, filters=['Gaia_G_MAW', 'PS_g', 'PS_r', '2MASS_J'])

   # Generate photometry for a 1 Gyr, solar metallicity population
   result = pop.get_seds(
       feh=0.0,        # Metallicity
       loga=9.0,       # log10(age/yr) = 9.0 → 1 Gyr
       av=0.1,         # V-band extinction
       dist=2000.0     # Distance in parsecs
   )

   seds, params1, params2 = result
   # seds: (N_stars, N_filters) array of flux densities along the isochrone
   # params1: stellar parameters for each point
   # params2: derived quantities

From Parameters to Photometry
-----------------------------

Converting stellar parameters to observable magnitudes requires two steps:

Bolometric Corrections
^^^^^^^^^^^^^^^^^^^^^^

Stellar atmosphere models (ATLAS12/SYNTHE) predict spectral flux distributions :math:`F_\lambda(T_{\rm eff}, \log g, [{\rm Fe/H}])`. These are integrated through filter transmission curves to yield synthetic magnitudes:

.. math::

   M_{\rm band} = M_{\rm bol} - {\rm BC}_{\rm band}(T_{\rm eff}, \log g, [{\rm Fe/H}])

where BC is the **bolometric correction** for each filter.

Neural Network Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computing full spectral synthesis at every model evaluation would be prohibitively slow. ``brutus`` uses trained neural networks (``FastNN``) to predict bolometric corrections:

.. math::

   (T_{\rm eff}, \log g, [{\rm Fe/H}]) \xrightarrow{\text{neural net}} \{{\rm BC}_{\rm band}\}

This provides:

- **Speed**: Orders of magnitude faster than full spectral synthesis
- **Accuracy**: Sufficient for photometric precision of typical surveys
- **Flexibility**: Any combination of supported filters

Extinction Modeling
^^^^^^^^^^^^^^^^^^^

Interstellar dust modifies photometry through wavelength-dependent :term:`extinction`. ``brutus`` models this with :term:`reddening vectors <reddening vector>`:

.. math::

   m_{\rm band} = M_{\rm band} + \mu + A_V \times \left( R_{\rm band} + R_V \times R'_{\rm band} \right)

where :math:`R_{\rm band}` and :math:`R'_{\rm band}` encode how extinction affects each filter for a given stellar spectrum.

The reddening vectors are pre-computed using the Fitzpatrick & Massa (2009) extinction curve. This extinction law is parameterized by :term:`R_V` and is baked into the neural network predictions, allowing efficient evaluation of reddened magnitudes for any :math:`(A_V, R_V)` combination.

.. seealso::
   See :doc:`priors` for the default R_V prior distribution, which is based on Schlafly et al. (2016).

.. _available-filters:

Available Photometric Filters
-----------------------------

``brutus`` supports these photometric systems through its neural network models:

**Space-Based**

- **Gaia DR3**: ``Gaia_G_MAW``, ``Gaia_BP_MAWf``, ``Gaia_RP_MAW``
- **HST ACS/WFC**: ``ACS_WFC_F475W``, ``ACS_WFC_F606W``, ``ACS_WFC_F814W``
- **HST WFC3/UVIS**: ``WFC3_UVIS_F275W``, ``WFC3_UVIS_F336W``, ``WFC3_UVIS_F438W``
- **HST WFC3/IR**: ``WFC3_IR_F110W``, ``WFC3_IR_F125W``, ``WFC3_IR_F160W``
- **WISE**: ``WISE_W1``, ``WISE_W2``, ``WISE_W3``, ``WISE_W4``

**Ground-Based Optical**

- **Pan-STARRS**: ``PS_g``, ``PS_r``, ``PS_i``, ``PS_z``, ``PS_y``
- **SDSS**: ``SDSS_u``, ``SDSS_g``, ``SDSS_r``, ``SDSS_i``, ``SDSS_z``
- **DECam**: ``DECam_g``, ``DECam_r``, ``DECam_i``, ``DECam_z``, ``DECam_Y``
- **Johnson-Cousins**: ``Bessell_U``, ``Bessell_B``, ``Bessell_V``, ``Bessell_R``, ``Bessell_I``

**Ground-Based Near-IR**

- **2MASS**: ``2MASS_J``, ``2MASS_H``, ``2MASS_Ks``

Pre-computed Grids vs On-the-Fly Models
---------------------------------------

``brutus`` offers two strategies:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * -
     - **Pre-computed Grids**
     - **On-the-Fly Models**
   * - **Class**
     - ``StarGrid``
     - ``StarEvolTrack``, ``StellarPop``
   * - **Speed**
     - Very fast (~ms per star)
     - Slower (~100 ms per evaluation)
   * - **Flexibility**
     - Fixed filter set
     - Any filter combination
   * - **Memory**
     - Large (several GB)
     - Minimal
   * - **Best for**
     - Large surveys, production fitting
     - Exploration, custom filters, clusters

**Pre-computed grids** (for large-scale fitting):

.. code-block:: python

   from brutus.core import StarGrid
   from brutus.data import load_models

   models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)
   grid = StarGrid(models, labels, label_mask)

**On-the-fly models** (for flexibility):

.. code-block:: python

   from brutus.core import EEPTracks, StarEvolTrack

   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks, filters=['Gaia_G_MAW', '2MASS_J', '2MASS_Ks'])

.. seealso::
   See :doc:`grid_generation` for creating custom pre-computed grids.

Model Limitations
-----------------

MIST models have known limitations:

- **Non-rotating**: Rotation affects stellar structure and lifetimes, especially for massive stars
- **Single stars**: Binary evolution (mass transfer, mergers) not included
- **Solar-scaled abundances**: No alpha-enhancement or individual element variations
- **M dwarf temperatures**: Models predict slightly incorrect temperatures for low-mass stars
- **Radius inflation**: Magnetic activity inflates radii of active low-mass stars by 5–15%

For empirical corrections to some of these issues, see :doc:`photometric_offsets`.

References
----------

**MIST Stellar Models:**

- Choi et al. (2016), "Mesa Isochrones and Stellar Tracks (MIST). I. Solar-scaled Models", `ApJ, 823, 102 <https://ui.adsabs.harvard.edu/abs/2016ApJ...823..102C>`_
- Dotter (2016), "MESA Isochrones and Stellar Tracks (MIST) 0: Methods for the Construction of Stellar Isochrones", `ApJS, 222, 8 <https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D>`_

**MESA Stellar Evolution Code:**

- Paxton et al. (2011, 2013, 2015, 2018, 2019), "Modules for Experiments in Stellar Astrophysics (MESA)", `ApJS series <https://ui.adsabs.harvard.edu/abs/2019ApJS..243...10P>`_

**Extinction Law:**

- Fitzpatrick & Massa (2009), "An Analysis of the Shapes of Interstellar Extinction Curves. VI. The Near-IR Extinction Law", `ApJ, 699, 1209 <https://ui.adsabs.harvard.edu/abs/2009ApJ...699.1209F>`_

**brutus Implementation:**

- Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_
