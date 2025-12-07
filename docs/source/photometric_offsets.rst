Photometric Offsets and Empirical Calibration
==============================================

This page explains the **empirical calibration** procedures used in brutus to correct systematic errors in theoretical stellar models. Understanding these corrections is important for achieving accurate stellar parameter estimates.

Why Empirical Calibration?
---------------------------

Theoretical stellar evolution models like MIST are based on fundamental physics, but they are not perfect. Known issues include:

**Effective Temperature Systematics**
   Models predict temperatures that are systematically offset from observations, particularly for M dwarfs. Interferometric measurements and empirical color-temperature relations show discrepancies of 100-300 K for cool stars.

**Radius Discrepancies**
   Theoretical radii disagree with interferometric measurements, especially for low-mass stars and evolved giants. Errors can reach 10-20% for M dwarfs.

**Photometric Systematics**
   Even with correct temperatures and radii, synthetic photometry can have systematic offsets due to:

   - Incomplete opacity tables (missing molecular lines)
   - Simplified atmospheric modeling (1D, LTE assumptions)
   - Uncertain bolometric corrections in certain wavelength regimes
   - Differences between model and observed photometric systems

**Impact on Results**
   Without corrections, these systematics propagate into:

   - Biased distance estimates (up to 10-20% errors)
   - Incorrect extinction measurements
   - Systematic errors in derived masses, ages, metallicities

Empirical calibration uses observations of well-characterized stars to **measure and correct** these systematic offsets.

Two Types of Corrections
-------------------------

brutus implements two complementary empirical corrections:

1. Isochrone Corrections (Temperature and Radius)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Fix systematic errors in fundamental stellar parameters (T_eff, radius)

**Method**: Use open clusters with well-known properties to derive corrections

**Application**: Modify the stellar models *before* computing photometry

**When to use**: Always recommended for main-sequence stars with accurate cluster calibrations

2. Photometric Offsets
^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Fix systematic errors in synthetic photometry for specific filters

**Method**: Compare observed and predicted photometry for nearby, low-reddening field stars

**Application**: Add offsets to model magnitudes in specific photometric bands

**When to use**: When fitting with specific survey data (e.g., Pan-STARRS, Gaia, SDSS)

.. note::
   These corrections are *complementary*. Isochrone corrections fix the stellar models, while photometric offsets fix the remaining systematic errors in converting models to observed magnitudes.

Isochrone Corrections: Open Cluster Approach
---------------------------------------------

Overview
^^^^^^^^

Open clusters provide ideal calibration targets because:

- **Known distances**: Often from parallax, kinematics, or eclipsing binaries
- **Known ages**: Isochrone fitting with multiple evolutionary phases
- **Known metallicities**: Spectroscopic measurements for many members
- **Low extinction**: Many nearby clusters have minimal reddening
- **Rich sequences**: Wide mass range from turnoff to lower main sequence

By comparing observed cluster sequences to theoretical isochrones, we can identify systematic offsets and derive empirical corrections.

Correction Parameterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus uses two linear correction parameters:

**Temperature correction** (``dtdm``):

.. math::

   T_{\rm eff,corrected} = T_{\rm eff,model} + {\rm dtdm} \times (M - M_{\rm TO})

where :math:`M` is stellar mass and :math:`M_{\rm TO}` is the turnoff mass for the isochrone. This corrects the temperature as a function of mass relative to the turnoff.

**Radius correction** (``drdm``):

.. math::

   R_{\rm corrected} = R_{\rm model} + {\rm drdm} \times (M - M_{\rm TO})

These mass-dependent corrections allow different adjustments for different stellar masses while maintaining smooth continuity along the isochrone.

**Additional parameters**:

- ``msto_smooth``: Smoothing scale near main sequence turnoff (avoids discontinuities)
- ``feh_scale``: Metallicity-dependent scaling of corrections

Typical Correction Values
^^^^^^^^^^^^^^^^^^^^^^^^^^

From cluster calibrations:

- **dtdm**: ~100-300 K / solar mass for lower main sequence
- **drdm**: ~0.05-0.15 R_sun / solar mass for lower main sequence
- **Effect**: Primarily impacts M and K dwarfs; minimal effect on FGK giants

Applying Isochrone Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrections are applied during model generation:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Define correction parameters from cluster calibration
   corr_params = [
       100.0,   # dtdm: Temperature correction (K/Msun)
       0.08,    # drdm: Radius correction (Rsun/Msun)
       0.05,    # msto_smooth: Smoothing scale (Msun)
       1.0      # feh_scale: Metallicity scaling
   ]

   # Initialize population model with corrections
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

   # Generate photometry with corrections
   seds, params1, params2 = pop.get_seds(
       feh=0.0, loga=9.0, av=0.1, dist=1000.0,
       corr_params=corr_params
   )

For grid generation:

.. code-block:: python

   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   generator = GridGenerator(tracks, filters=['g', 'r', 'i'])

   generator.make_grid(
       output_file='corrected_grid.h5',
       corr_params=corr_params
   )

Photometric Offsets: Field Star Approach
-----------------------------------------

Overview
^^^^^^^^

Even with isochrone corrections, residual photometric systematics remain. **Photometric offsets** measure the average difference between observed and model magnitudes for well-characterized field stars.

Selection Criteria
^^^^^^^^^^^^^^^^^^

Ideal calibration stars are:

- **Nearby**: d < 100 pc (parallax-based distances)
- **Low extinction**: A_V < 0.1 mag (avoid reddening uncertainties)
- **Well-measured**: High S/N photometry, accurate parallax
- **Main sequence**: Avoid evolved stars with uncertain modeling
- **Span color range**: Sample full range of stellar types

Method
^^^^^^

1. Select calibration sample of nearby, low-reddening stars
2. Fit each star with brutus using best available models (including isochrone corrections)
3. Compute residuals: :math:`\Delta m_{\rm band} = m_{\rm obs} - m_{\rm model}`
4. Bin residuals by color or stellar parameters
5. Measure median offset in each bin
6. Smooth and interpolate to get offset as function of stellar type

Result: Offset function :math:`\Delta m_{\rm band}({\rm color})` or :math:`\Delta m_{\rm band}(T_{\rm eff}, \log g, [{\rm Fe/H}])`

Applying Photometric Offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Offsets are applied *after* model generation, as additive corrections:

.. code-block:: python

   # Model magnitudes from brutus
   model_mags = grid.get_photometry(stellar_params)

   # Apply empirical offsets (band-dependent)
   offset_g = 0.02   # mag (from field star calibration)
   offset_r = -0.01  # mag
   offset_i = 0.00   # mag

   corrected_mags = model_mags + np.array([offset_g, offset_r, offset_i])

In practice, offsets are stored in lookup tables or polynomial fits and applied automatically during fitting.

Example: Pan-STARRS Offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typical offsets for Pan-STARRS grizy:

.. code-block:: python

   # Example offsets (magnitudes)
   offsets_ps1 = {
       'g': +0.025,  # Models too bright
       'r': -0.010,  # Models too faint
       'i': +0.005,
       'z': +0.015,
       'y': +0.020
   }

These offsets vary with:

- **Stellar type**: Different for dwarfs vs giants
- **Metallicity**: Metal-poor stars may have different offsets
- **Survey**: Each photometric system has unique systematics

Color-Magnitude Dependent Offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More sophisticated calibrations use color-magnitude dependent offsets:

.. math::

   \Delta m_{\rm band}(g-i, M_i) = a_0 + a_1 (g-i) + a_2 M_i + a_3 (g-i)^2 + \ldots

This captures systematic trends across the HR diagram. Implementation:

.. code-block:: python

   def photometric_offset_gband(color_gi, abs_mag_i):
       """Empirical g-band offset as function of color and magnitude."""
       a0, a1, a2, a3 = 0.01, 0.05, -0.002, 0.01
       offset = a0 + a1*color_gi + a2*abs_mag_i + a3*color_gi**2
       return offset

Estimating Offsets for Your Data
---------------------------------

If published offsets are unavailable for your photometric system, you can derive your own using this general procedure:

1. **Select calibration sample**: Query nearby stars (d < 100 pc) with accurate Gaia parallaxes (σ_π/π < 10%), low extinction, and good photometry. Filter for RUWE < 1.4 to avoid binaries.

2. **Cross-match**: Match your survey photometry to the Gaia calibration sample.

3. **Fit with brutus**: Run ``BruteForce.fit()`` on each calibration star using the Gaia parallax as a constraint.

4. **Compute residuals**: For each star, calculate Δm = m_observed - m_model in each band.

5. **Analyze trends**: Bin residuals by color (e.g., BP-RP) and compute median offsets. Look for systematic trends with stellar type.

6. **Parameterize**: Fit a polynomial (typically linear or quadratic in color) to the median offsets.

.. tip::
   See the ``tutorials/`` notebooks for worked examples of offset derivation with real data.

Validating Calibrations
------------------------

After deriving corrections, validate them using independent data:

- **Open clusters**: Fit well-studied clusters with and without corrections. Compare derived distances and ages to literature values.
- **Eclipsing binaries**: Compare brutus mass/radius estimates to model-independent EB measurements from radial velocities and eclipse modeling.
- **Benchmark stars**: Use stars with interferometric radii or asteroseismic constraints.

Good corrections should reduce scatter and systematic offsets relative to these independent constraints.

When to Apply Corrections
--------------------------

**Always Use**:

✓ Isochrone corrections for main-sequence stars (well-calibrated from clusters)
✓ Photometric offsets for surveys with known systematics

**Use with Caution**:

⚠ Corrections derived for specific metallicity/age ranges applied to very different populations
⚠ Extrapolating corrections beyond calibration sample (e.g., to brown dwarfs or very metal-poor stars)
⚠ Combining corrections from different sources without validating consistency

**Don't Use**:

✗ Corrections without understanding their origin or validation
✗ Multiple conflicting corrections simultaneously
✗ Corrections for photometric systems different from calibration

Limitations and Uncertainties
------------------------------

**Residual Systematics**
   Even with corrections, ~0.01-0.02 mag systematic errors remain in synthetic photometry.

**Metallicity Dependence**
   Most calibrations are for solar-metallicity stars. Metal-poor and metal-rich stars may have different systematics.

**Evolutionary Phase Dependence**
   Corrections derived for main-sequence stars may not apply to giants, white dwarfs, or pre-main-sequence stars.

**Survey-Specific Systematics**
   Each photometric survey has unique calibration issues. Offsets for one survey don't transfer to another.

**Time Dependence**
   Photometric systems drift over time. Calibrations should be updated for data release versions.

Future Directions
-----------------

Ongoing work to improve empirical calibrations:

- **Gaia DR4**: Improved parallaxes and photometry for calibration samples
- **JWST**: Near/mid-IR calibrations for cool stars and dusty environments
- **Spectrophotometry**: Direct comparison of model spectra to observed spectra
- **Expanded cluster samples**: More clusters across age/metallicity space
- **Physics-based corrections**: Understanding the physical origin of systematics to improve models

Summary
-------

brutus uses two types of empirical calibration:

1. **Isochrone corrections**: Fix T_eff and radius systematics using open clusters
   - Applied during model generation
   - Parameterized by dtdm (temperature) and drdm (radius)
   - Most important for main-sequence stars

2. **Photometric offsets**: Fix synthetic photometry systematics using field stars
   - Applied after model generation (additive to magnitudes)
   - Can be constant, color-dependent, or position-dependent in HR diagram
   - Survey-specific calibrations

Both corrections improve accuracy of stellar parameter estimates and reduce systematic errors in distances and extinctions.

Next Steps
----------

- Interpret fitting results: :doc:`understanding_results`
- Choose configuration options: :doc:`choosing_options`

References
----------

Stellar Model Calibrations:

- Torres et al. (2010), "Accurate Masses and Radii of Normal Stars from Detached Eclipsing Binaries", A&A Rev, 18, 67
- Choi et al. (2018), "Empirical Isochrone Calibration Using Open Clusters", ApJ, 863, 65
- Mann et al. (2015), "How to Constrain Your M Dwarf", ApJ, 804, 64

Photometric Systems:

- Scolnic et al. (2015), "Supercal: Cross-Calibration of Multiple Photometric Systems", ApJ, 815, 117
- Magnier et al. (2020), "Pan-STARRS Photometric and Astrometric Calibration", ApJS, 251, 6

brutus Implementation:

- Speagle et al. (2025), arXiv:2503.02227
