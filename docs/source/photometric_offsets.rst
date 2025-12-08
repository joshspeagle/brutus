Empirical Calibrations
======================

This page explains the **empirical calibration** procedures used in ``brutus`` to correct systematic errors in theoretical stellar models. For full details, see Speagle et al. (2025) §5.

Why Empirical Calibration?
--------------------------

Theoretical stellar models like MIST have known systematic errors:

- **Temperature offsets**: Models predict temperatures that differ from observations, particularly for M dwarfs (100-300 K discrepancies)
- **Radius discrepancies**: Theoretical radii disagree with interferometric measurements, especially for low-mass stars (up to 10-20% for M dwarfs)
- **Photometric systematics**: Synthetic photometry has band-dependent offsets due to incomplete opacities, atmospheric modeling assumptions, and photometric system differences

Without corrections, these propagate into biased distances, extinctions, and derived stellar parameters.

``brutus`` implements two complementary corrections:

1. **Isochrone corrections**: Adjust effective temperature and radius during model generation
2. **Photometric offsets**: Multiplicative flux corrections derived from fitting results

Isochrone Corrections
---------------------

Isochrone corrections modify the predicted :math:`T_{\rm eff}` and radius as a function of stellar mass. These are applied during model generation via the ``corr_params`` parameter.

Mathematical Form
^^^^^^^^^^^^^^^^^

The corrections are applied to :math:`\log T_{\rm eff}` and :math:`\log R`:

.. math::

   \Delta \log T_{\rm eff} = \log_{10}(1 + (M - 1) \times \texttt{dtdm}) \times f_{\rm EEP} \times f_{\rm [Fe/H]}

.. math::

   \Delta \log R = \log_{10}(1 + (M - 1) \times \texttt{drdm}) \times f_{\rm EEP} \times f_{\rm [Fe/H]}

where:

- :math:`M` is the initial mass in solar masses
- :math:`f_{\rm EEP} = 1 - 1/(1 + \exp(-({\rm EEP} - 454)/\texttt{msto\_smooth}))` suppresses corrections after the main-sequence turnoff
- :math:`f_{\rm [Fe/H]} = \exp(\texttt{feh\_scale} \times {\rm [Fe/H]})` scales corrections with metallicity
- Corrections are zeroed for :math:`M \geq 1 M_\odot`

Default Parameters
^^^^^^^^^^^^^^^^^^

The default correction parameters (from cluster calibration):

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dtdm``
     - 0.09
     - Temperature correction slope (dimensionless)
   * - ``drdm``
     - -0.09
     - Radius correction slope (dimensionless)
   * - ``msto_smooth``
     - 30.0
     - EEP smoothing scale for turnoff suppression
   * - ``feh_scale``
     - 0.5
     - Metallicity scaling factor

Usage
^^^^^

Corrections are applied automatically by default. To customize:

.. code-block:: python

   from brutus.core import Isochrone, StellarPop

   # Initialize models
   iso = Isochrone()
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              '2MASS_J', '2MASS_H', '2MASS_Ks']
   pop = StellarPop(iso, filters=filters)

   # Use default corrections (recommended)
   seds, params, params2 = pop.get_seds(
       feh=0.0, loga=9.0, av=0.1, dist=1000.0,
       apply_corr=True  # Default
   )

   # Custom correction parameters
   custom_corr = (0.10, -0.08, 25.0, 0.6)  # (dtdm, drdm, msto_smooth, feh_scale)
   seds, params, params2 = pop.get_seds(
       feh=0.0, loga=9.0, av=0.1, dist=1000.0,
       corr_params=custom_corr
   )

   # Disable corrections entirely
   seds_uncorr, _, _ = pop.get_seds(
       feh=0.0, loga=9.0, av=0.1, dist=1000.0,
       apply_corr=False
   )

For grid generation with :class:`~brutus.core.GridGenerator`:

.. code-block:: python

   from brutus.core import GridGenerator, EEPTracks

   tracks = EEPTracks()
   filters = ['Gaia_G_MAW', 'Gaia_BP_MAWf', 'Gaia_RP_MAW',
              'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']
   generator = GridGenerator(tracks, filters=filters)

   # Generate grid with default corrections
   generator.make_grid(
       output_file='my_grid.h5',
       apply_corr=True  # Default
   )

   # Or with custom parameters
   generator.make_grid(
       output_file='my_grid_custom.h5',
       corr_params=(0.10, -0.08, 25.0, 0.6)
   )

Photometric Offsets
-------------------

Photometric offsets are **multiplicative flux corrections** that account for systematic differences between model predictions and observed photometry. These are computed from fitting results using bootstrap resampling.

Overview
^^^^^^^^

The :func:`~brutus.analysis.photometric_offsets` function:

1. Takes posterior samples from :class:`~brutus.analysis.BruteForce` fitting
2. Generates model SEDs for each sample
3. Computes model/data flux ratios for each band
4. Uses likelihood reweighting to avoid circularity (for bands used in fitting)
5. Estimates median offsets and uncertainties via bootstrap

The key innovation is **likelihood reweighting**: when computing offsets for a band that was used in fitting, the function recomputes posteriors excluding that band to obtain unbiased estimates.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from brutus.analysis.offsets import photometric_offsets, PhotometricOffsetsConfig

   # After running BruteForce.fit() and extracting results...
   # phot: observed fluxes, shape (n_objects, n_filters)
   # err: flux errors, shape (n_objects, n_filters)
   # mask: observation mask, shape (n_objects, n_filters)
   # models: model grid coefficients
   # idxs: fitted model indices, shape (n_objects, n_samples)
   # reds: fitted A_V values, shape (n_objects, n_samples)
   # dreds: fitted R_V values, shape (n_objects, n_samples)
   # dists: fitted distances (kpc), shape (n_objects, n_samples)

   # Compute offsets with default settings
   offsets, errors, n_used = photometric_offsets(
       phot, err, mask, models, idxs, reds, dreds, dists
   )

   # Apply corrections to observed photometry
   phot_corrected = phot * offsets[None, :]

Configuration
^^^^^^^^^^^^^

Use :class:`~brutus.analysis.offsets.PhotometricOffsetsConfig` to customize behavior:

.. code-block:: python

   config = PhotometricOffsetsConfig(
       min_bands_used=5,        # Minimum bands for reweighted objects
       min_bands_unused=3,      # Minimum bands for non-reweighted objects
       n_bootstrap=500,         # Bootstrap iterations
       uncertainty_method='bootstrap_iqr',  # 'bootstrap_iqr' or 'bootstrap_std'
       random_seed=42           # For reproducibility
   )

   offsets, errors, n_used = photometric_offsets(
       phot, err, mask, models, idxs, reds, dreds, dists,
       config=config
   )

Additional Parameters
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   offsets, errors, n_used = photometric_offsets(
       phot, err, mask, models, idxs, reds, dreds, dists,
       sel=quality_mask,           # Boolean selection of objects to use
       weights=sample_weights,     # Per-sample weights
       mask_fit=fitting_mask,      # Which filters were used in fitting
       old_offsets=previous_offs,  # Remove previous offsets before recomputing
       dim_prior=True,             # Apply dimensionality prior in reweighting
       prior_mean=prior_offs,      # Gaussian prior means
       prior_std=prior_errs,       # Gaussian prior standard deviations
       verbose=True
   )

Interpreting Results
^^^^^^^^^^^^^^^^^^^^

The function returns:

- ``offsets``: Multiplicative flux corrections (model/data ratios). Apply as ``phot_corrected = phot * offsets``
- ``errors``: Uncertainties from bootstrap distribution
- ``n_used``: Number of objects used per filter (useful for quality assessment)

Offsets near 1.0 indicate good agreement between models and data. Values significantly different from 1.0 suggest systematic calibration differences.

Pre-Computed Offsets
--------------------

The default ``brutus`` grids include empirically calibrated offsets derived from:

1. **Open cluster fitting**: Six benchmark clusters (including M67) used to derive isochrone corrections and initial photometric offsets
2. **Field star validation**: ~23,000 nearby, low-reddening stars with precise Gaia parallaxes

See Speagle et al. (2025) §5.2-5.3 for calibration details and Table 5 for the derived offset values.

When to Use Custom Calibrations
-------------------------------

Use the default calibrations for most applications. Consider custom calibrations when:

- Working with photometric systems not in the default set
- Analyzing populations very different from the calibration sample (e.g., very metal-poor)
- Combining data from surveys with known cross-calibration issues

Limitations
-----------

- **Metallicity range**: Calibrations are optimized for near-solar metallicity. Metal-poor stars may have different systematics.
- **Evolutionary phase**: Corrections are derived primarily from main-sequence stars. Giants and other phases may require different treatment.
- **Survey-specific**: Each photometric system has unique calibration issues. Offsets don't transfer between surveys.

See Also
--------

- :doc:`grid_generation` - Grid-based fitting with ``BruteForce``
- :doc:`stellar_models` - MIST models and :ref:`available filters <available-filters>`
- :doc:`faq` - Troubleshooting and common questions

References
----------

Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", `arXiv:2503.02227 <https://arxiv.org/abs/2503.02227>`_ (see §5 for calibration details)
