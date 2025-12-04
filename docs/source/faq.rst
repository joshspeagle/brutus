Frequently Asked Questions
===========================

Getting Started
---------------

What is brutus?
^^^^^^^^^^^^^^^

**brutus** is a Bayesian inference package for deriving stellar properties (mass, age, metallicity), distances, and extinctions from photometry and astrometry. Well-suited for:

- Individual field stars with Gaia parallaxes
- Stellar clusters and coeval populations
- 3-D dust mapping from stellar ensembles

Do I need parallax?
^^^^^^^^^^^^^^^^^^^

**No**, but it helps enormously. Parallax breaks the distance-extinction degeneracy. Even low-precision parallax (20-30% errors) significantly improves constraints.

How many bands do I need?
^^^^^^^^^^^^^^^^^^^^^^^^^

**Minimum**: 3 bands. **Recommended**: 4-6 bands spanning optical to near-IR.

- **Optical-only** (e.g., ugriz): Sensitive to temperature but struggles with extinction
- **Optical + near-IR** (e.g., Gaia + 2MASS): Breaks distance-extinction degeneracy

Model Selection
---------------

StarGrid vs StarEvolTrack?
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``StarGrid`` (with ``BruteForce``): Fast (1-10 sec/star), for large samples with standard filters
- ``StarEvolTrack``: Flexible (any filters), for small samples or prototyping

See :doc:`choosing_options` for details.

Isochrone vs EEPTracks?
^^^^^^^^^^^^^^^^^^^^^^^

- ``EEPTracks``: For individual stars (unknown evolutionary state)
- ``Isochrone`` + ``StellarPop``: For coeval populations with fixed age

What stellar types are covered?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MIST models cover: 0.1-300 solar masses, [Fe/H] -4.0 to +0.5, pre-MS through AGB.

**Not covered**: White dwarfs, brown dwarfs, exotic objects.

Priors
------

Do I need priors?
^^^^^^^^^^^^^^^^^

**Good data** (bright stars, accurate parallax): Priors have minimal impact.

**Poor data** (faint stars, no parallax): Priors are essential to break degeneracies.

**Default**: Always use priors unless you have a specific reason not to. See :doc:`priors`.

How sensitive are results to priors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test by fitting with and without priors:

.. code-block:: python

   # With priors
   fitter.fit(data, data_err, data_mask, labels, save_file='with.h5', data_coords=coords)

   # Without Galactic prior
   fitter.fit(data, data_err, data_mask, labels, save_file='without.h5',
              lngalprior=lambda *args: 0.0)

If results change >30%, data may be insufficient.

Performance
-----------

How can I speed up fitting?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Use coarser grid (fewer mass/EEP points)
2. Limit parameter ranges (``avlim``, ``rvlim``)
3. Parallelize across stars
4. Use fewer posterior draws (``Ndraws=100``)

See :doc:`grid_generation` for details.

How much memory does brutus use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Grid files: 200 MB (500k models) to 5-10 GB (10M models)
- Runtime: 1-4 GB per fitting process

For memory issues, reduce grid size or process in batches.

Results
-------

Why do distance and parallax disagree?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Possible causes**: Bad parallax (binary, crowding), bad photometry, unresolved binary, star outside model coverage.

**Check**: Gaia RUWE > 1.4 suggests binary or poor astrometry.

Why are my uncertainties large?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Causes**: Too few bands, no parallax, degeneracies, multi-modal posteriors.

**Solutions**: Add near-IR bands, include parallax, check corner plots for degeneracies.

How do I know if results are reliable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- χ² ~ 1 (good fit)
- Residuals < 0.1-0.2 mag
- Parallax and photometric distance agree
- Posteriors are unimodal
- Results stable with/without priors

See :doc:`understanding_results` for detailed diagnostics.

Cluster Modeling
----------------

Which outlier model should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Chi-square** (default): More conservative, retains borderline members
- **Uniform**: More aggressive at excluding outliers

See :doc:`cluster_modeling` for details.

Error Messages
--------------

"No valid models found"
^^^^^^^^^^^^^^^^^^^^^^^

Check photometry for bad data, verify filters match grid, widen ``avlim``/``rvlim``.

"Grid does not cover observed star"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extend grid parameter ranges or check if star is outside MIST coverage (WD, BD).

Data Formats
------------

What photometric systems are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Built-in: Gaia, 2MASS, WISE, Pan-STARRS, SDSS, Johnson-Cousins, and more. Custom filters can be added.

What units should input data have?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Photometry**: Magnitudes (any system) - brutus converts to flux internally
- **Parallax**: Milliarcseconds (mas)
- **Distances**: Parsecs (pc)
- **Extinction**: Magnitudes (A_V)

Citation
--------

Please cite:

   Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", arXiv:2503.02227

.. code-block:: bibtex

   @ARTICLE{2025arXiv250302227S,
       author = {{Speagle}, Joshua S. and others},
       title = "{Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS}",
       journal = {arXiv e-prints},
       year = 2025,
       eprint = {2503.02227},
   }

Also cite MIST: Choi et al. (2016), ApJ, 823, 102; Dotter (2016), ApJS, 222, 8.

Getting Help
------------

1. Read the documentation: :doc:`quickstart`, :doc:`tutorials`
2. Check the :doc:`glossary` for unfamiliar terms
3. GitHub Issues: https://github.com/joshspeagle/brutus/issues
4. Email: j.speagle@utoronto.ca

When reporting issues, include: brutus version, Python version, minimal reproducible example, full error traceback.

License: MIT. Source: https://github.com/joshspeagle/brutus
