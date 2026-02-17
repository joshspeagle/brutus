Frequently Asked Questions
===========================

Getting Started
---------------

What is brutus?
^^^^^^^^^^^^^^^

``brutus`` is a Bayesian inference package for deriving stellar properties, distances, and extinctions from photometry and astrometry.

Do I need parallax?
^^^^^^^^^^^^^^^^^^^

No, but it helps significantly. Parallax breaks the distance-extinction degeneracy.

How many bands do I need?
^^^^^^^^^^^^^^^^^^^^^^^^^

Minimum 3 bands. Optical + near-IR (e.g., Gaia + 2MASS) is recommended to break degeneracies.

Models
------

StarGrid vs StarEvolTrack?
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``StarGrid`` + ``BruteForce``: Pre-computed grids for fast fitting of large samples
- ``StarEvolTrack``: On-the-fly model generation, flexible filter selection

See :doc:`quickstart` for examples.

What stellar types are covered?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MIST models cover 0.1-300 M☉, [Fe/H] -4.0 to +0.5, pre-MS through AGB. White dwarfs and brown dwarfs are not included.

What photometric systems are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaia, 2MASS, WISE, Pan-STARRS, SDSS, DECam, Johnson-Cousins, and others. See :ref:`available-filters` for the complete list.

Priors
------

Do I need priors?
^^^^^^^^^^^^^^^^^

For bright stars with good parallax, priors have minimal impact. For faint stars or without parallax, priors are important. Use priors by default. See :doc:`priors`.

Performance
-----------

How long does BruteForce fitting take?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the default grid (614K models) and 8 photometric bands (Pan-STARRS + 2MASS), ``BruteForce`` processes approximately **2--3 stars per second** (including parallax and dust map priors). The dominant cost is the magnitude-space screening (``loglike_grid``), which evaluates all grid models for each star. Fitting scales roughly linearly with grid size; reducing to ~60K models yields a ~5--6x speedup. See :doc:`grid_generation` for detailed benchmarks.

How long does population loglikelihood evaluation take?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single evaluation of :func:`~brutus.analysis.isochrone_population_loglike` takes approximately **40 ms** (fixed grid generation cost) plus **~1 ms per star**. For 100 stars with 3 Gaia bands, expect ~120 ms per evaluation. MCMC with 128 walkers and 5000 steps would take ~9 hours serially; use multiprocessing to parallelize across walkers. See :doc:`population_modeling` for full benchmarks.

How do I tune the population grid resolution?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default grid (1000 EEP :math:`\times` 21 SMF points) balances accuracy and speed. Key findings:

- **EEP**: Converges fast. 500 points are within :math:`\Delta \ln \mathcal{L} \approx 0.4` of the 5000-point reference. Reducing from 1000 to 200 gives a 6x speedup with :math:`\Delta \ln \mathcal{L} \approx 1.4`.
- **SMF**: Converges slower. Uniform spacing outperforms non-uniform grids. At least 15 points are needed for :math:`|\Delta \ln \mathcal{L}| < 4` per 100 stars.
- **For development**: ``eep_grid=np.linspace(202, 808, 200)`` with ``smf_grid=np.linspace(0, 1, 7)`` provides a ~10x speedup suitable for testing.

See :doc:`population_modeling` for convergence tables.

Data Formats
------------

What units should input data have?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Photometry**: Linear flux densities (``flux = 10**(-0.4 * mag)``). Use :func:`~brutus.utils.inv_magnitude` to convert.
- **Parallax**: Milliarcseconds (mas)
- **Coordinates**: Galactic (l, b) in degrees

See :doc:`quickstart` for data preparation examples.

Citation
--------

Please cite Speagle et al. (2025), arXiv:2503.02227.

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

- Documentation: :doc:`quickstart`
- Glossary: :doc:`glossary`
- Issues: https://github.com/joshspeagle/brutus/issues

License: MIT
