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
