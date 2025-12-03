Frequently Asked Questions
===========================

This page answers common questions about using brutus, troubleshooting issues, and understanding results. Questions are organized by topic.

Getting Started
---------------

What is brutus and when should I use it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**brutus** is a Bayesian inference package for deriving stellar properties (mass, age, metallicity), distances, and extinctions from photometry and astrometry. Use brutus when you have:

- Multi-band photometry (ideally 4+ filters from optical to near-IR)
- Optionally: parallax measurements (strongly recommended)
- Goal: Robust distance/extinction estimates with proper uncertainties

brutus is particularly well-suited for:

✓ Individual field stars with Gaia parallaxes
✓ Stellar clusters and coeval populations
✓ 3-D dust mapping from stellar ensembles
✓ Any application requiring careful uncertainty quantification

Do I need parallax measurements?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**No**, but they help enormously. Parallax breaks the distance-extinction degeneracy and dramatically improves parameter constraints.

**Without parallax**: Results rely heavily on priors (Galactic structure, dust maps). Uncertainties can be large, especially for faint or reddened stars.

**With parallax**: Even low-precision parallax (20-30% errors) significantly constrains distance and helps separate extinction from intrinsic color.

**Recommendation**: Always include parallax when available (e.g., from Gaia).

How many photometric bands do I need?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Minimum**: 3 bands

**Recommended**: 4-6 bands spanning optical to near-IR

**Why multi-wavelength matters**:

- **Optical-only** (e.g., ugriz): Sensitive to temperature but struggles with extinction
- **Optical + near-IR** (e.g., Gaia + 2MASS): Breaks distance-extinction degeneracy
- **Full coverage** (optical + near-IR + mid-IR): Best constraints on all parameters

**Example combinations**:

- Basic: Gaia G, BP, RP (3 bands) - minimal but usable with parallax
- Good: Gaia + 2MASS (6 bands) - excellent for most stars
- Optimal: Gaia + 2MASS + WISE (8 bands) - best for dusty regions

Model Selection
---------------

Should I use StarGrid or StarEvolTrack?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Short answer**: Use ``StarGrid`` for large samples, ``StarEvolTrack`` for flexibility.

**``StarGrid`` (with ``BruteForce``)**:

✓ Fast: 1-10 seconds per star
✓ Batch processing: Fit thousands to millions of stars
✓ Pre-computed grids: Fixed filter combinations
✗ Large files: 1-10 GB grids
✗ Less flexible: Must regenerate grid for new filters

**``StarEvolTrack`` (with custom fitting)**:

✓ Flexible: Any filter combination
✓ Small memory: No large grid files
✓ Easy modification: Tweak models on the fly
✗ Slower: 10-60 seconds per star (neural network evaluations)
✗ Not optimized for large samples

**Decision tree**:

- Fitting > 1000 stars with standard filters → ``StarGrid``
- Exploring different filters → ``StarEvolTrack``
- Cluster modeling with MCMC → ``StellarPop`` (similar to ``StarEvolTrack``)
- Prototyping / testing → ``StarEvolTrack``

When should I use Isochrone vs EEPTracks?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**``EEPTracks``**: For individual stars with unknown evolutionary state

- Parameterized by (mass, EEP, [Fe/H], [α/Fe])
- Fits stars anywhere along evolutionary tracks
- Use with ``StarEvolTrack`` or ``StarGrid``

**``Isochrone``**: For coeval populations with fixed age

- Parameterized by ([Fe/H], log(age), [α/Fe])
- Represents snapshot of stellar population at given age
- Use with ``StellarPop`` for cluster fitting

**Example**:

.. code-block:: python

   # Individual field star (age unknown)
   from brutus.core import EEPTracks, StarEvolTrack
   tracks = EEPTracks()
   star = StarEvolTrack(tracks=tracks)

   # Cluster (all stars same age)
   from brutus.core import Isochrone, StellarPop
   iso = Isochrone()
   pop = StellarPop(isochrone=iso)

What stellar types does brutus cover?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus uses MIST models, which cover:

✓ **Mass**: 0.1 to 300 solar masses
✓ **Evolutionary phases**: Pre-main-sequence through AGB (for low/intermediate mass) or core collapse (for massive stars)
✓ **Metallicity**: [Fe/H] from -4.0 to +0.5 dex
✓ **Ages**: ~1 Myr to 14 Gyr (depending on mass)

**Not covered** (would need custom models):

✗ White dwarfs
✗ Brown dwarfs (< 0.08 Msun)
✗ Exotic objects (CVs, X-ray binaries, etc.)
✗ Stars with extreme rotation or peculiar abundances

Priors and Configuration
-------------------------

Do I need to use priors?
^^^^^^^^^^^^^^^^^^^^^^^^^

**It depends** on your data quality.

**Good data** (bright stars, accurate parallax, low photometric errors):
   Priors have minimal impact. You can verify this by fitting with/without priors and checking that results don't change significantly.

**Poor data** (faint stars, no parallax, large photometric errors):
   Priors are **essential** to break degeneracies. Without priors, you'll get very uncertain or biased results.

**Default recommendation**: Always use priors unless you have a specific reason not to. They encode well-validated astrophysical knowledge and rarely hurt.

How sensitive are results to prior choices?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Test prior sensitivity**:

.. code-block:: python

   import h5py
   import numpy as np

   fitter = BruteForce(grid)

   # Fit with default priors
   fitter.fit(data, data_err, data_mask, labels, save_file='with_prior.h5',
              data_coords=coords)

   # Fit without Galactic prior
   fitter.fit(data, data_err, data_mask, labels, save_file='no_prior.h5',
              lngalprior=lambda *args: 0.0)

   # Compare
   with h5py.File('with_prior.h5', 'r') as f:
       dist_default = np.median(f['samps_dist'][0])
   with h5py.File('no_prior.h5', 'r') as f:
       dist_no_gal = np.median(f['samps_dist'][0])

   dist_change_pct = 100 * abs(dist_default - dist_no_gal) / dist_default
   print(f"Distance changed by {dist_change_pct:.1f}% without Galactic prior")

**Interpretation**:

- Change < 10%: Results robust, priors unimportant
- Change 10-30%: Moderate prior influence, check if priors are appropriate
- Change > 30%: Strong prior dependence, data may be insufficient

When should I customize priors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Customize priors when**:

- Studying non-Galactic objects (LMC, SMC stars need different priors)
- Specific regions with unusual properties (Galactic bulge, Local Bubble)
- You have independent information (spectroscopic metallicity, asteroseismic mass)
- Default dust maps inappropriate for your field

**Example - fixing metallicity**:

.. code-block:: python

   # If you know [Fe/H] from spectroscopy, apply tight prior
   def custom_feh_prior(feh, feh_spec=0.1, feh_err=0.05):
       """Gaussian prior around spectroscopic metallicity."""
       return -0.5 * ((feh - feh_spec) / feh_err)**2

   # Integrate into fitting (requires modifying BruteForce)

**See**: :doc:`priors` for detailed guidance on customization.

Performance and Optimization
-----------------------------

Fitting is very slow. How can I speed it up?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Diagnostics**:

- 1-10 sec/star → Normal for ``StarGrid``
- 10-60 sec/star → Normal for ``StarEvolTrack``
- >60 sec/star → Something is wrong

**Solutions**:

1. **Use coarser grid**:

   .. code-block:: python

      import numpy as np

      # Reduce grid resolution by using coarser arrays
      generator.make_grid(
          output_file='fast_grid.h5',
          mini_grid=np.linspace(0.1, 10.0, 100),  # Fewer mass points
          eep_grid=np.linspace(200, 600, 80),     # Fewer EEP points
      )

2. **Limit parameter space**:

   .. code-block:: python

      # Tighter bounds if you have prior knowledge
      output_file = fitter.fit(
          data, data_err, data_mask, labels,
          save_file='results.h5',
          avlim=(0.0, 2.0),   # Limit extinction range
          rvlim=(2.5, 4.5)    # Limit R_V range
      )

3. **Parallelize across stars**:

   .. code-block:: python

      from multiprocessing import Pool

      with Pool(processes=32) as pool:
          results_list = pool.map(fit_one_star, star_list)

4. **Use fewer posterior samples** (sacrifices precision):

   .. code-block:: python

      output_file = fitter.fit(data, data_err, data_mask, labels,
                               save_file='results.h5', Ndraws=100)  # vs default 250

How much memory does brutus use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Grid storage** (on disk):

- Small grid (500k models): ~200 MB
- Medium grid (2M models): ~1 GB
- Large grid (10M models): ~5-10 GB

**Runtime memory** (in RAM):

- Per fitting process: 1-4 GB depending on grid size and number of filters
- Parallelization: Memory × number of processes

**Solutions for memory issues**:

1. **Use smaller grids**: Limit parameter ranges or use coarser spacing during grid generation

2. **Process in batches**: Split large catalogs into batches, save results between batches

Can I run brutus on a cluster?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Yes!** brutus is well-suited for HPC environments:

**Embarrassingly parallel**: Each star fit is independent

**Example SLURM job**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks=32
   #SBATCH --mem=64GB
   #SBATCH --time=10:00:00

   python fit_catalog_parallel.py --ncores 32 --catalog my_stars.fits

**Python script**:

.. code-block:: python

   from multiprocessing import Pool
   import argparse

   parser = argparse.ArgumentParser()
   parser.add_argument('--ncores', type=int, default=1)
   parser.add_argument('--catalog', type=str)
   args = parser.parse_args()

   # Load catalog
   stars = load_catalog(args.catalog)

   # Parallel fitting
   with Pool(processes=args.ncores) as pool:
       results = pool.map(fit_one_star, stars)

   # Save results
   save_results('output.fits', results)

Results and Interpretation
---------------------------

What do I do if distance and parallax disagree?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Disagreement** means photometric distance (from brutus) and parallax-implied distance differ significantly (> 2-3σ).

**Possible causes**:

1. **Bad parallax**: Gaia parallax errors, binary contamination, crowding
2. **Bad photometry**: Systematic errors, wrong magnitudes, blended sources
3. **Unresolved binary**: Companion adds flux, making star appear brighter → underestimated distance
4. **Wrong model**: Star outside MIST coverage (WD, exotic object)
5. **High extinction**: Distance-extinction degeneracy despite parallax

**Diagnostics**:

.. code-block:: python

   # Check Gaia quality flags
   # - RUWE > 1.4 suggests binary or poor astrometry
   # - ipd_frac_multi_peak > 5 suggests non-single star
   # - high phot_bp_rp_excess_factor suggests blending

   # Check photometric residuals
   # - Large residuals indicate poor model fit
   # - Systematic trends suggest wrong stellar type

   # Try fitting without parallax
   fitter.fit(data, data_err, data_mask, labels,
              save_file='no_plx.h5')  # Omit parallax arguments
   # If photometric distance matches parallax distance without using parallax,
   # the disagreement is real (not a degeneracy issue)

**Actions**:

- Inspect Gaia RUWE and quality flags
- Check for nearby companions (imaging, proper motion)
- Look for RV variations (binarity)
- Consider flagging star as unreliable

Why are my uncertainties so large?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Causes of large uncertainties**:

1. **Insufficient data**: Too few photometric bands, no parallax, large errors
2. **Degeneracies**: Distance-extinction, mass-age-metallicity not broken by data
3. **Multi-modal posteriors**: Multiple solutions in parameter space
4. **Prior dominated**: Data too weak, priors control result

**Solutions**:

- Add more photometric bands (especially near-IR)
- Include parallax if available
- Check for and understand degeneracies (corner plots)
- Assess prior sensitivity (fit with/without priors)
- Accept large uncertainties if data genuinely insufficient

**When large uncertainties are OK**:

- Faint stars with poor photometry naturally have uncertain parameters
- Very reddened stars have intrinsic distance-extinction degeneracy
- Old, metal-poor stars have complex evolutionary histories

How do I know if results are reliable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Reliability checklist**:

✓ χ² ~ 1 (good fit quality)
✓ Residuals < 0.1-0.2 mag across all bands
✓ Parallax and photometric distance agree (if parallax available)
✓ Posteriors are unimodal (single clear solution)
✓ Results don't change much without priors (data-driven)
✓ No warnings about convergence failures

**See**: :doc:`understanding_results` for detailed diagnostic procedures.

Cluster Modeling
----------------

How do I choose the outlier model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Options**:

1. **Chi-square outlier** (``outlier_model='chisquare'``):

   - Assumes outliers follow cluster model but with additional scatter
   - Good for: Photometric binaries, differential extinction, modest contamination
   - More conservative (retains borderline members)

2. **Uniform outlier** (``outlier_model='uniform'``):

   - Assigns constant low likelihood to all outliers
   - Good for: Clean clusters with well-defined field population
   - More aggressive (excludes borderline cases)

3. **Custom function**:

   - Encode specific knowledge about field star properties
   - Good for: Known contaminant populations, complex fields

**Default recommendation**: Start with chi-square, switch to uniform if you see obvious outliers being retained.

What is mixture-before-marginalization and why does it matter?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Mixture-before-marginalization** means applying the cluster/outlier mixture model **before** integrating over unknown stellar masses.

**Wrong** (naive approach):

1. Marginalize each star over mass independently
2. Multiply likelihoods across stars
3. Mix in outliers

**Right** (brutus approach):

1. For each star, mix cluster and outlier likelihoods at each mass grid point
2. Then marginalize over mass
3. Multiply across stars

**Why it matters**:

Naive approach can **severely bias** results when field contamination is present. Mixture-before-marginalization properly accounts for outliers during the mass marginalization, preventing contamination bias.

**See**: :doc:`cluster_modeling` for mathematical details and examples.

How do I set the field contamination fraction?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``field_fraction`` parameter represents the fraction of observed stars that are field contaminants (not cluster members).

**Two approaches**:

1. **Fixed value** (if you know contamination level):

   .. code-block:: python

      lnl = isochrone_population_loglike(
          ..., field_fraction=0.15,  # 15% contamination
          cluster_prob=1.0            # All stars equally likely members a priori
      )

2. **Fitted parameter** (let data determine):

   .. code-block:: python

      def lnprob(theta):
          feh, loga, av, rv, dist, field_frac = theta  # field_frac is fitted
          return isochrone_population_loglike(
              feh=feh, loga=loga, av=av, rv=rv, dist=dist,
              field_fraction=field_frac,
              ...
          )

**Guidance**:

- Use spatial/kinematic selection to pre-clean sample → lower field_fraction
- Wide-field surveys of distant clusters → higher field_fraction (0.2-0.5)
- Nearby, well-separated clusters → lower field_fraction (0.05-0.15)

Error Messages
--------------

"No valid models found"
^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: No grid models produce reasonable likelihoods for the observed photometry.

**Solutions**:

1. Check photometry for bad data (negative fluxes, unrealistic errors)
2. Verify photometric system matches grid filters
3. Widen parameter limits: increase ``avlim`` or ``rvlim`` bounds
4. If persistent, star may be outside model grid coverage

"Grid does not cover observed star"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Star's parameters fall outside grid boundaries.

**Solutions**:

1. Extend grid coverage:

   .. code-block:: python

      import numpy as np

      generator.make_grid(
          output_file='extended_grid.h5',
          mini_grid=np.linspace(0.1, 100.0, 300),   # Wider mass range
          feh_grid=np.linspace(-3.0, 0.5, 20),      # Wider [Fe/H] range
          eep_grid=np.linspace(150, 800, 200)       # Wider EEP range
      )

2. Check if star is exotic object outside MIST coverage (WD, BD, etc.)

"Out of memory" during grid generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cause**: Grid too large for available RAM.

**Solutions**:

1. Reduce grid size (fewer points in mass, EEP, [Fe/H], [α/Fe])
2. Generate grid in chunks (split by metallicity, for example)
3. Use machine with more RAM
4. For fitting (not generation), use memory-mapped grids

Data Formats
------------

What photometric systems does brutus support?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

brutus supports any photometric system with defined filter transmission curves. Built-in filters include:

- Gaia DR2/DR3: G, BP, RP
- 2MASS: J, H, Ks
- WISE: W1, W2, W3, W4
- Pan-STARRS: g, r, i, z, y
- SDSS: u, g, r, i, z
- Johnson-Cousins: U, B, V, R, I
- Many others...

**Custom filters**: You can add new filters by providing transmission curves. See ``brutus.data.filters`` for examples.

What units should my input data have?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Photometry**: Magnitudes (any system: Vega, AB, ST) or flux densities (Jy, erg/s/cm²/Å, etc.)

brutus converts to flux internally, but you provide observed magnitudes:

.. code-block:: python

   phot = np.array([16.5, 15.2, 14.8])  # magnitudes
   phot_err = np.array([0.01, 0.02, 0.03])  # magnitude errors

   # brutus converts: F = 10^(-0.4 * m)

**Parallax**: Milliarcseconds (mas), as in Gaia

.. code-block:: python

   parallax = 2.5  # mas
   parallax_err = 0.1  # mas

**Distances**: Parsecs (pc)

**Extinction**: Magnitudes (A_V, A_λ, E(B-V))

Citation and Attribution
-------------------------

How do I cite brutus?
^^^^^^^^^^^^^^^^^^^^^

Please cite the main brutus paper:

   Speagle et al. (2025), "Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS", arXiv:2503.02227

BibTeX:

.. code-block:: bibtex

   @ARTICLE{2025arXiv250302227S,
       author = {{Speagle}, Joshua S. and {Zucker}, Catherine and {Beane}, Angus and others},
       title = "{Deriving Stellar Properties, Distances, and Reddenings using Photometry and Astrometry with BRUTUS}",
       journal = {arXiv e-prints},
       year = 2025,
       month = mar,
       eid = {arXiv:2503.02227},
       pages = {arXiv:2503.02227},
       archivePrefix = {arXiv},
       eprint = {2503.02227},
   }

Also cite the MIST models:

   Choi et al. (2016), ApJ, 823, 102
   Dotter (2016), ApJS, 222, 8

Is brutus open source?
^^^^^^^^^^^^^^^^^^^^^^^

**Yes!** brutus is released under the MIT License.

- Source code: https://github.com/joshspeagle/brutus
- PyPI: https://pypi.org/project/astro-brutus/
- Documentation: (will be) https://brutus.readthedocs.io

Contributions welcome via GitHub pull requests!

Getting Help
------------

Where can I get help?
^^^^^^^^^^^^^^^^^^^^^

1. **Read the documentation**: Start with :doc:`quickstart` and :doc:`tutorials`
2. **Check the FAQ**: You're already here!
3. **GitHub Issues**: https://github.com/joshspeagle/brutus/issues
4. **Email the authors**: j.speagle@utoronto.ca

When reporting issues, please include:

- brutus version: ``import brutus; print(brutus.__version__)``
- Python version
- Minimal reproducible example
- Error message and full traceback

How do I report a bug?
^^^^^^^^^^^^^^^^^^^^^^^

**GitHub Issues**: https://github.com/joshspeagle/brutus/issues

Please include:

1. Clear description of the bug
2. Minimal code example that reproduces the issue
3. Expected vs actual behavior
4. brutus version and Python version
5. Operating system

Can I contribute to brutus?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Absolutely!** Contributions are welcome:

- Bug fixes
- New features
- Documentation improvements
- Tutorial notebooks
- Test coverage
- Performance optimizations

See ``CONTRIBUTING.md`` in the repository for guidelines.

Future Development
------------------

What features are planned?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Roadmap includes:

- White dwarf models
- More empirical calibrations (additional surveys)
- GPU acceleration for large-scale fitting
- Integration with asteroseismic constraints
- More flexible prior specifications
- Improved cluster modeling (non-coeval populations, rotation)

**Suggestions welcome!** Open a GitHub issue to discuss new features.

Summary
-------

**Key takeaways**:

- brutus is for Bayesian stellar parameter estimation from photometry ± parallax
- Use ``StarGrid`` for speed, ``StarEvolTrack`` for flexibility
- Priors help but aren't always essential—test sensitivity
- Parallax dramatically improves results
- Multi-wavelength photometry (optical + IR) is highly recommended
- Check diagnostics (χ², residuals, posterior shapes) to assess reliability
- For clusters, use mixture-before-marginalization
- When in doubt, consult the documentation or ask for help

.. seealso::
   - Quick start: :doc:`quickstart`
   - Tutorials: :doc:`tutorials`
   - Configuration guide: :doc:`choosing_options`
   - Understanding results: :doc:`understanding_results`
