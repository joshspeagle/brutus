# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — statistical correctness *(output-changing)*

- **`fit(apply_dlabels=False)` evaluates the Galactic prior without model
  labels.** `logpost_grid` replaced `dlabels=None` with the grid labels, so
  the option had no effect. `logpost_grid` and `_fit` now default to the
  grid labels, and `dlabels=None` passes `labels=None` to the prior.
- **Dust-map A(V) prior scatter restored**: `logp_extinction` again adds a
  0.2 mag systematic scatter in quadrature to the map spread (new `scatter`
  argument, default `0.2`, as in brutus <= 0.8.3; the 1.x refactor had
  dropped it). Bayestar19 spreads are ~0.02 mag (median), so without it the
  prior pinned A(V) — and, through the A(V)-distance profile, the distance —
  far more tightly than the map supports, producing overconfident and
  occasionally badly wrong distances. Map bins with zero spread now also
  receive a proper prior instead of falling back to uniform.
- **Dust-map prior is proper in masked bins**: where the map is
  uninformative (reliability-masked distance bins, no coverage), the prior
  is now the normalized uniform density over `avlim`,
  `-ln(avlim[1] - avlim[0])`, instead of `ln p = 0`. The improper value
  acted as a density of 1 per mag and up-weighted reliability-masked
  distances relative to reliable ones by an arbitrary, unit-dependent factor.
- **`BruteForce` passes `avlim` to the default dust prior**, so its
  truncation and uniform normalization match the fitted A(V) range (custom
  `lndustprior` functions keep the 4-argument signature).
- **An integer 0/1 `data_mask` masks bands** in `BruteForce.fit` and
  `loglike_grid`. It was used as an index array, so the fit used band 0 and
  repeated band 1 in place of the masked selection. Boolean masks give the
  same results as before. `loglike_grid` no longer changes the caller's
  `data_mask`, `av_init` or `rv_init`.
- **`loglike_grid` normalizes every model's `lnl`**. The Gaussian constant
  `-0.5 * (Ndim * ln(2 pi) + sum(ln sigma^2))` went only to the models that
  passed the initial cut, so the culled models sat that constant (about 90
  log-units for fluxes in maggies) too low. Culled models keep their
  magnitude-space fit of (scale, A(V), R(V)).
- **`logpost_grid` truncates the Monte Carlo draws** of A(V), R(V) and
  distance at `avlim`, `rvlim` and (0.001, 1e6) kpc. A draw outside the
  support now gets zero weight. Before, it was clipped to the bound at full
  weight: about half of the A(V) draws of an unreddened model sat at
  exactly 0, and such a model gained up to ln 2 in `lnp`. This changes
  `samps_red`, `samps_dred`, `samps_dist`, `obj_log_post`, `obj_log_evid`
  and `mc_ess`; `mc_ess` is 0 for a model with no draw of nonzero weight.
- **A `functools.partial` of `logp_extinction` gets the fit's `avlim`**, as
  the bare function does, unless the partial binds `avlim` itself. A prior
  such as `partial(logp_extinction, scatter=0.3)` was normalized over its
  default (0, 20) instead of the fitted range.
- **`BruteForce.fit` skips an object with no valid bands** instead of
  stopping the whole run with `ValueError`. The object is written with
  `model_idx = -99`, `obj_Nbands = 0` and NaN values, and `fit` warns once
  with the number of such objects. The output file now closes on every exit
  path.
- **`quantile` drops samples with zero weight** before it builds the CDF.
  A zero-weight sample still got a CDF node, so `quantile([0, 1, 100],
  [0.5, 0.9], weights=[1, 1, 0])` gave `[0.5, 60.4]` instead of
  `[0.5, 1.0]`. Negative or non-finite weights, or weights that sum to 0,
  raise `ValueError`.
- **An object with no degrees of freedom adds 0 to a cluster fit.** With
  `dim_prior=True`, `phot_loglike` and `chisquare_outlier_loglike` gave
  such an object (`Ndim - dof_reduction <= 0`, for example all bands NaN)
  `-inf`, which made the cluster log-likelihood `-inf` for every parameter.
  It now gets 0, the same as with `dim_prior=False`.
- **`truncnorm_logpdf` and `logp_imf` return `LOG_ZERO` (-1e300)** for
  zero probability, not `-inf`, as the rest of brutus does. The fused and
  numpy paths of `logp_galactic_structure` now agree for ages above
  `max_age` (they gave -1e300 and -inf). `truncnorm_logpdf` and
  `truncnorm_pdf` accept numpy scalars and 0-d arrays, so
  `logp_age_from_feh(5.0)` no longer raises `TypeError`. The value
  `LOG_ZERO` lives in `brutus.utils.math`.
- **`chisquare_logpdf` returns `LOG_ZERO` (-1e300)** where the density is
  zero, not `-inf`, like `truncnorm_logpdf`. At `x == loc` it now agrees
  with `scipy.stats.chi2.logpdf` (`-ln 2` for `df = 2`, `+inf` for
  `df < 2`), and float32 input no longer overflows to `-inf`.
- **The line-of-sight and cluster log-likelihoods return `LOG_ZERO`
  (-1e300)** for a zero probability, not `-inf`:
  `los_clouds_loglike_samples`, `kernel_tophat`, `kernel_gauss`,
  `kernel_lorentz`, `marginalize_isochrone_grid`,
  `isochrone_population_loglike` and `uniform_outlier_loglike`. With
  `dim_prior=True`, `phot_loglike` gives `LOG_ZERO` for an exact fit
  (`chi2 = 0`) with 3 or more degrees of freedom.
- **`kernel_gauss` and `kernel_lorentz` normalize over `rlims` stably in
  the tails.** `kernel_gauss` took the mass from an `ndtr` difference that
  rounds to 0 for a kernel mean about 8 sigma below `rlims` or 38 sigma
  above; it now uses `log_ndtr`. `kernel_lorentz` took it from an `arctan`
  difference that lost its digits for a mean far from `rlims` (a relative
  error of 2e-4 at 1e7 widths).
- **The fused Galactic prior normalizes the age prior with the tail-stable
  log-mass of the fused extinction prior.** It used a floored `erf`
  difference. The prior changes by at most 1.8e-15 for valid inputs; the
  fits of `bench/fullfit.py` do not change.
- **`logp_extinction` normalizes its truncated normal stably in the
  tails.** It took the mass on `avlim` from an `erf` difference with a
  `1e-300` floor, so a map mean far outside `avlim` (for example 9 mag
  with `avlim=(0, 6)`) gave a log-prior near +631 instead of about +1. It
  now uses `log_ndtr` in the numpy path and a mirrored `erfc` in the fused
  kernel; both agree with `scipy.stats.truncnorm.logpdf`.
- **Batched `inverse3(regularize=True)` regularizes like the single-matrix
  path.** Its smallest-eigenvalue check cancelled for ill-conditioned
  matrices and shifted some that needed no shift. The saved results of the
  `bench/fullfit.py` fits do not change.
- **`draw_sar` and the plots that regenerate draws use `avlim=(0, 20)`**,
  the `BruteForce.fit` default, not `(0, 6)`. This affects `cornerplot`,
  `dist_vs_red` and `bin_pdfs_distred`. A star with A(V) above 6 mag got
  every draw replaced by its mean. Without `span`, the A(V) axis of
  `dist_vs_red` and `bin_pdfs_distred` now covers 0 to 20 mag, also for
  saved draws, so the default bins are 3.3 times wider; pass `span` for a
  narrower range. `draw_sar` pads a distribution with its mean after 100
  passes with no accepted draw, instead of after `max_attempts`.
- **`dist_vs_red` and `bin_pdfs_distred` smooth A(V) by 0.005 of its range
  by default** (0.1 mag over the default `avlim`): the default `smooth` is
  `(0.015, 0.005)` and `(0.01, 0.005)`.

### Changed

- The default fit limits are the constants `brutus.utils.math.AVLIM`,
  (0, 20) mag, and `RVLIM`, (1, 8). Every function that takes `avlim` or
  `rvlim` uses them as its default.
- `import brutus` now raises the original `ImportError` when a dependency is
  missing. Before, it warned with a hidden `ImportWarning` and left most names
  undefined.
- `brutus.__version__` comes from the installed package metadata, so the
  `brutus_version` attribute of generated grids follows the installed release.
  A source tree that is not installed reports `"unknown"`.
- The top level re-exports every data fetcher and loader: `fetch_nns`,
  `fetch_offsets`, `fetch_tracks` and `load_offsets` join `fetch_dustmaps`,
  `fetch_grids`, `fetch_isos` and `load_models`.
- `brutus.core` no longer exports the private `_get_seds`; import it from
  `brutus.core.sed_utils`. `brutus.dust.__version__` is removed.
- `StarGrid` is in `brutus.core.individual.__all__`.
- `logp_extinction` no longer accepts `avlim=None`, which gave masked map
  bins an improper `ln p = 0` next to normalized Gaussians in reliable
  bins. Pass the A(V) support of the fit instead; `None` raises
  `ValueError`.
- `brutus.utils` re-exports `chisquare_outlier_loglike` and
  `uniform_outlier_loglike`. The private, unused `_function_wrapper` is
  removed from `brutus.utils.math`.
- Package metadata declares the license as the SPDX expression `MIT`
  (`License-Expression: MIT`) instead of a license table and classifier.
  Only `LICENSE` is packaged as a license file; `AUTHORS.md` stays in the
  source distribution. Building from source needs `setuptools>=77`.
- `BruteForce.loglike_grid` and `logpost_grid` no longer accept unused
  `**kwargs`, so a misspelled keyword such as `av_gaus` raises `TypeError`.
  Before, it was dropped and the default applied.
- Python 3.10 or later and numpy 2.0 or later are required (the floors were 3.9
  and 1.19). Python 3.9 is past its end of life, and no astropy release for
  Python 3.9 supports numpy 2.

### Fixed

- `logp_extinction` raises `ValueError` for a 3D dust map called without
  `distance`, and `TypeError` when `dustmap.query` returns something other
  than a tuple, list or array of length 2 or 3. Both cases returned the
  uniform prior with no warning, which turned off the dust prior. A list or
  array result is now read like a tuple.
- `logp_galactic_structure` raises `ValueError` when `labels` does not have
  one entry per distance. With JIT on and more than 1000 distances, it read
  past the end of `labels`; otherwise it warned and dropped the metallicity
  and age priors. A scalar distance with `labels` or `feh` now works. The
  fused kernels of the Galactic and extinction priors no longer catch every
  error and fall back to numpy with a warning.
- `logp_imf` raises `ValueError` naming the argument for a `mass_min` or
  `mass_break` that is not finite and positive, a `mass_max` that does not
  exceed `mass_min`, or a slope that is not finite. A NaN `mass_break`
  returned `-inf` everywhere.
- `inverse3` converts `A` to float and raises `ValueError` unless its shape
  is `(3, 3)` or `(N, 3, 3)`. An integer matrix gave an integer inverse that
  truncated 0.5 to 0.
- `sample_multivariate_normal` converts `mean` and `cov` to float and raises
  `ValueError` when `cov` does not have shape `(Ndist, dim, dim)`. An
  integer `cov` shrank the spread of the draws, and a short `cov` returned
  garbage with JIT on.
- `core.sed_utils.get_seds` raises `ValueError` unless `mag_coeffs` has
  shape `(Nmodels, Nbands, 3)`. With fewer coefficients, the JIT kernel
  read past the coefficient axis.
- `plotting.posterior_predictive` treats an integer 0/1 `data_mask` as a
  mask and raises `ValueError` for a mask of the wrong shape. It plotted
  band 0 and repeated band 1 in place of the unmasked bands.
- `BruteForce.fit` and `loglike_grid` check their input shapes and raise
  `ValueError` that names the argument, before `fit` creates its output
  file. A `parallax` given without `parallax_err`, which the fit silently
  ignored, now raises. `logl_initthresh` (`init_thresh`) must be in (0, 1).
  A parallax with an error of 0 or less counts as no parallax, as in the
  parallax priors; it stopped the run with an unclear error.
- `BruteForce.logpost_grid` raises `ValueError` when `lnprior` or `dlabels`
  has a different length from the likelihood results. After
  `loglike_grid(indices=...)`, the default `dlabels` gave the Galactic prior
  the labels of the wrong models.
- `BruteForce.get_sed_grid` raises `ValueError` for an `av` or `rv` array
  whose length is not the number of models. With JIT on, it read past the
  end of the array and returned garbage SEDs.

### Documentation

- New SVG logo, with light and dark variants, and a favicon. The isochrone
  "S" is drawn as model-grid points coloured by temperature, over a faint
  posterior across the grid.
- The documentation site moves to the Furo theme and Markdown (MyST) pages,
  with a brutus colour style and a new landing page.
- Rewritten getting-started pages and method overview, with schematics of
  the forward model, the fit and the distance-extinction degeneracy, and a
  References page.
- Rewritten stellar models and model grid pages. The EEP values of the main
  sequence are corrected, and the coverage of the default grid (0.5 to 2.0
  solar masses, [Fe/H] -3.0 to +0.45) is stated.

## [1.2.0] - 2026-07-17

Systematic audit of every module outside the `BruteForce` hot path (covered in
1.1.1): population fitting, LOS dust, offsets, stellar models, priors, dust
maps, plotting, data loading, and utilities. 113 verified findings were fixed
across pull requests #61-#69, each with a regression test (the suite grew from
725 to over 1000 tests). Highlights below; full details are in the PR
descriptions. Items marked *(output-changing)* alter numerical results — in
every case toward the statistically correct answer.

### Fixed — statistical correctness *(output-changing)*

- **Cluster fitting**: binary modeling was silently non-functional (an
  `smf=` keyword typo swallowed by `**kwargs`); post-MS models were
  underweighted ~21x; the (mass, SMF) integration measure is now normalized;
  the uniform outlier model is a proper density (it previously *favored*
  flagging the best-measured stars as outliers).
- **LOS dust**: monotonicity constraints now apply to the effective
  extinction profile, and kernels renormalize for truncation at the
  reddening bounds.
- **Photometric offsets**: leave-one-out reweighting used incorrect
  importance weights, shrinking fitted-band offsets toward 1.
- **Priors**: metallicity/age priors were silently dropped when passed as
  plain arrays; the disk/halo normalization was inconsistent near the Sun;
  age-prior weights now condition on metallicity; the dust-map A(V) prior
  renormalizes over `avlim`.
- **Dust maps**: Bayestar queries honor the map's per-pixel reliability
  metadata by default (`apply_reliability_mask=False` restores raw values).
- **Plotting / post-processing**: distance posteriors regenerated from fit
  summaries were biased low (missing scale-to-distance Jacobian);
  `bin_pdfs_distred(cdf=True)` now cumulates along the reddening axis as
  documented; offset-diagnostic mask indexing fixed.
- **BruteForce**: parallax-based model pre-selection now uses the marginal
  (not conditional) scale uncertainty, so heavily reddened stars with good
  parallaxes no longer lose valid models before the posterior stage.
- **Utilities**: `quantile` uses one midpoint-CDF convention for weighted
  and unweighted input (unweighted values shift slightly); truncated-normal
  densities are tail-stable; a singular covariance in a batch no longer
  corrupts the parallel sampler.

### Fixed — robustness and API contracts

- Silent failure modes now raise clear errors: omitting a multi-valued grid
  axis (was quietly pinned to the grid edge), zero surviving filters in
  `load_models` (which also now drops a constant `afe` column), a dust map
  without sky coordinates, invalid `avlim`, mis-shaped weights/coordinates
  in the plotting utilities, and half-specified offset priors.
- NaN fluxes and zero errors are masked per band instead of invalidating
  the whole object; generated grids no longer contain invalid models;
  caller-supplied arrays (`lnprior`, `hist2d_kwargs`, labels) are no longer
  mutated in place.
- Additions: `dustfile` accepts `pathlib.Path`; `bin_pdfs_distred` /
  `dist_vs_red` support importance weights and multi-object input;
  `phot_loglike` supports shared model grids and extra chi-square terms;
  `draw_sar` gained `max_attempts` (its RNG stream changed, so fixed-seed
  draws differ; the distribution is verified identical).

### Performance (equivalence-tested)

Grid generation 16.7x per model; photometric-offset calibration ~12x
end-to-end; LOS dust likelihood up to 4.2x (prior transform 51x);
`draw_sar` 2.6-4.5x; `Bayestar()` init 8x; galactic prior 1.75x
(`logp_extinction` ~9x); `Isochrone` memory roughly halved; `EEPTracks`
caches ~264 MB smaller (now versioned `_cachev2.pkl`).

### Infrastructure

- The `docs` CI check no longer fails instantly on every pull request: the
  build was decoupled from the master-only GitHub Pages deployment
  environment, so PRs now actually build the documentation (deploys still
  run only from master).

## [1.1.1] - 2026-06-27

Performance pass on the individual-star `BruteForce` fitter (`loglike_grid` ->
`logpost_grid` -> `_fit`). Benchmarked on the real MIST grid (`grid_mist_v9.h5`,
613,530 models) and the real Orion field (Gaia parallaxes), across a cohort
spanning the full selected-model range (4 .. the `max_models=50000` cap).
End-to-end **~2.6x faster `loglike_grid`, ~2.3x faster `fit()`** on a 4-core host
(the parallel kernels scale further with physical cores). Every change is either
verified bitwise-identical in the saved (float32) outputs or proven *more
accurate* against a high-Nmc reference; see `bench/RESULTS.md` for the full
methodology and numbers.

### Changed

- **Monte-Carlo prior integration in `logpost_grid` now uses antithetic
  sampling** *(affects output -- more accurate)*: the per-model prior integral
  draws the proposal normals in antithetic pairs `(z, -z)`. The estimator stays
  unbiased; because the integrand is dominated along `ln(d)` by the
  galactic-density falloff and the `+log(d)` volume Jacobian (both monotone in
  the sampling direction), this cuts the MC-integration variance ~20x and reduces
  the finite-`Nmc` (Jensen) bias of `log(mean(w))`. Versus an `Nmc -> inf` gold
  standard the per-model log-posterior RMSE drops to ~0.28x and the bias 3-65x,
  at the same `Nmc` and half the Gaussian draws. The resulting shift in posterior
  summaries is within the procedure's intrinsic Monte-Carlo scatter. An
  adversarial review confirmed the one regime where antithetic could mildly
  *increase* variance (a very precise parallax centered on the photometric
  proposal mode with poorly-constraining photometry) is empirically negligible
  for real Gaia data (worst observed std ratio ~1.06) and never introduces bias.
- **`mc_ess` output is now a weight-concentration diagnostic, not a strict
  effective-sample count**: under antithetic correlation `1 / sum(w_i^2)` no
  longer counts independent samples (it can read up to ~2x optimistic for
  well-mixed models). It remains monotone and useful as a proposal/target
  mismatch indicator -- low values still flag poor overlap.
- `sample_multivariate_normal` gained an optional `antithetic` keyword (default
  `False`); `logp_galactic_structure` gained optional `feh`/`loga` array keywords
  (a fast path for the per-point metallicity/age prior). Existing calls are
  unaffected.

### Performance

All of the following are verified **bitwise-identical** to the prior results in
the saved `fit()` outputs (any differences are float64 round-off below float32
storage precision that flip no discrete selection):

- `loglike_grid`'s per-model numba kernels (`_get_seds`, `_optimize_fit_mag`,
  `_optimize_fit_flux`, `_get_sed_mle`) and the multivariate-normal sampler are
  parallelized with `prange` (rows are independent; reductions are exact). The
  two convergence max-reductions in `_optimize_fit_mag` are parallelized as well.
- Removed dead computation in `_get_sed_mle` (`models_int`/`reddening` were
  computed -- 4.9M `pow` calls per evaluation -- but never read).
- Fused kernels `_chi2_from_resid` and `_init_mag_resid` replace large NumPy
  temporaries; `np.zeros` -> `np.empty` for fully-overwritten arrays.
- `_get_seds` flux conversion uses `exp(fac*mag)` instead of `10**(-0.4*mag)`
  (~1.6x faster on that hot path; agreement ~1e-15).
- The batched 3x3 covariance regularization uses an analytic symmetric-3x3
  minimum-eigenvalue kernel instead of `numpy.linalg.eigvalsh` (no less accurate;
  the regularization decision is unchanged).
- `logpost_grid` no longer tiles a *structured* label array across all MC
  samples for the default galactic prior (a numpy structured-dtype `np.tile` is
  ~100x slower than the float equivalent; it dominated logpost for large
  selected-model counts). It now hands the prior the per-point `feh`/`loga` as
  plain float arrays. Bitwise-identical; ~1.24x faster `logpost_grid` at
  `Nsel=50000` (~165 ms/object), no change for small selections.

## [1.1.0] - 2026-05-29

Maintenance and polish release: verified bug fixes (several affecting numerical
output on specific paths), documentation accuracy, test-suite hygiene, and
release-process improvements. All fixes ship with regression tests.

### Fixed

- **`FastNN.encode` silent mis-scaling for 6-sample batches** *(affects output)*:
  a 2-D input of shape `(6 params, 6 samples)` broadcast against the parameter
  bounds without error and normalized along the wrong axis, producing silently
  corrupted SEDs/bolometric corrections only when exactly 6 valid samples were
  evaluated (reached via `StellarPop` synthesis). Now dispatches on input
  dimensionality explicitly.
- **`los_dust` kernels collapsing per-object cloud means** *(affects output)*:
  `kernel_gauss`/`kernel_lorentz`/`kernel_tophat` collapsed an array-valued mean
  to a scalar, so line-of-sight fits using a reddening template (`template_reds`,
  `additive_foreground`) evaluated every object against object 0's cloud mean.
  Means are now kept broadcastable. The default uniform-cloud path is unchanged.
- **Binary companions no longer discard a valid primary SED** *(affects binary
  grid generation)*: `StarEvolTrack.get_seds` returned an all-NaN combined SED
  (dropping the valid primary) when the secondary's age could not be matched to
  the unrealistically tight default tolerance. The default `tol` is relaxed
  `1e-6 -> 1e-2` dex and a primary-only fallback is used when the companion does
  not converge. *Regenerate binary (smf>0) grids to benefit; single-star fitting
  and the shipped default grids are unaffected.*
- `load_offsets` no longer crashes on a single-filter offsets file.
- `logp_imf` no longer raises `ZeroDivisionError` for a power-law slope of
  exactly 1.0 (flat-in-log); uses the logarithmic mass integral.
- `_fetch` (data download) removes a stale/broken symlink before re-linking and
  falls back to a file copy on filesystems without symlink support.
- `BruteForce.logpost_grid` floors the Monte-Carlo sample count at 1 (a very
  small `mem_lim` could drive it to 0 and crash).
- `BruteForce.loglike_grid` guards the dimensional-prior `log(Ndim)` term and
  fails fast with a clear error on a fully-masked object (was an opaque
  `ZeroDivisionError` deep in the optimizer).
- `cornerplot` no longer mutates a caller-supplied `labels` list in place.
- `dist_vs_red` uses `interpolation="none"` so binned PDF images show raw bins;
  fixed a no-op smoothing guard in `hist2d`.
- Removed a dead `from scipy import polyfit` import (`scipy` has no top-level
  `polyfit`; the code always used `numpy.polyfit`).

### Changed

- **Python support**: dropped end-of-life Python 3.8 (`requires-python>=3.9`);
  added 3.13 to the classifiers and tooling targets.
- **Dependencies**: raised the `numba` floor to `>=0.59.0` (the old 0.53 floor
  predates `numpy>=1.22`/2.x support); `numpy` remains uncapped (the code is
  numpy-2.0 clean, verified on 2.2).
- **Documentation engine**: standardized on a single NumPy-docstring processor
  (`napoleon`); removed the redundant `numpydoc` extension. The docs now build
  with zero warnings.
- **Tutorials**: standardized on the committed `Orion_l209.1_b-19.9` example
  field; replaced chi2/Nbands quality cuts with goodness-of-fit p-values.

### Documentation

- Corrected numerous stale code examples across the docs and docstrings to match
  the real API (`StarGrid(models, labels)`, `get_seds` 3-tuple in magnitudes,
  `magnitude(flux, err)`, `sample_multivariate_normal(size=)`, the prior call
  signatures, `samps_dred`, the data cache path/env var, real filter names, and
  the maggies flux convention).
- Added `summary_plot` to the API reference; documented `BruteForce.fit`'s
  performance/accuracy parameters (`max_models`, `precision_shrinkage`,
  `subsample_mode`, `R_solar`, `Z_solar`); reframed fit-quality guidance around
  the p-value methodology.

### Testing & packaging

- Added regression tests covering every fix above; reconciled the `conftest`
  coverage guidance with what CI actually runs
  (`NUMBA_DISABLE_JIT=1 pytest --cov`); the default `pytest` run no longer
  enables coverage (which crashed on some WSL/DrvFs paths).
- Added a tag-triggered PyPI publish workflow (Trusted Publishing); fixed the
  stale `MANIFEST.in` data-file path.

## [1.0.0] - 2025-12-08

First stable release of brutus following code verification, testing, and documentation improvements.

### Added

- **Documentation**: Scientific background pages, user guides, API documentation with examples, and ReadTheDocs hosting
- **Testing**: 92% code coverage, 606 tests, GitHub Actions CI/CD with Codecov integration
- **Code verification**: All functions verified for correctness; fixed IMF normalization bug, StarGrid distance reference, and other issues

### Changed

- Development status updated to Production/Stable
- Added `tqdm` as formal dependency
- Enforced Black formatting across codebase

## [0.9.0] - 2024-08-28

Major refactoring to improve usability and maintainability while preserving scientific functionality.

### Added

- **Modern packaging**: Migrated to `pyproject.toml` with black, isort, flake8, mypy
- **Testing**: pytest framework with 100+ tests, coverage reporting, multi-platform CI
- **Modular architecture**: Split into `brutus.core`, `brutus.analysis`, `brutus.plotting`, `brutus.dust`, `brutus.utils`, `brutus.data`, `brutus.priors`
- **Performance**: Numba JIT compilation, vectorized operations, improved caching

### Changed

- Minimum Python version: 3.8+ (dropped Python 2.7)
- Split large modules (`utils.py`, `plotting.py`) into focused submodules
- Updated all dependencies to modern versions

### Fixed

- Windows/WSL compatibility documentation
- Circular imports and module loading issues
- Infinite loop bug in `hist2d` function

### Migration

Update imports:

```python
# Old
from brutus.seds import Isochrone
from brutus.fitting import BruteForce

# New
from brutus import Isochrone, BruteForce
# or
from brutus.core import Isochrone
from brutus.analysis import BruteForce
```

All scientific algorithms, file formats, and core APIs remain unchanged.

## [0.8.3] - Previous Release

Final release using old project structure and Python 2 compatibility.

Features: individual star fitting, cluster modeling, 3D dust mapping, MIST support, neural network SED prediction.

---

For migration questions or bug reports, see the [issue tracker](https://github.com/joshspeagle/brutus/issues).
