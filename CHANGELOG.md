# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Systematic audit of every module outside the `BruteForce` hot path (which was
covered in 1.1.1): isochrone/cluster population fitting, LOS dust inference,
photometric offsets, stellar-model machinery, priors, dust maps, plotting, data
loading, and utilities. A multi-agent review produced 200 candidate findings;
113 survived independent adversarial verification (refutation, reproduction,
and impact review) and are fixed here. Every behavior fix carries a regression
test that fails on the old code; every performance rewrite carries an
equivalence test against the previous implementation. Changes marked
*(output-changing)* alter numerical results — in every case toward the
statistically correct answer.

### Fixed — isochrone cluster fitting (`analysis.populations`, `core.populations`)

- **Binary modeling was silently non-functional** *(output-changing)*:
  `generate_isochrone_population_grid` passed `smf=...` to
  `StellarPop.get_seds`, whose parameter is named `binary_fraction` and whose
  unused `**kwargs` swallowed the typo. Every SMF slice evaluated to the same
  single-star isochrone; unresolved binaries were pushed into the outlier
  component. `get_seds` no longer accepts `**kwargs` (unknown arguments now
  raise `TypeError`).
- **Post-main-sequence models were underweighted ~21x** *(output-changing)*:
  models above `eep_binary_max` are SMF-independent and stored once, but
  carried a single SMF grid spacing (~0.05) instead of the full SMF measure.
  Red giants were penalized by ~3 nats relative to MS stars, biasing cluster
  age/distance toward MS-only solutions.
- **The (mass, SMF) integration measure is now normalized** *(output-changing)*:
  previously the theta-dependent grid volume multiplied every mixture
  component — including the theta-independent field-outlier model — biasing
  the population parameters and field fraction.
- Binary companions that are invalid (post-MS primary, secondary below the
  mass bound, or no age-consistent EEP) no longer NaN-poison the primary SED
  through `add_mag`; the equal-mass shortcut (`binary_fraction=1.0`) now
  respects `eep_binary_max`; `_add_binary_components` uses the caller's EEP
  grid instead of the isochrone default.
- A stray NaN flux or zero error now drops just that band instead of driving
  the total log-likelihood to `-inf` for every theta; the cluster and outlier
  components now see the same effective band mask; under `dim_prior=True` the
  parallax enters the chi-square with an extra degree of freedom (matching the
  `BruteForce` convention) rather than mixing a Gaussian density into a
  chi-square log-pdf.
- `uniform_outlier_loglike` returned a dimensionless quantity that *grew* as
  measurement errors shrank (best-measured stars were easiest to call
  outliers); it is now a proper uniform-over-data-range flux density
  *(output-changing)*. `chisquare_outlier_loglike` accepts the inlier's mask
  and `dof_reduction` and guards DOF <= 0.
- `corr_params` is plumbed through `isochrone_population_loglike`;
  `Isochrone(predictions=...)` now actually selects/orders predicted columns
  (previously silently mislabeled); `Isochrone()` falls back to the pooch
  cache for pip installs; `FastNNPredictor.sed_batch` no longer crashes for
  single-filter predictors.

### Fixed — LOS dust inference (`analysis.los_dust`)

- Monotonicity was enforced on raw theta values, wrongly rejecting monotone
  extinction profiles (and accepting non-monotone ones) in `template_reds` and
  `additive_foreground` modes; it now constrains the effective profile
  *(output-changing)*.
- Built-in kernels are renormalized for truncation at the reddening bounds,
  removing the likelihood penalty on low-extinction foregrounds near A_V=0
  *(output-changing; pass `rlims=None` to kernels for the old behavior)*.
- A NaN reddening sample no longer silently zeroes the whole sightline
  likelihood; NaN samples are masked consistently in distance and reddening.
- The docstring's "cumulative extinction by summing clouds" description now
  matches the actual per-segment absolute-extinction model, and the evidence
  implications of `monotonic=True` for cloud-count selection are documented.

### Fixed — photometric offsets (`analysis.offsets`)

- **Leave-one-out reweighting used incorrect importance weights**
  *(output-changing)*: band-i circularity was only partially removed, shrinking
  fitted-band offsets toward 1. Corrected weights recover injected offsets on
  synthetic data for both fitted and unfitted bands.
- With a Gaussian prior and a band with no usable objects, the result is now
  `prior_mean +/- prior_std` (previously offset 1 with claimed zero error).
- Input validation no longer rejects NaN flux / non-positive errors in
  *masked* bands (the standard BruteForce data layout); float masks work;
  zero/negative observed fluxes are excluded from ratio medians; collapsed
  per-object weights exclude the object instead of silently using sample 0.
- Fitted-band selection restored to the documented legacy threshold (one band
  stricter than the refactor had made it).

### Fixed — stellar models (`core.individual`), grids and data IO

- `StarGrid` multilinear interpolation: missing bracketing corners previously
  either collapsed the query onto a far-away grid point or divided by zero
  (all-NaN SEDs); both now fall back safely (nearest-neighbor when the corner
  mass dominates) with a `UserWarning` *(output-changing near truncated track
  edges)*. Omitting a multi-valued grid axis no longer silently pins it to the
  lowest value; the KD-tree is no longer frozen with the first query's label
  set.
- Binary secondary EEPs are solved on the primary's [a/Fe] tracks (was always
  [a/Fe]=0) and via a direct monotone 1-D inversion instead of Nelder-Mead
  (6.4x faster); age weights use d(age)/d(EEP) over the actual EEP coordinate;
  the `get_corrections` docstring described the EEP dependence backwards.
- **Generated grids no longer contain invalid (NaN) models**: `_save_grid`
  applies the validity selection before writing (previously every generated
  grid crashed `BruteForce`), model validity is re-checked after the reddening
  fit, and `load_models` drops any non-finite rows from files written by older
  versions. `load_models` also: decides label availability per-column (not
  from row 0), no longer returns zero models for files lacking `eep`, warns
  with the surviving filter list, and includes `afe` in the default labels.
- `mini_bound` now masks sub-threshold primaries as documented;
  `sed_utils.get_seds` validates `av`/`rv` shapes (previously silent
  out-of-bounds reads in the parallel numba kernel); download SHA verification
  is no longer disabled by ambient `CI=true` (opt out with
  `BRUTUS_SKIP_HASH_CHECK=1`), and interrupted copies can no longer
  short-circuit future fetches.

### Fixed — priors (`priors.*`)

- `logp_galactic_structure` silently dropped the metallicity/age priors when
  `feh`/`loga` were passed as plain arrays on the small-N path and on the
  numba-fallback path *(output-changing — the priors now actually apply)*.
- The thin/thick-disk solar-position normalization now includes `R_smooth`
  consistently with the halo term (halo weight was ~10% high near the Sun)
  *(output-changing)*.
- The joint (feh, age) prior is no longer mis-factorized as independent
  marginals: age-prior mixture weights now condition on the star's
  metallicity *(output-changing for chemically extreme stars)*.
- `logp_imf` normalization was wrong (NaN for all masses) when
  `mass_min > mass_break`; `convert_parallax_to_scale` no longer divides by
  zero for `p_err=0`; a `pathlib.Path` dust file no longer silently disables
  the dust prior; the dust-map A_V prior is truncation-renormalized over
  `avlim` *(output-changing)*.

### Fixed — dust maps (`dust.maps`), plotting

- **Bayestar queries now honor the map's per-pixel reliability metadata**
  (`converged`, `DM_reliable_min/max`) by default, degrading to an
  uninformative prior outside each sightline's reliable range
  *(output-changing; `Bayestar(apply_reliability_mask=False)` restores the old
  behavior)*. Single-coordinate queries return 1-D profiles as documented, and
  the internal distance grid is no longer returned by reference.
- **SAR-regenerated distance posteriors were biased low**: the scale-to-
  distance Jacobian (d^3) was missing from the reweighting in `cornerplot` and
  `bin_pdfs_distred` *(output-changing — regenerated posteriors now agree with
  the saved ones)*. `bin_pdfs_distred(cdf=True)` cumulates along the reddening
  axis as documented; `dist_vs_red` honors its `weights` argument and handles
  multi-object input; integer 0/1 masks no longer fancy-index (silently
  duplicating/corrupting rows) in the offset diagnostics; `cornerplot` no
  longer mutates the caller's `hist2d_kwargs`.

### Fixed — utilities (`utils.*`)

- `truncnorm_logpdf`/`truncnorm_pdf` are tail-stable (no catastrophic erf
  cancellation beyond ~8 sigma, no underflowing normalization); the parallel
  multivariate-normal sampler raises on PSD-singular covariances instead of
  returning uninitialized memory; `inverse3` uses a scale-invariant
  conditioning threshold; `quantile` uses one (midpoint-CDF) convention for
  both weighted and unweighted inputs *(output-changing for unweighted calls;
  all internal callers pass weights)*.
- `phot_loglike` folds data validity into the band mask (NaN flux or zero
  error in an unmasked band no longer poisons the object), accepts shared 2-D
  model grids via a matrix-product fast path, and supports auxiliary
  chi-square terms (`extra_chi2`/`extra_dims`).

### Fixed — `analysis.individual` (BruteForce, surgical)

- `_setup` no longer mutates a user-supplied `lnprior` array in place
  (repeat `fit()` calls compounded age-weight/gradient terms).
- Parallax-based model pre-selection uses the marginal (not conditional) scale
  uncertainty, so correlated-Fisher models are no longer over-pruned
  *(output-changing: `Nsel` can grow for parallax-constrained objects)*.
- The dust-prior block no longer crashes deep in `Bayestar.query` when
  `data_coords` is omitted; the dead `apply_av_prior` parameter was removed
  and the `av_gauss` docstring corrected.

### Performance (measured on this 4-core host, equivalence-tested)

- Grid generation: reddening-coefficient fit batched over the (A_V, R_V)
  lattice — 16.7x per model (11.0 -> 0.66 ms); a full `make_grid` run is no
  longer dominated by redundant track interpolation.
- Photometric offsets: bootstrap vectorized via per-object CDF inversion
  (60x), one-pass leave-one-out reweighting, chunked SED generation —
  end-to-end 12.4x on a realistic calibration workload.
- LOS dust likelihood: disjoint distance slices assigned via `searchsorted`
  instead of per-cloud masked kernel evaluation (up to 4.2x at 5 clouds, net
  of the added truncation renormalization); prior transform 51x via
  precomputed `ndtri` forms.
- `draw_sar` batched through the existing numba sampler: 4.5-6.5x for
  dust-map post-processing paths.
- `Bayestar()` init 8x faster (lexsort instead of structured argsort);
  query allocations halved.
- Galactic prior: loop-invariant age-prior constants hoisted from the fused
  kernel and a leaner wrapper — 1.75x on 2.5M-point calls (bitwise-identical
  kernel results); `logp_extinction` 3-D path fused (~9x).
- `Isochrone` retains a zero-copy broadcast view instead of a duplicated
  alpha dimension (351 -> 176 MB resident; peak init RSS 629 -> 286 MB) and
  caches SMF-invariant primary SEDs across the population grid (2.2x).
- `EEPTracks` no longer pickles ~264 MB of dead construction intermediates
  (cache files are versioned `_cachev2.pkl`; old caches are ignored) and
  reads the MIST HDF5 once instead of ~8x on first load.
- `phot_loglike` shared-model fast path: ~4x with Nfilt-fold less memory on
  the isochrone-fitting path.

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
