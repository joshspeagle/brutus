# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
