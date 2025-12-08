# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
