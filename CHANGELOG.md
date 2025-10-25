# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-06

### 🎉 First Stable Release

This is the first stable release of brutus following comprehensive code verification, testing, and documentation improvements.

### Added

#### 📚 Publication-Quality Documentation

- **Scientific Background Pages**: 6 comprehensive guides covering Bayesian framework, stellar models, grid generation, priors, cluster modeling, and photometric offsets
- **User Guides**: 3 detailed guides for understanding results, choosing options, and FAQ with 40+ questions
- **Enhanced API Documentation**: All API pages include module overviews, typical usage patterns, and code examples
- **Tutorial Descriptions**: Complete learning path with time estimates, prerequisites, and learning objectives
- **ReadTheDocs Hosting**: Full documentation hosted at [brutus.readthedocs.io](https://brutus.readthedocs.io)
- **LaTeX Math Rendering**: Mathematical equations and derivations throughout documentation
- **Cross-References**: Extensive interconnections between conceptual guides and API documentation
- **arXiv Paper Badge**: Added badge linking to Speagle et al. (2025) paper ([arXiv:2503.02227](https://arxiv.org/abs/2503.02227))

#### 🧪 Testing Infrastructure Improvements

- **Comprehensive Test Coverage**: Achieved 92% overall code coverage across all modules
- **CI/CD Enhancements**:
  - Fixed all 45 failing CI tests
  - Removed all test skip mechanisms - 606 tests now run without skipping
  - Updated GitHub Actions workflow with better data caching
  - Integrated Codecov for dynamic coverage reporting
- **Data File Management**:
  - Fixed file discovery issues for neural network models and grids
  - Improved pooch cache handling to prevent re-downloads in CI
  - Added fallback paths for both `nn_c3k.h5` and `nnMIST_BC.h5` (identical files)
- **Test Fixes**:
  - Fixed `test_data_module_imports` hanging issue
  - Corrected all pytest.skip calls to use proper assertions
  - Fixed path resolution for CI environments

#### 🔍 Code Verification

- **Complete Function Verification**: All 106 functions across 17 modules verified for mathematical correctness
- **Docstring Consistency**: Comprehensive verification of docstring-implementation alignment
- **Critical Bug Fixes**:
  - Fixed IMF normalization in `priors/stellar.py` (swapped power-law indices)
  - Fixed StarGrid distance reference in `core/individual.py` (corrected from 10 pc to 1 kpc)
  - Fixed unpacking error in `bin_pdfs_distred` for different data formats
  - Fixed exception handling for missing "agewt" field in structured arrays
- **Grid Generation Restored**: Fully modernized `GridGenerator` class with comprehensive tests

### Changed

- **Development Status**: Updated from Beta to Production/Stable
- **Documentation Standards**: All documentation follows NumPy docstring conventions
- **Project Instructions**: Updated CLAUDE.md with comprehensive documentation structure section
- **Dependencies**: Added `tqdm` as formal dependency for progress bars
- **CI/CD Workflow**:
  - Upgraded cache action to v4 for better reliability
  - Updated Codecov action to v5 with token support
  - Improved cache key strategy to prevent stale caches
  - Added CI-specific optimization to skip SHA verification for cached files
- **Code Style**: Enforced Black formatting (88 char line length) across entire codebase

## [0.9.0] - 2024-08-28

### 🚀 Major Refactoring Release

This release represents a major refactoring of brutus to improve usability, maintainability, and development workflow while preserving all existing scientific functionality.

### Added

#### 📦 Modern Python Packaging

- **Modern build system**: Migrated from `setup.py` to `pyproject.toml`
- **Development tools**: Added black, isort, flake8, mypy configurations
- **CI/CD pipeline**: Added GitHub Actions for automated testing, linting, and coverage
- **Multi-platform testing**: Automated testing on Linux, macOS, and Windows (WSL)

#### 🧪 Testing Infrastructure

- **Comprehensive test suite**: Added pytest-based testing framework with 100+ tests
- **Test fixtures**: Created reusable test data and utilities in `tests/conftest.py`
- **Coverage reporting**: Added code coverage tracking with custom `run_coverage.py` script
- **Module-specific coverage**: Individual test suites for core, analysis, plotting, utils, dust, and priors
- **Test categories**: Organized unit tests, integration tests, and slow tests with proper markers
- **Coverage optimization**: Resolved coverage instrumentation conflicts with separate test runs

#### 📁 Modular Architecture

- **Organized modules**: Split functionality into logical modules:
  - `brutus.core` - Individual stellar models, isochrones, neural networks, and SED utilities
  - `brutus.analysis` - BruteForce fitting, population analysis, photometric offsets, and line-of-sight dust
  - `brutus.plotting` - Visualization utilities (corner plots, SEDs, distance-reddening, photometric offsets)
  - `brutus.dust` - 3D dust mapping and extinction models
  - `brutus.utils` - Mathematical, photometric, and sampling utilities (split from monolithic utils.py)
  - `brutus.data` - Data downloading and loading utilities
  - `brutus.priors` - Stellar, astrometric, galactic, and extinction priors
- **Refactored major components**:
  - Core stellar evolution models (EEPTracks, StarGrid, Isochrone)
  - Analysis workflows (BruteForce, population modeling)
  - Plotting utilities (moved from monolithic plotting.py into organized submodules)
  - Utility functions (split ~2000 lines utils.py into focused submodules)

#### 📚 Documentation Improvements

- **Enhanced README**: Comprehensive installation and usage guide
- **Installation notes**: Clear Windows/WSL requirements and alternatives
- **Quick start examples**: Added code examples for common workflows
- **API documentation**: Structured documentation for new module organization

#### ⚡ Performance Improvements

- **Numba optimization**: Added JIT compilation for critical mathematical functions in utils.math
- **Vectorized operations**: Optimized MISTtracks with numba and vectorization (commit a70f6d9)
- **EEPTrack caching**: Improved caching mechanisms for evolutionary track interpolation
- **Memory efficiency**: Optimized memory usage in grid-based computations
- **Batch processing**: Improved batch processing capabilities for stellar parameter estimation

### Changed

#### 🐍 Python Support

- **Minimum Python version**: Now requires Python 3.8+ (dropped Python 2.7 support)
- **Dependency updates**: Updated all dependencies to modern versions
- **Removed legacy code**: Cleaned up Python 2 compatibility code

#### 🔧 Internal Reorganization

- **Split large modules**: Broke up the massive `utils.py` file into logical components:
  - `brutus.utils.math` - Matrix operations, statistical functions, numba-optimized routines
  - `brutus.utils.photometry` - Magnitude/flux conversions, photometric likelihoods
  - `brutus.utils.sampling` - Monte Carlo sampling and quantile functions
- **Plotting module refactor**: Reorganized monolithic `plotting.py` into focused submodules:
  - `brutus.plotting.corner` - Corner plots and posterior visualization
  - `brutus.plotting.sed` - SED visualization and posterior predictive plots
  - `brutus.plotting.distance` - Distance vs reddening plots
  - `brutus.plotting.offsets` - Photometric offset visualization
  - `brutus.plotting.binning` - Posterior binning utilities
  - `brutus.plotting.utils` - Low-level plotting utilities
- **Core refactoring**: Separated individual star models, population models, and neural networks
- **Analysis refactoring**: Reorganized fitting algorithms and population analysis tools
- **Prior refactoring**: Separated stellar, astrometric, galactic, and extinction priors

#### 📋 Dependency Management

- **Removed direct dependencies**: Removed `six` package (Python 2/3 compatibility)
- **Updated constraints**: Added minimum version requirements for all dependencies
- **Optional dependencies**: Organized development and documentation dependencies

### Fixed

#### 🐛 Platform Compatibility

- **Windows installation**: Added clear documentation about healpy/Windows issues
- **WSL support**: Provided guidance for Windows users to use WSL
- **Alternative installations**: Documented conda and Docker alternatives

#### 🔧 Build System

- **Modern packaging**: Fixed various packaging issues with modern Python build tools
- **Dependency resolution**: Improved dependency management and conflict resolution

#### 🔧 Code Quality and Bug Fixes

- **Backward compatibility cleanup**: Removed legacy function aliases and attributes
- **Import structure**: Fixed circular imports and improved module loading
- **Test stability**: Fixed infinite loop bugs in plotting utilities (hist2d function)
- **Coverage instrumentation**: Resolved conflicts between coverage tools and numba compilation
- **Documentation accuracy**: Removed hallucinated examples and added working code samples
- **API consistency**: Standardized function signatures and parameter naming across modules

### Migration Guide

#### For Existing Users

**New modular import structure** - update your imports:

```python
# OLD (no longer available)
from brutus.seds import Isochrone
from brutus.fitting import BruteForce
from brutus.utils import magnitude

# NEW (current structure)
from brutus import Isochrone, BruteForce, magnitude
# or
from brutus.core import Isochrone
from brutus.analysis import BruteForce
from brutus.utils import magnitude
```

**Key changes for users:**
- Main classes now available from `brutus` root import
- Specialized functionality available from focused submodules
- All scientific algorithms and APIs remain unchanged
- Data file formats and model grids remain compatible

#### For Developers

**Development workflow changes:**

```bash
# OLD
python setup.py install

# NEW
pip install -e ".[dev]"
pytest  # instead of custom test runners
```

### Backward Compatibility

- ✅ **All scientific algorithms preserved**: No changes to core Bayesian inference
- ✅ **Existing import patterns work**: Old imports show warnings but function correctly
- ✅ **Same file formats**: No changes to data file formats or model grids
- ✅ **API compatibility**: Core function signatures unchanged

### Known Issues

- **Windows native installation**: Requires WSL due to healpy dependency
- **Transition warnings**: Some import patterns may show deprecation warnings
- **Documentation**: Full API documentation still being updated for new structure

## [0.8.3] - Previous Release

### Last release before major refactoring

This was the final release using the old project structure and Python 2 compatibility.

#### Features in 0.8.3 and earlier

- Individual star fitting with brute-force Bayesian inference
- Star cluster modeling and analysis
- 3D dust mapping with Bayestar integration
- MIST isochrone and evolutionary track support
- Neural network SED prediction
- Comprehensive photometric system support
- Data downloading utilities with Pooch

---

## Support and Migration

- **Migration questions**: Please open an [issue](https://github.com/joshspeagle/brutus/issues)
- **Bug reports**: Use the issue tracker with the "bug" label
- **Feature requests**: Use the issue tracker with the "enhancement" label
- **Documentation**: Check the updated README and tutorial notebooks

The development team is committed to maintaining backward compatibility during this transition period.
