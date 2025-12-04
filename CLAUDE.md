# CLAUDE.md - Repository Guide for brutus

## Overview

`brutus` is a Pure Python package for **"brute force" Bayesian inference** to derive distances, reddenings, and stellar properties from photometry. It supports individual star fitting, star cluster analysis, and 3D dust mapping.

**Package name on PyPI**: `astro-brutus`
**Version**: 1.0.0

## Repository Structure

```
brutus/
├── src/brutus/           # Main package source
│   ├── __init__.py       # Package exports
│   ├── core/             # Core stellar modeling
│   ├── analysis/         # Fitting and analysis workflows
│   ├── data/             # Data downloading/loading
│   ├── dust/             # 3D dust map utilities
│   ├── priors/           # Bayesian priors
│   ├── plotting/         # Visualization utilities
│   └── utils/            # Math, photometry, sampling utilities
├── tests/                # Test suite (pytest)
├── docs/source/          # Sphinx documentation
├── tutorials/            # Jupyter notebook tutorials
└── pyproject.toml        # Package configuration
```

## Key Modules and Classes

### `brutus.core` - Stellar Modeling
- **`EEPTracks`**: Load and interpolate stellar evolution tracks (EEP-based)
- **`StarGrid`**: Pre-computed stellar model grid for fast fitting
- **`StarEvolTrack`**: Single star evolution along tracks
- **`Isochrone`**: Stellar isochrone models (constant age populations)
- **`StellarPop`**: Synthetic stellar population generation
- **`FastNN` / `FastNNPredictor`**: Neural network utilities for SED prediction
- **`GridGenerator`**: Generate model grids for fitting

### `brutus.analysis` - Fitting Workflows
- **`BruteForce`**: Main fitting class for individual stars
- **`photometric_offsets`**: Compute photometric zeropoint offsets
- **`isochrone_population_loglike`**: Population-level likelihood for clusters
- **`los_clouds_*`**: Line-of-sight dust modeling functions

### `brutus.data` - Data Management
- **`fetch_grids()`, `fetch_isos()`, `fetch_tracks()`**: Download model data
- **`fetch_dustmaps()`**: Download 3D dust maps
- **`load_models()`**: Load HDF5 model files

### `brutus.dust` - Dust Mapping
- **`Bayestar`**: Interface to Bayestar 3D dust maps
- **`lb2pix`**: Coordinate to HEALPix conversion

### `brutus.priors` - Bayesian Priors
- **`logp_imf`**: Initial mass function prior
- **`logp_parallax`**: Parallax prior (with scale conversion)
- **`logp_galactic_structure`**: Galactic structure (disk + halo) prior
- **`logp_extinction`**: Extinction prior

### `brutus.plotting` - Visualization
- **`cornerplot`**: Corner plots for posteriors
- **`posterior_predictive`**: SED posterior predictive plots
- **`dist_vs_red`**: Distance vs reddening plots
- **`photometric_offsets`**: Offset visualization

### `brutus.utils` - Utilities
- **`magnitude`, `inv_magnitude`**: Flux/magnitude conversions
- **`luptitude`, `inv_luptitude`**: Asinh magnitude conversions
- **`phot_loglike`**: Photometric log-likelihood calculation
- **`quantile`**: Weighted quantile computation
- **`sample_multivariate_normal`**: MVN sampling utilities

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=brutus

# Build documentation
cd docs && make html
```

## Testing

Tests are in `tests/` directory, organized by module:
- `test_core/` - Core module tests
- `test_analysis/` - Analysis module tests
- `test_data/` - Data loading tests
- `test_priors/` - Prior function tests
- `test_plotting/` - Plotting tests
- `test_utils/` - Utility function tests
- `test_dust/` - Dust module tests

Run specific test modules:
```bash
pytest tests/test_core/
pytest tests/test_analysis/test_individual.py -v
```

## Key Data Files

The package uses external data files (downloaded via `fetch_*` functions):
- **MIST stellar grids**: Pre-computed SED grids
- **MIST isochrones**: Age-metallicity isochrone tables
- **Bayestar dust maps**: 3D extinction maps

Data is cached via `pooch` in user's cache directory.

## Code Style

- Uses `black` for formatting (line length 88)
- Uses `isort` for import sorting
- Type hints encouraged but not enforced
- Docstrings follow NumPy style (see `DOCSTRING_STYLE.md`)

## Common Patterns

### Fitting individual stars
```python
from brutus import BruteForce, StarGrid, load_models

# Load model grid
models = load_models('path/to/grid.h5')
grid = StarGrid(models)

# Create fitter and run
fitter = BruteForce(grid=grid)
results = fitter.fit(phot, phot_err, ...)
```

### Working with stellar populations
```python
from brutus import Isochrone, StellarPop

iso = Isochrone()
pop = StellarPop(isochrone=iso)
seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0)
```
