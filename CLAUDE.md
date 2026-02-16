# CLAUDE.md - Repository Guide for brutus

## Overview

`brutus` is a Python package for Bayesian inference of stellar properties, distances, and extinctions from photometry. It supports individual star fitting, cluster analysis, and 3D dust mapping.

**Package name on PyPI**: `astro-brutus`
**Version**: 1.0.0
**Paper**: Speagle et al. (2025), [arXiv:2503.02227](https://arxiv.org/abs/2503.02227)

## Repository Structure

```
brutus/
├── src/brutus/           # Main package source
│   ├── core/             # Stellar modeling (EEPTracks, StarGrid, Isochrone, StellarPop)
│   ├── analysis/         # Fitting (BruteForce, population modeling, photometric offsets)
│   ├── data/             # Data downloading/loading (Pooch-based, Harvard Dataverse)
│   ├── dust/             # 3D dust map utilities
│   ├── priors/           # Bayesian priors (IMF, Galactic structure, extinction)
│   ├── plotting/         # Visualization utilities
│   └── utils/            # Math, photometry, sampling utilities
├── tests/                # Test suite (pytest)
├── tutorials/            # 8 Jupyter tutorial notebooks
├── docs/source/          # Sphinx documentation
├── pyproject.toml        # Package configuration
└── CHANGELOG.md          # Version history
```

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage (NUMBA_DISABLE_JIT=1 required for accurate numba coverage)
NUMBA_DISABLE_JIT=1 pytest --cov=brutus

# Build documentation
cd docs && make html
```

## Testing

Tests are in `tests/`, organized by module (`test_core/`, `test_analysis/`, etc.).

```bash
pytest tests/test_core/
pytest tests/test_analysis/test_individual.py -v
```

**Important**: When making code changes, flag any tests that may need updating. Check for:
- Tests that directly test modified functions
- Tests that use modified functions as dependencies
- Integration tests that exercise modified code paths

## Documentation

Documentation lives in `docs/source/`:
- **Getting Started**: `installation.rst`, `quickstart.rst`, `glossary.rst`
- **Scientific Background**: `scientific_background.rst`, `stellar_models.rst`, `priors.rst`, `grid_generation.rst`, `population_modeling.rst`, `photometric_offsets.rst`
- **User Guide**: `understanding_results.rst`, `faq.rst`
- **API Reference**: `api/` directory with per-module documentation
- **Development**: `changelog.rst`, `contributing.rst`

Build with `cd docs && make html`. Output in `docs/build/html/`.

## Tutorials

8 Jupyter notebooks in `tutorials/` covering individual stars, populations, grids, priors, fitting, cluster analysis, dust mapping, and photometric calibration. These use real data files (FITS/HDF5) and require downloaded model data.

## Code Style

- `black` for formatting (line length 88)
- `isort` for import sorting
- NumPy-style docstrings

## Key APIs

### Input Data Format

brutus expects **linear flux densities**, not magnitudes:
```python
from brutus.utils import inv_magnitude
flux, flux_err = inv_magnitude(mag, mag_err)
```

### Loading Models

```python
from brutus.data import load_models
from brutus.core import StarGrid

models, labels, label_mask = load_models('grid_mist_v9.h5', filters=filters)
grid = StarGrid(models, labels)
```

### Fitting

```python
from brutus.analysis import BruteForce

fitter = BruteForce(grid)
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask,
    data_labels=obj_ids, save_file='results.h5',
    parallax=parallax, parallax_err=parallax_err,
    data_coords=coords
)
```

### Populations

```python
from brutus.core import Isochrone, StellarPop

iso = Isochrone()
pop = StellarPop(iso, filters=filters)
seds, params, params2 = pop.get_seds(feh=0.0, loga=9.0, av=0.1, dist=1000.0)
```
