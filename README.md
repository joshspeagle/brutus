# brutus

*Et tu, Brute?*

![brutus logo](https://github.com/joshspeagle/brutus/blob/master/brutus_logo.png?raw=true)

[![Tests](https://github.com/joshspeagle/brutus/workflows/Tests/badge.svg)](https://github.com/joshspeagle/brutus/actions)
[![Coverage](https://codecov.io/gh/joshspeagle/brutus/branch/master/graph/badge.svg)](https://codecov.io/gh/joshspeagle/brutus)
[![Documentation Status](https://readthedocs.org/projects/brutus/badge/?version=latest)](https://brutus.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![Python](https://img.shields.io/pypi/pyversions/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02227-b31b1b.svg)](https://arxiv.org/abs/2503.02227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`brutus` is a Pure Python package for **"brute force" Bayesian inference** to derive distances, reddenings, and stellar properties from photometry. The package is designed to be highly modular and user-friendly, with comprehensive support for modeling individual stars, star clusters, and 3-D dust mapping.

**Comprehensive documentation can be found at [brutus.readthedocs.io](https://brutus.readthedocs.io)**.

Please contact Josh Speagle (<j.speagle@utoronto.ca>) with any questions.

## Installation

The most recent **stable** release can be installed via `pip` by running

```bash
pip install astro-brutus
```

## Key Features

🌟 **Individual Star Modeling**: Fit distances, reddenings, and stellar properties for individual stars using Bayesian inference (with either pre-computed stellar grids or evolutionary mass tracks)

🌟 **Cluster Analysis**: Model stellar clusters with consistent ages, metallicities, and distances (with isochrones)

🌟 **3D Dust Mapping**: Integrate with 3D dust maps and model extinction along lines of sight

## Detailed Installation

### Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows with WSL (see Windows note below)

### Quick Install

For most users, install the most recent stable release from PyPI:

```bash
pip install astro-brutus
```

### Development Install

For development or to get the latest features:

```bash
git clone https://github.com/joshspeagle/brutus.git
cd brutus
pip install -e ".[dev]"
```

### Windows Users - Important Note

⚠️ **Windows Compatibility**: Due to the `healpy` dependency (required for interacting with dust maps), brutus does not work reliably on native Windows. **Windows users should install and run brutus in WSL (Windows Subsystem for Linux)**.

### Conda Installation

If you use conda, you can install from conda-forge:

```bash
conda install -c conda-forge astro-brutus
```

### Dependencies

Core dependencies that will be automatically installed:

- `numpy` (≥1.19) - Numerical computing
- `scipy` (≥1.6) - Scientific computing
- `matplotlib` (≥3.3) - Plotting
- `h5py` (≥3.0) - HDF5 file support
- `healpy` (≥1.14) - HEALPix utilities for incorporating dust maps
- `numba` (≥0.53) - Just-in-time compilation for performance
- `pooch` (≥1.4) - Data downloading and management
- `tqdm` (≥4.50) - Progress bars and live tracking

## Data

Brutus requires stellar evolution models and other data files to function. These can be downloaded automatically using built-in utilities:

```python
from brutus import fetch_grids, fetch_isos, fetch_dustmaps

# Download MIST stellar evolution grids
fetch_grids()

# Download MIST isochrones
fetch_isos()

# Download 3D dust maps
fetch_dustmaps()
```

Alternately, you can download them manually from the [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/astro-brutus).

## Contributing

We welcome contributions! Please see the [contribution guide](https://brutus.readthedocs.io/en/latest/contributing.html) for guidelines.

### Development Setup

```bash
git clone https://github.com/joshspeagle/brutus.git
cd brutus
pip install -e ".[dev]"
```

### Running Tests

```bash
# Basic tests
pytest

# With coverage
pytest --cov=brutus
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Joshua S. Speagle
- **Email**: <j.speagle@utoronto.ca>
- **GitHub**: [https://github.com/joshspeagle/brutus](https://github.com/joshspeagle/brutus)
- **Issues**: [https://github.com/joshspeagle/brutus/issues](https://github.com/joshspeagle/brutus/issues)

## Acknowledgments

`brutus` by default uses [MIST](http://waps.cfa.harvard.edu/MIST/) stellar evolution models and [Bayestar](https://argonaut.skymaps.info/) dust maps via the [https://dustmaps.readthedocs.io/en/latest/](`dustmaps` package). We thank the developers of these initiatives and codebases for making their data and/or code publicly available.
