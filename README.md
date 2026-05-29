# brutus

*Et tu, Brute?*

![brutus logo](https://github.com/joshspeagle/brutus/blob/master/brutus_logo.png?raw=true)

[![Tests](https://github.com/joshspeagle/brutus/workflows/Tests/badge.svg)](https://github.com/joshspeagle/brutus/actions)
[![Coverage](https://codecov.io/gh/joshspeagle/brutus/branch/master/graph/badge.svg)](https://codecov.io/gh/joshspeagle/brutus)
[![Documentation Status](https://readthedocs.org/projects/brutus/badge/?version=latest)](https://brutus.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02227-b31b1b.svg)](https://arxiv.org/abs/2503.02227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`brutus` is a Python package for Bayesian inference of stellar properties, distances, and extinctions from photometry. It supports individual star fitting, cluster analysis, and 3D dust mapping.

**Documentation**: [brutus.readthedocs.io](https://brutus.readthedocs.io)

## Installation

```bash
pip install astro-brutus
```

Requires Python 3.9+ on Linux, macOS, or Windows with WSL.

**Windows users**: Due to the `healpy` dependency, brutus requires WSL (Windows Subsystem for Linux) on Windows.

### Development Install

```bash
git clone https://github.com/joshspeagle/brutus.git
cd brutus
pip install -e ".[dev]"
```

## Data Files

Download required stellar models and dust maps:

```python
from brutus import fetch_grids, fetch_isos, fetch_dustmaps

fetch_grids()      # MIST stellar grids
fetch_isos()       # MIST isochrones
fetch_dustmaps()   # 3D dust maps
```

Data files are also available from the [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/astro-brutus).

## Citation

If you use `brutus`, please cite [Speagle et al. (2025)](https://arxiv.org/abs/2503.02227).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Author: Joshua S. Speagle (<j.speagle@utoronto.ca>)
- Issues: [GitHub Issues](https://github.com/joshspeagle/brutus/issues)

## Acknowledgments

`brutus` uses [MIST](http://waps.cfa.harvard.edu/MIST/) stellar evolution models and [Bayestar](https://argonaut.skymaps.info/) dust maps. We thank the developers for making their data publicly available.
