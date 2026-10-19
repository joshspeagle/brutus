# brutus

*Et tu, Brute?*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joshspeagle/brutus/master/docs/source/_static/brutus-logo-dark.svg">
  <img alt="brutus logo" src="https://raw.githubusercontent.com/joshspeagle/brutus/master/docs/source/_static/brutus-logo-light.svg" width="480">
</picture>

[![Tests](https://github.com/joshspeagle/brutus/workflows/Tests/badge.svg)](https://github.com/joshspeagle/brutus/actions)
[![Coverage](https://codecov.io/gh/joshspeagle/brutus/branch/master/graph/badge.svg)](https://codecov.io/gh/joshspeagle/brutus)
[![Documentation Status](https://readthedocs.org/projects/brutus/badge/?version=latest)](https://brutus.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02227-b31b1b.svg)](https://arxiv.org/abs/2503.02227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`brutus` infers the distances, extinctions and properties of stars from their photometry and parallaxes. It evaluates every model in a grid of stellar models instead of running Markov chain Monte Carlo. It fits individual stars, star clusters and 3D dust along a line of sight.

```bash
pip install astro-brutus
```

The documentation at [brutus.readthedocs.io](https://brutus.readthedocs.io) has the installation steps and model data, a quickstart, the method, tutorials and the API reference.

## Citation

If you use brutus, please cite [Speagle et al. (2025)](https://arxiv.org/abs/2503.02227). brutus uses the [MIST](https://waps.cfa.harvard.edu/MIST/) stellar models and the [Bayestar](https://argonaut.skymaps.info/) dust maps.

## License

MIT. See [LICENSE](https://github.com/joshspeagle/brutus/blob/master/LICENSE). Report problems at [GitHub Issues](https://github.com/joshspeagle/brutus/issues).
