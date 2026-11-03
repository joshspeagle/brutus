# brutus

*Et tu, Brute?*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joshspeagle/brutus/main/docs/source/_static/brutus-logo-dark.svg">
  <img alt="brutus logo" src="https://raw.githubusercontent.com/joshspeagle/brutus/main/docs/source/_static/brutus-logo-light.svg" width="480">
</picture>

[![Tests](https://github.com/joshspeagle/brutus/workflows/Tests/badge.svg)](https://github.com/joshspeagle/brutus/actions)
[![Coverage](https://codecov.io/gh/joshspeagle/brutus/branch/main/graph/badge.svg)](https://codecov.io/gh/joshspeagle/brutus)
[![Documentation Status](https://readthedocs.org/projects/brutus/badge/?version=latest)](https://brutus.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/astro-brutus.svg)](https://pypi.org/project/astro-brutus/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02227-b31b1b.svg)](https://arxiv.org/abs/2503.02227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`brutus` is a Python package for Bayesian inference of stellar distances, dust extinctions, and intrinsic stellar parameters from broadband photometry and astrometric parallaxes. Rather than relying on iterative Markov chain Monte Carlo (MCMC) sampling over individual stars, `brutus` evaluates the likelihood across a pre-computed grid of stellar evolution models while profiling and integrating over continuous distance and reddening parameters. The package supports fitting large catalogs of individual field stars, modeling coeval stellar populations and open clusters, and mapping 3D interstellar dust extinction along lines of sight.

```bash
pip install astro-brutus
```

Full documentation, data download instructions, scientific background, tutorials, and the API reference are available at [brutus.readthedocs.io](https://brutus.readthedocs.io).

## Citation

If you use `brutus` in your research, please cite [Speagle et al. (2025)](https://arxiv.org/abs/2503.02227). `brutus` builds on the [MIST](https://waps.cfa.harvard.edu/MIST/) stellar evolution models and the [Bayestar](https://argonaut.skymaps.info/) 3D dust maps.

## License

`brutus` is released under the MIT License (see [LICENSE](https://github.com/joshspeagle/brutus/blob/main/LICENSE)). Bug reports and feature requests can be submitted on [GitHub Issues](https://github.com/joshspeagle/brutus/issues).
