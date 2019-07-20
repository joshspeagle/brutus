# brutus
#### _**Et tu, Brute?**_

`brutus` is a Pure Python package whose core modules involve using
"brute force" Bayesian inference to derive distances, reddenings, and 
stellar properties from photometry using a grid of stellar models.

The package is designed to be highly modular, with current modules including
utilities for modeling individual stars, co-eval stellar associations, and
stellar-based 3-D dust mapping.

### Documentation
**Currently nonexistent.**

### Data

Various files needed to run different `brutus` modules can be downloaded
[here](https://www.dropbox.com/sh/ozq9tk8iyy8fhte/AAC_G0wA9eQ8shHbZzAKwLe-a?dl=0).
Various components of these are described below.

#### Stellar Models
Note that while `brutus` can (in theory) be run over an arbitrary set of
stellar models, it is configured for two by default: 
[MIST](http://waps.cfa.harvard.edu/MIST/)
and [Bayestar](https://arxiv.org/pdf/1401.1508.pdf).

#### Zero-points
Zero-point offsets in several bands have been estimated using Gaia data
and can be included during runtime. 
**These are currently not thoroughly vetted, so use at your own risk.**

#### Dust Map
`brutus` is able to incorporate a 3-D dust prior. The current prior is
based on the "Bayestar19" dust map from
[Green et al. (2019)](https://arxiv.org/abs/1905.02734).

#### Generating SEDs
`brutus` contains built-in SED generation utilities based on the MIST
stellar models, modeled off of
[`minesweeper`](https://github.com/pacargile/MINESweeper).
These are optimized for either generating photometry from stellar mass
tracks or for a single-age stellar isochrone based on
artificial neural networks trained on bolometric correction tables.

An empirical correction table to the models derived using several clusters is
also provided, which improves the models down to ~0.5 solar masses.
**These are currently not thoroughly vetted, so use at your own risk.**

Please contact Phil Cargile (pcargile@cfa.harvard.edu) and Josh Speagle
(jspeagle@cfa.harvard.edu) for more information on the provided data files.

### Installation
`brutus` can be installed by running
```
python setup.py install
```
from inside the repository.

### Demos
Several Jupyter notebooks currently outline very basic usage of the code.
Please contact Josh Speagle (jspeagle@cfa.harvard.edu) with any questions.
