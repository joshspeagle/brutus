# brutus
#### _**Et tu, Brute?**_

`brutus` is a Pure Python package that uses Bayesian inference
to derive distances, reddening, and stellar properties from photometry over
a grid of stellar models.

### Documentation
**Currently nonexistent.**

### Data
While `brutus` can be run over an arbitrary set of stellar models,
it is configured for two by default: MIST and Bayestar.

The current MIST grid (v4.1) can be found
[here](https://www.dropbox.com/s/mxx2m07am2tptc0/grid_v4.1.h5?dl=0).
The current Bayestar grid (v1) can be found
[here](https://www.dropbox.com/s/hvivkp5ll0tyf5d/grid_bayestar_v1.h5?dl=0).

### Generating SEDs
`brutus` contains built-in SED generation utilities that run over the MIST
stellar models and utilize the SED prediction engine taken from `minesweeper`.
This capability is not necessary to apply to code to data given
the pre-computed grids defined above, but can be useful if you'd like to
generate your own grid of MIST models.

### Installation
`brutus` can be installed by running
```
python setup.py install
```
from inside the repository.

### Demos
Several ipython notebooks currently outline very basic usage of the code.
