# brutus
#### _**Et tu, Brute?**_

`brutus` is a Pure Python package that uses Bayesian inference
to derive distances, reddening, and stellar properties from photometry over
a grid of stellar models.

### Data
While `brutus` can be run over an arbitrary set of models, it is configured
for two models by default: MIST and Bayestar.

The current MIST grid (v4.1) can be found
[here](https://www.dropbox.com/s/mxx2m07am2tptc0/grid_v4.1.h5?dl=0).
The current Bayestar grid (v1) can be found
[here](https://www.dropbox.com/s/hvivkp5ll0tyf5d/grid_bayestar_v1.h5?dl=0).

Note that the built-in SED generation utilities run over the MIST stellar
models and SED prediction engine taken from `minesweeper`.

### Documentation
**Currently nonexistent.**

### Installation
`brutus` can be installed by running
```
python setup.py install
```
from inside the repository.

### Demos
Several ipython notebooks currently outline very basic usage of the code.
