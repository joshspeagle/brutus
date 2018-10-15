# brutus
#### _**Et tu, Brute?**_

`brutus` is a Pure Python package that uses "brute force" Bayesian inference
to derive distances, reddenings, and stellar properties from photometry over
a grid of stellar models.

### Documentation
**Currently nonexistent.**

### Data
While `brutus` can be run over an arbitrary set of stellar models,
it is configured for two by default: [MIST](http://waps.cfa.harvard.edu/MIST/)
and [Bayestar](https://arxiv.org/pdf/1401.1508.pdf).

The current MIST grid (v5) can be found
[here](https://www.dropbox.com/s/v2wfuisjaapi4qb/grid_v5.h5?dl=0).
The current Bayestar grid (v1) can be found
[here](https://www.dropbox.com/s/hvivkp5ll0tyf5d/grid_bayestar_v1.h5?dl=0).

### Generating SEDs
`brutus` contains built-in SED generation utilities that run over the MIST
stellar models and utilize the SED prediction engine taken from 
[`minesweeper`](https://github.com/pacargile/MINESweeper).
**`brutus` can be installed and run without setting up this capability** using
the pre-computed grids defined above. This functionality is provided so that
users can generate their own grid of MIST models if desired. Please contact
Phil Cargile (pcargile@cfa.harvard.edu) for the relevant data files.

### Installation
`brutus` can be installed by running
```
python setup.py install
```
from inside the repository.

### Demos
Several Jupyter notebooks currently outline very basic usage of the code.
Please contact Josh Speagle (jspeagle@cfa.harvard.edu)
if you have any questions.
