# Core Module (`brutus.core`)

`brutus.core` provides the stellar modeling and synthetic photometry infrastructure for individual stars and coeval stellar populations.

**Key Components:**

- **Individual Stars:** `EEPTracks` predicts physical stellar parameters along evolutionary tracks, `StarEvolTrack` generates synthetic SEDs along tracks on the fly, and `StarGrid` evaluates pre-computed SED grids.
- **Populations:** `Isochrone` interpolates coeval stellar populations, and `StellarPop` synthesizes single and binary population photometry.
- **Neural Networks:** `FastNN` and `FastNNPredictor` evaluate neural network bolometric corrections across photometric passbands.
- **Grid Generation:** `GridGenerator` builds pre-computed HDF5 model grids with linear reddening vectors for `BruteForce`.

**Architecture:**

- **Parameter Prediction (`EEPTracks`, `Isochrone`):** Maps intrinsic stellar labels (initial mass, evolutionary phase or age, metallicity, alpha enhancement) to physical observables (effective temperature, luminosity, radius, surface gravity).
- **Photometry Synthesis (`StarEvolTrack`, `StellarPop`):** Converts predicted physical observables into reddened broadband magnitudes and fluxes via `FastNNPredictor`.

**Typical Usage Patterns:**

For **individual field stars** with unknown evolutionary state:

```python
from brutus.core import EEPTracks, StarEvolTrack

# Parameter prediction
tracks = EEPTracks()

# Photometry generation (on-the-fly)
star = StarEvolTrack(tracks=tracks, filters=['PS_g', 'PS_r', 'PS_i', 'PS_z'])
sed, params1, params2 = star.get_seds(
    mini=1.0, eep=400, feh=0.0,
    av=0.1, dist=1000.0
)
```

For **large samples** requiring speed:

```python
from brutus.core import StarGrid
from brutus.data import load_models

# Load pre-computed grid
models, labels, label_mask = load_models('grid_mist_v9.h5')
grid = StarGrid(models, labels)

# Fast photometry lookup (no re-computation)
# Use with BruteForce fitter for thousands to millions of stars
```

For **stellar clusters** with shared age and metallicity:

```python
from brutus.core import Isochrone, StellarPop

# Population parameter prediction
iso = Isochrone()

# Population photometry
pop = StellarPop(isochrone=iso)
seds, params1, params2 = pop.get_seds(
    feh=0.0, loga=9.0, av=0.5, dist=2000.0
)
```

**See Also:**

- {doc}`/stellar_models` - Conceptual overview of MIST models, EEP, and isochrones
- {doc}`/grid_generation` - Guide to creating and using pre-computed grids
- {doc}`/population_modeling` - Population-based cluster fitting

```{eval-rst}
.. py:module:: brutus.core
.. currentmodule:: brutus.core
```

## Individual Star Models

```{eval-rst}
.. autoclass:: EEPTracks
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: StarGrid
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: StarEvolTrack
   :members:
   :undoc-members:
   :show-inheritance:
```

## Population Models

```{eval-rst}
.. autoclass:: Isochrone
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: StellarPop
   :members:
   :undoc-members:
   :show-inheritance:
```

## Neural Networks

```{eval-rst}
.. autoclass:: FastNN
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: FastNNPredictor
   :members:
   :undoc-members:
   :show-inheritance:
```

## Grid Generation

```{eval-rst}
.. autoclass:: GridGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```

## SED Utilities

```{eval-rst}
.. autofunction:: get_seds
```

```{eval-rst}
.. autofunction:: brutus.core.sed_utils._get_seds
```
