# Priors Module (`brutus.priors`)

The priors module implements prior probability distributions for Bayesian inference in brutus. These priors encode astrophysical knowledge about the Galaxy and are critical for breaking parameter degeneracies in stellar parameter estimation.

**Why Priors Matter:**

Photometry alone cannot uniquely determine stellar properties—a faint red star could be either a nearby cool dwarf or a distant reddened giant. Priors resolve these ambiguities by incorporating knowledge about:

- Where different stellar types are located in the Galaxy (Galactic structure)
- How dust extinction varies with distance and direction (3-D dust maps)
- The relative abundance of different stellar masses (IMF)
- The metallicity and age structure of different Galactic populations

**Prior Categories:**

1. **Stellar Priors**: Initial Mass Function (IMF) and luminosity functions
2. **Astrometric Priors**: Parallax-based distance constraints
3. **Galactic Structure Priors**: 3-D spatial distribution of stars (thin/thick disk, halo)
4. **Kinematics**: velocity moments of the same components, projected to proper motions and radial velocity
5. **Extinction Priors**: 3-D dust maps and R_V variation

**Typical Usage:**

Priors are automatically applied by the `BruteForce` fitter, but can also be evaluated directly:

```python
from brutus.priors.stellar import logp_imf
from brutus.priors.galactic import logp_galactic_structure
from brutus.priors.extinction import logp_extinction
import numpy as np

# Evaluate IMF prior for range of masses
masses = np.array([0.5, 1.0, 2.0, 5.0])
log_prior_imf = logp_imf(masses)

# Evaluate Galactic structure prior
distances = np.array([0.1, 0.5, 1.0, 5.0])  # kpc
gal_l, gal_b = 45.0, 10.0  # Galactic coordinates (deg)
log_prior_gal = logp_galactic_structure(distances, (gal_l, gal_b))

# Evaluate extinction prior from 3-D dust map
from brutus.dust import Bayestar
dustmap = Bayestar()  # load a 3-D dust map object
av_values = np.array([0.1, 0.5, 1.0, 2.0])  # mag
log_prior_dust = logp_extinction(av_values, dustmap, (gal_l, gal_b),
                                 distance=distances[0])
```

**Customization:**

Advanced users can customize or disable priors by passing callables to `fit()`.
The Galactic structure prior is supplied via `lngalprior`, which is called as
`lngalprior(dist, coord, labels=...)` where `coord` is a single `(l, b)` tuple:

```python
from brutus.analysis import BruteForce

fitter = BruteForce(grid)

# Disable the Galactic structure prior with a flat (zero) prior
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask,
    data_labels=obj_ids, save_file='results.h5',
    data_coords=coords,
    lngalprior=lambda dist, coord, labels=None: np.zeros(len(np.atleast_1d(dist))),
)

# Or provide a custom Galactic prior function with the same signature
```

**See Also:**

- {doc}`/priors` - Detailed conceptual guide to all prior distributions
- {doc}`/scientific_background` - How priors fit into the Bayesian framework

```{eval-rst}
.. currentmodule:: brutus.priors
```

## Stellar Priors

```{eval-rst}
.. autofunction:: logp_imf
```

```{eval-rst}
.. autofunction:: logp_ps1_luminosity_function
```

## Astrometric Priors

```{eval-rst}
.. autofunction:: logp_parallax
```

```{eval-rst}
.. autofunction:: logp_parallax_scale
```

```{eval-rst}
.. autofunction:: convert_parallax_to_scale
```

## Galactic Structure Priors

```{eval-rst}
.. autofunction:: logp_galactic_structure

.. autofunction:: galactic_prior_summary
```

```{eval-rst}
.. autofunction:: logn_disk
```

```{eval-rst}
.. autofunction:: logn_halo
```

## Kinematics

Velocity moments of the Galactic components and their projection to proper motions and radial velocity. The Galactic prior uses the disk vertical dispersions; the fit does not use the velocities yet ([#275](https://github.com/joshspeagle/brutus/issues/275)).

```{eval-rst}
.. autofunction:: solar_velocity
```

```{eval-rst}
.. autofunction:: thin_disk_velocity_moments
```

```{eval-rst}
.. autofunction:: thick_disk_velocity_moments
```

```{eval-rst}
.. autofunction:: halo_velocity_moments
```

```{eval-rst}
.. autofunction:: velocity_to_observables
```

## Extinction Priors

```{eval-rst}
.. autofunction:: logp_extinction
```

## Submodules

For advanced users who need access to internal implementations:

```{eval-rst}
.. automodule:: brutus.priors.stellar
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.priors.astrometric
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.priors.galactic
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.priors.kinematic
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.priors.extinction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```
