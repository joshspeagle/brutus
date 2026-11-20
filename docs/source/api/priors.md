# Priors Module (`brutus.priors`)

`brutus.priors` implements the astrophysical prior probability distributions used by `BruteForce` to break degeneracies between stellar luminosity, distance, and dust reddening.

**Role in Inference:**

Because broadband photometry alone often permits both a nearby cool dwarf and a distant reddened giant, `brutus` incorporates prior distributions over:

- **Stellar Mass and Luminosity:** Kroupa initial mass function (`logp_imf`) and Pan-STARRS $r$-band luminosity function (`logp_ps1_luminosity_function`).
- **Astrometry:** Gaussian parallax likelihood (`logp_parallax`) and scale-factor marginalization (`logp_parallax_scale`).
- **Galactic Structure:** Thin-disk, thick-disk, halo, and bar number density, metallicity, and age distributions (`logp_galactic_structure`, `galactic_prior_summary`, `logn_disk`, `logn_halo`).
- **Kinematics:** Velocity moments of the Galactic components, projected to proper motions and radial velocity (`thin_disk_velocity_moments`, `velocity_to_observables`).
- **Interstellar Extinction:** Truncated Gaussian $A_V$ prior from 3D dust maps (`logp_extinction`).

**Typical Usage:**

Priors are evaluated automatically inside `BruteForce.fit`, and can also be called directly:

```python
from brutus.priors.stellar import logp_imf
from brutus.priors.galactic import logp_galactic_structure
from brutus.priors.extinction import logp_extinction
from brutus.dust import Bayestar
import numpy as np

# Evaluate IMF prior for a range of initial masses (in Msun)
masses = np.array([0.5, 1.0, 2.0, 5.0])
log_prior_imf = logp_imf(masses)

# Evaluate Galactic structure prior along a sightline
distances = np.array([0.1, 0.5, 1.0, 5.0])  # kpc
gal_l, gal_b = 45.0, 10.0  # Galactic coordinates (deg)
log_prior_gal = logp_galactic_structure(distances, (gal_l, gal_b))

# Evaluate extinction prior from a 3D dust map
dustmap = Bayestar()
av_values = np.array([0.1, 0.5, 1.0, 2.0])  # mag
log_prior_dust = logp_extinction(
    av_values, dustmap, (gal_l, gal_b), distance=distances
)
```

**Customization:**

Pass custom callables to `BruteForce.fit` to replace or disable specific priors. The Galactic structure prior is supplied via `lngalprior(dist, coord, labels=...)`, where `coord` is a `(l, b)` tuple in degrees:

```python
from brutus.analysis import BruteForce

fitter = BruteForce(grid)

# Replace the Galactic structure prior with a flat log-prior in distance
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask,
    data_labels=obj_ids, save_file="results.h5",
    data_coords=coords,
    lngalprior=lambda dist, coord, labels=None: np.zeros_like(dist, dtype=float),
)
```

**See Also:**

- {doc}`/priors` - Detailed conceptual guide to all prior distributions
- {doc}`/scientific_background` - How priors fit into the Bayesian framework

```{eval-rst}
.. py:module:: brutus.priors
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

The Galactic prior uses the disk vertical dispersions. The fit does not use the velocities yet ([#275](https://github.com/joshspeagle/brutus/issues/275)).

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
