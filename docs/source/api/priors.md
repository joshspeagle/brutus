# Priors Module (`brutus.priors`)

`brutus.priors` implements the astrophysical prior probability distributions used by `BruteForce` to break degeneracies between stellar luminosity, distance, and dust reddening.

**Role in Inference:**

Because broadband photometry alone often permits both a nearby cool dwarf and a distant reddened giant, `brutus` incorporates prior distributions over:

- **Stellar Mass and Luminosity:** Kroupa initial mass function (`logp_imf`) and Pan-STARRS $r$-band luminosity function (`logp_ps1_luminosity_function`).
- **Astrometry:** Gaussian parallax likelihood (`logp_parallax`) and scale-factor marginalization (`logp_parallax_scale`).
- **Galactic Structure:** Three-component thin-disk, thick-disk, and halo number density, metallicity, and age distributions (`logp_galactic_structure`, `logn_disk`, `logn_halo`, `logp_feh`, `logp_age_from_feh`).
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
```

```{eval-rst}
.. autofunction:: logn_disk
```

```{eval-rst}
.. autofunction:: logn_halo
```

```{eval-rst}
.. autofunction:: logp_feh
```

```{eval-rst}
.. autofunction:: logp_age_from_feh
```

## Extinction Priors

```{eval-rst}
.. autofunction:: logp_extinction
```

## Submodules

```{eval-rst}
.. automodule:: brutus.priors.stellar
   :no-members:
```

```{eval-rst}
.. automodule:: brutus.priors.astrometric
   :no-members:
```

```{eval-rst}
.. automodule:: brutus.priors.galactic
   :no-members:
```

```{eval-rst}
.. automodule:: brutus.priors.extinction
   :no-members:
```
