# Analysis Module (`brutus.analysis`)

`brutus.analysis` implements the Bayesian inference engines for fitting individual stars, coeval star clusters, empirical photometric offsets, and 3D line-of-sight dust profiles.

**Key Components:**

- **Individual Star Fitting:** `BruteForce` evaluates Bayesian posteriors over pre-computed stellar model grids.
- **Population Analysis:** `isochrone_population_loglike` and its component functions evaluate cluster likelihoods with binary stars and field-star mixture models.
- **Photometric Offsets:** `photometric_offsets` and `PhotometricOffsetsConfig` derive leave-one-out empirical zero-point corrections across survey bands.
- **Line-of-Sight Dust:** `los_clouds_priortransform` and `los_clouds_loglike_samples` fit multi-cloud 3D extinction profiles to stellar distance-reddening posteriors.

**Inference Workflow:**

1. **Grid Likelihood Evaluation:** Compute photometric and parallax likelihoods across the stellar model grid.
2. **Continuous Parameter Optimization:** Solve for the maximum-likelihood $(s, A_V, R_V)$ and $3 \times 3$ Fisher matrix per model via Gauss-Newton iteration.
3. **Prior Integration:** Integrate Galactic structure, 3D dust, and parallax priors via antithetic Monte Carlo importance sampling in $(\ln d, A_V, R_V)$.
4. **Posterior Resampling:** Draw equally weighted posterior samples of stellar models, distances, and extinctions.

**Typical Usage Patterns:**

For **individual field stars** with photometry and parallax:

```python
from brutus.analysis import BruteForce
from brutus.core import StarGrid
from brutus.data import load_models
from brutus.utils import inv_magnitude
import numpy as np
import h5py

# Load pre-computed grid
models, labels, label_mask = load_models("grid_mist_v9.h5")
grid = StarGrid(models, labels)

# Initialize fitter
fitter = BruteForce(grid)

# Observed data (Nstars=1 in this example)
# Convert observed magnitudes and errors to linear flux densities (maggies)
mag = np.array([[16.5, 15.2, 14.8, 13.5, 13.1]])  # shape (1, 5)
mag_err = np.array([[0.02, 0.02, 0.02, 0.03, 0.03]])
flux, flux_err = inv_magnitude(mag, mag_err)  # maggies = 10**(-0.4 * mag)
mask = np.ones_like(flux, dtype=bool)
obj_ids = np.array([1])
coords = np.array([[209.1, -19.9]])  # Galactic (l, b) in degrees
parallax = np.array([2.5])  # mas
parallax_err = np.array([0.1])  # mas

# Fit and save results to HDF5
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask,
    data_labels=obj_ids, save_file="results.h5",
    parallax=parallax, parallax_err=parallax_err,
    data_coords=coords,
    Ndraws=250,
)

# Read posterior samples from HDF5
with h5py.File("results.h5", "r") as f:
    dist = f["samps_dist"][0]  # (Ndraws,) distances in kpc
    av = f["samps_red"][0]     # (Ndraws,) A_V values in mag
print(f"Distance: {np.median(dist) * 1000:.0f} pc")
print(f"Extinction: {np.median(av):.2f} mag")
```

For **stellar clusters** with MCMC:

```python
from brutus.analysis import isochrone_population_loglike
from brutus.core import Isochrone, StellarPop
import emcee
import numpy as np

# Initialize population model
iso = Isochrone()
pop = StellarPop(isochrone=iso)

# Observed cluster data (N stars, M filters)
obs_flux = np.array([...])  # shape (N, M)
obs_err = np.array([...])   # shape (N, M)

# Define log-likelihood for MCMC
# theta = [feh, loga, av, rv, dist, field_frac]
# (field_frac is the field contamination fraction, 0 to 1;
# a 5-element theta raises ValueError)
def lnprob(theta):
    return isochrone_population_loglike(
        theta,
        stellarpop=pop,
        obs_phot=obs_flux,
        obs_err=obs_err,
        cluster_prob=0.9  # External membership prior
    )

# Run MCMC
ndim, nwalkers = 6, 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(initial_pos, 5000, progress=True)

# Extract posterior samples
samples = sampler.get_chain(discard=1000, thin=10, flat=True)
```

**Methodological Highlights:**

- **Full-grid evaluation:** Evaluates every model in the stellar grid, avoiding MCMC burn-in and multimodal trapping.
- **Two-stage Gauss-Newton optimization:** Profiles distance out in magnitude space across all models before refining high-weight candidates in flux space.
- **Pointwise cluster mixture modeling:** Evaluates the cluster-versus-field mixture probability at each $(m, q)$ grid point prior to marginalizing over mass and binary mass ratio.
- **Antithetic Monte Carlo integration:** Integrates continuous spatial and dust priors in $(\ln d, A_V, R_V)$ using paired draws $(z, -z)$.

**See Also:**

- {doc}`/grid_generation` - Understanding the brute-force fitting algorithm
- {doc}`/population_modeling` - Cluster population fitting methodology
- {doc}`/photometric_offsets` - Empirical calibration procedures
- {doc}`/understanding_results` - Interpreting posterior distributions

```{eval-rst}
.. currentmodule:: brutus.analysis
```

## Individual Star Fitting

```{eval-rst}
.. autoclass:: BruteForce
   :members:
   :undoc-members:
   :show-inheritance:
```

## Population Analysis

```{eval-rst}
.. autofunction:: isochrone_population_loglike
```

```{eval-rst}
.. autofunction:: generate_isochrone_population_grid
```

```{eval-rst}
.. autofunction:: compute_isochrone_cluster_loglike
```

```{eval-rst}
.. autofunction:: compute_isochrone_outlier_loglike
```

```{eval-rst}
.. autofunction:: apply_isochrone_mixture_model
```

```{eval-rst}
.. autofunction:: marginalize_isochrone_grid
```

## Photometric Offsets

```{eval-rst}
.. autoclass:: PhotometricOffsetsConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: photometric_offsets
```

## Line-of-Sight Dust

```{eval-rst}
.. autofunction:: los_clouds_priortransform
```

```{eval-rst}
.. autofunction:: los_clouds_loglike_samples
```

```{eval-rst}
.. autofunction:: kernel_tophat
```

```{eval-rst}
.. autofunction:: kernel_gauss
```

```{eval-rst}
.. autofunction:: kernel_lorentz
```
