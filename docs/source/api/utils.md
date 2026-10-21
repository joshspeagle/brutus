# Utilities Module (`brutus.utils`)

The utils module provides low-level mathematical, photometric, and sampling utilities that support brutus's core functionality. These functions handle magnitude/flux conversions, likelihood calculations, statistical distributions, and posterior sampling.

**Module Organization:**

- **\`\`brutus.utils.photometry\`\`**: Magnitude/flux conversions and photometric likelihoods
- **\`\`brutus.utils.math\`\`**: Mathematical utilities (matrix operations, distributions, special functions)
- **\`\`brutus.utils.sampling\`\`**: Posterior sampling and Monte Carlo methods

**Typical Usage:**

Most users won't call these utilities directly (they're used internally by `BruteForce` and other high-level classes). However, they're useful for:

- Custom analysis scripts
- Implementing new fitting methods
- Debugging and validation

**Photometry utilities**:

```python
from brutus.utils.photometry import magnitude, add_mag, phot_loglike
import numpy as np

# Convert flux to magnitude (returns (mag, mag_err))
flux = np.array([1e-10, 5e-11, 2e-11])  # maggies
flux_err = np.array([1e-12, 5e-13, 3e-13])  # maggies
mags, mag_errs = magnitude(flux, flux_err)

# Add magnitudes (combine flux from binaries)
mag1, mag2 = 5.0, 6.0
mag_combined = add_mag(mag1, mag2)  # Brighter than either component

# Compute photometric log-likelihood. Data are 2-D with shape
# (Nobj, Nfilt); models are either a shared (Nmod, Nfilt) grid or a
# per-object (Nobj, Nmod, Nfilt) array. Returns shape (Nobj, Nmod).
obs_flux = np.array([[1.2e-10, 4.8e-11, 2.1e-11]])
obs_err = np.array([[1e-12, 5e-13, 3e-13]])
model_flux = np.array([[1.15e-10, 5.0e-11, 2.0e-11]])

lnl = phot_loglike(obs_flux, obs_err, model_flux, dim_prior=True)
```

**Mathematical utilities**:

```python
from brutus.utils.math import inverse3, chisquare_logpdf

# Fast 3x3 matrix inversion (used in covariance calculations)
cov_matrix = np.array([[1.0, 0.1, 0.0],
                       [0.1, 2.0, 0.2],
                       [0.0, 0.2, 1.5]])
inv_cov = inverse3(cov_matrix)

# Chi-square log-PDF (used in outlier models)
chi2_values = np.array([1.0, 5.0, 10.0])
dof = 3
log_prob = chisquare_logpdf(chi2_values, dof)
```

**Sampling utilities**:

```python
from brutus.utils.sampling import sample_multivariate_normal, quantile

# Draw samples from multivariate normal
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.5], [0.5, 2.0]])
samples = sample_multivariate_normal(mean, cov, size=10000)

# Compute weighted quantiles
data = np.random.randn(10000)
weights = np.random.rand(10000)
q16, q50, q84 = quantile(data, [0.16, 0.5, 0.84], weights=weights)
```

**Performance Notes:**

- Many functions are JIT-compiled with `numba` for speed
- Matrix operations optimized for small dimensions (common in stellar fitting)
- Photometric likelihoods vectorized for batch processing

**See Also:**

- {doc}`/grid_generation` - How utilities are used in fitting algorithm
- {doc}`/understanding_results` - Sampling and quantile calculations

```{eval-rst}
.. currentmodule:: brutus.utils
```

## Photometry Functions

```{eval-rst}
.. autofunction:: magnitude
```

```{eval-rst}
.. autofunction:: inv_magnitude
```

```{eval-rst}
.. autofunction:: luptitude
```

```{eval-rst}
.. autofunction:: inv_luptitude
```

```{eval-rst}
.. autofunction:: add_mag
```

```{eval-rst}
.. autofunction:: phot_loglike
```

```{eval-rst}
.. autofunction:: chisquare_outlier_loglike
```

```{eval-rst}
.. autofunction:: uniform_outlier_loglike
```

## Constants

```{eval-rst}
.. autodata:: brutus.utils.math.LOG_ZERO
.. autodata:: brutus.utils.math.AVLIM
.. autodata:: brutus.utils.math.RVLIM
```

## Mathematical Functions

```{eval-rst}
.. autofunction:: galactic_to_galactocentric_cyl
```

```{eval-rst}
.. autofunction:: pm_equatorial_to_galactic
```

```{eval-rst}
.. autofunction:: inverse3
```

```{eval-rst}
.. autofunction:: isPSD
```

```{eval-rst}
.. autofunction:: chisquare_logpdf
```

```{eval-rst}
.. autofunction:: truncnorm_pdf
```

```{eval-rst}
.. autofunction:: truncnorm_logpdf
```

## Sampling Utilities

```{eval-rst}
.. autofunction:: quantile
```

```{eval-rst}
.. autofunction:: sample_multivariate_normal
```

```{eval-rst}
.. autofunction:: draw_sar
```

## Submodules

For advanced users who need access to internal implementations:

```{eval-rst}
.. automodule:: brutus.utils.math
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.utils.photometry
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.utils.sampling
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```
