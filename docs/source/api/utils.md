# Utilities Module (`brutus.utils`)

`brutus.utils` provides the mathematical, photometric, and Monte Carlo sampling routines used across `brutus`.

**Module Organization:**

- **`brutus.utils.photometry`:** Conversions between magnitudes, luptitudes, and linear flux densities in maggies (`magnitude`, `inv_magnitude`, `luptitude`, `inv_luptitude`, `add_mag`), plus Gaussian and outlier photometric log-likelihoods (`phot_loglike`, `chisquare_outlier_loglike`, `uniform_outlier_loglike`).
- **`brutus.utils.math`:** Analytical $3 \times 3$ matrix inversion and eigenvalue regularization (`inverse3`, `isPSD`), coordinate transformations (`galactic_to_galactocentric_cyl`), truncated normal and $\chi^2$ densities (`truncnorm_pdf`, `truncnorm_logpdf`, `chisquare_logpdf`), and numerical bounds (`LOG_ZERO`, `AVLIM`, `RVLIM`).
- **`brutus.utils.sampling`:** Weighted midpoint-CDF quantiles (`quantile`), antithetic multivariate normal sampling (`sample_multivariate_normal`), and $(s, A_V, R_V)$ posterior resampling (`draw_sar`).

**Typical Usage:**

**Photometry utilities:**

```python
from brutus.utils.photometry import magnitude, add_mag, phot_loglike
import numpy as np

# Convert linear flux densities (maggies) to magnitudes
flux = np.array([1e-10, 5e-11, 2e-11])
flux_err = np.array([1e-12, 5e-13, 3e-13])
mags, mag_errs = magnitude(flux, flux_err)

# Combine magnitudes of two unresolved binary components
mag1, mag2 = 5.0, 6.0
mag_combined = add_mag(mag1, mag2)

# Evaluate photometric log-likelihood for data of shape (Nobj, Nfilt)
# against a model grid of shape (Nmod, Nfilt), returning (Nobj, Nmod)
obs_flux = np.array([[1.2e-10, 4.8e-11, 2.1e-11]])
obs_err = np.array([[1e-12, 5e-13, 3e-13]])
model_flux = np.array([[1.15e-10, 5.0e-11, 2.0e-11]])
lnl = phot_loglike(obs_flux, obs_err, model_flux, dim_prior=True)
```

**Mathematical utilities:**

```python
from brutus.utils.math import inverse3, isPSD, chisquare_logpdf
import numpy as np

cov_matrix = np.array([[1.0, 0.1, 0.0],
                       [0.1, 2.0, 0.2],
                       [0.0, 0.2, 1.5]])
is_valid = isPSD(cov_matrix)
inv_cov = inverse3(cov_matrix, regularize=True)

chi2_values = np.array([1.0, 5.0, 10.0])
log_prob = chisquare_logpdf(chi2_values, dof=3)
```

**Sampling utilities:**

```python
from brutus.utils.sampling import sample_multivariate_normal, quantile
import numpy as np

rng = np.random.default_rng(42)
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.5], [0.5, 2.0]])
samples = sample_multivariate_normal(mean, cov, size=1000, rstate=rng)

data = rng.standard_normal(1000)
weights = rng.random(1000)
q16, q50, q84 = quantile(data, [0.16, 0.5, 0.84], weights=weights)
```

**See Also:**

- {doc}`/grid_generation` - How utilities are used in fitting algorithm
- {doc}`/understanding_results` - Sampling and quantile calculations

```{eval-rst}
.. py:module:: brutus.utils
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
