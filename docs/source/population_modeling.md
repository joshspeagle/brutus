# Star clusters

The stars of a cluster share one age, one metallicity, one distance and, nearly, one extinction. brutus fits them together with one {term}`isochrone` {cite:p}`speagle2025`. It does not run a sampler for this: {func}`~brutus.analysis.isochrone_population_loglike` returns the log-likelihood of six cluster parameters, and you pass it to an optimizer or a sampler of your choice.

```{raw} html
:file: figures/cluster-cmd.svg
:class: figure-svg
```

:::{div} figure-caption
A simulated cluster, drawn from the `BruteForce` model grid (`grid_mist_v9`) rather than the isochrones that the likelihood uses. Unresolved binaries lie up to 0.75 mag above the single-star isochrone; field stars lie anywhere. The likelihood models all three.
:::

## The cluster parameters

| Index | Parameter | Unit |
|---|---|---|
| 0 | [Fe/H] | dex |
| 1 | $\log_{10}$ age | age in years |
| 2 | $A_V$ | mag |
| 3 | $R_V$ | |
| 4 | distance $d$ | **pc** |
| 5 | field fraction $f_{\rm field}$ | 0 to 1 |

The distance is in parsecs here, while `BruteForce` uses kiloparsecs.

## The likelihood

For given cluster parameters $\boldsymbol\theta$, brutus builds an isochrone on a grid of initial mass $m$ and secondary mass fraction $q$ (the mass ratio of a binary). Each grid point predicts fluxes $F_b(m, q, \boldsymbol\theta)$ and the parallax $1000/d$ mas.

**Cluster member.** For star $i$ with fluxes $\hat F_{i,b} \pm \sigma_{i,b}$ and parallax $\hat\varpi_i \pm \sigma_{\varpi,i}$,

$$
\chi^2_i(m, q) = \sum_b \frac{\left( \hat F_{i,b} - F_b(m, q) \right)^2}{\sigma_{i,b}^2} + \frac{\left( \hat\varpi_i - 1000/d \right)^2}{\sigma_{\varpi,i}^2},
$$

and the cluster likelihood is the $\chi^2$ density with $k_i$ degrees of freedom, where $k_i$ counts the valid bands plus 1 for a parallax:

$$
P_c(\hat F_i \mid m, q, \boldsymbol\theta) = p_{\chi^2}\!\left( \chi^2_i(m, q);\, k_i \right).
$$

With `dim_prior=False`, the likelihood is instead the product of Gaussian densities in each band and in the parallax.

**Field star.** A field star is not on the isochrone. Its likelihood is the $\chi^2$ density at the value that a cluster member exceeds with probability $10^{-5}$ (`p_value_cut`):

$$
P_o(\hat F_i) = p_{\chi^2}\!\left( \chi^2_{\max}(k_i);\, k_i \right),
\qquad
\int_{\chi^2_{\max}}^\infty p_{\chi^2}(x; k_i)\, dx = 10^{-5}.
$$

A grid point whose $\chi^2$ is well above $\chi^2_{\max}$ describes the star worse than the field model does. With `dim_prior=False`, the field model is uniform over the observed range of fluxes in each band, widened by 3 errors (`sigma_clip`), and over the range of parallaxes. `outlier_model_func` replaces it with your own: it is called as `outlier_model_func(flux, flux_err, stellar_params=..., parallax=..., parallax_err=..., mask=...)` and returns the log-likelihood with shape `(Nstars,)` or `(Ngrid, Nstars)`.

**Mixture.** Each star is a member with prior probability

$$
w_c = p_{{\rm mem},i}\,(1 - f_{\rm field}),
$$

where $p_{{\rm mem},i}$ is `cluster_prob` (0.95 by default) and $f_{\rm field}$ is fitted. At each grid point,

$$
P(\hat F_i \mid m, q, \boldsymbol\theta) = w_c\, P_c(\hat F_i \mid m, q, \boldsymbol\theta) + (1 - w_c)\, P_o(\hat F_i).
$$

`cluster_prob` can be one number or one value per star, for example a membership probability from proper motions.

**Marginalization.** brutus integrates each star over the grid with a prior uniform in $m$ and $q$, normalized over the valid grid points:

$$
P(\hat F_i \mid \boldsymbol\theta) = \frac{\sum_{j} P(\hat F_i \mid m_j, q_j, \boldsymbol\theta)\, \Delta m_j\, \Delta q_j}{\sum_{j} \Delta m_j\, \Delta q_j},
\qquad
\ln \mathcal{L}(\boldsymbol\theta) = \sum_i \ln P(\hat F_i \mid \boldsymbol\theta).
$$

The prior on mass is uniform, not an initial mass function. The normalization keeps the grid volume, which changes with $\boldsymbol\theta$, from biasing the parameters and the field fraction.

## Binaries

A binary's flux is the sum of the fluxes of its two stars. An equal-mass binary ($q = 1$) is therefore $2.5 \log_{10} 2 = 0.75$ mag brighter than a single star of the same colour. The default grid has 21 values of $q$ from 0 to 1 (`smf_grid`) and 1000 EEPs from 202 to 808 (`eep_grid`). Only primaries up to EEP 480 (`eep_binary_max`) get a companion. Later phases are stored once, with the full weight of all $q$ values.

## Fitting a cluster

`fetch_isos()` downloads the MIST isochrones and `fetch_nns()` the neural network ({doc}`stellar_models`):

```python
import numpy as np

from brutus.analysis import isochrone_population_loglike
from brutus.core import Isochrone, StellarPop
from brutus.data import fetch_isos, fetch_nns
from brutus.data.filters import gaia, tmass

pop = StellarPop(
    Isochrone(mistfile=fetch_isos()),
    filters=gaia + tmass,
    nnfile=fetch_nns(),
)

theta = np.array([0.0, 9.6, 0.1, 3.3, 1000.0, 0.05])  # [Fe/H], log age, A(V), R(V), pc, f_field
lnl = isochrone_population_loglike(
    theta, pop, flux, flux_err,
    parallax=parallax, parallax_err=parallax_err,
)
```

`flux` and `flux_err` have shape `(Nstars, Nbands)` in {term}`maggies <maggie>`, in the order of `filters`. Masked or non-finite fluxes are skipped band by band; `mask` sets more. Malformed inputs raise `ValueError`. If no isochrone model is valid at `theta`, for example outside the isochrone grid, the function warns and returns `LOG_ZERO` (-1e300).

To sample the posterior, add a prior. With [emcee](https://emcee.readthedocs.io):

```python
import emcee

bounds = np.array([[-2.5, 0.5], [8.0, 10.1], [0.0, 3.0], [2.0, 5.0], [100.0, 1e4], [0.0, 0.5]])

def log_posterior(theta):
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        return -np.inf
    return isochrone_population_loglike(
        theta, pop, flux, flux_err, parallax=parallax, parallax_err=parallax_err
    )

rng = np.random.default_rng(42)
nwalkers = 32
start = theta + 1e-3 * rng.standard_normal((nwalkers, 6))
sampler = emcee.EnsembleSampler(nwalkers, 6, log_posterior)
sampler.run_mcmc(start, 5000)
samples = sampler.get_chain(discard=1000, flat=True)
```

A nested sampler such as [dynesty](https://dynesty.readthedocs.io) takes the same likelihood with a prior transform that maps the unit cube to `bounds`.

## Diagnostics

`return_components=True` also returns a dictionary:

| Key | Content | Shape |
|---|---|---|
| `lnl_total` | the same total $\ln \mathcal{L}$ | scalar |
| `lnl_per_object` | $\ln P(\hat F_i \mid \boldsymbol\theta)$ per star | `(Nstars,)` |
| `lnl_cluster`, `lnl_outlier`, `lnl_mixture` | the three terms at each grid point | `(Ngrid, Nstars)` |
| `isochrone_grid` | the grid: photometry, masses, $q$ and the weights $\Delta m$, $\Delta q$ | |

The stars with the lowest `lnl_per_object` fit the cluster worst.

## Cost

Each call builds the isochrone grid and evaluates every star at every grid point, so the cost grows with the number of grid points times the number of stars. The default grid has at most 21,000 points (1000 EEPs × 21 values of $q$), fewer after the binary and mass cuts. A coarser grid (`eep_grid`, `smf_grid`) is faster but less accurate; check that the result does not change before you use one. The matrix products inside can use several threads through BLAS. If you parallelize at the level of the sampler, for example with `emcee.EnsembleSampler(..., pool=pool)`, set `OMP_NUM_THREADS=1` so the two do not compete. `bench/bench_populations.py` measures the cost and the convergence with grid size on your machine.

## Limitations

- **One population.** The model has one cluster and a field; two populations need your own likelihood.
- **One extinction.** All members share $A_V$ and $R_V$; differential reddening is not modelled.
- **No kinematics.** Proper motions and radial velocities enter only through `cluster_prob`.
- **Unresolved binaries only.** Companions add flux; there is no mass transfer or orbital information.
