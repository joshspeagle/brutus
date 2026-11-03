# Star Clusters and Coeval Populations

Stars in a coeval open or globular cluster share a common age, chemical composition, heliocentric distance, and approximate foreground dust extinction. `brutus` models all stars in a cluster jointly along a single {term}`isochrone` {cite:p}`speagle2025` via {func}`~brutus.analysis.isochrone_population_loglike`, which evaluates the marginal log-likelihood of six population-level parameters so that it can be paired directly with an external optimizer, Markov chain Monte Carlo (MCMC) sampler, or nested sampler.

```{raw} html
:file: figures/cluster-cmd.svg
:class: figure-svg
```

:::{div} figure-caption
This simulated Gaia color-magnitude diagram ($G$ vs. $G_{\rm BP} - G_{\rm RP}$) shows a $4\,{\rm Gyr}$, solar-metallicity (${\rm [Fe/H]} = 0.00$) cluster at $d = 1\,{\rm kpc}$ with zero extinction ($A_V = 0$), drawn from the `grid_mist_v9` stellar grid. Cluster members (amber filled circles) include single stars along the solid black isochrone and unresolved main-sequence binaries lying up to $2.5\log_{10} 2 \approx 0.75\,{\rm mag}$ brighter (bounded by the dashed copper equal-mass binary sequence), while unassociated field stars (open red circles) are distributed across the color-magnitude plane.
:::

## Population Parameter Vector

{func}`~brutus.analysis.isochrone_population_loglike` takes a 6-element parameter vector `theta`:

| Index | Parameter | Units and Domain |
|---|---|---|
| `0` | Metallicity ${\rm [Fe/H]}$ | dex relative to solar |
| `1` | Log age $\log_{10}(t / {\rm yr})$ | dex ($t$ in years) |
| `2` | Visual extinction $A_V$ | magnitudes |
| `3` | Extinction curve shape $R_V$ | dimensionless |
| `4` | Heliocentric distance $d$ | **parsecs (pc)** |
| `5` | Field contamination fraction $f_{\rm field}$ | dimensionless in $[0, 1]$ |

In `isochrone_population_loglike`, distance $d$ is parameterized in **parsecs (pc)**, whereas `BruteForce` outputs individual stellar distances in kiloparsecs (kpc).

## Marginal Cluster Likelihood

Given population parameters $\boldsymbol\theta = ({\rm [Fe/H]}, \log_{10} t, A_V, R_V, d, f_{\rm field})$, `brutus` interpolates the MIST isochrone across a 2D grid of primary initial mass $m$ (indexed by EEP) and secondary binary mass ratio $q = M_2 / M_1$. Each grid node $(m, q)$ predicts broadband fluxes $F_b(m, q, \boldsymbol\theta)$ and the cluster parallax $\varpi = 1000 / d\,{\rm mas}$.

**Cluster member likelihood.** For star $i$ with observed fluxes $\hat F_{i,b} \pm \sigma_{i,b}$ and optional parallax $\hat\varpi_i \pm \sigma_{\varpi,i}$, the goodness-of-fit statistic at grid node $(m, q)$ is

$$
\chi^2_i(m, q) = \sum_b \frac{\left( \hat F_{i,b} - F_b(m, q) \right)^2}{\sigma_{i,b}^2} + \frac{\left( \hat\varpi_i - 1000/d \right)^2}{\sigma_{\varpi,i}^2}.
$$

By default (`dim_prior=True`), the cluster member likelihood is evaluated using the $\chi^2$ probability density with $k_i$ degrees of freedom (where $k_i$ equals the number of valid photometric bands plus $1$ when a valid parallax is present):

$$
P_c(\hat F_i \mid m, q, \boldsymbol\theta) = p_{\chi^2}\!\left( \chi^2_i(m, q);\, k_i \right).
$$

When `dim_prior=False`, $P_c$ is evaluated instead as the product of independent Gaussian densities across the valid bands and parallax.

**Field-star outlier model.** Because foreground and background field stars do not lie on the cluster isochrone, `brutus` models field contamination with an outlier density $P_o(\hat F_i)$. Under `dim_prior=True`, $P_o(\hat F_i)$ is set to the $\chi^2$ density evaluated at the critical threshold $\chi^2_{\max}(k_i)$ that a true cluster member exceeds with tail probability `p_value_cut` ($10^{-5}$ by default):

$$
P_o(\hat F_i) = p_{\chi^2}\!\left( \chi^2_{\max}(k_i);\, k_i \right),
\qquad
\int_{\chi^2_{\max}}^\infty p_{\chi^2}(x; k_i)\, dx = 10^{-5}.
$$

Any isochrone point whose $\chi^2_i(m, q)$ substantially exceeds $\chi^2_{\max}(k_i)$ contributes less likelihood than the background field model. Under `dim_prior=False`, the default outlier model is a uniform density over the observed flux range in each band (padded by `sigma_clip = 3` measurement errors) and over the observed parallax range. You can also supply a custom outlier log-likelihood callable via `outlier_model_func(flux, flux_err, stellar_params=..., parallax=..., parallax_err=..., mask=...)` returning an array of shape `(Nstars,)` or `(Ngrid, Nstars)`.

**Pointwise cluster-field mixture.** Each star $i$ has an effective prior cluster membership probability

$$
w_c = p_{{\rm mem},i}\,(1 - f_{\rm field}),
$$

where $p_{{\rm mem},i}$ is the external membership prior `cluster_prob` ($0.95$ by default, passed either as a scalar or as a per-star array derived from proper-motion or radial-velocity kinematics) and $f_{\rm field} = \theta_5$ is the fitted field contamination fraction. Combining the member and outlier likelihoods at each grid node $(m, q)$ before integration yields

$$
P(\hat F_i \mid m, q, \boldsymbol\theta) = w_c\, P_c(\hat F_i \mid m, q, \boldsymbol\theta) + (1 - w_c)\, P_o(\hat F_i).
$$

**Marginalization over mass and binary mass ratio.** Finally, `brutus` marginalizes over $(m, q)$ for each star using trapezoidal quadrature weights $\Delta m_j\,\Delta q_j$ normalized over the valid isochrone grid nodes, and sums the marginal log-likelihoods across all $N_{\rm stars}$ stars:

$$
P(\hat F_i \mid \boldsymbol\theta) = \frac{\sum_{j} P(\hat F_i \mid m_j, q_j, \boldsymbol\theta)\, \Delta m_j\, \Delta q_j}{\sum_{j} \Delta m_j\, \Delta q_j},
\qquad
\ln \mathcal{L}(\boldsymbol\theta) = \sum_i \ln P(\hat F_i \mid \boldsymbol\theta).
$$

Normalizing by the total valid grid volume $\sum_j \Delta m_j\,\Delta q_j$ at every evaluation of $\boldsymbol\theta$ ensures that changes in the number of valid isochrone nodes with age or metallicity do not bias the inferred population parameters or field fraction.

## Modeling Unresolved Binaries

An unresolved binary system combines the broadband fluxes of a primary star of mass $m$ and a coeval secondary star of mass $q\,m$, making an equal-mass binary ($q = 1$) brighter than a single star of the same color by $2.5 \log_{10} 2 \approx 0.75\,{\rm mag}$. By default, `isochrone_population_loglike` evaluates 21 secondary mass ratios from $q = 0$ to $q = 1$ (`smf_grid`) across 1,000 EEP nodes from $202$ to $808$ (`eep_grid`). Unresolved companions are added only for main-sequence and turnoff primaries up to ${\rm EEP} = 480$ (`eep_binary_max`); evolved post-turnoff stars (${\rm EEP} > 480$) are evaluated once as single stars and assigned the full marginal quadrature weight across $q$.

## Fitting a Cluster with MCMC or Nested Sampling

Download the MIST isochrone library (`fetch_isos()`) and bolometric-correction neural network (`fetch_nns()`) to construct a {class}`~brutus.core.StellarPop` instance ({doc}`stellar_models`):

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

The input arrays `flux` and `flux_err` have shape `(Nstars, Nbands)` in linear {term}`maggies <maggie>` ordered to match `filters`. Non-finite fluxes and entries masked via `mask` are excluded band-by-band, while malformed input shapes raise `ValueError`. If `theta` falls outside the valid domain of the isochrone library so that no valid grid models exist, `isochrone_population_loglike` issues a warning and returns `LOG_ZERO` (`-1e300`).

To sample the posterior distribution of $\boldsymbol\theta$ with [emcee](https://emcee.readthedocs.io), combine `isochrone_population_loglike` with a prior over `theta`:

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

For nested sampling with [dynesty](https://dynesty.readthedocs.io), pass `isochrone_population_loglike` as `loglikelihood` alongside a `prior_transform` mapping the unit hypercube $[0, 1]^6$ to `bounds`.

## Diagnostic Component Outputs

Passing `return_components=True` returns a `(lnl_total, components)` tuple where `components` is a dictionary containing intermediate likelihood arrays:

| Key | Description | Array Shape |
|---|---|---|
| `lnl_total` | Total marginal log-likelihood $\ln \mathcal{L}(\boldsymbol\theta)$ across the catalog | scalar `float` |
| `lnl_per_object` | Marginal log-likelihood $\ln P(\hat F_i \mid \boldsymbol\theta)$ for each star | `(Nstars,)` |
| `lnl_cluster`, `lnl_outlier`, `lnl_mixture` | Unmarginalized cluster, outlier, and mixture log-likelihoods at each grid node | `(Ngrid, Nstars)` |
| `isochrone_grid` | Dictionary containing generated model photometry, masses, $q$ values, and quadrature weights $\Delta m$, $\Delta q$ | `dict` |

Inspecting `lnl_per_object` identifies individual stars that are poorly described by the cluster isochrone.

## Computational Cost and Parallelization

Each likelihood evaluation interpolates the isochrone and evaluates all $N_{\rm stars}$ stars across up to $21,000$ `(EEP, q)` grid points ($1,000$ EEPs $\times$ $21$ binary mass ratios prior to mass and evolutionary phase cuts), so runtime scales linearly with $N_{\rm grid} \times N_{\rm stars}$. Passing coarser `eep_grid` or `smf_grid` arrays reduces per-call latency, though you should verify that coarser quadrature does not shift the posterior on your dataset (`bench/bench_populations.py` benchmarks runtime and quadrature convergence as a function of grid resolution). Because the internal matrix operations use multi-threaded BLAS routines, set `OMP_NUM_THREADS=1` whenever parallelizing across walker or live-point evaluations at the sampler level (such as `emcee.EnsembleSampler(..., pool=pool)`).

## Scope and Limitations

- **Single coeval population:** The built-in likelihood models one coeval cluster plus a field population; fitting multiple overlapping stellar populations requires a custom mixture wrapper.
- **Uniform foreground extinction:** All cluster members share a single $(A_V, R_V)$ pair without star-to-star differential reddening.
- **External kinematic priors:** Proper motions and radial velocities enter only through the per-star prior membership probabilities `cluster_prob`.
- **Detached, non-interacting binaries:** Binary companions contribute additive broadband flux without mass transfer or binary stellar evolution.
