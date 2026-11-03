# Understanding results

`BruteForce.fit` writes its posterior draws, maximum-likelihood parameters, and fit diagnostics to a single HDF5 file.

## The output file

Each row corresponds to one star in the order of the input catalog, where $N_d$ is the number of posterior draws (`Ndraws`, 250 by default):

| Field | Shape | Content |
|---|---|---|
| `labels` | as passed | the star IDs you passed as `data_labels` |
| `samps_dist` | $(N, N_d)$ | posterior draws of distance, in kpc |
| `samps_red` | $(N, N_d)$ | posterior draws of $A_V$, in mag |
| `samps_dred` | $(N, N_d)$ | posterior draws of $R_V$ |
| `samps_logp` | $(N, N_d)$ | log-weight of each draw's Monte Carlo point (diagnostic only) |
| `model_idx` | $(N, N_d)$ | grid model of each draw |
| `ml_scale`, `ml_av`, `ml_rv` | $(N, N_d)$ | best-fit scale factor, $A_V$ and $R_V$ of each draw's model |
| `ml_cov_sar` | $(N, N_d, 3, 3)$ | covariance of (scale, $A_V$, $R_V$) of each draw's model |
| `obj_log_post` | $(N, N_d)$ | log posterior weight of each draw's model |
| `obj_log_evid` | $(N,)$ | log-{term}`evidence` (log of the sum of the weights of the models kept by the selection, each divided by its inclusion probability under `max_models`) |
| `obj_chi2min` | $(N,)$ | smallest $\chi^2$ over all models, with the parallax term if given |
| `obj_Nbands` | $(N,)$ | bands used, plus 1 if a parallax was given |
| `mc_ess` | $(N, N_d)$ | concentration of the Monte Carlo weights of each draw's model |

The `samps_*` fields are written only with `save_dar_draws=True` (the default). {doc}`scientific_background` describes how the draws are made.

Because the saved samples are already drawn in proportion to their posterior weights, treat them as equally weighted posterior draws without reweighting by `samps_logp` or `obj_log_post`. When more than `max_models` models pass the selection, `obj_log_post` still records each model's own log-posterior, while the resampled draws and `obj_log_evid` already include the inverse-probability correction for subsampling.

With `running_io=True` (the default), each star is written as soon as it is fitted; rows not yet fitted hold `model_idx = -99`.

## Summaries

{func}`~brutus.utils.quantile` gives medians and intervals:

```python
import h5py
import numpy as np

from brutus.utils import quantile

with h5py.File("results.h5", "r") as f:
    dist = f["samps_dist"][:]
    av = f["samps_red"][:]

d16, d50, d84 = quantile(dist[0], [0.16, 0.5, 0.84])
```

## Stellar properties

Each entry in `model_idx` indexes a model in the grid, so indexing `labels` with `model_idx` yields the posterior draws of any stellar property:

```python
from brutus.data import load_models

models, labels, label_mask = load_models("grid_mist_v9.h5", filters=filters)

with h5py.File("results.h5", "r") as f:
    idx = f["model_idx"][0]

mass = labels["mini"][idx]          # initial mass, M☉
age = 10 ** labels["loga"][idx] / 1e9  # Gyr
teff = 10 ** labels["logt"][idx]    # K
```

Load the grid with the same arguments as for the fit, so that the indices refer to the same models.

## Checking a fit

### Goodness of fit

Evaluate goodness of fit using the $\chi^2$ survival-function p-value of `obj_chi2min` with $N_{\rm bands} - 3$ degrees of freedom rather than $\chi^2 / N_{\rm bands}$, because each model fits 3 free parameters (scale, $A_V$, and $R_V$):

```python
from scipy.stats import chi2

with h5py.File("results.h5", "r") as f:
    chi2min = f["obj_chi2min"][:]
    nbands = f["obj_Nbands"][:]

pvalue = chi2.sf(chi2min, np.maximum(nbands - 3, 1))
```

| p-value | Fit | Typical cause of a poor fit |
|---|---|---|
| $p > 10^{-3}$ | good | |
| $10^{-6} < p \le 10^{-3}$ | marginal | errors slightly too small, one bad band |
| $p \le 10^{-6}$ | poor | a binary, a young star, a white dwarf, blended or saturated photometry |

The p-value is only as good as the errors. Without the systematic errors of the models ({doc}`photometric_offsets`), many good fits of bright stars look poor.

### Posterior predictive check

Compare, band by band, the spread of the predicted fluxes across the draws with the measurement error:

```python
from brutus.core import get_seds

i = 0  # one star
with h5py.File("results.h5", "r") as f:
    idx, av, rv, dist = (f[k][i] for k in ("model_idx", "samps_red", "samps_dred", "samps_dist"))

pred = get_seds(models[idx], av=av, rv=rv, return_flux=True) / dist[:, None] ** 2
ratio = pred.std(axis=0) / (flux_err[i] * offsets)  # the errors as fitted
```

Here `models` must come from the grid loaded with the bands of the fit, in the same order, and `offsets` are the `phot_offsets` of the fit (1 if none). A ratio much above 1 in a band means the parameters are poorly constrained there.

```{raw} html
:file: figures/star-posterior.svg
:class: figure-svg
```

:::{div} figure-caption
Posterior summary for one star from the Orion sightline in {doc}`scientific_background`. The left panel shows the posterior draws of distance and extinction, where about three quarters of the draws fall within the $1\sigma$ parallax interval and a secondary tail extends toward larger distances and lower extinction. The right panel shows the band-by-band residuals between predicted and observed magnitudes, spanning about 0.1 mag (two to three times the measurement errors in several bands).
:::

{func}`~brutus.plotting.posterior_predictive` plots the same comparison.

### Monte Carlo weights

`mc_ess` is $1 / \sum_j w_j^2$ over the normalized Monte Carlo weights of a model. It is at most the number of Monte Carlo draws per model (`Nmc_prior`, 50 by default, or fewer when `mem_lim` reduces it). A value of 0 means no Monte Carlo draw of the model has nonzero weight, for example when every draw falls outside `avlim`, `rvlim`, or the distance range. A value near 1 indicates that a single draw carries nearly all the weight because the local Gaussian approximation around the best fit and the position-dependent priors overlap poorly. Because the Monte Carlo draws come in antithetic pairs, `mc_ess` measures weight concentration rather than a count of independent draws.

### Parallax and prior

If you gave a parallax, compare it with $1/d$ of the draws. A large difference points to a binary, a bad parallax (check the Gaia RUWE) or a model failure. For faint stars without a parallax, the priors can set the distance; {doc}`priors` shows how to test this.

## Plots

{mod}`brutus.plotting` draws the common figures from the output:

| Function | Figure |
|---|---|
| {func}`~brutus.plotting.cornerplot` | the posterior of the stellar properties, distance and extinction |
| {func}`~brutus.plotting.dist_vs_red` | the joint posterior of distance and $A_V$ |
| {func}`~brutus.plotting.posterior_predictive` | the predicted SED against the data |
| {func}`~brutus.plotting.bin_pdfs_distred` | binned distance-extinction posteriors of many stars |
