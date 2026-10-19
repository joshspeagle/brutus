# Understanding results

`BruteForce.fit` writes one HDF5 file. This page describes each field, how to turn the draws into stellar properties, and how to check each fit.

## The output file

Each row is one star, in the order of the input. $N_d$ is the number of draws (`Ndraws`, 250 by default).

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
| `obj_log_evid` | $(N,)$ | log-{term}`evidence`: the log of the sum of the weights of the models kept by the selection |
| `obj_chi2min` | $(N,)$ | smallest $\chi^2$ over all models, with the parallax term if given |
| `obj_Nbands` | $(N,)$ | bands used, plus 1 if a parallax was given |
| `mc_ess` | $(N, N_d)$ | concentration of the Monte Carlo weights of each draw's model |

The `samps_*` fields are written only with `save_dar_draws=True` (the default). {doc}`scientific_background` describes how the draws are made.

The draws are already drawn from the posterior: give each one equal weight. Do not reweight them by `samps_logp` or `obj_log_post`.

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

`model_idx` points into the grid. The labels of the drawn models give the posterior of each stellar property:

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

Do not use $\chi^2 / N_{\rm bands}$. Use the p-value of `obj_chi2min` with $N_{\rm bands} - 3$ degrees of freedom, since each fit has 3 free parameters (scale, $A_V$, $R_V$):

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

Here `models` must come from the grid loaded with the bands of the fit, in the same order, and `offsets` are the `phot_offsets` of the fit (1 if none). A ratio much above 1 in a band means the parameters are poorly constrained there. {func}`~brutus.plotting.posterior_predictive` plots the same comparison.

### Monte Carlo weights

`mc_ess` is $1 / \sum_j w_j^2$ over the normalized Monte Carlo weights of a model. It is at most the number of Monte Carlo draws per model: `Nmc_prior` (50), or fewer when `mem_lim` reduces it. A value of 0 means every weight underflowed. A value near 1 means one point carries the whole weight: the Gaussian around the best fit and the priors overlap poorly, and the weight of that model is noisy. The draws come in antithetic pairs, so `mc_ess` measures how concentrated the weights are, not a count of independent draws.

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
