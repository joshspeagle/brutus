# Understanding Results

{meth}`~brutus.analysis.BruteForce.fit` writes its resampled posterior draws, maximum-likelihood extrinsic parameters, and fit diagnostics to a single HDF5 output file.

## Output HDF5 Schema

Each dataset in the output HDF5 file has leading dimension $N$ matching the order of the input catalog, where $N_d$ is the number of posterior draws saved per star (`Ndraws`, equal to $250$ by default):

| Dataset | Shape | Contents |
|---|---|---|
| `labels` | `(N,)` | Object identifiers passed via `data_labels` |
| `samps_dist` | `(N, N_d)` | Resampled posterior draws of heliocentric distance in kpc |
| `samps_red` | `(N, N_d)` | Resampled posterior draws of visual extinction $A_V$ in magnitudes |
| `samps_dred` | `(N, N_d)` | Resampled posterior draws of the extinction curve shape $R_V$ |
| `samps_logp` | `(N, N_d)` | Log-weight of the selected Monte Carlo importance sample for each draw (diagnostic only) |
| `model_idx` | `(N, N_d)` | Index of the stellar grid model selected for each draw |
| `ml_scale`, `ml_av`, `ml_rv` | `(N, N_d)` | Maximum-likelihood scale factor $s$, $A_V$, and $R_V$ of each draw's grid model |
| `ml_cov_sar` | `(N, N_d, 3, 3)` | Inverse Fisher covariance matrix of $(s, A_V, R_V)$ for each draw's grid model |
| `obj_log_post` | `(N, N_d)` | Marginalized log-posterior weight of each draw's grid model |
| `obj_log_evid` | `(N,)` | Log-{term}`evidence` (log of the sum of posterior model weights across retained models, divided by their systematic subsampling inclusion probabilities under `max_models`) |
| `obj_chi2min` | `(N,)` | Minimum $\chi^2$ across all evaluated grid models (including the parallax term when provided) |
| `obj_Nbands` | `(N,)` | Number of valid photometric bands used in the fit, plus $1$ when a valid parallax was included |
| `mc_ess` | `(N, N_d)` | Effective weight-concentration metric $1 / \sum_j w_j^2$ across the Monte Carlo importance samples of each draw's grid model |

The continuous draw arrays (`samps_*`) are written whenever `save_dar_draws=True` (the default; see {doc}`scientific_background` for the resampling procedure).

Because `samps_dist`, `samps_red`, `samps_dred`, and `model_idx` are already resampled in proportion to the posterior weights, treat them as equally weighted posterior draws without reweighting by `samps_logp` or `obj_log_post`. When more than `max_models` templates pass the Step 2 selection threshold, `obj_log_post` records each model's unadjusted log-posterior weight while both the resampled draws and `obj_log_evid` incorporate the inverse-probability weighting correction for systematic subsampling.

When `running_io=True` (the default), each star's results are flushed to the HDF5 file immediately upon completion; stars that have not yet been fitted (or that had zero valid bands) retain the sentinel value `model_idx = -99`.

## Computing Posterior Summaries

Use {func}`~brutus.utils.quantile` to compute marginal posterior medians and credible intervals from the equally weighted draws:

```python
import h5py
import numpy as np

from brutus.utils import quantile

with h5py.File("results.h5", "r") as f:
    dist = f["samps_dist"][:]
    av = f["samps_red"][:]

d16, d50, d84 = quantile(dist[0], [0.16, 0.5, 0.84])
```

## Extracting Intrinsic Stellar Properties

Each integer in `model_idx` indexes a row in the stellar grid, so indexing the `labels` structured array returned by {func}`~brutus.data.load_models` with `model_idx` yields the corresponding posterior draws of any intrinsic or predicted stellar quantity:

```python
from brutus.data import load_models

models, labels, label_mask = load_models("grid_mist_v9.h5", filters=filters)

with h5py.File("results.h5", "r") as f:
    idx = f["model_idx"][0]

mass = labels["mini"][idx]             # initial mass in Msun
age = 10 ** labels["loga"][idx] / 1e9  # stellar age in Gyr
teff = 10 ** labels["logt"][idx]       # effective temperature in K
```

Ensure that `load_models` is called with the same phase and binary cuts (`include_ms`, `include_postms`, `include_binaries`) used during fitting so that row indices in `model_idx` map to the identical grid models.

## Fit Quality and Diagnostic Checks

### Goodness of Fit ($\chi^2$ Survival Probability)

Evaluate goodness of fit using the $\chi^2$ survival-function p-value of `obj_chi2min` with $N_{\rm bands} - 3$ degrees of freedom rather than the reduced $\chi^2 / N_{\rm bands}$, because each grid model optimizes 3 continuous parameters ($s$, $A_V$, and $R_V$):

```python
from scipy.stats import chi2

with h5py.File("results.h5", "r") as f:
    chi2min = f["obj_chi2min"][:]
    nbands = f["obj_Nbands"][:]

pvalue = chi2.sf(chi2min, np.maximum(nbands - 3, 1))
```

| $\chi^2$ p-value | Fit Quality | Typical Astrophysical or Instrumental Causes |
|---|---|---|
| $p > 10^{-3}$ | Good fit | Consistent with a single-star template in the grid |
| $10^{-6} < p \le 10^{-3}$ | Marginal fit | Underestimated photometric errors or a single mild outlier band |
| $p \le 10^{-6}$ | Poor fit | Unresolved binary, pre-main-sequence star, white dwarf, or blended/saturated photometry |

Because $\chi^2$ p-values are sensitive to the assumed uncertainty model, omitting the $0.02\text{--}0.03\,{\rm mag}$ systematic calibration uncertainty of the stellar models ({doc}`photometric_offsets`) from the measurement errors will cause bright stars with small formal errors to exhibit artificially low p-values.

### Posterior Predictive Checks

To verify how well the posterior models reproduce each passband individually, compare the dispersion and residuals of the predicted fluxes across posterior draws against the fitted measurement uncertainties:

```python
from brutus.core import get_seds

i = 0  # star index
with h5py.File("results.h5", "r") as f:
    idx, av, rv, dist = (f[k][i] for k in ("model_idx", "samps_red", "samps_dred", "samps_dist"))

pred = get_seds(models[idx], av=av, rv=rv, return_flux=True) / dist[:, None] ** 2
ratio = pred.std(axis=0) / (flux_err[i] * offsets)  # compared against offset-scaled errors
```

Here `models` must be loaded with the same passbands and column ordering used in the fit, and `offsets` is the `phot_offsets` vector applied during fitting ($1.0$ if none). A posterior predictive dispersion much larger than the measurement error in a given band indicates that the available passbands leave the SED weakly constrained at that wavelength.

```{raw} html
:file: figures/star-posterior.svg
:class: figure-svg
```

:::{div} figure-caption
This figure summarizes the posterior inference for one star from the Orion sightline in {doc}`scientific_background`. The left panel shows the posterior draws of distance and extinction, where about three quarters of the draws fall within the $1\sigma$ parallax interval and a secondary tail extends toward larger distances and lower extinction. The right panel shows the band-by-band residuals between predicted and observed magnitudes, spanning about $0.1\,{\rm mag}$ (two to three times the measurement errors in several bands).
:::

Calling {func}`~brutus.plotting.posterior_predictive` generates a posterior predictive SED comparison figure directly from the fit outputs.

### Monte Carlo Importance Weight Concentration (`mc_ess`)

For each grid model, `mc_ess` records $1 / \sum_j w_j^2$ evaluated over its normalized Step 3 Monte Carlo importance weights, bounded above by the number of Monte Carlo draws per model (`Nmc_prior`, equal to $50$ by default or fewer when capped by `mem_lim`). A value of `mc_ess == 0` indicates that every Monte Carlo draw for that model received zero prior weight (for example, when all draws fell outside `avlim`, `rvlim`, or the valid distance support). Values of `mc_ess` near $1$ indicate that a single Monte Carlo draw carries nearly all of the model's weight because the local Gaussian likelihood approximation around the best fit and the position-dependent priors overlap poorly. Since the Monte Carlo samples are drawn in antithetic pairs $(z, -z)$, `mc_ess` serves as a diagnostic of importance-weight concentration rather than a count of independent random draws.

### Parallax Consistency and Prior Sensitivity

When a parallax measurement was included in the fit, compare the observed parallax $\hat\varpi \pm \sigma_\varpi$ against the posterior distribution of $1 / d$. A strong discrepancy between the photometric distance and the astrometric parallax typically signals an unresolved binary (whose extra flux makes the star appear closer than $1 / \hat\varpi$), an astrometric solution affected by orbital motion (check the Gaia RUWE diagnostic), or an object outside the stellar grid domain. For faint stars lacking informative parallaxes, compare fits under the default Galactic prior and a flat distance prior ({doc}`priors`) to determine whether the inferred distances are prior-dominated.

## Diagnostic Plotting Utilities

{mod}`brutus.plotting` provides diagnostic visualization routines that accept the HDF5 output arrays directly:

| Plotting Function | Visualization |
|---|---|
| {func}`~brutus.plotting.cornerplot` | Corner plot of marginal and pairwise posteriors over intrinsic stellar parameters, distance, and extinction |
| {func}`~brutus.plotting.dist_vs_red` | Joint 2D posterior density of distance and extinction ($A_V$ or $E(B-V)$) for a single star |
| {func}`~brutus.plotting.posterior_predictive` | Posterior predictive SED distribution overlaid on the observed broadband photometry |
| {func}`~brutus.plotting.bin_pdfs_distred` | Binned stack of stellar distance-extinction posteriors along a line of sight |
