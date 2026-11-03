# Photometric Offsets

Synthetic stellar atmosphere spectra, filter transmission curves, and survey zero-point calibrations exhibit systematic residuals of a few percent across broadband passbands. `brutus` corrects these zero-point discrepancies by applying an empirical multiplicative flux offset per passband to the observed photometry and its uncertainties.

In addition to per-band flux offsets, `brutus` applies an empirical calibration to the effective temperatures and radii of low-mass main-sequence stars ({doc}`stellar_models`).

## Default Zero-Point Offsets

Calling `fetch_offsets()` downloads the empirical zero-point offsets calibrated for the default `mist_v9` grid {cite:p}`speagle2025` using open clusters and solar-neighborhood field stars with high-precision Gaia parallaxes (Section 5 of Speagle et al. 2025).

```{raw} html
:file: figures/offsets.svg
:class: figure-svg
```

:::{div} figure-caption
This bar chart displays the empirical multiplicative flux offsets (expressed as percentage corrections $(\text{offset} - 1) \times 100\%$) from `offsets_mist_v9.txt` across the calibrated optical and infrared passbands. Positive percentage offsets increase the observed fluxes, while passbands absent from the calibration file receive a unit scale factor ($1.0$).
:::

Load the offsets in the column order of your photometric catalog with {func}`~brutus.data.load_offsets` and pass the resulting array to `BruteForce.fit` via `phot_offsets`:

```python
from brutus.data import fetch_offsets, load_offsets

offsets = load_offsets(fetch_offsets(), filters=filters)
fitter.fit(..., phot_offsets=offsets)
```

Because `fit()` multiplies both `data` and `data_err` by `phot_offsets`, applying zero-point offsets shifts the flux calibration without altering the per-band signal-to-noise ratio.

## Including Systematic Offset Uncertainties

The empirical zero-point offsets carry residual calibration uncertainties of approximately $0.02\,{\rm mag}$ in optical bands and $0.03\,{\rm mag}$ in near-infrared bands (Table 5 of Speagle et al. 2025). Because `brutus` does not add these systematic floors automatically, propagate them into your linear flux uncertainties in quadrature before fitting:

$$
\sigma_b' = \sqrt{\sigma_b^2 + \left( \frac{\ln 10}{2.5}\, \hat F_b\, \sigma_{{\rm sys},b} \right)^2},
$$

where $\sigma_{{\rm sys},b}$ is the systematic magnitude uncertainty in band $b$ ({doc}`quickstart` demonstrates this step for an 11-band Gaia + Pan-STARRS + 2MASS catalog). Without this systematic error floor, bright stars with small photon-counting uncertainties receive artificially large $\chi^2$ values and overly narrow posteriors dominated by unmodeled atmosphere and filter systematics.

## Calibrating Custom Photometric Offsets

To derive empirical zero-point offsets for a new photometric survey or a custom model grid, fit a calibration sample of well-measured stars and compare the observed fluxes against the posterior model predictions using {func}`~brutus.analysis.photometric_offsets`:

```python
import h5py
import numpy as np

from brutus.analysis import photometric_offsets
from brutus.utils import magnitude

# Construct the mask of bands actually used during fitting (BruteForce drops
# bands with magnitude errors >= 0.25 mag after applying phot_offsets).
mag, mag_err = magnitude(flux * offsets_used_in_fit, flux_err * offsets_used_in_fit)
mask_used = mask & (mag_err < 0.25)

with h5py.File("results.h5", "r") as f:
    idxs = f["model_idx"][:]
    avs, rvs, dists = f["samps_red"][:], f["samps_dred"][:], f["samps_dist"][:]

offsets, offset_errors, n_used = photometric_offsets(
    flux, flux_err, mask_used, models, idxs, avs, rvs, dists,
    old_offsets=offsets_used_in_fit,
    rng=np.random.default_rng(42),
)
```

Here `models` is the coefficient array returned by {func}`~brutus.data.load_models`, and `flux` and `flux_err` are the uncorrected catalog fluxes and uncertainties.

For each passband $b$, {func}`~brutus.analysis.photometric_offsets` evaluates the model-to-data flux ratio $r = F_b / \hat F_b$ across all calibration stars and posterior draws and estimates the band offset from the weighted median ratio. Because including band $b$ in the original fit pulls the posterior draws toward the observed flux in band $b$ (which would bias the residual toward the offset assumed during fitting), each posterior draw is importance-reweighted to the leave-one-out posterior excluding band $b$:

$$
w \propto \frac{\mathcal{L}_{-b}}{\mathcal{L}} = \exp\!\left(\tfrac{1}{2}\,\chi^2_b\right),
$$

where $\mathcal{L}_{-b}$ is the photometric likelihood evaluated over all bands except $b$ and $\chi^2_b$ is the single-band residual chi-square of band $b$ (computed using the scaled fluxes `flux * old_offsets` that entered the fit). This Gaussian ratio holds both with and without `logl_dim_prior`, whose dimensional penalty is constant across all draws of a given star, and depends only on the photometry rather than the parallax. Passbands marked as excluded from the fit via `mask_fit` require no leave-one-out reweighting. Passing `old_offsets` ensures that the leave-one-out weights are evaluated at the exact flux scale used during the fit while returning the total multiplicative offset relative to the raw, uncorrected fluxes.

Uncertainties on the derived offsets are estimated via bootstrap resampling over calibration stars and posterior draws. When `prior_mean` and `prior_std` are supplied together, the bootstrap estimate in each band is combined with an independent Gaussian prior; any band with zero valid calibration stars (`n_used == 0`) falls back to `prior_mean` $\pm$ `prior_std` (or $1.0 \pm \infty$ when no prior is provided).

Additional calibration settings are configured via {class}`~brutus.analysis.offsets.PhotometricOffsetsConfig`:

| Option | Default | Description |
|---|---|---|
| `min_bands_used` | `4` | Minimum number of other valid bands required for a star when band $b$ was included in the fit |
| `min_bands_unused` | `3` | Minimum number of valid bands required for a star when band $b$ was held out of the fit |
| `n_bootstrap` | `300` | Number of bootstrap resampling iterations |
| `uncertainty_method` | `"bootstrap_iqr"` | Uncertainty estimator across bootstrap medians (`"bootstrap_iqr"` for the normalized interquartile range or `"bootstrap_std"` for the standard deviation) |

Because updating the photometric offsets shifts the stellar posteriors and vice versa, iterate the calibration by refitting the sample with the newly derived offsets (passing the previous iteration's offsets as `old_offsets`) until the per-band offsets converge. Select calibration stars that lie well within the physical domain of the model grid (such as low-extinction main-sequence stars with high-precision parallaxes and clean goodness-of-fit p-values; see {doc}`understanding_results`), and note that offsets calibrated on solar-neighborhood main-sequence stars may not extrapolate perfectly to very metal-poor stars or evolved giants.
