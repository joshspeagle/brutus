# Photometric offsets

The models and the data disagree by a few percent in each band, from the stellar models, the atmospheres and the photometric calibration. brutus corrects this with one multiplicative offset per band, applied to the data. This page describes the default offsets, how to add their uncertainties, and how to compute your own.

The models themselves also carry an empirical correction to the temperatures and radii of low-mass stars; {doc}`stellar_models` gives its form.

## The default offsets

`fetch_offsets()` downloads the offsets that go with the default grid {cite:p}`speagle2025`. They come from fits to open clusters and to nearby field stars with precise parallaxes (paper Section 5).

```{raw} html
:file: figures/offsets.svg
:class: figure-svg
```

:::{div} figure-caption
The offsets of the default grid, `offsets_mist_v9.txt`. A positive value makes the data brighter. Bands not in the file get an offset of 1.
:::

{func}`~brutus.data.load_offsets` reads them in the order of your bands, and `fit()` takes them as `phot_offsets`:

```python
from brutus.data import fetch_offsets, load_offsets

offsets = load_offsets(fetch_offsets(), filters=filters)
fitter.fit(..., phot_offsets=offsets)
```

`fit()` multiplies both the fluxes and their errors by the offsets, so the signal-to-noise ratio of each band does not change.

## Adding the offset uncertainties

The offsets have uncertainties of about 0.02 mag in the optical and 0.03 mag in the near-infrared (paper Table 5). brutus does not add them to the errors. To include them, convert each to flux and add it to the error in quadrature:

$$
\sigma_b' = \sqrt{\sigma_b^2 + \left( \frac{\ln 10}{2.5}\, \hat F_b\, \sigma_{{\rm sys},b} \right)^2},
$$

where $\sigma_{{\rm sys},b}$ is the uncertainty in magnitudes. {doc}`quickstart` shows the code for its 11 bands.

Without this term, the fits of bright stars, whose photometric errors are small, are limited by the offsets but do not know it: their $\chi^2$ is too large and their posteriors too narrow.

## Computing your own offsets

To calibrate a survey or a grid that has no offsets, fit a sample of well-measured stars and then compare the data with the models of the posterior draws. {func}`~brutus.analysis.photometric_offsets` does the comparison:

```python
import h5py
import numpy as np

from brutus.analysis import photometric_offsets
from brutus.utils import magnitude

# The bands the fit used: it drops bands with errors of 0.25 mag or more
# (merr_max) after applying the offsets.
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

`models` is the array from {func}`~brutus.data.load_models`. `flux` and `flux_err` are the inputs of the fit, without offsets.

For each band $b$, the function computes the ratio of model to observed flux, $r = F_b / \hat F_b$, for every star and draw, and takes the median over stars and draws. A band used in the fit pulls the draws toward its own data, which would bias the offset toward the value the fit assumed. Each draw is therefore reweighted to the posterior without band $b$:

$$
w \propto \frac{\mathcal{L}_{-b}}{\mathcal{L}},
$$

where $\mathcal{L}_{-b}$ is the likelihood of the other bands. `mask_fit` marks the bands that were used in the fit; the others need no reweighting. `old_offsets` gives the offsets that the fit applied, so the reweighting uses the same fluxes as the fit. The result is the full offset relative to the uncorrected data. The likelihoods follow the $\chi^2$-density convention of the fit's default; pass `dim_prior=False` if the fit used `logl_dim_prior=False`. They include the photometry only, not the parallax.

The uncertainty comes from a bootstrap over stars and draws. `prior_mean` and `prior_std` combine the result with a Gaussian prior per band; they must be given together. A band that no star constrains (`n_used == 0`) gets `prior_mean` ± `prior_std`, or 1 with an infinite uncertainty if there is no prior.

{class}`~brutus.analysis.offsets.PhotometricOffsetsConfig` sets the other options:

| Option | Default | Meaning |
|---|---|---|
| `min_bands_used` | 4 | bands besides $b$ that a star needs when $b$ was used in the fit |
| `min_bands_unused` | 3 | bands a star needs when $b$ was not used |
| `n_bootstrap` | 300 | bootstrap samples |
| `uncertainty_method` | `"bootstrap_iqr"` | the scaled interquartile range of the bootstrap medians, or `"bootstrap_std"` |

The offsets change the fits, and the fits change the offsets. To converge, refit with the new offsets and compute them again, passing the offsets of the last fit as `old_offsets`, until they stop changing.

Use stars that the models describe well: main-sequence stars with precise parallaxes, low extinction and good fits (see the fit-quality checks in {doc}`understanding_results`). The default offsets come from such a sample, so they may not suit stars far from it, such as metal-poor stars or giants.
