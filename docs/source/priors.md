# Astrophysical Priors

Because broadband photometry alone often permits both a nearby cool dwarf and a distant reddened giant ({doc}`scientific_background`), `brutus` incorporates informative Bayesian priors over stellar evolution, 3D Galactic structure, and interstellar dust to regularize poorly constrained fits:

| Prior Component | Conditioning Variables | Default Model | Configuration Arguments |
|---|---|---|---|
| [Model prior](#model-prior) | Grid model $\theta$ | Kroupa IMF $\times$ age weight $\times$ grid volume | `lnprior`, `apply_agewt`, `apply_grad` |
| [Galactic structure](#galactic-structure) | Distance $d$, sightline $(\ell, b)$, [Fe/H], age | Thin disk + thick disk + stellar halo | `lngalprior`, `data_coords` |
| [Dust map](#dust-map) | $A_V$, distance $d$, sightline $(\ell, b)$ | Uniform over `avlim` (or Bayestar19 when `dustfile` is provided) | `dustfile`, `lndustprior` |
| [Extinction curve](#extinction-curve) | $R_V$, $A_V$ | $R_V \sim \mathcal{N}(3.32, 0.18^2)$, uniform $A_V$ | `rv_gauss`, `rvlim`, `av_gauss`, `avlim` |

An astrometric parallax enters the inference as an observational likelihood term rather than a prior ({doc}`scientific_background`).

Astrophysical priors have the strongest influence on faint stars that lack precise parallax measurements; for bright stars with high signal-to-noise photometry and astrometry, the likelihood dominates the posterior.

## Model Prior

The intrinsic model prior assigns a single scalar log-weight to each node in the stellar grid:

$$
\ln \pi_{\rm model} = \ln \pi_{\rm IMF}(M_{\rm init}) + \ln \frac{dt}{d\,{\rm EEP}} + \sum_k \ln \Delta x_k .
$$

**Initial mass function.** {func}`~brutus.priors.logp_imf` evaluates the {cite:t}`kroupa2001` two-part broken power-law initial mass function, normalized over $0.08 \le M/M_\odot \le 100$:

$$
\pi_{\rm IMF}(M) \propto \begin{cases} M^{-1.3} & 0.08 \le M/M_\odot \le 0.5 \\ M^{-2.3} & 0.5 < M/M_\odot \le 100 \end{cases}
$$

Under this IMF, the prior density at $0.5\,M_\odot$ is $2^{2.3} \approx 4.9$ times higher than at $1.0\,M_\odot$. When fitting an empirical grid that tabulates absolute $r$-band magnitude `Mr` rather than initial mass `mini` (such as `grid_bayestar_v5`), `BruteForce` automatically substitutes the Pan-STARRS $r$-band luminosity function ({func}`~brutus.priors.logp_ps1_luminosity_function`).

**Evolutionary lifetime weight.** The `agewt` field of a MIST grid stores the evolutionary time derivative $dt/d\,{\rm EEP}$. Multiplying by $dt/d\,{\rm EEP}$ alongside the EEP grid spacing transforms a grid uniform in EEP into a prior uniform in stellar age, assigning greater prior weight to long-lived main-sequence phases than to rapid post-main-sequence transitions such as the Hertzsprung gap. Setting `apply_agewt=False` disables this factor, which is required when fitting empirical grids such as `bayestar_v5` that do not tabulate `agewt`.

**Grid volume element.** For each active grid axis $k$, $\Delta x_k$ is the local grid spacing computed via `np.gradient` over the unique coordinate values. Only the intrinsic grid axes (`mini`, `eep`, `feh`, `afe`, and `smf`, or `Mr` on a grid without `mini`) receive spacing factors; predicted stellar quantities such as `loga` or `logt` do not. Including $\prod_k \Delta x_k$ ensures that summing over discrete grid nodes approximates a continuous Riemann integral over the parameter space, preventing regions sampled at finer grid spacing (such as post-main-sequence phases above EEP 454, where the default grid steps EEP by 2 rather than 6) from receiving artificial extra prior mass. Setting `apply_grad=False` disables the grid-volume correction.

Passing a custom array to `lnprior` (with one log-prior entry per grid model) replaces the default IMF term while retaining the `agewt` and grid-volume factors unless `apply_agewt=False` or `apply_grad=False` is also specified.

## Galactic Structure

{func}`~brutus.priors.logp_galactic_structure` implements a three-component spatial, chemical, and chronological model of the Milky Way following {cite:t}`blandhawthorn2016`. Each structural component $c \in \{\text{thin disk}, \text{thick disk}, \text{halo}\}$ has a 3D stellar number density $n_c(R, Z)$ normalized to unity at the solar neighborhood and weighted by its local solar-neighborhood density ratio $f_c$. Along sightline $(\ell, b)$, the spatial prior per unit distance $d$ is

$$
\pi_{\rm gal}(d \mid \ell, b) \propto d^2 \sum_c f_c\, n_c\big(R(d), Z(d)\big),
$$

where $d^2$ is the spherical volume Jacobian of a conical sightline element and $(R, Z)$ are the Galactocentric cylindrical radius and vertical height above the midplane. By default, the Sun is placed at $R_\odot = 8.2\,{\rm kpc}$ and $Z_\odot = 0.025\,{\rm kpc}$ (configurable via `R_solar` and `Z_solar` in `fit()`, which forward to the default Galactic prior).

The thin and thick disks follow double-exponential density profiles with a core smoothing radius $R_s$ near the Galactic center, $R_{\rm eff} = \sqrt{R^2 + R_s^2}$:

$$
\ln n_{\rm disk} = -\frac{R_{\rm eff} - R_{{\rm eff},\odot}}{h_R} - \frac{|Z| - |Z_\odot|}{h_Z}.
$$

The stellar halo follows an oblate power-law profile whose axis ratio $q(r)$ transitions smoothly from a flattened inner halo ($q_0$) to a rounder outer halo ($q_\infty$):

$$
n_{\rm halo} = \left( \frac{R_{\rm eff}}{R_{{\rm eff},\odot}} \right)^{-\eta},
\qquad
R_{\rm eff} = \sqrt{R^2 + (Z/q)^2 + R_s^2},
\qquad
q(r) = q_\infty - (q_\infty - q_0)\, e^{1 - r'/r_q},
$$

where $r' = \sqrt{R^2 + Z^2 + r_q^2}$. All three components share the core smoothing scale $R_s = 2\,{\rm kpc}$:

| Component | Spatial Scale Parameters | Local Fraction $f_c$ | [Fe/H] Distribution | Mean Age $\bar t$ | Age Spread $\sigma_t$ |
|---|---|---|---|---|---|
| Thin disk | $h_R = 2.6\,{\rm kpc}$, $h_Z = 0.3\,{\rm kpc}$ | $1.000$ | $-0.2 \pm 0.3\,{\rm dex}$ | $4.9\,{\rm Gyr}$ | $4.0\,{\rm Gyr}$ |
| Thick disk | $h_R = 2.0\,{\rm kpc}$, $h_Z = 0.9\,{\rm kpc}$ | $0.040$ | $-0.7 \pm 0.4\,{\rm dex}$ | $8.3\,{\rm Gyr}$ | $2.8\,{\rm Gyr}$ |
| Stellar halo | $\eta = 4.2$, $q_0 = 0.2$, $q_\infty = 0.8$, $r_q = 6\,{\rm kpc}$ | $0.005$ | $-1.6 \pm 0.5\,{\rm dex}$ | $12.4\,{\rm Gyr}$ | $1.0\,{\rm Gyr}$ |

```{raw} html
:file: figures/galactic-prior.svg
:class: figure-svg
```

:::{div} figure-caption
These curves show the spatial number-density and volume prior $d^3 \sum_c f_c n_c(R, Z)$ per unit $\ln d$ (the coordinate sampled during Monte Carlo integration, omitting the metallicity and age factors) at Galactic longitude $\ell = 90^\circ$. The left panel compares the total spatial prior across Galactic latitudes $b = 0^\circ$, $30^\circ$, and $90^\circ$ normalized to each sightline's peak; the right panel decomposes the $b = 30^\circ$ sightline into its thin-disk, thick-disk, and halo contributions. While the $d^3$ volume factor places the spatial prior peak between $10$ and $20\,{\rm kpc}$, the photometric and parallax likelihoods typically confine the posterior to much narrower distance intervals.
:::

**Metallicity distribution.** When the grid labels include `feh`, each Galactic component contributes a Gaussian metallicity prior $\mathcal{N}({\rm [Fe/H]} \mid \mu_{{\rm [Fe/H]},c}, \sigma_{{\rm [Fe/H]},c}^2)$ weighted by that component's spatial density at distance $d$. Consequently, a metal-poor star near the solar neighborhood receives higher relative membership weight in the halo than a solar-metallicity star at the same position.

**Age distribution.** When the grid labels include `loga`, each component contributes a truncated normal age prior over $[0, 13.8]\,{\rm Gyr}$ ({func}`~brutus.priors.logp_age_from_feh`) whose mean $\bar t$ and standard deviation $\sigma_t$ are determined by the component's mean metallicity $\overline{\rm [Fe/H]}$:

$$
\bar t = \frac{13.8\ {\rm Gyr}}{1 + \exp\!\left[ \left( \overline{\rm [Fe/H]} + 0.5 \right) / 0.5 \right]},
\qquad
\sigma_t = \frac{13.8\ {\rm Gyr} - \bar t}{2}, \quad \sigma_t \in [1, 4]\,{\rm Gyr}.
$$

Because the component membership weights for the age term multiply both the spatial density $n_c(R, Z)$ and the metallicity likelihood $p_c({\rm [Fe/H]})$, a star's 3D position and metallicity jointly determine which age distribution governs its prior.

Evaluating the default Galactic structure prior requires passing Galactic coordinates `data_coords` as an array of shape `(Nstars, 2)` in degrees (`fit()` raises `ValueError` if `data_coords` is omitted under the default prior).

## Dust Map

Passing `dustfile` to `BruteForce.fit` activates a distance-dependent 3D extinction prior on $A_V$. Calling `fetch_dustmaps()` downloads the {term}`Bayestar` 3D dust map {cite:p}`green2019`, which can be passed to `dustfile` either as a file path or as a pre-loaded {class}`~brutus.dust.Bayestar` instance. At distance $d$ along sightline $(\ell, b)$, the map provides a mean extinction $\mu_{A_V}(d)$ and statistical uncertainty $\sigma_{A_V}(d)$, and {func}`~brutus.priors.logp_extinction` evaluates a Gaussian density truncated and normalized over `avlim`:

$$
\pi_{\rm dust}(A_V \mid d, \ell, b) = \mathcal{N}_{\rm trunc}\!\left( A_V \;\middle|\; \mu_{A_V}(d),\; \sigma_{A_V}^2(d) + 0.2^2,\; {\rm avlim} \right).
$$

Adding `scatter = 0.2` mag in quadrature accounts for sub-pixel interstellar medium substructure and systematic calibration uncertainties in the dust map, preventing the map's small formal uncertainties (often $\sim 0.02\,{\rm mag}$) from over-constraining the distance posterior.

Where the dust map has no reliable constraint (south of declination $\delta = -30^\circ$ outside the Pan-STARRS footprint, at distances beyond a sightline's reliable distance bounds, or in pixels where the Bayestar fit did not converge), `logp_extinction` returns a properly normalized uniform density $1 / ({\rm avlim}[1] - {\rm avlim}[0])$. Instantiating `Bayestar(apply_reliability_mask=False)` disables distance-range and convergence masking so that raw map values are used wherever sky coverage exists.

## Extinction Curve

During the Step 1 Gauss-Newton optimization of each grid model, `brutus` applies truncated Gaussian priors on $R_V$ and $A_V$ that also shape the local Gaussian proposal sampled during Step 3:

| Argument | Default | Description |
|---|---|---|
| `rv_gauss` | `(3.32, 0.18)` | Mean and standard deviation of the Gaussian $R_V$ prior {cite:p}`schlafly2016` |
| `rvlim` | `(1.0, 8.0)` | Allowed support bounds for $R_V$ |
| `av_gauss` | `None` | Mean and standard deviation of a Gaussian $A_V$ prior (`None` applies a flat prior over `avlim`) |
| `avlim` | `(0.0, 20.0)` | Allowed support bounds for $A_V$ in magnitudes |

Because `av_gauss` multiplies the 3D dust-map prior when both are active, avoid passing an informative `av_gauss` simultaneously with `dustfile` so that the extinction constraint is not double-counted.

## Customizing the Priors

To replace the default Galactic structure prior, pass a callable `lngalprior(dist, coord, labels=labels)` that accepts an array of distances `dist` in kpc, the object's sky coordinates `coord`, and the structured grid `labels` (`None` when `apply_dlabels=False`) and returns an array of log-prior values matching the shape of `dist` (`lngalprior` is evaluated only when `data_coords` is provided).

For example, to fit stars under a flat distance prior so that distances are constrained solely by the photometry and parallax:

```python
import numpy as np

fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    save_file="flat.h5",
    data_coords=coords,
    lngalprior=lambda dist, coord, labels=None: np.zeros_like(dist),
)
```

To customize parameters of the built-in Galactic model (such as reducing the thick-disk vertical scale height to $0.7\,{\rm kpc}$), bind keyword arguments with `functools.partial`:

```python
from functools import partial

from brutus.priors import logp_galactic_structure

fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    save_file="thin.h5",
    data_coords=coords,
    lngalprior=partial(logp_galactic_structure, Z_thick=0.7),
)
```

Similarly, passing a custom callable `lndustprior(av, dustmap, coord, distance=dist)` overrides the dust-map prior whenever `dustfile` is supplied.

To assess how strongly your scientific conclusions depend on the Galactic and dust priors, fit the sample once with the default priors and once with a flat distance prior; stars whose posterior median distances shift by more than their $1\sigma$ credible intervals are prior-dominated. For coeval stellar populations that share a common distance, age, and metallicity, see {doc}`population_modeling`.
