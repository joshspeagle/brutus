# Priors

Photometry alone often allows several solutions: a nearer, cooler star or a farther, hotter star behind more dust ({doc}`scientific_background`). The priors decide between them when the data cannot. This page gives each prior's equation, its defaults and the argument of {meth}`~brutus.analysis.BruteForce.fit` that changes it.

| Prior | Depends on | Default | Changed by |
|---|---|---|---|
| [Model prior](#model-prior) | grid model | Kroupa IMF × age weight × grid volume | `lnprior`, `apply_agewt`, `apply_grad` |
| [Galactic structure](#galactic-structure) | distance, sightline, [Fe/H], age | thin disk + thick disk + halo | `lngalprior`, `data_coords` |
| [Dust map](#dust-map) | $A_V$, distance, sightline | none | `dustfile`, `lndustprior` |
| [Extinction curve](#extinction-curve) | $R_V$, $A_V$ | $R_V \sim \mathcal{N}(3.32, 0.18^2)$, flat $A_V$ | `rv_gauss`, `rvlim`, `av_gauss`, `avlim` |

A parallax is a likelihood, not a prior; {doc}`scientific_background` describes how it enters.

The priors matter most for faint stars without a parallax. For a bright star with a precise parallax, the likelihood dominates.

## Model prior

The model prior is one number per grid model:

$$
\ln \pi_{\rm model} = \ln \pi_{\rm IMF}(M_{\rm init}) + \ln \frac{dt}{d\,{\rm EEP}} + \sum_k \ln \Delta x_k .
$$

**Initial mass function.** {func}`~brutus.priors.logp_imf` is the {cite:t}`kroupa2001` broken power law, normalized from 0.08 to 100 M☉:

$$
\pi_{\rm IMF}(M) \propto \begin{cases} M^{-1.3} & 0.08 \le M/M_\odot \le 0.5 \\ M^{-2.3} & 0.5 < M/M_\odot \le 100 \end{cases}
$$

At 0.5 M☉ the density is $2^{2.3} \approx 4.9$ times that at 1 M☉. A grid with an `Mr` label and no `mini` label uses the Pan-STARRS r-band luminosity function instead ({func}`~brutus.priors.logp_ps1_luminosity_function`).

**Age weight.** The `agewt` column of the grid is $dt/d\,{\rm EEP}$, the time a star spends per unit of EEP. Together with the grid volume below, it makes the prior uniform in age. Main-sequence models therefore weigh more than models of short phases such as the subgiant branch. `apply_agewt=False` removes it.

**Grid volume.** $\Delta x_k$ is the local spacing of grid parameter $k$ (`np.gradient` of its unique values). It makes the sum over models approximate an integral over the parameters, so a region sampled twice as densely does not count twice. The default grid steps EEP by 6 to EEP 454 and by 2 after it. `apply_grad=False` removes it.

`lnprior` replaces the IMF term with an array of your own, one value per model. The age weight and grid volume are still added unless you turn them off.

## Galactic structure

{func}`~brutus.priors.logp_galactic_structure` is a three-component model of the Milky Way with the structure of {cite:t}`blandhawthorn2016`. Each component $c$ has a number density $n_c$, normalized to 1 at the Sun and weighted by its local fraction $f_c$. The distance prior along the sightline $(\ell, b)$ is

$$
\pi_{\rm gal}(d \mid \ell, b) \propto d^2 \sum_c f_c\, n_c\big(R(d), Z(d)\big),
$$

where $d^2$ is the volume of a shell and $(R, Z)$ are the Galactocentric radius and height. The Sun is at $R_\odot = 8.2$ kpc and $Z_\odot = 0.025$ kpc (`R_solar`, `Z_solar` in `fit()`, which apply to the default prior only).

The disks are exponential in radius, with the radius smoothed near the Galactic centre, $R_{\rm eff} = \sqrt{R^2 + R_s^2}$:

$$
\ln n_{\rm disk} = -\frac{R_{\rm eff} - R_{{\rm eff},\odot}}{h_R} + \ln \nu(R, |Z|) - \ln \nu(R, |Z_\odot|).
$$

The vertical profile $\nu$ solves the vertical Jeans equation, $\partial(\nu \sigma_z^2)/\partial z = -\nu K_z$, in one vertical potential for both disks. The vertical force has the form of {cite:t}`kuijken1989`, a disk of surface density $\Sigma_0$ and thickness $D$ plus a constant dark matter density $\rho_{\rm DM}$:

$$
K_z(R, z) = 2\pi G \left[ S(R)\, \Sigma_0 \frac{z}{\sqrt{z^2 + D^2}} + 2 \rho_{\rm DM} z \right],
\qquad
S(R) = e^{-(R - R_\odot)/2.7\,{\rm kpc}}.
$$

$S(R)$ keeps its edge values outside 4 to 9 kpc, the radii of the measurement.

$\Sigma_0 = 39.3\ M_\odot\,{\rm pc}^{-2}$ and $D = 234$ pc reproduce $K_z(1.1\ {\rm kpc}) / 2\pi G = 67\ M_\odot\,{\rm pc}^{-2}$ and its 2.7 kpc scale length {cite:p}`bovy2013`, and a midplane density of $0.097\ M_\odot\,{\rm pc}^{-3}$ with $\rho_{\rm DM} = 0.013\ M_\odot\,{\rm pc}^{-3}$ {cite:p}`mckee2015`.

The thin-disk dispersion $\sigma_z$ depends on the star's age $\tau$, [Fe/H] and angular momentum, and grows with height as $1 + \gamma_z |z|$ {cite:p}`sharma2021`. The profile is then

$$
\ln \nu(z) = -2 \ln(1 + \gamma_z |z|) - \frac{1}{\sigma_{z,0}^2} \int_0^{|z|} \frac{K_z(t)}{(1 + \gamma_z t)^2}\, dt,
$$

with $\gamma_z = 0.20\ {\rm kpc}^{-1}$. Younger stars have smaller $\sigma_{z,0}$ and lie closer to the midplane. Without `loga`, the thin-disk profile is the average over the thin-disk age distribution (the table below), by 32-point Gauss-Legendre quadrature over the ages where it has weight. The star's `feh` sets $\sigma_{z,0}$ with or without `loga`; without `feh`, the thin-disk mean [Fe/H] does. [Fe/H] outside $-1$ to $0.5$, the range of the fit, takes the edge value. The thick disk is isothermal, with $\sigma_z = 43.6$ km/s from {func}`~brutus.priors.thick_disk_velocity_moments`, so $\ln \nu = -\Phi_z / \sigma_z^2$. Both profiles are normalized at the Sun's height at every radius, so the age and [Fe/H] distributions are those at the Sun's position.

The scale heights follow from these dispersions instead of being set. At the Sun, the age-averaged thin disk falls by $e$ in 0.33 kpc between 0.3 and 1.5 kpc, and the thick disk in 0.99 kpc between 0.5 and 2 kpc. Star counts give 0.3 and 0.9 kpc {cite:p}`juric2008`. A single thin-disk age is close to Gaussian near the plane, so its height depends on how it is measured; its column height $\Sigma / 2\nu_0$ at [Fe/H] $= -0.2$ is 0.16 kpc at 1 Gyr and 0.60 kpc at 12 Gyr.

The thick disk is normalized by its surface density. `f_thick_surface` is the ratio of the thick- to the thin-disk surface density at the Sun, 0.18 by default, the high-[α/Fe] surface mass fraction of {cite:t}`mackereth2017`. The prior converts it to the ratio at the Sun's position with the column heights $h = \Sigma / 2\nu(Z_\odot)$ of the two profiles:

$$
f_{\rm thick} = f_\Sigma \, \frac{h_{\rm thin}}{h_{\rm thick}}.
$$

The thin-disk column height averages over the thin-disk age and [Fe/H] distributions. With the defaults it is 0.38 kpc, the thick disk's is 1.24 kpc, and $f_{\rm thick} = 0.055$. A change to a parameter that sets the heights, such as `thick_sigma_z` or the thin-disk mean [Fe/H], therefore keeps the surface ratio and moves the ratio at the Sun. {func}`~brutus.priors.galactic_prior_summary` returns these quantities for any parameters:

```python
from brutus.priors import galactic_prior_summary

galactic_prior_summary()["f_thick_midplane"]  # 0.055
```

The halo is a power law, flattened near the centre and rounder far from it:

$$
n_{\rm halo} = \left( \frac{R_{\rm eff}}{R_{{\rm eff},\odot}} \right)^{-\eta},
\qquad
R_{\rm eff} = \sqrt{R^2 + (Z/q)^2 + R_s^2},
\qquad
q(r) = q_\infty - (q_\infty - q_0)\, e^{1 - r'/r_q},
$$

with $r' = \sqrt{R^2 + Z^2 + r_q^2}$.

| Component | Shape | Local fraction $f_c$ | [Fe/H] | Mean age | Age spread |
|---|---|---|---|---|---|
| Thin disk | $h_R = 2.6$ kpc, $\sigma_z$ by age | 1 | $-0.2 \pm 0.3$ | 4.9 Gyr | 4.0 Gyr |
| Thick disk | $h_R = 2.0$ kpc, $\sigma_z = 43.6$ km/s | 0.055 (surface ratio 0.18) | $-0.7 \pm 0.4$ | 8.3 Gyr | 2.8 Gyr |
| Halo | $\eta = 4.2$, $q_0 = 0.2$, $q_\infty = 0.8$, $r_q = 6$ kpc | 0.005 | $-1.6 \pm 0.5$ | 12.4 Gyr | 1.0 Gyr |

All components use $R_s = 2$ kpc.

```{raw} html
:file: figures/galactic-prior.svg
:class: figure-svg
```

:::{div} figure-caption
The density and volume terms of the default Galactic prior, per unit $\ln d$, the variable that brutus samples. The metallicity and age terms are left out, so a given grid model's prior can differ. Left: each sightline scaled to a peak of 1. Right: the b = 30° total and its components, scaled by the total's peak. The growing volume pushes the peak out to 10 to 20 kpc. Off the plane, the disks add a second peak at 2 to 4 kpc: at b = 30° it is the higher one, and at b = 90° it is 0.47 times the halo peak. The likelihood usually confines the distance far more than this.
:::

**Metallicity.** When the labels include `feh`, each component has a Gaussian [Fe/H] distribution (the table). The components are weighted by their density at the star's position. A metal-poor star near the Sun is therefore more likely to belong to the halo.

**Age.** When the labels include `loga`, each component has a normal distribution in age, truncated to 0 to 13.8 Gyr ({func}`~brutus.priors.logp_age_from_feh`). Its mean follows from the component's mean [Fe/H], $\overline{\rm [Fe/H]}$:

$$
\bar t = \frac{13.8\ {\rm Gyr}}{1 + \exp\!\left[ \left( \overline{\rm [Fe/H]} + 0.5 \right) / 0.5 \right]},
\qquad
\sigma_t = \frac{13.8\ {\rm Gyr} - \bar t}{2}, \text{ limited to 1 to 4 Gyr}.
$$

The component weights for the age term include the star's [Fe/H], so the metallicity and the position together set which age distribution applies.

The default prior needs the star's coordinates: pass `data_coords`, an array of shape `(Nstars, 2)` with $(\ell, b)$ in degrees. `fit()` raises `ValueError` without them.

## Dust map

With `dustfile`, the prior on $A_V$ comes from a 3D dust map. `fetch_dustmaps()` downloads the {term}`Bayestar` map {cite:p}`green2019`. `dustfile` takes its path or a loaded {class}`~brutus.dust.Bayestar`. At distance $d$ on the sightline, the map gives a mean $\mu_{A_V}(d)$ and a spread $\sigma_{A_V}(d)$, and {func}`~brutus.priors.logp_extinction` evaluates

$$
\pi_{\rm dust}(A_V \mid d, \ell, b) = \mathcal{N}\!\left( A_V \;\middle|\; \mu_{A_V}(d),\; \sigma_{A_V}^2(d) + 0.2^2 \right),
$$

truncated and normalized over `avlim`. The added 0.2 mag (`scatter`) covers structure smaller than a map pixel and the map's systematic errors; the map's own spreads are often a few hundredths of a magnitude.

Where the map has no reliable value, the prior is uniform over `avlim`, normalized like the Gaussian. This applies outside the Pan-STARRS footprint (declination below −30°), outside each sightline's reliable distance range, and in pixels whose fit did not converge. `Bayestar(apply_reliability_mask=False)` uses the raw map values instead.

## Extinction curve

The per-model fit has Gaussian priors on $R_V$ and $A_V$:

| Argument | Default | Meaning |
|---|---|---|
| `rv_gauss` | `(3.32, 0.18)` | mean and standard deviation of $R_V$ {cite:p}`schlafly2016` |
| `rvlim` | `(1.0, 8.0)` | allowed range of $R_V$ |
| `av_gauss` | `None` | mean and standard deviation of $A_V$; `None` is flat |
| `avlim` | `(0.0, 20.0)` | allowed range of $A_V$ |

These priors act in the Gauss-Newton fit of each model, so they also shape the Gaussian that the Monte Carlo step draws from. `av_gauss` adds to the dust-map prior: an informative `av_gauss` together with `dustfile` counts the $A_V$ information twice.

## Changing the priors

`lngalprior` replaces the Galactic prior. It applies only when `data_coords` is given; without coordinates, `fit()` uses no Galactic prior. `fit()` calls it as `lngalprior(dist, coord, labels=labels)`, where `dist` is an array of distances in kpc, `coord` is the star's $(\ell, b)$, and `labels` holds the grid labels of each point (`None` if `apply_dlabels=False`). It returns the log-prior with the shape of `dist`.

A prior flat in distance, for example to see what the data alone say:

```python
import numpy as np

fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    save_file="flat.h5",
    data_coords=coords,
    lngalprior=lambda dist, coord, labels=None: np.zeros_like(dist),
)
```

A prior with a thinner thick disk, from a smaller vertical dispersion:

```python
from functools import partial

from brutus.priors import logp_galactic_structure

fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    save_file="thin.h5",
    data_coords=coords,
    lngalprior=partial(logp_galactic_structure, thick_sigma_z=34.0),
)
```

`lndustprior` replaces the dust prior. It applies only when `dustfile` is given. `fit()` calls it as `lndustprior(av, dustmap, coord, distance=dist)`.

To check how much a result depends on the priors, fit the same stars twice, with the default and the flat prior, and compare the posterior distances. Stars whose distances change by more than their uncertainty are prior-dominated.

For star clusters, where the age, metallicity and distance are shared, see {doc}`population_modeling`.
