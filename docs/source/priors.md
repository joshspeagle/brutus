# Priors

Photometry alone often allows several solutions: a nearer, cooler star or a farther, hotter star behind more dust ({doc}`scientific_background`). The priors decide between them when the data cannot. This page gives each prior's equation, its defaults and the argument of {meth}`~brutus.analysis.BruteForce.fit` that changes it.

| Prior | Depends on | Default | Changed by |
|---|---|---|---|
| [Model prior](#model-prior) | grid model | Kroupa IMF × age weight × grid volume | `lnprior`, `apply_agewt`, `apply_grad` |
| [Galactic structure](#galactic-structure) | distance, sightline, [Fe/H], age | thin disk + thick disk + halo + bar | `lngalprior`, `data_coords` |
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

**Age weight.** The `agewt` column of the grid is $dt/d\,{\rm EEP}$, the time a star spends per unit of EEP. Together with the grid volume below, it makes the prior uniform in age. Main-sequence models therefore weigh more than models of short phases such as the subgiant branch. `apply_agewt=False` removes it; a grid without `agewt`, such as the Bayestar grid, needs it.

**Grid volume.** $\Delta x_k$ is the local spacing of grid parameter $k$ (`np.gradient` of its unique values). The grid parameters are `mini`, `eep`, `feh`, `afe` and `smf`, and `Mr` on a grid without `mini`; predictions such as `loga` get no term. It makes the sum over models approximate an integral over the parameters, so a region sampled twice as densely does not count twice. The default grid steps EEP by 6 to EEP 454 and by 2 after it. `apply_grad=False` removes it.

`lnprior` replaces the IMF term with an array of your own, one value per model. The age weight and grid volume are still added unless you turn them off.

## Galactic structure

{func}`~brutus.priors.logp_galactic_structure` is a four-component model of the Milky Way: the thin disk, thick disk and halo, with the structure of {cite:t}`blandhawthorn2016`, and the bar. Each component $c$ has a number density $n_c$, relative to the thin disk at the Sun. The distance prior along the sightline $(\ell, b)$ is

$$
\pi_{\rm gal}(d \mid \ell, b) \propto d^2 \sum_c n_c\big(\mathbf{x}(d)\big),
$$

where $d^2$ is the volume of a shell and $\mathbf{x}$ the Galactocentric position: the radius $R$ and height $Z$ for the disks and the halo, and also the azimuth for the bar. The Sun is at $R_\odot = 8.275$ kpc {cite:p}`gravity2021` and $Z_\odot = 0.0208$ kpc {cite:p}`bennett2019` (`R_solar`, `Z_solar` in `fit()`, which apply to the default prior only).

Each disk has an exponential surface density $\Sigma(R)$, with the radius smoothed near the Galactic centre, $R_{\rm eff} = \sqrt{R^2 + R_s^2}$. A disk population with vertical profile $\nu(R, z)$, 1 at the midplane, has the density

$$
n_{\rm pop}(R, Z) = \Sigma(R)\, p({\rm labels} \mid R)\, \frac{\nu(R, |Z|)}{2 H(R)},
\qquad
H(R) = \int_0^{100\ {\rm kpc}} \nu(R, z)\, dz,
$$

where $p({\rm labels} \mid R)$ is the population's share of the surface density and $H$ its column. A disk that flares therefore spreads its surface density over a greater height, instead of gaining stars. The scale lengths are those of the surface density: 2.74 kpc for the thin disk, the mass-weighted low-[α/Fe] disk of {cite:t}`imig2025`, and 1.9 kpc for the thick disk {cite:p}`mackereth2017`. brutus tabulates $\ln H$ against $R / R_\odot$, at the steps of the potential tables below, and $\ln(\sigma_z^2 / \phi)$, where $\phi$ scales the potential, and interpolates it with bicubic Catmull-Rom splines.

The vertical profile $\nu$ solves the vertical Jeans equation, $\partial(\nu \sigma_z^2)/\partial z = -\nu K_z$, in one potential for both disks: that of the Milky Way model of {cite:t}`binney2024`, which fits a self-consistent potential and distribution functions to APOGEE DR17 and Gaia DR3. brutus tabulates its vertical potential $\Phi_z(R, z) = \Phi(R, z) - \Phi(R, 0)$ and force $K_z = \partial \Phi / \partial z$ against $R / R_\odot$ and $z$, from a rebuild with AGAMA of the paper's distribution functions (`.github/scripts/make_bv24_tables.py`). The rebuilt model has $K_z(R_\odot, 1.1\ {\rm kpc}) / 2\pi G = 63.9\ M_\odot\,{\rm pc}^{-2}$, against 66.4 in the paper, and a circular speed of 235.7 km/s at the Sun. brutus scales the potential by $(v_{c,\odot} / 235.7\ {\rm km\,s^{-1}})^2$, so that it matches the rotation curve at the Sun (239.2 km/s by default), as $R / R_\odot$ scales its radii.

The thin-disk dispersion $\sigma_z$ depends on the star's age $\tau$, [Fe/H] and angular momentum, and grows with height as $f_z = 1 + \gamma_z |z|$ {cite:p}`sharma2021`. The profile is then

$$
\ln \nu(z) = -2 \ln f_z(|z|) - \frac{1}{\sigma_{z,0}^2} \int_0^{|z|} \frac{K_z(t)}{f_z(t)^2}\, dt,
$$

with $\gamma_z = 0.20\ {\rm kpc}^{-1}$. Above 2 kpc, the greatest height of the fit, $f_z$ keeps its value (with a quadratic blend over 1.5 to 2.5 kpc), so each population is isothermal there; extrapolated, the linear growth would give the disk a tail falling only as $z^{-2}$. Younger stars have smaller $\sigma_{z,0}$ and lie closer to the midplane. Extrapolated to age 0, the relation gives 2.8 km/s at the Sun, while stars younger than about 200 Myr toward the anticentre have 8 km/s, like the gas {cite:p}`gaia2021anticentre`; $\sigma_{z,0}$ is therefore joined to a floor of 8 km/s as $(\sigma_{z,0}^4 + 8^4)^{1/4}$, which raises it by 0.5% at 10 Gyr and 7% at 2 Gyr. The angular-momentum factor of $\sigma_{z,0}$, at the population's mean angular momentum, depends on the population only through its radial dispersion; brutus tabulates it against $R / R_\odot$ and $\ln \sigma_{R,0}$ and interpolates it with bicubic Catmull-Rom splines. [Fe/H] outside $-1$ to $0.5$, the range of the fit, takes the edge value in $\sigma_{z,0}$. The $L_z^2$ term of the angular-momentum factor lets $\sigma_z$ rise outward to account for the flare {cite:p}`sharma2021`; Gaia data show no upturn of $\sigma_z$ out to 14 kpc {cite:p}`gaia2018kin`, so the factor of $\sigma_z$ keeps its value at the $L_z$ of a circular orbit at 14 kpc; populations that lag reach it a little further out. The thin-disk column height $\Sigma / 2\nu$ is then 0.35 kpc at the Sun, 0.87 kpc at 14 kpc and 1.68 kpc at 20 kpc. Measurements at 14–16 kpc range from about 0.5 to 2.2 kpc {cite:p}`yu2024,uppal2023`, and {cite:t}`lopezcorredoira2014` find about 1.9 kpc at 20 kpc. Thin-disk density fits reach $|z| = 5$ kpc {cite:p}`yu2021`; above that, the hottest outer-disk populations would spread out like a halo, so the thin-disk density is tapered to 0 between $|z| = 5$ and 7 kpc with the cosine taper of the bar. The taper removes those stars: 6.3% of the thin-disk column at 20 kpc and 14.8% at 25 kpc. With the thick disk, halo and bar switched off, the prior is therefore 0 above $|z| = 7$ kpc. The thick disk is isothermal at each radius, so $\ln \nu = -\Phi_z / \sigma_z^2$, with $\sigma_z = 43.6$ km/s at the Sun and the radial profile of the high-α disk of {cite:t}`binney2024`, as in {func}`~brutus.priors.thick_disk_velocity_moments`. That disk is truncated beyond about 12 kpc, and inside about 3 kpc its $\sigma_z$ rises to 119 km/s, which would put two more density peaks kiloparsecs above the Galactic centre. The profile is therefore held outside $0.36 R_\odot$ to $1.5 R_\odot$. With the dispersions held, the rising $K_z$ toward the bulge would still squeeze both disks threefold in height inside 3 kpc, so inside 4 kpc each disk keeps the density it has at 4 kpc, with a 1 kpc quadratic blend; the bar fills the inner Galaxy.

The scale heights follow from these dispersions instead of being set. At the Sun, the thin disk summed over its ages and [Fe/H] falls by $e$ in 0.31 kpc between 0.3 and 1.5 kpc, and the thick disk in 1.01 kpc between 0.5 and 2 kpc. Star counts give 0.3 and 0.9 kpc {cite:p}`juric2008`. A single thin-disk age is close to Gaussian near the plane, so its height depends on how it is measured; its column height $\Sigma / 2\nu(Z_\odot)$ at [Fe/H] $= -0.2$ is 0.19 kpc at 1 Gyr and 0.62 kpc at 12 Gyr.

The thick disk is normalized by its surface density. `f_thick_surface` is the ratio of the thick- to the thin-disk surface density at the Sun, 0.18 by default, the high-[α/Fe] surface mass fraction of {cite:t}`mackereth2017`. The prior converts it to the ratio at the Sun's position with the column heights $h = \Sigma / 2\nu(Z_\odot)$ of the two profiles:

$$
f_{\rm thick} = f_\Sigma \, \frac{h_{\rm thin}}{h_{\rm thick}}.
$$

The thin-disk column height sums over the thin-disk ages and [Fe/H]. With the defaults it is 0.35 kpc, the thick disk's is 1.27 kpc, and $f_{\rm thick} = 0.050$. A change to a parameter that sets the heights, such as `thick_sigma_z` or `v_circ_solar`, therefore keeps the surface ratio and moves the ratio at the Sun. {func}`~brutus.priors.galactic_prior_summary` returns these quantities for any parameters.

Setting `f_thick_surface`, `halo_mass` or `bar_mass` to 0 removes that component, and the prior then skips it. The thin disk sets the normalization, so it always stays.

The halo is the broken power law of {cite:t}`amarante2024` in a flattened radius $m$, with a core $R_s$ that keeps it finite at the Galactic centre:

$$
\ln n_{\rm halo} = -\eta_{\rm in} \ln m - (\eta_{\rm out} - \eta_{\rm in}) \max\!\left(\ln \frac{m}{r_b}, 0\right) + {\rm const},
\qquad
m = \sqrt{R^2 + (Z/q)^2 + R_s^2}.
$$

Amarante et al. step $q$ from 0.77 to 0.99 at a Galactocentric radius of 30 kpc. In a prior, that step raises the density along the poles by a factor of 3.1 and adds a false distance peak at high latitude, so brutus uses $q = 0.77$ everywhere. About 90% of the halo mass within 70 kpc lies inside 30 kpc.

The halo is normalized by its stellar mass, $M_{\rm halo} = 7.5 \times 10^8\ M_\odot$ between Galactocentric radii of 2 and 70 kpc, the middle of the $6.7$ to $8.4 \times 10^8\ M_\odot$ of {cite:t}`lane2023`. Its density at the Sun is $M_{\rm halo}$ divided by the integral of its shape over that range, $5.3 \times 10^{-5}\ M_\odot\,{\rm pc}^{-3}$ with the defaults. Its ratio to the thin disk divides that by the thin disk's local density in living stars, $\rho_{\rm thin,\odot} = 0.030\ M_\odot\,{\rm pc}^{-3}$ (`rho_thin_solar`; {cite:t}`everall2022` find 0.032 including brown dwarfs and stars below 0.1 $M_\odot$), so $f_{\rm halo} = 1.8 \times 10^{-3}$. The ratio is of masses; brutus uses it as a ratio of star counts.

The bar is the analytic density of {cite:t}`sormani2022`, a fit to the dynamical model of {cite:t}`portail2017`, which reproduces the red-clump star counts and kinematics of the inner Galaxy. It has an X-shaped bar of $1.28 \times 10^{10}\ M_\odot$ and a long bar of $0.55 \times 10^{10}\ M_\odot$ that reaches 5 kpc from the centre in the plane. Its long axis is at $25°$ to the line from the Sun to the centre, with the near end at $\ell > 0$ (`bar_angle`). Its long-bar terms decay slowly beyond the end of the bar, whose half-length is $5.0 \pm 0.2$ kpc {cite:p}`wegg2015`. brutus therefore multiplies the density by a cosine taper, $[1 + \cos(\pi (r - 5\ {\rm kpc}) / 1\ {\rm kpc})] / 2$ between 5 and 6 kpc from the centre, and leaves the bar out beyond 6 kpc. Along the long axis in the midplane, the published density is 0.31 of the disks and halo at 6 kpc and 0.026 at 8 kpc, and $7 \times 10^{-4}$ of the thin disk at the Sun. The bulge is normalized by the mass of the tapered density, $1.81 \times 10^{10}\ M_\odot$ (`bar_mass`; 0 removes the bar and the spheroid), divided by $\rho_{\rm thin,\odot}$ like the halo. The bar's shape takes 0.4 of it and the metal-poor spheroid below 0.6. Inside the box $|x| < 2.2$, $|y| < 1.4$, $|z| < 1.2$ kpc of the bar frame the bar holds $0.51 \times 10^{10}\ M_\odot$ and the spheroid $0.39 \times 10^{10}$, against $0.14 \times 10^{10}$ in the two disks. Along one sightline the bar density depends only on $d$, so brutus interpolates its logarithm in 4096 nodes from 1 pc to 25 kpc, uniform in $\ln d$; the error is at most $1.2 \times 10^{-3}$ in $\ln n_{\rm bar}$, and the taper is applied after the interpolation. {mod}`brutus.priors.kinematic` has no velocity model for the bar.

| Component | Shape | Density at the Sun, relative to the thin disk | [Fe/H] at the Sun | Age at the Sun |
|---|---|---|---|---|
| Thin disk | $h_R = 2.74$ kpc, $\sigma_z$ by age and [Fe/H] | 1 | $-0.01 \pm 0.21$ | $5.4 \pm 2.8$ Gyr |
| Thick disk | $h_R = 1.9$ kpc, $\sigma_z = 43.6$ km/s | 0.050 (surface ratio 0.18) | $-0.45 \pm 0.27$ | $10.5 \pm 1.8$ Gyr |
| Halo | $\eta_{\rm in} = 2.85$, $\eta_{\rm out} = 4.55$, $r_b = 19.1$ kpc, $q = 0.77$ | 0.0018 (mass $7.5 \times 10^8\ M_\odot$) | $-1.21 \pm 0.37$ | $11.1 \pm 1.8$ Gyr |
| Bar | {cite:t}`sormani2022`, at $25°$, and a metal-poor spheroid | 0 (mass $1.81 \times 10^{10}\ M_\odot$) | $-0.08 \pm 0.42$ | $9.4 \pm 2.7$ Gyr |

The label columns give means and standard deviations, with the systematics. The disks and the halo use $R_s = 2$ kpc.

```{raw} html
:file: figures/galactic-prior.svg
:class: figure-svg
```

:::{div} figure-caption
The density and volume terms of the default Galactic prior, per unit $\ln d$, the variable that brutus samples. The thin disk is summed over its ages and [Fe/H]; a grid model with given labels has its own prior. Left: each sightline at ℓ = 90° scaled to a peak of 1. Middle: the b = 30° total and its components, scaled by the total's peak. Right: the same toward the bulge, at ℓ = 0° and b = 4°, where the bar peaks at 8.3 kpc. At ℓ = 90° the growing volume pushes the peak out to 8 to 18 kpc. Off the plane, the disks add a second peak at 2 to 4 kpc: 0.25 times the main peak at b = 90° and 0.67 times at b = 30°. The likelihood usually confines the distance far more than this.
:::

**Labels.** With `feh` and `loga` in the labels, each component's density is that of stars with those labels, per dex and per Gyr; a label not given is summed over. A metal-poor star near the Sun is therefore more likely to belong to the halo, and an old one to the thick disk. Ages are capped at 13.8 Gyr, the age of the Universe.

The thin-disk ages and [Fe/H] follow the star formation, enrichment and radial migration model of {cite:t}`frankel2018`, fitted to APOGEE red-clump stars at Galactocentric radii of 5 to 14 kpc. Stars form at the rate $e^{-(\tau_m - \tau)/\tau_{\rm SFR}}$, at birth radii $R_0$ with $p(R_0 \mid \tau) = e^{-R_0 / R_{\rm exp}} / R_{\rm exp}$, $R_{\rm exp} = 3\ {\rm kpc}\,(1 - \alpha_R \tau / 8\ {\rm Gyr})$, and with the [Fe/H] of the gas there,

$$
{\rm [Fe/H]}_0 = F_m + \nabla R_0 - (F_m + \nabla R_{\rm now}) \left(1 - \frac{\tau}{\tau_m}\right)^{\gamma}.
$$

They then migrate from $R_0$ by a Gaussian of width $\sigma_{\rm RM} \sqrt{\tau / 8\ {\rm Gyr}}$, normalized over $R \ge 0$. brutus uses the posterior values: $\tau_m = 12$ Gyr, $\tau_{\rm SFR} = 5.9$ Gyr, $\alpha_R = 0.42$, $F_m = -1$, $\nabla = -0.075$ dex/kpc, $R_{\rm now} = 8.8$ kpc, $\gamma = 0.36$ and $\sigma_{\rm RM} = 3.6$ kpc (`THIN_DISK_*` in `brutus.priors.galactic`). It reads the model as the distribution per unit area, $p({\rm labels} \mid R)$ above. Inside 5 kpc the distribution keeps its value there. Beyond 14 kpc the age mix $p(\tau \mid R)$ keeps its value there, because the inside-out growth was fitted only to 14 kpc and would otherwise make the outer disk ever younger, while [Fe/H] at each age follows the model. Both holds blend over 2 kpc. Five additions complete it:

- The [Fe/H] at birth stops falling beyond $R_0 = 15.3$ kpc (`THIN_DISK_GRADIENT_KNEE`). Classical Cepheids flatten beyond 15.3 kpc at [Fe/H] $= -0.48$ {cite:p}`albarracin2026`, close to the $-0.49$ that the gradient above reaches there today; other Cepheid samples show no break out to 20 kpc {cite:p}`trentin2024`. Stars born beyond the knee have its [Fe/H], so young stars there scatter about $-0.49$, and the integral over $R_0$ stays analytic.
- A share of 2% of the thin disk has a birth [Fe/H] spread of 0.3 dex (`THIN_DISK_FEH_TAIL`). The narrow scatter alone gives Gaussian tails that the model was never fitted in: a 0.5 Gyr star of solar [Fe/H] at 20 kpc would have $6 \times 10^{-6}$ of the peak density there, and the tail keeps it at $2 \times 10^{-3}$, so such a star is not handed to the thick disk by the tails alone.
- The [Fe/H] at birth scatters by 0.03 dex, the open-cluster value that Frankel et al. cite. Migration supplies the spread of all but the youngest stars; this floor keeps them from a single [Fe/H].
- Frankel et al. fit stars younger than 8 Gyr; older stars enter their fit only through a separate component, and their star formation ends at $\tau_m = 12$ Gyr. brutus leaves out that component. Instead, star formation fades beyond 8 Gyr, the start of the low-α disk {cite:p}`xiang2022`, by a half-Gaussian 1 Gyr wide, and ends at 12 Gyr. The age density therefore has no step. At the Sun the thin disk has a mean age of 5.4 Gyr, and 21% of its stars are older than 8 Gyr.
- The integral over $R_0$ is analytic except for the migration kernel's normalization over $R \ge 0$, which brutus takes at the mean $R_0$. The error is largest for stars born near the Galactic centre. With the 0.03 dex birth scatter alone it is 2.5% at most where the density exceeds $10^{-3}$ of its peak at that age and radius; with the default [Fe/H] systematic below, the age density is off by up to 3.2% (at 4 Gyr and 6 kpc).

Without the systematics below, all components together give [Fe/H] $= -0.04 \pm 0.22$ at the Sun, against $-0.06 \pm 0.22$ (Geneva-Copenhagen Survey) and $+0.01 \pm 0.20$ {cite:p}`hayden2015`. With them it is $-0.03 \pm 0.24$. Young stars follow the gradient of the gas, $-0.075$ dex/kpc.

The thick-disk [Fe/H] is skew-normal, with mean $-0.45$, standard deviation 0.22 and skewness $-0.5$ (`thick_feh`): the high-[α/Fe] distributions of {cite:t}`hayden2015` with their metal-poor tail. Its age is normal with a standard deviation of 0.8 Gyr, truncated to 0 to 13.8 Gyr, about a mean of 13 Gyr at [Fe/H] $\le -1$ that falls by 4 Gyr per dex above (`thick_age`). That is the old sequence of {cite:t}`xiang2022`, from 13 Gyr at [Fe/H] $= -1$ to about 7 Gyr at $+0.5$, with an intrinsic spread below 0.82 Gyr. The halo [Fe/H] has a skew-normal core through the 16th, 50th and 84th percentiles of {cite:t}`naidu2020`, $-1.48$, $-1.15$ and $-0.91$. Below [Fe/H] $= -1.95$ it continues as $10^{k\,{\rm [Fe/H]}}$ with $k = 1$, the slope that {cite:t}`youakim2020` measure between $-3.4$ and $-2.5$ (`halo_feh`: the mean, standard deviation and skewness of the core, the break and $k$). Of the halo stars, 2.7% lie below $-2$ and 0.27% below $-3$. Its age is $12 \pm 1.5$ Gyr, truncated likewise (`halo_age`): isochrone and colour-magnitude-diagram ages give 11.5 to 12.3 Gyr and asteroseismology about 10 Gyr, and the width covers that spread.

The bar has two populations, each normal in [Fe/H] and in age (`bar_feh`, `bar_age`). In [Fe/H] they are the two components that {cite:t}`gonzalez2015` fit to bulge red-clump stars: $-0.31 \pm 0.31$ and $+0.26 \pm 0.20$. The two populations have their own shapes. The metal-rich one keeps the bar's density with 0.4 of `bar_mass`. The metal-poor one, 0.6 of it (`bar_feh[0]`), is a spherical cored power law, $\rho \propto (r^2 + 0.2^2\ {\rm kpc}^2)^{-1.1}$, tapered like the bar (`SPHEROID_SHAPE`). {cite:t}`zoccali2017` find the metal-poor population more centrally concentrated and more axisymmetric than the boxy metal-rich one, with a share of 0.74 at $b = -8.5°$, 0.33 at $-3.5°$ and 0.53 at $-1°$. The shape and share are fitted to the metal-poor fractions of their fields. Near the peak of the bulge density toward $\ell = 0$ the share is then 0.70 at $b = -8°$, 0.47 at $-3.5°$ and 0.48 at $-1°$, and over 23 fields the rms difference falls from 0.18 with a fixed share to 0.11. A star's bulge labels therefore depend on where it is. The metal-poor population is $11 \pm 1.5$ Gyr old and the metal-rich one $8 \pm 3$ Gyr, truncated to 0 to 13.8 Gyr: among microlensed bulge dwarfs, most stars below [Fe/H] $= -0.5$ are older than 10 Gyr, and over 35% of those above solar are younger than 8 Gyr {cite:p}`bensby2017`.

**Systematics.** These distributions are measured on the scales of their surveys and models: APOGEE [Fe/H], and the ages of Frankel et al. and Xiang & Rix. brutus labels a star on the scale of its model grid, so it blurs each distribution by the mismatch between the scales. It widens them but does not shift them, because the offset depends on the grid.

- `feh_systematic`, 0.1, 0.15 and 0.15 dex for the thin disk, thick disk and halo (the bar takes the thick-disk value), is the standard deviation of a Gaussian in [Fe/H]. Surveys that measure the same stars differ by about 0.1 dex {cite:p}`hegedus2023`. A solar-scaled grid, such as the default MIST grid, returns about [M/H] for an α-rich star: 0.14 to 0.21 dex above its [Fe/H] at [α/Fe] $= 0.2$ to $0.3$ {cite:p}`salaris1993`.
- `age_systematic`, 0.1, 0.15 and 0.15, is the standard deviation of a Gaussian in age as a fraction of the true age, with `age_floor` (0.2 Gyr, the floor that Frankel et al. use) added in quadrature. Isochrone grids differ by about 10% in age {cite:p}`tayar2022,pinsonneault2025`. Solar-scaled ages of α-rich stars run 1 to 2 Gyr older than α-enhanced ones {cite:p}`xiang2022`.

Each blurred distribution stays normalized. The thin-disk birth scatter becomes $\sqrt{0.03^2 + \sigma_{\rm [Fe/H]}^2}$. A skew-normal convolved with a Gaussian is again skew-normal, and the halo [Fe/H] with its tail is convolved numerically. The thick-disk and halo age widths add the systematic at their mean ages. A thin-disk star's labelled age is a normal about its true age, truncated to 0 to 13.8 Gyr, and its vertical profile is that of the true age; brutus sums over the true ages by Gauss-Legendre quadrature, with 16 nodes about the labelled age and the onset. With one label, brutus interpolates in cached tables of the thin disk against that label, $R / R_\odot$ and $z$, built once per parameter set. With both, the sum over true ages runs at each distance, but its terms that depend only on the labels are computed once per model.

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
