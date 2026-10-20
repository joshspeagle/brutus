# How brutus works

brutus infers the distance, extinction and properties of a star from its photometry and, if available, its parallax {cite:p}`speagle2025`. It does not run a sampler. It evaluates every model in a grid of stellar models, integrates the priors for each one, and builds the posterior as a weighted sum over models.

```{raw} html
:file: figures/pipeline.svg
:class: figure-svg
```

:::{div} figure-caption
The four steps of a fit, described in [The fit in four steps](#the-fit-in-four-steps).
:::

## The forward model

A model star has intrinsic parameters $\theta$, which set its spectral energy distribution ({term}`SED`), and extrinsic parameters $\phi$, which set how it appears to us:

| | Parameter | Meaning |
|---|---|---|
| $\theta$ | $M_{\rm init}$ | {term}`initial mass`, in $M_\odot$ |
| | EEP | {term}`equivalent evolutionary point <EEP>`: the position along the evolutionary track |
| | [Fe/H] | {term}`metallicity` |
| $\phi$ | $d$ | distance |
| | $A_V$ | {term}`extinction` in the V band, in mag |
| | $R_V$ | shape of the extinction curve |

Age is not a parameter: it follows from $(M_{\rm init}, {\rm EEP}, {\rm [Fe/H]})$. {doc}`stellar_models` explains the tracks and why EEP is used.

Each grid model stores its magnitudes $M_b(\theta)$ in every band $b$ at a distance of 1 kpc, and two {term}`reddening vectors <reddening vector>` $R_b(\theta)$ and $R'_b(\theta)$. The predicted flux in band $b$ is

$$
F_b(\theta, \phi) = s \; 10^{-0.4\,\left[ M_b(\theta) + A_V \left( R_b(\theta) + R_V\, R'_b(\theta) \right) \right]},
\qquad s = \left( \frac{1\,{\rm kpc}}{d} \right)^2 ,
$$

where $s$ is the {term}`scale factor`. Distance dims all bands by the same factor; dust dims blue bands more than red ones. The extinction curve follows {cite:t}`fitzpatrick2009`, with $R_V$ as its shape parameter.

```{raw} html
:file: figures/forward-model.svg
:class: figure-svg
```

## The likelihood

brutus compares fluxes, not magnitudes, because flux errors are close to Gaussian and magnitude errors are not. For the observed fluxes $\hat F_b$ with errors $\sigma_b$, over the $N_b$ valid bands,

$$
\chi^2(\theta, \phi) = \sum_{b=1}^{N_b} \frac{\left( \hat F_b - F_b(\theta, \phi) \right)^2}{\sigma_b^2},
\qquad
\ln \mathcal{L}_{\rm phot} = -\tfrac{1}{2}\,\chi^2 + \text{normalization}.
$$

The normalization holds the Gaussian terms in $\sigma_b$ and, by default, a term in the number of bands (`logl_dim_prior`), so that stars with different numbers of bands are weighted consistently.

A parallax $\hat\varpi \pm \sigma_\varpi$, in mas, adds a Gaussian term in $\varpi = 1/d$ with $d$ in kpc:

$$
-2 \ln \mathcal{L}_\varpi(d) = \frac{\left( \hat\varpi - 1/d \right)^2}{\sigma_\varpi^2} + {\rm const}.
$$

A fit needs at least 4 bands: each model has 3 free parameters ($s$, $A_V$, $R_V$).

## The priors

The model prior depends only on the grid model; the position-dependent priors depend on where the star is; the $R_V$ prior is the same for all stars.

$$
\pi(\theta, \phi) = \underbrace{\pi_{\rm IMF}(M_{\rm init}) \; \frac{dt}{d\,{\rm EEP}} \; \Delta V(\theta)}_{\text{model prior}} \;\times\; \underbrace{\pi_{\rm gal}(d, \theta \mid \ell, b) \; \pi_{\rm dust}(A_V \mid d, \ell, b)}_{\text{position-dependent priors}} \;\times\; \pi(R_V)
$$

- **Initial mass function**: the {cite:t}`kroupa2001` {term}`IMF`.
- **Age weight** $dt/d\,{\rm EEP}$: the time a star spends per unit of EEP. It turns a prior uniform in EEP into one uniform in age.
- **Grid volume** $\Delta V$: the spacing of the grid in each parameter, so that a sum over models approximates an integral. The default grid steps EEP by 6 on the main sequence and by 2 after it.
- **Galactic structure** $\pi_{\rm gal}$: the density of a thin disk, a thick disk and a halo along the sightline $(\ell, b)$, each with its own metallicity distribution and an age distribution set by that metallicity.
- **Dust map** $\pi_{\rm dust}$: the {term}`Bayestar` 3D dust map {cite:p}`green2019`, if given, with 0.2 mag of scatter added to the map's own spread. Where the map has no reliable value, this prior is uniform in $A_V$.
- **Extinction curve** $\pi(R_V)$: Gaussian with mean 3.32 and standard deviation 0.18 {cite:p}`schlafly2016`. It enters the fit of each model in step 1, and so the Gaussian that step 3 draws from.

{doc}`priors` gives each prior's equation, defaults and function.

## The fit in four steps

**1. Fit every model.** For each grid model, brutus finds the $(s, A_V, R_V)$ that maximize $\mathcal{L}_{\rm phot}\,\pi(R_V)$ by Gauss-Newton iteration: first in magnitudes for every model, then in fluxes for the models whose magnitude-space fit is close to the best. It also computes their 3×3 Fisher matrix, which describes a Gaussian around the best fit.

**2. Keep the good fits.** Each model gets the weight $\mathcal{L}_{\rm phot}\,\pi_{\rm model}$. If the parallax has a signal-to-noise ratio above 4, an approximate parallax term on $s$ is added, using the uncertainty of $s$ after marginalizing over $A_V$ and $R_V$. brutus keeps the models whose weight is within a factor $10^{-3}$ of the best (`wt_thresh`). If more than 50,000 remain (`max_models`), it keeps about 50,000 at random: model $i$ is kept with probability $p_i = \min(1, c\,w_i)$, proportional to this weight $w_i$, and its posterior weight in step 4 is divided by $p_i$. This keeps the evidence and the posterior draws unbiased.

**3. Integrate the priors.** For each kept model $i$, brutus draws $N_{\rm mc} = 50$ points from its Gaussian in $(\ln d, A_V, R_V)$, in antithetic pairs $(z, -z)$. Each point $j$ is weighted by the priors that depend on position and by the exact parallax likelihood. The model's posterior weight is the average:

$$
w_i \;\propto\; \mathcal{L}_{{\rm phot},i}\, \pi_{{\rm model},i} \; \frac{1}{N_{\rm mc}} \sum_{j=1}^{N_{\rm mc}} \pi_{\rm gal}(d_{ij}, \theta_i)\; \pi_{\rm dust}(A_{V,ij} \mid d_{ij})\; \mathcal{L}_\varpi(d_{ij})\; d_{ij}.
$$

The factor $d_{ij}$ is the Jacobian of sampling in $\ln d$. Sampling in $s$ instead would bias distances by a factor $d^{-3}$.

The parallax is used three times. During step 1, models whose magnitude-space fit, with the parallax added at the best-fit $s$, is more than a factor $10^6$ worse than the best (`logl_initthresh`) keep that fit and are not refined. The approximate term in step 2 then decides which models are kept. Only the exact term here sets the weights.

**4. Draw the posterior.** brutus draws 250 models (`Ndraws`) in proportion to the posterior weight $w_i$ of step 3, divided by $p_i$ for the models kept by `max_models`, and for each one draws a point $(d, A_V, R_V)$ from its weighted Monte Carlo points. These are the saved posterior draws.

## Why the parallax matters

Dust reddens a star as well as dimming it, so the colours measure the dust, but only as well as they measure the star's temperature. A hotter star behind more dust has nearly the same colours as a cooler star behind less. On the main sequence the hotter star is also more luminous, so it lies farther away. Photometry alone therefore allows a nearer, less reddened star or a farther, more reddened one. A parallax limits the distance directly, and the posterior is where the two agree.

```{raw} html
:file: figures/degeneracy.svg
:class: figure-svg
```

Without a parallax, the Galactic and dust priors set where along this degeneracy the posterior lies.

## What brutus returns

For each star, the output file holds the posterior draws of distance, $A_V$ and $R_V$, the grid model of each draw (from which mass, age, temperature and the other properties follow), and fit diagnostics: the best $\chi^2$, the number of bands, and the log-{term}`evidence`. {doc}`understanding_results` describes each field.

## An example: a sightline towards Orion

The tutorials include 207 stars in one small field towards Orion ($\ell = 209.1°$, $b = -19.9°$), with Pan-STARRS and 2MASS photometry and Gaia parallaxes. Fitted with the default grid and priors, the stars that fit well trace the dust along the sightline: their extinction is low out to about 0.4 kpc and, with scatter, rises to 2 to 5 mag beyond 0.5 kpc, behind the Orion molecular cloud.

```{raw} html
:file: figures/orion-sightline.svg
:class: figure-svg
```

:::{div} figure-caption
Distance and extinction of the stars with p-value above $10^{-3}$ ({doc}`understanding_results`) and median distance below 2.5 kpc, without a dust-map prior. Each star contributes 25 posterior draws and its median.
:::

Fitting a dust profile to many such posteriors is how brutus maps dust in 3D; tutorial 7 shows it.

## Assumptions

- **Single stars.** The default grid has no unresolved binaries. A binary looks too bright, so its distance comes out too small.
- **Non-rotating, solar-scaled MIST models** {cite:p}`choi2016,dotter2016`. Fast rotators and chemically peculiar stars are not modelled.
- **One extinction curve family**, with $R_V$ free within its prior.
- **Systematic differences between models and data** remain at the level of a few hundredths of a magnitude per band. {doc}`photometric_offsets` describes the corrections and how to add their uncertainties to the errors.

For faint stars without a parallax, the result depends on the priors. Check how the result changes with them for your science case.
