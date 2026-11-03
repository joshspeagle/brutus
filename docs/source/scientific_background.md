# Scientific Background

`brutus` infers the distance, dust extinction, and intrinsic physical properties of a star from its broadband photometry and optional astrometric parallax {cite:p}`speagle2025`. Rather than exploring the multimodal stellar parameter space with iterative Markov chain Monte Carlo (MCMC) chains, `brutus` evaluates every template in a pre-computed stellar grid, integrates the continuous priors over distance and reddening for each viable model via importance sampling, and assembles the joint posterior as a weighted mixture across the grid.

```{raw} html
:file: figures/pipeline.svg
:class: figure-svg
```

:::{div} figure-caption
The four stages of a `BruteForce` fit proceed from optimizing distance scale and dust extinction across every model in the stellar grid (Stage 1) to pruning low-weight templates (Stage 2), integrating Galactic structure, dust-map, and parallax priors via antithetic Monte Carlo sampling in $(\ln d, A_V, R_V)$ (Stage 3), and resampling equally weighted draws from the combined posterior mixture (Stage 4).
:::

## The Forward Model

A model star is parameterized by intrinsic stellar labels $\theta$ that determine its unreddened spectral energy distribution ({term}`SED`), together with extrinsic parameters $\phi$ that govern how its flux is attenuated by interstellar dust and diluted by distance along the line of sight:

| Group | Parameter | Physical Meaning |
|---|---|---|
| $\theta$ | $M_{\rm init}$ | {term}`Initial mass <initial mass>` at birth, in $M_\odot$ |
| | EEP | {term}`Equivalent evolutionary point <EEP>` along the stellar evolutionary track |
| | [Fe/H] | Initial {term}`metallicity` in dex relative to solar |
| $\phi$ | $d$ | Heliocentric distance in kpc |
| | $A_V$ | Visual {term}`extinction` in the $V$ band, in magnitudes |
| | $R_V$ | Total-to-selective extinction ratio characterizing the shape of the dust curve |

Stellar age is determined uniquely by the triplet $(M_{\rm init}, {\rm EEP}, {\rm [Fe/H]})$ rather than varied as an independent grid axis ({doc}`stellar_models`).

Each grid model stores its unreddened magnitudes $M_b(\theta)$ in every photometric passband $b$ at a reference distance of $1\,{\rm kpc}$, alongside two linear {term}`reddening vectors <reddening vector>` $R_b(\theta)$ and $R'_b(\theta)$. The predicted linear flux density in band $b$ is

$$
F_b(\theta, \phi) = s \; 10^{-0.4\,\left[ M_b(\theta) + A_V \left( R_b(\theta) + R_V\, R'_b(\theta) \right) \right]},
\qquad s = \left( \frac{1\,{\rm kpc}}{d} \right)^2 ,
$$

where $s$ is the inverse-square {term}`scale factor`. Increasing the distance $d$ dims all passbands uniformly through $s$, whereas interstellar dust ($A_V > 0$) attenuates shorter wavelengths more strongly than longer wavelengths following the {cite:t}`fitzpatrick2009` extinction curve parameterized by $R_V$.

```{raw} html
:file: figures/forward-model.svg
:class: figure-svg
```

## The Photometric and Astrometric Likelihood

Because photon-counting and background-subtraction uncertainties are approximately Gaussian in linear flux density whereas magnitude uncertainties become skewed at low signal-to-noise ratios, `brutus` evaluates the photometric likelihood directly in flux space. Given observed fluxes $\hat F_b$ with measurement uncertainties $\sigma_b$ across $N_b$ valid bands, the photometric log-likelihood is

$$
\chi^2(\theta, \phi) = \sum_{b=1}^{N_b} \frac{\left( \hat F_b - F_b(\theta, \phi) \right)^2}{\sigma_b^2},
\qquad
\ln \mathcal{L}_{\rm phot} = -\tfrac{1}{2}\,\chi^2 + \text{normalization}.
$$

The normalization constant includes the Gaussian prefactors $-\frac{1}{2}\sum_b \ln(2\pi\sigma_b^2)$ and, by default (`logl_dim_prior=True`), a dimensional degrees-of-freedom penalty so that objects observed in different numbers of bands are weighted consistently.

When an astrometric parallax measurement $\hat\varpi \pm \sigma_\varpi$ (in mas) is available, it contributes an independent Gaussian likelihood term in $\varpi = 1/d$ (with $d$ in kpc):

$$
-2 \ln \mathcal{L}_\varpi(d) = \frac{\left( \hat\varpi - 1/d \right)^2}{\sigma_\varpi^2} + {\rm const}.
$$

Constrained optimization requires at least 4 valid photometric bands because each grid model fits 3 continuous extrinsic parameters ($s$, $A_V$, and $R_V$).

## Astrophysical Priors

The joint prior $\pi(\theta, \phi)$ factors into an intrinsic model prior that depends only on the grid node $\theta$, a position-dependent prior over distance and extinction along sightline $(\ell, b)$, and an extinction-curve prior on $R_V$ ({doc}`priors`):

$$
\pi(\theta, \phi) = \underbrace{\pi_{\rm IMF}(M_{\rm init}) \; \frac{dt}{d\,{\rm EEP}} \; \Delta V(\theta)}_{\text{model prior}} \;\times\; \underbrace{\pi_{\rm gal}(d, \theta \mid \ell, b) \; \pi_{\rm dust}(A_V \mid d, \ell, b)}_{\text{position-dependent priors}} \;\times\; \pi(R_V)
$$

- **Initial mass function ($\pi_{\rm IMF}$):** Weights initial stellar masses according to the {cite:t}`kroupa2001` broken power-law {term}`IMF`.
- **Evolutionary lifetime weight ($dt/d\,{\rm EEP}$):** Weights each model by the physical time a star spends per unit of EEP, transforming a grid uniform in EEP into a prior uniform in stellar age.
- **Grid volume element ($\Delta V$):** Multiplies by the local coordinate spacing of the grid axes so that summing over discrete grid nodes approximates a continuous integral over stellar parameters (the default grid steps EEP by 6 on the main sequence and by 2 in post-main-sequence phases).
- **Galactic structure ($\pi_{\rm gal}$):** Models the 3D stellar number density, metallicity distribution, and age distribution along sightline $(\ell, b)$ as a superposition of a thin disk, a thick disk, and a stellar halo.
- **3D dust map ($\pi_{\rm dust}$):** Evaluates a truncated Gaussian prior on $A_V$ at distance $d$ from the {term}`Bayestar` 3D dust map {cite:p}`green2019` (when provided), adding a $0.2\,{\rm mag}$ systematic scatter floor in quadrature to the map's statistical uncertainty and falling back to a normalized uniform prior over `avlim` where the map is masked.
- **Extinction curve shape ($\pi(R_V)$):** Applies a truncated Gaussian prior with mean $3.32$ and standard deviation $0.18$ {cite:p}`schlafly2016` during the per-model Gauss-Newton optimization in Step 1, which shapes the local Gaussian proposal distribution sampled in Step 3.

## The Inference Pipeline in Four Steps

**1. Optimize extrinsic parameters across the grid.** For every grid model, `brutus` solves for the extrinsic parameters $(s, A_V, R_V)$ that maximize $\mathcal{L}_{\rm phot}\,\pi(R_V)$ via two-stage Gauss-Newton iteration. It first solves the linear least-squares problem in magnitudes across all grid models, and then refines the fit in linear flux space for models whose magnitude-space likelihood (including the parallax likelihood evaluated at the magnitude-space best-fit scale $s$) lies within `logl_initthresh` ($10^{-6}$ by default) of the global maximum, while models below that threshold retain their magnitude-space fit without refinement. At convergence, `brutus` records the best-fit parameters and their $3 \times 3$ Fisher information matrix, which defines a local Gaussian proposal distribution around the maximum-likelihood point.

**2. Prune low-weight models.** Each model receives an initial selection weight proportional to $\mathcal{L}_{\rm phot}\,\pi_{\rm model}$. When a parallax measurement is provided, this weight is multiplied by the parallax likelihood marginalized over the model's local Gaussian distribution in $\ln d$, $\int \mathcal{N}(\ln d)\,\mathcal{N}(\varpi \mid 1/d,\,\sigma_\varpi)\,d\ln d$ (where the variance in $\ln d$ accounts for marginalization over $A_V$ and $R_V$), evaluated via five-node adaptive Gauss-Hermite quadrature centered on the integrand peak. `brutus` retains models whose selection weight lies within `wt_thresh` ($10^{-3}$ by default) of the maximum. If more than `max_models` (50,000 by default) survive, `brutus` subsamples approximately 50,000 models using systematic random sampling with inclusion probability $p_i = \min(1, c\,w_i)$ proportional to weight $w_i$, and divides the surviving models' final posterior weights in Step 4 by $p_i$ so that the marginal likelihood and posterior draws remain statistically unbiased.

**3. Integrate position-dependent priors via importance sampling.** For each retained model $i$, `brutus` draws $N_{\rm mc} = 50$ samples from its local Gaussian proposal in $(\ln d, A_V, R_V)$ using antithetic pairs $(z, -z)$ to reduce variance. Evaluating the 3D Galactic structure prior, the 3D dust prior, and the exact parallax likelihood at each sample $j$ yields the marginalized posterior weight of model $i$:

$$
w_i \;\propto\; \mathcal{L}_{{\rm phot},i}\, \pi_{{\rm model},i} \; \frac{1}{N_{\rm mc}} \sum_{j=1}^{N_{\rm mc}} \pi_{\rm gal}(d_{ij}, \theta_i)\; \pi_{\rm dust}(A_{V,ij} \mid d_{ij})\; \mathcal{L}_\varpi(d_{ij})\; d_{ij}.
$$

The Jacobian factor $d_{ij}$ arises because the local Gaussian proposal is sampled in $\ln d$ while the astrophysical priors are defined per unit physical distance $d$ (sampling directly in scale $s$ without this transformation would bias distances by $d^{-3}$). Across the pipeline, the parallax measurement screens models during the magnitude-space pre-filter in Step 1 (`logl_initthresh`), guides model pruning via the Gauss-Hermite marginalized integral in Step 2, and sets the final posterior weights through the exact point-by-point evaluation $\mathcal{L}_\varpi(d_{ij})$ in Step 3.

**4. Resample the joint posterior.** Finally, `brutus` draws `Ndraws` (250 by default) grid models in proportion to their inverse-probability-corrected posterior weights $w_i / p_i$, and for each selected model draws a continuous sample $(d, A_V, R_V)$ in proportion to its internal Monte Carlo importance weights.

## Breaking the Distance-Extinction Degeneracy with Parallax

Because interstellar dust both dims and reddens starlight, broadband colors constrain dust extinction only to the extent that the star's intrinsic effective temperature is known. Photometrically, a hotter main-sequence star behind a thicker dust column produces nearly the same broadband colors as a cooler, less luminous main-sequence star behind less dust located at a smaller distance. Broadband photometry alone therefore leaves a banana-shaped degeneracy between nearer, weakly reddened solutions and farther, heavily reddened solutions. An astrometric parallax constrains the distance directly, isolating the intersection of the photometric locus and the parallax band.

```{raw} html
:file: figures/degeneracy.svg
:class: figure-svg
```

When no parallax measurement is available (or when its signal-to-noise ratio is low), the 3D Galactic structure and dust-map priors determine how probability mass is distributed along this photometric degeneracy.

## Posterior Outputs

For every fitted star, the output HDF5 file stores the resampled posterior draws of distance, $A_V$, and $R_V$, the corresponding grid model indices `model_idx` (from which mass, age, effective temperature, and surface gravity follow immediately), and fit diagnostics including the minimum $\chi^2$, the number of valid bands, the effective Monte Carlo sample size, and the log-{term}`evidence` ({doc}`understanding_results`).

## Example Sightline Toward Orion

The tutorial dataset includes 207 stars in a field toward the Orion molecular cloud ($\ell = 209.1^\circ$, $b = -19.9^\circ$) observed with Pan-STARRS optical photometry, 2MASS near-infrared photometry, and Gaia parallaxes. Even when fitted without an external 3D dust-map prior, the individual stellar posteriors trace the dust column along the sightline, where extinction remains low out to roughly $0.4\,{\rm kpc}$ and rises to $2\text{--}5\,{\rm mag}$ beyond $0.5\,{\rm kpc}$ behind the Orion molecular cloud wall.

```{raw} html
:file: figures/orion-sightline.svg
:class: figure-svg
```

:::{div} figure-caption
This figure shows the posterior distance and $V$-band extinction ($A_V$) estimates for the 92 well-fitted stars ($\chi^2$ p-value $> 10^{-3}$ and median distance $< 2.5\,{\rm kpc}$; {doc}`understanding_results`) toward Orion ($\ell = 209.1^\circ, b = -19.9^\circ$) fitted without a 3D dust-map prior, plotting 25 posterior draws (amber points) and the posterior median (red circles) for each star.
:::

Combining these individual stellar $(d, A_V)$ posteriors along a line of sight allows `brutus` to fit parametric 3D dust-cloud profiles via {func}`~brutus.analysis.los_clouds_loglike_samples` ({doc}`Tutorial 7 <tutorials/tutorial_07_dust_mapping>`).

## Model Assumptions and Caveats

- **Single-star templates:** Because the default `mist_v9` grid contains only single stars, an unresolved binary has higher apparent flux than a single star of the same color and is placed at a systematically smaller photometric distance.
- **Non-rotating, solar-scaled MIST tracks:** The default grid is built from non-rotating MIST v1.2 models {cite:p}`choi2016,dotter2016` with solar-scaled alpha abundances (${\rm [\alpha/Fe]} = 0$), and therefore does not capture rapid rotation or chemically peculiar atmospheres.
- **Single-family extinction law:** Dust attenuation is modeled by a single {cite:t}`fitzpatrick2009` extinction curve family parameterized by $R_V$.
- **Residual calibration systematics:** Synthetic bolometric corrections and survey zero points retain systematic offsets at the level of $0.02\text{--}0.03\,{\rm mag}$ per band ({doc}`photometric_offsets` details the empirical zero-point corrections and quadrature error floors).

For faint stars lacking precise parallaxes, posterior inferences can depend sensitively on the adopted Galactic and dust priors; comparing fits under default and flat priors ({doc}`priors`) quantifies this prior sensitivity for a given science sample.
