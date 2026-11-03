# Glossary

Definitions of astrophysical, photometric, and statistical terms used throughout the `brutus` documentation.

## Stellar Evolution

```{glossary}
EEP
Equivalent Evolutionary Point
   Dimensionless index along an evolutionary track that marks the same physical evolutionary phase on every track regardless of initial mass. Primary values include 202 ({term}`ZAMS`), 454 ({term}`TAMS`), and 605 (tip of the red giant branch). The default grid spans EEP 202 to 808 ({doc}`stellar_models`).

ZAMS
Zero-Age Main Sequence
   Onset of stable core hydrogen fusion (EEP 202), marking the start of the main sequence.

TAMS
Terminal-Age Main Sequence
   Exhaustion of core hydrogen (EEP 454), after which a star evolves off the main sequence toward the subgiant and red giant branches.

isochrone
   Curve in the Hertzsprung-Russell diagram connecting stars of the same age across a range of initial masses. Used to model coeval stellar populations such as open and globular clusters (compare with {term}`evolutionary track`).

evolutionary track
   Path traced through the Hertzsprung-Russell diagram by a star of fixed initial mass and composition as it ages, parameterized in `brutus` by {term}`EEP` (compare with {term}`isochrone`).

initial mass
   Mass of a star at birth ($M_{\rm init}$), in solar masses ($M_\odot$). Together with evolutionary phase ({term}`EEP`) and chemical composition, initial mass determines the star's present-day physical properties.

MIST
   MESA Isochrones and Stellar Tracks {cite:p}`choi2016,dotter2016`, the stellar evolution library used by `brutus`. MIST models span $0.1$ to $300\,M_\odot$ and ${\rm [Fe/H]}$ from $-4.0$ to $+0.5$ across evolutionary phases from the pre-main sequence through the asymptotic giant branch.

IMF
Initial Mass Function
   Probability distribution of stellar initial masses at birth. `brutus` uses the {cite:t}`kroupa2001` broken power-law IMF, $\pi_{\rm IMF}(M) \propto M^{-1.3}$ for $0.08 \le M/M_\odot \le 0.5$ and $\pi_{\rm IMF}(M) \propto M^{-2.3}$ for $0.5 < M/M_\odot \le 100$.
```

## Chemical Abundances

```{glossary}
metallicity
[Fe/H]
   Logarithmic iron-to-hydrogen number abundance relative to the Sun, ${\rm [Fe/H]} = \log_{10}(N_{\rm Fe}/N_{\rm H}) - \log_{10}(N_{\rm Fe}/N_{\rm H})_\odot$ in dex. The Sun has ${\rm [Fe/H]} = 0$, and metal-poor stars have ${\rm [Fe/H]} < 0$.

alpha enhancement
[α/Fe]
   Logarithmic abundance of $\alpha$-process elements (O, Mg, Si, Ca, Ti) relative to iron, in dex relative to the Sun. Old halo and thick-disk stars typically have ${\rm [\alpha/Fe]} \approx +0.3$. While `brutus` model interfaces accept `afe`, the shipped MIST model files are solar-scaled (${\rm [\alpha/Fe]} = 0$).
```

## Photometry

```{glossary}
magnitude
   Logarithmic brightness scale $m = -2.5 \log_{10}(F / F_0)$, where fainter objects have larger magnitudes and a difference of 5 magnitudes corresponds to a factor of 100 in flux.

flux density
maggie
   Linear flux density unit used by `brutus`, where $1\text{ maggie}$ is the flux of a zero-magnitude source in the survey's native photometric system ($F = 10^{-0.4\,m}$). {func}`~brutus.utils.inv_magnitude` converts magnitudes and their uncertainties into maggies.

SED
Spectral Energy Distribution
   Distribution of emitted or observed flux across photometric passbands. `brutus` predicts synthetic broadband SEDs from stellar parameters and compares them against observed fluxes.

parallax
   Apparent angular shift of a star caused by Earth's orbit around the Sun, measured in milliarcseconds (mas). For a true distance $d$ in kpc, the model parallax is $\varpi = 1 / d\text{ mas}$ (equivalently $1000 / d\text{ mas}$ for $d$ in pc).

distance modulus
   Logarithmic distance scale $\mu = m - M = 5 \log_{10}(d / 10\text{ pc})$, equal to $10\text{ mag}$ at $d = 1\text{ kpc}$.

photometric offset
   Multiplicative per-band calibration factor applied to observed fluxes and uncertainties to remove systematic zero-point differences between synthetic models and survey data ({doc}`photometric_offsets`). Loaded via {func}`~brutus.data.load_offsets` and passed to `BruteForce.fit` via `phot_offsets`.
```

## Extinction & Dust

```{glossary}
extinction
   Wavelength-dependent attenuation of starlight by interstellar dust, measured in magnitudes. $A_V$ denotes extinction in the Johnson $V$ band ($\sim 550\text{ nm}$).

reddening
   Shift toward redder colors caused by stronger dust attenuation at shorter wavelengths, often expressed as the color excess $E(B - V) = (B - V)_{\rm obs} - (B - V)_0$.

R_V
   Total-to-selective extinction ratio $R_V = A_V / E(B - V)$, which characterizes the shape of the dust extinction curve and tracks the typical dust grain size. Diffuse interstellar dust averages $R_V \approx 3.1$ to $3.3$, whereas dense molecular clouds can reach $R_V \approx 4$ to $5$.

reddening vector
   Change in magnitude per unit $A_V$ across photometric passbands, parameterized in `brutus` model grids as $R_b(\theta) + R_V\,R'_b(\theta)$ for each model $\theta$ and filter $b$.

line of sight
   Direction from the observer toward sky coordinates $(\ell, b)$. Because dust extinction accumulates monotonically with distance along a sightline, the joint $(d, A_V)$ posteriors of stars along that direction trace the 3D dust distribution.

Bayestar
   Three-dimensional dust map of {cite:t}`green2019` inferred from Pan-STARRS 1 and 2MASS photometry across the sky north of $\delta = -30^\circ$, used in `brutus` as an optional distance-dependent prior on $A_V$ ({doc}`priors`).
```

## Statistical Inference

```{glossary}
posterior
   Conditional probability distribution of model parameters given the observed data, $P(\theta, \phi \mid D) \propto \mathcal{L}(D \mid \theta, \phi)\,\pi(\theta, \phi)$.

prior
   Probability distribution $\pi(\theta, \phi)$ encoding astrophysical knowledge of the initial mass function, evolutionary lifetimes, Galactic structure, and 3D dust before fitting the observed fluxes ({doc}`priors`).

likelihood
   Probability density $\mathcal{L}(D \mid \theta, \phi)$ of observing the measured fluxes and parallax given model parameters $(\theta, \phi)$.

evidence
   Marginal likelihood $P(D) = \int \mathcal{L}(D \mid \theta, \phi)\,\pi(\theta, \phi)\,d\theta\,d\phi$ integrated over the parameter space. `brutus` saves the log-evidence per star as `obj_log_evid`, which is comparable across fits that share the same model grid and priors.

marginalization
   Integrating the joint posterior distribution over nuisance parameters to obtain the marginal posterior for parameters of interest, such as integrating over $(d, A_V, R_V)$ to weight each grid model or summing over stellar models to obtain the marginal distance distribution.

brute force
   Grid-based Bayesian inference strategy in which the likelihood is evaluated across every model in a pre-computed stellar grid, continuous extrinsic parameters $(d, A_V, R_V)$ are optimized and integrated via local Gaussian approximations and Monte Carlo importance sampling, and the posterior is formed as a weighted mixture over the grid.
```

## brutus Classes

```{glossary}
model grid
   Pre-computed table of stellar models evaluated across initial mass, {term}`EEP`, and metallicity (plus optional alpha enhancement and binary mass ratio), storing 1 kpc magnitudes and reddening vectors alongside physical stellar properties. The default `grid_mist_v9` grid contains 613,530 models.

scale factor
   Flux dilution factor $s = (1\text{ kpc} / d)^2$ between a model stored at $1\text{ kpc}$ and a star at distance $d$ in kpc.

StarGrid
   Container class ({class}`~brutus.core.StarGrid`) holding a pre-computed {term}`model grid` for fast SED evaluation and multi-linear grid interpolation.

BruteForce
   Inference engine ({class}`~brutus.analysis.BruteForce`) that fits individual stars over a {term}`StarGrid` and writes posterior draws of distance, extinction, and model indices to HDF5.

EEPTracks
   Interpolator class ({class}`~brutus.core.EEPTracks`) that predicts stellar physical properties along MIST evolutionary tracks as a function of $(M_{\rm init}, {\rm EEP}, {\rm [Fe/H]}, {\rm [\alpha/Fe]})$.

Isochrone
   Interpolator class ({class}`~brutus.core.Isochrone`) that predicts stellar physical properties along a coeval MIST {term}`isochrone` at specified $({\rm [Fe/H]}, {\rm [\alpha/Fe]}, \log_{10} t)$.

StellarPop
   Population synthesis class ({class}`~brutus.core.StellarPop`) that combines an {term}`Isochrone` with neural network bolometric corrections to generate synthetic photometry for single and binary stellar populations.
```

```{note}
**Model Coverage**

The MIST models shipped with `brutus` cover main-sequence stars, subgiants, red giants, core helium-burning stars, and asymptotic giant branch stars ($0.1$ to $300\,M_\odot$ in the full tracks, and $0.5$ to $2.0\,M_\odot$ in the default `mist_v9` grid). Compact remnants (white dwarfs, neutron stars, black holes), brown dwarfs ($< 0.08\,M_\odot$), and interacting binaries are outside the grid domain.
```

## See Also

- {doc}`stellar_models` for MIST evolutionary tracks, EEPs, and neural network bolometric corrections
- {doc}`scientific_background` for the statistical forward model and four-step inference pipeline
- {doc}`priors` for the IMF, Galactic structure, and 3D dust priors
- {ref}`available-filters` for the 49 supported photometric passbands
