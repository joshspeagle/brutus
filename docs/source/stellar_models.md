# Stellar models

brutus turns the intrinsic parameters of a star into its stellar properties with the MIST v1.2 models {cite:p}`choi2016,dotter2016`, and its properties into photometry with a neural network. This page describes both steps. {doc}`grid_generation` describes the pre-computed grids built from them.

## Evolutionary tracks and EEP

A MIST track follows one star of fixed initial mass and composition through its life. brutus indexes each track by the {term}`equivalent evolutionary point <EEP>`, not by age. Points with the same EEP are at the same phase on every track, so tracks of different masses can be interpolated at fixed EEP. Age follows from (initial mass, EEP, [Fe/H]). Along a track, age changes by very different amounts per step: a star spends most of its life on the main sequence, so the prior includes the time per unit of EEP ({doc}`scientific_background`).

The primary EEPs mark the same physical transition on every track {cite:p}`dotter2016`:

| EEP | Phase |
|---|---|
| 1 | start of the pre-main sequence |
| 202 | {term}`zero-age main sequence <ZAMS>` (ZAMS) |
| 353 | intermediate-age main sequence: central hydrogen fraction 0.3 |
| 454 | {term}`terminal-age main sequence <TAMS>` (TAMS): central hydrogen exhausted |
| 605 | tip of the red giant branch |
| 631 | zero-age core helium burning |
| 707 | end of core helium burning |
| 808 | start of the thermally pulsing asymptotic giant branch |

The default grid spans EEP 202 to 808: from the ZAMS to the thermally pulsing AGB.

```{raw} html
:file: figures/hr-tracks.svg
:class: figure-svg
```

:::{div} figure-caption
Tracks of the default grid at solar metallicity. The dots mark the primary EEPs from the ZAMS to the TP-AGB; on the 1 M☉ track they are labelled.
:::

## Tracks and isochrones

brutus reads the MIST models in two forms:

| | Evolutionary tracks | Isochrones |
|---|---|---|
| Fixed | initial mass, composition | age, composition |
| Varies along it | EEP | initial mass |
| Parameter classes | {class}`~brutus.core.EEPTracks` | {class}`~brutus.core.Isochrone` |
| Photometry classes | {class}`~brutus.core.StarEvolTrack`, {class}`~brutus.core.GridGenerator` | {class}`~brutus.core.StellarPop` |
| Used for | individual stars and the model grids | star clusters ({doc}`population_modeling`) |
| Data file | `fetch_tracks()` | `fetch_isos()` |

`EEPTracks.get_predictions` takes `[mini, eep, feh, afe]` and returns the columns named in `tracks.predictions`: by default log age, log luminosity, log effective temperature, log surface gravity, the surface [Fe/H] and [α/Fe], and the age weight `agewt`.

### Empirical corrections

By default (`apply_corr=True`), brutus corrects the MIST temperatures and radii of main-sequence stars below 1 M☉, where the models are systematically offset from observations:

$$
\Delta \log_{10} T_{\rm eff} = f\, \log_{10}\!\left(1 + 0.09\,\Delta M\right),
\qquad
\Delta \log_{10} R = f\, \log_{10}\!\left(1 - 0.09\,\Delta M\right),
\qquad
f = \frac{e^{0.5\,{\rm [Fe/H]}}}{1 + e^{({\rm EEP} - 454)/30}},
$$

with $\Delta M = M_{\rm init} / M_\odot - 1$. The factor $f$ falls from about $e^{0.5\,{\rm [Fe/H]}}$ on the main sequence to 0 after the turnoff (EEP 454). `corr_params` sets the four coefficients (0.09, −0.09, 30, 0.5).

## From stellar parameters to photometry

A neural network ({class}`~brutus.core.neural_nets.FastNNPredictor`, file `nn_c3k.h5` from `fetch_nns()`) predicts the bolometric correction ${\rm BC}_b$ in each band $b$ from the effective temperature, surface gravity, surface [Fe/H], [α/Fe], $A_V$ and $R_V$. The extinction is part of the network's input, so the reddened magnitude comes out directly:

$$
m_b = -2.5 \log_{10} \frac{L}{L_\odot} + 4.74 - {\rm BC}_b\left(T_{\rm eff}, \log g, {\rm [Fe/H]}_{\rm surf}, {\rm [\alpha/Fe]}, A_V, R_V\right) + \mu(d),
$$

where 4.74 is the bolometric magnitude of the Sun and $\mu(d) = 5 \log_{10}(d / 10\,{\rm pc})$ is the {term}`distance modulus`. The network returns NaN outside the range it was trained on: $T_{\rm eff}$ from 2,500 to 50,000 K, $\log g$ from −1 to 5.5, [Fe/H] from −4 to +0.5, [α/Fe] from −0.2 to +0.6, $A_V$ from 0 to 6 and $R_V$ from 2 to 5.

A model grid does not call the network during a fit. It stores each model's magnitudes at 1 kpc and two {term}`reddening vectors <reddening vector>` fitted to the network's output ({doc}`grid_generation`).

## Binaries

`StarEvolTrack.get_seds` and `StellarPop.get_seds` can add an unresolved companion, whose flux is added to the primary's:

- `StarEvolTrack.get_seds(smf=...)` sets the secondary mass fraction of one star.
- `StellarPop.get_seds(binary_fraction=...)` sets the same mass ratio for every star of a population. Despite its name, it is a mass ratio, not a fraction of stars that are binaries.

Companions are added only to primaries with EEP up to `eep_binary_max` (480 by default). The default grid has no binaries ({doc}`grid_generation`).

(available-filters)=

## Available filters

The network and the default grid cover 49 bands. `brutus.data.filters` holds each survey's list (for example `filters.gaia` or `filters.tmass`), and `filters.FILTERS` holds all of them.

| Survey | List | Bands |
|---|---|---|
| Gaia | `gaia` | `Gaia_G_MAW`, `Gaia_BP_MAWf`, `Gaia_RP_MAW` |
| SDSS | `sdss` | `SDSS_u`, `SDSS_g`, `SDSS_r`, `SDSS_i`, `SDSS_z` |
| Pan-STARRS | `ps` | `PS_g`, `PS_r`, `PS_i`, `PS_z`, `PS_y`, `PS_w`, `PS_open` |
| DECam | `decam` | `DECam_u`, `DECam_g`, `DECam_r`, `DECam_i`, `DECam_z`, `DECam_Y` |
| Johnson-Cousins | `bessell` | `Bessell_U`, `Bessell_B`, `Bessell_V`, `Bessell_R`, `Bessell_I` |
| 2MASS | `tmass` | `2MASS_J`, `2MASS_H`, `2MASS_Ks` |
| VISTA | `vista` | `VISTA_Z`, `VISTA_Y`, `VISTA_J`, `VISTA_H`, `VISTA_Ks` |
| UKIDSS | `ukidss` | `UKIDSS_Z`, `UKIDSS_Y`, `UKIDSS_J`, `UKIDSS_H`, `UKIDSS_K` |
| WISE | `wise` | `WISE_W1`, `WISE_W2`, `WISE_W3`, `WISE_W4` |
| Tycho | `tycho` | `Tycho_B`, `Tycho_V` |
| Hipparcos | `hipp` | `Hipparcos_Hp` |
| Kepler | `kepler` | `Kepler_D51`, `Kepler_Kp` |
| TESS | `tess` | `TESS` |

## Limitations

- **Rotation**: the default models and grid do not rotate. `fetch_isos(iso="MIST_1.2_vvcrit0.4")` gives isochrones with rotation for population work.
- **Composition**: the default grid is solar-scaled ([α/Fe] = 0).
- **Stellar types**: the models do not include white dwarfs, brown dwarfs or binary evolution (mass transfer, mergers).
- **Model systematics**: the empirical corrections reduce, but do not remove, the offsets of low-mass stars. {doc}`photometric_offsets` describes the per-band corrections that remain.
