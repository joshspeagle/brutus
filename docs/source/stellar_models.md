# Stellar Models

`brutus` maps intrinsic stellar labels to physical stellar parameters using the MIST v1.2 stellar evolution library {cite:p}`choi2016,dotter2016`, and converts those physical parameters into reddened broadband photometry using a neural network bolometric-correction emulator ({doc}`grid_generation` describes how these components are pre-tabulated into model grids).

## Evolutionary Tracks and Equivalent Evolutionary Points (EEP)

A stellar evolutionary track follows a star of fixed initial mass and chemical composition as it ages. Rather than indexing tracks directly by physical age, `brutus` parameterizes each track by its {term}`equivalent evolutionary point <EEP>` because equal EEP values correspond to the same physical evolutionary stage across stars of widely different masses, enabling smooth multi-dimensional interpolation at fixed EEP. Physical stellar age is then predicted uniquely as a function of $(M_{\rm init}, {\rm EEP}, {\rm [Fe/H]})$. Since a star spends the vast majority of its lifetime on the main sequence and traverses post-main-sequence phases rapidly, physical age advances at highly non-uniform rates per unit EEP, and the intrinsic model prior weights each EEP step by the evolutionary lifetime derivative $dt / d\,{\rm EEP}$ ({doc}`scientific_background`).

Primary EEP values mark equivalent physical transitions across every evolutionary track {cite:p}`dotter2016`:

| EEP | Evolutionary Phase |
|---|---|
| 1 | Pre-main-sequence birthline |
| 202 | {term}`Zero-age main sequence <ZAMS>` (ZAMS; onset of stable core hydrogen burning) |
| 353 | Intermediate-age main sequence (central hydrogen mass fraction $X_c = 0.3$) |
| 454 | {term}`Terminal-age main sequence <TAMS>` (TAMS; core hydrogen exhaustion) |
| 605 | Tip of the red giant branch (RGB tip; helium flash or ignition) |
| 631 | Zero-age core helium burning (horizontal branch or red clump) |
| 707 | Terminal-age core helium burning (core helium exhaustion) |
| 808 | Onset of the thermally pulsing asymptotic giant branch (TP-AGB) |

The default `mist_v9` grid spans EEP 202 to 808, covering evolution from the zero-age main sequence through the onset of the thermally pulsing asymptotic giant branch.

```{raw} html
:file: figures/hr-tracks.svg
:class: figure-svg
```

:::{div} figure-caption
This Hertzsprung-Russell diagram shows the MIST v1.2 evolutionary tracks in the default `brutus` grid at solar metallicity (${\rm [Fe/H]} = 0.00$) for initial masses between $0.6\,M_\odot$ and $2.0\,M_\odot$. Filled circles mark the primary EEP transitions from the ZAMS (EEP 202) to the TP-AGB (EEP 808), with phase labels shown along the $1.0\,M_\odot$ track.
:::

## Evolutionary Tracks vs. Isochrones

`brutus` provides interpolators for the MIST library in two complementary representations:

| Property | Evolutionary Tracks | Isochrones |
|---|---|---|
| Held fixed along curve | Initial mass, chemical composition | Stellar age, chemical composition |
| Varies along curve | {term}`EEP` | Initial mass (parameterized by EEP) |
| Parameter interpolator | {class}`~brutus.core.EEPTracks` | {class}`~brutus.core.Isochrone` |
| Photometry generator | {class}`~brutus.core.StarEvolTrack`, {class}`~brutus.core.GridGenerator` | {class}`~brutus.core.StellarPop` |
| Primary application | Individual field stars and pre-computed grids | Coeval star clusters ({doc}`population_modeling`) |
| Data fetcher | `fetch_tracks()` | `fetch_isos()` |

Calling `EEPTracks.get_predictions([mini, eep, feh, afe])` interpolates the track grid and returns the physical quantities listed in `tracks.predictions` (by default `loga`, `logl`, `logt`, `logg`, `feh_surf`, `afe_surf`, and the evolutionary lifetime weight `agewt`).

### Empirical Low-Mass Radius and Temperature Corrections

Because standard 1D convective stellar models systematically overpredict effective temperatures and underpredict radii for low-mass main-sequence dwarfs ($M_{\rm init} < 1\,M_\odot$), `brutus` applies an empirical main-sequence calibration by default (`apply_corr=True`):

$$
\Delta \log_{10} T_{\rm eff} = f\, \log_{10}\!\left(1 + 0.09\,\Delta M\right),
\qquad
\Delta \log_{10} R = f\, \log_{10}\!\left(1 - 0.09\,\Delta M\right),
\qquad
f = \frac{e^{0.5\,{\rm [Fe/H]}}}{1 + e^{({\rm EEP} - 454)/30}},
$$

where $\Delta M = M_{\rm init} / M_\odot - 1$. The logistic taper $f$ scales with metallicity as $e^{0.5\,{\rm [Fe/H]}}$ on the main sequence and decays smoothly to zero past the main-sequence turnoff (${\rm EEP} = 454$). The four calibration coefficients $(0.09, -0.09, 30, 0.5)$ can be customized via `corr_params`.

## From Stellar Parameters to Broadband Photometry

A three-layer neural network ({class}`~brutus.core.neural_nets.FastNNPredictor`, loaded from `nn_c3k.h5` via `fetch_nns()`) emulates the C3K stellar atmosphere grid to predict the reddened bolometric correction ${\rm BC}_b$ in each photometric band $b$ as a function of effective temperature, surface gravity, surface iron abundance, alpha enhancement, $A_V$, and $R_V$. Because interstellar extinction is evaluated directly inside the bolometric-correction network, the apparent magnitude at distance $d$ is

$$
m_b = -2.5 \log_{10} \frac{L}{L_\odot} + 4.74 - {\rm BC}_b\left(T_{\rm eff}, \log g, {\rm [Fe/H]}_{\rm surf}, {\rm [\alpha/Fe]}, A_V, R_V\right) + \mu(d),
$$

where $4.74$ is the solar bolometric magnitude and $\mu(d) = 5 \log_{10}(d / 10\,{\rm pc})$ is the {term}`distance modulus`. The emulator returns `NaN` for inputs outside its training domain: $T_{\rm eff} \in [2500, 50000]\,{\rm K}$, $\log g \in [-1.0, 5.5]$, ${\rm [Fe/H]} \in [-4.0, +0.5]$, ${\rm [\alpha/Fe]} \in [-0.2, +0.6]$, $A_V \in [0, 6]$, and $R_V \in [2, 5]$.

Pre-computed model grids (`StarGrid`) do not evaluate the neural network at fit time; instead, they store each model's unreddened magnitudes at $1\,{\rm kpc}$ together with two linear {term}`reddening vectors <reddening vector>` fitted to the network's predictions across $(A_V, R_V)$ ({doc}`grid_generation`).

## Unresolved Binary Systems

Both `StarEvolTrack.get_seds` and `StellarPop.get_seds` can model an unresolved binary companion by summing the linear fluxes of a primary star of mass $M_1$ and a coeval secondary star of mass $M_2 = q\,M_1$ at the same metallicity and distance:

- `StarEvolTrack.get_seds(smf=q)` specifies the secondary mass fraction $q = M_2 / M_1 \in [0, 1]$ for an individual star.
- `StellarPop.get_seds(binary_fraction=q)` evaluates the isochrone at a fixed secondary mass ratio $q$ across the population (note that in `StellarPop.get_seds`, `binary_fraction` denotes the mass ratio $q = M_2 / M_1$ rather than the fraction of cluster stars in binary systems).

Secondary companions are added only for primaries with EEP up to `eep_binary_max` ($480$ by default, just past the main-sequence turnoff). The default `mist_v9` grid contains only single stars ($q = 0$; see {doc}`grid_generation`).

(available-filters)=

## Supported Photometric Passbands

The bolometric-correction neural network and the default `mist_v9` grid include 49 photometric passbands across 13 optical and infrared surveys. Survey-specific filter lists are available in `brutus.data.filters` (for example, `filters.gaia` or `filters.tmass`), and `filters.FILTERS` contains the full list of 49 passbands:

| Survey | Filter List | Passbands |
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

## Model Scope and Limitations

- **Stellar rotation:** The default evolutionary tracks and pre-computed grids use non-rotating MIST v1.2 models ($v / v_{\rm crit} = 0.0$). Rotating isochrones ($v / v_{\rm crit} = 0.4$) can be downloaded for population modeling via `fetch_isos(iso="MIST_1.2_vvcrit0.4")`.
- **Alpha-element abundances:** The shipped MIST model files are solar-scaled (${\rm [\alpha/Fe]} = 0.0$).
- **Exotic and compact objects:** White dwarfs, brown dwarfs ($M_{\rm init} < 0.08\,M_\odot$), and interacting binaries undergoing mass transfer or common-envelope evolution are not included in the MIST grid.
- **Residual photometric systematics:** While the empirical low-mass radius and temperature corrections reduce color-magnitude offsets on the lower main sequence, per-band zero-point residuals of a few hundredths of a magnitude remain and are calibrated via {doc}`photometric_offsets`.
