# Model Grids

{class}`~brutus.analysis.BruteForce` fits stars against a pre-computed grid of stellar evolution models (`StarGrid`). Each grid node tabulates one stellar template from {doc}`stellar_models` with its synthetic broadband photometry and linear reddening derivatives stored on disk, allowing `brutus` to fit large photometric catalogs without evaluating evolutionary track interpolators or neural networks at runtime.

## Grid File Structure

Each pre-computed HDF5 grid file contains three aligned datasets with one row per stellar model:

| Dataset | Contents |
|---|---|
| `labels` | Intrinsic grid parameters: `mini`, `eep`, `feh`, `afe`, and `smf` |
| `parameters` | Predicted stellar properties: `loga`, `logl`, `logt`, `logg`, `feh_surf`, `afe_surf`, and the evolutionary lifetime weight `agewt` |
| `mag_coeffs` | Three polynomial reddening coefficients per passband: $(m_{0,b},\, a_b,\, b_b)$ |

At distance $d$ and extinction $(A_V, R_V)$, the apparent magnitude in passband $b$ is evaluated as

$$
m_b = m_{0,b} + A_V \left( a_b + b_b\, R_V \right) + 5 \log_{10} \frac{d}{1\,{\rm kpc}},
$$

where $m_{0,b}$ is the unreddened magnitude at $1\,{\rm kpc}$ and $a_b + b_b R_V$ is the linear {term}`reddening vector`. Storing $(m_{0,b}, a_b, b_b)$ allows a single grid table to predict reddened photometry analytically at any distance and extinction.

## Shipped Model Grids

Calling `fetch_grids(grid=...)` downloads one of the three pre-computed grids:

| Grid Name | Description |
|---|---|
| `mist_v9` (default) | MIST v1.2 evolutionary grid with empirical low-mass corrections across 49 passbands |
| `mist_v8` | Previous release of the MIST v1.2 grid |
| `bayestar_v5` | Empirical Pan-STARRS and 2MASS stellar main-sequence/giant locus from the Bayestar dust maps {cite:p}`green2019` |

Download the corresponding empirical zero-point calibration table by passing the same `grid` identifier to `fetch_offsets(grid=...)`.

The default `mist_v9` grid contains 613,530 single-star models spanning the following parameter lattice:

| Grid Axis | Range | Grid Spacing |
|---|---|---|
| Initial mass (`mini`) | $0.5$ to $2.0\,M_\odot$ | $0.025\,M_\odot$ |
| Evolutionary phase (`eep`) | $202$ to $808$ | $\Delta{\rm EEP} = 6$ up to ${\rm EEP} = 454$ (TAMS), then $\Delta{\rm EEP} = 2$ |
| Metallicity (`feh`) | $-3.0$ to $+0.45\,{\rm dex}$ | $0.1\,{\rm dex}$ up to $-2.0$, then $0.05\,{\rm dex}$ |
| Alpha enhancement (`afe`) | $0.0\,{\rm dex}$ | Fixed (solar-scaled) |
| Secondary mass fraction (`smf`) | $0.0$ | Fixed (single stars) |

Because stellar models older than $13.8\,{\rm Gyr}$ are excluded during grid construction, stars below the $13.8\,{\rm Gyr}$ main-sequence turnoff mass (approximately $0.9\,M_\odot$ at ${\rm [Fe/H]} = 0.0$, $0.8\,M_\odot$ at ${\rm [Fe/H]} = -1.0$, and $0.78\,M_\odot$ at ${\rm [Fe/H]} = -3.0$) remain on the main sequence across all valid ages.

```{raw} html
:file: figures/grid-coverage.svg
:class: figure-svg
```

:::{div} figure-caption
This diagram illustrates the coverage of the default `mist_v9` grid in initial mass and EEP at solar metallicity (${\rm [Fe/H]} = 0.00$), plotting one vertical track per initial mass node from the zero-age main sequence (EEP 202) up to the last evolutionary point younger than $13.8\,{\rm Gyr}$.
:::

```{important}
The default `mist_v9` grid covers initial masses between $0.5\,M_\odot$ and $2.0\,M_\odot$. Fitting M dwarfs ($M_{\rm init} < 0.5\,M_\odot$) or massive stars ($M_{\rm init} > 2.0\,M_\odot$) requires generating a custom grid that spans the target mass range.
```

## Loading a Grid

{func}`~brutus.data.load_models` loads the subset of passbands required for your catalog in the exact column order you specify:

```python
from brutus.core import StarGrid
from brutus.data import load_models
from brutus.data.filters import gaia, tmass

models, labels, label_mask = load_models("grid_mist_v9.h5", filters=gaia + tmass)
grid = StarGrid(models, labels)
```

By default, `labels` combines the intrinsic grid axes with `loga`, `logl`, `logt`, `logg`, and `agewt` (pass `labels=[...]` to request additional columns such as `feh_surf`), while `label_mask` marks which fields correspond to intrinsic grid axes. Unresolved binary templates (`smf > 0`) are excluded unless `include_binaries=True` is set. Constant grid axes (such as `afe` on the solar-scaled MIST grid or `smf` when binaries are excluded) are retained as single-value columns so that `StarGrid` can validate queries against them. Passing `include_ms=False` or `include_postms=False` restricts the loaded models to post-main-sequence (${\rm EEP} > 454$) or main-sequence (${\rm EEP} \le 454$) phases, respectively. Because the loaded grid array is held in RAM, pass only the filters present in your photometric catalog.

## Generating a Custom Grid

{class}`~brutus.core.GridGenerator` builds custom HDF5 model grids from the continuous MIST evolutionary tracks (`fetch_tracks()`) and the bolometric-correction neural network (`fetch_nns()`):

```python
import numpy as np
from brutus.core import EEPTracks, GridGenerator
from brutus.data.filters import gaia, tmass

generator = GridGenerator(EEPTracks(), filters=gaia + tmass)
generator.make_grid(
    mini_grid=np.arange(0.1, 0.5, 0.01),  # low-mass M dwarfs below the default grid
    mini_bound=0.1,  # minimum primary mass threshold (default 0.5 Msun)
    afe_grid=np.array([0.0]),
    output_file="grid_mdwarfs.h5",
)
```

Any grid axis omitted from `make_grid` falls back to its default lattice (note that the default `make_grid` lattice extends `[Fe/H]` to $+0.5$ and evaluates five `afe_grid` values from $-0.2$ to $+0.6$, producing a grid more than five times larger than `mist_v9` unless `afe_grid=np.array([0.0])` is specified). At each grid node, `make_grid` evaluates the neural network across a 2D lattice of $A_V \in [0.0, 1.5]$ and $R_V \in [2.4, 4.2]$ and fits the linear reddening coefficients $(m_{0,b}, a_b, b_b)$ in two weighted linear regression stages, first regressing magnitude against $A_V$ at each fixed $R_V$ and then regressing those slopes against $R_V$ (weighted via `av_wt` and `rv_wt` to anchor $A_V = 0$ and emphasize $R_V \approx 3.3$).

## Computational Performance and Tuning

Because `BruteForce` evaluates `loglike_grid` across every template in the loaded grid, per-star runtime scales linearly with the number of grid models and passbands. On the reference benchmark in `bench/RESULTS.md` (`brutus` v1.1.1, default `mist_v9` grid, 8 passbands with parallax on a 4-core CPU), fitting one star takes approximately $0.5\,{\rm s}$.

The primary tuning parameters of {meth}`~brutus.analysis.BruteForce.fit` that govern runtime and memory consumption are:

| Parameter | Default | Effect on Computation |
|---|---|---|
| `Nmc_prior` | `50` | Number of antithetic Monte Carlo importance samples per retained model |
| `Ndraws` | `250` | Number of resampled posterior draws written to disk per star |
| `max_models` | `50000` | Maximum number of grid models retained for Step 3 Monte Carlo integration |
| `mem_lim` | `8000` | Maximum memory budget (in MB) for Monte Carlo arrays (`Nmc_prior` is automatically reduced if needed) |
| `running_io` | `True` | Flush each star's posterior draws to HDF5 immediately upon completion so interrupted runs retain completed stars |
| `rstate` | `None` | Explicit random number generator (`numpy.random.Generator` or `RandomState`) for reproducible sampling |

Loading only the observed passbands and restricting the grid to the relevant stellar regime (for example, narrowing the mass or metallicity bounds or setting `include_postms=False` for main-sequence samples) provides the largest speedup.
