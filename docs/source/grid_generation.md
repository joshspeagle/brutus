# Model grids

`BruteForce` fits stars against a pre-computed grid of stellar models. Each grid point is one star from {doc}`stellar_models`, with its photometry stored so that a fit never runs the stellar models or the neural network. This page describes what a grid holds, the grids you can download, and how to build your own.

## What a grid holds

A grid file (HDF5) has three datasets, with one row per model:

| Dataset | Content |
|---|---|
| `labels` | the grid parameters: `mini`, `eep`, `feh`, `afe`, `smf` |
| `parameters` | the stellar properties: `loga`, `logl`, `logt`, `logg`, `feh_surf`, `afe_surf`, and the age weight `agewt` |
| `mag_coeffs` | three coefficients per band: $(m_{0,b},\, a_b,\, b_b)$ |

The magnitude in band $b$ at distance $d$ and extinction $(A_V, R_V)$ is

$$
m_b = m_{0,b} + A_V \left( a_b + b_b\, R_V \right) + 5 \log_{10} \frac{d}{1\,{\rm kpc}},
$$

where $m_{0,b}$ is the unreddened magnitude at 1 kpc and $a_b + b_b R_V$ is the {term}`reddening vector`. One grid therefore covers every distance and extinction without storing them.

## Available grids

`fetch_grids(grid=...)` downloads:

| Grid | Content |
|---|---|
| `mist_v9` (default) | MIST v1.2 with the empirical corrections, 49 bands |
| `mist_v8` | the previous version of the MIST grid |
| `bayestar_v5` | the grid of the Bayestar dust maps {cite:p}`green2019`: Pan-STARRS and 2MASS bands |

Each grid has its matching photometric offsets: `fetch_offsets(grid=...)` with the same name.

The default `mist_v9` grid has 613,530 models:

| Parameter | Range | Step |
|---|---|---|
| initial mass | 0.5 to 2.0 M☉ | 0.025 M☉ |
| EEP | 202 to 808 | 6 to EEP 454, then 2 |
| [Fe/H] | −3.0 to +0.45 | 0.1 to −2.0, then 0.05 |
| [α/Fe] | 0 | |
| secondary mass fraction | 0 (single stars) | |

Models older than 13.8 Gyr are removed, so low-mass stars never leave the main sequence in the grid. The limit depends on metallicity: about 0.9 M☉ at solar [Fe/H], 0.8 M☉ at [Fe/H] = −1 and 0.78 M☉ at [Fe/H] = −3.

```{raw} html
:file: figures/grid-coverage.svg
:class: figure-svg
```

:::{div} figure-caption
The models of the default grid at solar metallicity: one line per initial mass, from EEP 202 up to the last EEP younger than 13.8 Gyr.
:::

```{important}
The default grid does not cover stars below 0.5 M☉ or above 2.0 M☉. To fit M dwarfs or massive stars, build a grid that covers them.
```

## Loading a grid

{func}`~brutus.data.load_models` reads a grid with the bands you choose, in your order:

```python
from brutus.core import StarGrid
from brutus.data import load_models
from brutus.data.filters import gaia, tmass

models, labels, label_mask = load_models("grid_mist_v9.h5", filters=gaia + tmass)
grid = StarGrid(models, labels)
```

`labels` holds the grid parameters and, by default, `loga`, `logl`, `logt`, `logg` and `agewt`; pass `labels=[...]` to choose others, such as `feh_surf`. `label_mask` marks which columns are grid parameters. A constant `afe` column is dropped. Binary models (`smf` > 0) are dropped unless `include_binaries=True`. `include_ms` and `include_postms` keep only the main sequence (EEP ≤ 454) or only later phases. Load only the bands you fit: the grid is held in memory.

## Building a grid

{class}`~brutus.core.GridGenerator` builds a grid from the evolutionary tracks (`fetch_tracks()`) and the neural network (`fetch_nns()`):

```python
import numpy as np
from brutus.core import EEPTracks, GridGenerator
from brutus.data.filters import gaia, tmass

generator = GridGenerator(EEPTracks(), filters=gaia + tmass)
generator.make_grid(
    mini_grid=np.arange(0.1, 0.5, 0.01),  # M dwarfs, below the default grid
    mini_bound=0.1,  # make_grid drops primaries below this (default 0.5)
    afe_grid=np.array([0.0]),
    output_file="grid_mdwarfs.h5",
)
```

Unset grids take the defaults of `make_grid`. They match the default grid in mass and EEP, but [Fe/H] runs to +0.5 and `afe_grid` has five values (−0.2 to +0.6), so the grid is more than five times larger. For each model, `make_grid` evaluates the network on a lattice of $A_V$ (0 to 1.5) and $R_V$ (2.4 to 4.2) and fits the reddening vector in two linear steps: magnitude against $A_V$ at each $R_V$, then that slope against $R_V$. The weights force the fit through $A_V = 0$ and favour $R_V$ near 3.3 (`av_wt`, `rv_wt`).

## Performance

Most of the time of a fit goes to fitting every model of the grid (`loglike_grid`); the time scales with the number of models. In the measurements of `bench/RESULTS.md` (brutus v1.1.1, the default grid, 8 bands, parallaxes, a 4-core machine), one star took about 0.5 s.

The options of {meth}`~brutus.analysis.BruteForce.fit` that set the cost:

| Option | Default | Effect |
|---|---|---|
| `Nmc_prior` | 50 | Monte Carlo draws per kept model |
| `Ndraws` | 250 | posterior draws saved per star |
| `max_models` | 50,000 | most models kept after selection |
| `mem_lim` | 8000 | memory for the Monte Carlo draws, in MB; `Nmc_prior` is reduced to fit |
| `running_io` | `True` | write each star when it is done, so an interrupted run keeps its results |
| `rstate` | `None` | a `numpy.random.RandomState` for reproducible draws |

A smaller grid (fewer bands, a narrower range of mass or metallicity, or `include_postms=False`) is the most effective way to make fits faster.
