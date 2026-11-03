# Installation

`brutus` requires Python 3.10 or later on Linux or macOS and is distributed on PyPI under the package name `astro-brutus`:

```bash
pip install astro-brutus
```

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/joshspeagle/brutus.git
```

For editable development installations and testing instructions, see {doc}`contributing`.

Installing `astro-brutus` via `pip` automatically installs all required runtime dependencies (`numpy`, `scipy`, `matplotlib`, `h5py`, `healpy`, `astropy`, `numba`, `pooch`, and `tqdm`).

```{admonition} Windows Support via WSL
:class: warning
Because `healpy` (used by `brutus` to query HEALPix-indexed 3D dust maps) does not provide native Windows wheels, Windows users should install and run `brutus` inside the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).
```

## Model Data Files

`brutus` evaluates stellar models, bolometric-correction neural networks, 3D dust maps, and empirical calibration tables that are distributed separately from the source repository due to their size. The `fetch_*` functions in {mod}`brutus.data` download each file on demand, cache it locally in the user cache directory (or in `$ASTRO_BRUTUS_DATA_DIR` when that environment variable is set), and verify its SHA-256 checksum:

| Function | File | Size | Purpose |
|---|---|---|---|
| `fetch_grids()` | `grid_mist_v9.h5` | 780 MB | Default MIST v1.2 grid for fitting individual stars |
| `fetch_grids(grid="bayestar_v5")` | `grid_bayestar_v5.h5` | 13 MB | Empirical Pan-STARRS stellar grid (optional) |
| `fetch_offsets()` | `offsets_mist_v9.txt` | < 1 kB | Empirical zero-point offsets for `mist_v9` (recommended) |
| `fetch_dustmaps()` | `bayestar2019_v1.h5` | 2.2 GB | Bayestar19 3D dust map for extinction priors (optional) |
| `fetch_isos()` | `MIST_1.2_iso_vvcrit0.0.h5` | 264 MB | MIST v1.2 isochrones for star clusters and populations |
| `fetch_tracks()` | `MIST_1.2_EEPtrk.h5` | 1.1 GB | Continuous MIST v1.2 evolutionary tracks |
| `fetch_nns()` | `nn_c3k.h5` | 0.2 MB | Neural network bolometric corrections for tracks and isochrones |

When {class}`~brutus.core.EEPTracks` loads `MIST_1.2_EEPtrk.h5` for the first time, it writes a pre-built interpolation cache of approximately 0.13 GB alongside the track file. Downloading the complete set of model files and caches requires roughly 4.5 GB of disk space. All data files are also archived in the [astro-brutus collection](https://dataverse.harvard.edu/dataverse/astro-brutus) on the Harvard Dataverse for manual download.

To fit individual stars with {class}`~brutus.analysis.BruteForce`, download the default stellar grid and photometric offsets:

```python
from brutus.data import fetch_grids, fetch_offsets

grid_file = fetch_grids(grid="mist_v9")
offsets_file = fetch_offsets(grid="mist_v9")
```

Each `fetch_*` call creates a symbolic link to the cached file in the current working directory (or in `target_dir` if specified) and returns its path, falling back to a file copy on filesystems that do not support symbolic links.

## Verifying the Installation

Confirm that `brutus` imports cleanly and reports its installed version:

```python
import brutus
from brutus.analysis import BruteForce

print(brutus.__version__)
```

Proceed to {doc}`quickstart` to run an end-to-end stellar fit.
