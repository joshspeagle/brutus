# Installation

brutus needs Python 3.10 or later on Linux or macOS. It installs from PyPI as `astro-brutus`:

```bash
pip install astro-brutus
```

To get the latest code from GitHub instead:

```bash
pip install git+https://github.com/joshspeagle/brutus.git
```

To change the code, follow {doc}`contributing`.

pip installs the dependencies: numpy, scipy, matplotlib, h5py, healpy, numba, pooch and tqdm.

```{admonition} Windows
:class: warning
healpy, which brutus uses for the 3D dust maps, does not run on native Windows. Use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and install brutus there.
```

## Model data

brutus fits stars against stellar models that ship separately from the code. Download them once: the `fetch_*` functions in {mod}`brutus.data` store each file in your user cache directory, or in `$ASTRO_BRUTUS_DATA_DIR` if you set it, and check its checksum.

| Function | File | Size | Needed for |
|---|---|---|---|
| `fetch_grids()` | `grid_mist_v9.h5` | 780 MB | Fitting individual stars |
| `fetch_offsets()` | `offsets_mist_v9.txt` | < 1 kB | Photometric calibration (recommended) |
| `fetch_dustmaps()` | `bayestar2019_v1.h5` | 2.2 GB | Dust-map priors on extinction (optional) |
| `fetch_isos()` | `MIST_1.2_iso_vvcrit0.0.h5` | 264 MB | Star clusters and populations |
| `fetch_tracks()` | `MIST_1.2_EEPtrk.h5` | 1.1 GB | Individual stars off the grid |
| `fetch_nns()` | `nn_c3k.h5` | 0.2 MB | Synthetic photometry for the two rows above |

The files are also in the [astro-brutus collection](https://dataverse.harvard.edu/dataverse/astro-brutus) of the Harvard Dataverse, for a manual download.

To fit individual stars, download the grid and the offsets:

```python
from brutus.data import fetch_grids, fetch_offsets

grid_file = fetch_grids(grid="mist_v9")
offsets_file = fetch_offsets(grid="mist_v9")
```

Each function returns the path of a link to the cached file, made in the current directory (or in `target_dir`). Where the file system cannot make links, it copies the file there instead.

## Check the installation

```python
import brutus
from brutus.analysis import BruteForce

print(brutus.__version__)
```

Next, fit a first star with the {doc}`quickstart`.
