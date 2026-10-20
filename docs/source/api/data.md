# Data Module (`brutus.data`)

The data module manages all external data dependencies for brutus, including MIST stellar evolution grids, isochrones, 3-D dust maps, neural network weights, and photometric calibration offsets. It uses `pooch` to download and cache the data files.

**Data Dependencies:**

brutus requires several types of external data:

1. **Model grid**: a precomputed grid of stellar models and their spectral energy distributions (SEDs), `grid_mist_v9.h5` from `fetch_grids` (~780 MB)
2. **MIST EEP tracks**: evolutionary tracks for `EEPTracks`, `MIST_1.2_EEPtrk.h5` from `fetch_tracks` (~1.1 GB). By default, the first load writes a cache of ~0.26 GB (`*_cachev2.pkl`) next to the file.
3. **MIST Isochrones**: tabulated isochrones for population synthesis, from `fetch_isos` (~260 MB)
4. **Dust Maps**: HEALPix 3-D extinction maps (Bayestar19, from `fetch_dustmaps`, ~2.2 GB)
5. **Neural Networks**: trained weights for bolometric corrections, from `fetch_nns` (~250 KB)
6. **Photometric Offsets**: empirical calibration tables, from `fetch_offsets` (\<1 KB)

**Data Management:**

No file downloads on first use. Each `fetch_*` function downloads its file into the brutus data directory and makes a link to it in `target_dir` (default: the current directory). The data directory is `~/.cache/astro-brutus/` (via `pooch.os_cache('astro-brutus')`), or the directory that the `ASTRO_BRUTUS_DATA_DIR` environment variable names. The loaders that find a default file (`find_nn_file`, `EEPTracks` and `Isochrone`) look in the data directory, then in `data/DATAFILES` in the repository. If the file is in neither, they raise `FileNotFoundError` with the `fetch_*` function to run.

**Typical Usage:**

**First-time setup** (download all data files):

```python
from brutus.data import fetch_grids, fetch_isos, fetch_dustmaps

# Download MIST stellar evolution grids
fetch_grids()  # Downloads the default grid ('mist_v9', all supported filters)

# Download MIST isochrones
fetch_isos()

# Download 3-D dust maps
fetch_dustmaps(dustmap='bayestar19')
```

**Loading data for fitting**:

```python
from brutus.data import load_models
from brutus.core import StarGrid

# Load pre-computed grid (third return is the ancillary-label mask)
models, labels, label_mask = load_models('grid_mist_v9.h5')
grid = StarGrid(models, labels)

# Grid is now ready for use with BruteForce fitter
```

**Custom data locations**:

```python
# Load grid from custom path
models, labels, label_mask = load_models('/path/to/my_custom_grid.h5')
```

**Data File Formats:**

- **Grids**: HDF5 with datasets for models, labels, reddening coefficients
- **Isochrones**: HDF5 with structured arrays
- **Dust Maps**: HDF5 files on a HEALPix pixelization (pixel lookups via `healpy`)
- **Neural Networks**: HDF5 files with layer weights

**Storage Requirements:**

- Grid alone: ~0.8 GB
- Grid and isochrones: ~1 GB
- Everything (grid, isochrones, EEP tracks and dust map): ~4.3 GB, plus the 0.26 GB EEP track cache

**See Also:**

- {doc}`/installation` - Setting up data files
- {doc}`/grid_generation` - Creating custom grids
- {doc}`/quickstart` - Basic data loading examples

```{eval-rst}
.. currentmodule:: brutus.data
```

## Data Downloading

```{eval-rst}
.. autofunction:: fetch_grids
```

```{eval-rst}
.. autofunction:: fetch_isos
```

```{eval-rst}
.. autofunction:: fetch_tracks
```

```{eval-rst}
.. autofunction:: fetch_dustmaps
```

```{eval-rst}
.. autofunction:: fetch_offsets
```

```{eval-rst}
.. autofunction:: fetch_nns
```

## Data Loading

```{eval-rst}
.. autofunction:: load_models
```

```{eval-rst}
.. autofunction:: load_offsets
```

```{eval-rst}
.. autofunction:: find_nn_file
```

## Submodules

For advanced users who need access to internal implementations:

```{eval-rst}
.. automodule:: brutus.data.download
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```

```{eval-rst}
.. automodule:: brutus.data.loader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```
