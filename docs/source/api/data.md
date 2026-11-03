# Data Module (`brutus.data`)

`brutus.data` downloads, verifies, and loads the external model grids, MIST evolutionary tracks, isochrones, neural network weights, 3D dust maps, and photometric calibration offsets used by `brutus`.

**Data Dependencies:**

{doc}`/installation` lists every external data file with its download size and corresponding `fetch_*` function.

**Storage and Lookup Order:**

No file downloads automatically on import or class instantiation. Each `fetch_*` function downloads its file via `pooch` into the `brutus` cache directory (`~/.cache/astro-brutus/` by default, or `$ASTRO_BRUTUS_DATA_DIR` when set), verifies its SHA-256 checksum, and creates a symbolic link (or copy) in `target_dir` (the current working directory by default). Loaders that resolve a default filename (`find_nn_file`, `EEPTracks`, and `Isochrone`) check the `brutus` cache directory first and `data/DATAFILES` in the repository second, raising `FileNotFoundError` with the required `fetch_*` command when the file is absent.

**Typical Usage:**

**First-time setup** (download data files):

```python
from brutus.data import fetch_grids, fetch_isos, fetch_dustmaps

# Download the default MIST v9 model grid (all 49 supported filters)
fetch_grids()

# Download MIST v1.2 isochrones
fetch_isos()

# Download the Bayestar19 3D dust map
fetch_dustmaps(dustmap="bayestar19")
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

```{eval-rst}
.. automodule:: brutus.data.download
   :no-members:
```

```{eval-rst}
.. automodule:: brutus.data.loader
   :no-members:
```
