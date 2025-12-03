# API Ground Truth Reference

This document contains the **verified actual API** for brutus, based on source code inspection.
Use this as the authoritative reference when revising documentation.

**Created**: 2025-12-03
**Source**: Direct inspection of `src/brutus/` implementation

---

## Critical Findings Summary

### What Docs Claim vs Reality

| Docs Claim | Reality |
|------------|---------|
| `BruteForce(grid, use_galactic_prior=..., use_dust_prior=...)` | **WRONG** - No such parameters |
| `fitter.fit(..., n_samples=..., dist_bounds=..., av_max=...)` | **WRONG** - Different parameter names |
| `results['dist_samples']`, `results['av_samples']` | **WRONG** - fit() returns file path, not dict |
| `cornerplot(results, show_titles=True)` | **WRONG** - Signature is `cornerplot(idxs, data, params, ...)` |
| `load_models(..., memmap=True)` | **WRONG** - No memmap parameter |
| `pop.synthesize(...)` | **WRONG** - Method is `get_seds()` |

---

## BruteForce Class

### `__init__` Signature

```python
def __init__(self, star_grid, verbose=True):
```

**Parameters**:
- `star_grid` (StarGrid): Required. Must be a StarGrid instance.
- `verbose` (bool): Whether to print info. Default True.

**NOT AVAILABLE** (contrary to docs):
- ❌ `use_galactic_prior`
- ❌ `use_dust_prior`
- ❌ `use_imf_prior`

### `fit()` Signature

```python
def fit(
    self,
    data,                    # Required: flux densities (Ndata, Nfilt)
    data_err,                # Required: flux errors (Ndata, Nfilt)
    data_mask,               # Required: validity mask (Ndata, Nfilt)
    data_labels,             # Required: object labels (Ndata, Nlabels)
    save_file,               # Required: output HDF5 path
    phot_offsets=None,       # Multiplicative flux offsets
    parallax=None,           # Parallax in mas (Ndata,)
    parallax_err=None,       # Parallax errors
    Nmc_prior=50,            # MC samples for prior integration
    avlim=(0.0, 20.0),       # A_V bounds
    av_gauss=None,           # Gaussian A_V prior (mean, std)
    rvlim=(1.0, 8.0),        # R_V bounds
    rv_gauss=(3.32, 0.18),   # Gaussian R_V prior
    lnprior=None,            # Custom log-prior per model
    wt_thresh=1e-3,          # Weight threshold for model selection
    cdf_thresh=2e-3,         # CDF threshold
    Ndraws=250,              # Number of posterior draws to save
    apply_agewt=True,        # Apply age weighting
    apply_grad=True,         # Apply grid spacing corrections
    lngalprior=None,         # Custom Galactic prior function
    lndustprior=None,        # Custom dust prior function
    dustfile=None,           # Path to 3D dust map
    apply_dlabels=True,      # Pass labels to Galactic prior
    data_coords=None,        # Galactic (l, b) in degrees (Ndata, 2)
    logl_dim_prior=True,     # Dimensional correction to likelihood
    ltol=3e-2,               # Convergence tolerance
    ltol_subthresh=1e-2,     # Sub-threshold tolerance
    logl_initthresh=5e-3,    # Initial likelihood threshold
    mag_max=50.0,            # Max allowed magnitude
    merr_max=0.25,           # Max magnitude error
    rstate=None,             # Random state
    save_dar_draws=True,     # Save distance/A_V/R_V draws
    running_io=True,         # Stream results to disk
    mem_lim=8000.0,          # Memory limit in MB
    verbose=True,            # Print progress
):
```

**Returns**: `str` - Path to output HDF5 file

**NOT AVAILABLE** (contrary to docs):
- ❌ `n_samples` (use `Ndraws`)
- ❌ `dist_bounds` (no direct equivalent)
- ❌ `av_max` (use `avlim`)
- ❌ `rv_bounds` (use `rvlim`)
- ❌ `ftol`
- ❌ `maxiter`

### HDF5 Output Format

The output file contains these datasets:

**Always present**:
- `labels` (Ndata, Nlabels): Object labels
- `model_idx` (Ndata, Ndraws): Resampled model indices (int32)
- `ml_scale` (Ndata, Ndraws): ML scale factors
- `ml_av` (Ndata, Ndraws): ML A_V values
- `ml_rv` (Ndata, Ndraws): ML R_V values
- `ml_cov_sar` (Ndata, Ndraws, 3, 3): Covariance matrices
- `obj_log_post` (Ndata, Ndraws): Log-posteriors per draw
- `obj_log_evid` (Ndata,): Log-evidence per object
- `obj_chi2min` (Ndata,): Minimum chi-squared
- `obj_Nbands` (Ndata,): Number of bands used

**If `save_dar_draws=True`**:
- `samps_dist` (Ndata, Ndraws): Distance draws in **kpc**
- `samps_red` (Ndata, Ndraws): A_V draws
- `samps_dred` (Ndata, Ndraws): R_V draws
- `samps_logp` (Ndata, Ndraws): Log-weights for draws

**NOT in output** (contrary to docs):
- ❌ `dist_samples`, `av_samples`, `rv_samples`
- ❌ `dist_median`, `dist_std`, `dist_16`, `dist_84`
- ❌ `best_fit_idx`, `best_fit_mags`, `chi2_best`, `lnL_max`
- ❌ `converged`

---

## Data Loading Functions

### `load_models()`

```python
def load_models(
    filepath,
    filters=None,
    labels=None,
    include_ms=True,
    include_postms=True,
    include_binaries=False,
    verbose=True,
):
```

**Returns**: `(models, labels, label_mask)` - tuple of 3 arrays

**NOT AVAILABLE**:
- ❌ `memmap` parameter

### `fetch_*()` Functions

All fetch functions exist in `brutus.data.download`:

```python
fetch_isos(target_dir=".", iso="MIST_1.2_vvcrit0.0")
fetch_tracks(target_dir=".", track="MIST_1.2_vvcrit0.0")
fetch_dustmaps(target_dir=".", dustmap="bayestar19")
fetch_grids(target_dir=".", grid="mist_v9")
fetch_offsets(target_dir=".", grid="mist_v9")
fetch_nns(target_dir=".", model="c3k")
```

---

## Core Classes

### `EEPTracks`

```python
class EEPTracks:
    def __init__(
        self,
        mistfile=None,
        predictions=["loga", "logl", "logt", "logg", "feh_surf", "afe_surf"],
        ageweight=True,
        verbose=True,
        use_cache=True,  # ✓ This exists!
    ):
```

**Key method**:
```python
def get_predictions(self, labels):
    # labels: [mini, eep, feh, afe] or array of shape (N, 4)
    # Returns: array of predictions
```

### `Isochrone`

```python
class Isochrone:
    def __init__(
        self,
        mistfile=None,
        predictions=None,  # Default: ["mini", "mass", "logl", "logt", "logr", "logg", "feh_surf", "afe_surf"]
        verbose=True,
    ):
```

**NOT AVAILABLE**:
- ❌ `use_cache` parameter

**Key method**:
```python
def get_predictions(self, feh, afe, loga, eep=None):
    # Returns: array of predictions for the isochrone
```

### `StellarPop`

```python
class StellarPop:
    def __init__(
        self,
        isochrone,      # Required: Isochrone instance
        filters=None,
        nnfile=None,
        verbose=True,
    ):
```

**Key method** (NOT `synthesize()`!):
```python
def get_seds(self, feh, afe, loga, av, rv, dist, ...):
    # Returns: (seds, params, params2) tuple
```

### `StarGrid`

```python
class StarGrid:
    def __init__(self, models, labels, label_mask, filters=None):
```

---

## GridGenerator

```python
class GridGenerator:
    def __init__(
        self,
        tracks,          # EEPTracks or Isochrone instance
        filters=None,
        nnfile=None,
        verbose=True,
    ):
```

### `make_grid()` Method

```python
def make_grid(
    self,
    mini_grid=None,
    eep_grid=None,
    feh_grid=None,
    afe_grid=None,
    smf_grid=None,
    av_grid=None,
    av_wt=None,
    rv_grid=None,
    rv_wt=None,
    dist=1000.0,
    loga_max=10.14,
    eep_binary_max=480.0,
    mini_bound=0.5,
    apply_corr=True,
    corr_params=None,
    output_file=None,
    verbose=True,
):
```

**NOT AVAILABLE** (contrary to docs):
- ❌ `mini_range` (use `mini_grid` array)
- ❌ `eep_range` (use `eep_grid` array)
- ❌ `feh_range` (use `feh_grid` array)
- ❌ `n_mini`, `n_eep` (use explicit arrays)

---

## Population Analysis

### `isochrone_population_loglike()`

```python
def isochrone_population_loglike(
    theta,              # Array [feh, loga, av, rv, dist]
    stellarpop,         # StellarPop instance
    obs_phot,           # (N_objects, N_filters)
    obs_err,            # (N_objects, N_filters)
    parallax=None,
    parallax_err=None,
    cluster_prob=0.95,
    dim_prior=True,
    outlier_model_func=None,
    smf_grid=None,
    eep_grid=None,
    mini_bound=0.08,
    eep_binary_max=480.0,
    return_components=False,
    mask=None,
    **outlier_kwargs,
):
```

**Note**: First argument is `theta` (array), not separate kwargs for each parameter.

### `generate_isochrone_population_grid()`

```python
def generate_isochrone_population_grid(
    stellarpop,
    feh, loga, av, rv, dist,
    smf_grid=None,
    eep_grid=None,
    mini_bound=0.08,
    eep_binary_max=480.0,
    corr_params=None,
):
```

---

## Plotting

### `cornerplot()`

```python
def cornerplot(
    idxs,               # Resampled model indices (Nsamps,)
    data,               # 3-tuple (dists, reds, dreds) or 4-tuple (scales, avs, rvs, covs_sar)
    params,             # Structured array of model parameters
    lndistprior=None,
    coord=None,
    avlim=(0.0, 6.0),
    rvlim=(1.0, 8.0),
    weights=None,
    parallax=None,
    parallax_err=None,
    Nr=500,
    applied_parallax=True,
    pcolor="blue",
    parallax_kwargs=None,
    span=None,
    quantiles=[0.025, 0.5, 0.975],
    color="black",
    smooth=10,
    hist_kwargs=None,
    hist2d_kwargs=None,
    labels=None,
    label_kwargs=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    title_quantiles=[0.025, 0.5, 0.975],
    truths=None,
    truth_color="red",
    truth_kwargs=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    verbose=False,
    fig=None,
    rstate=None,
):
```

**CRITICAL**: This is NOT `cornerplot(results, show_titles=True)`!

---

## Priors Module

All priors are exported from `brutus.priors` directly (not submodules):

```python
from brutus.priors import (
    # Stellar
    logp_imf,
    logp_ps1_luminosity_function,
    # Astrometric
    logp_parallax,
    logp_parallax_scale,
    convert_parallax_to_scale,
    # Galactic
    logp_galactic_structure,
    logn_disk,
    logn_halo,
    logp_feh,           # NOT logp_metallicity!
    logp_age_from_feh,  # NOT logp_age!
    # Extinction
    logp_extinction,
)
```

**NOT AVAILABLE** (contrary to docs):
- ❌ `brutus.priors.stellar.logp_imf` - use `brutus.priors.logp_imf`
- ❌ `brutus.priors.galactic.logp_metallicity` - use `brutus.priors.logp_feh`
- ❌ `brutus.priors.galactic.logp_age` - use `brutus.priors.logp_age_from_feh`
- ❌ `brutus.dust.maps.get_dust_prior` - doesn't exist

---

## Usage Example (Correct)

```python
import h5py
import numpy as np
from brutus.data import load_models, fetch_grids
from brutus.core import StarGrid
from brutus.analysis import BruteForce

# Download data if needed
fetch_grids(target_dir="./data")

# Load models
models, labels, label_mask = load_models("./data/grid_mist_v9.h5")
grid = StarGrid(models, labels, label_mask)

# Create fitter
fitter = BruteForce(grid)

# Fit data - note required parameters!
output_file = fitter.fit(
    data=flux_array,           # (Nstars, Nfilters)
    data_err=flux_err_array,   # (Nstars, Nfilters)
    data_mask=mask_array,      # (Nstars, Nfilters)
    data_labels=label_array,   # (Nstars, Nlabels)
    save_file="results.h5",
    parallax=parallax_array,   # (Nstars,) in mas
    parallax_err=parallax_err_array,
    data_coords=coords_array,  # (Nstars, 2) galactic l,b
    Ndraws=250,
)

# Read results from HDF5
with h5py.File(output_file, "r") as f:
    distances = f["samps_dist"][:]  # (Nstars, Ndraws) in kpc
    av_values = f["samps_red"][:]   # (Nstars, Ndraws)
    model_indices = f["model_idx"][:]
    log_evidence = f["obj_log_evid"][:]

# Compute summary statistics yourself
dist_median = np.median(distances, axis=1)
dist_16, dist_84 = np.percentile(distances, [16, 84], axis=1)
```

---

## Files to Update

Based on this ground truth, the following documentation files need correction:

1. **quickstart.rst** - Fix output format, cornerplot example
2. **understanding_results.rst** - Completely rewrite output section
3. **choosing_options.rst** - Remove fake parameters
4. **priors.rst** - Fix import paths, remove fake BruteForce params
5. **faq.rst** - Fix all code examples
6. **api/analysis.rst** - Fix narrative examples
