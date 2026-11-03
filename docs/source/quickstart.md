# Quickstart

Fitting a photometric catalog of stars with `brutus` proceeds in four stages: loading a pre-computed stellar model grid, converting observed magnitudes into linear flux densities with systematic uncertainty floors, running {class}`~brutus.analysis.BruteForce` to evaluate the posterior across the grid, and reading the resulting posterior draws and fit diagnostics from the output HDF5 file. Before running the example below, {doc}`install brutus and download the default grid and photometric offsets <installation>`.

## 1. Load the Stellar Model Grid

Specify the photometric passbands in the exact column order of your input catalog (`brutus.data.filters` provides pre-defined filter lists for common surveys, and {ref}`available-filters` lists all 49 supported passbands):

```python
from brutus.core import StarGrid
from brutus.data import load_models, load_offsets
from brutus.data.filters import gaia, ps, tmass

filters = gaia + ps[:5] + tmass  # Gaia G/BP/RP, Pan-STARRS grizy, 2MASS JHKs

models, labels, _ = load_models("grid_mist_v9.h5", filters=filters)
grid = StarGrid(models, labels)
offsets = load_offsets("offsets_mist_v9.txt", filters=filters)
```

The default `grid_mist_v9.h5` file contains approximately 600,000 {term}`stellar models <model grid>`, storing each model's unreddened {term}`SED` at 1 kpc, linear reddening coefficients, and intrinsic stellar properties ({term}`initial mass`, age, {term}`metallicity`, {term}`EEP`, effective temperature, and surface gravity).

## 2. Prepare the Photometric and Astrometric Inputs

Because flux measurement errors are approximately Gaussian whereas magnitude errors become asymmetric at low signal-to-noise ratios, `brutus` evaluates likelihoods in linear {term}`flux densities <flux density>` (maggies). Convert your observed magnitudes and uncertainties using {func}`~brutus.utils.inv_magnitude` and construct a boolean validity mask:

```python
import numpy as np
from brutus.utils import inv_magnitude

# mag, mag_err: (Nstars, Nfilters) arrays; NaN where a band is missing
mask = np.isfinite(mag) & np.isfinite(mag_err)
flux, flux_err = inv_magnitude(np.nan_to_num(mag), np.nan_to_num(mag_err, nan=1.0))
```

By default, `brutus` masks any band with a magnitude uncertainty exceeding 0.25 mag (`merr_max`). Because synthetic stellar atmospheres and empirical survey calibrations still differ at the level of roughly 0.02 mag in optical bands and 0.03 mag in near-infrared bands ({doc}`photometric_offsets`), add this systematic uncertainty floor in quadrature so that bright stars with tiny photon-counting errors do not receive artificially inflated $\chi^2$ values:

```python
sys_mag = np.array([0.02] * 8 + [0.03] * 3)  # Gaia + Pan-STARRS, then 2MASS
flux_err = np.hypot(flux_err, flux * sys_mag * np.log(10) / 2.5)
```

Each star in the catalog also takes the following auxiliary inputs:

| Input | Shape | Units and Description |
|---|---|---|
| `parallax`, `parallax_err` | `(Nstars,)` | Astrometric parallax and uncertainty in mas (optional; breaks the distance-extinction degeneracy) |
| `coords` | `(Nstars, 2)` | Galactic longitude and latitude $(\ell, b)$ in degrees |
| `ids` | `(Nstars,)` | Integer or byte-string identifier per star (`h5py` does not store fixed-width NumPy Unicode strings) |

A constrained fit requires at least 4 valid photometric bands because each grid model optimizes 3 continuous extrinsic parameters (distance, $A_V$, and $R_V$) and requires at least 1 residual degree of freedom. When a star has between 1 and 3 valid bands, `brutus` issues a warning and proceeds with the fit; when a star has 0 valid bands, `brutus` skips the fit and records `model_idx = -99`. Filter the catalog to retain stars with at least 4 valid bands before fitting:

```python
keep = (mask & (mag_err < 0.25)).sum(axis=1) >= 4
```

The steps below assume `keep` has been applied to all input arrays.

## 3. Run the Grid Fit

Initialize {class}`~brutus.analysis.BruteForce` with the loaded `StarGrid` and run {meth}`~brutus.analysis.BruteForce.fit`:

```python
from brutus.analysis import BruteForce

fitter = BruteForce(grid)
fitter.fit(
    data=flux, data_err=flux_err, data_mask=mask, data_labels=ids,
    phot_offsets=offsets,
    parallax=parallax, parallax_err=parallax_err,
    data_coords=coords,
    save_file="results.h5",
)
```

For each star, `fit` evaluates the likelihood across every model in the grid, optimizes its distance and extinction, integrates the astrophysical {term}`priors <prior>` ({doc}`scientific_background`), and writes 250 equally weighted draws from the joint {term}`posterior` to `results.h5`. To include the Bayestar 3D dust map as a distance-dependent prior on extinction, download the map with `fetch_dustmaps()` and pass `dustfile="bayestar2019_v1.h5"`.

## 4. Inspect Posterior Draws and Diagnostics

The output HDF5 file stores one row per star in the order of the input catalog:

```python
import h5py
from scipy import stats

from brutus.utils import quantile

with h5py.File("results.h5", "r") as f:
    dist = f["samps_dist"][:]       # distance draws, kpc
    av = f["samps_red"][:]          # A_V draws, mag
    idx = f["model_idx"][:]         # grid model of each draw
    chi2 = f["obj_chi2min"][:]      # best-fit chi-square per star
    nbands = f["obj_Nbands"][:]     # bands used, parallax counted as one

lo, med, hi = quantile(dist[0], [0.16, 0.5, 0.84])
print(f"distance = {med:.2f} (+{hi - med:.2f} / -{med - lo:.2f}) kpc")
print(f"A_V = {np.median(av[0]):.2f} mag")
print(f"initial mass = {np.median(labels['mini'][idx[0]]):.2f} Msun")
```

To obtain posterior draws of intrinsic stellar properties (such as initial mass, age, metallicity, or effective temperature), index the grid's `labels` array with `model_idx` for each draw.

Evaluate goodness of fit using the $\chi^2$ survival-function p-value for $N_{\rm bands} - 3$ degrees of freedom rather than the reduced $\chi^2 / N_{\rm bands}$, because each model optimizes 3 free extrinsic parameters:

```python
pvalue = stats.chi2.sf(chi2, np.maximum(nbands - 3, 1))
good = pvalue > 1e-3
```

A p-value below $10^{-6}$ indicates that no single-star template in the grid reproduces the observed fluxes within the stated uncertainties, which typically occurs for unresolved binaries, pre-main-sequence stars, or corrupted photometry ({doc}`understanding_results`).

## Further Reading

| Task | Documentation Guide |
|---|---|
| Work through real survey data and diagnostic plots | {doc}`tutorials/index` |
| Generate synthetic photometry for individual stars or populations | {doc}`stellar_models` |
| Fit coeval star clusters with binaries and field stars | {doc}`population_modeling` |
| Infer 3D dust clouds along a line of sight | {doc}`Tutorial 7 <tutorials/tutorial_07_dust_mapping>` |
| Customize Galactic structure, IMF, and dust priors | {doc}`priors` |
