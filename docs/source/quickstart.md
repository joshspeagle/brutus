# Quickstart

This page fits stars from their photometry and parallaxes, then reads the distance, extinction, mass and fit quality from the results. It assumes you have {doc}`installed brutus <installation>` and downloaded the grid and the offsets.

## 1. Load the models

Choose the filters of your photometry, in the order of your data columns. `brutus.data.filters` has lists for common surveys; {ref}`available-filters` lists all 49.

```python
from brutus.core import StarGrid
from brutus.data import load_models, load_offsets
from brutus.data.filters import gaia, ps, tmass

filters = gaia + ps[:5] + tmass  # Gaia G/BP/RP, Pan-STARRS grizy, 2MASS JHKs

models, labels, _ = load_models("grid_mist_v9.h5", filters=filters)
grid = StarGrid(models, labels)
offsets = load_offsets("offsets_mist_v9.txt", filters=filters)
```

The grid holds about 600,000 {term}`stellar models <model grid>`. Each has an {term}`SED` at 1 kpc and its stellar properties ({term}`initial mass`, age, {term}`metallicity`, {term}`EEP`, temperature, surface gravity).

## 2. Prepare the data

brutus fits {term}`flux densities <flux density>`, not magnitudes. Convert them, and mark the bands each star has:

```python
import numpy as np
from brutus.utils import inv_magnitude

# mag, mag_err: (Nstars, Nfilters) arrays; NaN where a band is missing
mask = np.isfinite(mag) & np.isfinite(mag_err)
flux, flux_err = inv_magnitude(np.nan_to_num(mag), np.nan_to_num(mag_err, nan=1.0))
```

brutus drops any band with an error above 0.25 mag (`merr_max`). The models and the data also differ by about 0.02 mag in optical bands and 0.03 mag in near-infrared bands ({doc}`photometric_offsets`). Add this systematic error in quadrature; without it, good fits of bright stars look poor:

```python
sys_mag = np.array([0.02] * 8 + [0.03] * 3)  # Gaia + Pan-STARRS, then 2MASS
flux_err = np.hypot(flux_err, flux * sys_mag * np.log(10) / 2.5)
```

You also need, per star:

| Input | Shape | Units |
|---|---|---|
| `parallax`, `parallax_err` | `(Nstars,)` | mas (optional, but they break the distance-extinction degeneracy) |
| `coords` | `(Nstars, 2)` | Galactic longitude and latitude, degrees |
| `ids` | `(Nstars,)` | an integer or byte-string label for each star (h5py cannot save unicode strings) |

A star needs at least 4 valid bands: the fit has 3 free parameters per model (distance, $A_V$ and $R_V$). With fewer, brutus warns; with none, it skips the star and writes it with `model_idx = -99`. Keep only the stars with enough bands:

```python
keep = (mask & (mag_err < 0.25)).sum(axis=1) >= 4
```

The examples below assume you have applied `keep` to every input.

## 3. Fit

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

For each star, `fit` evaluates every model, fits its distance and extinction, weights it by the {term}`priors <prior>`, and saves 250 draws from the {term}`posterior`. {doc}`scientific_background` explains the steps. To add the 3D dust map as a prior on extinction, download it with `fetch_dustmaps()` and pass `dustfile="bayestar2019_v1.h5"`.

## 4. Read the results

The draws are in the HDF5 file, one row per star:

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

Stellar properties come from the grid: index `labels` with the model of each draw.

Check the fit quality with the p-value of the best fit, not with `chi2 / nbands`:

```python
pvalue = stats.chi2.sf(chi2, np.maximum(nbands - 3, 1))
good = pvalue > 1e-3
```

A p-value below $10^{-6}$ means no model matches the star: it may be a binary, a young star or a bad measurement. {doc}`understanding_results` describes every output and check.

## Next steps

| To | See |
|---|---|
| Work through real data, with plots | {doc}`tutorials/index` |
| Make synthetic photometry for a star or a population | {doc}`stellar_models` |
| Fit a star cluster | {doc}`population_modeling` |
| Map dust along a line of sight | {doc}`Tutorial 7 <tutorials/tutorial_07_dust_mapping>` |
| Change the priors | {doc}`priors` |
