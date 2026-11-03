# Frequently Asked Questions

## Input Preparation

### Do I pass magnitudes or linear fluxes to `BruteForce.fit`?

Pass linear flux densities in {term}`maggies <maggie>`, defined as $F = 10^{-0.4\,m}$ in each survey's native zero-point system. {func}`~brutus.utils.inv_magnitude` converts apparent magnitudes and their uncertainties into maggies:

```python
from brutus.utils import inv_magnitude

flux, flux_err = inv_magnitude(mag, mag_err)
```

### What physical units do distances and parallaxes use?

| Quantity | Physical Unit |
|---|---|
| Input `parallax` and `parallax_err` | milliarcseconds (`mas`) |
| `BruteForce.fit` output distances (`samps_dist`) | kiloparsecs (`kpc`) |
| `isochrone_population_loglike` cluster distance (`theta[4]`) | parsecs (`pc`) |
| Visual extinction $A_V$ (`samps_red`, `theta[2]`) | magnitudes (`mag`) |

### How many valid photometric bands does a star require?

A constrained fit requires at least 4 valid photometric bands because each grid model optimizes 3 continuous extrinsic parameters (distance scale, $A_V$, and $R_V$) and needs at least 1 residual degree of freedom. When a star has between 1 and 3 valid bands, `fit()` issues a warning and fits the star anyway; when a star has 0 valid bands, `fit()` skips it after a single summary warning and records `model_idx = -99` with `NaN` posterior draws. A band is treated as valid when its `data_mask` entry is `True`, its flux and flux uncertainty are positive and finite, and its magnitude uncertainty is below `merr_max` ($0.25\,{\rm mag}$ by default). Filter your catalog to retain stars with at least 4 valid bands before fitting:

```python
keep = (mask & np.isfinite(mag) & (mag_err < 0.25)).sum(axis=1) >= 4
```

### Why does `fit()` raise an error when saving string star IDs?

Because `h5py` does not support fixed-width NumPy Unicode string arrays (`dtype="<U..."`), pass integer identifiers or ASCII byte strings as `data_labels` (for example, `np.asarray(ids, dtype="S")`).

### Why does `fit()` require `data_coords`?

Both the default 3D Galactic structure prior and the Bayestar 3D dust-map prior depend on the sightline direction $(\ell, b)$. Pass Galactic longitude and latitude in degrees as `data_coords` with shape `(Nstars, 2)`. Omitting `data_coords` disables the Galactic structure prior and is permitted only when you explicitly supply a custom `lngalprior` ({doc}`priors`).

## Interpreting Fit Results

### Why do normal stars have poor $\chi^2$ values?

In most cases, the input flux uncertainties include only random photon-counting errors and omit model and calibration systematics. Synthetic stellar models and empirical survey zero points differ at the level of roughly $0.02\,{\rm mag}$ in optical passbands and $0.03\,{\rm mag}$ in near-infrared passbands, and `brutus` does not add this floor automatically. Add the systematic uncertainty in quadrature to `flux_err` before fitting ({doc}`photometric_offsets`), and evaluate goodness of fit using the $\chi^2$ survival p-value for $N_{\rm bands} - 3$ degrees of freedom rather than $\chi^2 / N_{\rm bands}$ ({doc}`understanding_results`).

### Why is the 3D dust prior flat for certain sightlines or distances?

The Bayestar19 dust map covers only the sky north of declination $\delta = -30^\circ$. Within that footprint, {class}`~brutus.dust.Bayestar` also masks distances outside each sightline's reliable distance range as well as HEALPix pixels where the Bayestar fit did not converge. Wherever the map is masked or lacks sky coverage, {func}`~brutus.priors.logp_extinction` evaluates a properly normalized uniform prior over `avlim`. Passing `Bayestar(apply_reliability_mask=False)` disables the distance-range and convergence mask so that raw map values are used wherever sky coverage exists.

### Why are M dwarfs or massive stars fitted poorly by the default grid?

The default `mist_v9` grid spans initial masses from $0.5\,M_\odot$ to $2.0\,M_\odot$. Fitting low-mass M dwarfs ($M_{\rm init} < 0.5\,M_\odot$) or massive stars ($M_{\rm init} > 2.0\,M_\odot$) requires generating a custom grid that covers the appropriate mass range ({doc}`grid_generation`).

### Why is an unresolved binary assigned too small a distance?

Because the default `mist_v9` grid contains only single-star templates, an unresolved binary has higher apparent flux than a single star of the same broadband color, causing photometric fits to place it at a smaller distance (up to a factor of $\sqrt{2}$ closer for an equal-mass binary). When a precise parallax is also included, this tension between the photometric and astrometric distances typically produces a poor $\chi^2$ p-value.

## Runtime and Data Management

### Where are the downloaded model files stored?

The `fetch_*` functions in {mod}`brutus.data` download the HDF5 model grids, isochrones, evolutionary tracks, neural networks, and dust maps from the Harvard Dataverse into your user cache directory (or `$ASTRO_BRUTUS_DATA_DIR` when set) and create a symbolic link in the target directory ({doc}`installation` lists all files and sizes).

### How can I speed up `BruteForce.fit` on large catalogs?

Because runtime per star scales linearly with the number of grid templates and passbands ({doc}`grid_generation`), load only the filters present in your catalog and restrict the grid to the stellar regime of interest (for example, passing `include_postms=False` when fitting main-sequence stars or building a narrower mass/metallicity grid).

### How do I make posterior draws reproducible across runs?

Pass an explicitly seeded NumPy random generator via `fit(..., rstate=np.random.default_rng(42))`.
