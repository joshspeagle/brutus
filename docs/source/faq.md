# FAQ

## Inputs

### Do I pass magnitudes or fluxes?

Fluxes. brutus works in {term}`maggies <maggie>`, $F = 10^{-0.4\,m}$, in each survey's native system with no zero-point offset. {func}`~brutus.utils.inv_magnitude` converts magnitudes and their errors:

```python
from brutus.utils import inv_magnitude

flux, flux_err = inv_magnitude(mag, mag_err)
```

### What units do distances and parallaxes use?

| Quantity | Unit |
|---|---|
| parallax (input) | mas |
| distance in `BruteForce` output (`samps_dist`) | kpc |
| distance in `isochrone_population_loglike` (`theta[4]`) | pc |
| $A_V$ | mag |

### How many bands does a star need?

At least 4: each model has 3 free parameters (scale, $A_V$, $R_V$), and the fit needs one more for a degree of freedom. With fewer than 4 valid bands, `fit()` warns and fits anyway. A star with none is not fitted: `fit()` writes it with `model_idx = -99` and NaN values, warns once with the number of such stars, and fits the rest. A band is valid when its mask is true, its flux is positive, its error is positive and finite, and its error is below 0.25 mag (`merr_max`). Select the stars before fitting:

```python
keep = (mask & np.isfinite(mag) & (mag_err < 0.25)).sum(axis=1) >= 4
```

### Why does `fit()` fail to save my star IDs?

h5py cannot store unicode strings. Pass integers or byte strings as `data_labels`, for example `np.array(ids, dtype="S")`.

### Why does `fit()` ask for `data_coords`?

The default Galactic prior and the dust-map prior depend on the position on the sky. Pass `data_coords`, an array of $(\ell, b)$ in degrees with shape `(Nstars, 2)`. Without coordinates, `fit()` applies no Galactic prior at all, and it accepts this only when you pass your own `lngalprior` ({doc}`priors`).

## Results

### My fits look poor, but the stars are normal. Why?

Most often, the errors are too small. The models differ from the data by about 0.02 mag in the optical and 0.03 mag in the near-infrared, and brutus does not add this to your errors. Add it in quadrature ({doc}`photometric_offsets`). Judge a fit by its p-value, not by $\chi^2 / N_{\rm bands}$ ({doc}`understanding_results`).

### Why is the dust prior flat for some stars?

The Bayestar map covers only the sky north of declination −30°. Within that area, `Bayestar` also masks distances outside each sightline's reliable range and pixels whose fit did not converge. Where the map is masked, the prior on $A_V$ is uniform over `avlim`. `Bayestar(apply_reliability_mask=False)` returns the raw map values instead.

### Why are my M dwarfs or massive stars fitted badly?

The default grid covers initial masses from 0.5 to 2.0 M☉ only. Build a grid that covers your stars ({doc}`grid_generation`).

### Why does a binary come out too close?

The default grid has single stars only. An unresolved binary is brighter than a single star of the same colour, so brutus places it closer. With a parallax, the fit is then often poor.

## Running

### Where are the data files?

The `fetch_*` functions in {mod}`brutus.data` download them from the Harvard Dataverse and return their paths. {doc}`installation` lists each file and where it is stored.

### How do I make fits faster?

Load only the bands you fit, and use a smaller grid: a narrower range of mass or metallicity, or `include_postms=False` for main-sequence stars. The time per star grows with the number of models ({doc}`grid_generation`).

### How do I get the same draws again?

Pass a seeded random state: `fit(..., rstate=np.random.RandomState(42))`.
