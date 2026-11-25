# Migrating from 2.x to 3.0

Version 3.0 replaces the Galactic prior, so most of its changes are in `logp_galactic_structure` and the halo density.
2.x code that passes a removed keyword fails with a `TypeError`, and code that imports a removed function fails with an `ImportError`.
Each table maps the 2.x usage to the 3.0 usage.
The [changelog](changelog.md) lists every change.

## Galactic prior

| 2.x usage | 3.0 usage |
|---|---|
| `from brutus.priors import logp_feh, logp_age_from_feh` | Remove the import. `logp_galactic_structure(..., return_components=True)` returns the label term of each component under `'labels'`. |
| `logp_galactic_structure(..., f_thick=0.04)`, the density ratio at the Sun | Pass `f_thick_surface`, the ratio of the surface densities (default 0.18). `galactic_prior_summary` reports the ratio at the Sun that it implies. |
| `Z_thin=...`, `Z_thick=...` | Remove them. The disk heights follow from the vertical Jeans equation. Pass `thick_sigma_z` to change the thick-disk height. |
| `f_halo=0.005`, the halo to thin-disk density ratio at the Sun | Pass `halo_mass` (default 7.5e8 Msun between 2 and 70 kpc). With `rho_thin_solar` it sets that ratio, which `galactic_prior_summary` reports. |
| `eta_halo`, `q_halo_ctr`, `q_halo_inf`, `r_q_halo` | Pass `eta_halo_inner`, `eta_halo_outer`, `r_break_halo` and `q_halo`, which describe a broken power law with one flattening. |
| `logn_halo(R, Z, eta=..., q_ctr=..., q_inf=..., r_q=...)` | Pass `eta_inner`, `eta_outer`, `r_break` and `q`. |
| `feh_thick`, `feh_thick_sigma` | Pass `thick_feh`, the mean, standard deviation and skewness. |
| `feh_halo`, `feh_halo_sigma` | Pass `halo_feh`, a skew-normal core with a metal-poor tail. |
| `feh_thin`, `feh_thin_sigma` | Remove them. The thin-disk [Fe/H] and ages come from its enrichment and radial-migration model, which takes no keyword. |
| `max_age`, `min_age`, `feh_age_ctr`, `feh_age_scale`, `nsigma_from_max_age`, `max_sigma`, `min_sigma` | Pass `thick_age`, `halo_age` and `bar_age`. Ages above 13.8 Gyr get zero prior. `feh_systematic`, `age_systematic` and `age_floor` set how much the label distributions are blurred. |
| `return_components=True` returned `'feh'` and `'age'`, and lists of 3 components | It returns `'number_density'`, `'labels'` when labels are given and `'kinematics'` when `pm` or `rv` is given, each a list of 4 components: thin disk, thick disk, halo and bulge. |
| The defaults `R_solar=8.2`, `Z_solar=0.025` of `logp_galactic_structure`, `logn_disk` and `logn_halo` | The defaults are 8.275 kpc and 0.0208 kpc. Pass the old values to keep the old Sun. |
| The disk scale lengths `R_thin=2.6`, `R_thick=2.0` (`R_scale=2.6` in `logn_disk`) | The defaults are 2.74 and 1.9 kpc, the scale lengths of the surface density. Pass the old values to keep the old disks. |
| A non-finite `feh` or `loga` | Pass finite labels. Non-finite labels raise `ValueError`. |

## Fitting

| 2.x usage | 3.0 usage |
|---|---|
| `fit(..., wt_thresh=1e-3)` or `cdf_thresh`, the old defaults | Remove them. They default to None and raise a `FutureWarning` when given. `max_models` (default 5000) sets how many models a fit keeps. Pass `max_models=None` to keep every model. |
| `fit(...)` with `max_models=50000`, the old default | The default is 5000, sampled with the distance priors. A larger value lowers the Monte Carlo noise of the evidence. |
| The `lnl` of `loglike_grid` was the likelihood | It adds the normalized log-density of the `av_gauss` and `rv_gauss` priors at each model's fit. Subtract it to recover the likelihood. |
| `fit` and `logpost_grid` without `rstate` used the global numpy random state | They use `numpy.random.default_rng()`. Pass `rstate` for reproducible fits. |
| `lndustprior` without `dustfile` had no effect | It applies, with `dustmap=None`. `lndustprior=logp_extinction` without `dustfile` raises `ValueError`. |

## Results that change

Fits with the default Galactic prior change, because each of its components changed and a bar was added.
The model weights now include the Laplace volume of each model's fit, and the model selection includes the distance priors, so the evidence and the posterior change for every star.
The entries marked *(output-changing)* under 3.0.0 in the [changelog](changelog.md) list each change for the same inputs and seed.
