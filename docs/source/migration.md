# Migrating from 1.x to 2.0

Version 2.0 removes some arguments, rejects some inputs that 1.x ignored, and changes some results.
No public name was renamed. 1.x code that uses a removed argument, attribute or import fails with a `TypeError`, `AttributeError` or `ImportError`, and a rejected input raises `ValueError`.
Each table maps the 1.x usage to the 2.0 usage.
The [changelog](changelog.md) lists every change.

Results from 2.0 differ from those in the paper (Speagle et al. 2025, arXiv:2503.02227).
See [Results that change](#results-that-change).

## Fitting

| 1.x usage | 2.0 usage |
|---|---|
| `fit(..., subsample_mode="topk")` or any `subsample_mode` | Remove the argument. The `max_models` cap keeps each model with probability `min(1, c * w)` and weights it by the inverse. |
| `fit(..., ltol_subthresh=...)` | Remove the argument. Each model checks its own convergence. |
| `fit(ltol=3e-2, logl_initthresh=5e-3)`, the old defaults | The defaults are `ltol=1e-3` (the convergence tolerance) and `logl_initthresh=1e-6` (the first cut; `init_thresh` in `loglike_grid`). Passing the old values does not restore 1.x results, because the optimizer changed. |
| `max_models=0` to keep every model | Pass `max_models=None`. Other values must be positive integers. |
| `logpost_grid` returns 8 values | It returns 9. The ninth is `ln_incl`, the log-probability that the `max_models` cap kept each model. |
| `fit(..., apply_dlabels=False)` had no effect | It now evaluates the Galactic prior without model labels. |
| `loglike_grid(..., unknown_keyword=...)` | Unknown keywords raise `TypeError`. Fix the spelling. |
| An integer 0/1 `data_mask`, which 1.x read as band indices | Nothing to change: it now masks bands, so the results change. Boolean masks give the same results as before. |
| `parallax` without `parallax_err` | Pass both, or neither. One alone raises `ValueError`. |
| An object with no valid bands stopped `fit` | `fit` skips it and writes `model_idx = -99` and `obj_Nbands = 0`. |
| Zero probability returned `-inf` | It returns `LOG_ZERO` (-1e300) in `brutus.utils.math`. |

## Priors and dust

| 1.x usage | 2.0 usage |
|---|---|
| `logp_extinction(..., avlim=None)` | Pass the A(V) range of the fit, for example `brutus.utils.math.AVLIM`. `None` raises `ValueError`. |
| `logp_extinction` without scatter | It adds `scatter=0.2` mag in quadrature to the map spread. Pass `scatter=0` to remove it. |
| `logp_extinction(dustmap)` with a 3D map and no `distance` | Pass `distance`. It raises `ValueError` otherwise. |
| `Bayestar.get_query_size(coords)` | Remove the call. The method no longer exists. |
| `brutus.dust.__version__` | Use `brutus.__version__`. |
| `logp_galactic_structure` with `labels` of the wrong length | Pass one label per distance. It raises `ValueError`. |

## Stellar models and grids

| 1.x usage | 2.0 usage |
|---|---|
| `GridGenerator.make_grid(dist=...)` | Remove `dist`. Grids are always at 1 kpc. |
| `make_grid` with a one-value `av_grid` or `rv_grid` | Pass at least 2 distinct values for each. |
| `StarEvolTrack.get_seds(..., Av=5.0)`, or any unknown keyword | Use the documented argument names. Unknown keywords raise `TypeError`, also in `StarGrid.get_seds` and `get_predictions`. |
| `StarGrid.get_seds` with a label outside the grid | Pass a label inside the grid. It raises `ValueError`. |
| `StellarPop.get_seds(return_dict=False)` returned `params` of shape (Npred, Neep) | `params` has shape (Neep, Npred), like `params2`. Transpose old code. |
| `StellarPop.get_seds` fell back to per-star SEDs when `sed_batch` failed | The error propagates. |
| `RuntimeError` from `StarEvolTrack` or `Isochrone.get_predictions` | The original error type passes through. Catch that type. |
| `EEPTracks.null`, `EEPTracks.binwidths` | Remove the attributes. They were unused. |
| `from brutus.core import _get_seds` | Use `from brutus.core.sed_utils import _get_seds`. |
| Old `EEPTracks` caches (`*_cachev2.pkl`) | The first load builds a `*_cachev3.pkl` file. Delete the old file. |

## Data

| 1.x usage | 2.0 usage |
|---|---|
| `fetch_isos("./data")` | `fetch_isos(target_dir="./data")`. The file choice comes first. `target_dir` is keyword-only for every `fetch_*` function. |
| `fetch_offsets("./data", "mist_v9")` | `fetch_offsets("mist_v9", target_dir="./data")` |
| `load_models` without an `afe` or `smf` column | The columns are present as one-value columns when the grid is constant in them. Do not index by column position. |
| `load_models(labels=['mini', 'feh'], include_postms=False)` kept every model | The cuts apply whatever `labels` holds. A phase cut on a file without `eep` raises `ValueError`. |
| `load_models(include_ms=False, include_postms=False)` | It raises `ValueError`. |

## Offsets

| 1.x usage | 2.0 usage |
|---|---|
| `analysis.photometric_offsets(..., dim_prior=...)` | Remove the argument. The function always uses Gaussian leave-one-out weights. |
| `plotting.photometric_offsets(..., dim_prior=...)`, also `photometric_offsets_2d` | Remove the argument. |
| `PhotometricOffsetsConfig(random_seed=seed)` | Pass `rng=np.random.default_rng(seed)` to `photometric_offsets`. |
| `PhotometricOffsetsConfig(use_vectorized_bootstrap=..., validate_inputs=..., progress_interval=...)` | Remove the arguments. Input validation always runs, and `verbose` controls progress messages. |
| `sel` as an index array, which 1.x accepted and then excluded the wrong objects | Pass a boolean or 0/1 array of shape `(Nobj,)`. An index array raises `ValueError`. |

## Plotting

| 1.x usage | 2.0 usage |
|---|---|
| `hist2d(..., unknown_keyword=...)` | Remove the keyword. It raises `TypeError`, also for an unknown key in `cornerplot(hist2d_kwargs=...)`. |
| `cornerplot(applied_parallax=True)` | The default is `None`: the parallax applies when `parallax` and `parallax_err` are both given. |
| `dist_vs_red` and `bin_pdfs_distred` with the A(V) range 0 to 6 | The default range is 0 to 20 mag. Pass `span` for a narrower range. |
| `smooth=0.015` in `dist_vs_red`, `smooth=0.01` in `bin_pdfs_distred` | The defaults are `(0.015, 0.005)` and `(0.01, 0.005)`. Pass the old scalar to keep the old smoothing. |
| `summary_plot` with the global numpy random stream | Pass `rstate`. |
| `dist_vs_red` always drew on the current axes | It draws on `ax` from `fig=(fig, ax)` when given. |

## Package

| 1.x usage | 2.0 usage |
|---|---|
| Python 3.9, numpy below 2.0 | Use Python 3.10 or later and numpy 2.0 or later. |
| Building from source with an old `setuptools` | Use `setuptools>=77`. |
| `import brutus` with a missing dependency warned | It raises the original `ImportError`. |
| `brutus.__version__` from a source tree | Install the package. An uninstalled tree reports `"unknown"`. |

## Results that change

The [changelog](changelog.md) lists each change in results for the same inputs and seed. The entries marked *(output-changing)* change fit results, and most of them are in the first "Fixed" section under "Unreleased". Other entries change plotted values and the log-likelihoods of the cluster and line-of-sight fits.
