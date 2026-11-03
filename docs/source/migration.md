# Migrating from 1.x to 2.0

Version 2.0 resolves statistical biases across the individual-star, cluster, line-of-sight dust, and photometric-offset pipelines, tightens input validation, and removes deprecated configuration switches. While all public module, class, and function names remain unchanged, v1.x scripts that use removed keyword arguments, attributes, or private imports will now raise `TypeError`, `AttributeError`, or `ImportError`, and invalid inputs raise `ValueError`. Because the statistical fixes improve posterior accuracy, numerical results from v2.0 differ from v1.x and from the benchmark runs in Speagle et al. (2025, arXiv:2503.02227; see [Results That Change](#results-that-change) and the full [changelog](changelog.md)).

## Individual-Star Fitting (`BruteForce`)

| v1.x Usage | v2.0 Migration |
|---|---|
| `fit(..., subsample_mode="topk")` (or any `subsample_mode`) | Remove `subsample_mode`. When more than `max_models` templates survive the selection threshold, `logpost_grid` now subsamples models via systematic sampling with inclusion probability $p_i = \min(1, c\,w_i)$ and weights kept models by $1 / p_i$. |
| `fit(..., ltol_subthresh=...)` | Remove `ltol_subthresh`. Each model now checks its own Gauss-Newton convergence criterion independently. |
| Old optimization defaults `fit(ltol=3e-2, logl_initthresh=5e-3)` | The v2.0 defaults are `ltol=1e-3` (Gauss-Newton log-posterior convergence tolerance) and `logl_initthresh=1e-6` (magnitude-space pre-filter threshold, named `init_thresh` in `loglike_grid`). Passing the v1.x parameter values does not reproduce v1.x outputs because the underlying Gauss-Newton solver was also updated. |
| `max_models=0` to disable model subsampling | Pass `max_models=None` to retain all surviving models; integer values of `max_models` must be strictly positive. |
| Unpacking 8 return values from `logpost_grid` | Unpack 9 return values. The ninth element is `ln_incl`, the log inclusion probability $\ln p_i$ under the `max_models` cap. |
| Calling `fit(...)` on an empirical grid without `agewt` (such as `bayestar_v5`) | Pass `apply_agewt=False` explicitly (`apply_agewt=True` now raises `ValueError` when `agewt` is absent from the grid labels). |
| `fit(..., apply_dlabels=False)` (previously ignored) | Passing `apply_dlabels=False` now evaluates the Galactic structure prior with `labels=None` as documented. |
| `loglike_grid(..., unknown_keyword=...)` | Remove unrecognized keyword arguments (`loglike_grid` and `logpost_grid` now raise `TypeError` on unknown `**kwargs`). |
| Integer `0`/`1` `data_mask` arrays (previously misinterpreted as band index arrays) | Integer `0`/`1` masks are now converted to boolean masks and mask bands identically to `bool` arrays. |
| Passing `parallax` without `parallax_err` (or vice versa) | Pass both `parallax` and `parallax_err`, or omit both; supplying only one raises `ValueError`. |
| Stars with 0 valid bands (previously aborted `fit()` with `ValueError`) | `fit()` now emits a single summary warning, skips stars with 0 valid bands, and writes `model_idx = -99` and `obj_Nbands = 0` with `NaN` draws. |
| Functions returning `-inf` for zero probability density | Zero-density evaluations now return `LOG_ZERO` (`-1e300`, defined in `brutus.utils.math`) consistently across all likelihoods and priors. |

## Priors and 3D Dust Maps (`brutus.priors`, `brutus.dust`)

| v1.x Usage | v2.0 Migration |
|---|---|
| `logp_extinction(..., avlim=None)` | Pass an explicit `(av_min, av_max)` tuple such as `brutus.utils.math.AVLIM`; passing `avlim=None` raises `ValueError`. |
| `logp_extinction` without systematic scatter | `logp_extinction` now adds `scatter=0.2` mag in quadrature to the dust map's statistical uncertainty by default (matching `brutus <= 0.8.3`); pass `scatter=0.0` for zero added scatter. |
| Calling `logp_extinction` on a 3D dust map with `distance=None` | Pass `distance`; omitting `distance` for a 3D dust map raises `ValueError`. |
| `Bayestar.get_query_size(coords)` | Remove the call (`get_query_size` was unused and has been removed). |
| `brutus.dust.__version__` | Use `brutus.__version__`. |
| Calling `logp_galactic_structure` with `labels` of mismatched length | Ensure `labels` has the same shape/length as `dists` (mismatched lengths now raise `ValueError`). |

## Stellar Models and Grids (`brutus.core`)

| v1.x Usage | v2.0 Migration |
|---|---|
| `GridGenerator.make_grid(dist=...)` | Remove `dist`; all pre-computed grids are tabulated at the reference distance of $1\,{\rm kpc}$. |
| `make_grid` with a single-element `av_grid` or `rv_grid` | Pass at least 2 distinct values in both `av_grid` and `rv_grid` so the linear reddening regression is well-posed. |
| Passing unknown keyword arguments (such as `Av=5.0`) to `StarEvolTrack.get_seds`, `StarGrid.get_seds`, or `StarGrid.get_predictions` | Use the documented parameter names; unrecognized keyword arguments now raise `TypeError`. |
| Querying `StarGrid.get_seds` or `get_predictions` outside the grid domain | Queries with `NaN`, infinite, or out-of-range grid labels now raise `ValueError` instead of clamping to the grid boundary. |
| `StellarPop.get_seds(return_dict=False)` returning `params` of shape `(Npred, Neep)` | `params` now has shape `(Neep, Npred)`, matching `params2` and `StarEvolTrack.get_seds`; remove any downstream transpose. |
| `StellarPop.get_seds` falling back to per-star evaluation on `sed_batch` errors | Errors in `sed_batch` or binary companion interpolation now propagate directly instead of silently returning single-star or `NaN` SEDs. |
| Catching `RuntimeError` from `StarEvolTrack` or `Isochrone.get_predictions` | Underlying interpolator and file errors now propagate with their native exception types (`ValueError`, `FileNotFoundError`, `OSError`, `KeyError`). |
| Accessing `EEPTracks.null` or `EEPTracks.binwidths` | Remove references to these unused attributes. |
| `from brutus.core import _get_seds` | Import `get_seds` from `brutus.core` or `_get_seds` from `brutus.core.sed_utils`. |
| Legacy `EEPTracks` pickle caches (`*_cachev2.pkl`) | First initialization automatically generates a new `*_cachev3.pkl` cache; old `*_cachev2.pkl` files can be deleted. |

## Data Loading and Fetching (`brutus.data`)

| v1.x Usage | v2.0 Migration |
|---|---|
| Positional `target_dir` in `fetch_isos("./data")` | Pass `target_dir` as a keyword argument (`fetch_isos(target_dir="./data")`). Across all `fetch_*` functions, the dataset selector is the first positional argument and `target_dir` is keyword-only. |
| `fetch_offsets("./data", "mist_v9")` | Call `fetch_offsets("mist_v9", target_dir="./data")`. |
| Indexing `labels` from `load_models` by integer column position | Index `labels` by field name (`labels["mini"]`). Constant grid axes (`afe`, `smf`) are now retained as single-value columns so `StarGrid` can validate queries against them. |
| `load_models(labels=["mini", "feh"], include_postms=False)` (previously skipped phase cuts when `"eep"` was not in `labels`) | Phase and binary cuts (`include_ms`, `include_postms`, `include_binaries`) are now always applied using the file's `eep` and `smf` columns regardless of which fields are requested in `labels`. |
| `load_models(include_ms=False, include_postms=False)` | Raises `ValueError` immediately before reading the HDF5 file. |

## Photometric Offsets (`brutus.analysis.offsets`)

| v1.x Usage | v2.0 Migration |
|---|---|
| `analysis.photometric_offsets(..., dim_prior=...)` (and `plotting.photometric_offsets`, `photometric_offsets_2d`) | Remove `dim_prior`; leave-one-out importance weights now always use the exact Gaussian likelihood ratio $\ln w = \chi^2_b / 2$. |
| `PhotometricOffsetsConfig(random_seed=seed)` | Pass an explicit generator `rng=np.random.default_rng(seed)` to `photometric_offsets`. |
| `PhotometricOffsetsConfig(use_vectorized_bootstrap=..., validate_inputs=..., progress_interval=...)` | Remove these configuration fields; vectorized bootstrap and input validation are always active, and `verbose` controls progress reporting. |
| Passing an integer index array as `sel` in plotting offset routines | Pass a boolean mask or `0`/`1` indicator array of shape `(Nobj,)`; integer index arrays raise `ValueError`. |

## Plotting (`brutus.plotting`)

| v1.x Usage | v2.0 Migration |
|---|---|
| Passing unrecognized keyword arguments to `hist2d` or `cornerplot(hist2d_kwargs=...)` | Remove unrecognized keywords (`hist2d` now raises `TypeError` on unknown kwargs). |
| `cornerplot(applied_parallax=True)` (old default) | `applied_parallax` now defaults to `None`, which automatically includes the parallax when both `parallax` and `parallax_err` are provided and avoids raising an error when plotting pre-saved `(dists, reds, dreds)` draws without parallax inputs. |
| Default $A_V$ axis range of $[0, 6]\,{\rm mag}$ in `dist_vs_red` and `bin_pdfs_distred` | The default $A_V$ range now matches `AVLIM = (0.0, 20.0)` mag; pass `span` explicitly for a narrower plotting window. |
| Scalar `smooth` defaults in `dist_vs_red` (`0.015`) and `bin_pdfs_distred` (`0.01`) | Default `smooth` tuples are now `(0.015, 0.005)` and `(0.01, 0.005)` so that the $0.1\,{\rm mag}$ kernel width in $A_V$ is preserved over the wider $20\,{\rm mag}$ default range; pass a scalar to use identical fractional smoothing on both axes. |
| Seeding `summary_plot` via global `np.random.seed` | Pass an explicit generator via `summary_plot(..., rstate=rng)`. |
| `dist_vs_red(fig=(fig, ax))` drawing on `plt.gca()` | `dist_vs_red` now draws on the supplied `ax` as documented. |

## Packaging and Runtime Environment

| v1.x Usage | v2.0 Migration |
|---|---|
| Python 3.9 or `numpy < 2.0` | Upgrade to Python $\ge 3.10$ and `numpy >= 2.0` (`astropy >= 6.1` is also declared as a runtime dependency). |
| Building from source with older `setuptools` | Use `setuptools >= 77` (required for SPDX `license = "MIT"` metadata). |
| `import brutus` emitting `ImportWarning` when a dependency is missing | `import brutus` now raises the underlying `ImportError` immediately. |
| Reading `brutus.__version__` from an uninstalled source checkout | Install the package in editable mode (`pip install -e .`); an uninstalled checkout reports `"unknown"`. |

## Results That Change

The [changelog](changelog.md) documents every statistical and numerical fix under `[2.0.0]` with quantitative benchmarks showing how outputs change for identical inputs and random seeds. Entries marked *(output-changing)* in the `Fixed: statistical correctness` section of [CHANGELOG.md](changelog.md) affect individual-star posteriors, cluster and line-of-sight log-likelihoods, empirical photometric offsets, and regenerated plotting draws.
