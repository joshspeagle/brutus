# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-10-29

Version 2.0.0 collects pull requests `#71` through `#306` merged since v1.2.0
(primarily from the audit and remediation plan in `#99`). This release fixes
statistical biases across individual-star grid fitting, astrophysical priors,
cluster and line-of-sight dust likelihoods, and leave-one-out photometric
offsets, while adding strict input validation across public functions. Because
these statistical fixes alter numerical outputs for identical inputs and seeds,
v2.0.0 results differ from v1.2.0 and from Speagle et al. (2025), as detailed
in the entries marked *(output-changing)* below. Deprecated arguments and
attributes have also been removed, and invalid inputs that v1.2.0 silently
ignored now raise explicit exceptions (see the [migration
guide](https://brutus.readthedocs.io/en/latest/migration.html) for upgrade
instructions).

### Fixed: statistical correctness *(output-changing)*

- **The parallax selection gate no longer removes the true models.**
  `logpost_grid` selects models by a weight that includes the parallax.
  In v1.2.0, that factor was approximated as a Gaussian in the flux scale $s$
  and omitted whenever the parallax signal-to-noise ratio was $4$ or less,
  allowing a dwarf favored by the IMF prior to prune a reddened giant required
  by the parallax. On 120 mock giants at 6-10 kpc with parallax SNR -1.3 to
  3.7, the posterior median was below half the true distance for 15.8% of stars
  (reduced to 2.5% in v2.0.0). The selection weight now multiplies by the
  parallax likelihood marginalized over each model's local Gaussian in $\ln d$
  via five-node adaptive Gauss-Hermite quadrature at all SNR values.
- **`BruteForce` reads the predictions in `StarGrid.params`.** With
  `StarGrid(models, labels, params)`, the fit previously read `agewt`, `feh`,
  and `loga` from `labels` only, dropping the age weights without warning and
  omitting `loga` from the Galactic prior. `BruteForce.models_labels` now
  combines the fields of `labels` and `params` so that both forms of `StarGrid`
  produce identical fits. Passing `apply_agewt=True` on a grid without `agewt`
  (such as the Bayestar grid) now raises `ValueError`; pass
  `apply_agewt=False`.
- **The grid-spacing correction (`apply_grad`) uses only the grid axes.**
  `BruteForce` previously treated every field of `StarGrid.labels` as a grid
  axis. When initialized with `StarGrid(models, labels)` using the combined
  label array from `load_models` (as in Tutorial 5), it erroneously multiplied
  by spacing factors for predicted quantities (`loga`, `logl`, `logt`, `logg`,
  `agewt`). For a mock bulge giant at 8.3 kpc, those extra terms favored a
  dwarf model by 17.6 log-units over the true model, causing 88 of 200 mock
  bulge giant posteriors to miss the true distance. Only intrinsic grid axes
  (`mini`, `eep`, `feh`, `afe`, and `smf`, or `Mr` on a grid without `mini`)
  now receive the spacing correction.
- **The `max_models` cap no longer biases the posterior.** Above `max_models`,
  `logpost_grid` previously drew a weighted subsample of models and retained
  their unadjusted weights, narrowing distance posteriors by up to 31% and
  lowering `obj_log_evid` by up to 0.49. It now retains model $i$ with
  inclusion probability $p_i = \min(1, c\,w_i)$ via systematic sampling,
  returns $\ln p_i$ as `ln_incl`, and divides each kept model's posterior
  weight in `fit` by $p_i$. Fits with fewer than `max_models` surviving models
  are unchanged.
- **`loglike_grid` fits each model to its best (scale, A(V), R(V)).**
  The flux-space $R_V$ derivative previously omitted the factor $A_V$ in
  $\partial F / \partial R_V$, causing refinement to stop at a point that
  over-weighted the $R_V$ prior whenever $A_V > 1$. Meanwhile, the
  magnitude-space solver alternated coordinate-descent steps in $A_V$ and $R_V$
  and stalled when the two parameters were strongly correlated, allowing the
  initial pre-filter to discard the best-fitting model. Both stages now take
  joint Gauss-Newton steps in $(A_V, R_V)$ with the distance scale profiled
  out, clamp active boundary constraints while optimizing the remaining
  parameter, apply Levenberg-Marquardt damping when a full step does not
  improve the objective, and iterate until the predicted log-posterior gain
  falls below `ltol` (default lowered from `3e-2` to `1e-3`). On 24 Orion
  stars, the v1.2.0 best-model `lnl` was up to 9.8 log-units below the optimum.
- **The first cut of `loglike_grid` keeps models within a factor `1e-6` of
  the best** (`init_thresh` and `logl_initthresh`, lowered from `5e-3`).
  Because the initial cut ranks models by their magnitude-space $\chi^2$ rather
  than their flux-space $\chi^2$, the looser `1e-6` threshold retains models
  that gain significant weight during flux-space refinement. On 24 Orion stars,
  the maximum total-variation distance between the resulting model weights and
  an unpruned fit drops from 0.106 to 0.0005 at the same runtime.
- **`fit(apply_dlabels=False)` evaluates the Galactic prior without model
  labels.** `logpost_grid` previously overwrote `dlabels=None` with the grid
  labels, rendering `apply_dlabels=False` a no-op. `logpost_grid` and `_fit`
  now forward `labels=None` to the Galactic prior when `dlabels=None`.
- **Dust-map A(V) prior scatter restored.** `logp_extinction` again adds a
  0.2 mag systematic scatter in quadrature to the dust map's statistical spread
  via the new `scatter=0.2` argument (matching `brutus <= 0.8.3`, after the
  v1.x refactor had dropped the scatter floor). Because Bayestar19 statistical
  uncertainties have a median of ~0.02 mag, omitting the scatter floor pinned
  $A_V$ and distance far more tightly than the map resolution supports. Map
  distance bins with zero reported spread now also evaluate a proper Gaussian
  prior with width `scatter` instead of falling back to a uniform density.
- **Dust-map prior is proper in masked bins.** Where the 3D dust map is
  uninformative (reliability-masked distance bins or sky positions outside the
  survey footprint), `logp_extinction` now returns the normalized uniform
  log-density $-\ln({\rm avlim}[1] - {\rm avlim}[0])$ instead of $\ln p = 0$,
  which previously acted as an unnormalized density of $1\,{\rm mag}^{-1}$ and
  up-weighted masked distance bins relative to reliable ones.
- **`BruteForce` passes `avlim` to the default dust prior**, ensuring that its
  Gaussian truncation and uniform fallback normalization match the fitted $A_V$
  support (custom `lndustprior` callables retain the 4-argument signature).
- **An integer 0/1 `data_mask` masks bands** in `BruteForce.fit` and
  `loglike_grid`. Integer masks were previously interpreted as index arrays,
  causing the fit to evaluate band 0 and repeat band 1 across all masked
  entries. Boolean masks produce the same results as before, and `loglike_grid`
  no longer mutates the caller's `data_mask`, `av_init`, or `rv_init` arrays.
- **`loglike_grid` normalizes every model's `lnl`.** The Gaussian normalization
  constant $-\frac{1}{2}(N_{\rm dim}\ln(2\pi) + \sum \ln \sigma^2)$ was
  previously added only to models that passed the initial cut, leaving culled
  models offset by roughly 90 log-units for fluxes in maggies. Culled models
  now include the normalization constant and retain their magnitude-space
  $(s, A_V, R_V)$ estimates.
- **`logpost_grid` truncates the Monte Carlo draws** of $A_V$, $R_V$, and
  distance at `avlim`, `rvlim`, and $(0.001, 10^6)\,{\rm kpc}$, assigning zero
  weight to out-of-bounds draws. Previously, out-of-bounds draws were clipped
  to the boundary at full weight, placing roughly half of the $A_V$ draws of an
  unreddened model at $A_V = 0$ and inflating its log-posterior weight by up to
  $\ln 2$. This fix updates `samps_red`, `samps_dred`, `samps_dist`,
  `obj_log_post`, `obj_log_evid`, and `mc_ess` (setting `mc_ess = 0` when no
  Monte Carlo draw has positive weight).
- **A `functools.partial` of `logp_extinction` receives the fit's `avlim`**
  just like the bare function unless the `partial` object explicitly binds
  `avlim` itself (previously, `partial(logp_extinction, scatter=0.3)` normalized
  over the default `(0, 20)` range instead of the fit's `avlim`).
- **`BruteForce.fit` skips an object with no valid bands** instead of aborting
  the catalog run with `ValueError`. Skipped objects are recorded with
  `model_idx = -99`, `obj_Nbands = 0`, and `NaN` draw values after a single
  summary warning, and the output HDF5 file is now closed cleanly on all exit
  paths.
- **`quantile` drops samples with zero weight** before constructing the
  empirical CDF. Previously, zero-weight samples retained CDF nodes so that
  `quantile([0, 1, 100], [0.5, 0.9], weights=[1, 1, 0])` returned `[0.5, 60.4]`
  instead of `[0.5, 1.0]`. Negative, non-finite, or all-zero weights now raise
  `ValueError`.
- **An object with no degrees of freedom adds 0 to a cluster fit.** Under
  `dim_prior=True`, `phot_loglike` and `chisquare_outlier_loglike` previously
  returned `-inf` for objects with $N_{\rm dim} - {\rm dof\_reduction} \le 0$
  (such as stars with all bands masked), driving the total cluster
  log-likelihood to `-inf` across all parameters. Such objects now contribute
  `0.0`, matching `dim_prior=False`.
- **`truncnorm_logpdf` and `logp_imf` return `LOG_ZERO` (-1e300)** for
  zero probability density instead of `-inf`, matching the convention across
  the rest of `brutus`. The Numba-fused and NumPy evaluation paths of
  `logp_galactic_structure` now agree for stellar ages above `max_age` (where
  they previously returned `-1e300` and `-inf`), and `truncnorm_logpdf` and
  `truncnorm_pdf` accept NumPy scalars and 0-D arrays so `logp_age_from_feh(5.0)`
  no longer raises `TypeError`. The constant `LOG_ZERO` is exported from
  `brutus.utils.math`.
- **`chisquare_logpdf` returns `LOG_ZERO` (-1e300)** where the density vanishes
  instead of `-inf`. At `x == loc`, it now matches `scipy.stats.chi2.logpdf`
  ($-\ln 2$ for ${\rm df} = 2$ and $+\infty$ for ${\rm df} < 2$), and `float32`
  inputs no longer underflow to `-inf`.
- **The line-of-sight and cluster log-likelihoods return `LOG_ZERO`
  (-1e300)** for zero probability density instead of `-inf` across
  `los_clouds_loglike_samples`, `kernel_tophat`, `kernel_gauss`,
  `kernel_lorentz`, `marginalize_isochrone_grid`,
  `isochrone_population_loglike`, and `uniform_outlier_loglike`. Under
  `dim_prior=True`, `phot_loglike` also returns `LOG_ZERO` for an exact zero
  residual ($\chi^2 = 0$) with 3 or more degrees of freedom.
- **`kernel_gauss` and `kernel_lorentz` normalize over `rlims` stably in
  the tails.** `kernel_gauss` now evaluates the truncation mass with `log_ndtr`
  instead of a direct `ndtr` difference that rounded to zero for a kernel mean
  roughly $8\sigma$ below `rlims` or $38\sigma$ above, and `kernel_lorentz`
  avoids catastrophic cancellation in its `arctan` normalization when the
  kernel center lies far outside `rlims` (reducing the relative error at $10^7$
  widths from $2 \times 10^{-4}$ to machine precision).
- **The fused Galactic prior normalizes the age prior with the tail-stable
  log-mass of the fused extinction prior** in place of a floored `erf`
  difference, changing valid log-prior evaluations by at most $1.8 \times 10^{-15}$
  and leaving `bench/fullfit.py` outputs unchanged.
- **`logp_extinction` normalizes its truncated normal stably in the
  tails.** Previously, evaluating the truncation mass on `avlim` via a floored
  `erf` difference caused dust-map means far above `avlim` (such as $9\,{\rm mag}$
  with `avlim=(0, 6)`) to yield a spurious log-prior near $+631$ instead of
  $\sim +1$. Both the NumPy `log_ndtr` path and the mirrored `erfc` Numba kernel
  now match `scipy.stats.truncnorm.logpdf` across extreme tails.
- **Batched `inverse3(regularize=True)` regularizes identically to the
  single-matrix path**, fixing catastrophic cancellation in its minimum
  eigenvalue check on ill-conditioned $3 \times 3$ matrices without altering the
  saved `bench/fullfit.py` benchmark outputs.
- **`draw_sar` and the plotting functions that regenerate draws default to
  `avlim=(0, 20)`**, matching `BruteForce.fit` instead of clipping at
  `(0, 6)` in `cornerplot`, `dist_vs_red`, and `bin_pdfs_distred` (where stars
  with $A_V > 6\,{\rm mag}$ previously had all regenerated draws replaced by the
  mean). When `span` is omitted, the default $A_V$ axis in `dist_vs_red` and
  `bin_pdfs_distred` spans $0$ to $20\,{\rm mag}$ (including for pre-saved draws,
  making the default bins $3.3$ times wider; pass `span` for a narrower
  window), and `draw_sar` pads with the mean after 100 consecutive empty
  rejection-sampling passes rather than waiting for `max_attempts`.
- **`analysis.photometric_offsets` weights the draws of a fitted band by
  the Gaussian leave-one-out ratio** $\ln w = \chi^2_b / 2$, replacing the
  previous $\chi^2$-density ratio (`dim_prior=True`) that did not match the
  Gaussian likelihood sampled by `BruteForce`. In synthetic injection tests
  with a 10% offset in one fitted band (400 stars, 1,000 draws, 8 seeds), the
  mean recovery bias drops from 0.0026 to 0.0008.
- **`dist_vs_red` and `bin_pdfs_distred` smooth A(V) by 0.005 of its range
  by default** ($0.1\,{\rm mag}$ over the default $20\,{\rm mag}$ `avlim`),
  setting default `smooth` tuples of `(0.015, 0.005)` and `(0.01, 0.005)`.
- **`load_models` applies the main-sequence and binary cuts regardless of what
  `labels` contains.** Previously, passing a `labels` list that omitted `"eep"`
  or `"smf"` bypassed the phase or binary filter (for example,
  `load_models("grid_mist_v9.h5", labels=["mini", "feh"], include_postms=False)`
  returned all 613,530 models instead of the 143,856 main-sequence models).
  Requesting a phase cut on an HDF5 file that lacks an `"eep"` column now
  raises `ValueError`.
- **`StellarPop.get_seds` finds each companion's EEP on the isochrone's
  full native EEP grid.** Previously, inverting $M_{\rm init}({\rm EEP})$ on the
  caller's `eep` array dropped companions lighter than the caller's minimum mass
  and misinterpolated coarser `eep` slices (and never added companions for a
  single-element `eep`). At ${\rm [Fe/H]} = 0$, $\log_{10}({\rm age}) = 9.5$,
  and `binary_fraction=0.6`, the SDSS $g$ magnitude at ${\rm EEP} = 350$ is now
  14.621 for any `eep` array (compared to 14.645 with
  `eep=np.arange(300, 480, 10)`). With the default cluster grids,
  `isochrone_population_loglike` shifts by about 0.01 on a synthetic cluster of
  30 stars, and companions that fall on the pre-main sequence on young
  isochrones are now included.
- **`StellarPop.get_seds` returns NaN `params2` for every star whose SED
  has no companion** across all binary fractions (previously,
  `0 < binary_fraction < 1` populated `params2` even for companions below
  `mini_bound` or without valid SEDs whose flux was omitted from the combined
  SED).
- **`los_clouds_loglike_samples` averages each star's likelihood over its
  finite draws only.** Dividing by the total draw count previously penalized
  stars containing `NaN` draws relative to the outlier model, and a star with
  all-`NaN` draws added $\ln p_b - \ln({\rm area})$, biasing the outlier
  fraction $p_b$ upward. Non-finite draws are now excluded from the per-star
  average, finite distances outside $[0, 10^{10})$ continue to contribute zero
  inlier likelihood, and stars with zero finite draws are omitted from the sum
  (for 5 stars with 25 draws each and one all-`NaN` star, the log-likelihoods
  at $p_b = 0.01, 0.1, 0.5$ change from $-8.25, -6.22, -6.14$ to the four-star
  values $-1.85, -2.13, -3.66$).
- **`StarEvolTrack` finds a companion's EEP at the primary's [α/Fe]**
  without clamping to the track library boundary, matching the `[α/Fe]` value
  already passed for the companion's stellar parameters (tracks without a
  `gridpoints` EEP axis previously used ${\rm [\alpha/Fe]} = 0.0$). Results at
  ${\rm [\alpha/Fe]} = 0$ on `MIST_1.2_EEPtrk.h5` are unchanged, and for
  $0 < |{\rm [\alpha/Fe]}| < 10^{-5}$ the companion EEP shifts by at most
  $9 \times 10^{-13}$ across 200 random binaries evaluated at $\pm 3 \times 10^{-6}$.
- **`logp_galactic_structure` computes in float64 for float32 `labels`.**
  Evaluating the metallicity and age priors in the native `float32` dtype of
  `labels` previously caused `float32` evaluations on the NumPy path ($\le 1,000$
  distances) and on the fused path under `NUMBA_DISABLE_JIT=1`. Both paths now
  cast `feh` and `loga` to `float64` (changing log-priors across 500 random
  points by at most $2.0 \times 10^{-6}$, or $2.7 \times 10^{-7}$ relative), and
  passing a multi-dimensional `labels` array whose shape differs from `dists`
  now raises `ValueError`.
- **`StarGrid.get_seds` and `get_predictions` reject queries they cannot
  answer.** Passing a `NaN`, infinite, or out-of-bounds grid label now raises
  `ValueError` naming the offending label instead of silently clamping to the
  grid boundary (where `mini=50` or `mini=NaN` on `grid_mist_v9.h5` previously
  returned the `mini=2.0` model). Grid labels within a relative tolerance of
  $10^{-6}$ of an axis boundary remain accepted, and omitting a multi-valued
  axis under `use_multilinear=False` now raises `ValueError` just as under
  `use_multilinear=True`.

### Changed

- **`load_models` keeps constant grid labels.** Single-value grid axes (`afe` on
  a solar-scaled grid and `smf` when `include_binaries=False`) are now retained
  as constant columns in `labels` so `StarGrid` can validate queries against
  them. `BruteForce` skips grid-spacing factors for single-value axes, leaving
  fit outputs unchanged.
- **`hist2d` no longer accepts `**kwargs`.** Unrecognized keyword arguments
  passed to `hist2d` or via `cornerplot(hist2d_kwargs=...)` now raise
  `TypeError` instead of being silently ignored, and `cornerplot` removes the
  unused default `alpha` key from `hist2d_kwargs`.
- `StarEvolTrack.get_seds` and `Isochrone.get_predictions` propagate underlying
  track, interpolator, and neural-network exceptions with their native types
  instead of wrapping them in `RuntimeError` or converting companion errors into
  `RuntimeWarning` with `NaN` SEDs. `EEPTracks` still rebuilds unreadable pickle
  caches and skips cache writes only on `OSError`.
- **`GridGenerator.make_grid` has no `dist` argument.** Grids are always
  tabulated at $1\,{\rm kpc}$ to match the reference distance assumed by
  `load_models`, `StarGrid`, and `BruteForce` (passing `dist` now raises
  `TypeError`).
- `GridGenerator.make_grid` raises `ValueError` naming the offending parameter
  when `av_grid` or `rv_grid` contains non-finite values or fewer than 2
  distinct points, or when `av_wt` or `rv_wt` contains non-finite values or has
  a mismatched shape (Tutorial 3 now passes 4 $R_V$ grid values).
- **`EEPTracks` drops the unused attributes `null` and `binwidths`.** Its
  `ygrid` attribute stores the unpadded prediction grid (with a single
  `[α/Fe]` slice when the track file has only one `[α/Fe]` value), while
  `xgrid` and `grid_dims` continue to describe the padded interpolator grid.
- **`StarEvolTrack.get_seds`, `StarGrid.get_seds`, and
  `StarGrid.get_predictions` no longer accept `**kwargs`**, raising `TypeError`
  on misspelled keyword arguments such as `Av=5.0` or `fe_h=-1`.
- **`subsample_mode` is removed** from `fit`, `_fit`, and `logpost_grid` along
  with its biased `"topk"` mode, and `logpost_grid` returns `ln_incl` as its
  ninth return value.
- **`max_models` must be `None` or a positive integer** in `fit` and
  `logpost_grid` (passing `0` or a negative integer now raises `ValueError`).
- `load_models` raises `ValueError` when both `include_ms=False` and
  `include_postms=False` are passed, and emits a `UserWarning` listing any
  requested `labels` absent from the HDF5 file. `load_offsets` warns when a
  requested filter is absent from both `brutus.FILTERS` and the offset file, and
  formats its verbose summary as signed percentages (`+12.34%`).
- **Every `fetch_*` function takes the dataset selector first and `target_dir`
  as a keyword-only argument** (for example,
  `fetch_offsets("mist_v9", target_dir="./data")`), raising `ValueError` with
  the allowed dataset names when given an unknown selector.
- `Bayestar.get_query_size` is removed because it was unused and returned twice
  the actual query length for `[l, b]` coordinate arrays.
- **`ltol_subthresh` is removed** from `loglike_grid`, `fit`, and `_fit`, and
  non-positive `ltol` values raise `ValueError`.
- **`dim_prior` is removed** from `analysis.photometric_offsets`,
  `plotting.photometric_offsets`, and `plotting.photometric_offsets_2d` (which
  always use Gaussian leave-one-out weights), and non-positive or non-finite
  `old_offsets` values raise `ValueError`.
- **`PhotometricOffsetsConfig` removes four deprecated fields**
  (`use_vectorized_bootstrap`, `validate_inputs`, `random_seed`, and
  `progress_interval`). Pass `rng=np.random.default_rng(seed)` directly to
  `photometric_offsets` for reproducible bootstrap resampling, and use `verbose`
  to control progress output.
- `logpost_grid` accepts `return_precision=True` to return the $3 \times 3$
  Fisher precision matrices of retained models so that `fit` inverts matrices
  only for the final resampled draws.
- Default extinction bounds are centralized in `brutus.utils.math.AVLIM`
  `(0.0, 20.0)` and `RVLIM` `(1.0, 8.0)` across all fitting, prior, and plotting
  functions.
- `import brutus` raises the underlying `ImportError` immediately when a
  required dependency is missing instead of emitting `ImportWarning`.
- `brutus.__version__` is read from `importlib.metadata` so that the
  `brutus_version` attribute of generated grids tracks the installed release
  (reporting `"unknown"` only when imported from an uninstalled source tree).
- Top-level `brutus` re-exports all data fetchers and loaders (`fetch_nns`,
  `fetch_offsets`, `fetch_tracks`, `load_offsets`, `fetch_dustmaps`,
  `fetch_grids`, `fetch_isos`, and `load_models`).
- `brutus.core` no longer exports the private helper `_get_seds` (importable
  from `brutus.core.sed_utils`), `brutus.dust.__version__` is removed, and
  `StarGrid` is included in `brutus.core.individual.__all__`.
- `logp_extinction` requires a finite `(av_min, av_max)` tuple for `avlim` and
  raises `ValueError` when `avlim=None`.
- `brutus.utils` re-exports `chisquare_outlier_loglike` and
  `uniform_outlier_loglike`, and removes the unused `_function_wrapper` from
  `brutus.utils.math`.
- `pyproject.toml` declares `license = "MIT"` (SPDX expression, packaging
  `LICENSE` as the license file while keeping `AUTHORS.md` in the source
  distribution) with `setuptools >= 77`, `requires-python = ">=3.10"` (up from
  `3.9`), `numpy >= 2.0` (up from `1.19`), and `astropy >= 6.1`.
- `BruteForce.loglike_grid` and `logpost_grid` reject unrecognized `**kwargs`
  with `TypeError`.
- `StellarPop.get_seds(return_dict=False)` and `get_seds_smf_grid` return
  `params` with shape `(Neep, Npred)` matching `params2` and
  `StarEvolTrack.get_seds` (replacing the transposed `(Npred, Neep)` layout).
- `StellarPop` and `StarEvolTrack` propagate neural-network initialization
  errors at construction, `FastNNPredictor` raises a `ValueError` naming any
  filters missing from `nnfile`, `StellarPop` validates that
  `isochrone.predictions` includes `mini`, `logl`, `logt`, `logg`, `feh_surf`,
  and `afe_surf`, and `StarEvolTrack` validates that `tracks.predictions`
  includes `loga`, `logl`, `logt`, `logg`, `feh_surf`, and `afe_surf`.
- `StellarPop.get_seds` evaluates SEDs exclusively via
  `FastNNPredictor.sed_batch` and propagates batch or companion evaluation
  errors directly.
- `Isochrone.get_predictions` and `EEPTracks.get_corrections` share a single
  empirical correction implementation, `Isochrone` validates `corr_params`
  (length 4) and required track predictions (`mini`, `logt`, `logl`, `logg`) at
  construction, and `StellarPop` propagates exceptions from
  `Isochrone.get_predictions` with their original exception types instead of
  wrapping them in `RuntimeError`.
- `cornerplot` defaults `applied_parallax` to `None`, applying the parallax
  automatically when both `parallax` and `parallax_err` are provided and
  avoiding errors when plotting pre-saved `(dists, reds, dreds)` draws without
  parallax inputs (passing `applied_parallax=True` without both parallax inputs
  now raises only for `(scales, avs, rvs, covs_sar)` inputs).
- `dist_vs_red` honors the axes passed via `fig=(fig, ax)`, and `summary_plot`
  accepts `rstate` and forwards it to `cornerplot` and `posterior_predictive`.

### Fixed

- Numba kernel caches are invalidated automatically whenever any Python file in
  `brutus` changes, preventing cached `@njit(cache=True)` kernels from returning
  stale results after a callee in another module is modified.
- `StarGrid.get_seds(afe=0.4)` on `grid_mist_v9.h5` now raises `ValueError`
  because `load_models` retains constant grid axes (`afe`, `smf`) and `StarGrid`
  validates queries against them (unrecognized label keywords emit a warning
  and are ignored).
- `Isochrone` chains the underlying `OSError` or `KeyError` as `__cause__` when
  raising `RuntimeError` on a failed HDF5 load.
- `EEPTracks` writes its pickle cache atomically via a temporary file and
  `os.replace`, preventing interrupted writes from leaving corrupted
  `*_cachev3.pkl` files.
- `StarGrid` constructs its grid axes and KD-tree from finite label values only
  and raises `ValueError` if any grid axis contains no finite values.
- `logpost_grid` raises `ValueError` when only one of `parallax` or
  `parallax_err` is supplied (matching `loglike_grid` and `fit`), and raises a
  descriptive `ValueError` naming `dlabels` when an unstructured array is
  passed to the default Galactic prior.
- `StarGrid` preserves the caller's `filters` ordering for structured `models`
  arrays (matching `load_models`), warns on skipped unknown filter names,
  raises `ValueError` when no requested filter exists, and validates the length
  of `filters` for unstructured `models` arrays.
- `StarGrid` suppresses false-positive warnings for missing multilinear
  interpolation hypercube corners whose interpolation weight is zero (on
  `grid_mist_v9.h5`, 96 of 3,000 queries on grid nodes previously emitted
  warnings without changing the returned SEDs).
- `StarGrid.get_seds(use_multilinear=False)` now supports single-model grids,
  raises `ValueError` on grids with no valid grid axes under both interpolation
  modes, and raises `ValueError` for non-positive or non-finite `dist`.
- `find_nn_file`, `EEPTracks`, and `Isochrone` search the `brutus` download
  directory (`strato.path`, configured by `$ASTRO_BRUTUS_DATA_DIR`) before
  `data/DATAFILES/` and raise a descriptive `FileNotFoundError` naming the
  searched paths and the corresponding `fetch_*()` helper when a data file is
  missing.
- `EEPTracks` validates `predictions` at construction and raises `ValueError`
  when required fields (`loga`, `logl`, `logt`, `logg`) are missing or unknown
  names are present.
- `lb2pix` returns `-1` for `NaN` or infinite Galactic longitudes just as for
  invalid latitudes, preventing `Bayestar.query` from mapping non-finite
  longitudes to an arbitrary HEALPix pixel.
- `Bayestar.query` validates that array inputs have shape `(2,)` or
  `(Ncoords, 2)` (previously, a `(2, 3)` array of `[l, b]` values was read row
  by row and an `(N, 3)` array silently dropped its third column), and
  `query_gal` with scalar `ell` and `b` returns 1D profiles of shape
  `(n_dist,)` matching `query_equ` instead of `(1, n_dist)`.
- `logp_extinction` raises `ValueError` when called on a 3D dust map without
  `distance`, and raises `TypeError` if `dustmap.query` returns anything other
  than a 2- or 3-element tuple, list, or array.
- `logp_galactic_structure` validates that `labels` has the same length and
  shape as `dists`, supports scalar `dists` with `labels` or `feh`, and removes
  the broad exception-catching fallback in the Numba-fused Galactic and
  extinction prior kernels.
- `logp_imf` validates that `mass_min` and `mass_break` are finite and positive,
  `mass_max > mass_min`, and `alpha1` and `alpha2` are finite.
- `inverse3` casts input matrices to floating point and validates that `A` has
  shape `(3, 3)` or `(N, 3, 3)`.
- `sample_multivariate_normal` casts `mean` and `cov` to floating point and
  validates that `cov` has shape `(Ndist, dim, dim)`.
- `core.sed_utils.get_seds` validates that `mag_coeffs` has shape
  `(Nmodels, Nbands, 3)` before invoking the Numba kernel.
- `plotting.posterior_predictive` converts integer `0`/`1` `data_mask` arrays
  to boolean masks and validates mask shapes.
- `BruteForce.fit` and `loglike_grid` validate all input array shapes before
  creating the output HDF5 file, require `logl_initthresh` (`init_thresh`) to
  lie in `(0, 1)`, and treat non-positive `parallax_err` values as unmeasured
  parallaxes.
- `BruteForce.logpost_grid` validates that `lnprior` and `dlabels` match the
  length of `results` (preventing mismatched model labels after
  `loglike_grid(indices=...)`).
- `BruteForce.get_sed_grid` raises `ValueError` for an `av` or `rv` array
  whose length is not the number of models. With JIT on, it read past the
  end of the array and returned garbage SEDs.
- `isochrone_population_loglike` checks its inputs before it builds the
  model grid and raises `ValueError` that names the argument: `obs_phot`,
  `obs_err` and `mask` must be 2-D with one shape, `parallax` and
  `parallax_err` must come together with shape (N_objects,), and
  `smf_grid` and `eep_grid` must be 1-D, finite and strictly increasing,
  with `smf_grid` in [0, 1]. These inputs returned `LOG_ZERO` with a warning
  for every `theta`. Only a `theta` with no valid model in the isochrone
  grid still returns `LOG_ZERO`; any other error, such as a misspelled
  outlier keyword or a `ValueError` from the model code, now raises.
  `generate_isochrone_population_grid` checks the grids and no longer
  turns a failed SED slice into a skipped slice. `compute_isochrone_cluster_loglike` raises `ValueError`
  for a `parallax` without `parallax_err` or `distance`, which it dropped
  without a warning.
- `StellarPop.get_seds` raises `ValueError` for a `binary_fraction` that
  is not finite or lies outside [0, 1], and `get_seds_smf_grid` does the
  same for `smf_values`. Such values gave single-star SEDs. An integer or
  scalar `eep` now works with binaries; it raised `ValueError` when a
  companion had no valid EEP.
- `los_clouds_priortransform` raises `ValueError` unless `u` has
  4 + 2 * n entries (n >= 1), all finite. A short `u` raised an
  `IndexError`, and a NaN entry gave NaN parameters.
  `los_clouds_loglike_samples` raises `ValueError` unless `Ndraws` is an
  integer >= 1 and `s0` and `s` are positive. `Ndraws=-1` silently dropped
  the last draw, `Ndraws=0` returned NaN, and `s=0` failed inside the
  kernel with a message that did not name the argument.
- `analysis.photometric_offsets` checks `sel`, `weights`, `prior_mean` and
  `prior_std` before it generates the model SEDs, and raises `ValueError`
  that names the argument. `sel` must be a bool or 0/1 array of shape `(Nobj,)`:
  an index array excluded objects with no error. `weights` must be
  finite and >= 0, and `prior_std` finite and > 0. A scalar prior failed
  only after the full computation.
- `plotting.photometric_offsets` and `photometric_offsets_2d` weight the
  draws of each band with the leave-one-out weights of
  `analysis.photometric_offsets`, computed in flux space. They weighted by
  the likelihood of the other bands alone, which keeps part of the pull of
  the band on the draws. With a 10% offset injected into a fitted band, the
  plotted offset differed from the analysis offset by 0.013 mag; it now
  agrees to 0.0001 mag.
- `dist_vs_red` and `bin_pdfs_distred` accept a float parallax and
  parallax error for several objects and apply them to every object, and
  raise `ValueError` that names the argument for any shape other than
  `(Nobj,)`. A float raised `TypeError`, and a short array left the last
  objects without parallax smoothing. `bin_pdfs_distred` accepts a tuple
  `span` for every `dist_type`, and raises `ValueError` unless `span` has
  shape `(2, 2)`; `dist_type='scale'` and `'parallax'` raised `TypeError`.
- `plotting.photometric_offsets` leaves the panel of a band that no object
  qualifies for blank and draws the other bands; it raised `IndexError`. It
  raises `ValueError` that names `bins` unless `bins` is an int or one int
  per band; a length-2 `bins` raised `TypeError`. `photometric_offsets_2d`
  reads a `bins` of length `Nfilt` as one entry per band, also with 2 bands.
  Both raise `ValueError` that names `weights` unless it has shape `(Nobj,)`
  or `(Nobj, Nsamps)`; a `(Nsamps,)` array was read as per-object weights
  when `Nsamps == Nobj`. The titles of `photometric_offsets_2d` give the
  offset in fixed point (`+12.0%`, not `1.2e+01%`).
- `plotting.photometric_offsets_2d` draws each map over its bin edges. The
  image extent was the outer bin centres, half a bin too narrow on each
  side. `xspan` and `yspan` now set the range of the bins, as in
  `photometric_offsets`; they relabelled the axes of bins taken over the
  data range. The last bin includes its right edge, as in `np.histogram`;
  the object with the largest x or y was left out of every bin.
- `posterior_predictive`, `photometric_offsets` and
  `photometric_offsets_2d` lay out only a figure that they create. With
  `fig=(fig, ax)`, they called `plt.tight_layout()`, which re-laid out the
  current figure, not the one passed.
- `hist2d` and `dist_vs_red` no longer write their defaults into the
  keyword dicts the caller passes (`data_kwargs`, `contourf_kwargs`,
  `contour_kwargs`, `truth_kwargs`). A `contour_kwargs` reused across
  panels kept the colour of the first call.
- `cornerplot` and `bin_pdfs_distred` choose the input format by the
  length of `data`, and raise `ValueError` that names `data` unless it has
  3 or 4 arrays. `cornerplot` accepts saved draws as lists; it sent them to
  the `(scales, avs, rvs, covs_sar)` path, which failed with an unpacking
  error. With `(scales, avs, rvs, covs_sar)` input, `bin_pdfs_distred`
  gives zero weight to a sample when the priors give zero weight to every
  draw of it; the binned PDF of its object was NaN.

### Performance

- `photometric_offsets_2d` groups the objects of each band into the bins
  once, instead of scanning every object for each bin. The maps are bitwise
  identical. With 3000 objects, 5 bands, 20 draws and 50 x 50 bins, a call
  takes a median 0.54 s instead of 0.71 s (3 alternated runs, 4 CPUs).
- `StarGrid.get_seds` and `get_predictions` with `use_multilinear=False`
  keep the scaling of the grid axes with the KD-tree instead of scanning
  every model per query. Results are bitwise identical. On
  `grid_mist_v9.h5` with 5 bands, a KD-tree `get_seds` call takes a median
  0.11 ms instead of 35 ms, and a multilinear call 0.14 ms instead of
  0.19 ms (5 interleaved runs, 4 CPUs).
- `GridGenerator.make_grid` computes each model without a companion once
  per (mini, eep, feh, afe) and copies it to the other `smf_grid` values.
  The saved values are bitwise identical. A 960-model grid with 6 `smf`
  values takes a median 0.78 s instead of 1.29 s (3 interleaved runs).
- `EEPTracks` pads a single [α/Fe] value with a broadcast view instead of a
  copy, and its cache stores the unpadded grid. On `MIST_1.2_EEPtrk.h5`, the
  cache falls from 255 MB to 128 MB, the peak memory of a build from 945 MB
  to 750 MB and of a cached load from 459 MB to 337 MB. Predictions are
  bitwise identical. The cache file is now `*_cachev3.pkl`; the first load
  rebuilds it.
- `EEPTracks` computes the age weights with one numpy path (a stable
  `np.lexsort` by track and EEP, then `np.gradient` per track) instead of
  a pandas groupby with a slower loop when pandas is absent. The weights
  are bitwise identical on `MIST_1.2_EEPtrk.h5`, and computing them takes
  a median 0.72 s instead of 1.08 s with pandas (5 interleaved runs).
- The bootstrap of `analysis.photometric_offsets` draws each sample by a
  per-row binary search of the weight CDF, in a jitted helper, instead of
  comparing every CDF entry. The medians are bitwise identical. With 20,000
  objects, 500 draws and 30 replicates, a bootstrap takes 0.33 s instead of
  1.66 s (medians of 7 interleaved runs, 4 CPUs).
- `marginalize_isochrone_grid` makes two fewer full-size temporaries, and
  `compute_isochrone_cluster_loglike` adds the parallax term only when one
  exists. Results are bitwise identical. With 21000 models and 1000 stars,
  the marginalization takes a median 0.59 to 0.67 s against 0.70 s on one
  core (25 interleaved runs, two repeats).
- `cornerplot` resamples the regenerated draws with one batch of uniform
  deviates and a search of each sample's weight CDF, instead of one
  `rstate.choice` call per sample. The draws are bitwise identical for a
  seeded `RandomState` or `Generator`. For 2000 samples and 500 draws, the
  resampling takes 12 ms instead of 35 ms (medians of 7 interleaved runs,
  4 CPUs).

### Documentation

- New page "Migrating from 1.x to 2.0" with the old and new usage of each
  changed argument.
- The docstring examples of `core.individual`, `core.neural_nets` and
  `core.grid_generation` run on the shipped data: binary examples use
  [α/Fe] = 0 (the MIST tracks have no other value) and filter names are
  real ones such as `PS_g` and `Bessell_V`. `FastNNPredictor.sed` gives the
  input ranges of `nn_c3k.h5` (R(V) in [2, 5], [α/Fe] in [-0.2, 0.6]) and
  says that magnitudes are in each survey's native system, not AB.
  `_get_seds` gives the meaning and units of `rvecs` and `drvecs`.
- The `GridGenerator` docstrings give the default grid size (4,093,100
  labels before masking, not ~300,000), drop the unmeasured run times and
  memory figures, say that cost is linear in `len(av_grid) *
  len(rv_grid)`, and give the `mini_grid` and `av_grid` defaults that the
  code uses.
- New SVG logo, with light and dark variants, and a favicon. The isochrone
  "S" is drawn as model-grid points coloured by temperature, over a faint
  posterior across the grid.
- The documentation site moves to the Furo theme and Markdown (MyST) pages,
  with a brutus colour style and a new landing page.
- Rewritten getting-started pages and method overview, with schematics of
  the forward model, the fit and the distance-extinction degeneracy, and a
  References page.
- Rewritten stellar models and model grid pages. The EEP values of the main
  sequence are corrected, and the coverage of the default grid (0.5 to 2.0
  solar masses, [Fe/H] -3.0 to +0.45) is stated.
- The data module page says that only the `fetch_*` functions download
  files, and where the loaders look for them. It describes the model grid
  as a precomputed SED grid, adds the 1.1 GB EEP track file and its cache,
  and corrects the storage totals.
- The tutorials keep their data requirements in one table in
  `tutorial_utils.py`, which each setup cell checks, and leave file sizes
  to the installation page, which now lists the 13 MB Bayestar grid.
  Tutorial 5 no longer downloads the grid, the offsets and the dust map
  when it runs, and Tutorials 7 and 8 save their figures to
  `tutorials/plots/` like the others, not to the working directory.
- The `Bayestar.query` docstring states that `av_mean` and `av_std` keep
  the storage dtype of the file (float16 for `bayestar2019_v1.h5`), and
  `query_equ` states that it ignores the distance `d`.
- The `los_dust` docstrings state that the outlier model is uniform in
  reddening over `rlims`, independent of distance, and that `s_params`
  sets the kernel width as a fraction of the `rlims` range, not in
  percent.
- Tutorial 1 and the `Isochrone.get_predictions` docstring use the MIST
  primary EEPs: the main sequence is EEP 202 to 454, the subgiant and red
  giant branches 454 to 605, and core helium burning 631 to 707.
- The population docstrings give units (distance in pc), allow a
  per-star `cluster_prob`, describe the grid's model selection as the code
  does it, name `EEPTracks`, and use `afe=0.0` in their examples: the
  shipped isochrone file has only [α/Fe] = 0, and other values give NaN.
- The `cornerplot` docstring says that `labels` are field names of
  `params` and also the axis text. The `hist2d` docstring documents
  `smooth` and drops its "internal function" note. The `dist_vs_red`
  docstring gives the `ebv` default as `None`.
- The `loglike_grid` and `logpost_grid` docstrings document every
  parameter and each value they return, with shapes and units. The
  `get_sed_grid` docstring gives the distance in kpc, not parsecs.
- Module docstrings no longer list their classes and functions; the API
  pages list each module's members.
- The equations in 18 docstrings now render on the API pages.

## [1.2.0] - 2026-07-17

Systematic audit of every module outside the `BruteForce` hot path (covered in
1.1.1): population fitting, LOS dust, offsets, stellar models, priors, dust
maps, plotting, data loading, and utilities. 113 verified findings were fixed
across pull requests #61-#69, each with a regression test (the suite grew from
725 to over 1000 tests). Highlights below; full details are in the PR
descriptions. Items marked *(output-changing)* alter numerical results (in
every case toward the statistically correct answer).

### Fixed: statistical correctness *(output-changing)*

- **Cluster fitting**: binary modeling was silently non-functional (an
  `smf=` keyword typo swallowed by `**kwargs`); post-MS models were
  underweighted ~21x; the (mass, SMF) integration measure is now normalized;
  the uniform outlier model is a proper density (it previously *favored*
  flagging the best-measured stars as outliers).
- **LOS dust**: monotonicity constraints now apply to the effective
  extinction profile, and kernels renormalize for truncation at the
  reddening bounds.
- **Photometric offsets**: leave-one-out reweighting used incorrect
  importance weights, shrinking fitted-band offsets toward 1.
- **Priors**: metallicity/age priors were silently dropped when passed as
  plain arrays; the disk/halo normalization was inconsistent near the Sun;
  age-prior weights now condition on metallicity; the dust-map A(V) prior
  renormalizes over `avlim`.
- **Dust maps**: Bayestar queries honor the map's per-pixel reliability
  metadata by default (`apply_reliability_mask=False` restores raw values).
- **Plotting / post-processing**: distance posteriors regenerated from fit
  summaries were biased low (missing scale-to-distance Jacobian);
  `bin_pdfs_distred(cdf=True)` now cumulates along the reddening axis as
  documented; offset-diagnostic mask indexing fixed.
- **BruteForce**: parallax-based model pre-selection now uses the marginal
  (not conditional) scale uncertainty, so heavily reddened stars with good
  parallaxes no longer lose valid models before the posterior stage.
- **Utilities**: `quantile` uses one midpoint-CDF convention for weighted
  and unweighted input (unweighted values shift slightly); truncated-normal
  densities are tail-stable; a singular covariance in a batch no longer
  corrupts the parallel sampler.

### Fixed: robustness and API contracts

- Silent failure modes now raise clear errors: omitting a multi-valued grid
  axis (was quietly pinned to the grid edge), zero surviving filters in
  `load_models` (which also now drops a constant `afe` column), a dust map
  without sky coordinates, invalid `avlim`, mis-shaped weights/coordinates
  in the plotting utilities, and half-specified offset priors.
- NaN fluxes and zero errors are masked per band instead of invalidating
  the whole object; generated grids no longer contain invalid models;
  caller-supplied arrays (`lnprior`, `hist2d_kwargs`, labels) are no longer
  mutated in place.
- Additions: `dustfile` accepts `pathlib.Path`; `bin_pdfs_distred` /
  `dist_vs_red` support importance weights and multi-object input;
  `phot_loglike` supports shared model grids and extra chi-square terms;
  `draw_sar` gained `max_attempts` (its RNG stream changed, so fixed-seed
  draws differ; the distribution is verified identical).

### Performance (equivalence-tested)

Grid generation 16.7x per model; photometric-offset calibration ~12x
end-to-end; LOS dust likelihood up to 4.2x (prior transform 51x);
`draw_sar` 2.6-4.5x; `Bayestar()` init 8x; galactic prior 1.75x
(`logp_extinction` ~9x); `Isochrone` memory roughly halved; `EEPTracks`
caches ~264 MB smaller (now versioned `_cachev2.pkl`).

### Infrastructure

- The `docs` CI check no longer fails instantly on every pull request: the
  build was decoupled from the master-only GitHub Pages deployment
  environment, so PRs now actually build the documentation (deploys still
  run only from master).

## [1.1.1] - 2026-06-27

Performance pass on the individual-star `BruteForce` fitter (`loglike_grid` ->
`logpost_grid` -> `_fit`). Benchmarked on the real MIST grid (`grid_mist_v9.h5`,
613,530 models) and the real Orion field (Gaia parallaxes), across a cohort
spanning the full selected-model range (4 .. the `max_models=50000` cap).
End-to-end **~2.6x faster `loglike_grid`, ~2.3x faster `fit()`** on a 4-core host
(the parallel kernels scale further with physical cores). Every change is either
verified bitwise-identical in the saved (float32) outputs or proven *more
accurate* against a high-Nmc reference; see `bench/RESULTS.md` for the full
methodology and numbers.

### Changed

- **Monte-Carlo prior integration in `logpost_grid` now uses antithetic
  sampling** *(affects output -- more accurate)*: the per-model prior integral
  draws the proposal normals in antithetic pairs `(z, -z)`. The estimator stays
  unbiased; because the integrand is dominated along `ln(d)` by the
  galactic-density falloff and the `+log(d)` volume Jacobian (both monotone in
  the sampling direction), this cuts the MC-integration variance ~20x and reduces
  the finite-`Nmc` (Jensen) bias of `log(mean(w))`. Versus an `Nmc -> inf` gold
  standard the per-model log-posterior RMSE drops to ~0.28x and the bias 3-65x,
  at the same `Nmc` and half the Gaussian draws. The resulting shift in posterior
  summaries is within the procedure's intrinsic Monte-Carlo scatter. An
  adversarial review confirmed the one regime where antithetic could mildly
  *increase* variance (a very precise parallax centered on the photometric
  proposal mode with poorly-constraining photometry) is empirically negligible
  for real Gaia data (worst observed std ratio ~1.06) and never introduces bias.
- **`mc_ess` output is now a weight-concentration diagnostic, not a strict
  effective-sample count**: under antithetic correlation `1 / sum(w_i^2)` no
  longer counts independent samples (it can read up to ~2x optimistic for
  well-mixed models). It remains monotone and useful as a proposal/target
  mismatch indicator -- low values still flag poor overlap.
- `sample_multivariate_normal` gained an optional `antithetic` keyword (default
  `False`); `logp_galactic_structure` gained optional `feh`/`loga` array keywords
  (a fast path for the per-point metallicity/age prior). Existing calls are
  unaffected.

### Performance

All of the following are verified **bitwise-identical** to the prior results in
the saved `fit()` outputs (any differences are float64 round-off below float32
storage precision that flip no discrete selection):

- `loglike_grid`'s per-model numba kernels (`_get_seds`, `_optimize_fit_mag`,
  `_optimize_fit_flux`, `_get_sed_mle`) and the multivariate-normal sampler are
  parallelized with `prange` (rows are independent; reductions are exact). The
  two convergence max-reductions in `_optimize_fit_mag` are parallelized as well.
- Removed dead computation in `_get_sed_mle` (`models_int`/`reddening` were
  computed -- 4.9M `pow` calls per evaluation -- but never read).
- Fused kernels `_chi2_from_resid` and `_init_mag_resid` replace large NumPy
  temporaries; `np.zeros` -> `np.empty` for fully-overwritten arrays.
- `_get_seds` flux conversion uses `exp(fac*mag)` instead of `10**(-0.4*mag)`
  (~1.6x faster on that hot path; agreement ~1e-15).
- The batched 3x3 covariance regularization uses an analytic symmetric-3x3
  minimum-eigenvalue kernel instead of `numpy.linalg.eigvalsh` (no less accurate;
  the regularization decision is unchanged).
- `logpost_grid` no longer tiles a *structured* label array across all MC
  samples for the default galactic prior (a numpy structured-dtype `np.tile` is
  ~100x slower than the float equivalent; it dominated logpost for large
  selected-model counts). It now hands the prior the per-point `feh`/`loga` as
  plain float arrays. Bitwise-identical; ~1.24x faster `logpost_grid` at
  `Nsel=50000` (~165 ms/object), no change for small selections.

## [1.1.0] - 2026-05-29

Maintenance and polish release: verified bug fixes (several affecting numerical
output on specific paths), documentation accuracy, test-suite hygiene, and
release-process improvements. All fixes ship with regression tests.

### Fixed

- **`FastNN.encode` silent mis-scaling for 6-sample batches** *(affects output)*:
  a 2-D input of shape `(6 params, 6 samples)` broadcast against the parameter
  bounds without error and normalized along the wrong axis, producing silently
  corrupted SEDs/bolometric corrections only when exactly 6 valid samples were
  evaluated (reached via `StellarPop` synthesis). Now dispatches on input
  dimensionality explicitly.
- **`los_dust` kernels collapsing per-object cloud means** *(affects output)*:
  `kernel_gauss`/`kernel_lorentz`/`kernel_tophat` collapsed an array-valued mean
  to a scalar, so line-of-sight fits using a reddening template (`template_reds`,
  `additive_foreground`) evaluated every object against object 0's cloud mean.
  Means are now kept broadcastable. The default uniform-cloud path is unchanged.
- **Binary companions no longer discard a valid primary SED** *(affects binary
  grid generation)*: `StarEvolTrack.get_seds` returned an all-NaN combined SED
  (dropping the valid primary) when the secondary's age could not be matched to
  the unrealistically tight default tolerance. The default `tol` is relaxed
  `1e-6 -> 1e-2` dex and a primary-only fallback is used when the companion does
  not converge. *Regenerate binary (smf>0) grids to benefit; single-star fitting
  and the shipped default grids are unaffected.*
- `load_offsets` no longer crashes on a single-filter offsets file.
- `logp_imf` no longer raises `ZeroDivisionError` for a power-law slope of
  exactly 1.0 (flat-in-log); uses the logarithmic mass integral.
- `_fetch` (data download) removes a stale/broken symlink before re-linking and
  falls back to a file copy on filesystems without symlink support.
- `BruteForce.logpost_grid` floors the Monte-Carlo sample count at 1 (a very
  small `mem_lim` could drive it to 0 and crash).
- `BruteForce.loglike_grid` guards the dimensional-prior `log(Ndim)` term and
  fails fast with a clear error on a fully-masked object (was an opaque
  `ZeroDivisionError` deep in the optimizer).
- `cornerplot` no longer mutates a caller-supplied `labels` list in place.
- `dist_vs_red` uses `interpolation="none"` so binned PDF images show raw bins;
  fixed a no-op smoothing guard in `hist2d`.
- Removed a dead `from scipy import polyfit` import (`scipy` has no top-level
  `polyfit`; the code always used `numpy.polyfit`).

### Changed

- **Python support**: dropped end-of-life Python 3.8 (`requires-python>=3.9`);
  added 3.13 to the classifiers and tooling targets.
- **Dependencies**: raised the `numba` floor to `>=0.59.0` (the old 0.53 floor
  predates `numpy>=1.22`/2.x support); `numpy` remains uncapped (the code is
  numpy-2.0 clean, verified on 2.2).
- **Documentation engine**: standardized on a single NumPy-docstring processor
  (`napoleon`); removed the redundant `numpydoc` extension. The docs now build
  with zero warnings.
- **Tutorials**: standardized on the committed `Orion_l209.1_b-19.9` example
  field; replaced chi2/Nbands quality cuts with goodness-of-fit p-values.

### Documentation

- Corrected numerous stale code examples across the docs and docstrings to match
  the real API (`StarGrid(models, labels)`, `get_seds` 3-tuple in magnitudes,
  `magnitude(flux, err)`, `sample_multivariate_normal(size=)`, the prior call
  signatures, `samps_dred`, the data cache path/env var, real filter names, and
  the maggies flux convention).
- Added `summary_plot` to the API reference; documented `BruteForce.fit`'s
  performance/accuracy parameters (`max_models`, `precision_shrinkage`,
  `subsample_mode`, `R_solar`, `Z_solar`); reframed fit-quality guidance around
  the p-value methodology.

### Testing & packaging

- Added regression tests covering every fix above; reconciled the `conftest`
  coverage guidance with what CI actually runs
  (`NUMBA_DISABLE_JIT=1 pytest --cov`); the default `pytest` run no longer
  enables coverage (which crashed on some WSL/DrvFs paths).
- Added a tag-triggered PyPI publish workflow (Trusted Publishing); fixed the
  stale `MANIFEST.in` data-file path.

## [1.0.0] - 2025-12-08

First stable release of brutus following code verification, testing, and documentation improvements.

### Added

- **Documentation**: Scientific background pages, user guides, API documentation with examples, and ReadTheDocs hosting
- **Testing**: 92% code coverage, 606 tests, GitHub Actions CI/CD with Codecov integration
- **Code verification**: All functions verified for correctness; fixed IMF normalization bug, StarGrid distance reference, and other issues

### Changed

- Development status updated to Production/Stable
- Added `tqdm` as formal dependency
- Enforced Black formatting across codebase

## [0.9.0] - 2024-08-28

Major refactoring to improve usability and maintainability while preserving scientific functionality.

### Added

- **Modern packaging**: Migrated to `pyproject.toml` with black, isort, flake8, mypy
- **Testing**: pytest framework with 100+ tests, coverage reporting, multi-platform CI
- **Modular architecture**: Split into `brutus.core`, `brutus.analysis`, `brutus.plotting`, `brutus.dust`, `brutus.utils`, `brutus.data`, `brutus.priors`
- **Performance**: Numba JIT compilation, vectorized operations, improved caching

### Changed

- Minimum Python version: 3.8+ (dropped Python 2.7)
- Split large modules (`utils.py`, `plotting.py`) into focused submodules
- Updated all dependencies to modern versions

### Fixed

- Windows/WSL compatibility documentation
- Circular imports and module loading issues
- Infinite loop bug in `hist2d` function

### Migration

Update imports:

```python
# Old
from brutus.seds import Isochrone
from brutus.fitting import BruteForce

# New
from brutus import Isochrone, BruteForce
# or
from brutus.core import Isochrone
from brutus.analysis import BruteForce
```

All scientific algorithms, file formats, and core APIs remain unchanged.

## [0.8.3] - Previous Release

Final release using old project structure and Python 2 compatibility.

Features: individual star fitting, cluster modeling, 3D dust mapping, MIST support, neural network SED prediction.

---

For migration questions or bug reports, see the [issue tracker](https://github.com/joshspeagle/brutus/issues).
