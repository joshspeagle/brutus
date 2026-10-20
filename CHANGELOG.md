# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — statistical correctness *(output-changing)*

- **The `max_models` cap no longer biases the posterior.** Above the
  cap, `logpost_grid` kept a weighted sample of models but gave them their
  full weights, which narrowed the distance posterior by up to 31% and
  lowered `obj_log_evid` by up to 0.49. It now keeps model i with probability
  `min(1, c * w_i)` by systematic sampling, returns that log-probability as
  `ln_incl`, and `fit` divides each kept model's weight by it. Fits below the
  cap do not change.
- **`loglike_grid` fits each model to its best (scale, A(V), R(V)).**
  The flux-space R(V) step left out the factor A(V) in `df/dR(V)`, so the
  refinement stopped at a point that gave the R(V) prior too much weight
  when A(V) > 1. The magnitude-space fit alternated A(V) and R(V) steps and
  stopped far from its minimum when the two are correlated, so the first
  cut could drop the best model of a star that no model fits exactly. Both
  fits now take Gauss-Newton steps in (A(V), R(V)) with the distance
  profiled out, hold a parameter at its bound while they fit the other,
  retry a step that does not improve the fit with Levenberg-Marquardt
  damping, and stop each model when the predicted gain
  in log-posterior is below `ltol`, whose default is now `1e-3` (was
  `3e-2`). On 24 Orion stars, master's best-model `lnl` was up to 9.8 too
  low.
- **The first cut of `loglike_grid` keeps models within a factor `1e-6` of
  the best** (`init_thresh`, `logl_initthresh`), not `5e-3`. The cut ranks
  models by their magnitude-space fit, whose chi2 differs from the
  flux-space fit, so the old cut left models unrefined that carry posterior
  weight. On 24 Orion stars, the largest total-variation distance between
  the model weights and those of a fit without the cut falls from 0.106 to
  0.0005, at the same run time.
- **`fit(apply_dlabels=False)` evaluates the Galactic prior without model
  labels.** `logpost_grid` replaced `dlabels=None` with the grid labels, so
  the option had no effect. `logpost_grid` and `_fit` now default to the
  grid labels, and `dlabels=None` passes `labels=None` to the prior.
- **Dust-map A(V) prior scatter restored**: `logp_extinction` again adds a
  0.2 mag systematic scatter in quadrature to the map spread (new `scatter`
  argument, default `0.2`, as in brutus <= 0.8.3; the 1.x refactor had
  dropped it). Bayestar19 spreads are ~0.02 mag (median), so without it the
  prior pinned A(V) — and, through the A(V)-distance profile, the distance —
  far more tightly than the map supports, producing overconfident and
  occasionally badly wrong distances. Map bins with zero spread now also
  receive a proper prior instead of falling back to uniform.
- **Dust-map prior is proper in masked bins**: where the map is
  uninformative (reliability-masked distance bins, no coverage), the prior
  is now the normalized uniform density over `avlim`,
  `-ln(avlim[1] - avlim[0])`, instead of `ln p = 0`. The improper value
  acted as a density of 1 per mag and up-weighted reliability-masked
  distances relative to reliable ones by an arbitrary, unit-dependent factor.
- **`BruteForce` passes `avlim` to the default dust prior**, so its
  truncation and uniform normalization match the fitted A(V) range (custom
  `lndustprior` functions keep the 4-argument signature).
- **An integer 0/1 `data_mask` masks bands** in `BruteForce.fit` and
  `loglike_grid`. It was used as an index array, so the fit used band 0 and
  repeated band 1 in place of the masked selection. Boolean masks give the
  same results as before. `loglike_grid` no longer changes the caller's
  `data_mask`, `av_init` or `rv_init`.
- **`loglike_grid` normalizes every model's `lnl`**. The Gaussian constant
  `-0.5 * (Ndim * ln(2 pi) + sum(ln sigma^2))` went only to the models that
  passed the initial cut, so the culled models sat that constant (about 90
  log-units for fluxes in maggies) too low. Culled models keep their
  magnitude-space fit of (scale, A(V), R(V)).
- **`logpost_grid` truncates the Monte Carlo draws** of A(V), R(V) and
  distance at `avlim`, `rvlim` and (0.001, 1e6) kpc. A draw outside the
  support now gets zero weight. Before, it was clipped to the bound at full
  weight: about half of the A(V) draws of an unreddened model sat at
  exactly 0, and such a model gained up to ln 2 in `lnp`. This changes
  `samps_red`, `samps_dred`, `samps_dist`, `obj_log_post`, `obj_log_evid`
  and `mc_ess`; `mc_ess` is 0 for a model with no draw of nonzero weight.
- **A `functools.partial` of `logp_extinction` gets the fit's `avlim`**, as
  the bare function does, unless the partial binds `avlim` itself. A prior
  such as `partial(logp_extinction, scatter=0.3)` was normalized over its
  default (0, 20) instead of the fitted range.
- **`BruteForce.fit` skips an object with no valid bands** instead of
  stopping the whole run with `ValueError`. The object is written with
  `model_idx = -99`, `obj_Nbands = 0` and NaN values, and `fit` warns once
  with the number of such objects. The output file now closes on every exit
  path.
- **`quantile` drops samples with zero weight** before it builds the CDF.
  A zero-weight sample still got a CDF node, so `quantile([0, 1, 100],
  [0.5, 0.9], weights=[1, 1, 0])` gave `[0.5, 60.4]` instead of
  `[0.5, 1.0]`. Negative or non-finite weights, or weights that sum to 0,
  raise `ValueError`.
- **An object with no degrees of freedom adds 0 to a cluster fit.** With
  `dim_prior=True`, `phot_loglike` and `chisquare_outlier_loglike` gave
  such an object (`Ndim - dof_reduction <= 0`, for example all bands NaN)
  `-inf`, which made the cluster log-likelihood `-inf` for every parameter.
  It now gets 0, the same as with `dim_prior=False`.
- **`truncnorm_logpdf` and `logp_imf` return `LOG_ZERO` (-1e300)** for
  zero probability, not `-inf`, as the rest of brutus does. The fused and
  numpy paths of `logp_galactic_structure` now agree for ages above
  `max_age` (they gave -1e300 and -inf). `truncnorm_logpdf` and
  `truncnorm_pdf` accept numpy scalars and 0-d arrays, so
  `logp_age_from_feh(5.0)` no longer raises `TypeError`. The value
  `LOG_ZERO` lives in `brutus.utils.math`.
- **`chisquare_logpdf` returns `LOG_ZERO` (-1e300)** where the density is
  zero, not `-inf`, like `truncnorm_logpdf`. At `x == loc` it now agrees
  with `scipy.stats.chi2.logpdf` (`-ln 2` for `df = 2`, `+inf` for
  `df < 2`), and float32 input no longer overflows to `-inf`.
- **The line-of-sight and cluster log-likelihoods return `LOG_ZERO`
  (-1e300)** for a zero probability, not `-inf`:
  `los_clouds_loglike_samples`, `kernel_tophat`, `kernel_gauss`,
  `kernel_lorentz`, `marginalize_isochrone_grid`,
  `isochrone_population_loglike` and `uniform_outlier_loglike`. With
  `dim_prior=True`, `phot_loglike` gives `LOG_ZERO` for an exact fit
  (`chi2 = 0`) with 3 or more degrees of freedom.
- **`kernel_gauss` and `kernel_lorentz` normalize over `rlims` stably in
  the tails.** `kernel_gauss` took the mass from an `ndtr` difference that
  rounds to 0 for a kernel mean about 8 sigma below `rlims` or 38 sigma
  above; it now uses `log_ndtr`. `kernel_lorentz` took it from an `arctan`
  difference that lost its digits for a mean far from `rlims` (a relative
  error of 2e-4 at 1e7 widths).
- **The fused Galactic prior normalizes the age prior with the tail-stable
  log-mass of the fused extinction prior.** It used a floored `erf`
  difference. The prior changes by at most 1.8e-15 for valid inputs; the
  fits of `bench/fullfit.py` do not change.
- **`logp_extinction` normalizes its truncated normal stably in the
  tails.** It took the mass on `avlim` from an `erf` difference with a
  `1e-300` floor, so a map mean far outside `avlim` (for example 9 mag
  with `avlim=(0, 6)`) gave a log-prior near +631 instead of about +1. It
  now uses `log_ndtr` in the numpy path and a mirrored `erfc` in the fused
  kernel; both agree with `scipy.stats.truncnorm.logpdf`.
- **Batched `inverse3(regularize=True)` regularizes like the single-matrix
  path.** Its smallest-eigenvalue check cancelled for ill-conditioned
  matrices and shifted some that needed no shift. The saved results of the
  `bench/fullfit.py` fits do not change.
- **`draw_sar` and the plots that regenerate draws use `avlim=(0, 20)`**,
  the `BruteForce.fit` default, not `(0, 6)`. This affects `cornerplot`,
  `dist_vs_red` and `bin_pdfs_distred`. A star with A(V) above 6 mag got
  every draw replaced by its mean. Without `span`, the A(V) axis of
  `dist_vs_red` and `bin_pdfs_distred` now covers 0 to 20 mag, also for
  saved draws, so the default bins are 3.3 times wider; pass `span` for a
  narrower range. `draw_sar` pads a distribution with its mean after 100
  passes with no accepted draw, instead of after `max_attempts`.
- **`analysis.photometric_offsets` weights the draws of a fitted band by
  the Gaussian leave-one-out ratio**, `ln w = chi2_b / 2`. The old default
  (`dim_prior=True`) used a ratio of chi-square densities, which does not
  match the posterior `BruteForce` samples, with or without
  `logl_dim_prior`. The default offsets change. With a 10% offset injected
  into a fitted band (400 stars, 1000 draws, 8 seeds), the mean bias fell
  from 0.0026 to 0.0008.
- **`dist_vs_red` and `bin_pdfs_distred` smooth A(V) by 0.005 of its range
  by default** (0.1 mag over the default `avlim`): the default `smooth` is
  `(0.015, 0.005)` and `(0.01, 0.005)`.
- **`load_models` applies the main-sequence and binary cuts whatever
  `labels` holds.** The cuts read the file's `eep` and `smf` columns. A
  `labels` list without `eep` or `smf` skipped the cut: on `grid_mist_v9.h5`,
  `labels=['mini', 'feh'], include_postms=False` returned 613530 models
  instead of 143856. A phase cut on a file without `eep` now raises
  `ValueError` instead of keeping every model.
- **`StellarPop.get_seds` finds each companion's EEP on the isochrone's
  own EEP grid.** It inverted mini(EEP) on the caller's `eep` only, so a
  coarse or truncated `eep` dropped companions below its smallest mass and
  misplaced the others; a single `eep` never got a companion. At
  [Fe/H] = 0, log(age) = 9.5 and `binary_fraction=0.6`, the SDSS g
  magnitude at EEP 350 is now 14.621 for any `eep`, against 14.645 with
  `eep=np.arange(300, 480, 10)`. With the default cluster grids,
  `isochrone_population_loglike` changes by about 0.01 on a synthetic
  cluster of 30 stars. Companions that a young isochrone places before the
  zero-age main sequence are now included.
- **`StellarPop.get_seds` returns NaN `params2` for every star whose SED
  has no companion**, for all binary fractions. For `0 < binary_fraction
  < 1`, it also returned the parameters of companions that it left out of
  the SED (below `mini_bound`, or with no valid SED).
- **`los_clouds_loglike_samples` averages each star's likelihood over its
  finite draws only.** It divided by the count of all draws, so NaN draws
  lowered the star's inlier term against the outlier term, and a star with
  only NaN draws added `ln(pb) - ln(area)`, which pulled the outlier
  fraction `pb` upward. A draw with a NaN or infinite distance or
  reddening is now left out, a finite distance outside [0, 1e10) still
  counts as a zero-likelihood draw, and a star with no finite draw is left
  out of the sum. With 5 stars, 25 draws each, and one star all NaN, the
  log-likelihood at `pb` = 0.01, 0.1 and 0.5 was -8.25, -6.22 and -6.14;
  it is now -1.85, -2.13 and -3.66, the values without that star.
- **`StarEvolTrack` finds a companion's EEP at the primary's [α/Fe]**,
  with no clamp to the library range, as `get_seds` already used for the
  companion's parameters. Tracks without a `gridpoints` EEP axis used
  [α/Fe] = 0.0. On `MIST_1.2_EEPtrk.h5`, results at [α/Fe] = 0 do not
  change; for 0 < |[α/Fe]| < 1e-5 the companion EEP changes by at most
  9e-13 (200 random binaries each at ±3e-6).
- **`logp_galactic_structure` computes in float64 for float32 `labels`.**
  It used the `feh` and `loga` fields of a structured `labels` array at
  their own dtype, so a float32 grid gave float32 metallicity and age
  priors on the numpy path (1000 points or fewer) and, with
  `NUMBA_DISABLE_JIT=1`, on the fused path. Both call styles now take the
  same float64 path. On 500 random points the log-prior changes by at most
  2.0e-6 (relative 2.7e-7). A multi-dimensional `labels` must have the
  shape of `dists`; one of the same size but another shape now raises
  `ValueError`, where it could pair distances with the wrong labels.
- **`StarGrid.get_seds` and `get_predictions` reject queries they cannot
  answer.** A NaN, infinite or out-of-range grid label raises `ValueError`
  that names it; before, it returned the edge model with no warning
  (`grid_mist_v9.h5`: `mini=50` and `mini=NaN` gave the `mini=2` SED).
  Values within a relative 1e-6 of an axis edge are accepted. With
  `use_multilinear=False`, an omitted multi-valued axis now raises
  `ValueError`, as with the default; before, the KD-tree ignored it.

### Changed

- **`hist2d` no longer accepts `**kwargs`.** It ignored unknown keywords,
  so an unknown key in `cornerplot(hist2d_kwargs=...)` had no effect. An
  unknown keyword now raises `TypeError`. `cornerplot` no longer sets a
  default `alpha` for the 2-D histograms, which `hist2d` never applied;
  the figures do not change.
- `StarEvolTrack.get_seds` and `Isochrone.get_predictions` pass errors of
  the tracks, the interpolator and the neural network through with their
  own type. `get_seds` wrapped a track error in `RuntimeError`, and turned
  an error in the primary or companion SED into a `RuntimeWarning` with a
  NaN or single-star SED; the companion EEP search turned errors into a
  NaN EEP. `Isochrone.get_predictions` wrapped interpolator errors in
  `RuntimeError`. `EEPTracks` still rebuilds any cache that it cannot load,
  and skips the cache write only on `OSError`.
- **`GridGenerator.make_grid` has no `dist` argument.** Grids are always
  computed at 1 kpc, the distance that `load_models`, `StarGrid` and
  `BruteForce` assume; a grid made at another `dist` gave distances off by
  a factor of (dist / 1000)^2. Passing `dist` now raises `TypeError`.
- `GridGenerator.make_grid` raises `ValueError` naming the argument when
  `av_grid` or `rv_grid` has a non-finite value or fewer than 2 distinct
  values, or when `av_wt` or `rv_wt` has a non-finite value or a different
  shape from its grid. Before, a one-value lattice gave a rank-deficient
  fit with only a `RankWarning`, so the reddening coefficients that it
  could not determine were arbitrary; a NaN marked every model invalid.
  Tutorial 3 now passes 4 R(V) values.
- **`EEPTracks` drops the unused attributes `null` and `binwidths`.** Its
  `ygrid` attribute holds the unpadded grid (one [α/Fe] layer when the
  tracks have one [α/Fe] value); `xgrid` and `grid_dims` still describe the
  padded grid of the interpolator.
- **`StarEvolTrack.get_seds`, `StarGrid.get_seds` and
  `StarGrid.get_predictions` no longer accept `**kwargs`.** They ignored
  unknown keywords, so a misspelled argument such as `Av=5.0` or `fe_h=-1`
  gave a wrong result or a misleading error. An unknown keyword now raises
  `TypeError`.
- **`subsample_mode` is removed** from `fit`, `_fit` and `logpost_grid`,
  with its `"topk"` option, which truncated the posterior. `logpost_grid`
  returns a ninth value, `ln_incl`, the log-probability that each selected
  model was kept by the `max_models` cap.
- **`max_models` must be None or a positive integer**; `fit` and
  `logpost_grid` raise `ValueError` otherwise. With the old cap, `0` kept
  every model.
- `load_models` raises `ValueError` for `include_ms=False,
  include_postms=False` before it reads the file. It warns with a
  `UserWarning` that lists requested `labels` absent from the file.
  `load_offsets` warns for a filter name that is neither in `brutus.FILTERS`
  nor in the file, and its verbose summary prints each offset as a signed percentage, such as
  `+12.34%`.
- **Every `fetch_*` function takes the file choice first and `target_dir`
  only as a keyword**, for example `fetch_offsets("mist_v9",
  target_dir=...)`. A call that passes `target_dir` by position, such as
  `fetch_isos("./data")`, now fails. Before, `fetch_grids` alone took the
  file choice first, so `fetch_offsets("mist_v9")` made a link in a new
  directory `./mist_v9`. An unknown option raises `ValueError` that names
  the argument, the valid options and the given value.
- `Bayestar.get_query_size` is removed. Nothing called it, and it gave
  twice the result size for an `[l, b]` array.
- **`ltol_subthresh` is removed** from `loglike_grid`, `fit` and `_fit`.
  Each model's fit now checks its own convergence. `ltol` must be positive.
- **`dim_prior` is removed** from `analysis.photometric_offsets`, which
  always uses the Gaussian leave-one-out weights. Passing it raises
  `TypeError`. `old_offsets` must be finite and > 0.
- **`dim_prior` is removed** from `plotting.photometric_offsets` and
  `plotting.photometric_offsets_2d`, which use the weights of
  `analysis.photometric_offsets`.
- **`PhotometricOffsetsConfig` loses four switches.**
  `use_vectorized_bootstrap` and the per-object bootstrap loop it selected
  are removed. `validate_inputs` is removed, so input validation always
  runs. `random_seed` is removed: pass a seeded `rng` to
  `photometric_offsets` instead (`rng=np.random.default_rng(seed)` gives
  the old results of `random_seed=seed`). `progress_interval` is removed,
  and `verbose` alone controls the progress messages.
- `logpost_grid` takes `return_precision=True` to return the precision
  matrices of the selected models in place of their covariances. `fit`
  uses it to invert only the resampled models.
- The default fit limits are the constants `brutus.utils.math.AVLIM`,
  (0, 20) mag, and `RVLIM`, (1, 8). Every function that takes `avlim` or
  `rvlim` uses them as its default.
- `import brutus` now raises the original `ImportError` when a dependency is
  missing. Before, it warned with a hidden `ImportWarning` and left most names
  undefined.
- `brutus.__version__` comes from the installed package metadata, so the
  `brutus_version` attribute of generated grids follows the installed release.
  A source tree that is not installed reports `"unknown"`.
- The top level re-exports every data fetcher and loader: `fetch_nns`,
  `fetch_offsets`, `fetch_tracks` and `load_offsets` join `fetch_dustmaps`,
  `fetch_grids`, `fetch_isos` and `load_models`.
- `brutus.core` no longer exports the private `_get_seds`; import it from
  `brutus.core.sed_utils`. `brutus.dust.__version__` is removed.
- `StarGrid` is in `brutus.core.individual.__all__`.
- `logp_extinction` no longer accepts `avlim=None`, which gave masked map
  bins an improper `ln p = 0` next to normalized Gaussians in reliable
  bins. Pass the A(V) support of the fit instead; `None` raises
  `ValueError`.
- `brutus.utils` re-exports `chisquare_outlier_loglike` and
  `uniform_outlier_loglike`. The private, unused `_function_wrapper` is
  removed from `brutus.utils.math`.
- Package metadata declares the license as the SPDX expression `MIT`
  (`License-Expression: MIT`) instead of a license table and classifier.
  Only `LICENSE` is packaged as a license file; `AUTHORS.md` stays in the
  source distribution. Building from source needs `setuptools>=77`.
- `BruteForce.loglike_grid` and `logpost_grid` no longer accept unused
  `**kwargs`, so a misspelled keyword such as `av_gaus` raises `TypeError`.
  Before, it was dropped and the default applied.
- Python 3.10 or later and numpy 2.0 or later are required (the floors were 3.9
  and 1.19). Python 3.9 is past its end of life, and no astropy release for
  Python 3.9 supports numpy 2.
- `StellarPop.get_seds(return_dict=False)` and `get_seds_smf_grid`
  return `params` with shape (Neep, Npred), like `params2` and
  `StarEvolTrack.get_seds`. It was the transpose, (Npred, Neep).
- `StellarPop` and `StarEvolTrack` raise the error of the neural network
  when it fails to load, at construction. Before, they set `predictor` to
  None, and `get_seds` raised a `RuntimeError` without the cause.
  `FastNNPredictor` raises `ValueError` that names the filters missing from
  `nnfile`, in place of an h5py `KeyError`. `StellarPop` raises
  `ValueError` when `isochrone.predictions` lacks mini, logl, logt, logg,
  feh_surf or afe_surf, and `StarEvolTrack` when `tracks.predictions`
  lacks loga, logl, logt, logg, feh_surf or afe_surf.
- `StellarPop.get_seds` computes SEDs only with `FastNNPredictor.sed_batch`.
  The per-star `sed` fallback, which ran when `sed_batch` raised, is
  removed; such an error now propagates, as does an error in the
  companions' parameters, which gave single-star SEDs with a warning.
- `Isochrone.get_predictions` and `EEPTracks.get_corrections` share one
  correction function, with bitwise-identical results. `Isochrone` raises
  `ValueError` for a `corr_params` without 4 entries, and at construction
  for a file without the mini, logt, logl or logg label; it warned and
  returned uncorrected predictions. `StellarPop` passes errors of
  `Isochrone.get_predictions` through, not as `RuntimeError`.
- `cornerplot` defaults `applied_parallax` to `None`, which applies the
  parallax when both `parallax` and `parallax_err` are given. The default
  `True` raised for saved `(dists, reds, dreds)` draws given without a
  parallax, where the flag has no effect. `applied_parallax=True` without
  both values now raises only for `(scales, avs, rvs, covs_sar)` input.
- `dist_vs_red` takes `fig=(fig, ax)` and draws on `ax`, as the other
  plotting functions do. Without it, it draws on the current axes, as
  before.
- `summary_plot` takes `rstate` and passes it to `cornerplot` and
  `posterior_predictive`. The violins of the SED inset came from the global
  numpy stream, which only `np.random.seed` could seed.

### Fixed

- `logpost_grid` raises `ValueError` that names `dlabels` when the default
  Galactic prior gets a plain, not structured, `dlabels` array. It raised a
  `TypeError` about `NoneType`.
- `StarGrid` keeps the requested `filters` order for a structured `models`
  array, as `load_models` does. It ordered the columns as in the file and
  dropped unknown names silently; it now warns with the skipped names and
  raises `ValueError` when none exists. For an unstructured `models`
  array, a `filters` list of the wrong length raises `ValueError`.
- `StarGrid` no longer warns about missing grid corners that carry zero
  weight. On `grid_mist_v9.h5`, 96 of 3000 queries on grid nodes warned;
  the SEDs do not change.
- `StarGrid.get_seds(use_multilinear=False)` works on a grid of one
  model; it raised `AttributeError`. On a grid with no grid label (`mini`,
  `eep`, `feh`, `afe` or `smf`), both neighbour methods raise `ValueError`;
  the default method returned the first model for any query. A `dist` that
  is not a finite distance > 0 raises `ValueError`; a negative `dist` gave
  NaN magnitudes.
- `find_nn_file`, `EEPTracks` and `Isochrone` look for their default data
  files in the brutus download directory (`strato.path`, which
  `ASTRO_BRUTUS_DATA_DIR` sets) first, then in `data/DATAFILES`. They
  ignored `ASTRO_BRUTUS_DATA_DIR`, so they missed files that the `fetch_*`
  functions had downloaded there. A missing default file raises
  `FileNotFoundError` that names the directories and the `fetch_*`
  function to run.
- `EEPTracks` raises `ValueError` naming `predictions` when it lacks
  `loga`, `logl`, `logt` or `logg` or has an unknown name, and
  `FileNotFoundError` with a `fetch_tracks()` hint for a missing
  `mistfile`; `Isochrone` raises it with a `fetch_isos()` hint. It no longer wraps errors from reading the file in
  `RuntimeError`.
- `lb2pix` returns -1 for a NaN or infinite longitude, as for an invalid
  latitude. It returned a valid pixel, so `Bayestar.query` could give such
  a star the extinction profile of another sightline.
- `Bayestar.query` raises `ValueError` naming `coords` when an array input
  does not have shape `(2,)` or `(Ncoords, 2)`. A `(2, 3)` array of
  `[l values, b values]` was read row by row, and an `(N, 3)` array lost its
  third column. `query_gal` with a scalar `ell` and `b` returns profiles of
  shape `(n_dist,)`, as `query_equ` does; it returned `(1, n_dist)`.
- `logp_extinction` raises `ValueError` for a 3D dust map called without
  `distance`, and `TypeError` when `dustmap.query` returns something other
  than a tuple, list or array of length 2 or 3. Both cases returned the
  uniform prior with no warning, which turned off the dust prior. A list or
  array result is now read like a tuple.
- `logp_galactic_structure` raises `ValueError` when `labels` does not have
  one entry per distance. With JIT on and more than 1000 distances, it read
  past the end of `labels`; otherwise it warned and dropped the metallicity
  and age priors. A scalar distance with `labels` or `feh` now works. The
  fused kernels of the Galactic and extinction priors no longer catch every
  error and fall back to numpy with a warning.
- `logp_imf` raises `ValueError` naming the argument for a `mass_min` or
  `mass_break` that is not finite and positive, a `mass_max` that does not
  exceed `mass_min`, or a slope that is not finite. A NaN `mass_break`
  returned `-inf` everywhere.
- `inverse3` converts `A` to float and raises `ValueError` unless its shape
  is `(3, 3)` or `(N, 3, 3)`. An integer matrix gave an integer inverse that
  truncated 0.5 to 0.
- `sample_multivariate_normal` converts `mean` and `cov` to float and raises
  `ValueError` when `cov` does not have shape `(Ndist, dim, dim)`. An
  integer `cov` shrank the spread of the draws, and a short `cov` returned
  garbage with JIT on.
- `core.sed_utils.get_seds` raises `ValueError` unless `mag_coeffs` has
  shape `(Nmodels, Nbands, 3)`. With fewer coefficients, the JIT kernel
  read past the coefficient axis.
- `plotting.posterior_predictive` treats an integer 0/1 `data_mask` as a
  mask and raises `ValueError` for a mask of the wrong shape. It plotted
  band 0 and repeated band 1 in place of the unmasked bands.
- `BruteForce.fit` and `loglike_grid` check their input shapes and raise
  `ValueError` that names the argument, before `fit` creates its output
  file. A `parallax` given without `parallax_err`, which the fit silently
  ignored, now raises. `logl_initthresh` (`init_thresh`) must be in (0, 1).
  A parallax with an error of 0 or less counts as no parallax, as in the
  parallax priors; it stopped the run with an unclear error.
- `BruteForce.logpost_grid` raises `ValueError` when `lnprior` or `dlabels`
  has a different length from the likelihood results. After
  `loglike_grid(indices=...)`, the default `dlabels` gave the Galactic prior
  the labels of the wrong models.
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
  and corrects the storage totals. Tutorials 00 to 04 give the same file
  sizes.
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

## [1.2.0] - 2026-07-17

Systematic audit of every module outside the `BruteForce` hot path (covered in
1.1.1): population fitting, LOS dust, offsets, stellar models, priors, dust
maps, plotting, data loading, and utilities. 113 verified findings were fixed
across pull requests #61-#69, each with a regression test (the suite grew from
725 to over 1000 tests). Highlights below; full details are in the PR
descriptions. Items marked *(output-changing)* alter numerical results — in
every case toward the statistically correct answer.

### Fixed — statistical correctness *(output-changing)*

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

### Fixed — robustness and API contracts

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
