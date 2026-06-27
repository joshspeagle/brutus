# BruteForce pipeline performance optimization — results

Benchmarks use the **real** MIST grid (`grid_mist_v9.h5`, 613,530 models),
the **real** Orion field data (207 stars w/ Gaia parallaxes), the realistic
Orion-tutorial filter set (Pan-STARRS *grizy* + 2MASS *JHKs*, 8 bands), and the
real photometric offsets. Host: 4 physical cores.

## Headline (min-of-N estimator, real data)

| Stage          | Baseline | Optimized | Speedup |
|----------------|---------:|----------:|--------:|
| `loglike_grid` | ~790 ms  | ~332 ms   | **2.0–2.4×** |
| `logpost_grid` | ~178 ms  | ~166 ms   | ~1.08× (eigvalsh→analytic) |
| **`_fit` (full per object)** | **~977 ms** | **~515 ms** | **~1.9×** |

The wall-clock host is shared/noisy (±15%); the loglike/full-fit speedups are
stable across repeated clean A/B runs (git-stash baseline vs working tree, numba
cache cleared between, JIT enabled throughout).

## Verification — results are *exact*, not merely "within MC noise"

End-to-end `fit()` over 60 real Orion stars with a single threaded RNG
(`RandomState(12345)`), baseline vs optimized, comparing **every** HDF5 dataset:

```
model_idx, ml_scale, ml_av, ml_rv, ml_cov_sar, obj_log_post, obj_log_evid,
obj_chi2min, obj_Nbands, mc_ess, samps_dist, samps_red, samps_dred, samps_logp
   ->  WORST relative error over all datasets: 0.000e+00  (bitwise identical)
```

The only differences anywhere are at the ~1e-16 level in float64 *intermediate*
chi² (from summation reassociation), which (a) flip no discrete model selection
and (b) are below float32 storage precision, so the saved science outputs are
bitwise-identical. This is far stronger than the requested
"consistent to within Monte Carlo noise."

## What changed (all numerically identical)

`src/brutus/core/sed_utils.py`
- `_get_seds`: serial → `prange` (parallel); `np.zeros`→`np.empty` (all written).

`src/brutus/analysis/individual.py`
- `_optimize_fit_mag`, `_optimize_fit_flux`, `_get_sed_mle`: per-model loops
  serial → `prange`. The two convergence max-reductions in `_optimize_fit_mag`
  parallelized (max is exact & order-independent → bitwise-identical).
- **Dead-code removal** in `_get_sed_mle`: `models_int = 10**(-0.4*mag)` and the
  `reddening` term were computed (4.9M `pow`s per call) but never read. Removed.
- New fused parallel kernels: `_chi2_from_resid` (replaces
  `np.sum(np.square(resid)/tot_var, axis=1)`) and `_init_mag_resid` (fuses the
  initial mag-space `_get_seds` with the `mags - models` residual).
- `np.zeros`→`np.empty` for fully-written arrays.

`src/brutus/utils/math.py`
- `_batch_invert_3x3_preconditioned`: replaced batched `np.linalg.eigvalsh`
  (used only for the smallest-eigenvalue PD regularization check) with an
  analytic symmetric-3×3 min-eigenvalue kernel `_batch_min_eig_sym3` (parallel).
  Agrees with `eigvalsh` to ~5e-14; the regularization threshold (1e-12) decision
  was identical for every matrix in the 60-star end-to-end test (bitwise output).

## Why not more aggressive?

The brute-force evaluation over all 613k grid points is the *design* of the
method; the safe `prange` parallelization captures the bulk of the available
speedup and **scales with physical core count** — the ~2× seen here on 4 cores
would be substantially larger on production many-core hardware. Further gains
would require changing the statistics (earlier model culling, fewer MC samples),
which would trade away the exactness verified above; those were intentionally
not taken.

## Session 2 — statistical efficiency + further compute

After the parallelization above, a second pass targeted statistical efficiency
and any remaining compute, verified with a distributional-equivalence harness
(`bench/disteq.py`) calibrated against the procedure's own seed-to-seed Monte
Carlo scatter, plus a high-Nmc gold-standard accuracy test (`bench/mcgold.py`)
and a direct MC-variance test (`bench/mcvar.py`).

**Antithetic Monte-Carlo integration (kept).** The prior integral in
`logpost_grid` now draws the proposal normals in antithetic pairs `(z, -z)`.
Each sample is still marginally N(mean,cov), so the estimator is unbiased, but:
- **~4.6× lower std (≈20× lower variance)** of the per-model integrated
  log-posterior `lnp` (direct measurement, `mcvar.py`).
- **More accurate, not just different**: vs an Nmc→∞ gold standard, RMSE ratio
  **0.276** (≈3.6× more accurate) and the finite-Nmc Jensen bias of
  `log(mean(w))` shrinks 3–65× — at the *same* Nmc=50, using *half* the Gaussian
  draws.
- Final-posterior shift is within the procedure's intrinsic MC scatter: across a
  cohort spanning the full selected-model range (Nsel = 4 … 50000-cap), the
  antithetic-vs-plain z-profile (frac|z|>3 = 0.07) is no larger than the
  plain-vs-plain null (0.08), and output scatter drops (log-evidence 0.76×).
- Benefit concentrates in well-constrained stars (where MC-integration variance
  dominates); for poorly-constrained stars that hit the `max_models` subsampling
  cap the subsampling variance dominates and antithetic is neutral (no harm).

  *Adversarial review (resolved).* A reviewer initially objected that antithetic
  increases variance for even-symmetric integrands. Reconciliation: the brutus
  integrand is dominated along ln(d) by the galactic-density falloff and the
  +log(d) Jacobian — both monotone (odd) in the sampling direction — which is the
  regime where antithetic *reduces* variance; the reviewer reproduced this and
  withdrew the objection. The one corner where antithetic can mildly *increase*
  variance (never bias) — a very precise parallax centered on the photometric
  proposal mode with poorly-constraining photometry — was spot-checked on real
  high-SNR Orion stars reduced to 4 bands and found negligible (worst std ratio
  ~1.06). `mc_ess` is documented as a weight-concentration diagnostic (not a
  strict independent-sample count) under antithetic correlation.

**Parallel multivariate-normal sampler (kept, exact).** `_sample_multivariate_
normal_jit` now runs `prange` over distributions — bitwise-identical, parallel.

**exp instead of pow (kept, machine-eps).** `_get_seds` flux conversion uses
`exp(fac·mag)` rather than `10**(-0.4·mag)`: ~1.6× faster on that hot path,
agreement ~3e-15 (vanishes under float32 storage; verified machine-eps across
the Nsel = 4 … 50000 cohort).

**Investigated and deliberately NOT taken** (analysis in commit/notes):
- *Early model culling / iteration capping* — measured that the mag-space
  optimizer already converges in ≤3 iterations (the top models have zero
  improvement past iter 3), so capping yields no speedup; the cost is the
  inherent full-grid SED passes.
- *Deferred Fisher (icov) for selected models only* — exact but saves only
  ~20 ms (the icov build is a small part of the MLE) for an invasive
  loglike/logpost API change; not worth the risk.
- *float32 SED/optimize path* — a real but partial (~1.3–1.5×, memory-bound
  parts only) win that requires a full-chain precision change and full
  distributional re-validation; recommended as a separate opt-in change.

## Session 3 — logpost galactic-prior block (large Nsel)

For the default galactic-structure prior, `logpost_grid` was tiling the full
*structured* label array across all `Nmc` MC samples. A numpy structured-dtype
`np.tile` is ~100× slower than the equivalent float tile (~200 ms vs ~2 ms at
Nsel=50000) and was the single biggest logpost cost for poorly-constrained
(large-Nsel) stars. `logp_galactic_structure` now accepts the per-point
`feh`/`loga` as plain float arrays; logpost float-tiles only the fields the
prior uses. **Bitwise-identical** across the full Nsel cohort (every fit() output
0.000e+00, disteq scatter 1.000). Clean A/B (min-of-8): logpost **~1.24×** for
Nsel=50000 (~165 ms/object saved); no change for small Nsel.

Remaining logpost headroom (large-Nsel tail only, deprioritized): the Gaussian
RNG generation (`_antithetic_normals`, intrinsic), the prior math itself
(`_galactic_prior_fused`, intrinsic), and a coordinate-transform fusion (~30 ms
net, machine-eps, needs a shared-kernel refactor). Typical (small-Nsel) stars
already spend only ~25–33 ms in logpost.

## Reproduce

```bash
python bench/harness.py bench            # min-of-N per-object timings
python bench/harness.py capture <tag>    # freeze deterministic + draw outputs
python bench/harness.py compare a b      # regression compare
bash    bench/ab.sh                       # stash-based baseline vs optimized A/B
bash    bench/fullfit_ab.sh               # end-to-end fit() HDF5 bitwise A/B
```
