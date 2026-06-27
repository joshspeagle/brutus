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

## Reproduce

```bash
python bench/harness.py bench            # min-of-N per-object timings
python bench/harness.py capture <tag>    # freeze deterministic + draw outputs
python bench/harness.py compare a b      # regression compare
bash    bench/ab.sh                       # stash-based baseline vs optimized A/B
bash    bench/fullfit_ab.sh               # end-to-end fit() HDF5 bitwise A/B
```
