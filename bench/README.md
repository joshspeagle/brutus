# `bench/` — internal developer tooling (not part of the package)

> **This directory is for brutus developers, not users.** It is **not** part of
> the installed `astro-brutus` package (excluded from both the wheel and the
> sdist — see `MANIFEST.in`), is **not** imported by `brutus`, and is **not** run
> in CI. Nothing here is a supported public API. End users should ignore it; the
> user-facing summary of the performance work lives in `CHANGELOG.md`.

## What this is

A benchmark + correctness-verification harness built for the `BruteForce`
individual-star fitter performance pass (v1.1.1). Its job is to prove that
performance changes to `loglike_grid` / `logpost_grid` / `_fit` either leave the
saved results **bitwise-identical** or are **provably more accurate**, and to
measure the speedups — all on the **real** MIST grid and **real** Orion data
rather than toy inputs.

## Prerequisites (why it's not in CI)

These are manual tools that need large data downloaded locally:

```python
from brutus.data import fetch_grids, fetch_offsets
fetch_grids()              # grid_mist_v9.h5  (~780 MB)
fetch_offsets('mist_v9')   # photometric offsets
# Orion field data ships in tutorials/Orion_l209.1_b-19.9.h5
```

All tools resolve these paths via `harness.py`. Run with JIT **enabled** (do not
set `NUMBA_DISABLE_JIT=1`) so the real parallel codegen is exercised.

## Contents

| file | purpose |
|------|---------|
| `harness.py` | shared setup (loads grid + Orion data); per-object stage timings (`bench`); freeze/compare deterministic loglike outputs + posterior draws (`capture`/`compare`) |
| `disteq.py` | distributional-equivalence tester — multi-seed posterior summaries vs the procedure's own Monte-Carlo scatter; defines the `COHORT` (Nsel = 4 … 50000-cap) |
| `fullfit.py` | end-to-end `fit()` → HDF5, then bitwise-compare two runs (the strongest regression check; threads one RNG across objects) |
| `mcvar.py` | direct measurement of MC-integration variance, antithetic vs plain |
| `mcgold.py` | accuracy of antithetic vs plain against a high-`Nmc` gold standard |
| `profile_cohort.py` | per-stage timings + aggregated `cProfile` over the cohort |
| `*_ab.sh` | A/B the current **working tree** against committed `HEAD` via `git stash` (make an uncommitted change, run, read the speedup + regression/distributional deltas) |
| `RESULTS.md` | the dated findings record: methodology, per-change results, and the investigated-but-deferred ideas |
| `artifacts/` | generated outputs (git-ignored) |

See `RESULTS.md` for the full methodology and numbers, and
`docs/superpowers/specs/` for the project's other dated work-records.
