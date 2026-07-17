#!/usr/bin/env python
"""
Benchmark for the isochrone population-fitting pipeline.

Reproduces the four tables in ``docs/source/population_modeling.rst``
("Performance Considerations"): the pipeline-stage breakdown, the scaling
with number of stars, and the EEP / SMF convergence studies. Run on an
otherwise-idle machine; wall-clock numbers are min-of-N over repeats.

Methodology (matching the documented setup):
- 3 Gaia bands (G, BP, RP), synthetic coeval cluster at 1 kpc drawn from
  the population grid itself (feh=0.0, loga=9.0, A_V=0.2, R_V=3.3) with
  2% flux errors, fixed RNG seed.
- Times are per evaluation of the full log-likelihood (or per stage).
- Convergence is the change in total log-likelihood at the true theta
  versus a high-resolution reference grid, for the same 100 stars.

Usage:
    python bench/bench_populations.py            # full run (~10-20 min)
    python bench/bench_populations.py --smoke    # small sanity run
"""

import argparse
import sys
import time

import numpy as np

from brutus.analysis.populations import (
    apply_isochrone_mixture_model,
    compute_isochrone_cluster_loglike,
    compute_isochrone_outlier_loglike,
    generate_isochrone_population_grid,
    isochrone_population_loglike,
    marginalize_isochrone_grid,
)
from brutus.core import Isochrone, StellarPop

FILTERS = ["Gaia_G_MAW", "Gaia_BP_MAWf", "Gaia_RP_MAW"]
THETA = dict(feh=0.0, loga=9.0, av=0.2, rv=3.3, dist=1000.0)
FIELD_FRAC = 0.05
CLUSTER_PROB = 0.95
SEED = 42


def timeit(fn, repeats=5):
    """Min-of-N wall-clock time in ms (min is robust to scheduling noise)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return min(times), out


def make_stars(pop, nstars, seed=SEED):
    """Draw synthetic cluster members from the population grid itself."""
    grid = generate_isochrone_population_grid(pop, **THETA)
    flux = np.asarray(grid["photometry"])
    valid = np.all(np.isfinite(flux) & (flux > 0), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.where(valid)[0], size=nstars, replace=True)
    truth = flux[idx]
    err = 0.02 * truth
    obs = truth + rng.normal(size=truth.shape) * err
    return obs, err


def total_loglike(pop, obs, err, smf_grid=None, eep_grid=None):
    theta = [THETA["feh"], THETA["loga"], THETA["av"], THETA["rv"],
             THETA["dist"], FIELD_FRAC]
    return isochrone_population_loglike(
        theta, pop, obs, err, cluster_prob=CLUSTER_PROB,
        smf_grid=smf_grid, eep_grid=eep_grid,
    )


def bench_stages(pop, obs, err, repeats):
    print("\n== Pipeline stage breakdown "
          f"({len(obs)} stars, default grid, min of {repeats}) ==")
    t_grid, grid = timeit(
        lambda: generate_isochrone_population_grid(pop, **THETA), repeats)
    t_clus, lnl_c = timeit(
        lambda: compute_isochrone_cluster_loglike(
            obs, err, grid, distance=THETA["dist"]), repeats)
    t_outl, lnl_o = timeit(
        lambda: compute_isochrone_outlier_loglike(obs, err, grid), repeats)
    t_mix, lnl_m = timeit(
        lambda: apply_isochrone_mixture_model(
            lnl_c, lnl_o, CLUSTER_PROB, FIELD_FRAC), repeats)
    t_marg, _ = timeit(
        lambda: marginalize_isochrone_grid(
            lnl_m, grid["mass_jacobians"], grid["smf_jacobians"]), repeats)
    t_tot, _ = timeit(lambda: total_loglike(pop, obs, err), repeats)

    stages = [("Grid generation (fixed cost)", t_grid),
              ("Cluster loglike", t_clus),
              ("Outlier loglike", t_outl),
              ("Mixture model", t_mix),
              ("Marginalization", t_marg)]
    ssum = sum(t for _, t in stages)
    for name, t in stages:
        print(f"  {name:32s} {t:8.1f} ms   {100 * t / ssum:4.0f}%")
    print(f"  {'Sum of stages':32s} {ssum:8.1f} ms")
    print(f"  {'Total (isochrone_population_loglike)':32s} {t_tot:8.1f} ms")
    print(f"  Grid points: {len(grid['masses'])}")
    return grid


def bench_nstars(pop, obs, err, counts, repeats):
    print(f"\n== Scaling with number of stars (min of {repeats}) ==")
    rng = np.random.default_rng(SEED + 1)
    for n in counts:
        if n <= len(obs):
            o, e = obs[:n], err[:n]
        else:
            idx = rng.choice(len(obs), size=n, replace=True)
            o, e = obs[idx], err[idx]
        t, _ = timeit(lambda: total_loglike(pop, o, e), repeats)
        print(f"  N={n:4d}: {t:8.1f} ms   ({t / n:5.2f} ms/star)")


def bench_eep(pop, obs, err, sizes, ref_size, repeats):
    print(f"\n== EEP convergence vs {ref_size}-point reference "
          f"({len(obs)} stars, min of {repeats}) ==")
    results = {}
    for n in sizes + [ref_size]:
        if n in results:
            continue
        eep = np.linspace(202.0, 808.0, n)
        grid = generate_isochrone_population_grid(pop, **THETA, eep_grid=eep)
        t, lnl = timeit(
            lambda e=eep: total_loglike(pop, obs, err, eep_grid=e), repeats)
        results[n] = (len(grid["masses"]), lnl, t)
    ref_lnl = results[ref_size][1]
    for n in sizes + [ref_size]:
        size, lnl, t = results[n]
        tag = " (reference)" if n == ref_size else (
            " (default)" if n == 1000 else "")
        print(f"  N_EEP={n:5d}{tag:13s} grid={size:6d}  "
              f"dlnL={lnl - ref_lnl:+8.2f}  {t:8.1f} ms")


def bench_smf(pop, obs, err, counts, ref_count, repeats):
    print(f"\n== SMF convergence vs {ref_count}-point uniform reference "
          f"({len(obs)} stars, min of {repeats}) ==")
    results = {}
    for n in counts + [ref_count]:
        if n in results:
            continue
        smf = np.array([0.0]) if n == 1 else np.linspace(0.0, 1.0, n)
        grid = generate_isochrone_population_grid(pop, **THETA, smf_grid=smf)
        t, lnl = timeit(
            lambda s=smf: total_loglike(pop, obs, err, smf_grid=s), repeats)
        results[n] = (len(grid["masses"]), lnl, t)
    ref_lnl = results[ref_count][1]
    for n in counts + [ref_count]:
        size, lnl, t = results[n]
        label = "Singles only (N=1)" if n == 1 else f"{n} uniform"
        tag = " (reference)" if n == ref_count else (
            " (default)" if n == 21 else "")
        print(f"  {label:18s}{tag:13s} grid={size:6d}  "
              f"dlnL={lnl - ref_lnl:+8.2f}  {t:8.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="small fast sanity run (numbers not publishable)")
    args = ap.parse_args()

    print(f"filters={FILTERS}  theta={THETA}  seed={SEED}")
    iso = Isochrone(verbose=False)
    pop = StellarPop(iso, filters=FILTERS, verbose=False)

    if args.smoke:
        nstars, repeats = 10, 2
        eep_sizes, eep_ref = [200], 500
        smf_counts, smf_ref = [1, 7], 11
        nstar_counts = [10]
    else:
        nstars, repeats = 100, 5
        eep_sizes, eep_ref = [200, 500, 1000, 2000], 5000
        smf_counts, smf_ref = [1, 7, 15, 21], 31
        nstar_counts = [10, 50, 100, 200, 500]

    obs, err = make_stars(pop, nstars)
    # Warm-up (JIT compilation, lazy loads) before any timed section.
    total_loglike(pop, obs[:5], err[:5])

    bench_stages(pop, obs, err, repeats)
    bench_nstars(pop, obs, err, nstar_counts, repeats)
    bench_eep(pop, obs, err, eep_sizes, eep_ref, max(3, repeats - 2))
    bench_smf(pop, obs, err, smf_counts, smf_ref, max(3, repeats - 2))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
