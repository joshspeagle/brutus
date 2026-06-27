#!/usr/bin/env python
"""
Full profiling run over the Nsel cohort (tiny .. 50000-cap), current code.

Reports, per cohort object:
  - Nsel (selected models), and per-stage wall time (min-of-N, to suppress the
    shared-host noise) for loglike_grid / logpost_grid / full _fit.
Then an aggregated cProfile over the whole cohort attributing time to the
individual kernels, so we can see exactly where the remaining time goes.
"""

import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402
import disteq as D  # noqa: E402


def stage_timings(repeats=3):
    s = H.load_setup()
    bf = s["bf"]
    ph, er = H._apply_offsets(s)
    cohort = D.COHORT
    i0 = cohort[0]
    H._warmup(
        bf,
        ph[i0],
        er[i0],
        s["mask"][i0],
        s["parallax"][i0],
        s["parallax_err"][i0],
        tuple(s["coords"][i0]),
    )
    rows = []
    for i in cohort:
        i = int(i)
        mk = s["mask"][i]
        par, pe, co = s["parallax"][i], s["parallax_err"][i], tuple(s["coords"][i])
        ll = lp = ft = np.inf
        nsel = 0
        for _ in range(repeats):
            t = time.perf_counter()
            lr = bf.loglike_grid(
                ph[i], er[i], mk, return_vals=True, parallax=par, parallax_err=pe
            )
            ll = min(ll, time.perf_counter() - t)
            t = time.perf_counter()
            r = bf.logpost_grid(
                lr,
                parallax=par,
                parallax_err=pe,
                coord=co,
                Nmc_prior=50,
                wt_thresh=1e-3,
                rstate=np.random.RandomState(i),
            )
            lp = min(lp, time.perf_counter() - t)
            nsel = len(r[0])
            t = time.perf_counter()
            bf._fit(
                ph[i],
                er[i],
                mk,
                parallax=par,
                parallax_err=pe,
                coord=co,
                Nmc_prior=50,
                Ndraws=250,
                wt_thresh=1e-3,
                rstate=np.random.RandomState(i),
            )
            ft = min(ft, time.perf_counter() - t)
        rows.append((i, nsel, ll * 1e3, lp * 1e3, ft * 1e3))
    print("=== per-object stage timing over cohort (min of %d) ===" % repeats)
    print(
        f"{'obj':>4s} {'Nsel':>7s} {'loglike_ms':>11s} {'logpost_ms':>11s} "
        f"{'fit_ms':>9s}"
    )
    for i, nsel, ll, lp, ft in rows:
        print(f"{i:>4d} {nsel:>7d} {ll:11.1f} {lp:11.1f} {ft:9.1f}")
    arr = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"\n{'stage':>10s} {'min':>9s} {'median':>9s} {'max':>9s} {'mean':>9s}")
    for k, name in enumerate(["loglike", "logpost", "fit"]):
        c = arr[:, k]
        print(
            f"{name:>10s} {c.min():9.1f} {np.median(c):9.1f} {c.max():9.1f} "
            f"{c.mean():9.1f}"
        )
    return rows


def aggregate_profile():
    s = H.load_setup()
    bf = s["bf"]
    ph, er = H._apply_offsets(s)
    cohort = D.COHORT

    def workload():
        for i in cohort:
            i = int(i)
            bf._fit(
                ph[i],
                er[i],
                s["mask"][i],
                parallax=s["parallax"][i],
                parallax_err=s["parallax_err"][i],
                coord=tuple(s["coords"][i]),
                Nmc_prior=50,
                Ndraws=250,
                wt_thresh=1e-3,
                rstate=np.random.RandomState(i),
            )

    workload()  # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        workload()
    pr.disable()
    st = io.StringIO()
    pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(22)
    print("\n=== aggregated cProfile over cohort (2 passes, tottime) ===")
    for line in st.getvalue().splitlines():
        if any(
            t in line
            for t in [
                "brutus/",
                "tottime",
                "function calls",
                "method 'reduce'",
                "logsumexp",
                "eigvalsh",
                "method 'normal'",
                "{method 'repeat'",
            ]
        ):
            print(line)


if __name__ == "__main__":
    stage_timings()
    aggregate_profile()
