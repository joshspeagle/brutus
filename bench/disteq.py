#!/usr/bin/env python
"""
Distributional-equivalence tester for non-exact pipeline changes.

For changes that legitimately alter the RNG stream or introduce ~float-noise
(float32, antithetic sampling, early culling), bitwise comparison is the wrong
bar. The right bar is: the change must shift posterior summaries by LESS than the
intrinsic Monte-Carlo scatter of the procedure.

Protocol
--------
Run the full per-object `_fit` over a fixed object set with K independent RNG
seeds, and reduce each object's 250 posterior draws to summary statistics
(distance/Av/Rv medians + spreads, log-evidence, chi2min). Do this for both the
baseline and the candidate code (separate processes via git stash). Then for each
(object, statistic):

    z = (mean_cand - mean_base) / SE_base ,   SE_base = std_base / sqrt(K)

If the candidate is statistically identical, the per-statistic z-scores behave
like standard normals (|z| ~ O(1)); a real bias shows up as |z| >> 1 systematically.
We also report the candidate's own seed-to-seed scatter vs the baseline's, and the
shift measured in units of the posterior width (practical significance).

Usage
-----
    python bench/disteq.py run <tag> [K] [Nobj]   # -> artifacts/disteq_<tag>.npz
    python bench/disteq.py cmp <base> <cand>
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402

STATS = ["dist_med", "dist_iqr", "av_med", "av_iqr", "rv_med", "levid", "chi2min"]


def _summarize(dists, reds, dreds, levid, chi2min):
    def med(x):
        return float(np.median(x))

    def iqr(x):
        return float(np.subtract(*np.percentile(x, [84, 16])))

    return np.array(
        [
            med(dists),
            iqr(dists),
            med(reds),
            iqr(reds),
            med(dreds),
            float(levid),
            float(chi2min),
        ]
    )


# Deliberate cohort spanning the selected-model-count (Nsel) range so that BOTH
# the tiny-Nsel edge cases (Nsel < Ndraws) and the large-Nsel path that triggers
# max_models=50000 subsampling (Gumbel-max) are exercised. Indices are into the
# full-coverage+parallax object set of the Orion field.
COHORT = [13, 14, 25, 60, 0, 2, 3, 5, 9, 29, 39, 52]


def run(tag, K=8, Nobj=24, seed0=0, objs=None):
    s = H.load_setup()
    bf = s["bf"]
    phot_all, err_all = H._apply_offsets(s)
    if objs is not None:
        good = [int(i) for i in objs]
    else:
        good = np.where(s["mask"].all(axis=1) & np.isfinite(s["parallax"]))[0][:Nobj]
        good = [int(i) for i in good]
    i0 = good[0]
    H._warmup(
        bf,
        phot_all[i0],
        err_all[i0],
        s["mask"][i0],
        s["parallax"][i0],
        s["parallax_err"][i0],
        tuple(s["coords"][i0]),
    )

    out = np.zeros((K, len(good), len(STATS)))
    for t in range(K):
        for j, i in enumerate(good):
            ph, er, mk = phot_all[i], err_all[i], s["mask"][i]
            par, pe, co = s["parallax"][i], s["parallax_err"][i], tuple(s["coords"][i])
            res = bf._fit(
                ph,
                er,
                mk,
                parallax=par,
                parallax_err=pe,
                coord=co,
                Nmc_prior=50,
                Ndraws=250,
                wt_thresh=1e-3,
                rstate=np.random.RandomState(7919 * (t + seed0) + i),
                return_distreds=True,
            )
            (
                idxs,
                scales,
                avs,
                rvs,
                covs,
                Ndim,
                lnprob,
                levid,
                chi2min,
                dists,
                reds,
                dreds,
                logwts,
                mc_ess,
            ) = res
            out[t, j] = _summarize(dists, reds, dreds, levid, chi2min)
    path = os.path.join(H.OUTDIR, f"disteq_{tag}.npz")
    np.savez_compressed(path, summ=out, obj=np.array(good))
    print(f"wrote {path}  shape={out.shape}")
    return path


def cmp(base, cand):
    b = np.load(os.path.join(H.OUTDIR, f"disteq_{base}.npz"))["summ"]  # (K,Nobj,S)
    c = np.load(os.path.join(H.OUTDIR, f"disteq_{cand}.npz"))["summ"]
    K = b.shape[0]
    mb, sb = b.mean(0), b.std(0, ddof=1)
    mc, sc = c.mean(0), c.std(0, ddof=1)
    se = sb / np.sqrt(K)
    # A statistic is effectively DETERMINISTIC (RNG-independent, e.g. chi2min)
    # when its seed-to-seed scatter is negligible vs its magnitude. For those the
    # z-score (shift / ~0 SE) is meaningless and explodes; equivalence must be
    # judged by the relative shift instead (which should be ~machine epsilon).
    deterministic = sb <= 1e-9 * np.maximum(np.abs(mb), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (mc - mb) / se, 0.0)
        z = np.where(deterministic, 0.0, z)  # don't report meaningless z
        # mean shift in units of the posterior spread (practical significance)
        rel = np.where(np.abs(mb) > 0, (mc - mb) / np.maximum(np.abs(mb), 1e-30), 0.0)
    # Per-object scatter ratio: candidate seed-to-seed std / baseline std.
    # <1 means the candidate is LESS noisy (variance reduction, e.g. antithetic).
    with np.errstate(divide="ignore", invalid="ignore"):
        scatter_ratio = np.where(sb > 0, sc / sb, np.nan)
    print(f"=== distributional equivalence: {base} vs {cand}  (K={K} seeds) ===")
    print(
        f"{'stat':9s} {'|z|.mean':>9s} {'|z|.max':>8s} {'|relshift|.max':>14s} "
        f"{'scatter(cand/base)':>19s}"
    )
    for si, name in enumerate(STATS):
        zz = np.abs(z[:, si])
        rr = np.abs(rel[:, si])
        sr = scatter_ratio[:, si]
        sr = sr[np.isfinite(sr)]
        srmed = np.median(sr) if sr.size else np.nan
        det = "  (deterministic)" if deterministic[:, si].all() else ""
        print(
            f"{name:9s} {zz.mean():9.2f} {zz.max():8.2f} {rr.max():14.3e} "
            f"{srmed:19.3f}{det}"
        )
    # Overall: fraction of |z|>3 (would-be outliers if truly equivalent ~0.3%)
    frac = float((np.abs(z) > 3).mean())
    print(
        f"\nfraction |z|>3 : {frac:.3f}  (expect ~0.003 if equivalent; "
        f"high => systematic bias)"
    )
    print(f"max |z| overall: {np.abs(z).max():.2f}")
    return np.abs(z).max()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(
            sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 8,
            int(sys.argv[4]) if len(sys.argv) > 4 else 24,
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
        )
    elif sys.argv[1] == "cmp":
        cmp(sys.argv[2], sys.argv[3])
