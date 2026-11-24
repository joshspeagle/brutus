#!/usr/bin/env python
"""
Accuracy and speed of fit() on a broad set of stars.

The Orion cohort of disteq.py shares one sightline, and every star in it has
a parallax. Approximations of the Galactic prior can pass there and fail
toward the bulge, at high latitude, for giants without a parallax, or with
proper motions. This tool adds stars from the validation samples of #335:
Gaia DR3, GeDR3mock and a mock drawn from the prior, 4 fields each, fitted
from Gaia and 2MASS photometry with the Bayestar prior and no parallax. Each
validation star is fitted twice: without its motion, and with its proper
motion (and the radial velocity, for Gaia DR3).

Each run fits every star with one seed and saves the summaries of
disteq.py. `cmp` compares the mean summaries of two tags over their seeds by
Welch t-tests, flagged below a Bonferroni-corrected p of 0.05, tests each
summary for a shift common to all stars, and compares the median fit
times. With few seeds the seed-to-seed variance is itself uncertain, so the
t-test, not |z| > 3, sets the flags.

Usage
-----
    python bench/broad.py build <stars.npz> <sample.npz> ...  # pick the stars
    python bench/broad.py run <tag> <seed>                    # one seed
    python bench/broad.py cmp <base> <cand>
    BASE_REF=<ref> ROUNDS=<K> bench/broad_ab.sh               # A/B, interleaved

Each sample file holds, per star: ``mag`` and ``mag_err`` (N, 6) in the
order of ``filters``, ``l``, ``b``, ``field``, ``pmra``, ``pmdec``,
``pmra_error``, ``pmdec_error``, ``pmra_pmdec_corr``, ``rv`` and ``rv_err``.
The stars file is read from ``$BRUTUS_BENCH_STARS``, by default
``bench/broad_stars.npz``: 48 stars, 4 from each field of the three samples
of #335 (Gaia DR3 by its ADQL there, GeDR3mock, and the mock).
"""

import glob
import os
import sys
import time
import warnings

import h5py
import numpy as np
from scipy import stats

from brutus.dust import Bayestar
from brutus.utils import inv_magnitude

sys.path.insert(0, os.path.dirname(__file__))
import disteq as D  # noqa: E402
import harness as H  # noqa: E402

STARS = os.environ.get(
    "BRUTUS_BENCH_STARS", os.path.join(os.path.dirname(__file__), "broad_stars.npz")
)
DUST = os.path.join(H.CACHE, "bayestar2019_v1.h5")
COLUMNS = ("mag", "mag_err", "l", "b", "field", "pmra", "pmdec", "pmra_error")
COLUMNS += ("pmdec_error", "pmra_pmdec_corr", "rv", "rv_err")
PER_FIELD = 4


def build(out, samples):
    """Keep the first PER_FIELD stars of each field of each sample."""
    picked = {k: [] for k in COLUMNS + ("sample", "use_rv")}
    for k, path in enumerate(samples):
        s = dict(np.load(path))
        if "pm_err" in s:  # the mock: one error for both components
            s["pmra_error"] = s["pmdec_error"] = s["pm_err"]
            s["pmra_pmdec_corr"] = np.zeros(len(s["l"]))
        for f in np.unique(s["field"]):
            idx = np.flatnonzero(s["field"] == f)[:PER_FIELD]
            for c in COLUMNS:
                picked[c].append(s[c][idx])
            picked["sample"].append(np.full(len(idx), k))
            # Radial velocities only for the first sample (Gaia DR3), so
            # that both the 2- and 3-observable paths run.
            picked["use_rv"].append(np.full(len(idx), k == 0))
    np.savez(
        out,
        filters=np.load(samples[0])["filters"],
        names=np.array([os.path.basename(p) for p in samples]),
        **{c: np.concatenate(v) for c, v in picked.items()},
    )
    print(f"wrote {out}")


def _fit_summaries(bf, path, data, **kwargs):
    """Fit, time and summarize every object of `data`."""
    t0 = time.perf_counter()
    bf.fit(*data, path, Nmc_prior=50, Ndraws=250, verbose=False, **kwargs)
    dt = time.perf_counter() - t0
    with h5py.File(path + ".h5", "r") as f:
        cols = [f[k][()] for k in ("samps_dist", "samps_red", "samps_dred")]
        levid, chi2 = f["obj_log_evid"][()], f["obj_chi2min"][()]
    summ = [
        D._summarize(*(c[i] for c in cols), levid[i], chi2[i])
        for i in range(len(levid))
    ]
    return np.array(summ), dt


def run(tag, seed):
    s = H.load_setup()
    orion = [int(i) for i in D.COHORT]
    ph, er = H._apply_offsets(s)
    stars = dict(np.load(STARS))
    n = len(stars["l"])
    bf_val, offsets = H.make_fitter(list(stars["filters"]))
    flux, ferr = inv_magnitude(stars["mag"], stars["mag_err"])
    mask = np.isfinite(flux) & np.isfinite(ferr)
    val = (
        np.nan_to_num(flux, nan=1.0),
        np.nan_to_num(ferr, nan=1.0),
        mask,
        np.arange(n),
    )
    coords = np.c_[stars["l"], stars["b"]]
    rv = np.where(stars["use_rv"], stars["rv"], np.nan)
    motion = {k: stars[k] for k in COLUMNS[5:10]}
    motion.update(
        radial_velocity=rv,
        radial_velocity_error=np.where(np.isnan(rv), np.nan, stars["rv_err"]),
    )
    # Read the dust map once, so the times measure the fits.
    common = dict(data_coords=coords, dustfile=Bayestar(DUST), phot_offsets=offsets)
    out = os.path.join(H.OUTDIR, f"broad_{tag}_s{seed}")
    groups = [
        (
            "orion",
            s["bf"],
            (ph[orion], er[orion], s["mask"][orion], np.arange(len(orion))),
            dict(
                parallax=s["parallax"][orion],
                parallax_err=s["parallax_err"][orion],
                data_coords=s["coords"][orion],
            ),
        ),
        ("val_nopm", bf_val, val, common),
        ("val_pm", bf_val, val, dict(common, **motion)),
    ]
    summ, times = {}, {}
    for g, (name, bf, data, kwargs) in enumerate(groups):
        rstate = np.random.RandomState(7919 * seed + g)
        summ[name], times[name] = _fit_summaries(
            bf, f"{out}_{name}", data, rstate=rstate, **kwargs
        )
        print(f"{name}: {times[name]:.1f} s for {len(data[0])} stars", flush=True)
    np.savez(
        f"{out}.npz",
        **{f"summ_{k}": v for k, v in summ.items()},
        **{f"time_{k}": v for k, v in times.items()},
    )
    for path in glob.glob(f"{out}_*.h5"):
        os.remove(path)


def _load(tag):
    runs = [
        np.load(p)
        for p in sorted(glob.glob(os.path.join(H.OUTDIR, f"broad_{tag}_s*.npz")))
    ]
    if not runs:
        raise SystemExit(f"no runs for tag {tag!r} in {H.OUTDIR}")
    names = [k[5:] for k in runs[0].files if k.startswith("summ_")]
    summ = {k: np.array([r[f"summ_{k}"] for r in runs]) for k in names}
    times = {k: np.array([float(r[f"time_{k}"]) for r in runs]) for k in names}
    return summ, times


def cmp(base, cand):
    """Welch t-test of each (star, summary) mean across the seeds, flagged
    below a Bonferroni-corrected p of 0.05, and a bias test per summary: the
    mean over stars of the shift in units of its standard error, which is
    near 0 within sqrt(2 / Nstar) when the two sides agree."""
    sb, tb = _load(base)
    sc, tc = _load(cand)
    tests = sum(b.shape[1] for b in sb.values()) * len(D.STATS)
    alpha = 0.05 / tests
    worst = 1.0
    print(f"=== broad check: {base} vs {cand} (flag p < {alpha:.1e}) ===")
    for name in sb:
        b, c = sb[name], sc[name]  # (K, Nstar, Nstat)
        shift = c.mean(0) - b.mean(0)
        se = np.sqrt(b.var(0, ddof=1) / len(b) + c.var(0, ddof=1) / len(c))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t, p = stats.ttest_ind(c, b, axis=0, equal_var=False)
        # chi2min does not depend on the seed: judge it by the shift alone.
        fixed = se <= 1e-9 * np.maximum(np.abs(b.mean(0)), 1e-30)
        t, p = np.where(fixed, 0.0, t), np.where(fixed, 1.0, p)
        worst = min(worst, p.min())
        print(f"-- {name}: K = {len(b)} / {len(c)} seeds, {b.shape[1]} stars")
        before, after = np.median(tb[name]), np.median(tc[name])
        print(
            f"   time: median {before:.1f} s -> {after:.1f} s ({after / before:.2f}x)"
        )
        for si, stat in enumerate(D.STATS):
            z = stats.norm.isf(p[:, si] / 2) * np.sign(t[:, si])  # t as a normal z
            bias = z.mean() * np.sqrt(len(z))
            print(
                f"   {stat:11s} p<0.01: {np.mean(p[:, si] < 0.01):.2f} (0.01 expected)"
                f"   bias z {bias:+5.2f}   max |shift| {np.abs(shift[:, si]).max():.3e}"
            )
        for star, si in np.argwhere(p < alpha):
            print(
                f"   FLAG star {star} {D.STATS[si]}: p = {p[star, si]:.1e}, "
                f"{b.mean(0)[star, si]:.4g} -> {c.mean(0)[star, si]:.4g}"
            )
    print(f"=== smallest p: {worst:.1e} (flag below {alpha:.1e}) ===")
    return worst


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "run":
        H.print_setup()
        run(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "cmp":
        cmp(sys.argv[2], sys.argv[3])
