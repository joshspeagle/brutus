#!/usr/bin/env python
"""
Verification + benchmark harness for the brutus BruteForce pipeline.

Uses the REAL MIST grid (grid_mist_v9.h5, ~613k models) and REAL data
(the Orion field, 207 stars w/ parallax) with the realistic Orion-tutorial
filter set (Pan-STARRS griz y + 2MASS JHKs, 8 bands) and photometric offsets.

Provides:
  * load_setup()            -> grid + data, cached
  * benchmark(...)          -> per-object timings of loglike/logpost/_fit
  * capture(tag, ...)       -> freeze deterministic loglike arrays + posterior
                              draws for a fixed object subset, with a
                              deterministic per-object RNG seed.
  * compare(base, cand)     -> identity check on deterministic loglike outputs
                              and (seed-matched) posterior draws.

Verification philosophy
-----------------------
loglike_grid is RNG-free => its outputs (lnl, chi2, scale, av, rv, icov_sar)
are deterministic and must match the baseline. Pure code-reorganization
(prange, loop fusion, precompute) is expected BITWISE identical; numeric
changes are checked at a tight rtol. logpost_grid/_fit draw RNG; with a fixed
per-object seed they are bit-identical iff the RNG draw sequence is preserved,
otherwise distributional-equivalence is required (see dist_equiv.py).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

GRID = os.path.expanduser("~/.cache/astro-brutus/grid_mist_v9.h5")
OFFSETS = os.path.expanduser("~/.cache/astro-brutus/offsets_mist_v9.txt")
ORION = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tutorials", "Orion_l209.1_b-19.9.h5"
)
OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(OUTDIR, exist_ok=True)

_SETUP = None


def load_setup():
    global _SETUP
    if _SETUP is not None:
        return _SETUP
    import h5py

    from brutus.analysis import BruteForce
    from brutus.core import StarGrid
    from brutus.data import filters, load_models, load_offsets
    from brutus.utils import inv_magnitude

    filt = filters.ps[:-2] + filters.tmass  # PS grizy + 2MASS JHKs (8 bands)
    models, labels, label_mask = load_models(GRID, filters=filt, verbose=False)
    grid_names = [n for n, g in zip(labels.dtype.names, label_mask[0]) if g]
    pred_names = [n for n, g in zip(labels.dtype.names, label_mask[0]) if not g]
    sg = StarGrid(models, labels[grid_names], labels[pred_names])
    bf = BruteForce(sg, verbose=False)
    offsets = load_offsets(OFFSETS, filters=filt, verbose=False)

    with h5py.File(ORION, "r") as f:
        fpix = f["photometry"]["pixel 0-0"]
        mag, magerr = fpix["mag"][:], fpix["err"][:]
        mask = np.isfinite(magerr)
        phot, err = inv_magnitude(mag, magerr)
        parallax = fpix["parallax"][:].astype(float)
        parallax_err = fpix["parallax_error"][:].astype(float)
        psel = (
            np.isclose(parallax_err, 0.0)
            | np.isclose(parallax, 0.0)
            | (parallax_err > 1e6)
        )
        parallax[psel], parallax_err[psel] = np.nan, np.nan
        coords = np.c_[fpix["l"][:], fpix["b"][:]]

    _SETUP = dict(
        bf=bf,
        phot=phot,
        err=err,
        mask=mask,
        parallax=parallax,
        parallax_err=parallax_err,
        coords=coords,
        offsets=offsets,
        filt=filt,
    )
    return _SETUP


def _apply_offsets(s):
    """Apply photometric offsets once (mirrors BruteForce._setup)."""
    phot = s["phot"] * s["offsets"]
    err = s["err"] * s["offsets"]
    return phot, err


def _warmup(bf, phot, err, mask, par, parerr, coord):
    """Trigger numba JIT compilation so timings are steady-state."""
    lr = bf.loglike_grid(
        phot, err, mask, return_vals=True, parallax=par, parallax_err=parerr
    )
    bf.logpost_grid(
        lr,
        parallax=par,
        parallax_err=parerr,
        coord=coord,
        Nmc_prior=50,
        wt_thresh=1e-3,
        rstate=np.random.RandomState(0),
    )
    bf._fit(
        phot,
        err,
        mask,
        parallax=par,
        parallax_err=parerr,
        coord=coord,
        Nmc_prior=50,
        Ndraws=250,
        wt_thresh=1e-3,
        rstate=np.random.RandomState(0),
    )


def benchmark(obj_idx=None, repeats=5, Nmc_prior=50, Ndraws=250, wt_thresh=1e-3):
    """Per-object timings using a MIN-of-repeats estimator per object.

    On a shared/noisy host the minimum over repeats is the cleanest estimate of
    true compute time (least contaminated by co-tenant interference). We time
    each object `repeats` times and keep the min per object, then average the
    per-object minima across the object sample.
    """
    s = load_setup()
    bf = s["bf"]
    phot_all, err_all = _apply_offsets(s)
    if obj_idx is None:
        good = np.where(s["mask"].all(axis=1) & np.isfinite(s["parallax"]))[0]
        obj_idx = good[:30]
    obj_idx = [int(i) for i in obj_idx]
    i0 = obj_idx[0]
    _warmup(
        bf,
        phot_all[i0],
        err_all[i0],
        s["mask"][i0],
        s["parallax"][i0],
        s["parallax_err"][i0],
        tuple(s["coords"][i0]),
    )

    ll = {i: np.inf for i in obj_idx}
    lp = {i: np.inf for i in obj_idx}
    ft = {i: np.inf for i in obj_idx}
    for rep in range(repeats):
        for i in obj_idx:
            ph, er, mk = phot_all[i], err_all[i], s["mask"][i]
            par, pe, co = s["parallax"][i], s["parallax_err"][i], tuple(s["coords"][i])
            t = time.perf_counter()
            lr = bf.loglike_grid(
                ph, er, mk, return_vals=True, parallax=par, parallax_err=pe
            )
            ll[i] = min(ll[i], time.perf_counter() - t)
            t = time.perf_counter()
            bf.logpost_grid(
                lr,
                parallax=par,
                parallax_err=pe,
                coord=co,
                Nmc_prior=Nmc_prior,
                wt_thresh=wt_thresh,
                rstate=np.random.RandomState(i),
            )
            lp[i] = min(lp[i], time.perf_counter() - t)
            t = time.perf_counter()
            bf._fit(
                ph,
                er,
                mk,
                parallax=par,
                parallax_err=pe,
                coord=co,
                Nmc_prior=Nmc_prior,
                Ndraws=Ndraws,
                wt_thresh=wt_thresh,
                rstate=np.random.RandomState(i),
            )
            ft[i] = min(ft[i], time.perf_counter() - t)
    n = len(obj_idx)
    return dict(
        n=n,
        loglike_ms=1e3 * sum(ll.values()) / n,
        logpost_ms=1e3 * sum(lp.values()) / n,
        fit_ms=1e3 * sum(ft.values()) / n,
    )


def capture(tag, obj_idx=None, Nmc_prior=50, Ndraws=250, wt_thresh=1e-3):
    """Freeze deterministic loglike arrays + posterior draws for comparison."""
    s = load_setup()
    bf = s["bf"]
    phot_all, err_all = _apply_offsets(s)
    if obj_idx is None:
        good = np.where(s["mask"].all(axis=1) & np.isfinite(s["parallax"]))[0]
        obj_idx = good[:6]
    obj_idx = [int(i) for i in obj_idx]
    i0 = obj_idx[0]
    _warmup(
        bf,
        phot_all[i0],
        err_all[i0],
        s["mask"][i0],
        s["parallax"][i0],
        s["parallax_err"][i0],
        tuple(s["coords"][i0]),
    )

    out = dict(obj_idx=np.array(obj_idx))
    for k, i in enumerate(obj_idx):
        ph, er, mk = phot_all[i], err_all[i], s["mask"][i]
        par, pe, co = s["parallax"][i], s["parallax_err"][i], tuple(s["coords"][i])
        # Deterministic loglike outputs (RNG-free)
        lnl, ndim, chi2, scale, av, rv, icov = bf.loglike_grid(
            ph, er, mk, return_vals=True, parallax=par, parallax_err=pe
        )
        out[f"o{k}_lnl"] = lnl.astype(np.float64)
        out[f"o{k}_chi2"] = chi2.astype(np.float64)
        out[f"o{k}_scale"] = scale.astype(np.float64)
        out[f"o{k}_av"] = av.astype(np.float64)
        out[f"o{k}_rv"] = rv.astype(np.float64)
        out[f"o{k}_icov"] = icov.astype(np.float64)
        # Posterior draws via _fit with fixed per-object seed
        res = bf._fit(
            ph,
            er,
            mk,
            parallax=par,
            parallax_err=pe,
            coord=co,
            Nmc_prior=Nmc_prior,
            Ndraws=Ndraws,
            wt_thresh=wt_thresh,
            rstate=np.random.RandomState(1000 + i),
            return_distreds=True,
        )
        (
            idxs,
            scales,
            avs,
            rvs,
            covs,
            Ndim2,
            lnprob,
            levid,
            chi2min,
            dists,
            reds,
            dreds,
            logwts,
            mc_ess,
        ) = res
        out[f"o{k}_idxs"] = idxs.astype(np.int64)
        out[f"o{k}_levid"] = np.array(levid, np.float64)
        out[f"o{k}_chi2min"] = np.array(chi2min, np.float64)
        out[f"o{k}_dists"] = dists.astype(np.float64)
        out[f"o{k}_reds"] = reds.astype(np.float64)
        out[f"o{k}_dreds"] = dreds.astype(np.float64)
        out[f"o{k}_scales"] = scales.astype(np.float64)
    path = os.path.join(OUTDIR, f"capture_{tag}.npz")
    np.savez_compressed(path, **out)
    print(f"captured {len(obj_idx)} objects -> {path}")
    return path


def _stats(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return f"  {name}: all-nonfinite"
    da = np.abs(a[fin] - b[fin])
    denom = np.maximum(np.abs(a[fin]), 1e-300)
    rel = da / denom
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    return (
        f"  {name}: max|abs|={da.max():.3e} max|rel|={rel.max():.3e} "
        f"bitwise={'Y' if np.array_equal(a, b) or (np.array_equal(a[fin],b[fin]) and nan_mismatch==0) else 'N'} "
        f"nanmis={nan_mismatch}"
    )


def compare(base_tag, cand_tag):
    base = np.load(os.path.join(OUTDIR, f"capture_{base_tag}.npz"))
    cand = np.load(os.path.join(OUTDIR, f"capture_{cand_tag}.npz"))
    nobj = len(base["obj_idx"])
    print(f"=== compare {base_tag} vs {cand_tag} ({nobj} objects) ===")
    det_keys = ["lnl", "chi2", "scale", "av", "rv", "icov"]
    draw_keys = ["idxs", "levid", "chi2min", "dists", "reds", "dreds", "scales"]
    worst_det = 0.0
    for k in range(nobj):
        print(f"-- object {k} (idx {int(base['obj_idx'][k])}) --")
        print("  [deterministic loglike]")
        for key in det_keys:
            a, b = base[f"o{k}_{key}"], cand[f"o{k}_{key}"]
            line = _stats(a, b, key)
            print(line)
            fin = np.isfinite(a) & np.isfinite(b)
            if fin.any():
                rel = np.abs(a[fin] - b[fin]) / np.maximum(np.abs(a[fin]), 1e-300)
                worst_det = max(worst_det, rel.max())
        print("  [posterior draws, seed-matched]")
        for key in draw_keys:
            print(_stats(base[f"o{k}_{key}"], cand[f"o{k}_{key}"], key))
    print(
        f"\n=== WORST deterministic relative error across all objects: {worst_det:.3e} ==="
    )
    return worst_det


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "bench":
        print(benchmark())
    elif cmd == "capture":
        capture(sys.argv[2])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
