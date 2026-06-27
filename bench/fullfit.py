#!/usr/bin/env python
"""
End-to-end fit() regression: run the REAL fit() over many Orion stars with a
single threaded RNG (exactly as a user would), write the HDF5, and compare all
datasets between two runs (baseline vs optimized). This is the strongest check:
fit() threads ONE RandomState across all objects, so any change in the RNG draw
sequence anywhere in the pipeline shows up as a divergence here.

Usage:
    python bench/fullfit.py run  <tag>     # run fit() -> bench/artifacts/fit_<tag>.h5
    python bench/fullfit.py cmp  <a> <b>   # compare two HDF5 outputs
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import harness as H  # noqa: E402

NOBJ = 60  # number of Orion stars to fit end-to-end


def run(tag):
    s = H.load_setup()
    bf = s["bf"]
    # Use the first NOBJ stars (mixed quality, mirrors a real catalog slice).
    sl = slice(0, NOBJ)
    data = s["phot"][sl]
    data_err = s["err"][sl]
    data_mask = s["mask"][sl]
    parallax = s["parallax"][sl]
    parallax_err = s["parallax_err"][sl]
    coords = s["coords"][sl]
    labels = np.array([(i,) for i in range(NOBJ)], dtype=[("id", "i8")])
    out = os.path.join(H.OUTDIR, f"fit_{tag}")
    bf.fit(
        data,
        data_err,
        data_mask,
        labels,
        out,
        phot_offsets=s["offsets"],
        parallax=parallax,
        parallax_err=parallax_err,
        data_coords=coords,
        Nmc_prior=50,
        Ndraws=250,
        wt_thresh=1e-3,
        save_dar_draws=True,
        running_io=True,
        verbose=False,
        rstate=np.random.RandomState(12345),
    )
    print(f"wrote {out}.h5")


def cmp(a, b):
    fa = h5py.File(os.path.join(H.OUTDIR, f"fit_{a}.h5"), "r")
    fb = h5py.File(os.path.join(H.OUTDIR, f"fit_{b}.h5"), "r")
    keys = [k for k in fa.keys()]
    worst = 0.0
    print(f"=== fit() HDF5 compare {a} vs {b} ({NOBJ} objects) ===")
    for k in keys:
        x, y = fa[k][:], fb[k][:]
        xf = np.asarray(x, float)
        yf = np.asarray(y, float)
        fin = np.isfinite(xf) & np.isfinite(yf)
        if fin.any():
            d = np.abs(xf[fin] - yf[fin])
            rel = d / np.maximum(np.abs(xf[fin]), 1e-300)
            bit = "Y" if np.array_equal(xf[fin], yf[fin]) else "N"
            worst = max(worst, rel.max())
            print(
                f"  {k:14s} max|abs|={d.max():.3e} max|rel|={rel.max():.3e} bitwise={bit}"
            )
        else:
            print(f"  {k:14s} (no finite overlap)")
    print(f"=== WORST relative error over all fit() datasets: {worst:.3e} ===")
    return worst


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "cmp":
        cmp(sys.argv[2], sys.argv[3])
