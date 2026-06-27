#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root (bench/..)
PY_CAP='import bench.harness as h, bench.disteq as d; h.capture("coh_%TAG%", obj_idx=d.COHORT)'
PY_DEQ='import bench.disteq as d; d.run("deq_%TAG%", K=6, objs=d.COHORT)'
echo "=== CANDIDATE: capture + disteq on cohort ==="
python -c "${PY_CAP/\%TAG\%/cand}" 2>/dev/null | tail -1
python -c "${PY_DEQ/\%TAG\%/cand}" 2>/dev/null | tail -1
echo "=== stash -> BASELINE ==="
git stash >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
python -c "${PY_CAP/\%TAG\%/base}" 2>/dev/null | tail -1
python -c "${PY_DEQ/\%TAG\%/base}" 2>/dev/null | tail -1
git stash pop >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
echo ""
echo "############ EXACT loglike (cohort, tiny..50000 Nsel) ############"
python bench/harness.py compare coh_base coh_cand 2>/dev/null | grep -E "WORST"
# also print per-object deterministic worst for the extremes
python -c "
import numpy as np, os
import bench.harness as H
b=np.load(os.path.join(H.OUTDIR,'capture_coh_base.npz')); c=np.load(os.path.join(H.OUTDIR,'capture_coh_cand.npz'))
import bench.disteq as d
for k in range(len(d.COHORT)):
    w=0.0
    for key in ['chi2','scale','icov']:
        a=b[f'o{k}_{key}']; e=c[f'o{k}_{key}']
        fin=np.isfinite(a)&np.isfinite(e)
        if fin.any(): w=max(w, (np.abs(a[fin]-e[fin])/np.maximum(np.abs(a[fin]),1e-300)).max())
    print(f'  cohort obj {d.COHORT[k]:>3d}: det-loglike worst rel = {w:.2e}')
"
echo ""
echo "############ ANTITHETIC distributional (cohort) ############"
python -c "import bench.disteq as d; d.cmp('deq_base','deq_cand')" 2>/dev/null
echo "=== DONE ==="
