#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root (bench/..)
OBJ="import bench.harness as h,numpy as np; print(h.benchmark(obj_idx=np.where(h.load_setup()['mask'].all(axis=1)&np.isfinite(h.load_setup()['parallax']))[0][:12], repeats=3))"
echo "=== OPTIMIZED ==="
python -c "$OBJ" 2>/dev/null | grep -E "loglike|^\{" | tail -1
echo "=== switch to BASELINE (stash) ==="
git stash >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
echo "=== BASELINE ==="
python -c "$OBJ" 2>/dev/null | grep -E "loglike|^\{" | tail -1
git stash pop >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
echo "=== DONE, restored optimized ==="
