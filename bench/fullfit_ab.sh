#!/bin/bash
set -e
cd /home/user/brutus
echo "=== run OPTIMIZED fit() ==="
python bench/fullfit.py run opt 2>/dev/null
echo "=== stash -> BASELINE ==="
git stash >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
# fullfit.py is untracked (in bench/), survives stash; harness too
python bench/fullfit.py run base 2>/dev/null
git stash pop >/dev/null 2>&1
find . -name __pycache__ -path '*brutus*' -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/numba_cache
echo "=== COMPARE ==="
python bench/fullfit.py cmp base opt 2>/dev/null
echo "=== DONE ==="
