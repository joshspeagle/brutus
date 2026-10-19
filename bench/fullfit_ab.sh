#!/bin/bash
# Compare fit() outputs of the working tree's src/ and $BASE_REF's (default
# HEAD). Both runs use the working tree's bench/ tools.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_baseline.sh"
echo "=== CANDIDATE (working tree) fit() ==="
PYTHONPATH=src python bench/fullfit.py run cand
echo "=== BASELINE ($BASE_REF) fit() ==="
PYTHONPATH="$BASE_SRC" python bench/fullfit.py run base
echo "=== COMPARE ==="
python bench/fullfit.py cmp base cand
