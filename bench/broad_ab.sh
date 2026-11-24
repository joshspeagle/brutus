#!/bin/bash
# A/B broad.py runs of the working tree's src/ against $BASE_REF's (default
# HEAD), alternating the two sides for $ROUNDS seeds (default 4), then cmp.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_baseline.sh"
for seed in $(seq 1 "${ROUNDS:-4}"); do
  echo "=== round $seed: BASELINE ($BASE_REF) ==="
  PYTHONPATH="$BASE_SRC" python bench/broad.py run base "$seed"
  echo "=== round $seed: CANDIDATE (working tree) ==="
  PYTHONPATH=src python bench/broad.py run cand "$seed"
done
python bench/broad.py cmp base cand
