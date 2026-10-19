#!/bin/bash
# Time the working tree's src/ against $BASE_REF's (default HEAD; use
# BASE_REF=origin/master for a pull request). The two sides alternate for
# $ROUNDS rounds (default 3); compare the medians. Both sides use the working
# tree's bench/ tools.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_baseline.sh"
OBJ="import bench.harness as h; h.print_setup(); print(h.benchmark(obj_idx=h.full_coverage_objects()[:12], repeats=3))"
for round in $(seq "${ROUNDS:-3}"); do
  echo "=== round $round: CANDIDATE (working tree) ==="
  PYTHONPATH=src python -c "$OBJ"
  echo "=== round $round: BASELINE ($BASE_REF) ==="
  PYTHONPATH="$BASE_SRC" python -c "$OBJ"
done
