#!/bin/bash
# Compare loglike outputs and posterior summaries of the working tree's src/
# and $BASE_REF's (default HEAD) on the cohort. Both runs use the working
# tree's bench/ tools.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_baseline.sh"
PY='import bench.harness as h, bench.disteq as d; h.print_setup(); h.capture("coh_TAG", obj_idx=d.COHORT); d.run("deq_TAG", K=6, objs=d.COHORT)'
echo "=== CANDIDATE (working tree): capture + disteq on the cohort ==="
PYTHONPATH=src python -c "${PY//TAG/cand}"
echo "=== BASELINE ($BASE_REF): capture + disteq on the cohort ==="
PYTHONPATH="$BASE_SRC" python -c "${PY//TAG/base}"
echo "=== EXACT loglike (cohort) ==="
python bench/harness.py compare coh_base coh_cand | grep WORST
echo "=== ANTITHETIC distributional (cohort) ==="
python -c "import bench.disteq as d; d.cmp('deq_base', 'deq_cand')"
