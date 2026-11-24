#!/bin/bash
# Section 18 follow-up: random_matched_P with the fixed headroom cap, replaying cert_P's logged schedule (tag rfix).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
S=rnj_wzzt_dr/experiments/hourly_probing_20261118/dev_scripts
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
$CONDA run -n Topo310 --cwd "$PWD" python $S/rerun_random.py 14 > $OUT/rfix.log 2>&1
echo "ran rfix $(date) exit $?"
