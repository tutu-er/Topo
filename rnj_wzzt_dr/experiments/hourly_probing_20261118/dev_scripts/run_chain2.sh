#!/bin/bash
# After the V = 100 check: (1) missing-truth check development run (tag preq), (2) fixed random arm (tag rfix),
# (3) V -> infinity point on the section 16 inputs (tag vinf).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
S=rnj_wzzt_dr/experiments/hourly_probing_20261118/dev_scripts
M=rnj_wzzt_dr.experiments.hourly_probing_20261118
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
$CONDA run -n Topo310 --cwd "$PWD" python $S/run_preq.py 14 > $OUT/preq.log 2>&1
echo "ran preq $(date)"
$CONDA run -n Topo310 --cwd "$PWD" python $S/rerun_random.py 14 > $OUT/rfix.log 2>&1
echo "ran rfix $(date)"
for c in hard_short01 hard_short005 hard_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms bgreedy_V100000_P --paths 2 --workers 14 --strata S1 --no-missing --per-stratum 32 \
    --tag vinf > $OUT/vinf_$c.log 2>&1
  echo "ran vinf $c $(date)"
done
