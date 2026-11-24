#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
ARMS=bgreedy_V50_P,bgreedy_V100_P,bgreedy_V200_P,bgreedy_V500_P,bgreedy_V1000_P,bgreedy_V2000_P
cd "$(git rev-parse --show-toplevel)"
for c in hard_short01 hard_short005 hard_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments \
    run $c --arms $ARMS --paths 2 --workers 14 --strata S1 --no-missing --per-stratum 32 --tag vsweep > $OUT/vsweep_$c.log 2>&1
  echo "swept $c $(date)"
done
