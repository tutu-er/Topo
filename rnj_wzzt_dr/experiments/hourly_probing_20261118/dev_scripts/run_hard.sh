#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
ARMS=passive_bic,cert_P,random_matched_P,qbc_P,bgreedy_P,bgreedy_lowV_P
cd "$(git rev-parse --show-toplevel)"
for c in "$@"; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments \
    run $c --arms $ARMS --paths 2 --workers 14 --strata S1 --no-missing --per-stratum 32 > $OUT/run_$c.log 2>&1
  echo "ran $c $(date)"
done
