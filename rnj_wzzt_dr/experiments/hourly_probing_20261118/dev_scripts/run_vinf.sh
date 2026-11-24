#!/bin/bash
# DEVELOPMENT_RUNS section 17 extension: the upper limit V -> infinity (V = 100,000 CNY) on the section 16
# ambiguous inputs, tag vinf; runs after the section 18 runs and the V = 100 check so it does not slow them.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
M=rnj_wzzt_dr.experiments.hourly_probing_20261118
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
for c in hard_short01 hard_short005 hard_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms bgreedy_V100000_P --paths 2 --workers 14 --strata S1 --no-missing --per-stratum 32 \
    --tag vinf > $OUT/vinf_$c.log 2>&1
  echo "ran vinf $c $(date)"
done
