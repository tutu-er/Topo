#!/bin/bash
# DEVELOPMENT_RUNS section 18 addendum (registered 2026-11-23 before any section 18 result):
# the strong greedy at V = 100 (secondary rule of section 17) on the same fresh natural S1 inputs, tag vcheck.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
M=rnj_wzzt_dr.experiments.hourly_probing_20261118
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
# (in the original session this line waited for the previous background chain to finish)
for c in lock_short01 lock_short005 lock_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms bgreedy_V100_P --paths 2 --workers 14 --strata S1 --no-missing --per-stratum 32 \
    --tag vcheck > $OUT/run_vcheck_$c.log 2>&1
  echo "ran vcheck $c $(date)"
done
