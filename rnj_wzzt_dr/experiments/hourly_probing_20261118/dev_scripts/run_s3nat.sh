#!/bin/bash
# DEVELOPMENT_RUNS section 19 (registered 2026-11-23 before running): inputs whose truth the bank construction
# itself missed (natural S3), passive and the strong greedy at V = 2,000, 2 paths, tag s3nat.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
M=rnj_wzzt_dr.experiments.hourly_probing_20261118
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
for c in hard_short01 hard_noisy lock_short01 lock_short005 lock_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms passive_bic,bgreedy_V2000_P --paths 2 --workers 14 --strata S3 --no-missing --per-stratum 32 \
    --tag s3nat > $OUT/s3nat_$c.log 2>&1
  echo "ran s3nat $c $(date)"
done
