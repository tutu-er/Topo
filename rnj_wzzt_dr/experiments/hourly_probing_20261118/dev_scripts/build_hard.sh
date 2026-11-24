#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
OUT=outputs/hourly_probing_dev_20261120
cd "$(git rev-parse --show-toplevel)"
for c in hard_short01 hard_short005 hard_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments build $c --workers 14 > $OUT/build_$c.log 2>&1
  echo "built $c $(date)"
done
