#!/bin/bash
# DEVELOPMENT_RUNS sections 17-18: after the V sweep, pick V by the registered rule, build the fresh-seed
# inputs (seeds 65-96), then run the natural ambiguous inputs and their truth-removed copies.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=${CONDA:-conda}
PY=${PY:-python}
OUT=outputs/hourly_probing_dev_20261120
M=rnj_wzzt_dr.experiments.hourly_probing_20261118
cd "$(git rev-parse --show-toplevel)"
# (in the original session this line waited for the previous background chain to finish)
$CONDA run -n Topo310 --cwd "$PWD" python -m $M.vsweep > $OUT/vsweep_summary.txt 2>&1
V=$($PY -c "import json;print(json.load(open(r'outputs/hourly_probing_dev_20261120/summary_vsweep.json'))['primary'])")
if [ -z "$V" ]; then echo "no V chosen; see vsweep_summary.txt"; exit 1; fi
echo "sweep analysed $(date): V = $V"
for c in lock_short01 lock_short005 lock_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments build $c --workers 14 > $OUT/build_$c.log 2>&1
  echo "built $c $(date)"
done
for c in lock_short01 lock_short005 lock_noisy; do
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms passive_bic,bgreedy_V${V}_P,cert_P,random_matched_P,qbc_P --paths 2 --workers 14 \
    --strata S1 --no-missing --per-stratum 32 > $OUT/run_$c.log 2>&1
  echo "ran natural $c $(date)"
  $CONDA run -n Topo310 --cwd "$PWD" python -m $M.experiments run $c \
    --arms passive_bic,bgreedy_V${V}_P --paths 2 --workers 14 --strata S1 --per-stratum 32 \
    --truth-missing --tag missing > $OUT/run_missing_$c.log 2>&1
  echo "ran truth-missing $c $(date)"
done
