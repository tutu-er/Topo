# case33: P/Q error correction and quality-score accuracy

This is a small known-line diagnostic, not an R/X or topology recovery study.
All six methods receive the original case33 topology and line impedances.
Only EIV methods fit the 64 single-period true P/Q values; FIX methods use
the noisy P/Q readings directly. No unknown source or EM is introduced.

Read `protocol.md` for the predeclared design and `results.md` for conclusions.
The full raw results and source snapshot are in `evidence/`.

Methods: `{FIX,EIV} x {linear,i2,ac}`. LD is lossless squared-voltage flow;
DF-I2 includes downstream loss propagation; AC is full nonlinear flow. All
six scores use voltage magnitude plus root P/Q and consistent meter sigmas.

## Reproduce

Use the Topo Python environment with NumPy/SciPy and one BLAS thread. From the
research checkout root, run:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B -m research_experiments.mobile_ev_mle.case33_quality_fixed_rx.experiment --output outputs/case33_quality_replay --seeds 8
python -B -m research_experiments.mobile_ev_mle.case33_quality_fixed_rx.audit_saved --results outputs/case33_quality_replay
python -B -m pytest -c research_experiments/mobile_ev_mle/case33_quality_fixed_rx/pytest.ini --confcutdir research_experiments/mobile_ev_mle/case33_quality_fixed_rx research_experiments/mobile_ev_mle/case33_quality_fixed_rx/test_experiment.py -q
```

`--seeds 1` is a pilot; the scientific configuration and eight-seed run remain
separate. The no-noise cases are arithmetic diagnostics only. Do not use their
true states to initialize or choose the noisy-data fits.

The source archive contains the original case33 table and every loaded local
Python dependency. Restore it into an empty directory to avoid accidentally
importing another checkout with the same package name. All failed starts and
reference qualification fields remain in the raw JSON.
