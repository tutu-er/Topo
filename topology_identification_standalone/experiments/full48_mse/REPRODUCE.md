# Recompute this self-contained archive

Extract the entire `full48_mse_20261111/` folder. All observed arrays and physical cases are
inside `runtime/data/` and `runtime/cases/`; no original project is required.
Install Python >=3.10 and the dependencies in `runtime/requirements.txt`.
MIQP needs an independently available Gurobi license, which is not bundled.

From inside this extracted folder, replay the numerical audit and plots:

```text
python -I post_source/analyze_full.py . --bootstrap 2000
python -I post_source/plot_full.py .
```

Optional post-fit true-family diagnostic (does not change the algorithms):

```text
python -I post_source/diagnose_full.py .
```
This optional calculation uses SciPy bounded least squares, with no MIQP and no model-selection feedback.


The audit uses NumPy and the standard library, without a native optimizer.
Plotting additionally needs Matplotlib. Existing archived report and figures
remain the original evidence; use a copy if you want to preserve them unchanged.

For a small algorithm demonstration:

```text
python -I runtime/run.py demo --methods rnj nj pure_wzzt rnj_wzzt rte
```

To rerun all 480 methods freshly, choose a NEW output folder:

```text
python -I runtime/tools/full_benchmark.py freeze ../recomputed_full48 --workers 8
python -I runtime/tools/full_benchmark.py run ../recomputed_full48 --workers 8
python -I post_source/analyze_full.py ../recomputed_full48 --bootstrap 2000
python -I post_source/plot_full.py ../recomputed_full48
```

This full recomputation has a nominal 1920 method-hour allowance, not a promised
elapsed duration. Failed, interrupted and budget-exceeded archived results are
retained; archive integrity does not mean all numerical models succeeded.
Historical absolute paths in payloads describe provenance only. Do not invoke
an archived worker payload directly after relocation; create a fresh run using
the frozen local runner as shown above.
