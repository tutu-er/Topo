# Known-equipment full-case33 candidate ranking

Read [results.md](results.md) for results, [protocol.md](protocol.md) for the
predeclared experiment, and [admm_analysis.md](admm_analysis.md) for the future
shared-R/X decomposition. The original LaTeX/PDF is not modified.

The experiment reuses the previous case33_quality_fixed_rx input archive.
It compares 10 statically declared full 33-bus trees, three operating states,
48 paired noisy conditions and three clean arithmetic checks. R/X and source
P/Q are fixed; EIV corrects the noisy terminal P/Q. No WZZT, EV optimization,
new forecasting model or ADMM solver is implemented here.

## Reproduce

Use the Topo NumPy/SciPy environment, single BLAS thread, with the research
checkout on PYTHONPATH. From the checkout root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B -m research_experiments.mobile_ev_mle.case33_candidate_ranking.experiment --input research_experiments/mobile_ev_mle/case33_candidate_ranking/evidence/input_archive.zip --output outputs/case33_ranking_replay --workers 4
python -B -m research_experiments.mobile_ev_mle.case33_candidate_ranking.audit_saved --output outputs/case33_ranking_replay
python -B -m pytest -c research_experiments/mobile_ev_mle/case33_candidate_ranking/pytest.ini --confcutdir research_experiments/mobile_ev_mle/case33_candidate_ranking research_experiments/mobile_ev_mle/case33_candidate_ranking/test_experiment.py -q -p no:cacheprovider
```

Use a fresh output directory for a rerun. Partial runs may resume only with the
same input hash, catalog and scientific configuration; completed runs are not
overwritten. `--seed-limit 1` is a separate pilot, not the complete experiment.

`evidence/full_results.zip` contains all endpoints, observations, rankings,
configurations and audit results. Extract it into an empty result directory
and call `audit_saved --output <directory> --input <input_archive.zip>` to
audit without refitting. All unqualified references remain in the archive.

`evidence/source_snapshot.zip` holds this study, imported local dependencies,
the independent physical replay and original case33 table. Restore into a
fresh directory and point PYTHONPATH there to avoid another checkout shadowing
the package. `source_manifest.json` records versions and SHA256 hashes;
`canonical_replay_receipt.json` records stage/canonical agreement.

The AC reference is a checked local fit. Rank agreement is distinct from true
topology recovery, and neither is a calibrated correctness probability.
