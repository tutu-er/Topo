# Continuous H-only failed diagnostics: continuation package

All V1–V5 diagnostics failed the original gates. No posterior is accepted and
no predictive EIG or action ranking was computed. The original proper prior and
physical target are unchanged; each final level uses H once. Pilot Lhat^kappa
weights are engineering proposal fitting, not exact tempered posterior weights.

The submitted scope contains exact frozen configs, deduplicated byte snapshots
in `frozen_source_history.tar.gz`, all 18 bank aggregates, all seven modes, fresh
repeat seeds, paired L32/L128 measurement/weight hashes and validation records.
`compact_manifest.json` distinguishes historical frozen bytes from integrated
package source hashes. The inverse source, inverse math and inverse tests remain
unchanged. Detailed particle arrays and raw NPZ stay local with paths and hashes.

`aggregate_diagnostics.json.gz` retains component ESS/log evidence/conditional
maximum weight, support counters (absent historical counters remain unknown),
per-time H-weighted latent ESS/relSE and the original failure denominator.
`full_bank_recheck_record.json.gz` records the actual local particle-weight
recomputation; the committed summary cannot independently rederive particle ESS.
The summary verifier checks hashes and algebra from archived sufficient
statistics only. All displayed posterior quantities are numerical diagnostics
and are not accepted estimates. Neither latent relSE nor conditional theta SE
is an unconditional evidence error bound; one CRN bank cannot identify a beta /
latent variance decomposition.

The initial trial-schema read accidentally exposed case/truth_topology/true_index.
The exposure remains disclosed. Formal readers used sanitized H-only six arrays;
actual future/latent/noise arrays were not indexed. These exposed fields were not
used to choose proposals, actions, budgets or filtering.

From the research root, reproducible read-only checks are:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_mle.continuous_eig_recheck_compact
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest -q research_experiments/mobile_ev_mle/test_continuous_ac_inverse.py research_experiments/mobile_ev_mle/test_continuous_eig_sampling.py research_experiments/mobile_ev_mle/test_continuous_eig_proposals.py research_experiments/mobile_ev_mle/test_continuous_eig_guided_latent.py research_experiments/mobile_ev_mle/test_continuous_eig_tempered_proposal.py research_experiments/mobile_ev_mle/test_continuous_eig_final_helpers.py -p no:cacheprovider
```

To repeat the local particle audit, pass `--compact` with the manifest's
`full_local_evidence_directory`; it loads the local diagnostic arrays and
reconstructs original seven-mode importance weights without any new MC.

The concrete remaining bottleneck is six modes with conditional ESS about 1,
despite final global ESS above 100. Inspect each mode's paired log evidence,
supported-theta counts and weighted per-time latent diagnostics before proposing
a new frozen repair. The archived final runner is callable as
`python -m research_experiments.mobile_ev_mle.continuous_eig_final_run --help`.
It requires a fresh output directory and an explicit pilot directory for any
new authorized run. No new run or algorithm freeze is part of this delivery.
