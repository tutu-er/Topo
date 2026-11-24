# Development scripts (pre-lock, 2026-11-20 to 2026-11-23)

These are the run chains and analysis scripts behind the numbers in [DEVELOPMENT_RUNS.md](../DEVELOPMENT_RUNS.md). They were written and run from a temporary working folder during the development runs and moved here on 2026-11-23 so the records can be reproduced. They are development tools, not part of the package; the maintained runners are `experiments.py`, `hard.py`, `vsweep.py` and `lock.py` one level up.

**How to run.** From the repository root, in the `Topo310` environment, with BLAS pinned to one thread (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`). They read and write `outputs/hourly_probing_dev_20261120/`, which is not in Git.
- Python scripts were copied unchanged. Several hard-code the checkout path `C:/0-workspace/Topo/repo`; change it if the repository lives elsewhere.
- Shell scripts were given repo-relative paths, and `CONDA` / `PY` default to `conda` / `python`. In the original session each chain first waited for the previous background chain to finish; those waits are now comments, so run the chains in the order below.
- Left out: one-off scripts that patched the code during development (their changes are in the code), smoke tests, environment checks, document-editing helpers, and a pickled test bank (fitted banks are never committed).

## Sections 17–19 (2026-11-23)

| Script | Section | What it does |
| --- | --- | --- |
| `run_vsweep.sh` | §17 | The strong greedy at V = 50–2,000 on the §16 inputs (tag `vsweep`) |
| `calib_U.py` | §17 | How loose the strong greedy's risk bound U is against the realised error |
| `headroom_diag.py` | §17 | Whether the pre-fix code delivered blocks beyond the battery and EV headroom |
| `run_lock.sh` | §17–18 | Picks V by the registered rule, builds seeds 65–96, runs natural and truth-removed inputs |
| `run_vcheck.sh` | §18 addendum | V = 100 on the same natural inputs (tag `vcheck`) |
| `run_vinf.sh` | §17 | V = 100,000 on the §16 inputs (tag `vinf`) |
| `t1_diag.py`, `t1_diag2.py`, `t1_diag3.py` | §18 exploratory | Why T1 failed; whether any logged statistic separates truth-missing banks; what the gains and harms look like |
| `rerun_random.py`, `run_rfix.sh` | §18 | The random arm rerun with the fixed headroom cap (tag `rfix`) |
| `rfix_analyse.py` | §18 | L2 against the fixed random arm, next to the old one |
| `run_preq.py`, `run_chain2.sh` | §18 exploratory | The one-hour-ahead prediction check (tag `preq`), then `rfix` and `vinf` |
| `preq_analyse.py` | §18 exploratory | ROC areas, calibration and the offline veto of the prediction check |
| `run_s3nat.sh` | §19 | Natural truth-missing inputs (tag `s3nat`) |
| `s3nat_analyse.py` | §19 | The §19 tables and the N1 test |
| `closure_split.py` | §19 exploratory | Natural misses and T1 split by whether the bank's closure finished |
| `lint_tex.py`, `lint_tables.py` | paper | Heuristic LaTeX checks for `paper/strategies.tex` (no TeX engine installed): `python lint_tex.py <file>` |

## Section 16 (2026-11-22/23)

`build_hard.sh` and `run_hard.sh` build and run the hard-case conditions. `energy.py`, `reprice.py` and `oneclass.py` checked the DR price model and the class weights; `hard_cap.py`, `hard_extra.py`, `cap_rerun.py` and `near_cap.py` produced the §16 tables and the daily-cap check.

## Sections 10–15 (2026-11-20 to 2026-11-22)

Earlier diagnostics, most without a docstring; each was run as written, and the date tells the section.
- 2026-11-20, first round (§1–9) and line realism (§10): `loss_check.py`, `inputs_summary.py`, `deep.py`, `deep2.py`, `diag.py`, `diag2.py`, `practical.py`, `case_strength.py`, `vdrop.py`, `cov_timing.py`, `cov_all.py`, `cov_stats.py`.
- 2026-11-21, revised step 1 and probing on the revised bank (§11–12): `strata.py`, `short_missing.py`, `old_gap.py`, `why_idle.py`, `pair_be.py`, `round2_stats.py`, `cov_left.py`, `cov_final.py`, `sec11.py`.
- 2026-11-22, head-to-head, GLS and fresh seeds (§13–15): `shapes.py`, `cmp_safety.py`, `noise_struct.py`, `gls_effect.py`, `dev_report.py`, `conf_rules.py`, `adq_sep.py`, `s3_summary.py`, `conf_extra.py`.
