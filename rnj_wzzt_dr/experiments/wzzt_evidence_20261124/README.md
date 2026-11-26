# WZZT evidence: does the MIQP do work that heuristics cannot, and does a revised WZZT do better?

Phases 1–3 of [WZZT_EVIDENCE_PLAN.md](../hourly_probing_20261118/WZZT_EVIDENCE_PLAN.md).
- **Phase 1** (the MIQP's role): [PROTOCOL.md](PROTOCOL.md), [RESULTS.md](RESULTS.md).
- **Phase 2** (the revised WZZT, WZZT-R; development): [PROTOCOL_PHASE2.md](PROTOCOL_PHASE2.md), [RESULTS_PHASE2.md](RESULTS_PHASE2.md).
- **Phase 3** (confirmation on unseen inputs): [PROTOCOL_PHASE3.md](PROTOCOL_PHASE3.md), registered before its runs.
- **WZZT-C confirmation**: [PROTOCOL_WZZT_C.md](PROTOCOL_WZZT_C.md), [RESULTS_WZZT_C.md](RESULTS_WZZT_C.md) (not confirmed: C1b not shown).

## Modules

| File | What it does |
| --- | --- |
| `capture.py` | Runs WZZT `111` unchanged on the 130 inputs (levels and demeaned). Hooks record every W step and the E stage. Also holds the B2 truth-removal hook |
| `fastfit.py` | Exact fixed-family fits on compressed data: `UniverseFitter` (any sub-family of the E pool), `ExtensionFitter` (current family plus any one support), and the enumeration of every compatible support |
| `heuristics.py` | E heuristics H1–H9 (greedy, local search, beam, RNJ and random multistarts, lasso path) |
| `generators.py` | W generators G1–G5 (RNJ pool, sibling pairs, residual RNJ, residual clustering, screening) |
| `certify.py` | The E MIQP with compressed rows: QR (first run, experiment F) and truncated SVD with `NumericFocus=2` (the certifier of deviations D1 and D3) |
| `run_phase1.py` | Runner: `capture`, `a` (experiments A and F), `a2` (D1: re-certification and F2), `a3` (D3: second certificate), `b1`, `b2` |
| `analyse_phase1.py` | Tables and the decision rules D-E (first run, D1, D3) and D-W |
| `test_phase1.py` | Fast fits match the package fit, the enumeration is complete, generators return valid supports, and the compressed MIQP matches the original |
| `PROTOCOL_PHASE2.md` | Phase 2 (the revised WZZT): design, variants, decision rule D2, deviations |
| `run_phase2.py` | Phase 2 runner: `ceiling` (data-supported truth segments), `ladder` (WZZT-R and ablations, experiment C), `bank` (the probing study's closure, with and without R3), `root` (root scenarios, experiment E) |
| `analyse_phase2.py` | Phase 2 tables and the decision rule D2 |
| `test_phase2.py` | WZZT-R: transformed columns and Gram statistics, fits against the explicit and package fits, both MIQPs against brute force, the screening gain, the noise estimate, a toy recovery |
| `judge_phase2.py` | Scores the selected families and the truth with the probing study's own BIC (is the search or the data the limit?) |
| `explore_price.py` | Exploratory: the study closure seeded with the lower-price (`R_alpha`) families, alone and together with `R`'s |
| `build_phase3.py` | Phase 3 real inputs: the 33 feeders on start days never used in development (1–5, 7–11) |
| `run_phase3.py` | Phase 3 runner: `ceiling`, `orig` (`111`, levels and demeaned), `R` (frozen WZZT-R), `bank`, `judge` |
| `analyse_phase3.py` | Phase 3 tables and the registered decision (P1, P2; S1–S4) |
| `PROTOCOL_MIQP.md` | Experiment M: where the exact search is critical (scale, interacting branches, certified resolution) |
| `build_large.py` | Experiment M inputs: whole transformer networks (26–77 meters), large feeders, 16-day histories |
| `run_miqp.py` | Experiment M runner: `fit`, `anatomy` (MIQP at every W step), `pairs` (W2), `resolution` (node certificates) |
| `analyse_miqp.py` | Experiment M tables, including the 16-day test of the certified data requirement |
| `test_miqp_ext.py` | The generalised unknown-support MIQP (`rnj_wzzt_dr/identification/miqp_ext.py`: k = 1 or 2 supports, node-restricted, forced) against brute force |

The revised method itself is `rnj_wzzt_dr/identification/revised.py` (WZZT-R); the original `algorithm.fit` is unchanged.

## Run

From the repository root, in `Topo310`, with BLAS pinned to one thread:

~~~sh
python -m pytest -q rnj_wzzt_dr/experiments/wzzt_evidence_20261124/test_phase1.py
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 capture --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 a --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 b1 --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 b2 --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 a2 --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 a3 --workers 14
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_phase1
# Phase 2
python -m pytest -q rnj_wzzt_dr/experiments/wzzt_evidence_20261124/test_phase2.py
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 ceiling
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 bank --sources orig
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 ladder --variants R
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 bank --sources R
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 ladder --variants R_pq,R_alpha,R_plain,R_c0,R_src,R_wmiqp,R_wpool,R_emiqp,R_train
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.judge_phase2
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase2 root --variants R_h,R_src_h,R_c0_h
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_phase2
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.explore_price
# Phase 3
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.build_phase3 --workers 2
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 ceiling
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 orig
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 R
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 bank
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase3 judge
python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_phase3
~~~

- **Inputs:** the hourly-probing development inputs in `outputs/hourly_probing_dev_20261120/` (real-feeder pilot, `lock_short01`, `lock_noisy`).
- **Outputs:** `outputs/wzzt_evidence_20261124/` (not in Git).
- **Resuming:** every stage skips finished jobs, so it can be restarted after an interruption.
