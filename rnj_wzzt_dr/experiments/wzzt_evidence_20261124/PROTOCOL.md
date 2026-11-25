# Phase 1 protocol: does WZZT's MIQP do work that heuristics cannot? (registered 2026-11-24, before any Phase 1 run)

This is Phase 1 of [WZZT_EVIDENCE_PLAN.md](../hourly_probing_20261118/WZZT_EVIDENCE_PLAN.md), covering experiments A, B and F. These are development runs: they decide the role of the MIQP inside WZZT. They are not confirmatory for hypotheses H1–H6. The thresholds and comparisons below are fixed before the runs. Anything added after the runs is labelled exploratory.

## Inputs

- **Real:** the 66 real-feeder pilot inputs (33 feeders × start days 0 and 6). Each has 144 training and 48 validation hourly rows, and the target is the identifiable truth.
- **Synthetic:** `lock_short01` and `lock_noisy`, taking the 8 lowest seeds of each of the four networks (32 per condition, 64 in all). Each has 96 training and 48 validation rows, and the target is the exact truth.
- **Stage 0 (capture):** every input gets the study's two identifier runs (levels and demeaned), method `111`, `SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)`, which makes 260 runs. The search is unchanged. Hooks only record the inputs and results of every W step and of the E stage.
- "Internal" supports are those with 1 < |C| < n. A family is "exact" when its internal supports equal the truth's.

## Experiment A: the E stage, same problem, different solvers

**Instance.** The E call of each run: training data, the finite pool, the mandatory supports, E's start point, α = 0.003 and coefficient bounds [0, 2]. **Problem:** choose a laminar subset of the pool's internal supports to minimise the normalised training MSE plus α per selected support (the E MIQP's own objective).

**Exact reference.**
- The E MIQP is rebuilt with the data rows compressed by QR, which is an exact reformulation. It runs with MIPGap 1e-6, a 900 s limit and one thread, started from the run's own E solution.
- An instance is *certified* when Gurobi reports optimal.
- The *optimum* is the lowest objective found by any solver on that instance. The MIQP bound is the lower bound.

**Heuristics.** Each searches with an exact compressed fit, then has its family rescored with the package's fixed-family fit. Each is capped at 300 s.
- H1: greedy forward addition from the base model (singletons and the whole feeder).
- H2: local search from the base model, taking the best improving move among adding a support, dropping a support, and adding a support while evicting the supports it crosses.
- H3: the same local search from E's own start (the current point or the RNJ reference, whichever E used).
- H4 and H5: beam search of width 5 and of width 20.
- H6: local search started from each RNJ tree of the guidance.
- H7: local search started from 20 random laminar subsets of the pool (fixed seed).
- H8: a nonnegative lasso path over the pool (box [0, 2], 20 penalties), followed by laminar repair and a refit.
- H9: H8 followed by local search.
- **Portfolio P = the best of H1–H9.** P is the pre-specified candidate to replace the MIQP.

**Primary metric.** P's reach rate: the share of certified instances where P's objective is at most the optimum + 1e-6 (normalised units; one support costs 0.003).

**Secondary metrics.**
- Each heuristic's reach rate, objective gap and time.
- When P misses: internal-segment recall, precision and exactness of P's family against the MIQP's family.
- The features of the instances P misses: n, pool size, crossing pairs, training rows, real or synthetic.

**Decision rule D-E.**
- *Replaceable* if P reaches the optimum in at least 99% of certified instances, and there is no instance where the MIQP's family is exact and P's is not. E's MIQP then moves to the certificate role: check P's answer with the MIQP bound, and solve only when the bound does not close.
- *Otherwise* the MIQP stays as E's solver. The instances P misses define where exactness pays.
- *Undecided* if fewer than 90% of instances are certified.

## Experiment B: the W stage

**B1, per step.**
- **Instance:** every W step of every run (its current family and training data).
- **Exact reference:** the W MIQP's result from the run, when Gurobi reported optimal. This is the best single compatible support added with all coefficients refit, scored by the W MIQP's own objective (old supports stay charged).
- **Generators.** Every candidate is refit with all coefficients and scored by the same objective.
  - G1: the RNJ pool (the same candidates as W's pool trials).
  - G2: unions of two sibling blocks.
  - G3: RNJ trees (3 weights × 5 tolerances) built from sensitivities fitted to the current residual.
  - G4: average-linkage clustering of each node's child blocks by residual correlation; every merged cluster is a candidate.
  - G5: screening. Every compatible support (every union of a node's child blocks, for nodes with at most 20 children) is ranked by its one-support gain on the current residual with old coefficients fixed, and the top 50 are refit.
  - Portfolio Gp: the best of G1–G5.
  - G6, brute force: every compatible support refit, where there are at most 20,000. It serves as an independent check of the MIQP.
- **Primary metric:** Gp's reach rate on certified steps (objective at most the MIQP's + 1e-6).
- **Also reported:**
  - whether the MIQP's support is in each generator's set;
  - comparisons after zero cleanup;
  - at W's final step, whether Gp agrees with the MIQP that no support gains.

**B2, truth-removed pools.**
- **Runs:** the demeaned run only, on all 130 inputs.
- **Removal:** every true support of size at least 3 is removed from the RNJ pool, the RNJ trees and the reference family. The training reference loss stays as it was. Supports of size 2 are excluded because E's pool always contains every pair.
- **Three variants:**
  - "MIQP W": W full, as now;
  - "pool W": W with pool trials only;
  - "generator W": pool trials with G2–G5 added to the pool at every step.
- **Metrics:** removed true supports recovered in the selected family and in WZZT's whole output; exact recovery; time.
- **Paired comparison** of MIQP W against generator W: per-input counts by Wilcoxon signed-rank; exact recovery by McNemar.

**Decision rule D-W.**
- *Replaceable* if Gp reaches the MIQP's result in at least 99% of certified steps, *and* generator W recovers at least as many removed supports as MIQP W (the deficit is no more than 2% of removed supports and is not significant at 5%). W's MIQP then moves to a single no-gain certificate at the end of W.
- *Otherwise* it stays. The same 90% certification floor as D-E applies.

## Experiment F: speed

- For each E instance, the original E MIQP's time from the run is compared with the compressed MIQP under the same settings: MIPGap 1e-4, the same start, and the same time allowance.
- **Exactness check:** same status, objective within 1e-7 relative, and the same family.
- **Report:** median and range of the speed-up, by condition and by training length.

## Not decided here

- Baselines and the benchmark (Phase 3), the source term (Phase 2), and any change to the hourly-probing protocol.
- Truth is used only to build the truth-removed pools and to score.

## Deviations (added after the first run, before the re-runs)

**D1 (2026-11-24): the E certificate.** In the first run, Gurobi reported "optimal" (status 2, bound equal to the objective) at points that a known family beats. This happened with both the original E MIQP and the QR-compressed one:
- `eulv_td008690_feeder21_s6` lv: certified 0.66017; local search found 0.65933.
- `eulv_td400828_feeder153_s6` lv: certified 0.36756; greedy found 0.34226.

The better families are feasible in the model. Given as a start, or as a two-family pool, the same MIQP returns them. So the fault is in Gurobi's bounding, not in the formulation.

The cause is the conditioning. The E pool's design is rank-deficient by construction: for example, 75 independent columns out of 126 on a 12-meter feeder, because the span of z z' has dimension at most n(n+1)/2. On the real feeders P and Q are also nearly collinear (correlation 0.996–1.000).

On the two failing instances, from a cold start, only the SVD-compressed rows combined with `NumericFocus` 1 or 2 found and proved the best known value. These failed at least once:
- the original formulation with `Presolve=0`;
- SVD with default numerics, `Presolve=0`, `PreQLinearize=0`, `Quad=1` or looser tolerances.

What changes (stage `a2`, `exp_a2`):
- **Certifier:** `certify.SVDSolverNF2`. This is the E MIQP with rows compressed by a truncated SVD (singular values above 1e-10 of the largest), with `NumericFocus=2`, MIPGap 1e-6 and 900 s. It is warm-started from the best known family of the first run, i.e. the best of all methods after the package refit.
- **Cold-start check:** the same formulation from the run's start, with the run's MIPGap (1e-4) and time allowance. Its result also replaces the F speed comparison (F2).
- **Optimum:** the lowest refitted objective over all methods, the re-certification and the cold-start check.
- **Certified:** the re-certification returns status 2 and no known family beats it by more than 1e-6.
- **Formulation error:** the cold-start check returns status 2 at an objective above the optimum by more than its MIPGap. These are counted and reported.
- **Unchanged:** D-E's thresholds (99% reach, no lost exact topology, 90% certification floor), the heuristics and their outputs.

The first-run decision, made with the QR certifier, is reported next to the D1 decision. D-W does not use the E certificate and is unchanged.

**D2 (2026-11-24): smoke rows.** Some rows for `eulv_td000653_feeder15_s0`/`_s6` were written by the smoke test before the full run started. All rows for these two inputs (4 E instances, their W steps and their B2 runs) are recomputed with the final code. The first-run files are kept in `outputs/wzzt_evidence_20261124/backup_run1/`.

**D3 (2026-11-24, registered after the D1 run, before this run): a second, independent certificate.** The D1 decision passes with a margin of one instance: the portfolio reached 258 of 259 certified optima, against a threshold of 257. In D1's cold-start check, the SVD formulation once claimed "optimal" at a point a known family beats (`eulv_td400291_feeder10_s0` lv, MIPGap 1e-4).

The D1 certificates were warm-started at the best known family, so a false bound there would go unnoticed. Stage `a3` therefore certifies every instance again:
- from the run's cold start;
- with the same formulation (`SVDSolverNF2`), MIPGap 1e-6 and 900 s;
- with a different Gurobi `Seed` (7), so the search path differs.

A certificate is *confirmed* when both certifications return status 2 and their refitted objectives agree within 1e-6. D-E is then recomputed on the confirmed instances, against the lowest objective known from any method.

- If D-E on the confirmed set agrees with the D1 decision, the D1 decision is final.
- If it disagrees, or fewer than 90% of instances are confirmed, the E verdict is reported as *undecided*.
