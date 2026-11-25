# WZZT evidence, Phase 1: results (2026-11-24)

Design and decision rules: [PROTOCOL.md](PROTOCOL.md), registered before the runs, with deviations D1–D3 added before each re-run. Plan: [WZZT_EVIDENCE_PLAN.md](../hourly_probing_20261118/WZZT_EVIDENCE_PLAN.md).

**Inputs.**
- 130 inputs: the 66 real EU LV pilot inputs (`eulv`), and 32 each from `lock_short01` and `lock_noisy` (8 seeds × 4 bundled feeders).
- Each input is run in levels (`lv`) and demeaned (`dm`) form, giving 260 E instances and 1,094 W steps.
- 128 inputs in B2. The two inputs of `eulv_td009471_feeder111` have no true support of size ≥ 3.
- No run errors.

## Decisions

| Rule | Registered test | Result | Verdict |
| --- | --- | --- | --- |
| **D-E** (E stage) | The heuristic portfolio reaches the certified optimum in ≥ 99% of certified instances; no exact topology lost; certification ≥ 90% | 258 of 259 confirmed optima (99.6%; threshold 257). No exact topology lost. 259/260 confirmed by two independent certificates (D3) | **Replaceable: E's MIQP moves to the certificate role** |
| **D-W** (W stage) | The generator portfolio G1–G5 reaches the MIQP's step result in ≥ 99% of certified steps, *and* B2 shows no loss | 96.9% of steps (1,054/1,088). B2 shows no loss | **Not replaceable: W's MIQP stays** |

## 1. A finding that comes first: Gurobi's "optimal" stamp is unreliable on these problems

In the first run, Gurobi reported "optimal" (status 2, gap zero) at points that a known family beats. This happened with the original E MIQP *and* with the QR-compressed one.

| Instance | "Certified" optimum | Better known family | Found by |
| --- | --- | --- | --- |
| `eulv_td008690_feeder21_s6` lv | 0.66017 | 0.65933 | local search |
| `eulv_td400828_feeder153_s6` lv | 0.36756 | 0.34226 | greedy (adds the single best branch at each step, never reconsiders) |

**How often each version stamped a wrong answer "optimal"**, counting a stamp as false when the result is above the best known value by more than its MIPGap:

| Version | False stamps |
| --- | --- |
| The original E MIQP, as WZZT ran it | 2 of 255 (`eulv_td008742_feeder82_s6` dm, `eulv_td400291_feeder10_s0` lv) |
| QR-compressed, run settings | 11 of 257, all on levels data |
| QR-compressed certification, MIPGap 1e-6 | 2 of 257 (the two in the table above) |
| SVD + `NumericFocus=2`, run settings | 1 of 257 (`eulv_td400291_feeder10_s0` lv) |
| SVD + `NumericFocus=2`, cold start, MIPGap 1e-6, `Seed` 7 (D3) | 1 of 260 (the same instance) |

**The formulation is not at fault.** The better families are feasible in the model. Given as a start, or as a two-family pool, the same MIQP returns them with the right objective. The fault is in Gurobi's bounding: from a cold start it cuts away the region that contains the better family.

**Why the bounding fails: conditioning.**
- The E pool's design matrix is rank-deficient *by construction*. Every pool support adds a matrix z zᵀ, and these span at most n(n+1)/2 dimensions. Examples: 75 independent columns out of 126 on a 12-meter feeder, and 297 out of 373 on a 24-meter one.
- On the real feeders, each meter's P and Q are also nearly collinear (correlation 0.996–1.000), so the R and X columns nearly coincide.

**What fixes it** (tested on the two failing instances from a cold start, 600 s each):

| Formulation and setting | `feeder21_s6` lv | `feeder153_s6` lv |
| --- | --- | --- |
| Original, `Presolve=0` | time limit, not found | false "optimal" (+0.0235) |
| SVD rows, default | false "optimal" (+0.0022) | right, 477 s |
| SVD rows, `Presolve=0` / `PreQLinearize=0` / `Quad=1` / loose tolerances | false or unproven | mixed |
| **SVD rows, `NumericFocus=1` or `2`** | **right, ~550 s** | **right, ~550 s** |

"SVD rows" means the data rows compressed by a truncated SVD: singular values below 1e-10 of the largest are dropped, which removes the redundant directions. This is `certify.SVDSolver`. With `NumericFocus=2` it became the certifier of deviation D1.

Even this version is not perfect. On `eulv_td400291_feeder10_s0` lv, it claimed "optimal" at 0.48668, with gap zero, both at MIPGap 1e-4 and at 1e-6, while 0.48506 is known.
- The original run stamped the same wrong value there.
- Only the heuristics found 0.48506; from there, no certifier could prove optimality in 900 s.
- On this instance, every Gurobi version makes the same pruning error.

**So a certificate needs confirmation.** A solve warm-started at the answer it is checking cannot expose a false bound. Deviation D3 adds a second certificate from a cold start with a different seed. The two agree on 259 of 260 instances; the exception is this one.

**The heuristics and the MIQP check each other.** The heuristics caught the MIQP's false stamps (§1 tables). The MIQP proved the heuristics' answers optimal on 259 instances.

**What this means for WZZT:**
- The earlier statement "every MIQP solved to optimality" (WZZT_EVIDENCE_PLAN §1, last row) does not hold as stated. Some of those certificates were false.
- An exact-certificate claim, novelty item (c), needs the SVD formulation with stronger numerics, and the certificates must be checked. The default formulation's certificates cannot be cited.
- The W-stage MIQP (`extend`) has the same weakness. In B1, generators beat its "optimal" result in 3 of 1,088 steps.

## 2. Experiment A: the E stage (does the MIQP find what heuristics cannot?)

Each E instance was frozen from a `111` run and solved again by nine heuristics, H1–H9. Every answer was refitted with the package fit (a bounded least-squares solve with its own optimality check), so all objectives are comparable. "Reach" means within 1e-6 of the optimum.

**Certification.**

| Run | Certifier | Certified | Notes |
| --- | --- | --- | --- |
| First run (registered) | QR-compressed MIQP, MIPGap 1e-6, 900 s, from the run's answer | 257/260 | 2 certificates contradicted by heuristics (§1) |
| D1 | SVD + `NumericFocus=2`, from the best known family | 259/260, none contradicted | `feeder10_s0` lv at the time limit (gap 1.7%) |
| D3 | SVD + `NumericFocus=2`, cold start, `Seed` 7; confirmed = both agree | 259/260 confirmed | `feeder10_s0` lv: the second certificate stamps 0.48668 "optimal" (§1). Median 24.9 s; 90th percentile 229 s |

No re-certification found a better objective than the best already known from the heuristics and the run. The D1 certified set and the D3 confirmed set are the same 259 instances, with the same optima.

**Reach against the confirmed optimum** (259 instances):

| Method | Reach | Real (`eulv`) | `lock_short01` | `lock_noisy` | Median time |
| --- | --- | --- | --- | --- | --- |
| MIQP in the run (as WZZT ran it) | 256 (98.8%) | 128/131 | 64/64 | 64/64 | 31.5 s |
| H1 greedy forward addition | 117 (45.2%) | 91/131 | 13/64 | 13/64 | 0.9 s |
| H2 local search from the base | 152 (58.7%) | 105/131 | 27/64 | 20/64 | 0.5 s |
| H3 local search from E's start | 173 (66.8%) | 107/131 | 45/64 | 21/64 | 0.0 s |
| H4 beam search, width 5 | 187 (72.2%) | 122/131 | 33/64 | 32/64 | 2.6 s |
| H5 beam search, width 20 | 232 (89.6%) | 130/131 | 56/64 | 46/64 | 8.4 s |
| H6 local search from each of the 15 RNJ trees | 241 (93.1%) | 128/131 | 62/64 | 51/64 | 13.2 s |
| H7 local search from random laminar starts | 243 (93.8%) | 130/131 | 59/64 | 54/64 | 33.8 s |
| H8 lasso path with laminar repair | 14 (5.4%) | 14/131 | 0/64 | 0/64 | 0.3 s |
| H9 lasso then local search | 183 (70.7%) | 104/131 | 48/64 | 31/64 | 0.4 s |
| **Portfolio P (best of H1–H9)** | **258 (99.6%)** | 131/131 | 64/64 | 63/64 | 62.7 s (all nine) |

**Reading.**
- No single heuristic is enough: the best (H7) misses 6% of instances. Together they miss one: `lock_noisy/flynn16_s70` lv, gap 0.0023, which is under one branch price (α = 0.003). The portfolio's best member there was beam search (H5).
- Three heuristics carry the portfolio. H5 + H6 + H7 reach 258, the same as all nine, at a median of 57 s. H6 is the only heuristic to reach the optimum on 6 instances, H7 on 2 and H5 on 1.
- **The portfolio is not cheaper than the MIQP.** Its median is 57–63 s against the MIQP's 31.5 s; the 90th percentiles are 212 s and 213 s. Replacing the solver would buy no speed. What changes is only *who searches*.
- **Truth:** the certified optimum and the portfolio both give the exact topology on 30 of 260 instances. There is no instance where one is exact and the other is not.
- The run's own E MIQP missed the optimum 3 times: 2 false "optimal" claims and 1 time limit.

## 3. Experiment B1: the W stage, step by step

At each of the 1,094 W steps of the `111` runs, the MIQP's best one-support extension was compared with the best support from each generator. Each candidate's objective is computed exactly: the current family plus the candidate, refitted. 1,088 steps had status 2; 6 hit the time limit.

| Generator | Reaches the MIQP's step result | Has the MIQP's support | Median set size |
| --- | --- | --- | --- |
| G1 the RNJ pool | 596 (54.8%) | 51.8% | 18 |
| G2 unions of sibling pairs | 187 (17.2%) | 11.7% | 49 |
| G3 RNJ on the residual sensitivities | 381 (35.0%) | 30.7% | 6 |
| G4 clusters of residual correlations | 187 (17.2%) | 11.5% | 11 |
| G5 screening of every subset of every node's children (top 50) | 1,003 (92.2%) | 91.6% | 50 |
| **Gp, best of G1–G5** | **1,054 (96.9%)** | 96.7% | — |
| G6 brute force, every compatible support (only when ≤ 20,000) | 933/933 (100%) | 100% | 274 |

- **The 34 steps the portfolio misses** are all accepted, non-final steps. In every one, the MIQP's support is large: 7–19 meters. The number of compatible supports ranges from 747 to 524,267.
- **Cost of a miss:** the median gap is 0.0036, about one branch price; 19 of 34 exceed one α; the maximum is 0.136. After zero cleanup, the MIQP's addition is better in all 34.
- **Where G6 could run**, it found the MIQP's result in all 23 of those misses. Post hoc (not registered), Gp + G6 reaches 1,077 of 1,088 (98.99%), just under the 99% bar.
- **G5 is the strong generator.** It misses large supports because of its top-50 cut.
- Generators beat the MIQP's "optimal" step in 3 steps (gaps 0.0003, 0.007, 0.054): the false-certificate problem of §1.
- **Final no-gain check.** On the last W step of each run, 68 MIQP calls proved that no compatible support adds a gain. The generator portfolio also found none in 67 of those 68.
- **Time:** the W MIQP's median step is 6.7 s; scoring every generator set takes 1.1 s.

## 4. Experiment B2: truth-removed pools (can W find branches nobody proposed?)

Every true support of size ≥ 3 was removed from the RNJ pool, the trees and the reference family. Each input was then run in demeaned form with three W variants. In total, 740 supports were removed from 128 inputs.

| W variant | Recovered in the selected family | Recovered in WZZT's output | Exact | Precision of the selected family | Median run time |
| --- | --- | --- | --- | --- | --- |
| MIQP W (as now) | 321 (43.4%) | 382 (51.6%) | 9/128 | 0.69 | 88 s |
| Pool W (pool trials only) | 166 (22.4%) | 190 (25.7%) | 6/128 | 0.37 | 31 s |
| Generator W (pool + G2–G5 at every step) | 323 (43.6%) | 386 (52.2%) | 8/128 | 0.69 | 46 s |

- **Generator W against MIQP W** (paired):
  - selected: +2 supports (5 inputs worse, 3 better), Wilcoxon p = 0.94;
  - output: +4 (5 worse, 4 better), p = 0.85;
  - exact: MIQP only 1, generator only 0, McNemar p = 1.
- **Reaching outside the pool is what matters.** Without it, recovery halves and precision drops from 0.69 to 0.37. The cheap generators reach outside the pool as well as the MIQP does, in about half the time.
- **By condition** (output recovered): real 176 → 180 of 372, `lock_short01` 120 → 120 of 184, `lock_noisy` 86 → 86 of 184 (MIQP W → generator W).

## 5. Experiment F: speed of the compressed E MIQP

| Version, run settings (MIPGap 1e-4, the run's start and time allowance) | Median speed-up over the run's MIQP | Same objective as the run |
| --- | --- | --- |
| QR rows (registered F) | 2.4× (range 0.1–110); real 4.2×, `lock_short01` 1.3×, `lock_noisy` 1.7× | 242/260; 11 false "optimal" stamps, all on levels data |
| SVD rows + `NumericFocus=2` (F2, D1) | 1.6× (range 0.0–102); real 3.2×, `lock_short01` 0.8×, `lock_noisy` 1.0× | 256/260; SVD better 3, original better 1 |

- **The QR version is not exact.** It fails the registered exactness check on 18 of 260 instances, so its speed-up cannot be claimed.
- **The robust SVD version** is faster on the real feeders and no faster on the synthetic ones. It is slower on 106 of 260 instances.
- So compression alone is not a speed result. The speed case for S1/S2 has to come from reducing the history (Phase 2), not from this compression.

## 6. What this means for WZZT

- **E stage: the MIQP is replaceable as the solver and moves to the certificate role** (D-E, confirmed by D3).
  - A portfolio of three heuristics finds the optimum on 258 of 259 instances: beam search (width 20), local search from each RNJ tree, and local search from random laminar starts.
  - The margin over the registered threshold is one instance, so this is a pass, not a landslide.
  - It brings no speed gain: the portfolio is slower than the MIQP at the median.
  - The case for the change is the design: heuristics search, the MIQP certifies, and each catches the other's failures.
  - The certificate step must be confirmed. One solve warm-started at the heuristic's answer, plus one cold start. If they disagree, escalate: longer solve, report the gap.
- **W stage: the MIQP keeps its role.**
  - Step by step, it finds better large supports that the cheap generators miss (3.1% of steps).
  - Brute force matches it whenever enumeration is small enough, so the MIQP is what makes the exhaustive search possible at scale.
  - At the outcome level (B2), the generators match it. A defensible design is generator-first W with the MIQP as the final no-gain certificate. That needs its own registered test (Phase 2); D-W as registered keeps the MIQP.
- **Novelty item (c), exact certificates,** holds only with the robust formulation and a confirmation step. The default formulation's "optimal" stamps were wrong on 2 of 255 E instances, and the QR version's on 11 of 257. This needs a short methods paragraph: why the pool design is rank-deficient, how SVD rows and stronger numerics fix it, and how certificates are confirmed.
- **The search is not the bottleneck on real feeders.** The run's MIQP, the certified MIQP, the portfolio, beam search (H5) and random multistart (H7) each reach the optimum on 128–131 of 131 real instances. The limit there is the criterion and the resolution floor (finding 4), as before.

## Files

- Outputs (not in Git): `outputs/wzzt_evidence_20261124/`. First run: `a.jsonl`, `b1.jsonl`, `b2.jsonl`. D1: `a2.jsonl`. D3: `a3.jsonl`. Also `summary.json`, logs, and the first-run copy in `backup_run1/`.
- To reproduce: see [README.md](README.md). Then:

  ```sh
  python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 a2
  python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_phase1 a3
  python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.analyse_phase1
  ```
