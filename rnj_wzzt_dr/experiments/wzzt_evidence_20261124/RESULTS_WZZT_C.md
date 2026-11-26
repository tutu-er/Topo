# WZZT-C confirmation: results (2026-11-25)

Registered protocol: [PROTOCOL_WZZT_C.md](PROTOCOL_WZZT_C.md), written before any confirmation run. Its corrections (section 3)
and the user's decisions (sections 3 and 6) were logged before any result was read. The analysis (`analyse_confirm_c.py`) was
run once, after these three runs were complete:
- the synthetic run, 192 of 192 inputs;
- the large run, 46 of 46 inputs, 4 of them failed jobs;
- the rerun of the comparators alone on those 4 jobs.

The printout is in `outputs/wzzt_evidence_20261124/confirm_c/analysis.txt` (not in Git).

## Decision

**Not confirmed.** Four of the five parts hold. C1b on the 13 clean real inputs does not. By the registered rule (top of the
protocol), the paper keeps WZZT-R as its identification method and reports WZZT-C as the fast certified variant.

| Part | Result | Verdict |
| --- | --- | --- |
| C1a, synthetic exact recovery, WZZT-C minus WZZT-R (192 inputs; margin −5 points) | 91 against 84; +3.6 points [+1.0, +6.8] | holds |
| C1b, real, the 13 clean inputs (7 networks; margin −5 points) | supported recall 82 against 87 of 110: −4.5 points [−29.4, +16.4]; precision +4.7 points [−1.0, +12.6] | **not shown** (the lower end of the recall interval is below −5) |
| C2a, synthetic exact recovery against the original WZZT | only WZZT-C exact on 42 inputs, only the original on 10; one-sided p = 4.5e-6 | holds |
| C2b, per-input F1 against RNJ, NJ and RG at their fixed settings (192 synthetic + 13 clean inputs) | +23.7, +25.1 and +39.9 points; Holm p ≤ 2e-30 | holds |
| C2c, root-line recall against RNJ, NJ and RG (the 11 clean inputs that have root lines) | +37.3, +44.5 and +61.6 points; Holm p = 0.031, 0.0098 and 0.0029 | holds |

## Where C1b's deficit comes from (descriptive, after the decision; it does not change it)

- **All of it comes from one input.** In the main run, the job for `eulv_td400859_feeder31_s0` ran WZZT-C to the end. WZZT-R
  then crashed (`revised.extend_miqp`: Gurobi gave no MIPGap). The runner kept no result from a job that raised, so WZZT-C's
  finished tree was lost. The user chose the strict reading before any result was read ("just WZZT-C"), so WZZT-C counts as
  empty there: 0 of the input's 15 supported lines. WZZT-R, rerun alone, completed and found 14.
- On each of the other 12 clean inputs, WZZT-C finds at least as many supported lines as WZZT-R. In total it finds 82 against
  73 of 95, with precision 72.9% against 66.7%.
- The interval is wide because the 13 inputs come from only 7 networks. The protocol said so in advance.

| Input | Meters | Supported lines | WZZT-C found | WZZT-R found | WZZT-C output / true | WZZT-R output / true | WZZT-C s | WZZT-R s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| td006145_feeder24_s0 | 31 | 7 | 3 | 2 | 7 / 3 | 7 / 2 | 741 | 530 |
| td006145_feeder24_s6 | 31 | 6 | 5 | 3 | 8 / 5 | 7 / 3 | 781 | 734 |
| td008690_feeder18_s0 | 29 | 7 | 7 | 5 | 8 / 7 | 8 / 5 | 375 | 514 |
| td008690_feeder18_s6 | 29 | 7 | 7 | 7 | 8 / 7 | 8 / 7 | 118 | 410 |
| td008742_feeder80_s0 | 29 | 6 | 5 | 4 | 10 / 8 | 10 / 7 | 500 | 526 |
| td008742_feeder80_s6 | 29 | 7 | 5 | 5 | 8 / 6 | 8 / 6 | 490 | 631 |
| td009451_feeder140_s0 | 44 | 2 | 1 | 0 | 8 / 4 | 9 / 2 | 515 | 1,678 |
| td009451_feeder140_s6 | 44 | 3 | 0 | 0 | 9 / 3 | 8 / 2 | 528 | 1,688 |
| td400079_feeder144_s0 | 34 | 9 | 9 | 8 | 16 / 9 | 15 / 8 | 217 | 666 |
| **td400859_feeder31_s0** (failed job) | 39 | 15 | **0** (empty by the rule) | 14 | 0 / 0 | 19 / 15 | – | 1,511 |
| td400859_feeder31_s6 | 39 | 11 | 11 | 11 | 18 / 13 | 17 / 13 | 625 | 1,177 |
| td401346_all_s0 | 26 | 15 | 15 | 15 | 17 / 17 | 17 / 17 | 22 | 614 |
| td401346_all_s6 | 26 | 15 | 14 | 13 | 16 / 15 | 15 / 14 | 16 | 604 |

The lesson is about the runner, not the method. Methods that share one job must be isolated, so that one method's crash cannot
erase another's result. The runner of the sealed identification test was changed that way before its registration
(`case_library_20261125/PROTOCOL_ID.md`, section 7).

## Secondary (no decision rests on these)

- **All 46 large inputs, which overlap with development (descriptive):**
  - supported recall: 437 against 362 of 606, +12.4 points [−1.6, +25.8];
  - precision: +19.0 points [+1.0, +32.8];
  - root-line recall against RNJ, NJ and RG: +48, +48 and +62 points.
- **Recall by line class on the large inputs (WZZT-C against WZZT-R):**
  - near root: 58.3% against 59.7% (72 lines);
  - terminal side: 42.5% against 35.5% (1,094 lines);
  - precision: 77.5% against 58.5%.

  On the same lines, RNJ (0.08) reaches 16.7% near the root, 28.8% on the terminal side and 23.1% precision.
- **Score on the same design** (lower is better; failed jobs left out):
  - synthetic: WZZT-C better on 19 inputs, worse on 5, tied on 168;
  - large: better on 26, worse on 12, tied on 4 (42 inputs).
- **Certified:** synthetic 192 of 192; large 40 of the 42 inputs where WZZT-C ran.
- **Time** (the two methods ran at different times under different machine load, so this is only indicative):
  - synthetic: median 3.7 s, maximum 33 s, against 84.5 s and 316 s;
  - large: median 539 s, maximum 1,492 s, against 1,434 s and 10,054 s.
- **Builders, pooled F1:**

  | | Synthetic, fixed setting | Large, fixed setting | Large, best after the fact | WZZT-C, synthetic / large |
  | --- | --- | --- | --- | --- |
  | RNJ | 0.656 | 0.253 | 0.262 (0.04) | 0.916 / 0.557 |
  | NJ | 0.653 | 0.261 | 0.261 | 0.916 / 0.557 |
  | RG | 0.503 | 0.115 | 0.128 (0.16) | 0.916 / 0.557 |
- **Crashes:**
  - In the main run, WZZT-R crashed on 3 of the 46 large inputs (twice no MIPGap, once an SVD that did not converge), and
    WZZT-C's set-up on 1 (ordered QP, Gurobi status NUMERIC).
  - In the rerun, WZZT-R crashed again on `td400291_all_s0` (no MIPGap). On `td009451_all_s0` the set-up that every method
    uses failed, so every method is empty there.
  - WZZT-R completed in the rerun on `td400859_feeder31_s0` and `td400079_all_s0`. These crashes depend on the time limits.
  - The case-library study's final method repeats a failed ordered QP with SLSQP (`final.guarded`).

## What this means for the paper

- **On new data from known topologies (synthetic):** WZZT-C is non-inferior to WZZT-R; in fact it is better, since the
  interval excludes 0.
- **Against the other methods:** WZZT-C beats the original WZZT, and it beats the distance builders. It also beats them on
  the root lines of the clean networks.
- **Not shown:** non-inferiority to WZZT-R on new real topologies. This test had low power, and one failed job decided it.
- **Registered consequence:** WZZT-R stays the reference identification method, and WZZT-C is reported as its fast certified
  variant.
- **Open (the user's decision):** whether this changes how the case-library study presents its final method
  (`identification/final.py`, built on WZZT-C's pipeline). That study's sealed test compares the final method with the
  distance methods, not with WZZT-R.
