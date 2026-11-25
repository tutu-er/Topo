# WZZT-C confirmation (registered 2026-11-25, before any confirmation run)

**Decision this test makes (the user's choice of 2026-11-25: "WZZT-C, confirm it").** The paper presents WZZT-C as
its identification method, with WZZT-R as the confirmed reference, if WZZT-C is non-inferior to WZZT-R and better
than the baselines on fresh inputs (§4). Otherwise the paper keeps WZZT-R, and WZZT-C is reported as the fast
certified variant.

## 1. The frozen method

WZZT-C as in `rnj_wzzt_dr/identification/combo.py` (`fit`), with the settings of `run_confirm_c.WZZT_C`:
- Model, residual operator (per-meter offsets, GLS slot term) and BIC score of WZZT-R.
- **Sources** (proposals only): RNJ, NJ, RG (tolerance 0) on the P-only matrix (Gurobi QP), and six clusterings
  (raw V², V² after M, star residual, matrix rows, UPGMA and Ward on the path distance).
- **Joint selection:** the exact pool MIQP over every source line → T*.
- **Blocks:** lines of T* whose evidence margin against removal and against every crossing candidate is at least
  **m = 4.6** half-BIC units (a Bayes factor of 100), from the largest down, no size limit.
- **Root search:** exact pool MIQP over the lines inside the blocks and every union of 2 or more blocks, warm-started
  at T*; skipped when more than k_max = 9 blocks remain.
- **Certificate and repair:** node-by-node MIQP; single-line repairs (at most 10), then pair moves inside a node
  (at most 3 rounds), then one re-selection with the added lines; the final node-by-node check gives `certified`.
- Time limits: 240 s per pool MIQP, 60 s per node MIQP, 30 s local search. A time-limited MIQP keeps its best
  solution; its status is recorded.

Development record (130 development inputs, `WZZT_C_REVIEW.md`, tag `ev_pairs`): synthetic exact 33 against
WZZT-R's 32; real supported lines 224 against 228 of 254; precision 90.5% against 91.3%; median 3.0 s against 85 s;
certified 130 of 130.

## 2. Comparators

- **WZZT-R**, frozen (D2), ordered fits by Gurobi (deviation D4, `qp_solver = "gurobi"`).
- **The original WZZT** (`O_dm`, demeaned), synthetic inputs only (its cost grows steeply with the number of
  meters).
- **NJ, RNJ, RG** on WZZT-R's P-only matrix, at settings fixed from the development analyses: real networks RNJ
  0.08, NJ 0.04, RG 0.08; synthetic feeders 0.24 for all three. Every other tolerance (0, 0.04, 0.08, 0.16, 0.24) is
  also run and reported as secondary, including each builder's best setting chosen after the fact.

## 3. Inputs (never used by any method's development)

- **Synthetic:** 192 inputs, the four feeders (soumalas11, pengwah18, flynn16, paper15) in the two Phase 3
  conditions (`lock_short01` settings: one 0.01 Ω segment; `lock_noisy`: meter stress), seeds 161–184, one day of
  history (96 training + 48 validation quarter-hours). Built by `build_confirm.py`.
- **Real (large):** `eulv_large`, the 17 transformer-level networks (26–77 meters) and 6 large feeders of the EU LV
  test system, start days 0 and 6, 6 days of hourly history (46 inputs from 23 networks). WZZT-R's results on these
  inputs have not been examined line by line; WZZT-C has never run on them.
- Truth: the identifiable truth (segments below the resistance floor merged). *Supported* lines: truth lines whose
  removal raises the study judge's BIC (dJ > 0, the Phase 3 ceiling routine with the input's own settings).

### Correction to §3 (2026-11-25, runs in progress, no result read): the large real inputs are NOT independent

The sentence "never used by any method's development" and "WZZT-C has never run on them" are wrong for the large real
inputs. What is true (checked by meter identity, not by results):
- The transformer networks contain the development feeders. Matching meters by their recorded load profile
  (correlation > 0.98, same transformer, same start day 0 or 6): 724 of the 2,034 meters in the 46 inputs (36%) are
  meters of the 33 feeders used in development, on the same days; up to 94% in one network. Only 13 of the 46 inputs
  (7 networks: `td006145_feeder24`, `td008690_feeder18`, `td008742_feeder80`, `td009451_feeder140`,
  `td400079_feeder144` (day 0 only), `td400859_feeder31`, `td401346_all`) contain no development meter.
- WZZT-C was run before the registration on `td400291_all_s0` (a feasibility probe, finished) and started on
  `td008690_all_s0` (stopped); only statuses and timings were printed, no accuracy.
- Scratch benchmarks (formulation speed and certificates on the true tree) used eight of these networks, and the
  stopped anatomy stage had finished 18 of the 46 inputs (unread). These are not accuracy tests of WZZT-C, but they are
  contact with the inputs.
- WZZT-C's settings (m, sources, pair moves) were chosen on feeders that are parts of these networks.

The synthetic inputs (seeds 161–184) have new loads, meter errors and noise, but **not new topologies**: the 192 inputs
hold only 4 distinct (feeder, true tree) pairs, 48 inputs each, and all 4 are also in the development set (checked by
identity, no result read). "Never used" is true of the seeds, not of the trees. WZZT-C's settings and the baselines'
synthetic settings were chosen on the same four trees. C1a and C2a therefore test new data on known topologies; the
bootstrap over inputs treats 192 inputs as independent and is too narrow for any claim about new topologies. The two
sets together hold 4 + 7 distinct topologies, of which only the 7 clean networks are outside development. This is a limit of the test, not a change of its rules; any claim must
say "new data on known topologies" for the synthetic part.

Consequence, fixed now: the analysis reports C1b and C2c on all 46 inputs, labelled *overlapping with development*, and
on the 13 inputs without development meters, with the same formulas. A "confirmed" verdict does not rest on the 46-input
real results alone. Whether the real part may count toward the decision, and how, is the user's call.

**User's decision (2026-11-25, runs in progress, no result read): "Only the 13 clean".** The real part of the decision
uses only the 13 inputs without development meters:
- C1b and C2c are decided on those 13 inputs; the results on all 46 are descriptive.
- C2b is decided on the 192 synthetic inputs plus the 13 (205 inputs), not on all 238.
- C1a and C2a are unchanged. *Confirmed* still needs all five parts. Margins and thresholds are unchanged.
- A failed job among the 13 blocks C1b until the comparators are rerun on it, as in §4.

What this costs, stated in advance: the 13 inputs come from only 7 networks, so the cluster bootstrap has 7 clusters. Its
interval is coarse and probably too narrow, and the part has low power: "not shown" here would be weak evidence of
anything. Nine of the 13 were also touched by the unread anatomy stage, two of those nine (`td401346_all`) by speed
benchmarks too; the four inputs of `td009451_feeder140` and `td400859_feeder31` were not. None was used to choose
WZZT-C's settings. The large run was restarted only to process these 13 first (§5 order; the
frozen methods and settings are the same).

## 4. Endpoints and decision rules

Errors and crashes count against the method that fails (an empty output). An input whose job fails as a whole
(the runner cannot tell which method) counts as a failure of WZZT-C: the comparators are rerun on it alone and
keep their results, and WZZT-C gets an empty output. Until that rerun, C1a scores it as WZZT-C wrong and WZZT-R
right, and C1b is not shown. (Clarification added before any result.)

**C1, non-inferiority to WZZT-R (both parts needed):**
- *C1a, synthetic:* exact recovery rate, WZZT-C minus WZZT-R, paired over the 192 inputs; the lower end of the 95%
  bootstrap interval (10,000 resamples of inputs, percentile) must exceed −5 percentage points.
- *C1b, real:* on `eulv_large`, pooled recall of supported lines and pooled precision, WZZT-C minus WZZT-R; for
  both, the lower end of the 95% cluster-bootstrap interval (10,000 resamples of networks) must exceed −5 points.

**C2, better than the baselines (all parts needed):**
- *C2a:* synthetic exact recovery, WZZT-C against the original WZZT: one-sided exact McNemar, p < 0.05.
- *C2b:* per-input F1 over all truth lines (all 238 inputs), WZZT-C against NJ, RNJ and RG at their fixed settings:
  one-sided Wilcoxon signed-rank, Holm-adjusted over the three, each < 0.05.
- *C2c, root lines:* on `eulv_large`, per-input recall of truth lines with more than half the meters downstream,
  WZZT-C against NJ, RNJ and RG at their fixed settings: one-sided Wilcoxon, Holm over the three, each < 0.05.

**Confirmed** when C1a, C1b, C2a, C2b and C2c all hold.

**Secondary (reported, no decision):** median and maximum time; share certified; share of pool MIQPs proved;
the score comparison with WZZT-R on the same design (better / worse / tie); recall by line class (near root,
terminal side; short, long) and by network size; WZZT-R against the builders on root lines (a replication of
Amendment D's HR1 on the large networks); every builder setting and the after-the-fact best.

## 5. Execution

`run_confirm_c.py --set synthetic` and `--set large` (3 workers), then `analyse_confirm_c.py`. Outputs in
`outputs/wzzt_evidence_20261124/confirm_c/` (not in Git). Nothing in §1–4 changes after the first run.

## 6. Log during the runs (no result read)

- 2026-11-25: the synthetic run is complete: 192 of 192 rows (96 names in 2 conditions), no errors. Its results have not been read.
- 2026-11-25: the large run has one failed job so far, `eulv_td400859_feeder31_s0` (one of the 13 clean inputs). The
  traceback is inside `revised.extend_miqp` (Gurobi returned no MIPGap), which only WZZT-R calls (`combo.fit` does not).
  The job wrote no result for any method. Nothing in the frozen code was changed. How this job counts is the user's
  decision, because two sentences of section 4 point to different outcomes: "errors and crashes count against the method
  that fails" (then WZZT-R gets an empty output) and "an input whose job fails as a whole counts as a failure of WZZT-C"
  (written for the case where the runner cannot tell which method failed; here the traceback can). Until then C1b stays
  undecided, and the analysis script is not run. Either way the comparators must be rerun on this input alone.
- **User's decision (2026-11-25, no result read): "just WZZT-C".** The failed clean input `eulv_td400859_feeder31_s0` counts against
  WZZT-C (an empty WZZT-C output; the comparators keep theirs), as the second sentence of section 4 says. It is applied as
  registered: after the large run, the comparators are rerun on this input alone, WZZT-C's result there is empty, and C1b is
  then decided. This is the strict reading, chosen before any result was read. (The run itself is unchanged.)
