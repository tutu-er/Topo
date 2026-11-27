# Development results (to 2026-11-27)

Everything here is development: no registered claim. Each section says which code produced it. The truth is the
identifiable truth (lines of at least 3 milliohm); F1, precision and recall are of the tree's meter groupings.

## 1. The registered test of the predecessor (2026-11-26; the set is spent)

The registered predecessor (`identification/final.py` at 71684ff) was run once on the sealed set: 240 cases on 60 new
random feeders, by the registered protocol (`rnj_wzzt_dr/experiments/case_library_20261125/PROTOCOL_ID.md`; full
report `RESULTS_SEALED.md` there). Feeder-cluster bootstrap within size class, Holm over H1 to H4 at 5%. F1 in points.

| | Comparison | Cases | Estimate | 95% interval | Holds |
| --- | --- | --- | --- | --- | --- |
| H1 | predecessor without PCC minus RNJ, no unmetered load | 60 | +26.9 | [23.4, 30.4] | yes |
| H2 | predecessor with PCC (load as a leaf) minus RNJ, with a load | 180 | +29.3 | [26.0, 32.7] | yes |
| H3 | with PCC minus without PCC, same cases | 180 | +8.7 | [6.9, 10.7] | yes |
| H4 | true load place within 4.6 of the best; false alarms without a load | 180 + 60 | coverage 96.7%, false alarms 0% | [93.9, 98.9]; [0, 0] | yes |

- Pooled F1: 70.1 without a load (precision 90.7, recall 57.1), 72.1 with a load (precision 91.5). RNJ: 38.2 and 37.7.
- The gap to the distance methods grows with feeder size (no load, small to huge: 77.5, 75.6, 70.9, 62.9 against
  RNJ's 60.9, 52.3, 36.7, 29.3).
- The predecessor's local margins were weak: only 78% of lines with a margin between 4.6 and 10 were true. This is
  what the labels of the present method (flip tests) are for.
- **This result does not carry over** to the present method (load gate, flip-test labels, anchored rounds, the solver
  fix) or to probing. The set is spent; a registered test of this method needs a new sealed set (PLAN.md, step 9).

## 2. Algorithm 1 before the solver fix (code ef9ec75, 2026-11-26): 240 twins with at most 25 meters

**Run.** `dev_v8.py run --max-n 25` (now `identify_cases run --max-n 25`): the 240 development twins with at most
25 meters, with the PCC (the gate decides). Code frozen at ef9ec75 (a separate checkout, so later edits could not
touch the run). Eight workers, about 1 h 45 min, 0 errors. (The run of the 48 larger twins was stopped at 10 of 48 when
the solver fault of section 3 was found.)

### Against the registered predecessor (same cases, both with the PCC)

The predecessor's numbers are S2 of `final_source.jsonl`. The truth is the identifiable truth.

| Cases | F1, v12 | F1, predecessor | v12 better / equal / worse |
| --- | --- | --- | --- |
| named feeders, 4 x 48 | 0.924 to 0.931 | 0.926 to 0.933 | 0 / 189 / 3 |
| random small and medium, 4 x 12 | 0.729 to 0.778 | 0.715 to 0.778 | 3 / 45 / 0 |
| all 240 | 0.890 | 0.890 | 3 / 234 / 3 |

- The trees are the same in 234 of 240 cases.
- What v12 adds is the labels and the load gate.
- It is slower: median 57 s per case against 9 s, maximum 4,040 s, because of the flip tests.
- **Gate:** it modelled the load in 180 of the 180 cases with a load and in 0 of the 60 without.
- **Stops:** 220 runs stopped by the stopping rule (Lemma 3) and 20 at the round limit.

### Labels

| Groupings of the tree | True | False |
| --- | --- | --- |
| decided | 1,475 | 3 (0.2%, in 3 cases) |
| open | 137 | 35 |
| unsettled (flip test at its time limit) | 82 | 6 |

| Open groupings outside the tree | True | False |
| --- | --- | --- |
| nested in the tree (one more line) | 235 | 350 |
| crossing the tree | 0 | 9 |

- **Decided labels are reliable:** 0.2% of them are wrong, against about 1% if read as a Bayes factor of 100. This is
  much better than the predecessor's local margins on the sealed set: only 78% of lines with a margin of 4.6 to 10
  were true (section 1).
- **Open is not a "no":** 80% of the open groupings inside the tree are true lines.
- **The trees miss 434 true groupings.** 235 of them (54%) are labelled open, as lines nested in the tree.

### What this says for the proposals (PROPOSALS.md)

- **SR1 passes** on these 240: the true tree is flagged by the S1 F-test in 0 cases (criterion: at most 2%). The leader
  is flagged in 1.
- **SR3, start rule:**
  - "No target" (every grouping decided, nothing open outside) happened in 25 cases.
  - None of the 25 has a wrong decision; 22 have a perfect tree.
  - The other 3 are the three twins of `flynn16_street_l6`. Each misses 4 true lines with no open alternative,
    probably lines the data cannot see; this still needs a look.
  - Decided groupings are wrong in 0.2% overall (criterion: at most 2%).
- **SR4:** pricing was complete in 240 of 240 cases, so S2 never blocks here and this set cannot test it.
- **TR2 and TR4, target rule:**
  - Open nested groupings average 2.4 per case, and 40% of them are true lines the tree lacks: 235 lines in 240 cases.
  - The TR4 criterion for T1 ("at most 1 such line per 50 cases") fails by far. So T1 would drop the targets most
    worth probing, unless probing cannot recover them under T0 or T2 either. That needs probing runs.
  - The simulated day on `dev_small_003_head` already showed one: (8, 9), true and nested, was T0's first target.

### Builders (`ptid/method/builders.py`, rewritten 2026-11-26)

The nine starting trees were rewritten inside the package, replacing `candidates.py` and its use of `identification/`.
On all 240 twins with at most 25 meters, with the load modelled exactly when the gate says so, every one of the nine
trees is identical to the predecessor's.

- One recursive-grouping tree differed at first, from a rounding difference of 1e-16 in how the load's profiles were
  projected out. The rewrite now does those operations in the predecessor's order.
- Algorithm 1's whole output on two cases is identical to the frozen ef9ec75.
- The 48 larger twins (26 to 60 meters) give identical trees too, so all nine builders match the predecessor on all
  288 twins. The rewrite is also faster there: 441 s against 788 s in total.

## 3. The solver fix (2026-11-26/27)

The run of the 48 larger twins was stopped at 10 of 48. Huge twins (47 to 60 meters) took 2.6 to 3.6 hours each, about
80% of it in flip tests that ended at their limit. Looking into why showed three problems.

**1. False solver claims.**
- All 11 anchored MIQPs that reported "optimal" on the huge twins had a bound above a tree in their own search space,
  by up to 82,000 score units.
- The Gurobi log of `dev_huge_004_head`'s first tree shows the root relaxation declared "infeasible" while Gurobi held a
  feasible solution, which it then called optimal.
- Cause: tolerances forced to 1e-9 (NumericFocus 2) on a badly scaled problem. The matrix coefficients spanned 1e-13 to
  2e3. The per-meter bounds are valid: the run's own tree satisfies all 120 of them.
- On the huge twins (old code), 4 of 35 decided groupings were wrong, against 3 of 1,478 below 26 meters. Whether these
  came from false proofs or from pools that lack the true lines is not separated.

First-tree MIQP on `dev_huge_006_none`, 240 s, started from the 40 s local-search tree. A claim is false if its bound is
above the best tree known (the converged local search, score 5,661.4):

| Setting | Outcome | Bound below the best known tree |
|---|---|---|
| tolerances forced to 1e-9 (the old code) | "optimal" in 10 s: false | none (bound above it by 69) |
| forced 1e-9 with NumericFocus 3 | consistent | 241.5 |
| Gurobi's defaults | consistent | 222.4 |
| defaults, coefficients scaled to their columns | consistent | 171.5 |
| forced 1e-9, coefficients scaled | consistent | 165.4 |

With scaled coefficients, `dev_huge_004_head`'s root relaxation solved in 19 to 79 s (unscaled: "infeasible" or
"numerical trouble"). Its claims were consistent at both tolerance settings, with bounds 518 to 525 units below the best
known tree.

Other formulations, not adopted:
- **Residual per meter and hour** (eq. 9 written directly; range 3e-7 to 1e2): also a false optimum at forced
  tolerances on `dev_huge_004_head`, and its root relaxation did not finish in 240 s on `dev_huge_006_none`.
- **MILP with cuts below the same quadratic:** not built. Its relaxation cannot be tighter than the quadratic's own.

**2. Local search stopped by its time limit.** At 40 s per start, the search made one or two steps. On the first pool,
the local-search tree was far from a local optimum:

| Twin | Paying single swaps left | Useless lines in the tree | To convergence | Score gained |
|---|---|---|---|---|
| dev_huge_006_none | 28 | 0 | 8 steps, 52 s | 69 |
| dev_huge_004_head | 104 | 35 (each saves its price) | 25 steps, 346 s | 398 |

In 240 s the MIQP did not find even the 69-unit improvement.

**3. No proof is possible at this size.** Even when consistent, the MIQP's bound stays 158 to 525 score units below the
best tree, so flip tests on 47-60 meter feeders can rarely decide a grouping. There, the trees rest on the local search.

**The fix:**
- Gurobi's default tolerances;
- coefficients scaled to the size of their columns;
- local search to convergence before and after the MIQP, so every candidate line's value is checked and no single
  removal or swap pays;
- every MIQP always started from the converged tree;
- every claim checked against the trees known. A bound above a known tree's score, or an "optimum" that one move still
  improves, is a numerical failure: no bound, nothing decided.

First-tree step with the fix (local search, MIQP, local search again):

| Twin | Score (run's 40 s pass, then fixed) | MIQP | Bound below the tree | Time |
|---|---|---|---|---|
| dev_huge_006_none | 5,730.4, then 5,661.4 | time limit, consistent | 158 | 687 s |
| dev_huge_004_head | 7,332.3, then 6,934.7 | time limit, consistent | 506 | 980 s |

Those two rows are without the scaling. With the scaling too (the committed code):

| Twin | Score | MIQP | Root relaxation | Bound below the tree | Time |
|---|---|---|---|---|---|
| dev_huge_006_none | 5,661.4 | time limit, consistent | solved in 8 s | 177 | 680 s |
| dev_huge_004_head | 6,934.7 | time limit, consistent | solved in 26 s | 525 | 943 s |

The scaling does not reliably tighten the bound (177 here against 158 unscaled, both with the per-meter bounds). Its value
is robustness: no false claims even at forced tolerances, and a root relaxation that always solves.

## 4. Algorithm 1 with the fix (code c5c4aae = ptid, 2026-11-27)

**Status:** both runs are complete.

| Run | Code | Cases | State |
| --- | --- | --- | --- |
| development, one twin per feeder | c5c4aae (frozen checkout) | 72 (PROTOCOL.md §2.2) | complete, 2026-11-27 00:50 to 05:25, 0 errors |
| real feeders | `ptid` | 66 (PROTOCOL.md §2.3) | complete, 2026-11-27 00:55 to 02:00, 0 errors |

**The package gives the same results as the code it was moved from.** On the first 20 real inputs that both the old
code (c5c4aae, `identification_v8`) and `ptid` finished, the tree, every label and the stopping reason are identical in
20 of 20. Of the two refinements of CHANGES.md §3, probing's check is never used by identification, and the node MIQP's
scaling is an exact change of variables that gave the same line as enumeration wherever it was compared (section 5.1).
So the development run made with c5c4aae is taken as a run of `ptid`. The one place the two codes could differ is a
node MIQP on a node wider than 9 children: in the 72 cases that happened twice, both on `dev_huge_004_end`, and the
recheck with `ptid` gives the same line and gain there (section 5.1).

### 4.1 Real feeders (66 inputs, complete; `ptid`, 0 errors; equal to c5c4aae on the 25 inputs both ran)

| Measure | Value |
| --- | --- |
| F1 / precision / recall (identifiable truth) | 0.669 / 0.760 / 0.625 (case means) |
| precision against the exact truth (is every grouping found a physical line?) | 0.826 |
| groupings of the trees: decided / open / unsettled | 162 / 192 / 29 |
| decided groupings not in the identifiable truth | 4 (2.5%) |
| decided groupings that are not physical lines at all | **0 of 162** |
| solver outcomes, anchored MIQPs | 74 optimal, 12 at the time limit, 0 numerical failures |
| solver outcomes, flip tests of the final labels | 155 cut off (decided), 7 infeasible (decided), 44 optimal, 36 at the time limit, 141 settled by local flips; 0 numerical failures |
| stop | 56 converged (Lemma 3), 10 at the round limit |
| S1 adequacy (1% level) | identifiable truth flagged in 2 of 66, exact truth in 0, found tree in 0 |
| S2 pricing complete | 66 of 66 |
| time per input | median 33 s, largest 2,075 s |

**The four "wrong" decided labels are physical lines.** Each of the four groupings is a line of the network: it is in
the exact truth, but its segment is shorter than the 3 milliohm floor below which the scoring truth merges lines. The
data see these lines anyway: the true (identifiable) tree scores 14 to 38 units worse than the found tree under the
method's own score in all four cases.

| Input | Grouping | H(identifiable truth) - H(found tree) |
| --- | --- | --- |
| eulv_td000653_feeder15_s0 | meters 2-7 | 38.1 |
| eulv_td000653_feeder15_s6 | meters 2-7 | 34.9 |
| eulv_td004454_feeder159_s0 | meters 12-14 | 14.5 |
| eulv_td008690_feeder19_s0 | meters 1-11 | 14.8 |

The same explains test S1's two flags: both are `feeder15`, whose identifiable truth has 2 lines where the network has
5 that the data see. Its exact truth passes (p = 0.86 and 0.62), and on day 0 the found tree is the exact truth. So
the flags came from the scoring truth, not from the model's fit to real feeders, as first suspected.

**Consequence for the measurements.** A decided label is wrong only if its grouping is not a physical line; the
3 milliohm floor stays the scoring truth for recall (lines the data cannot be expected to see). Both are reported from
now on (`identify_cases summary`: columns `wrong` and `not a line`). This was decided after seeing these results
(PROTOCOL.md §4 says so).

For comparison: on the same 66 inputs the earlier study's method got F1 62.5 with the plain BIC price and 66.7 with
the structural prior, and RNJ 41 at its best tolerance (`rnj_wzzt_dr/experiments/case_library_20261125/IDENTIFICATION_PLAN.md`).

### 4.2 Development twins, one per feeder (72 cases, complete)

| Kind | Cases | F1 | Precision | Recall | Decided | Wrong | Not a line | Open | Unsettled | Median time | Largest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| star | 24 | 1.000 | 1.000 | 1.000 | 138 | 0 | 0 | 0 | 0 | 41 s | 114 s |
| street | 24 | 0.856 | 0.985 | 0.761 | 188 | 0 | 0 | 26 | 0 | 68 s | 298 s |
| small | 6 | 0.776 | 0.944 | 0.664 | 21 | 0 | 0 | 7 | 0 | 31 s | 104 s |
| medium | 6 | 0.678 | 0.831 | 0.593 | 15 | 0 | 0 | 13 | 27 | 43 min | 77 min |
| large | 6 | 0.681 | 0.924 | 0.542 | 2 | 0 | 0 | 9 | 73 | 1.5 h | 3.3 h |
| huge | 6 | 0.684 | 0.892 | 0.559 | 3 | 0 | 0 | 13 | 107 | 4.0 h | 5.1 h |
| all | 72 | 0.854 | 0.961 | 0.784 | 367 | **0** | **0** | 68 | 207 | 75 s | 5.1 h |

- **Solver:** 0 numerical failures. Anchored MIQPs: 60 optimal, 30 at the time limit. Flip tests of the final labels:
  351 cut off and 16 infeasible (decided), 12 optimal, 207 at the time limit (unsettled), 56 settled by local flips.
- **S1:** the true tree is flagged in 0 of 72; the found tree in 0. **S2:** pricing complete in 71 of 72.
- **Stop:** 56 by the stopping rule (Lemma 3), 16 at the round limit (every medium, large and huge case but two).
- Times are upper bounds: the real-feeder run and checks shared the machine for the first hour.

**Against the old run, on the 63 cases both ran** (old: ef9ec75 for the twins up to 25 meters, before the fix; the
stopped run for 3 huge twins, under the faulty solver):

| Kind | Cases | Same tree | F1 old → new | Decided not a line, old → new |
| --- | --- | --- | --- | --- |
| star, street, small | 54 | 54 | unchanged (1.000, 0.856, 0.776) | 0/347 → 0/347 |
| medium | 6 | 5 | 0.699 → 0.678 | 0/20 → 0/15 |
| huge | 3 | 0 | 0.681 → 0.638 | 1/15 → 0/0 |

Where the tree changed, the method's own score tells which search did better:

| Case | H new − H old | H(true tree) − H new | F1 old → new |
| --- | --- | --- | --- |
| dev_medium_003_middle | −0.4 (new better) | +31.0 | 0.375 → 0.250 |
| dev_huge_001_none | −2.0 (new better) | +51.7 | 0.776 → 0.735 |
| dev_huge_006_head | −1.2 (new better) | +67.5 | 0.706 → 0.680 |
| dev_huge_004_end | **+70.5 (new worse)** | **−40.0** | 0.563 → 0.500 |

In three of the four the new tree scores better and the true tree scores clearly worse than either: the lower F1 is
the model's preference, not a search failure. **`dev_huge_004_end` is a search regression**, explained below.

**Criteria of PROTOCOL.md §4:**
- no contradicted claim used: **met** (0 numerical failures);
- decided labels wrong at most 2%: **met** (0 of 367 are not physical lines; 0 are outside the identifiable truth either);
- F1 below 26 meters not lower by more than 0.01: **met** (mean change −0.002 over the 60 comparable cases; the medium
  kind alone −0.021, from the one case above).

**What the honest solver shows on feeders of 26 to 60 meters:**
1. **Few decided labels.** Over the 12 large and huge twins, 5 groupings are decided, and 180 are unsettled: their
   flip tests reach the time limit.
2. **Anchors appear, then vanish.** Round 0 often decides a few groupings (13 on `dev_large_004_end`, 11 on
   `dev_huge_001_none`), but the anchored search adds hundreds of candidate lines. In the next round the same flip
   tests, now over the larger pool, reach their limit. Before the fix, false proofs kept the anchors.
3. **So the anchored search rarely widens the pool.** On `dev_huge_004_end` round 0 decided nothing, so no unions were
   offered. The old run's anchors, from false proofs, had opened 550 unions, and its tree scores 70.5 units better. The
   new final tree still has a paying single line (17.6 units): pricing adds at most 10 lines per round, and the round
   limit came first.
4. Time per huge case is about 4 hours, most of it in flip tests that end unsettled.

These are limits of the method at this size, not faults: every claim it makes is correct. PLAN.md (step 4) lists ways
to improve the search there without weakening the labels.

**The old run's 6 decided groupings that are not lines:** 4 were on huge twins under the faulty solver, and on the
huge twins rerun nothing is decided now. The other 2, on `dev_medium_003_head` and `_end`, are not in the 72 (that
feeder's twin here is `_middle`).

## 5. Checks

### 5.1 Wide nodes and the node MIQP

With the old switch (enumerate up to 12 children, node MIQP beyond), the node MIQP never ran on the 248 twins finished so
far (the 240 small to medium twins and the first 8 huge ones). Widest node per tree:

| Twins | Found trees | True trees |
|---|---|---|
| star, street (192) | 3 to 4 children | 2 to 4 |
| small, medium (48) | 3 to 8 | 3 to 7 |
| huge, first 8 | 7 to 11 (two trees with 11) | 5 to 6 |

A found tree with a node wider than any branching point of the feeder is missing lines there. Pricing now enumerates
nodes up to K = 9 children, the same K as the anchored universe, and uses the node MIQP beyond. Checked on the final trees
of the two huge twins with an 11-child node (`ptid.runs.checks wide-nodes`; pricing only):

| Twin | Pricing by enumeration | Pricing with the node MIQP | Node MIQP at the 11-child node |
|---|---|---|---|
| dev_huge_004_head | 73 s | 11 s, same best line and gain | optimal in 8 s |
| dev_huge_004_none | 63 s, no line pays | 13 s, no line pays | optimal in 9 s |

Both are exact, so results change only if a node MIQP stops at its 60 s limit, which is reported as an unproved node.

**With the node MIQP's coefficients scaled (ptid, 2026-11-27),** on the same trees (`ptid.runs.checks wide-nodes` on the
huge twins of the stopped run; 8 of 10 trees have no node wider than K = 9 and are enumerated):

| Twin | Enumeration | Scaled node MIQP |
|---|---|---|
| dev_huge_004_head | best line of 20 meters, gain 3.04e-3 (crit), 90 s | the same line and gain, proved, 13 s |
| dev_huge_004_none | no line pays, 67 s | no line pays, proved, 12 s |

(The machine was shared with two runs, so the times are upper bounds.)

**In the 72-case run** (c5c4aae, unscaled node MIQP) the node MIQP ran twice, both on `dev_huge_004_end`, and proved
both. Rechecked with `ptid` on that case's final tree (a node of 10 children): enumeration and the scaled node MIQP
give the same best line (55 leaves) and gain (3.05e-3 in crit, 17.6 score units: the paying line left), the MIQP
proved it, 6 s against 23 s. The other five huge trees have no node wider than 9 children.

### 5.2 The nested-pair cap on true trees

See PROPOSALS.md, section 4 (438 extra lines on 24 no-load cases: the cap held for every line; none beat the true tree;
6.2% were labelled open).
