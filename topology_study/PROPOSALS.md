# Proposals under test (revised 2026-11-27)

This file records proposed start and target rules for probing. Draft v17 uses T0 by default. T1 and T2 are
implemented comparison variants. The cases and pass criteria below are for development evaluation.

## 1. The fact both rules rest on (proved)

**Nested pair.** Let tree B be tree A plus one more line M (A's lines and M are compatible). Then

    H(B) - H(A) = beta - gain/2,   gain = (RSS(A) - RSS(B)) / sigma2 >= 0,   so   H(B) - H(A) <= beta.

*Proof.* B can copy A's fit by setting r_M = x_M = 0, which lies inside the box [0, BOUND]. So RSS(B) <= RSS(A), and B
pays beta for its one extra line. QED.

Consequences:

- (a) On any data, the smaller tree beats the bigger one by at most beta. "M is absent" can be decided (gap > m) only
  if beta > m. On the development cases beta = 5.6 to 7.4 and the prespecified margin m = 4.6 score units,
  leaving 1.0 to 2.8 score units.
- (b) If A generates the observations under the fitted linear model, adding a redundant line can absorb noise.
  How much it absorbs depends on the design and on active coefficient bounds. The deterministic cap is beta.
  In the development configuration, adding 24 hourly rows increases beta by about 0.12 with the penalty settings held fixed.
- (c) If B is true, the rival A can be separated when it cannot represent the response of the omitted line.
  Excitation changes the distinguishable signal defined in Theorem 1a.
- (d) The forecast in Lemma 4 holds the fitted predictions fixed for the next hour and assumes the leader's
  observation model. The next gate refits the trees. In a nested pair the larger tree can reproduce the smaller
  tree's fitted signal. The predicted q/2 and the realised change in their refitted score gap can differ.

## 2. Start rule (proposal S)

On a given day, probe only if all four hold:

| | Condition | Basis | Proved or assumed |
| --- | --- | --- | --- |
| S1 | The leader is adequate: F = [(RSS_L - RSS_sat)/df1] / sigma2_hat is below the 99% quantile of F(df1, df2), where df1 = p_sat - p_L and df2 = the saturated model's residual degrees of freedom | every tree is a special case of the saturated model (all R_ij, X_ij free) | the F distribution holds for a linear Gaussian model with the bounds inactive. **Assumed:** linearisation and PCC errors are small against the noise at dev N (test SR1) |
| S2 | The search is complete: pricing finds no single line that pays | Lemma 3 | proved for single-line additions only; Remark 1 shows a pair can pay when no single line does |
| S3 | At least one target exists (section 3) and some feasible block reaches it | Lemma 5 | proved (reach) |
| S4 | J > 0 for the best block | eq. 15 | a forecast; see 1(d) |

Outcomes:
- If S1 fails, report that the leader failed the fit check and skip probing under this proposal.
- If S3 finds no reachable target, report that outcome with the grouping labels. Unsettled groupings and
  open groupings that feasible blocks cannot distinguish can both remain.

**Open question.** The 4.6-unit margin applies to a grouping flip in its specified candidate pool. Measure
the fraction of decided groupings that disagree with physical lines and the fraction of feeders with at least
one such disagreement. Deriving a whole-tree error bound would require separately validated individual error
bounds or a joint analysis.

## 3. Target rule (proposal T)

Three variants, to be compared:

- **T0 (the draft).** Every open grouping is a target, together with every open load place, valued by eq. 15.
- **T1 (exclude).** Omit a grouping outside the leader and nested in it from the target list. Report its score
  margin. Section 1(a) caps the score advantage of the smaller tree for a single nested addition.
  **Risk:** a weak true line missing from the leader is also nested and outside, so T1 would never probe for it,
  although 1(c) says probing could confirm it.
- **T2 (two-world value).** Keep nested rivals as targets. T2 assigns zero to their leader-side contribution
  and uses kappa(m + gap, q) for the rival side, multiplied by the heuristic score weight 1/(1 + exp(gap)).
  Its value for recovering weak lines is an experimental question (TR3 and TR4).

Groupings of the leader and rivals that cross the leader stay targets in all variants.

## 4. Evidence so far

The check was run 2026-11-26 (`check_extra_lines.py` of the case library; now `python -m ptid.runs.checks extra-lines`).
It covers 24 no-load development cases with at most 20 meters.
Each case uses the true tree plus up to 40 extra lines per node, each refining one node: 438 extra lines in all.

| Quantity | Result |
| --- | --- |
| Cap cost <= beta (section 1) | held for all 438 |
| Extra lines that beat the true tree (cost < 0) | 0 of 438 |
| Extra lines labelled open (cost <= 4.6) | 27 of 438 (6.2%), in 14 of the 24 cases |
| Cost quantiles (0, 10, 50, 100%) | 1.83, 5.08, 5.94, 6.24. The median is about beta: the r, x >= 0 bounds give half the lines no gain |
| Largest voltage change from any extra line | 2.95e-4 pu (median 6.5e-6); all are below 5e-4 pu |
| Smallest true line per case (its own drop) | median 3.0e-4 pu; below 5e-4 pu in 58% of cases |

So the old 5e-4 pu equivalence level cannot mean "absent": it would also remove true lines.

Side finding: in `flynn16_street_l2_none` one true line is fitted at r = x = 0, with an effect of about 1e-22 pu. That
line is invisible to the data; this is the identifiability limit of Theorem 1, and it needs a look.

**Update (2026-11-26):** the development run of v12 on 240 twins (RESULTS.md, section 2) gives first numbers for SR1
(passes: 0 of 240 true trees flagged), SR3 (0.2% of decided groupings wrong) and TR2/TR4 (235 true lines nested and
open, so T1 fails its criterion by far). These numbers predate the solver fix (RESULTS.md, section 3) and are rechecked
on the v16 runs.

**Update (2026-11-27), SR1 on the 66 real inputs:** the true tree is flagged in 2 of 66 (3%, both days of
`eulv_td000653_feeder15`), against the criterion of at most 2%. Both flags come from the scoring truth, not from model
error: that feeder's identifiable truth lacks 3 lines shorter than 3 milliohm that the data see, and its exact truth
passes (p = 0.86 and 0.62; RESULTS.md §4.1). With the exact truth, S1 flags the true tree in **0 of 66** real inputs
(and the found tree in 0 of 66), so SR1 passes on the real inputs when the truth includes every physical line.

## 5. Tests that decide

The pass criteria are written here before the tests run.

| Test | Question | How | Decides |
| --- | --- | --- | --- |
| SR1 | Does S1 flag true trees? | F of the true tree on all 288 dev twins | Pass if at most 2% are flagged (nominal 1%). If it fails, model error is visible at this N: S1 needs a model-error term or is dropped |
| SR2 | Does S1 catch real failures? | the leader's F when a true line is removed from the pool; a second unmetered load; a load at 0.5-1% just under the gate; a biased meter | S1 is useful if it flags at least half of the cases whose decided labels are wrong |
| SR3 | How often does a decided grouping disagree with the reference when probing is skipped? | v8 on dev twins, restricted to cases where S skips probing | Development criterion of at most 2% among decided groupings. Also report the fraction of these cases with at least one disagreement. |
| SR4 | Does S2 add anything? | the wrong-label rate in complete against incomplete cases | keep S2 only if incomplete cases are clearly worse |
| TR1 | The cap holds in code | as in section 4 | **done:** held |
| TR2 | How often are nested rivals open? | as section 4, on all 288 twins (with loads, larger n) and with the real candidate pool | descriptive |
| TR3 | Leader true, nested target: what does probing buy? | a simulated probing day under T0; share decided and money spent | T1/T2 are preferred if T0 spends on these targets with little gain |
| TR4 | Leader lacks a weak true line: what does T1 lose? | v8 on dev twins: count true lines missing from the leader that are nested in it; simulate probing under T0, T1, T2 | T1 is acceptable only if such lines are rare (at most 1 per 50 cases) or T0/T2 recover few of them either; otherwise T2 |

**Prerequisites:** identification with labels on the development cases (done for v13; v16 running), Algorithm 2 in
code (done), the probing world (done: `ptid/cases/world.py`). Missing: probing runs on many development cases and the
psi knee (PLAN.md).

## 6. Related items, also not adopted

- **Stop identification when a round changes nothing** (proposed 2026-11-26). The cheaper stop would use
  unchanged anchors and an empty set of priced additions. Draft v17 also requires complete single-line pricing
  and a closed gap to tolerance for the preceding anchored solve. The v16 logs can be used to count cases in
  which a second round changed the result and to estimate the time saved by the proposed stop.
- **Stop rule for probing:** stop probing after 6 hours without progress (the old planner's `no_progress_hours = 6`).
- **Operational equivalence** (5e-4 pu) for near-twin crossing rivals, so that they do not become targets. Section 4
  shows it cannot be used to call a line absent.
- **Low-belief report in identification** (the F of S1), reported even when no probing is planned.

## 7. Issue found while coding Algorithm 2 (2026-11-26; still open)

**kappa is not monotone in q.** kappa(delta, q) = E min(delta, Z), Z ~ N(q/2, q), first falls and then rises as q
grows when the deficit delta is small. The spread of Z grows like sqrt(q), faster than its mean q/2, and the lower tail
counts in full.

- Numbers from the probing code on `dev_small_003_head`: a target with gap 4.58 (delta = 0.02) has
  kappa = -0.018 passively (q = 0.005) and -0.108 with a 9 kW block (q = 0.15).
- So eq. 15 can give a block a *negative* value for a target that is almost decided. J then avoids probing exactly the
  targets that need the least evidence.
- A monotone alternative to test: the chance that the hour's evidence closes the deficit,
  P(Z >= delta) = F0((q/2 - delta)/sqrt(q)), with psi in CNY per decision. Its argument increases in q for every
  delta >= 0.

**Draft gap fixed in the code and the draft (v13).** In Algorithm 2 the block bought at gate 22 runs in hour 23, after
the last relabelling, so its readings were never used. `probe.py` adds one gate after hour 23.
