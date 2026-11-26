# Proposals under test: when probing starts, and what it targets (2026-11-26)

**Status: recorded, NOT adopted.** Neither the draft (`procedure.tex`, v12) nor the code (`identification_v8/`) uses
these rules. We want the final method to follow from the score or from a proof, not from tuned thresholds. So each rule
below says what is proved, what is only assumed, and which tests decide it. All tests use development cases only; the
sealed set is never used for them.

## 1. The fact both rules rest on (proved)

**Nested pair.** Let tree B be tree A plus one more line M (A's lines and M are compatible). Then

    H(B) - H(A) = beta - gain/2,   gain = (RSS(A) - RSS(B)) / sigma2 >= 0,   so   H(B) - H(A) <= beta.

*Proof.* B can copy A's fit by setting r_M = x_M = 0, which lies inside the box [0, BOUND]. So RSS(B) <= RSS(A), and B
pays beta for its one extra line. QED.

Consequences:

- (a) On any data, the smaller tree beats the bigger one by at most beta. "M is absent" can be decided (gap > m) only
  if beta > m. On the development cases beta = 5.6 to 7.4 and m = ln 100 = 4.6, so the margin left is only 1.0 to 2.8.
- (b) If A is true, gain is noise: about chi-bar-square, whatever the design. Probing does not raise the cap or shift
  the distribution. Evidence against M cannot build up; only beta grows, like ln N. One probing day of 24 hourly rows
  adds about 0.12 to beta.
  A strong probe can still help, but only by replacing the history's unlucky noise with fresh noise. That is a one-off
  draw, and with dev values it decides at most about 94% of the time (P(gain/2 < beta - m)).
- (c) If B is true (A lacks a true line), gain grows with the excitation of M's flow (Theorem 1a). Probing confirms M.
- (d) The draft's forecast (Lemma 4, the q/2 in eq. 15) assumes the leader generates the data. It is wrong exactly
  when the rival contains the leader: there the refit absorbs whatever the probe shows, yet the formula still credits
  q/2.

For contrast, the user's guarantee (Theorem 1a) concerns a rival that lacks a true line. Its cost grows with the data,
so presence can be decided firmly, while absence of an extra line is capped by (a).

## 2. Start rule (proposal S)

On a given day, probe only if all four hold:

| | Condition | Basis | Proved or assumed |
| --- | --- | --- | --- |
| S1 | The leader is adequate: F = [(RSS_L - RSS_sat)/df1] / sigma2_hat is below the 99% quantile of F(df1, df2), where df1 = p_sat - p_L and df2 = the saturated model's residual degrees of freedom | every tree is a special case of the saturated model (all R_ij, X_ij free) | the F distribution holds for a linear Gaussian model with the bounds inactive. **Assumed:** linearisation and PCC errors are small against the noise at dev N (test SR1) |
| S2 | The search is complete: pricing finds no single line that pays | Lemma 3 | proved for single-line additions only; Remark 1 shows a pair can pay when no single line does |
| S3 | At least one target exists (section 3) and some feasible block reaches it | Lemma 5 | proved (reach) |
| S4 | J > 0 for the best block | eq. 15 | a forecast; see 1(d) |

Outcomes:
- If S1 fails, report **low belief** and do not probe, because probing between wrong trees wastes money.
- If S3 finds no target, every grouping of the leader is decided. Report the labels and do not probe; this is the
  "high confidence, no need to probe" case.

**Open question:** m = ln 100 is per grouping. For a whole tree with G groupings, the chance that some decision is wrong
is up to about G/100. A whole-tree level would use m = ln(100 G), but that can exceed beta (see 1a).

## 3. Target rule (proposal T)

Three variants, to be compared:

- **T0 (draft v12).** Every open grouping is a target, together with every open load place, valued by eq. 15.
- **T1 (exclude).** A grouping outside the leader and nested in it (a "one more line" rival) is not a target. It is
  reported as "a further line here is not worth its price (margin x)". This applies 1(b) and 1(d): in the world the
  planner assumes (leader true), such a target has no value that can build up.
  **Risk:** a weak true line missing from the leader is also nested and outside, so T1 would never probe for it,
  although 1(c) says probing could confirm it.
- **T2 (two-world value).** Keep nested rivals as targets, but value them correctly:
  - in the leader-true world, 0 (or at most the one-off draw of 1b);
  - in the rival-true world, kappa(m + gap, q), weighted by the rival's probability 1/(1 + e^gap).
  T2 uses the same objective as eq. 15, with no new threshold, and keeps the route to weak missing true lines.

Groupings of the leader and rivals that cross the leader stay targets in all variants.

## 4. Evidence so far

The script is `check_extra_lines.py`, run 2026-11-26. It covers 24 no-load development cases with at most 20 meters.
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

## 5. Tests that decide

The pass criteria are written here before the tests run.

| Test | Question | How | Decides |
| --- | --- | --- | --- |
| SR1 | Does S1 flag true trees? | F of the true tree on all 288 dev twins | Pass if at most 2% are flagged (nominal 1%). If it fails, model error is visible at this N: S1 needs a model-error term or is dropped |
| SR2 | Does S1 catch real failures? | the leader's F when a true line is removed from the pool; a second unmetered load; a load at 0.5-1% just under the gate; a biased meter | S1 is useful if it flags at least half of the cases whose decided labels are wrong |
| SR3 | Is "no target, so no probing" safe? | v8 on dev twins; in the cases where S says do not probe, count wrong decided groupings | Pass if at most 2% per grouping (nominal about 1%); also report the whole-tree rate against G/100 |
| SR4 | Does S2 add anything? | the wrong-label rate in complete against incomplete cases | keep S2 only if incomplete cases are clearly worse |
| TR1 | The cap holds in code | as in section 4 | **done:** held |
| TR2 | How often are nested rivals open? | as section 4, on all 288 twins (with loads, larger n) and with the real candidate pool | descriptive |
| TR3 | Leader true, nested target: what does probing buy? | a simulated probing day under T0; share decided and money spent | T1/T2 are preferred if T0 spends on these targets with little gain |
| TR4 | Leader lacks a weak true line: what does T1 lose? | v8 on dev twins: count true lines missing from the leader that are nested in it; simulate probing under T0, T1, T2 | T1 is acceptable only if such lines are rare (at most 1 per 50 cases) or T0/T2 recover few of them either; otherwise T2 |

**Prerequisites:**
- v8 identification with labels on the dev twins (not yet run at scale);
- Algorithm 2 in code;
- the case adapter for simulated probing days (load day profile, PCC day power, households).

## 6. Related items, also not adopted

- **Stop rule:** stop probing after 6 hours without progress (the old planner's `no_progress_hours = 6`).
- **Operational equivalence** (5e-4 pu) for near-twin crossing rivals, so that they do not become targets. Section 4
  shows it cannot be used to call a line absent.
- **Low-belief report in identification** (the F of S1), reported even when no probing is planned.
