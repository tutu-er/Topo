# Development Runs, 2026-11-20 (pre-lock, not evidence)

**Status:** first implementation of the v3 design, run on regenerated inputs. The protocol is still a draft and nothing here is Stage M evidence. Like Stage D, these runs exist to find plumbing errors and design weaknesses before the lock. They were run before the lock, so every setting they inform may still change.

**Second round, 2026-11-20/21:** single phase only, typical meter bias 0.05%, real line data, a revised step 1, and all four case feeders (Sections 10–12). Its verdict comes first; the first round's follows unchanged.

## Decision after §19 (2026-11-23): the block selector is locked, with T1's failure disclosed

**The user chose to lock the strong greedy at V = 2,000 CNY with linear DR prices and GLS scoring now, and to report the failed truth-removed test instead of first strengthening the trust gate.** This departs from the decision registered in §18, which required a stronger gate before the lock.
- **Why.** All of T1's harm came from completed banks without the truth, which arose naturally in 2.4% of the inputs where probing is allowed. On natural misses probing changed little (§19). Weighted by that rate, the extra wrong confirmations are about 0.1 point against 11 points of extra correct ones, and each was a near-twin at most 0.27% off in voltage. A check of the leader alone cannot detect a missing truth, and the remaining remedy could only be tested on about 3,200 new meter-stress inputs.
- **What it binds.** PROTOCOL §11 (Decided, 2026-11-23) lists the locked settings and the obligations: the paper reports the failure and the estimate; Stage M keeps measuring truth-removed and natural truth-missing runs, split by closure status. In code: `config.locked_settings()` and `config.LOCKED_ARM`, pinned by a unit test to the §18 settings (22 tests pass).
- **Not locked.** The rest of the protocol (PROTOCOL §11 open rows, Stage 0, a real feeder) is still a draft.

## Verdict after the fresh-seed lock test (2026-11-23, §18)

**Probing's benefit replicated on fresh seeds, but the registered safety test with the truth removed failed, so the lock is not confirmed.**

- **Efficacy replicated.** On 532 new ambiguous runs, the strong greedy (V = 2,000) confirmed the true topology in 43% against 32% passive (61 gained, 2 lost) and beat random blocks 47 to 8. It matched the certified planner and had the largest net value (88 CNY per run). No violations; the budget and headroom held.
- **V = 2,000 is better than V = 100** (38 / 14, p = 0.001; 88 against 55 CNY per run). §17's reading of a flat curve from 100 to 2,000 was underpowered; the registered choice stands. Going higher buys nothing: V = 100,000 was as accurate as V = 2,000 on the §16 inputs (214 against 215 of 550) and cost more.
- **T1 failed.** With the truth removed from the same banks, probing confirmed a wrong topology in 64% of runs against 60% passive (37 / 17, p = 0.009). The trust gate barely acted: probing ran on 55% of truth-removed days against 58% of natural ones.
- **Size of the harm.** The extra wrong confirmations were near-twins of the truth (wrong customer placement, an extra line, a merged short segment), at most 0.14% off in voltage on the short lines and 0.27% under meter stress. They are errors in the topology record more than in voltage.
- **Why the gate cannot simply be tightened.** No logged statistic separates a truth-missing bank from a natural one (ROC areas 0.45–0.64): a near-twin fits the meters as well as the truth. The truth was missing naturally in 26 of 128 meter-stress inputs, so this matters in practice.
- **Also found.** The random comparison arm overdrew batteries in 197 of 532 runs. Fixed and rerun: the strong greedy beats it 50 to 8 (registered 47 to 8), so L2 stands.
- **A prediction-based check failed too.** Asking each candidate to predict the next probed hour before seeing it is correctly scaled (the truth's ratio is 1.00) but does not separate truth-missing banks (ROC areas 0.47–0.58). A check of the leader alone lacks the power: the twin's excess misfit (3–10 units over a day) is smaller than the random spread of the test (11–15).
- **Natural misses (§19).** On the 62 inputs whose truth the bank itself missed, probing barely changed wrong confirmations (45% against 44% of runs, 4 / 2, p = 0.69). The gate's closure flag caught 51 of them: the closure hit a cap, so probing was forbidden. All of T1's harm came from banks whose closure had finished, a case that arose naturally in 2.4% of the inputs where probing is allowed (0.8% on short lines, 7.8% under meter stress). Weighted by that rate, the harm is about 0.1 point of wrong confirmations against 11 points of correct ones.
- **Next (a decision for the user).** The registered rule says the gate must be strengthened. The remaining blind spot is a completed bank without the truth. The candidate remedy is to rebuild the bank's neighbourhood around the leader each day with the probe data. The alternative is to lock with the failure disclosed and its size estimated; that departs from the registered rule. *Decided: the alternative (see the decision above).*

## Verdict after the V sweep (2026-11-23, §17)

**The price of certainty V matters little between 100 and 2,000 CNY. A battery-state defect inflated §16's accuracy by about 3 points; with it fixed, probing still clearly helps.** *(Corrected by §18: on fresh seeds V = 2,000 beat V = 100, 38 to 14, p = 0.001. The flat reading below was underpowered; the registered pick V = 2,000 stands.)*

- **No single best V.** Every V from 100 to 2,000 CNY confirmed the true topology in 37–39% of the 550 ambiguous runs (passive 29%), worth 62–72 CNY per run at 1,000 CNY per correct identification. The registered rule picks V = 2,000, by a margin inside the noise (10 runs, p = 0.16 against V = 100). V = 100 costs less than half as much (11.6 against 24.7 CNY a day). Only V = 50 is clearly too low.
- **Why it is flat.** From V ≈ 500, probing already runs in every hour the gates allow (3.2 a day), so a larger V only buys larger blocks.
- **Defect found.** In §16 the next block was sized against the batteries' charge before the current block ran, so it could draw on energy that was no longer there. With the fix, the strong greedy at V = 200 is 37% correct instead of 40% (47 to 3 against passive instead of 62 to 2).
- **Next.** The fresh-seed confirmation (§18, truth present and removed) is running with V = 2,000 as registered and V = 100 alongside.

## Verdict after the hard-case test (2026-11-23, §16)

**Where one day of passive data is not enough, targeted probing clearly helps, at realistic demand-response prices of about 17–27 CNY a day. The frugal strong greedy is the selector to lock.**

- **Realistic cost.** The draft contract understated the cost about ten times. At DR prices (reductions 4 CNY/kWh, increases 1.2 CNY/kWh), probing costs 29–45 CNY on the days it runs, at most 108 CNY on any day, and about 150–240 CNY per extra correct identification, against about 1,000 CNY for a field visit.
- **Probing works on hard inputs (pre-registered HC1, HC2).** On 550 ambiguous runs of three new hard conditions (short segments of 0.01 and 0.005 Ω, meter stress), the strong greedy confirmed the true topology in 40% of runs against 29% passive (62 gained, 2 lost, p = 2·10⁻¹⁶; 37% and 47 to 3 after the battery-state fix of §17). It beat random blocks on the same hours 55 to 6. The first correct confirmation came at hour 7 instead of hour 10. Each condition passed on its own.
- **Selector.** Strong greedy (V = 200 or 1,000 CNY) and certified planner are indistinguishable; the committee rule is significantly worse (29 to 12 against the strong greedy). By the pre-registered rule (the cheapest arm not significantly worse than the best) the selector is the strong greedy with V = 200 CNY: 16.8 CNY a day, the same accuracy as V = 1,000 at about 70% of its cost.
- **Safe.** No violations in 3,300 simulated days. Probing did not raise wrong confirmations (47–48% of ambiguous runs in every arm, almost all the short segment merged away, at most 0.15% off in voltage; under meter stress at most 0.25% with probing, 0.29% passive).
- **Limits.** A 0.005 Ω segment is still resolved in at most 22% of ambiguous runs, and under meter stress half of the runs stay undecided after one day. Two defects were found and fixed along the way: the strong greedy's twin-only risk sum (§16), and a daily budget that ignored the block already sealed for the current hour (9 of 2,750 runs over the cap by at most 8.3 CNY).
- **Still synthetic.** The four feeders are synthetic, and the truth was in the bank in every ambiguous input. Real AMI data from a surveyed feeder remain the test that matters.

## Verdict after the fresh-seed confirmation (2026-11-23, §14–15)

**The refined pipeline is reliable on these synthetic feeders; probing is safe but adds little once the scoring is right; the strong greedy is the selector to lock.**

- **Confirmed on fresh seeds.** Revised step 1, per-meter offsets and GLS scoring for the shared root-meter error, against the first-round pipeline on 64 new inputs per case (1-day history, 0.05% meter bias):
  - main case: correct in 100% of runs against 48%, with no wrong confirmation against 49%;
  - 0.01 Ω short line: correct in 78% against 27%, wrong confirmations 15% against 69%, none of them more than 0.2% off in voltage.

  Pooled, the refined pipeline wins 134 paired runs and loses 3. The repository identifier's own pick was right in 19 and 12 of 64.
- **GLS was the decisive correction.** Half of the measurement noise is shared by all terminals through the root meter. Modelling it roughly doubled the evidence: ambiguous short-line runs went from 42% to 77% correct during development.
- **Probing did not pass its test.** On top of the refined pipeline it gained 5 runs and lost 1 (p = 0.22). It stays safe (no violations in 1,916 days). *(Superseded by §16: "cheap" was a contract-price artefact, and on hard inputs probing passes clearly.)*
- **Selector.** The strong greedy (one hour at a time, minimise cost + V × the Bhattacharyya bound on misidentification risk), the committee rule and the certified planner were statistically indistinguishable. The pre-registered rule therefore picks the cheapest: the strong greedy, which bought about half as many hours.
- **The residual risk is a truth missing from the bank.** Then a near-twin is confirmed, at most 0.15% off in voltage. With the revised step 1 the truth was in the bank in 128 of 128 fresh inputs.

## Verdict, second round

**With the revised step 1 the method is reliable and practical in simulation. Targeted probing then pays off, but only if the planner may buy without the union-bound guarantee, and it should run only when the bank is trusted.**

- **Step 1 decides reliability, and it can be fixed.** Under typical 0.05% meter bias, the first-round bank missed the true topology in 25 of 64 inputs (all four feeders, 1-day history), and confirmed wrong models were up to 0.32% off in voltage. The revised step 1 contained the truth in 64 of 64 and its passive BIC leader was correct in 56 of 64. It adds per-meter offsets, keeps the identifier's pre-contraction candidates, and closes the bank under one-edit moves within a BIC margin. Offsets are the decisive part under bias; the closure adds the rest.
- **Passive data then do most of the work.** Ambiguous inputs (S1) were confirmed correctly in 89% of runs without probing; confident inputs in 100%.
- **The protocol's planner never buys.** With banks of up to 150 candidates, the union-bound tail cannot be covered by three active rivals: on all 862 triggered S1 gates the planner found no admissible rival. Under protocol v3 as written, targeted probing equals passive.
- **A best-effort planner works.** Targeting the three nearest reachable rivals, with the tail requirement dropped and the best block bought at the ceiling, made all 100 S1 runs correct (passive 89%) and halved the median time to a correct confirmation (8 h to 4 h), at 2.6 CNY a day. Random blocks of equal cost gained almost nothing (91%), so the targeting does the work.
- **The price is faster wrong confirmations when the truth is missing.** With the truth removed from the bank, best-effort probing confirmed a wrong topology in 76% of runs against 44% passive, up to 0.19% off in voltage. With the revised step 1, a natural truth-missing bank occurred in 0 of 64 typical inputs and 3 of 64 short-line inputs (all 3 from computational caps), so this is a worst case, not the expected one.
- **Realistic short lines remain unresolvable within a day.** A 0.005 Ω internal segment stays in the bank, but passive data or probing resolve it in at most 19% of runs. The merged topology that gets confirmed is at most 0.13% off in voltage.
- **The case feeders are synthetic and generous.** Every internal line is 46–95 m and 4–9 times above the identifier's deletion threshold; real underground feeders have many shorter segments. The first round's 0.0045 Ω short-line test sat below that threshold on three feeders, so its result measured the threshold, not probing.
- **Safe and fast.** 2,280 simulated days: no voltage or volt-var violations, voltages within 0.994–1.023 pu, 1.7 s per planning gate on average (at most 10.5 s), at most 18.3 CNY on any day.
- **Targeting matters; the planner's sophistication does not (§13, added 2026-11-22).** In a pre-registered comparison on four cases, a certified planner built on a Bhattacharyya miss bound (`PLANNER_REVISION.md`) beat passive (20 gained, 1 lost) and random blocks of the same cost (17 to 1). It did not beat a query-by-committee heuristic (6 to 3, p = 0.51), and it cost 1.4–1.6 times as much. A simple targeted rule does as well.

## Verdict, first round

**Reliable where it is least needed, unreliable where real feeders differ from the model.**

- **Safe and cheap.** No executed block caused an AC voltage violation or left the volt-var deadband. Blocks were at most 4.9 kW, a planning gate took under 10 s, and probing cost 0.05–3 CNY a day on average. Confident inputs (S2) almost never triggered probing.
- **Small benefit where the bank is ambiguous.** On S1 inputs targeted probing raised correct confirmations from 81% to 91% (broad check first: 97%), but it never confirmed sooner (median 10 h either way). The predeclared decision rules C1–C4 all fail at this development scale.
- **The planner rarely buys, and when it does the window is empty.** With banks of 45–122 candidates, the active-set rule leaves no active rival on 82% of triggered gates. Every executed targeted block ran at its headroom ceiling and delivered only 7–15% (median) of the information the 2-hour guarantee asks for.
- **The main risk is a true topology missing from the bank.** Then a wrong topology is confirmed at some hour in 84–91% of runs, and no arm prevents it: not the broad check, the no-progress rule or the reach test. With balanced physics and ideal meters these wrong topologies predict voltages within 0.1% of the truth, so they are nearly harmless. Under realistic conditions they are not.
- **Realistic conditions push the truth out of the bank.**
  - Unbalanced phases: truth missing in 30 of 30 banks; the confirmed models are off by up to 1.2% in voltage.
  - A 0.1% voltage-meter gain bias: truth missing in 28 of 30; up to 0.5% off.
  - A real 0.0045 Ω internal segment: missed in 96% of runs, but harmlessly (within 0.1%).

  These failures arise in the passive identifier and the balanced model, before any probing. Probing cannot repair them.
- **Proposition 6 holds under shared root-meter noise, but not under meter bias.** A 0.1% gain bias lets extra-line rivals come within ln 19 of the truth 8–14 times more often than the 19/N bound allows. Per-meter offsets (demeaned data) make the identifier immune to such bias, at some cost in accuracy.
- **A cheap fix makes the planner act.** Taking the tail of the union bound at Chernoff order 1 cut the time to a correct confirmation from 10 h to 7 h, for about 1 CNY a day more.

## What was run

| Step | What | Code |
| --- | --- | --- |
| Unit tests | The PROTOCOL §9 tests the code covers | `test_hourly_probing.py` |
| Main batch | 5 arms × S1/S2/S3 inputs × 4 paths, balanced AC, scenario R1 | `experiments.py run main` |
| Realism stresses | Unbalanced three-phase physics, meter gain bias, short internal line, low adoption, records-based floor, gate lead 0 h; arms `passive_bic`, `t_opt_P`, `check_then_t_opt` | `experiments.py run <condition>` |
| Candidate amendment | The union-bound tail at Chernoff order 1 | `experiments.py run tail1` |
| Proposition 6 under meter noise | Share of extra-line rivals within ln 19 of the truth, with white noise and with the physical meter model | `noise_check.py` |
| Identifier remedies | Identifier fed with demeaned data, with and without meter bias (differencing was tried once, by hand) | `identifier_remedies.py` |

In total, 3,464 simulated feeder-days, plus 800 for a reproducibility rerun of the main batch, which matched exactly.

Outputs are in `outputs/hourly_probing_dev_20261120/` (ignored by Git).

## How the inputs were made

The protocol's source banks (`history288_pq_noise_20261115`, `flexibility_grid_20261115`) live in `data/`, which is not on this PC. Every input was therefore regenerated:

1. **Load and physics.** The load law of `physical.backend.generate_day96`, extended to several days, drives the case feeders Soumalas11 and Pengwah18. Voltages come from the balanced AC sweep of `physical.backend`, or from the new three-phase sweep `unbalanced.py` in the unbalanced condition.
2. **Meters.** Voltage meters have 0.05% random error, P/Q meters 0.5%. The root voltage is metered like every terminal, so its error enters every terminal's squared-voltage drop at the same sample. The stress conditions add a fixed gain bias per voltage meter.
3. **Passive history.** 1.5 days (96 training and 48 validation slots) or 0.75 day (48 and 24), followed by the probing day.
4. **Bank.** The repository's passive identifier (`identification.algorithm.fit`, method 111, Gurobi) produces 6–13 candidates. Every one-edit neighbour of the passive BIC leader is added: one extra line, one missing line, one leaf moved to an adjacent support. Banks hold 45–122 candidates, so extra-line, missing-line and wrong-placement rivals are always present.
5. **Strata.** As PROTOCOL §3, with the v3 scorer on the passive history. S3 inputs are the natural ones (truth not in the bank) plus S1/S2 inputs with the truth removed.

## Deviations from the draft protocol

| Item | Protocol v3 | Development runs | Why |
| --- | --- | --- | --- |
| Source banks | Saved frozen banks | Regenerated (above) | `data/` is missing |
| Paths per input | 8 | 4 | Development scale |
| Trigger | A non-equivalent rival with `E_j < ln 19` | The posterior rule (20) fails | With 45–122 candidates, many rivals at `E_j` of 3–4 can block confirmation while none is below ln 19 |
| Same-sign blocks at several nodes | Not allowed | Allowed when the transformer limit holds | The dictionary's group blocks are same-sign |
| Guard (score test) | Open, Stage D | Not implemented | — |
| Refinement (step 6) | Off by default | Off | — |
| Unbalanced condition | Rerun the same paths on an unbalanced day, bank unchanged | History and day both unbalanced, bank regenerated | A real feeder is unbalanced in the history too |
| Unbalance limit | κ_U = κ_V / 6 | Same, plus the AC check on all three phases | — |

## Results

### 1. Unit tests

All 10 tests in `test_hourly_probing.py` pass (11 s):
- the loss-compensated model removes 99.9–100% of the linear model's gap to the AC drops, and its sensitivity matches an AC finite difference within 2%;
- the three-phase sweep reproduces `physical.backend` within 1e-15 pu at equal phase shares;
- with `r_min = 0` the refit matches `QPFitter` within 1e-9 in relative SSE; with `r_min > 0` the floor holds, and floored coordinates leave Σ_k;
- Proposition 6: under white noise the share of seeds with `E_W < ln 19` stays within 19/N; with an aligned error at 2β*, W leads by ln 19 in at least 90% of 1,000 seeds;
- with Σ = 0, q_j equals the closed form; the dictionary plan meets q_j ≥ η_j at κ_need and fails one bisection step below;
- the active set leaves `R_tail ≤ δ/2`, and the reach `Q̄_j` bounds 20 random dictionary plans of the rest of the day;
- a rival equal to the leader plus one branch held at the floor is beyond reach, and nothing is bought;
- check blocks sum to zero within headroom; `frozen_replay` repeats `t_opt_P`'s blocks; `random_matched_P` never exceeds its hourly cost.

### 2. Passive stage (60 main inputs)

| | Soumalas11 (36) | Pengwah18 (24) |
| --- | ---: | ---: |
| Identifier bank contains the truth | 27 | 16 |
| Identifier's validation-MSE pick correct | 26 | 16 |
| v3 BIC leader correct (extended bank) | 31 | 18 |
| Strata S1 / S2 / S3 | 5 / 27 / 4 | 3 / 17 / 4 |

- With 1.5 days of history the BIC leader was confident and correct in every input but one. Ambiguity (S1) came almost only from 0.75-day histories.
- The BIC refit leader was correct in 49 of 60 inputs against 42 for validation MSE. The banks differ (the BIC leader is chosen from the extended bank), so this is indicative only, not the Stage 0 test C0.

### 3. Main batch (balanced AC, scenario R1, 4 paths)

| Stratum | Arm | Correct | Ever confirmed wrong (extra / missing / placement) | Median h to correct | Probed h/day | Cost CNY/day (max) |
| --- | --- | ---: | --- | ---: | ---: | --- |
| S1 (32) | `passive_bic` | 81% | 6 (0/4/2) | 10 | 0 | 0 |
| | `t_opt_P` | 91% | 6 (0/4/2) | 10 | 1.4 | 0.41 (2.7) |
| | `check_then_t_opt` | 97% | 5 (0/4/1) | 7 | 3.2 | 2.32 (5.6) |
| | `random_matched_P` | 81% | 6 (0/4/2) | 10 | 1.4 | 0.37 (2.2) |
| | `frozen_replay` | 94% | 6 (0/5/1) | 9 | 1.4 | 0.41 (2.7) |
| S2 (64) | `passive_bic` | 100% | 0 | 1 | 0 | 0 |
| | `t_opt_P` | 98% | 0 | 1 | 0.1 | 0.05 (3.4) |
| S3 (64) | `passive_bic` | — | 54 (18/8/29) | — | 0 | 0 |
| | `t_opt_P` | — | 58 (22/8/31) | — | 1.5 | 1.11 (5.7) |
| | `check_then_t_opt` | — | 57 (22/8/29) | — | 3.2 | 3.02 (12.4) |
| | `frozen_replay` | — | 55 (21/8/29) | — | 1.5 | 1.11 (5.7) |

Type counts can exceed the number of runs, because one run can confirm wrong leaders of two types. In S3, `t_opt_P` ends 23% of runs as "unresolved: beyond probing reach" and 5% as "no progress"; the passive arm ends 47% undecided.

**Predeclared decisions at development scale** (paired McNemar counts; S1 = 32 paths):

| Decision | Result | Supported? |
| --- | --- | --- |
| C1 probing helps | +3 correct, 0 lost (needs ≥ 3.2); median time unchanged; ever-wrong in S2 ∪ S3 +4 vs passive | No |
| C2 design matters | +3 correct vs `random_matched_P` (needs ≥ 3.2) | No (borderline) |
| C3 broad check pays | Ever-wrong in S3: 4 more with the check, 1 fewer | No: the check adds wrong confirmations |
| C4 frozen scoring fails | Ever-wrong in S1 ∪ S2: 1 vs 1 | No |

### 4. Why targeted probing adds so little

**The planner rarely has an active rival.** Of 316 triggered planning gates in S1:

| Reason | Gates |
| --- | ---: |
| No active rival: the tail of the rest of the bank exceeds δ | 260 (82%) |
| Window empty: best shape run at its ceiling | 46 (15%) |
| Only rivals beyond probing reach | 10 (3%) |
| Admissible plan | 0 |

- The active-set rule requires `R_tail = Σ √19·e^(−E_j/2) ≤ δ/2` over the rivals left out.
- With 45–122 candidates, dozens of wrong placements sit at `E_j` of 4–15. Each rival only drops out of this tail at `E_j` ≳ 17.
- So the guarantee cannot be met even in principle, and the gate buys nothing.

**When it buys, the window is empty.**
- Headroom bound every executed targeted block (46 of 46 in S1, 96 of 97 in S3).
- At the ceiling, the plan reached a median of 7% of the required η_j in S1 (90th percentile 23%) and 15% in S3.
- Executed blocks had a median κ of 1.2 and a median largest node block of 1.5 kW (max 4.9 kW).

**Realised evidence matched the plan's scale.** Per probed hour, the active rival's `E_j` rose by a median of 0.67, against about 0.77 implied by the planned q_j. In 26% of blocks it fell.

**The pre-check's optimism came from its planner.** The pre-check aimed at an expected evidence and found P-only probing confirmed the truth in every run. The guaranteed condition of v3 asks 5–18 times more information, which R1 cannot supply in 2 hours.

### 5. Wrong confirmations: by type, and whether they matter

The table reports, for each run with a wrong confirmation, the largest deviation of that wrong leader's predicted voltages from the true AC voltages over the day.

| Condition | Stratum | Ever wrong (t_opt_P) | Within 0.05% | Within 0.1% | Largest deviation |
| --- | --- | ---: | ---: | ---: | ---: |
| Main (balanced, ideal meters) | S3 | 91% | 30/58 | 58/58 | 0.09% |
| Main | S1 | 19% | 2/6 | 6/6 | 0.07% |
| Low adoption (same inputs) | S3 | 84% | 26/54 | 54/54 | 0.09% |
| Short line | S1 | 96% | 41/46 | 46/46 | 0.05% |
| Meter bias 0.1% | S3 | 94% | 0/60 | 0/60 | 0.48% |
| Unbalanced | S3 | 80% | 0/51 | 0/51 | 1.15% |

- **Truth missing, balanced, ideal meters.** The BIC posterior is relative to the bank, so it confirms the closest remaining candidate. That candidate predicts voltages within 0.1% of the truth (about 0.2 V), so the error is cosmetic for voltage management. It still matters for anything that needs the exact topology (phase or fault location, asset records).
- **Short line.** A real 0.0045 Ω segment (1.5 R_min) is missed in 96% of runs. At the confirmation gate the truth was judged operationally equivalent to the leader (rule (19)) in every inspected case, so it never entered the posterior sum. This is the trade-off of Proposition 6, measured: the floor hides short lines, harmlessly.
- **Unbalance and meter bias.** The confirmed models are systematically wrong by 0.5–1.2%, that is 1–3 V, and the method cannot tell.

### 6. AC bias ratio b/J

As specified in PROTOCOL §8, b/J uses the leader's prediction error. When the leader is wrong, b/J is −2 by construction (the rival is the better model), so it is reported only for blocks whose leader at the sealing gate was the truth (main batch, `t_opt_P`):

| Rival type | Blocks | Whole response flagged (\|b/J\| > 0.5) | Probe part flagged | Median b/J (whole, probe) |
| --- | ---: | ---: | ---: | --- |
| Missing line, loss term | 26 | 31% | 35% | +0.24, −0.02 |
| Missing line, linear model | 26 | 27% | 42% | +0.06, +0.02 |
| Wrong placement, loss term | 67 | 39% | 30% | +0.08, +0.07 |
| Wrong placement, linear model | 67 | 34% | 30% | +0.06, +0.08 |
| Extra line (meter-bias condition) | 3 | 100% | 100% | −1.23, −1.16 |

- Amendment trigger (i) fires: far more than 10% of blocks are flagged, both with and without the loss term.
- **The flags are not physics bias.** The medians are near zero, the loss term barely changes the shares, and the loss-compensated model is exact on balanced AC (unit test). The scatter comes from the leader's parameter-estimation error over a single hour of 4 samples. As specified, b/J cannot separate parameter error from model bias. It should be computed at the true parameters (pure model bias), or against the predictive spread of (8).
- **The one systematic case is Proposition 6(b).** Under meter bias, with the truth leading, b/J against extra-line rivals was −1.2: the bias pushed the evidence towards the extra line. Only 3 blocks were seen.

### 7. Realism stresses

| Condition | Inputs (S1/S2/S3) | Main effect |
| --- | --- | --- |
| Unbalanced (single-phase households, Z0/Z1 = 3, voltage on one phase) | 0/0/30 | The identifier never finds the truth. Metered drops differ from the balanced model by 4–15 times the meter noise; even the true topology leaves a systematic residual of 0.6–1.7 × noise. Ever-wrong 80% (`t_opt_P`), 86% with the check. |
| Voltage-meter gain bias, σ = 0.1% per meter | 1/1/28 | Constant offsets of about 1.4 × noise per terminal break the identifier (2 of 30 correct picks). Ever-wrong in S3: 94–97%. One refit hit an SVD failure on biased data; the solver fallback caught it. |
| Short internal line, 1.5 R_min | 12/0/18 | The identifier never keeps the short line; the leader's neighbours supply it in 12 inputs. Missed in 92–96% of runs, within 0.1% of the true voltages. |
| Low adoption (10% batteries, 5% EVs) | as main | S1 correct 81% → 84% with probing; one gate in three finds only rivals beyond reach; plans reach a median 5% of η_j. |
| Records-based floor, R_min = 0.01 Ω | 7/11/12 | Probing helps most here: S1 correct 89% → 96% (`t_opt_P`), 100% (check), at 0.40 and 1.89 CNY a day. S3 unchanged. |
| Gate lead 0 h | as main | No measurable effect of the 1 h lead: S1 correct 88% with 0 h against 91% with 1 h; median 9 h against 10 h. |
| Tail at Chernoff order 1 (candidate amendment) | as main | The planner acts: idle on 55% of triggered S1 gates instead of 82%; probes 2.7 h a day instead of 1.4 h. S1 median time to a correct confirmation falls from 10 h to 7 h (passive: 10 h), at 1.41 CNY a day; correct confirmations 88%. In S3, ever-wrong is 89% against 84% passive, so C1 still fails on wrong confirmations. |

**Remedies tried at the identifier stage only** (16 inputs, identifier picks the truth):

| Data given to the identifier | Ideal meters | 0.1% gain bias |
| --- | ---: | ---: |
| Levels (as now) | 11/16 | 2/30 (stress build) |
| First differences in time | — | 0/16 |
| Per-terminal demeaned y, P, Q (equivalent to per-meter offsets) | 7/16 | 7/16 |

Demeaning makes identification immune to constant meter bias, at the cost of the information in the mean voltage drop. Differencing also removes the slow daily load swing that carries most of the information, so it fails.

### 8. Proposition 6 under realistic meter noise

Setup: true topology T against every one-extra-line neighbour W, 1.5 days of passive data, 300 seeds, v3 refit with the floor. The table gives the share of seeds with `E_W < ln 19` (`noise_check.py`).

| Noise | Soumalas11 (16 rivals, 19/N = 1.2%) | Pengwah18 (36 rivals, 19/N = 0.73%) |
| --- | --- | --- |
| White, same variance | 0.04% per rival; any rival 0.7% | 0.06% per rival; any rival 2% |
| Physical meters (root error shared by all terminals) | 0% | 0% |
| Physical, no floor | 0.02% | 0.03% |
| Physical + 0.1% gain bias | **9.3% per rival** (worst 15%); any rival 76%; W ahead by ln 19 in 3.3% | **10.2% per rival** (worst 19%); any rival 89%; W ahead by ln 19 in 4.4% |
| Physical + bias, no floor | 9.7%; any rival 77% | 10.7%; any rival 89% |

- **The open item of MANUSCRIPT §IX is answered: the shared root-meter error does not weaken the 19/N bound.** No extra-line rival came within ln 19 in 300 seeds on either feeder.
- **A constant meter bias breaks the bound by a factor of 8–14**, as Proposition 6(b) predicts for a systematic error. The resistance floor does not help against it.

### 9. Practicality

| Requirement | Finding |
| --- | --- |
| Computation | 0.65–1.1 s per gate on average, 9.8 s at most (Pengwah, about 100 candidates, one core with others busy); far inside the 60 s cap. NumPy/SciPy only; Gurobi only for the passive identifier |
| Grid safety | No AC band or volt-var deadband violations in any arm or condition (3,464 simulated days); phase voltages stayed within 0.972–1.028 pu even when unbalanced |
| Cost | 0.05–3 CNY a day on average, at most 14.6 CNY; the 5 CNY hourly or 50 CNY daily cap bound in 9 of about 1,400 planned blocks, headroom in all others |
| Flexibility | R1 (25% batteries, 20% EVs, 25% enrolled) supplies about 10% of what the guarantee needs; low adoption about 5%. Both are optimistic for today's LV feeders |
| Data | The hourly loop with a 1 h gate lead needs every terminal's 15-min V, P and Q within the hour. Many AMI systems deliver interval data once a day, so this needs edge or near-real-time metering |
| Meters | Voltage-meter offsets of 0.1% must be removed (per-meter offsets) or calibrated out before identification |
| Phases | The balanced single-phase-equivalent model fails on unbalanced feeders. A per-phase model with known phase connections is a precondition |

## Second round (2026-11-20): single phase, typical meter bias, real line data, revised step 1

The first round's realism stresses were revisited after three decisions and one question:
- the study is single-phase: the unbalanced condition is out of scope (PROTOCOL §11, decided 2026-11-20);
- a voltage-meter gain bias of σ = 0.05% is the typical case, and 0.1% is a stress test;
- whether the short-line test reflects real line parameters, checked on all four case feeders;
- whether step 1, the passive identification, can be changed so that the bank contains the true topology.

### 10. Are the case feeders and the short-line test realistic?

All four case feeders (`physical/cases`: Soumalas11, Pengwah18, Flynn16, Paper15) are synthetic. Each imitates the style of a published test feeder; none is a surveyed network. Their line data come from one uniform synthetic library, and they were first built at impedance scale 4.5–5.2, then reset to scale 1.

| Item | Case feeders | Real LV feeders | Consequence |
| --- | --- | --- | --- |
| Main conductor | 0.42 Ω/km, X/R 0.81 everywhere | Aerial bundled 95 mm² Al: 0.32 Ω/km, X ≈ 0.08–0.09 Ω/km (X/R ≈ 0.25); underground 185 mm² Al: about 0.16 Ω/km, X ≈ 0.07 Ω/km; bare open wire: X ≈ 0.3 Ω/km | R plausible for a small conductor; X realistic only for bare overhead lines |
| Service line | 0.55 Ω/km, X/R 0.69 | 25 mm² Al: 1.2 Ω/km at 20 °C (1.49 at 80 °C), X ≈ 0.10 Ω/km (X/R ≈ 0.07) | Service R is 2–3 times too low and X about 4 times too high. P-only probing does not use X, so this matters little for topology |
| Segment between branching nodes | 46–95 m (0.019–0.040 Ω) | One pole span (about 30–50 m) overhead; a joint or pit every 15–40 m underground | Real feeders have many shorter internal segments |
| Internal line strength `s_C` against the identifier's 0.08 contraction threshold | 0.30–0.73 on every true internal line: 4–9 times above the threshold | A 185 mm² cable segment of 20–35 m sits at the threshold | No true line of the case feeders comes near the threshold |
| The 0.08 threshold in ohms (true R/X, backbone X/R) | 0.0038 Ω (Soumalas), 0.0050 (Pengwah), 0.0059 (Flynn), 0.0057 (Paper15) | 9–14 m of the case conductor, 12–18 m of 95 mm² ABC, 23–36 m of 185 mm² cable | — |
| Terminal | 7–8 kW nominal peak, about 6 households per metered terminal | — | — |
| Peak voltage drop | 1.4–2.6% | A few per cent on a healthy short feeder | Plausible |
| Load | One smooth daily curve with phase shifts plus AR(1) noise; Q follows P | Spikier, less correlated household loads | Likely harder to identify than real data, not easier |

Sources: [Prysmian LOBAC datasheet](https://australia.prysmian.com/sites/australia.prysmian.com/files/media/documents/PGAU_LOBAC_Cables_2018.pdf), [Eland ABC datasheet](https://www.elandcables.com/media/a4gbf5wc/aerial-bundle-cable-abc.pdf), [Nexans U-1000 AR2V 4×95](https://www.nexans.fr/en/products/Building/Tertiary/Rigid-cables/U-1000-AR2-536932452/product~10043996~.html). The 185 mm² figures are typical catalogue values. The feeder-level `s_C` values are computed with the true R/X matrices (the identifier calibrates on estimated ones).

**The first round's short-line test was artificial.** It shortened one internal segment to 1.5·R_min = 0.0045 Ω. That is below the identifier's 0.08 threshold on Pengwah, Flynn and Paper15 and just above it on Soumalas, so the identifier was built to delete that line. "Missed in 96% of runs" measured the threshold, not probing, and no line of the case feeders is that short. Short internal segments are nevertheless realistic on urban underground feeders. The second round therefore tests two realistic values on all four feeders: 0.01 Ω (about 30 m of 95 mm² ABC or 60 m of 185 mm² cable) and 0.005 Ω (about 30 m of 185 mm² cable).

### 11. Revised step 1: does the bank contain the truth?

`coverage.py`: 240 inputs, 60 per feeder (Soumalas11, Pengwah18, Flynn16, Paper15). 1-day histories on seeds 1–8 and 3-day histories on seeds 1–4, each under ideal meters, 0.05% bias, 0.1% bias, and 0.05% bias with one internal segment shortened to 0.01 Ω or 0.005 Ω. Truth is used only for evaluation.

The revised step 1 (`bank.identifier_families`, `bank.closure`, `model.demean_rows`) has three changes, all using only metered data:
1. **Per-meter offsets.** Every refit gets a free constant per voltage meter, fitted by per-terminal demeaning (exactly equivalent, unit-tested). The identifier also runs once on demeaned data.
2. **Keep what the 0.08 contraction deletes.** The identifier's path before output contraction joins the bank, from both the levels run and the demeaned run.
3. **BIC neighbourhood closure.** Every one-edit neighbour (add, drop or move one line) of each candidate within 6 BIC units of the leader is added and refitted (loss term, offsets, resistance floor), repeated until no candidate within the margin is left unexpanded (caps: 4 rounds, 1,500 candidates). The bank is then pruned to a BIC gap of 30.

Bank labels: A = identifier output as used so far; A+ = A plus the leader's neighbours (the first round's bank); C = A + pre-contraction paths + demeaned run; D = C closed, with offsets (the revised step 1); D0 = the same closure without offsets. "Leader" is the BIC leader of D or D0.

| History | Meter bias | Short line | n | Truth in A | A+ | C | **D** | D0 | Leader correct, D | Leader correct, D0 | Truth's BIC gap in D, median (max) | Adequacy ratio of leader, D / D0 median (D0 max) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 day | 0 | — | 32 | 29 | 32 | 30 | **32** | 32 | 30 | 31 | 0 (3.0) | 0.99 / 0.99 (1.13) |
| 1 day | 0.05% | — | 32 | 10 | 18 | 25 | **32** | 32 | 30 | 22 | 0 (3.0) | 0.99 / 1.03 (1.19) |
| 1 day | 0.1% | — | 32 | 1 | 6 | 24 | **31** | 26 | 29 | 7 | 0 (3.0) | 0.99 / 1.14 (1.42) |
| 1 day | 0.05% | 0.01 Ω | 32 | 4 | 10 | 16 | **30** | 28 | 15 | 11 | 0.4 (7.1) | 0.99 / 1.03 (1.19) |
| 1 day | 0.05% | 0.005 Ω | 32 | 1 | 8 | 4 | **31** | 26 | 1 | 1 | 5.2 (8.0) | 0.99 / 1.03 (1.19) |
| 3 days | 0 | — | 16 | 16 | 16 | 16 | **16** | 16 | 16 | 16 | 0 (0) | 1.00 / 1.00 (1.06) |
| 3 days | 0.05% | — | 16 | 10 | 15 | 16 | **16** | 16 | 16 | 9 | 0 (0) | 1.00 / 1.03 (1.12) |
| 3 days | 0.1% | — | 16 | 0 | 2 | 16 | **16** | 16 | 16 | 1 | 0 (0) | 1.00 / 1.18 (1.36) |
| 3 days | 0.05% | 0.01 Ω | 16 | 6 | 11 | 13 | **16** | 16 | 13 | 5 | 0 (5.2) | 1.00 / 1.03 (1.13) |
| 3 days | 0.05% | 0.005 Ω | 16 | 2 | 11 | 3 | **16** | 15 | 3 | 1 | 4.7 (8.6) | 1.00 / 1.03 (1.13) |
| **All** | | | **240** | **79** | **129** | **163** | **236** | **223** | **169** | **104** | | |

Per feeder, D contained the truth in 60 of 60 inputs on Soumalas11, Pengwah18 and Paper15, and 56 of 60 on Flynn16.

Findings:
- **The revised bank contained the truth in 236 of 240 inputs** against 79 of 240 for the identifier output alone and 129 for the first round's bank. All four misses are Flynn16 inputs in which a closure cap bound (round cap in 2, size cap in 2). The caps bound in only 7 of 240 inputs; raising them costs seconds.
- **With a 1-day history and ideal meters the current step 1 is nearly fine** (truth in A in 29 of 32; in A+ in 32 of 32). The first round's ambiguous and truth-missing inputs all came from 12-hour histories or meter bias.
- **Offsets are decisive under bias.** At 0.05% bias (1 day), the leader of D was correct in 30 of 32 against 22 of 32 without offsets; at 0.1%, 29 against 7. Without offsets the closure also loses the truth in some biased inputs (D0 26 of 32 at 0.1%).
- **A longer history helps.** With 3 days, D's leader was correct in every input without a short line, at all bias levels.
- **Short lines stay in the bank but are not resolved passively.** At 0.005 Ω the truth is about 5 BIC units behind a merged topology (at most 8.6), whose voltages differ by at most about 0.1%; at 0.01 Ω the leader is right in about half the inputs (15 of 32 at 1 day, 13 of 16 at 3 days).
- **Adequacy check.** With offsets the leader's ratio has a median of 0.99–1.00 in every condition. Without offsets it rises with bias (median 1.14–1.18 and up to 1.42 at 0.1%), so a ratio test against the meter class flags stress-level bias. At typical bias it stays below about 1.2, too close to 1 to detect reliably.
- **Cost.** Two identifier runs took a median of 294 s (at most 914 s). The closure with offsets took a median of 6.7 s (at most 76 s), and closed banks held 55–2,332 candidates. The probing runs cap the bank at 150 by BIC order.
- **No bank can be guaranteed to contain the truth.** The closure guarantees only that every candidate within the margin has had all of its one-edit neighbours tried, and that no cap bound. The truth is found whenever a chain of single edits through plausible candidates leads to it. A bound cap or a high adequacy ratio is the signal that it may not have been.

**A defect found and fixed during the runs (2026-11-21).** The neighbour generator grouped children of a node in every combination only up to 5 children, and in pairs beyond. Merging away a segment with 3–4 terminals can leave its parent with 6 or more children, so the true topology was then unreachable in one edit: 1 of 7 internal segments on Pengwah18 and 2 of 6 on Flynn16 (none on Soumalas11 and Paper15). The limit is now 8 children (sizes 2–3 up to 12), and a unit test checks every merged segment on all four feeders. Coverage records written before the fix carry no `neighbour_max_children` field; the Pengwah18 ones were rerun with the fix, and the Soumalas11 ones (no unreachable segment) were kept. The short-line probing inputs were rebuilt with the fix; the main-case inputs already contained the truth in 64 of 64 and were kept.

### 12. Probing on the revised bank: single phase, typical meter bias, all four feeders

**Setup.** 16 seeds on each of the four feeders, a 1-day passive history (96 training and 48 validation slots), scenario R1, single-phase AC evaluation, every voltage meter with a fixed gain bias of σ = 0.05%. Inputs per stratum (PROTOCOL §3): up to 8 per feeder, with S3 made from S1/S2 inputs by removing the truth.

| Condition | Step 1 | Truth in bank | Passive BIC leader correct | Strata S1 / S2 / S3 (natural) | Runs |
| --- | --- | ---: | ---: | --- | --- |
| `bias_typical` | First-round bank (identifier + leader's neighbours), no offsets | 39 / 64 | 31 / 64 | 10 / 29 / 25 | 40 inputs × 3 arms × 2 paths |
| `v2_bias_typical` (new main case) | Revised step 1, offsets in every refit | **64 / 64** | **56 / 64** | 28 / 36 / 0 | 86 inputs × 3 arms × 4 paths |
| `v2_shortline` (one internal segment at 0.005 Ω) | Revised step 1 | 61 / 64 | 2 / 64 | 61 / 0 / 3 | 32 inputs × 3 arms × 2 paths |

Two development variants reuse those inputs with a **best-effort planner** (`tail_guarantee=False`): the active set is the three reachable non-equivalent rivals with the smallest BIC gap, the union-bound tail is not required, and when the 2-hour requirement cannot be met the best block at the ceiling is still bought. `random_matched_P` buys random blocks at the same hours and cost.

**Main case (`v2_bias_typical`, 4 paths).**

| Stratum | Arm | Correct | Ever wrong | Median h to correct | Probed h/day | CNY/day (max) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S1 (100 runs) | passive | 89% | 8% | 8 | 0 | 0 |
| | targeted, v3 planner (`t_opt_P`) | 89% | 8% | 8 | 0 | 0 |
| | broad check first | 88% | 6% | 7 | 2.0 | 1.77 (4.1) |
| | **targeted, best effort** | **100%** (11 more, none fewer) | 5% | **4** | 3.8 | 2.56 (13.0) |
| | random, matched cost | 91% | 8% | 7 | 3.8 | 1.47 (4.3) |
| S2 (116 runs) | all arms | 99–100% | 0% | 1 | ≤ 0.2 | ≤ 0.14 |
| S3 (128 runs, truth removed) | passive | — | 44% | — | 0 | 0 |
| | targeted, v3 planner | — | 44% | — | 0.2 | 0.15 |
| | broad check first | — | 50% | — | 2.2 | 2.02 (8.0) |
| | **targeted, best effort** | — | **76%** | — | 6.3 | 5.50 (18.3) |
| | random, matched cost | — | 53% | — | 6.3 | 2.94 (8.2) |

- **The v3 planner never buys.** On all 862 triggered S1 gates it found no admissible rival: 89% "no active rival", 11% "beyond reach". At 76% of unconfirmed gates some non-equivalent rival is within ln 19, so with banks of up to 150 candidates the union-bound tail can never be covered by three active rivals. Under the protocol as written, `t_opt_P` equals passive.
- **Targeted probing works once it is allowed to buy.** The best-effort planner made every S1 run correct and halved the median time to a correct confirmation (8 h to 4 h), for 2.6 CNY a day. The paired gain is 11 to 0 (exact McNemar p ≈ 0.001). Random blocks of the same cost gained 2 to 0, so the targeting, not the extra excitation, does the work. Every best-effort block ran at its ceiling ("window empty"), yet the evidence sufficed.
- **When the truth is missing, probing makes wrong confirmations more likely.** With the truth removed, best effort confirmed a wrong topology in 76% of runs against 44% passive (random: 53%). Probing sharpens whatever leads, and when the truth is absent the leader is a near-equivalent. These wrong models were at most 0.19% off the true voltages, and 28% of runs exceeded 0.1%. C1 therefore fails on its wrong-confirmation condition, even though the correct-confirmation part passes clearly.
- **The revised step 1 is what makes probing worth doing.** On the same 64 inputs, the first-round bank missed the truth in 25 and its confirmed wrong models were up to 0.32% off (extra-line and wrong-placement rivals, as Proposition 6(b) predicts for meter bias). On `bias_typical` S1, passive was correct in only 38%, and probing did not change that.

**Short line, 0.005 Ω (`v2_shortline`, 2 paths).**

| Stratum | Arm | Correct | Ever wrong | Probed h/day | CNY/day (max) |
| --- | --- | ---: | ---: | ---: | --- |
| S1 (32 runs) | passive / v3 planner | 6% | 66% | 0 | 0 |
| | broad check first | 6% | 62% | 1.2 | 1.12 (2.6) |
| | targeted, best effort | 19% | 62% | 3.8 | 3.53 (12.5) |
| | random, matched cost | 6% | 75% | 3.8 | 2.01 (8.4) |
| S3 (32 runs) | passive / best effort / random | — | 78% / 75% / 81% | 0 / 4.0 / 4.0 | 0 / 3.56 / 1.74 |

A 0.005 Ω internal segment (about 30 m of 185 mm² cable) stays in the revised bank but is resolved in at most 19% of runs within the day, even with best-effort probing. The confirmed merged topology is a missing-line error at most 0.13% off the true voltages. This matches Proposition 6: near the resistance floor, R1 headroom cannot buy enough evidence in one day.

**Safety and practicality (all second-round runs: 2,280 simulated days).** No AC voltage-band or volt-var deadband violation; voltages stayed within 0.994–1.023 pu. A planning gate took 1.7 s on average (99th percentile 5.3 s, at most 10.5 s) with banks of up to 150 candidates. The most expensive day cost 18.3 CNY (best effort, truth removed); the hourly and daily caps held. Two refits fell back from BVLS to the trust-region solver.

**Reproduce.**

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
C:\Users\Trace\miniconda3\Scripts\conda.exe run -n Topo310 --cwd C:\0-workspace\Topo\repo python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.coverage --workers 15 --seeds 8
C:\Users\Trace\miniconda3\Scripts\conda.exe run -n Topo310 --cwd C:\0-workspace\Topo\repo python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments build v2_bias_typical
C:\Users\Trace\miniconda3\Scripts\conda.exe run -n Topo310 --cwd C:\0-workspace\Topo\repo python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments run v2_bias_typical --arms passive_bic,t_opt_P,check_then_t_opt --paths 4
C:\Users\Trace\miniconda3\Scripts\conda.exe run -n Topo310 --cwd C:\0-workspace\Topo\repo python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.report followup
```

The other conditions (`bias_typical`, `v2_shortline`, `v2_besteffort`, `v2_shortline_besteffort`) run the same way with the arms and paths of the tables above. `v2_main` (ideal meters) and `v2_bias_stress` (0.1%) were built only in part and not run with probing; the coverage batch answers their step-1 question.

### 13. Head-to-head: the certified planner against simpler rules

**Pre-registration (written 2026-11-22, before any run of this section).**

*Question.* Does the certified planner of `PLANNER_REVISION.md` (`cert_P`) choose better blocks than simple rules, or would a heuristic do as well?

*Arms* (`config.CERT_ARMS`), all with the revised step 1, per-meter offsets, 0.05% meter bias and the trust gate. Every arm uses the same trigger: probe only while not confirmed.
- `passive_bic`: no probing.
- `cert_P`: the certified planner, with V = 50 CNY and the 20 nearest rivals in the bound.
- `qbc_matched_P`: the query-by-committee rule (largest posterior-weighted disagreement over the plausible candidates, single-node blocks), at the hours and hourly budget of `cert_P`. This isolates the choice of block.
- `random_matched_P`: a random single-node block, at the same hours and budget.
- `qbc_P` and `greedy_P`: free-running heuristics at the hour's ceiling. `greedy_P` maximises the predicted difference to the nearest rival.

*Cases*, chosen by where the planner's mechanism should matter, before seeing results:

| Case | Inputs | Reason | Prediction |
| --- | --- | --- | --- |
| `cmp_main` | Main-case inputs (`v2_bias_typical`), S1 and S3 | Weak excitation: Section 4 of the note predicts that the two rules choose alike | Parity on S1 (within 5 points); S3 wrong confirmations rise for every probing arm |
| `cmp_short01` | New build, one internal segment at 0.01 Ω, S1 | The separating direction is a group pattern; a group block is several times cheaper per unit of flow than a single-node block | `cert_P` > `qbc_matched_P` in S1 correct |
| `cmp_lowflex` | Main-case inputs, 10% batteries, 5% EVs, S1 | Scarce flexibility makes cost efficiency matter | `cert_P` ≥ heuristics |
| `cmp_highflex` | Main-case inputs, 50% batteries, 40% EVs, 50% enrolled, S1 | The bound becomes informative and the worst-rival focus matters | `cert_P` reaches correct sooner or cheaper |

*Metrics.* Primary: S1 correct confirmation by the end of the day, paired by input and path against `qbc_matched_P` (exact McNemar). Secondary: median hour of the first correct confirmation, cost per day, S3 ever-wrong.

*Decision.* The planner "beats the heuristic" only if, pooled over the four cases, `cert_P` gains more S1 runs than it loses against `qbc_matched_P` with p < 0.05, or matches it at no more than 70% of the free-running heuristics' cost. Otherwise the report says that a heuristic does as well.

*Known limitation of the matching.* Single-node blocks are capped by one home's headroom, so a matched heuristic may spend less than its budget. Actual costs are reported.

**Results** (2026-11-22; 1,668 simulated days, 2 paths per input).

S1, correct confirmation by the end of the day (median hour of the first correct confirmation; mean cost in CNY a day):

| Case (S1 runs) | Passive | Random, matched | Committee, matched | Committee, free | Greedy, free | **Certified** |
| --- | --- | --- | --- | --- | --- | --- |
| `cmp_main` (50) | 86% (8 h) | 88% (8 h; 1.59) | 96% (4 h; 1.95) | 98% (4 h; 2.32) | 96% (5 h; 2.33) | **98% (5 h; 3.24)** |
| `cmp_short01` (64) | 42% (9 h) | 42% (9 h; 2.03) | 48% (7 h; 2.46) | 44% (6 h; 2.70) | 48% (7 h; 2.75) | **48% (6 h; 4.25)** |
| `cmp_lowflex` (50) | 86% (8 h) | 88% (8 h; 0.72) | 90% (6 h; 0.68) | 90% (6 h; 0.78) | 90% (6 h; 0.76) | **92% (6 h; 1.14)** |
| `cmp_highflex` (50) | 86% (8 h) | 88% (5 h; 2.69) | 96% (3 h; 3.83) | 98% (3 h; 4.41) | 98% (3 h; 4.39) | **98% (3 h; 4.34)** |

Paired over all four cases (214 S1 runs), certified planner gains / losses against each arm:

| Against | Gains / losses | Exact McNemar p |
| --- | --- | --- |
| Passive | 20 / 1 | < 0.001 |
| Random, matched | 17 / 1 | 0.0001 |
| Committee, matched | 6 / 3 | 0.51 |
| Committee, free | 5 / 1 | 0.22 |
| Greedy, free | 4 / 2 | 0.69 |

S3 on `cmp_main` (64 runs, truth removed), wrong topology confirmed at some hour: passive 42%, random 55%, committee matched 67%, committee 64%, greedy 69%, certified 75% (21 more than passive, none fewer).

**Predictions against outcomes.**
- `cmp_main`, parity: **confirmed**.
- `cmp_short01`, certified better than the matched committee: **refuted** (48% against 48%, 3 gained and 3 lost).
- `cmp_lowflex`, certified at least as good: met in the letter (92% against 90%), but the difference is not significant.
- `cmp_highflex`, certified sooner or cheaper: **refuted** (3 h for every targeted rule; 4.34 against 4.41 CNY).

**Decision (as pre-registered): the certified planner does not beat a simple heuristic.** It gains no significant accuracy over the committee or greedy rules and costs 1.4–1.6 times as much in three of four cases.

**Why.**
- The certificate was informative (`B < 1`) in 0–3% of blocks under R1-type flexibility and in 19% with ample flexibility. In that regime Section 4 of `PLANNER_REVISION.md` predicts that the planner reduces to a weighted-information design, which the committee rule approximates.
- The planner's group blocks carried 1.5–1.8 times the power of the heuristics' single-node blocks, yet added no confirmations within the day.
- What decides the outcome is that a targeted block is bought at all.

**What does hold up.**
- **Targeting beats random excitation at the same budget**: 17 to 1.
- **Probing beats passive**: 20 to 1. Every targeted rule cuts the median time to a correct confirmation from 8–9 h to 3–7 h.
- **Every targeted rule speeds wrong confirmations when the truth is missing** (42% → 64–75%), the certified planner most. This is why the trust gate and the revised step 1 matter more than the planner.

**Safety.** No AC voltage-band or volt-var deadband violation in the 1,668 days; voltages within 0.993–1.023 pu; 1.9 s per gate on average, at most 11.6 s.

### 14. Development round 2: GLS scoring and a stronger greedy (seeds 1–16)

Two refinements were developed on the inputs of §12–13 (development seeds only):
- **GLS scoring** for the root-meter error that every terminal's measurement shares (`PLANNER_REVISION.md` §11). In the true model's residuals 50–52% of the variance is shared, and the scorer had treated it as independent.
- **A stronger greedy** (`bgreedy_P`, §10 of the note): one hour at a time, buy the block that minimises cost + V × the Bhattacharyya bound on the probability of naming the wrong operational class.

**Results** (development seeds 1–16, the S1 and S3 inputs of §13, 2 paths). `dev_short01` is complete. `dev_main` was stopped after 336 of 798 runs to free the machine for the confirmation, so its numbers are partial.

*GLS scoring, same inputs and paths as §13:*

| Case | Old scoring (§13): passive correct, median h | GLS: passive correct, median h |
| --- | --- | --- |
| Main, S1 | 86%, 8 h (50 runs) | 95%, 1 h (20 runs, partial) |
| Short line 0.01 Ω, S1 | 42%, 9 h (64 runs) | **77%, 3 h** (64 runs) |
| Main, S3 (truth removed), ever wrong | 42%, wrong models up to 0.20% off | 88%, wrong models up to 0.11% off |

Passive re-scoring of the S1 banks alone (no runs) gave the same picture: on the main case the leader was correct in 28 of 28 inputs against 20, and immediately confirmable in 22 against 0. On the short-line case the leader was correct in 30 of 52 against 16.

*Selectors under GLS* (S1, % correct by the end of the day):

| Case | Passive | Strong greedy | Certified | Random, matched | Committee, matched | Committee | Greedy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Short line (64 runs) | 77% | 78% | 81% | 81% | 81% | 83% | 81% |
| Main (20 runs, partial) | 95% | 95% | 100% | 95% | 100% | 100% | 100% |

The strong greedy did not beat the committee rule (0 gained, 3 lost on the short line). No selector differs significantly from another. With GLS the planners rarely need to buy (0.2–1.8 probed hours a day), because most inputs confirm within 1–3 hours.

*Stricter confirmation (offline, from the logged posteriors; exact for passive).* On the short line, going from (α = 0.05, 2 h) to (α = 0.01, 3 h) cut wrong confirmations from 17 to 11 of 64 runs. Correct confirmations fell from 49 to 43, and the median time grew from 3 h to 9 h. The current rule is kept.

*Adequacy at confirmation.* Under GLS, wrong confirmations on truth-removed inputs had the same adequacy ratio as correct ones (median 0.947 against 0.952). These wrong leaders are near-twins of the truth, at most 0.11% off in voltage, and no absolute fit test can catch them.

*Development conclusion.* GLS is the decisive refinement: it roughly doubles the evidence and does most of the work that probing was meant to do. The strong greedy has a sound theory (submodularity) but no measurable edge. The confirmation keeps every selector and makes no directional prediction among them.

### 15. Confirmation on fresh seeds 17–32 (pre-registered 2026-11-22, before building)

*Inputs.* 16 new seeds on each of the four feeders, 1-day history, 0.05% meter bias, single phase. Two cases: the main case and one internal segment at 0.01 Ω. None of these seeds was used in any development step.

*Pipelines.*
- **Baseline** (`conf_base_*`): the first-round bank (identifier output plus the leader's one-edit neighbours), ordinary least squares, no offsets; passive only. The repository identifier's own validation-MSE pick is reported alongside (no confirmation step).
- **Refined** (`conf_*`): the revised step 1 with GLS inside the closure, per-meter offsets, GLS scoring and the trust gate.

*Arms on the refined pipeline.* `passive_bic`, `cert_P`, `random_matched_P` (matched to `cert_P`), `qbc_P`, `bgreedy_P`; 2 paths per input. All inputs are run, not only ambiguous ones, so the rates estimate the population of these synthetic feeders. Truth-removed copies (S3) are run for the refined passive and probing arms as a robustness check.

*Hypotheses.*
- **H1, pipeline:** refined passive confirms the true topology by the end of the day in more runs than baseline passive, in both cases (paired by input and path, exact McNemar, p < 0.05).
- **H2, probing:** on the refined pipeline, `qbc_P` confirms correctly in more runs than refined passive (pooled over both cases; p < 0.05).
- **H3, selector:** no directional prediction between `bgreedy_P`, `cert_P` and `qbc_P`. The protocol locks the cheapest selector that is not significantly worse than the best.
- **H4, safety:** wrong confirmations and voltage deviations are reported for every arm, including the truth-removed copies.

**Results** (2026-11-23; 1,916 simulated days on fresh seeds, 2 paths per input; `confirm.py`).

*Passive stage.* Of 64 inputs per case, the bank and the passive leader:

| Case | Repository identifier's own pick correct | Baseline bank has the truth | Refined bank has the truth | Baseline passive leader correct | Refined passive leader correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main | 19 | 40 | **64** | 33 | **64** |
| Short line 0.01 Ω | 12 | 31 | **64** | 18 | **47** |

*End of day, all inputs* (128 runs per row and case):

| Pipeline / arm | Main: correct | Main: ever wrong (> 0.2% V) | Short line: correct | Short line: ever wrong (> 0.2% V) | Short line: median h to correct | CNY/day (both cases) |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Baseline passive | 48% | 49% (27 runs; up to 0.33%) | 27% | 69% (33; up to 0.32%) | 1 | 0 |
| Refined passive | **100%** | **0%** | **78%** | 15% (0; up to 0.10%) | 1 | 0 |
| Refined + committee | 100% | 0% | 81% | 15% (0) | 1 | 0.59 |
| Refined + strong greedy | 100% | 0% | 80% | 16% (0) | 1 | **0.47** |
| Refined + certified | 100% | 0% | 82% | 17% (0) | 1 | 0.96 |
| Refined + random, matched | 100% | 0% | 80% | 16% (0) | 1 | 0.48 |

*Hypotheses.*
- **H1 supported, decisively.** The refined pipeline beats the baseline by 66 to 0 (main) and 68 to 3 (short line) paired runs; pooled 134 to 3, p < 10⁻³⁰. Wrong confirmations fall from 49% and 69% to 0% and 15%, and none of the refined wrong confirmations is more than 0.2% off in voltage (baseline: 60 runs).
- **H2 not supported.** Committee probing over refined passive: 0 to 0 (main), 5 to 1 (short line), p = 0.22. On the short line every probing arm adds 2–4 points, which is not significant at this sample size.
- **H3: no selector differs.** Strong greedy against committee 3 to 4; certified against committee 4 to 3; strong greedy against certified 3 to 5 (p = 0.73).
  - By the pre-registered rule (the cheapest selector not significantly worse than the best), the protocol locks **the strong greedy `bgreedy_P`**. It matched the others' accuracy (231 of 256 against 232–233) while buying about half as many hours (0.55 against about 1.0 probed hours a day) at the lowest cost (0.47 CNY a day).
- **H4.** No AC voltage-band or volt-var deadband violation in 1,916 days (voltages 0.994–1.024 pu; 1.3 s per gate on average, at most 8.8 s). With the truth removed (64 runs per arm and case), a near-twin topology was confirmed:
- main case: in 83% of passive runs and 86–92% of probing runs;
- short line: in 69% of passive runs and 62–75% of probing runs (strong greedy 75%).

It was at most 0.16% off in voltage, and never more than 0.2%.

*What the confirmation establishes, and what it does not.*
- On these synthetic feeders with typical meter bias and one day of data, the refined pipeline is reliable: correct in every main-case run, 78% on the short line (against 27%), and no wrong confirmation that matters operationally.
- Its three parts are the revised step 1 (the truth was in the bank in 128 of 128 inputs), per-meter offsets and GLS scoring.
- Probing is safe and cheap, but adds little once the scoring is correct. Its measurable value is limited to short internal segments, and even there it is not yet significant.
- When probing is used, the strong greedy is the right selector: equally accurate, and it buys only when the risk it removes is worth the cost. *(Qualified by §16: part of its low cost came from a defect, fixed 2026-11-22.)*

### 16. Hard-case test with realistic demand-response prices (pre-registered 2026-11-22, before building)

*Why.* In §15 probing changed the outcome of 6 of 128 short-line runs, too few to measure what it can do. This test generates many new inputs under settings where one day of passive data is often not enough, and runs only the ambiguous ones.

*Two corrections found before this test.*
- **Cost.** The draft contract (3) charges price × u²/headroom at 0.30–0.60 CNY/kWh, so a block at a fraction of a home's headroom costs a fraction of its energy value. Re-priced as demand response (reductions at 4 CNY/kWh, increases at 1.2 CNY/kWh), the logged §15 blocks cost about 9–10 times the contract figure: 13–42 CNY a day on the days when probing ran, against 1–4 CNY. This test settles every block at those linear DR prices: reductions (discharge, paused charging) at 4 CNY/kWh, the upper end of the 3–4 CNY/kWh peak-shaving DR compensation reported for the Yangtze River Delta; increases (extra charging) at 1.2 CNY/kWh, the upper end of the 0.3–1.2 CNY/kWh valley-filling payments. The caps are raised to 20 CNY an hour and 100 CNY a day. The price of certainty becomes V = 1,000 CNY, about one field verification visit.
- **Strong greedy defect.** Its risk sum took the 10 candidates of largest posterior weight. When the leader had many operationally equivalent twins, all 10 were twins, the sum saw one class, and the planner stopped buying while real rivals remained. In §15 this happened at 716 short-line gates and in 29 of 128 natural short-line runs, 11 of which ended undecided. The fixed version treats each operational class as one hypothesis with weight W_c (the sum of its members' weights): the leader's class is the gate's equivalence set, the stopping rule's own, and the other candidates with E < 25 are grouped by the same full-day voltage test (`PLANNER_REVISION.md` §10). The §15 strong-greedy numbers are those of the defective version.

*Inputs.* New seeds 33–64 on all four feeders (128 per condition), none used before; 1-day history; refined pipeline (revised step 1, offsets, GLS, trust gate).

| Condition | What makes it hard |
| --- | --- |
| `hard_short01` | One internal segment at 0.01 Ω, meter bias 0.05% |
| `hard_short005` | One internal segment at 0.005 Ω (1.7 times the resistance floor), meter bias 0.05% |
| `hard_noisy` | True topology as in the case file; meter stress: 0.1% bias and 0.1% random voltage noise |

Only S1 inputs (after the passive day, the leader is wrong or not yet confirmable) are run, with 2 paths each.

*Arms.* `passive_bic`; `cert_P`; `random_matched_P` (random single-home blocks at `cert_P`'s hours and spend); `qbc_P`; `bgreedy_P` (V = 1,000 CNY); `bgreedy_lowV_P` (frugal strong greedy, V = 200 CNY).

*Hypotheses.*
- **HC1, probing works where passive is not enough:** `bgreedy_P` confirms the true topology by the end of the day in more S1 runs than passive, pooled over the three conditions (paired by input and path, exact McNemar, p < 0.05). The same test for `qbc_P` is secondary.
- **HC2, targeting:** `bgreedy_P` against `random_matched_P`, pooled (p < 0.05). The random arm is matched to `cert_P`'s spend, so costs are reported alongside.
- **HC3, selector:** no directional prediction. Report correct confirmations, median hour of the first correct confirmation, kWh and CNY a day. The rule of §15 applies: the cheapest selector not significantly worse than the best.
- **HC4, safety:** wrong confirmations (and their voltage error), voltage-band and volt-var violations for every arm.

*Decision.* If HC1 fails, the paper states that probing adds no measurable value even on hard inputs at realistic prices.

**What the §15 probing really cost (re-priced from the logged blocks, 2026-11-22).** The energy each arm moved is exact; only the price changes. Most blocks were increases (extra charging): reductions were 16–33% of the energy.

| §15 arm, short line, all 128 runs | kWh a day | Draft contract, CNY a day | DR prices (4 / 1.2 CNY/kWh), CNY a day | On days with probing | Every kWh at 4 CNY | Largest day at DR prices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong greedy (defective version) | 4.4 | 0.83 | 7.3 | 23 | 17.7 | 75 |
| Committee | 4.8 | 1.09 | 8.7 | 21 | 19.0 | 44 |
| Certified | 9.3 | 1.76 | 16.5 | 40 | 37.2 | 108 |
| Random, matched | 3.7 | 0.88 | 8.0 | 19 | 15.0 | 38 |

On the main case probing ran on 8% of days, at 10–21 CNY on those days. Over the whole confirmation (both cases, all strata, 384 runs per arm), the strong greedy would have cost about 4,300 CNY at DR prices (9,600 at a flat 4 CNY/kWh), the committee 4,100, the certified planner 9,700. With the truth removed, probing ran on 47–78% of days at 25–82 CNY per probed day, because the planners chase a near-twin they cannot rule out.

**Results (run 2026-11-22/23, 3,300 runs, 3.2 h on 14 workers).**

*Inputs.* All 384 inputs built without error. Inputs still ambiguous after the passive day (S1): 68 of 128 (0.01 Ω), 121 of 128 (0.005 Ω), 86 of 128 (meter stress). The truth was in the bank in every S1 input, so these runs measure what probing can add, not the truth-missing failure. Each S1 input ran 2 paths, 550 runs per arm.

*Accuracy and cost per arm, pooled over the three conditions (S1 only).* "Correct" means the true topology is confirmed at the end of the day. Costs are at the linear DR prices above.

| Arm | Correct | Undecided | Ever confirmed wrong | Median hour of first correct confirmation | Probed hours a day | kWh a day | CNY a day | CNY per probed day | Largest day | Net extra correct over passive | CNY per extra correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 162 (29%) | 212 | 260 | 10 | 0 | 0 | 0 | 0 | 0 | — | — |
| Strong greedy, V = 1,000 | 222 (40%) | 148 | 266 | 7 | 3.2 | 15.0 | 23.8 | 39.9 | 104 | 60 | 219 |
| Strong greedy, V = 200 | 222 (40%) | 156 | 264 | 7.5 | 3.0 | 12.1 | 16.8 | 28.6 | 84 | 60 | 154 |
| Committee | 205 (37%) | 164 | 260 | 7 | 3.3 | 8.9 | 16.3 | 27.3 | 104 | 43 | 209 |
| Certified | 224 (41%) | 144 | 264 | 7 | 3.2 | 16.6 | 26.7 | 44.6 | 108 | 62 | 237 |
| Random, same hours | 173 (31%) | 202 | 256 | 10 | 3.2 | 7.0 | 13.8 | 23.0 | 57 | 11 | 689 |

*By condition (correct confirmations of S1 runs).*

| Condition | S1 runs | Passive | Strong greedy, V = 1,000 | Strong greedy, V = 200 | Committee | Certified | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 Ω short line | 136 | 73 (54%) | 87 (64%) | 87 (64%) | 80 (59%) | 89 (65%) | 73 (54%) |
| 0.005 Ω short line | 242 | 30 (12%) | 51 (21%) | 53 (22%) | 46 (19%) | 53 (22%) | 33 (14%) |
| Meter stress (0.1% bias, 0.1% noise) | 172 | 59 (34%) | 84 (49%) | 82 (48%) | 79 (46%) | 82 (48%) | 67 (39%) |

*Hypotheses (exact McNemar on runs paired by input and path; "a / b" = runs only the first arm got right / only the second).*

| Test | 0.01 Ω | 0.005 Ω | Meter stress | Pooled | p (pooled) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **HC1** strong greedy vs passive | 14 / 0 | 22 / 1 | 26 / 1 | **62 / 2** | 2·10⁻¹⁶ | **Supported** |
| HC1 secondary: committee vs passive | 9 / 2 | 17 / 1 | 20 / 0 | 46 / 3 | 7·10⁻¹¹ | Supported |
| **HC2** strong greedy vs random | 16 / 2 | 20 / 2 | 19 / 2 | **55 / 6** | 5·10⁻¹¹ | **Supported** |
| Committee vs random (similar spend: 16.3 against 13.8 CNY a day) | 9 / 2 | 14 / 1 | 16 / 4 | 39 / 7 | 2·10⁻⁶ | Targeting, not energy, does the work |
| Strong greedy V = 200 vs random | | | | 54 / 5 | 2·10⁻¹¹ | |
| Random vs passive | | | | 17 / 6 | 0.035 | Untargeted energy helps a little |
| Strong greedy vs certified | 5 / 7 | 4 / 6 | 6 / 4 | 15 / 17 | 0.86 | Indistinguishable |
| Strong greedy V = 200 vs V = 1,000 | 3 / 3 | 3 / 1 | 3 / 5 | 9 / 9 | 1 | Indistinguishable |
| Strong greedy V = 200 vs certified | | | | 18 / 20 | 0.87 | Indistinguishable |
| Strong greedy vs committee | 9 / 2 | 11 / 6 | 9 / 4 | 29 / 12 | 0.012 | Strong greedy better |
| Strong greedy V = 200 vs committee | | | | 28 / 11 | 0.009 | Strong greedy better |
| Committee vs certified | | | | 17 / 36 | 0.013 | Committee worse |

- **HC1 and HC2 hold, in every condition separately and pooled.** Where one day of passive data is not enough, targeted probing confirms the true topology in 11 more runs per 100 (29% → 40%) and brings the first correct confirmation forward from hour 10 to hour 7–7.5. Random blocks on the same hours add 2 runs per 100.
- **HC3, selector.** The best arm is the certified planner (224). The strong greedy at V = 200 CNY (222) is not significantly worse (18 / 20) and is the cheapest such arm: 16.8 CNY a day against 26.7. The committee rule is cheaper by 0.5 CNY a day but significantly worse than the certified planner (17 / 36) and than either strong greedy. By the §15 rule the selector is **the strong greedy with V = 200 CNY**. V = 1,000 CNY bought 25% more energy and 42% more money for the same accuracy (9 / 9). Unlike §15, the committee rule now loses to the strong greedy; the fixed class weights are the likely reason, since the committee rule still spreads effort over the leader's twins.
- **HC4, safety.** No voltage-band or volt-var violation in 3,300 simulated days. Probing did not raise wrong confirmations: every arm confirmed a wrong topology at some point in 47–48% of runs. Almost all of these are the short segment merged away (missing line, at most 0.15% off in voltage). Under meter stress, 3–4 runs per arm had a wrong confirmation more than 0.2% off: passive 3, at most 0.29%; probing arms 3–4, at most 0.25%.
- **Cost at realistic prices.** On ambiguous inputs probing ran on 59–60% of days, at 29–45 CNY per probed day for the targeted arms and at most 108 CNY on any day. Each extra correct identification cost about 150 CNY with the frugal strong greedy and 220–240 CNY with the others, against about 690 CNY with random blocks and about 1,000 CNY for a field visit.
- **What probing cannot do in a day.** The 0.005 Ω segment is still resolved in at most 22% of ambiguous runs, and under meter stress about half of the runs stay undecided at the end of the day. Probing helps, but it does not make these cases easy.

*Disclosed after the run: the daily cap was broken in 9 of 2,750 probing runs, by at most 8.3 CNY.*
- **Cause.** The block for hour h+1 is sealed at gate h while the block for hour h has not yet run. The budget check and the battery state of charge ignored that in-flight block.
- **Fix (2026-11-23).** `run.committed` now counts the sealed, not yet executed blocks against the day's budget and the battery energy; a unit test covers it (21 tests pass).
- **Effect.** Rerunning the 9 runs with the fix kept all of them within the cap (53–97 CNY) with the same outcomes. The budget can only have bound in runs that spent more than 80 CNY, within one block of the cap: 47 of the 2,750 probing runs, 18 of them correct, spread over all probing arms. Even if every one of those 18 had failed, HC1 and HC2 would still hold by wide margins (62 / 2 and 55 / 6). The battery part of the fix can change block sizes in other runs; those are not re-run here.
- **The random arm is not matched on spend.** It is matched on hours and on `cert_P`'s target spend, but its single-home blocks hit the headroom ceiling, so it spent about half as much (13.8 against 26.7 CNY a day). The committee rule spends about the same as random and still wins 39 to 7, so HC2 does not depend on the spend gap.

### 17. Which V? A sweep of the price of certainty (rule registered 2026-11-23, while the sweep ran, before any result was read)

*Why.* V = 1,000 and V = 200 CNY were set by hand, and the hard-case test only compared those two. The logged runs show that the strong greedy's risk bound U is about twice the realised error (1.2–3 times; `paper/strategies.tex`, Section 4), so V is in bound units, not in CNY per wrong identification.

*Runs.* The strong greedy at V = 50, 100, 200, 500, 1,000 and 2,000 CNY, with the 2026-11-23 budget fix, on the same 275 ambiguous inputs as §16, 2 paths each (tag `vsweep`, started 03:07). Passive comes from §16, because passive buys nothing and the fix does not touch it. The V = 200 and V = 1,000 reruns also show what the fix changed. These are development data: V is chosen here and confirmed in §18 on new seeds.

*Rule for the best V* (`vsweep.py`).
- **Primary: the largest net value per run,** NV(V) = 1,000 CNY × (correct runs with V − correct runs passive) / runs − mean cost a day. 1,000 CNY is the value of one correct identification, about one field visit avoided.
- **Secondary:** the cheapest V whose correct confirmations are not significantly worse than those of the most accurate V (exact McNemar, p ≥ 0.05).
- If the two disagree, the primary rule decides and both are reported.

**Results (run 2026-11-23 03:07–06:13, 3,300 runs on 14 workers, none failed).**

*Accuracy and cost against V, pooled over the three conditions (550 S1 runs per arm).* "Gained / lost" counts the runs only this arm, or only passive, got right (exact McNemar, paired by condition, input and path).

| V (CNY) | Correct | Gained / lost vs passive | Undecided | Ever confirmed wrong | Median hour of first correct | Probed hours a day | kWh a day | CNY a day | Largest day | Net value per run (CNY) | CNY per extra correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 162 (29%) | — | 212 | 260 | 10 | 0 | 0 | 0 | 0 | 0 | — |
| 50 | 190 (35%) | 33 / 5 | 184 | 264 | 8 | 1.9 | 5.5 | 6.8 | 53 | 44.1 | 134 |
| 100 | 205 (37%) | 47 / 4 | 166 | 263 | 8 | 2.6 | 8.8 | 11.6 | 68 | 66.6 | 148 |
| 200 | 206 (37%) | 47 / 3 | 164 | 268 | 8 | 3.0 | 11.4 | 16.5 | 83 | 63.5 | 207 |
| 500 | 210 (38%) | 52 / 4 | 167 | 261 | 7 | 3.2 | 13.4 | 21.3 | 93 | 66.0 | 244 |
| 1,000 | 209 (38%) | 52 / 5 | 165 | 265 | 7 | 3.2 | 14.1 | 23.4 | 93 | 62.1 | 274 |
| 2,000 | 215 (39%) | 56 / 3 | 160 | 263 | 7 | 3.2 | 14.6 | 24.7 | 93 | 71.6 | 257 |
| 100,000 (limit, reported only) | 214 (39%) | 55 / 3 | 161 | 264 | 7 | 3.2 | 15.2 | 26.4 | 95 | 68.2 | 279 |

*By condition (correct confirmations of S1 runs).*

| Condition | S1 runs | Passive | V = 50 | V = 100 | V = 200 | V = 500 | V = 1,000 | V = 2,000 | V = 100,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 Ω short line | 136 | 73 | 80 | 81 | 83 | 81 | 82 | 87 | 86 |
| 0.005 Ω short line | 242 | 30 | 38 | 49 | 50 | 51 | 50 | 50 | 48 |
| Meter stress | 172 | 59 | 72 | 75 | 73 | 78 | 77 | 78 | 80 |

*Against the most accurate value (V = 2,000; runs only the other V got right / only V = 2,000).* V = 50: 13 / 38 (p = 0.0006, worse). V = 100: 16 / 26 (p = 0.16). V = 200: 10 / 19 (p = 0.14). V = 500: 6 / 11 (p = 0.33). V = 1,000: 0 / 6 (p = 0.03; the two arms differ in only 6 runs, all in the same direction). V = 100,000: 2 / 3 (p = 1).

*The V → ∞ point* (tag `vinf`, run 12:35–13:33 after the §18 runs, 550 runs, none failed; reported, never chosen). V = 100,000 CNY is the limit in which the budget caps alone decide how much is bought.

- **The rule's answer.** Primary rule: **V = 2,000** (71.6 CNY per run). Secondary rule: **V = 100** (11.6 CNY a day, not significantly worse than V = 2,000). They disagree, so by the registered rule V = 2,000 goes to §18, with V = 100 alongside (§18 addendum).
- **Which V is best: the data do not single one out.** Every V from 100 to 2,000 confirmed the true topology in 205–215 of 550 runs and is worth 62–72 CNY per run. V = 2,000 leads V = 100 by 10 correct runs, inside the noise (p = 0.16), and costs 13 CNY a day more. Only V = 50 is clearly too low: it probes 1.9 hours a day and loses 25 correct runs. *(Corrected by §18: on fresh seeds V = 2,000 beat V = 100, 38 / 14, p = 0.001. "Not significantly worse" was not equivalence.)*
- **Why the curve is flat.** From V ≈ 500 the number of probed hours stops growing (3.2 a day): the buy rule passes whenever the bound falls at all, so a larger V only buys larger blocks, up to the ceilings and caps (`paper/strategies.tex`, Section 4). This also answers whether the sweep should go past 2,000: V → ∞ can add only block size, not hours. **Confirmed by the V = 100,000 point:** the same accuracy as V = 2,000 (214 against 215 correct; the two differ in 5 runs), the same 3.2 probed hours a day, slightly larger blocks (15.2 against 14.6 kWh a day) and a lower net value (68 against 72 CNY per run). Above V = 2,000 nothing more is bought that helps.
- **Agreement with the calibration.** The decision-theoretic V ≈ L/k ≈ 1,000/2 = 500 CNY lies inside the flat range.
- **Safety.** No violations at any V. Wrong confirmations did not depend on V: 261–268 runs ever confirmed wrong against 260 passive; 3–5 runs more than 0.2% off against 3 passive, at most 0.29% off, the same largest error as passive.

*What the budget fix changed (V = 200 and V = 1,000 rerun against §16).* V = 200: 206 correct against 222 in §16 (4 gained, 20 lost, p = 0.002), 16.5 against 16.8 CNY a day. V = 1,000: 209 against 222 (7 / 20, p = 0.02), 23.4 against 23.8 CNY a day.
- The budget half of the fix barely moved spending, so the loss comes from the battery half. In §16 the block for hour h+1 was sized against the batteries' charge before hour h's block ran, so blocks could ask for energy the batteries no longer had. The simulator delivered the full block anyway (the battery gives what it holds, and the rest was assumed to come from EVs even when none were available), so some §16 accuracy most likely came from flexibility that did not exist.
- **Measured directly.** A new evaluation-only record in `run.py` (`headroom_excess`) compares each delivered block with what the batteries and EVs could give in that hour. On the first 30 ambiguous 0.005 Ω inputs at V = 200 (path 0), with the fix switched off, the runs reproduced §16 exactly (30 of 30: same outcome, cost and probed hours), so the record changes no decision and the runs are deterministic. In 9 of the 11 runs that probed, the old code delivered more than the resources could give, in 10 hours in all (9.4 kWh, up to 0.75 kW at one node). With the fix, none of the 11 did. In this sample the outcomes happened to be the same (8 correct either way), so the 20 / 4 loss above comes from other inputs; the diagnostic shows the mechanism, not its size.
- **Consequence for §16.** Every §16 probing arm had this defect (the certified planner, committee and random arms read the same battery state), so §16's accuracy is optimistic for all of them: by about 3 points (40% → 37–38%) for the strong greedy. Whether it changed the comparisons between arms is not known until §18, which runs every arm with the fix. With the fix, probing still clearly beats passive (47 / 3 at V = 200, p = 4·10⁻¹¹), and each extra correct identification costs 148 CNY at V = 100 to 257 CNY at V = 2,000, against 154–219 CNY in §16.

### 18. Confirmation of the lock on fresh seeds 65–96, with the truth present and removed (pre-registered 2026-11-23, before building)

*Why.* The selector, V and both fixes were chosen or found on the §16 inputs, so their result must be repeated on inputs never used before. Section 16 also had the truth in every ambiguous bank; the earlier finding that targeted probing confirms near-twins faster when the truth is missing (§12–13) has not been retested with the trust gate, the fixed strong greedy and DR prices.

*Inputs.* New seeds 65–96 on all four feeders, 128 per condition, with the same three conditions and settings as §16: `lock_short01` (0.01 Ω segment), `lock_short005` (0.005 Ω segment), `lock_noisy` (0.1% meter bias and noise). Refined pipeline, trust gate, linear DR prices (4 / 1.2 CNY/kWh, caps 20 / 100 CNY), budget fix.

*Runs.*
- **Natural ambiguous (S1) inputs,** 2 paths: `passive_bic`, the strong greedy at the V chosen in §17 (`bgreedy_V<V>_P`), `cert_P`, `random_matched_P`, `qbc_P`.
- **Truth-removed copies of the same inputs** (tag `missing`), 2 paths: `passive_bic` and the strong greedy at the chosen V.

*Hypotheses* (`lock.py`; exact McNemar, paired by input and path, pooled over the three conditions, p < 0.05).
- **L1, efficacy (primary):** the strong greedy confirms the true topology by the end of the day in more natural S1 runs than passive.
- **L2, targeting:** the strong greedy beats random blocks.
- **L3, selector:** the strong greedy is not significantly worse than the certified planner. Its comparison with the committee is reported.
- **L4, safety with the truth present:** no voltage-band or volt-var violations; wrong confirmations reported against passive.
- **T1, safety with the truth removed (primary):** the strong greedy does not confirm a wrong topology in significantly more runs than passive.
- **T2:** wrong confirmations more than 0.2% off in voltage, and the largest error, for both arms.
- **T3:** how often the trust gate lets probing run on truth-removed inputs, and what it costs.

*Decision.*
- Lock the strong greedy at the chosen V with DR prices if L1 and L2 hold and T1 passes.
- If T1 fails, the trust gate must be strengthened before the lock, and the paper reports the failure.
- If L1 fails, the §16 effect did not replicate, and the paper reports probing's value as unconfirmed.

*Addendum (registered 2026-11-23 06:20, after the §17 result and while the §18 inputs were still being built, before any §18 run).* The §17 rule picked V = 2,000 by a margin inside the noise: V = 100 to 2,000 do not differ significantly in accuracy, while the cost doubles. The strong greedy at V = 100, the §17 secondary choice, therefore also runs on the same natural S1 inputs (`bgreedy_V100_P`, tag `vcheck`, 2 paths). Secondary tests: V = 2,000 against V = 100, V = 100 against passive (exact McNemar, paired within condition, pooled), and the net value per run of every probing arm at 1,000 CNY per correct identification. They do not change the decision above; they show on fresh seeds what the extra spend of V = 2,000 buys.

**Results (run 2026-11-23 06:13–12:02, 3,724 runs on 14 workers, none failed).**

*Inputs.* All 384 inputs built without error.
- Ambiguous after the passive day (S1): 61 of 128 (0.01 Ω), 119 of 128 (0.005 Ω), 86 of 128 (meter stress). The truth was in the bank in every S1 input. Each ran 2 paths: 532 natural runs per arm, and 532 truth-removed runs per arm on copies of the same inputs.
- **The truth is missing in practice, not only in this test.** Outside S1, the bank construction itself missed the truth (stratum S3) in 2 of 128 inputs (0.01 Ω), 1 of 128 (0.005 Ω) and **26 of 128 under meter stress** (§16: 3, 0 and 30). These natural misses were not part of the registered runs, but they show that T1 is about a failure that happens in about one input in five when the meters are poor.

*Natural ambiguous runs, pooled over the three conditions (532 runs per arm).* Costs at the linear DR prices. Net value: 1,000 CNY per extra correct identification over passive, minus the mean cost a day (the §17 rule).

| Arm | Correct | Undecided | Ever confirmed wrong | Median hour of first correct | Probed hours a day | kWh a day | CNY a day | CNY per probed day | Largest day | Net value per run (CNY) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 170 (32%) | 197 | 230 | 11 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strong greedy, V = 2,000 | 229 (43%) | 145 | 221 | 8 | 3.2 | 13.4 | 23.2 | 40.4 | 99 | 87.7 |
| Certified | 223 (42%) | 151 | 219 | 8 | 3.1 | 14.0 | 23.6 | 41.1 | 96 | 76.0 |
| Committee | 217 (41%) | 156 | 224 | 8 | 3.3 | 8.2 | 15.6 | 27.1 | 65 | 72.8 |
| Random, same hours (defective cap, see below) | 190 (36%) | 179 | 221 | 11 | 3.1 | 6.6 | 13.0 | 22.6 | 43 | 24.6 |
| Random, same hours (fixed cap, rerun) | 187 (35%) | 181 | 224 | 11 | 3.1 | 6.8 | 14.9 | 25.9 | 56 | 17.1 |

*By condition (correct confirmations of S1 runs).*

| Condition | S1 runs | Passive | Strong greedy | Certified | Committee | Random | Random, fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 Ω short line | 122 | 56 | 77 | 75 | 78 | 62 | 60 |
| 0.005 Ω short line | 238 | 44 | 57 | 57 | 53 | 52 | 51 |
| Meter stress | 172 | 70 | 95 | 91 | 86 | 76 | 76 |

*Truth-removed copies of the same inputs (532 runs per arm).*

| Arm | Ever confirmed wrong | Undecided at the end | Wrong > 0.2% in voltage | Largest wrong voltage error | Days probed | Probed hours a day | CNY a day | CNY per probed day | Largest day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 318 (60%) | 292 | 2 | 0.24% | 0% | 0 | 0 | 0 | 0 |
| Strong greedy, V = 2,000 | 338 (64%) | 280 | 4 | 0.27% | 55% | 3.5 | 27.0 | 49.3 | 100 |

*Hypotheses ("a / b" = runs only the first arm met the criterion / only the second; exact McNemar, paired within condition by input and path).*

| Test | 0.01 Ω | 0.005 Ω | Meter stress | Pooled | p (pooled) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **L1** strong greedy vs passive, correct | 21 / 0 | 15 / 2 | 25 / 0 | **61 / 2** | 4·10⁻¹⁶ | **Supported** (each condition p < 0.01) |
| **L2** strong greedy vs random, correct | 17 / 2 | 10 / 5 | 20 / 1 | **47 / 8** | 8·10⁻⁸ | **Supported** (0.005 Ω alone p = 0.30) |
| L2 rerun, against the fixed random arm | 19 / 2 | 11 / 5 | 20 / 1 | 50 / 8 | 2·10⁻⁸ | Supported, slightly more strongly |
| L3 strong greedy vs certified, correct | 6 / 4 | 5 / 5 | 10 / 6 | 21 / 15 | 0.41 | Holds: not worse |
| L3 strong greedy vs committee, correct | 7 / 8 | 9 / 5 | 11 / 2 | 27 / 15 | 0.09 | Not significant |
| L4 ever confirmed wrong, strong greedy vs passive | 2 / 3 | 1 / 10 | 2 / 1 | 5 / 14 | 0.06 | Probing did not add wrong confirmations; no violations |
| **T1** ever confirmed wrong, truth removed | 17 / 5 | 13 / 6 | 7 / 6 | **37 / 17** | **0.009** | **Fails** |
| T2 wrong by more than 0.2%, truth removed | 0 / 0 | 0 / 0 | 2 / 0 | 2 / 0 | 0.5 | 4 against 2; at most 0.27% against 0.24% |

- **Efficacy replicated (L1, L2).** On fresh seeds and with every fix, the strong greedy confirmed the true topology in 43% of ambiguous runs against 32% passive (61 gained, 2 lost), and beat random blocks on the same hours 47 to 8. The first correct confirmation came at hour 8 instead of 11. The size matches §17 (+10 points there, +11 here).
- **Selector (L3).** The strong greedy is not worse than the certified planner (21 / 15) and has the largest net value of the four probing arms (88 CNY per run, against 76 for the certified planner and 73 for the committee rule).
- **Safety with the truth present (L4).** No voltage-band or volt-var violation in 3,724 simulated days. Probing did not add wrong confirmations (5 / 14, p = 0.06). No natural run had a wrong confirmation more than 0.2% off (largest 0.16–0.19% in every arm). The budget cap held in every run, and no arm except random exceeded the battery and EV headroom.
- **T1 fails: with the truth removed, probing confirms a wrong topology more often than passive** (64% against 60% of runs; 37 / 17, p = 0.009). The effect comes mostly from the 0.01 Ω condition (17 / 5, p = 0.02); under meter stress it is 7 / 6. Compared with the first round (76% against 44% before the trust gate), the gap has shrunk from 32 points to 4 but has not gone.
- **T3, the trust gate did almost nothing on truth-removed inputs.** It let probing run on 55% of their days, against 58% of natural days, and these runs cost more (27.0 against 23.2 CNY a day) because the planner chases rivals it cannot separate, up to the 100 CNY daily cap.
- **Decision, as registered.** L1 and L2 hold, but T1 fails. **The lock is not confirmed: the trust gate must be strengthened before the lock, and the paper reports the failure.** *(Overridden 2026-11-23 after §19 by the user's decision: locked with the failure disclosed; see the decision at the top.)*

*Addendum result: V = 2,000 against V = 100 on the same natural S1 inputs (tag `vcheck`, run 12:02–12:35, 532 runs, none failed).*

| Arm | Correct | Undecided | Ever confirmed wrong | Median hour of first correct | Probed hours a day | kWh a day | CNY a day | CNY per probed day | Largest day | Net value per run (CNY) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 170 (32%) | 197 | 230 | 11 | 0 | 0 | 0 | 0 | 0 | 0 |
| Strong greedy, V = 100 | 205 (39%) | 167 | 223 | 9 | 2.5 | 8.0 | 10.7 | 19.5 | 53 | 55.1 |
| Strong greedy, V = 2,000 | 229 (43%) | 145 | 221 | 8 | 3.2 | 13.4 | 23.2 | 40.4 | 99 | 87.7 |

| Test | 0.01 Ω | 0.005 Ω | Meter stress | Pooled | p (pooled) |
| --- | --- | --- | --- | --- | --- |
| V = 2,000 vs V = 100, correct | 13 / 5 | 11 / 6 | 14 / 3 | **38 / 14** | **0.001** |
| V = 100 vs passive, correct | 14 / 1 | 11 / 3 | 16 / 2 | 41 / 6 | 2·10⁻⁷ |

- **On fresh seeds the larger V is clearly better.** V = 2,000 confirmed 24 more true topologies than V = 100 (38 gained, 14 lost) and is worth 33 CNY more per run, although it costs 12.5 CNY a day more. V = 100 still beats passive (41 / 6). Neither had violations, headroom excess or a run over the cap.
- **This corrects §17.** There, V = 2,000 against V = 100 was 26 / 16 (p = 0.16), and the curve was read as flat from 100 to 2,000. The direction was the same; the test was underpowered. Pooled over both rounds it is 64 / 30. The registered primary rule (largest net value) picked the right value; the secondary rule ("not significantly worse") read absence of evidence as equivalence.
- **Where V = 100 loses.** It probes fewer hours (2.5 against 3.2 a day) and smaller blocks (8.0 against 13.4 kWh a day), most visibly on the 0.005 Ω segment (1.9 against 2.8 probed hours a day), where the evidence needed is largest.

*Exploratory analysis after the result (not registered).*
- **What the extra wrong confirmations are.** All 37 runs that only the strong greedy confirmed wrong are near-twins of the removed truth: 16 wrong placements of a customer, 13 extra lines, 8 merged short segments. On the short lines they were at most 0.14% off in voltage (median 0.07%). Under meter stress, 2 of 7 were more than 0.2% off (at most 0.27%). In every one of the 37, passive ended the day undecided.
- **The same mechanism produces the gain and the harm.** Probing turns "undecided" into "confirmed". With the truth present, 42 of the 61 gains were runs where passive already had the true topology in the lead but could not confirm it. With the truth removed, the leader that gets confirmed is the closest near-twin.
- **The wrong topologies predict voltages almost as well as the true one.** With the truth removed, wrong final confirmations were off by a median of 0.06% on the short lines (at most 0.14%); with the truth present, correct confirmations were off by 0.05%, because the fitted line parameters are not exact. Under meter stress: 0.14% against 0.10%. The harm is in the topology record (which customer hangs on which branch, a line that does not exist), not in voltage operation.
- **No logged statistic tells a truth-missing bank from a natural one.** At the gates where the strong greedy probed, the leader's adequacy ratio had the same distribution with and without the truth (median 0.97 in both; area under the ROC curve 0.48–0.52). So did the gap to the best non-equivalent rival (0.45–0.54), the rivals' posterior mass (0.54–0.64) and the size of the equivalence set (0.45–0.49). Leader switches during the day were more frequent with the truth removed (2.4 against 1.4 per run at 0.01 Ω, 3.6 against 2.0 under meter stress), but occurred in 52–66% of natural runs too. Confirmed leaders replaced by another confirmed leader were equally common (14–15% at 0.01 Ω). **So the trust gate cannot be strengthened by tightening its threshold:** a near-twin that is not the truth fits the meters as well as the truth does.

*A missing-truth check from one-hour-ahead predictions (development run after the result, not registered; tag `preq`, run 12:35–14:30, 1,064 runs, none failed).*
- **Idea.** A near-twin fits the data it was fitted on, but a probe moves the feeder to a new operating point. If the twin is wrong, it should predict the meters' response to the next probe worse than the truth would. At every gate, each candidate, fitted on the data up to the previous hour, predicts that hour's meter readings before seeing them. The squared prediction error is scaled by its expected size (meter noise plus the uncertainty of the fitted line parameters), so it averages 1 per reading when the candidate is right. It is summed separately over probed and passive hours (`run.py`, `prequential_check`; evaluation only, it changes no decision).
- **Runs.** Passive and the strong greedy (V = 2,000) on path 0 of all 266 natural S1 inputs of §18 and their truth-removed copies. On this path the T1 gap is 172 against 163 runs ever confirmed wrong.
- **The check is correctly scaled.** With the truth present, the truth's own ratio had a median of 1.00 over probed hours in all three conditions (95th percentile 1.10–1.28) and 1.00 over passive hours (1.07–1.09).
- **It does not separate truth-missing banks.** Areas under the ROC curve for the leader's probed-hour ratio, its probed-minus-passive contrast, a z-score and the best ratio in the bank or among the leader's equivalents: 0.53–0.58 at the first confirmation, 0.47–0.55 at the last gate and 0.48–0.54 over all probed gates, by condition; pooled 0.50–0.55. That is no better than the logged statistics above.
- **Vetoing on it only trades one error for the other.** Withholding a confirmation when the leader's contrast exceeds t: at t = 0.10, the strong greedy keeps 94 correct confirmations instead of 110 (passive 78) and is ever wrong in 161 truth-removed runs instead of 172 (passive 163). At t = 0.05 it falls below passive on correct confirmations (76) and is still wrong in 151. (Approximate: the vetoed paths were not re-simulated.)
- **Why.** The difference between a near-twin and the truth is a few units of fit over the whole day (E ≈ 3–10), spread over the 60–120 meter readings of the probed hours. A check that asks only "does the leader fit?" compares that excess with the random spread of a sum of 60–120 squared errors, about 11–15 units. The comparison with the truth inside the bank succeeds because it is relative: both candidates see the same noise. **So a missing truth cannot be detected by testing the leader alone; it can only be found by building the alternatives it would be among.** The candidate is to repeat the bank's neighbourhood closure around the current leader each day, using the probe data, and to withhold confirmation while a new neighbour fits as well as the leader. Its realistic test is the natural truth-missing inputs (§19).

*A defect in the random arm, found by the new headroom record.* `random_matched_P` delivered more than the batteries and EVs could give in 197 of its 532 natural runs (109 kWh in all); every other arm had none.
- **Cause.** Its block cap took a battery's rated power for upward blocks without checking whether the battery was already full, and ignored the charging efficiency.
- **Fix (2026-11-23).** `planner.random_block` now caps blocks with the same state-of-charge-aware headroom as every other arm (`sc.headroom`). The test now checks random blocks at empty, half and full batteries; 21 tests pass.
- **Rerun with the fixed cap** (tag `rfix`, run 13:33–14:28, 532 runs on the same inputs, replaying cert_P's logged hours and costs, none failed). No run exceeded the headroom (against 197). The fixed arm was correct in 187 runs against 190 with the defect (0 gained, 3 lost, p = 0.25) and still beat passive (20 / 3, p = 0.0005). It cost 14.9 against 13.0 CNY a day for about the same energy: the mean price rose from 1.98 to 2.19 CNY/kWh, so more of the energy was reductions (4 CNY/kWh) and less was charging a battery that had no room.
- **Effect on L2.** Against the fixed random arm the strong greedy wins 50 to 8 (p = 2·10⁻⁸; registered result 47 to 8). The defect had helped the random arm slightly, as expected, and the registered L2 verdict stands. §16 HC2 was not rerun; by the same direction it is conservative.

### 19. Inputs whose truth the bank itself missed (registered 2026-11-23 13:05, before any run)

*Why.* T1 removed the truth by hand. The bank construction missed the truth on its own (stratum S3) in 62 of the 768 hard-case and lock-test inputs, 56 of them under meter stress (§16: 33, §18: 29). These are the realistic truth-missing cases and have never been run.

*Runs.* Every natural S3 input of `hard_short01`, `hard_noisy`, `lock_short01`, `lock_short005` and `lock_noisy` (`hard_short005` has none), 2 paths: `passive_bic` and `bgreedy_V2000_P` (tag `s3nat`), with all fixes. They run after the fixed random arm and the V → ∞ point.

*Measures.* For both arms: runs ever confirmed wrong, wrong confirmations more than 0.2% off in voltage and the largest error, wrong types, undecided at the end of the day, days probed and cost. **N1:** exact McNemar on "ever confirmed wrong", strong greedy against passive, paired within condition by input and path, pooled. With about 124 pairs the test has little power, so the result is reported as a measurement, not as a pass or fail of the lock.

**Results (run 2026-11-23 14:31–14:49, 248 runs on 14 workers, none failed).** 62 inputs, 124 runs per arm. No input had the truth in its bank.

| Condition | Inputs | Ever confirmed wrong: passive | Strong greedy | Wrong > 0.2% (passive / strong greedy) | Largest wrong error | Days probed | CNY a day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 Ω short line (§16) | 3 | 6 of 6 | 5 of 6 | 0 / 0 | 0.10% | 33% | 18.3 |
| Meter stress (§16) | 30 | 28 of 60 | 30 of 60 | 11 / 12 | 0.32% | 17% | 8.5 |
| 0.01 Ω short line (§18) | 2 | 4 of 4 | 4 of 4 | 0 / 0 | 0.10% | 25% | 0.2 |
| 0.005 Ω short line (§18) | 1 | 2 of 2 | 2 of 2 | 0 / 0 | 0.11% | 0% | 0 |
| Meter stress (§18) | 26 | 14 of 52 | 15 of 52 | 0 / 0 | 0.17% | 12% | 7.9 |
| **Pooled** | **62** | **54 of 124 (44%)** | **56 of 124 (45%)** | 11 / 12 | 0.32% (both) | 15% | 8.3 |

- **N1: probing did not measurably raise wrong confirmations on natural misses** (4 / 2, p = 0.69; wrong at the end of the day 42 against 38, 4 / 0, p = 0.13). As registered, this is a measurement with little power, not a pass.
- **The four extra wrong confirmations** were 0.13–0.26% off in voltage (2 missing lines, 2 wrong placements); passive ended the day undecided in all four.
- **Natural misses are worse in voltage than hand-removed ones, but passive causes this.** Under meter stress, passive itself confirmed a wrong topology more than 0.2% off in 11 runs (up to 0.32%); probing added one. With the truth removed by hand, the wrong topologies were near-twins at most 0.27% off.
- No violations, no headroom excess, no run over the cap.

*Exploratory, found while this ran: the trust gate's closure flag is the part that works.* The trust gate has two parts: the bank's neighbourhood closure finished without hitting a cap, and the leader's adequacy ratio is within the meter class. Splitting the runs by the first part:

| Inputs | Truth-missing inputs | Runs per arm | Runs probed | Ever confirmed wrong: passive / strong greedy | McNemar | Wrong at end of day: passive / strong greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| §19, closure hit a cap | 51 | 102 | 0 | 45 / 45 | 0 / 0 | 33 / 33 |
| §19, closure finished | 11 | 22 | 19 | 9 / 11 | 4 / 2 | 5 / 9 |
| §18 T1, parent closure hit a cap | 51 inputs' copies | 102 | 0 | 22 / 22 | 0 / 0 | — |
| §18 T1, parent closure finished | 215 inputs' copies | 430 | 292 | 296 / 316 | **37 / 17** | — |

- **The closure flag caught 51 of the 62 natural misses (82%).** When the truth is missing because the bank could not be completed, the closure runs into its size or round cap, the gate forbids probing, and the strong greedy behaves exactly like passive. The adequacy part never acted: over all 6,200 gates of these runs the leader's ratio stayed between 0.89 and 1.05 (median 0.97), below the gate's 1.2.
- **All of T1's harm came from banks whose closure had finished.** Removing the truth by hand from a completed bank creates exactly the case the closure flag cannot see. That is a fair worst case, but in these inputs it arose naturally in only 11 of 461 inputs on which probing would be allowed (ambiguous or truth-missing, closure finished): 2.4% overall, 3 of 359 on the short lines (0.8%) and 8 of 102 under meter stress (7.8%).
- **On those 11 natural cases** probing ran in 19 of 22 runs and was wrong at the end of the day in 9 against 5, the same direction as T1, from a very small sample.
- **Size in practice (an estimate, not a test).** Among completed banks with the truth missing, T1 added 4.7 points of runs ever confirmed wrong (316 against 296 of 430). Weighted by the 2.4% of eligible inputs where this happens naturally, that is about 0.1 point of all probe-eligible runs, against the 11 points of extra correct confirmations on ambiguous inputs (§18 L1). Under meter stress (7.8% prevalence) it is about 0.4 point. Every extra wrong topology seen was 0.27% off in voltage or less.
- Closure caps were also common on natural ambiguous inputs under meter stress (78 of 172, 45%), where the gate also blocks probing. Probing's §18 meter-stress gain (25 / 0) therefore came from roughly half of those inputs.

## Recommendations before the lock

Updated after the second round (2026-11-21). Items 1–3 replace the first round's items 1 and 3.

1. **Adopt the revised step 1 as the protocol's bank construction.**
   - Per-meter voltage offsets in the identifier input (one extra demeaned run) and in every hourly refit (BIC counts n extra parameters).
   - Keep the identifier's pre-contraction path in the bank.
   - BIC neighbourhood closure (margin 6, neighbour grouping up to 8 children), with the round and size caps raised so that they never bind (closure costs seconds), and a bank cap set by BIC gap rather than by count.
   - Require at least one day of passive history; the first round's 12-hour histories caused all of its truth-missing banks under ideal meters.
0. **Adopt GLS scoring for the shared root-meter error** (§14–15, `PLANNER_REVISION.md` §11) in step 1, the hourly refit, the BIC, the adequacy ratio and the planner. It was the largest single improvement, and it follows from how the method measures, not from the simulator.
1a. **Lock the strong greedy as the block selector** (pre-registered rules, §15–17), with the class-weight fix and the committed-block fix, **at V = 2,000 CNY (the §17 rule).** *Locked 2026-11-23 by the user's decision, with T1's failure disclosed (decision at the top; PROTOCOL §11).* §18 replicated efficacy (61 to 2 against passive, 50 to 8 against the fixed random arm) but failed T1. On fresh seeds V = 2,000 beat V = 100 (38 to 14, p = 0.001; 88 against 55 CNY net value per run), so V = 2,000 is the value to lock; the §17 curve only looked flat because its test was underpowered. Settle every block at linear DR prices (reductions 4 CNY/kWh, increases 1.2 CNY/kWh; caps 20 CNY an hour and 100 CNY a day). Keep the certified planner, the committee rule and passive as comparison arms in Stage M, and report the certified bound B as the certificate. State in the paper that probing adds little where passive data already suffice (§15, 5 to 1), and that on ambiguous inputs it raised correct confirmations from 29% to 37–39% (§17 with all fixes, 47–56 gained and 3–5 lost) at about 150–260 CNY per extra correct identification.
2. *(Superseded by 1a.)* **Replace the planner's guarantee with the certified miss bound, and choose blocks with the simplest rule that matches it** (updated 2026-11-22 after §13). The union-bound tail makes the v3 planner idle on realistic banks; the miss bound of `PLANNER_REVISION.md` contains it as one allocation and explains why. In the head-to-head comparison, a query-by-committee rule matched the certified planner's accuracy at 64–72% of its cost in three of four cases (the same cost with ample flexibility). Suggested split:
   - the bound `B(u)` as the reported certificate and the stopping rule (buy only while `B` falls by at least cost/V);
   - the committee rule as the operational block selector, with `cert_P` kept as a comparison arm in Stage M;
   - in the paper, no claim that the planner beats heuristics. The supported claims are the revised step 1, targeting against random excitation, and the certificate.
3. **Gate probing on trust in the bank.** Because probing accelerates wrong confirmations when the truth is missing (76% against 44%), probe only when the leader's adequacy ratio is within the meter class and the closure completed without binding a cap. Report the operational equivalence (voltage deviation) of every confirmation.
   *Updated after §18:* this gate is not enough. With the truth removed, probing still raised wrong confirmations (64% against 60%), and the adequacy ratio was distributed the same way with and without the truth. A check that uses the probe responses themselves is needed (a test of whether the leader's fitted parameters or residuals shift when the probed hours are added); it must be developed and then confirmed on fresh seeds with the truth removed, before the lock. The truth was missing naturally in about one meter-stress input in five, so the check matters in practice.
   *Updated after the detector run and §19:* a check of the leader alone cannot work. The one-hour-ahead prediction check was correctly scaled but did not separate the cases (ROC area 0.47–0.58), because a near-twin's excess misfit is smaller than the random spread of any test of one candidate. The gate's closure flag already catches 82% of natural misses (51 of 62); the blind spot is a completed bank without the truth, which arose in 2.4% of probe-eligible inputs (7.8% under meter stress), and all of T1's harm came from it. Two ways forward: (a) re-run the neighbourhood closure around the leader each day with the probe data and withhold confirmation while a new neighbour fits as well, tested on natural completed-bank misses (rare: 8 arose in 256 meter-stress inputs, so about 100 need about 3,200 new ones); or (b) lock now and report T1's failure with its prevalence-weighted size (about 0.1 point of wrong confirmations against 11 points of correct ones), which departs from the registered decision and needs the user's approval. *Decided 2026-11-23: (b), by the user; (a) stays future work.* This also supersedes item 5 for missing-truth detection: absolute adequacy tests lack the power.
4. **Drop phase-unbalance work** (decided 2026-11-20) and keep the short-line set at realistic values (0.005–0.01 Ω) on all four feeders. Expect short segments near 0.005 Ω to stay unresolved within a day; report them as missing-line confirmations with their voltage deviation.
5. **Add an absolute model-adequacy check** before any confirmation. The posterior only compares candidates, so a test of the leader's residuals against the meter noise floor (a χ² or whiteness test) is needed to catch a truth missing from the bank. Implement and validate the score-test guard; it is the only planned mechanism aimed at this failure.
   With per-meter offsets the leader's ratio had a median of 0.98–1.01 in every second-round condition and never exceeded 1.16, so a threshold near 1.2 would rarely fire falsely. Without offsets it reached 1.19 at 0.05% bias (too close to detect) and 1.42 at 0.1%, so the check catches stress-level bias but does not replace the offsets. Near-equivalent wrong leaders also fit to meter accuracy, so the check cannot catch them.
6. **Redefine b/J** so that it measures model bias only: evaluate it at the true parameters, or normalise it by the predictive spread. As specified, parameter error alone flags 30–40% of blocks.
7. **Keep the records-based floor as a Stage S candidate.** It was the one first-round setting where probing clearly helped.
8. **Validate on real feeder data before any claim.** The four case feeders are synthetic, with long internal segments and a smooth common load shape; a surveyed LV feeder with real AMI data is the next test that matters.
