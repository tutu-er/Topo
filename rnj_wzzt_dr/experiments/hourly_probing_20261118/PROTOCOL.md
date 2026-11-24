# Hourly Market Probing for Topology Discrimination: Protocol

**Status: v4, 2026-11-23. Every setting is decided (§11). One item stays open before the lock commit: the Stage M input source (§9), which depends on whether a real surveyed feeder is available.** The block selector and its settings were locked on 2026-11-23 by the user's decision, with the failed truth-removed safety test (T1) disclosed. Nothing has run on real data.

v4 replaces v3 (2026-11-19/20; its text is in commit `be39809`) after the pre-lock development runs ([DEVELOPMENT_RUNS.md](DEVELOPMENT_RUNS.md) §1–19, 2026-11-20 to 23). Those runs served as Stage D. The two premises of v3 are unchanged:
- **Only controls that exist in today's LV networks.** Reactive power is not dispatched, because household inverters run local volt-var by default and remote reactive set-points exist only in trials. Active power is bought from an aggregator of household batteries and EV chargers.
- **Topology is the only goal.** R and X are nuisance parameters: refitted every hour, never targeted.

What changed from v3:

| Item | v3 | v4 | Evidence |
| --- | --- | --- | --- |
| Bank (step 1) | Identifier output as frozen | Revised step 1: per-meter offsets, pre-contraction candidates kept, BIC neighbourhood closure, at least one day of history | Truth in bank 64/64 against 39/64 (DEVELOPMENT_RUNS §11) |
| Scoring noise model | Terminals independent | GLS for the root-meter error shared by every terminal, plus per-meter offsets | Correct confirmations 100% / 78% against 48% / 27% (§14–15) |
| Main arm | `t_opt_P`, dictionary planner with a guaranteed condition | `bgreedy_P`, the strong greedy at V = 2,000 CNY | The v3 planner bought nothing on 862 triggered gates (§12); strong greedy 61 to 2 against passive on fresh seeds (§18) |
| Probing condition | Triggered by a close rival | Triggered, and only while the bank is trusted (trust gate) | First-round truth-missing harm 44% → 76% without a gate (§12) |
| Prices and caps | Quadratic contract, 5 / 50 CNY | Linear DR settlement 4 / 1.2 CNY/kWh, caps 20 / 100 CNY | The contract understated cost about ten times (§16) |
| Arms | 5, including broad check and frozen replay | 5: passive, strong greedy, certified planner, committee, random | §13, §16, §18 |
| Decision rules | Thresholds of 0.10·S1 | Exact McNemar tests as in §18, plus reported safety measures | §18 and the 2026-11-23 lock decision |
| Guard | Score test on nested extensions | Dropped | A test of the leader alone cannot detect a missing truth (§18) |
| Inputs | Saved frozen banks | Regenerated banks (the frozen banks' `data/` is not in this checkout) | — |

The method is written up in [MANUSCRIPT.md](MANUSCRIPT.md) (v3 planner) and, for the strong greedy, in [PLANNER_REVISION.md](PLANNER_REVISION.md) §10 and `paper/strategies.tex`; equation numbers below refer to the manuscript.

## 1. Question

A low-voltage distribution grid (LVDG) operator holds a candidate bank built from passive meter data. When the bank cannot separate the leading topologies, the operator buys hourly active-power blocks at chosen nodes from an aggregator. After each hour it refits every candidate, rescores it and decides the next block.

| ID | Comparison | Question |
| --- | --- | --- |
| M1 (primary) | `bgreedy_P` vs `passive_bic` | Does P-only probing raise the confirmed-correct rate on ambiguous inputs? |
| M2 | `bgreedy_P` vs `random_matched_P` | Does targeting matter at the same hours and spend? |
| M3 | `bgreedy_P` vs `cert_P` (and `qbc_P`) | Is the cheap one-step selector as good as the certified planner? |
| M4 | `bgreedy_P` vs `passive_bic`, truth present | Does probing add wrong confirmations or violations? |
| T, N (reported) | `bgreedy_P` vs `passive_bic`, truth missing | How much does probing add to wrong confirmations when the truth is not in the bank, removed by hand (T) or missed by the bank itself (N)? |

## 2. Market, resources and timing

- **Product:** one hourly active-power block per node, constant over the four quarter-hour slots of the hour. It is signed and delivered as `a·u`, with a fixed delivery gain per node for the day, `a ∈ [0.9, 1]`, learned only from telemetry already received. No reactive set-points are bought.
- **Interpretation:** a demand-response contract with an aggregator that dispatches enrolled devices (a VPP), or a local flexibility market. It is not a transmission spot market.
- **Resources, scenario R1 (main):** each node holds `N_i = p̄_i / 0.8 kW` household equivalents.
  - 25% of households own a 5 kW / 13.5 kWh battery; 20% own an EV with a 7 kW charger, plugged in 18:00–08:00, half of whose power is charging that can pause.
  - 25% of device owners are enrolled. Batteries keep 20–80% state of charge (efficiency 0.95).
  - Allocations follow from these rules in `scenario.resources`; no separate allocation file is needed.
- **Reactive power:** household inverters stay in their local volt-var mode (AS/NZS 4777.2:2020 Australia A defaults, zero Q between 220 and 240 V). Their Q is metered and enters the fit.
- **Price (linear DR settlement):** a block costs `c(u) = π↓ Σ_i u_i⁻ + π↑ Σ_i u_i⁺` CNY per hour, with reductions (battery discharge or paused charging) at π↓ = 4 CNY/kWh and increases (extra charging) at π↑ = 1.2 CNY/kWh. Root energy is settled separately and reported.
- **Caps:** 20 CNY per hour and 100 CNY per feeder-day, counting blocks already sealed for the current hour.
- **Gate lead 1 h:** the block for hour *h+1* is sealed at the start of hour *h*, using data up to the end of hour *h−1*. The battery state and the budget used for sizing include the block already sealed for hour *h*.

## 3. Inputs and strata

**Bank construction (revised step 1, DEVELOPMENT_RUNS §11).** Each input has at least one day of passive history (96 training and 48 validation quarter-hours, then the new day). The passive identifier runs with a free voltage offset per meter; its pre-contraction candidates are kept; then a BIC neighbourhood closure (margin 6, neighbour grouping up to 8 children) adds every one-edit neighbour within the margin. The closure's round and size caps are recorded; when one binds, the closure is "capped".

Strata are assigned with the scorer of §4 on the passive history. Truth is used only to assign strata and to evaluate.

| Stratum | Rule | Role |
| --- | --- | --- |
| S1 ambiguous | The true topology is in the bank, and on the passive history the leader is wrong or the posterior rule (§5) does not confirm it | Primary evidence |
| S2 confident | On the passive history the leader is the true topology and the posterior rule confirms it | Cost of false triggers |
| S3 truth missing | The true topology is not in the bank | Safety measure N |
| S1 truth-removed copies | S1 banks with the true topology's candidates removed by hand | Safety measure T |

- Inputs per condition and seeds: §9. Each input gets 2 paired paths covering forecast error, meter error and delivery gain (seed `2026111901`, then input seed and path).
- Stratum assignment uses known truth, so S1 results are conditional on ambiguity; the share of inputs in each stratum is reported per condition.

## 4. Scoring

After every hour, refit every retained candidate *k* on its fixed support, warm-started from its previous fit. The data are the whole passive history and the new day so far.

- **Bounds and resistance floor (manuscript (16)).** Box bounds as `QPFitter.fit` (0 ≤ r_C, x_C ≤ 2). Every non-singleton support carries the floor `r_C ≥ r_min`, `r_min = 2·R_min/Z_base`, with `R_min` = 0.003 Ω (the identifier's recommended floor). The floor is a bound, not a deletion: a branch held at its floor stays in the candidate. Reactance has no floor. Coordinates held at a bound are left out of the covariance.
- **Loss compensation (manuscript (1), (16)).** Before each refit, the candidate's branch-flow loss term is computed from its previous fit and the metered P/Q and root voltage, and subtracted from the voltage drops; one fixed-point step per hour, three at the first gate.
- **Per-meter offsets.** Every refit has a free constant per meter (one demeaned run), and the BIC counts the n offsets.
- **GLS for the shared root error (PLANNER_REVISION §11).** Every terminal's `y_i = V_0² − V_i²` subtracts the same root reading, so each slot's errors share a component. Residuals and designs are whitened by subtracting γ times the cross-terminal mean in each slot, `γ = 1 − 1/√(1+nρ)`, with ρ estimated from the data.

Then

$$
J_k=\frac{\mathrm{SSE}_k}{2\hat\sigma^2}+\tfrac12\,p_k\ln N,\qquad
\hat\sigma^2=\min_k\frac{\mathrm{SSE}_k}{N-p_k},
$$

with whitened SSE, *N* scalar residuals and `p_k` the fitted parameters including offsets. The leader is `argmin J`, `E_j = J_j − J_L`, and the posterior weights are `w_k ∝ e^{−E_k}`. The refitted R and X are reported for the confirmed topology but are not targets.

## 5. Feedback and the block selector

**Equivalence, confirmation, trigger and progress.**
- **Operational equivalence (manuscript (19)):** two candidates are equivalent when their refitted full-day voltage predictions differ by at most ε_V = 5·10⁻⁴ pu at every terminal and slot.
- **Confirmation (manuscript (20)):** the leader's posterior, over non-equivalent rivals, stays at or above 0.95 for two consecutive hourly checks. Probing then stops; a change of leader re-arms it.
- **Trigger:** probing is armed while the confirmation rule fails.
- **Trust gate:** probing is allowed only while (1) the bank's closure finished without a cap binding and (2) the leader's adequacy ratio `ρ_L = SSE_L / ((N − p_L − n) σ_y²)` is at most 1.2. Condition (1) is what stops probing on most banks that miss the truth (DEVELOPMENT_RUNS §19); its blind spot, a completed bank without the truth, is disclosed (§11, 2026-11-23).
- **No-progress rule:** if the closest rival's `E_j` has risen by less than 1 over the last 6 probed hours, buying stops for the day and the input is recorded as "unresolved, possible model mismatch".

**The strong greedy (`bgreedy_P`, the main arm; PLANNER_REVISION §10).** "Greedy" means it decides one hour at a time: at each gate it buys the single block that most lowers the risk of naming the wrong topology for its price, and does not plan later hours.
- **Classes.** The leader's equivalence set is one class; the other candidates with `E < 25` are grouped by the same equivalence test into at most 10 classes, each weighted by its posterior mass `W_c` and represented by its heaviest member.
- **Risk bound (Bhattacharyya).** For a block `u`, with `q_cc'(u)` the whitened separation of the representatives of classes c and c' over the next hour, including each candidate's parameter uncertainty,

  $$U(u)=\sum_{c<c'}\sqrt{W_cW_{c'}}\;e^{-q_{cc'}(u)/8}\;\ge\;\Pr(\text{wrong class after the hour}).$$

- **Rule.** Choose `u* ∈ argmin_{u ∈ 𝒰_g} c(u) + V·U(u)` with **V = 2,000 CNY**: a block is bought only if `V·(U(0) − U(u)) > c(u)`, and among those the one with the largest net value. Otherwise nothing is bought this hour.
- **Admissible blocks 𝒰_g.** Shapes: single-node ±P blocks, and group blocks on the nodes whose upstream paths differ between the leader and its three nearest rivals; each on a 24-point grid of levels up to its ceiling.
- **Code:** `config.LOCKED_ARM` with `config.locked_settings()`; a unit test pins them to the settings confirmed in DEVELOPMENT_RUNS §18.

**Comparison arms (§6).** `cert_P` minimises cost subject to the certified miss bound `B(u)` (PLANNER_REVISION §3–5, planning horizon `H_c = 2 h` and the 20 nearest rivals), and its bound is reported as the certificate. `qbc_P` is the query-by-committee rule (§13). `random_matched_P` replays `cert_P`'s hours and realised hourly spend with random single-node blocks within the state-of-charge-aware headroom.

**Constraints on every block** (all arms):
- device power and energy within the state-of-charge-aware headroom, delivery gain, the hourly and day caps;
- **robust voltage band** [0.96, 1.035] pu, held for the forecast background ± its 2.33σ error (8% relative load-forecast error) under every candidate with posterior weight ≥ 0.01, using each candidate's refitted resistances and their covariance (manuscript (14));
- **root loading:** root apparent power ≤ the transformer rating (125 kVA Soumalas, 200 kVA Pengwah, Flynn and the paper case); same-sign blocks at several nodes are allowed only while this holds;
- **unbalance:** single-phase delivery checked with a phase-voltage sensitivity of up to 6 times the balanced value;
- **linearisation remainder** below the voltage-drop noise (manuscript (27)), including the probe's cross term with the forecast-error band;
- a block that fails a check is scaled down until it passes, or replaced by zero action. Every executed block is checked post hoc against AC, and any AC violation is a reported failure.

## 6. Arms

| Arm | Probing | Role |
| --- | --- | --- |
| `passive_bic` | None | Baseline |
| `bgreedy_P` (main, locked) | Strong greedy, V = 2,000 CNY | M1–M4, T, N |
| `cert_P` | Certified planner | M3; its bound B is the reported certificate |
| `qbc_P` | Query by committee | M3, reported |
| `random_matched_P` | `cert_P`'s hours and spend, random single-node blocks | M2 |

All arms score with refit, GLS and BIC, and probing arms obey the trust gate. On S1 truth-removed copies and S3 inputs only `passive_bic` and `bgreedy_P` run. The v3 arms `t_opt_P`, `check_then_t_opt` and `frozen_replay` stay in the code for reference and are not run.

## 7. Physics and sealing

- The physical day is the input's new day with the true topology fixed; there are no switching events.
- Executed blocks are added to background P. `physical.backend` computes balanced single-phase-equivalent AC voltages, and the declared meter errors are applied: 0.05% random voltage error, 0.5% P/Q error, and a fixed per-meter voltage gain bias with σ = 0.05% (σ = 0.1% and 0.1% random error in the meter-stress condition).
- Household volt-var is modelled in the AC evaluation if any voltage leaves 0.953–1.039 pu; such hours are reported.
- Every round seals its plan and information cutoff before the evaluator releases that hour's data. Partial trajectories are never restarted with a fresh budget.

## 8. Outcomes and decision rules

**Per run (end of day):** correct confirmation, wrong confirmation, undecided, unresolved (no progress); ever-confirmed-wrong (a wrong leader confirmed at any hour), by type relative to the truth (extra line, missing line, wrong placement) and with its voltage error against the truth; hour of first correct confirmation; probed hours, kWh, cost (CNY) and root energy; AC violations and any delivered energy beyond the headroom; per probed block, the AC bias ratio b/J (manuscript; evaluation only); R and X errors of the confirmed topology (reported, not targets); the trust gate's state and the closure status.

All comparisons are paired on condition, input and path (exact McNemar on the discordant pairs, pooled over conditions, two-sided p < 0.05). Every result is also reported per condition.

**Predeclared decisions:**

| Decision | Supported when |
| --- | --- |
| M1, probing helps (primary) | `bgreedy_P` confirms the true topology in significantly more S1 runs than `passive_bic` |
| M2, targeting matters | `bgreedy_P` beats `random_matched_P` significantly in S1 correct confirmations |
| M3, selector adequate | `bgreedy_P` is not significantly worse than `cert_P` in S1 correct confirmations; the comparison with `qbc_P` is reported |
| M4, safe with the truth present | No AC violations, and `bgreedy_P` is not significantly worse than `passive_bic` in ever-confirmed-wrong over S1 ∪ S2 |

**Reported safety measures (not gates; lock obligation of 2026-11-23):**
- **T:** ever-confirmed-wrong of `bgreedy_P` against `passive_bic` on the S1 truth-removed copies, with the McNemar count, split by closure status.
- **N:** the same on natural S3 inputs, split by closure status.
- The share of probe-eligible inputs (S1 or S3 with a finished closure) whose bank misses the truth, and the estimated extra wrong confirmations in practice: T's excess among finished closures times that share.
- Wrong confirmations more than 0.2% off in voltage, and the largest, for both arms in every stratum.

**Also reported:** net value per run at 1,000 CNY per correct identification, `NV = 1000·(C_greedy − C_passive)/n − mean CNY a day`, and the cost per extra correct identification; the certificate B at confirmation.

Null results are reported as null. Thresholds, caps and settings are not tuned after the lock.

## 9. Stages

0. **Stage 0, offline re-scoring** of the frozen banks of `wzzt_long_grid_20261117_v2` and `w_importance_20261117_v1`: deferred until their `data/` is restored to this checkout. It answers the v3 question C0 (BIC refit against the validation-MSE selector) and is not a precondition of Stage M.
1. **Unit tests** (`test_hourly_probing.py`, 22 tests): loss term against AC; three-phase equal split; refit against `QPFitter` and the floor; the extra-line BIC tail; the v3 planner's closed forms and reach bound; zero-sum and random blocks within the state-of-charge-aware headroom; paired replay arms; per-meter offsets; one-edit closure; the certified bound; GLS whitening; monotone submodularity of the risk reduction and the strong greedy; linear DR cost; the committed block against the budget; the locked settings.
2. **Stage D, development:** done as the development runs (DEVELOPMENT_RUNS §1–19). This v4 is its amendment.
3. **Stage M, main batch.** *Input source (open before the lock commit):*
   - **M-real**, if a real surveyed feeder is available: its surveyed topology as the truth, its smart-meter voltage, P and Q at 15 min as the passive history, and probe responses added by the balanced AC model on the surveyed network (no real blocks are bought). Strata and arms as above.
   - **M-syn**, otherwise: fresh seeds never used in development (129–160) on the four case feeders, 128 inputs in each of four conditions: typical (0.05% bias, the feeders' own lines), 0.01 Ω short segment, 0.005 Ω short segment, meter stress (0.1% bias and noise).
   - In both: 2 paths per input; every S1 input with all five arms; every S2 input with `passive_bic` and `bgreedy_P`; truth-removed copies of every S1 input and every natural S3 input with `passive_bic` and `bgreedy_P`.
4. **Stage S, sensitivities** (on S1 and S3 inputs, `passive_bic` and `bgreedy_P`): low adoption (10% batteries, 5% EVs); delivery shortfall (gain `a ∈ [0.6, 0.9]`); two days of passive history instead of one; records-based floor (`R_min` = 0.01 Ω); short segment at 3·R_min; impedance scale 2 with regenerated banks; service X/R 0.25.
5. **Stage C, confirmation** (separate lock): independent physical dates, or a second real feeder.

## 10. Records

- Output: `outputs/hourly_probing_20261118_m/` (Stage M) and `_s/` (Stage S), ignored by Git.
- Store hashes of the source, the protocol commit, the inputs (banks), and the environment.
- Store per-round seals, blocks, device states, refit parameters, scores, costs, AC results and completion markers.
- An independent audit recomputes the scores and costs.
- The commit holds only code, this protocol, the development records, tests and summaries. Run Stage M from the lock commit.

## 11. Settings (decided 2026-11-23 unless dated otherwise)

| Setting | Value | Status and evidence |
| --- | --- | --- |
| Confirmation rule | Posterior ≥ 0.95 over non-equivalent rivals for 2 consecutive hours | Used in every development run (§12–19); the pairwise ln 19 rule is not adopted |
| Equivalence | ε_V = 5·10⁻⁴ pu on full-day predictions | Used throughout |
| Trigger | The confirmation rule fails | DEVELOPMENT_RUNS deviation table: with 45–122 candidates, many rivals at E of 3–4 block confirmation while none is below ln 19 |
| No-progress rule | ΔE < 1 over 6 probed hours | Used throughout; not binding on any reported result |
| Gate lead | 1 h, with the sealed block counted against budget and battery state | Used throughout; committed-block fix 2026-11-23 (§17) |
| Hourly / day cap and prices | **Locked:** linear DR settlement, reductions 4 CNY/kWh, increases 1.2 CNY/kWh, caps 20 / 100 CNY | The contract understated cost about ten times (§16); ambiguous inputs cost 23 CNY a day at V = 2,000 (§18) |
| Resistance floor `R_min` | 0.003 Ω; the records-based 0.01 Ω is a Stage S sensitivity | Used throughout |
| Bank construction (step 1) | Revised step 1 (decided 2026-11-22) | Truth in bank 64/64 against 39/64 (§11–12) |
| Scoring noise model | **Locked:** GLS for the shared root-meter error, with per-meter offsets | §14–15 |
| Trust gate | Closure finished and ρ_L ≤ 1.2 (decided 2026-11-22); blind spot disclosed | §18–19 |
| Block selector | **Locked (user decision):** the strong greedy at V = 2,000 CNY, `config.locked_settings()` and `config.LOCKED_ARM` | §15–19; see Decided, 2026-11-23 |
| Comparison planner | Certified planner `cert_P` (H_c = 2 h, 20 rivals) with its bound B reported | §13, §18 |
| Paths per input | 2 | §16–19 |
| Broad check | Not adopted (not part of the locked selector) | In the first round, at ρ = 1 for 3 h, it cut wrong confirmations in S3 from 35–40% to 5–20% at 32–151 CNY a day, before the trust gate; noted as a candidate remedy for the trust gate's blind spot, future work |
| Guard (score test) | Dropped | A test of the leader alone lacks the power to detect a missing truth (§18 prediction check) |
| Miss probability δ, program refinement, reach test | Not applicable | Parts of the v3 dictionary planner, which is no longer run |
| R2 community battery | Dropped from Stage S | Size and node were never set; a battery at the transformer gives no topology information |
| **Stage M input source** | ***Open:*** M-real or M-syn (§9) | Depends on whether a real surveyed feeder with smart-meter data is available |

**Decided:**
- **2026-11-19, impedance scale.** The main batch keeps the repository's impedance scale 1: feeders of at most 186 m, with a background voltage spread of about 2%. Impedance scale 2 is run with candidate banks regenerated by the passive identifier, and service X/R 0.25 is a further sensitivity. Neither changes the main-batch decision rules in §8.
- **2026-11-19, controls.** Probing uses active power only, and R/X calibration is not a goal.
- **2026-11-19, model bias.** Scoring and planning use the loss-compensated branch-flow model, and the linearisation remainder includes the probe's cross term with the forecast error. Every probed block reports the AC bias ratio b/J. A multiplicative sensitivity-uncertainty term is kept in reserve for the Stage D amendment.
- **2026-11-19, planner condition.** The dictionary planner uses the program's guaranteed condition `q_j ≥ η_j` in place of the pre-check's expected-evidence quota, so the planner and the optional refinement pursue the same requirement. The pre-check therefore does not test the main arm's planner. *(Superseded 2026-11-23: the dictionary planner is no longer the main arm.)*
- **2026-11-20, extra-line rivals.**
  - The refit holds a resistance floor `r_C ≥ r_min` on non-singleton supports, as a bound rather than a deletion.
  - The planner drops rivals that probing cannot reach before the end of the day, and buys nothing if those alone block the requirement.
  - Wrong confirmations and b/J are reported by type: extra line, missing line, wrong placement.
  - Stage D adds a short-line set and an unbalanced AC evaluation, with an amendment trigger on extra-line confirmations.
- **2026-11-20, single phase.** Fitting, planning and evaluation all use the balanced single-phase-equivalent model. The unbalanced AC evaluation, its amendment trigger and the unbalanced-evaluator setting are dropped; `unbalanced.py` stays in the code as an optional diagnostic only.
- **2026-11-20, meter bias.** Every voltage meter carries a fixed gain bias with σ = 0.05%, the typical value, in the main batch. σ = 0.1% is a stress test.
- **2026-11-22, step 1.** The bank is built by the revised step 1 (DEVELOPMENT_RUNS §11): per-meter voltage offsets in the identifier input and in every hourly refit (BIC counts the n offsets), the identifier's pre-contraction candidates kept, and a BIC neighbourhood closure (margin 6, neighbour grouping up to 8 children) whose caps must not bind. Every input has at least one day of passive history.
- **2026-11-22, trust gate.** Probing is allowed only while the bank is trusted: the closure finished without a cap binding, and the leader's adequacy ratio (its residual variance over the meter-class noise variance) is within the meter class. *2026-11-23: insufficient.* In the fresh-seed test (DEVELOPMENT_RUNS §18) the gate let probing run on 55% of truth-removed days against 58% of natural ones, and the adequacy ratio had the same distribution with and without the truth. A one-hour-ahead prediction check did not separate the cases either (ROC area 0.47–0.58). A stronger missing-truth check is required before the lock; since the leader cannot be tested alone, it must compare the leader with rebuilt alternatives. The closure part of the gate does work: it blocked probing on 51 of 62 inputs whose truth the bank missed naturally (DEVELOPMENT_RUNS §19). The blind spot is a completed bank without the truth (2.4% of probe-eligible inputs), which is where all of T1's harm arose. *2026-11-23, user decision: the gate is kept as it is and its blind spot is disclosed, not fixed (next item).*
- **2026-11-23, block selector lock (user decision; a departure from the registered §18 decision).**
  - *Locked:* the strong greedy (each hour it buys the one block that most lowers the risk of a wrong answer per CNY, without planning further ahead) at V = 2,000 CNY, with class weights and the committed-block fix; linear DR settlement (4 / 1.2 CNY/kWh, caps 20 / 100 CNY a day); GLS scoring; the revised step 1 and trust gate already decided above. Code: `config.locked_settings()`, `config.LOCKED_ARM`; a unit test pins them to the settings confirmed in DEVELOPMENT_RUNS §18.
  - *Evidence for:* fresh seeds 65–96 (§18), 43% against 32% correct (61 to 2), random blocks beaten 50 to 8 (fixed random arm), certified planner matched, largest net value (88 CNY per run). V = 2,000 beat V = 100 (38 to 14) and V = 100,000 added nothing (§17).
  - *Evidence against, disclosed:* T1 failed. With the truth removed by hand, probing confirmed a wrong topology in 64% of runs against 60% passive (37 to 17, p = 0.009). All of that harm came from completed banks, which the gate cannot flag; on the 62 inputs whose truth the bank missed by itself, probing changed little (45% against 44%, 4 to 2), because the closure flag blocked probing on 51 of them (§19). A completed bank without the truth arose in 2.4% of probe-eligible inputs (0.8% on short lines, 7.8% under meter stress); weighted by that rate the extra wrong confirmations are about 0.1 point (0.4 under meter stress) against 11 points of extra correct ones. Every extra wrong topology was a near-twin of the truth, at most 0.27% off in voltage.
  - *Why the registered remedy was not done first:* a check of the leader alone cannot detect a missing truth (DEVELOPMENT_RUNS §18, prediction check), and the remaining remedy (rebuilding the bank around the leader each day) can only be judged on natural completed-bank misses, of which about 100 would need about 3,200 new meter-stress inputs.
  - *Obligations:* (1) the paper reports T1's failure, the registered decision it overrode, the blind spot and the estimate above; (2) Stage M keeps the truth-removed copies and the natural truth-missing inputs as reported safety measures, split by closure status, not as gates; (3) wrong confirmations stay reported by type with their voltage error; (4) the daily re-closure check is future work, outside the lock.
- **2026-11-23, protocol v4.** The remaining v3 settings are decided as in the table above: the values used in every development run are kept (confirmation rule, equivalence, trigger, no-progress rule, gate lead, `R_min`, 2 paths); the v3 planner's own settings (δ, refinement, reach test), the guard, the broad check, the frozen-replay control and the R2 battery are dropped; Stage 0 is deferred; Stage S gains low adoption, delivery shortfall and a two-day history. The Stage M input source is the one open item.
