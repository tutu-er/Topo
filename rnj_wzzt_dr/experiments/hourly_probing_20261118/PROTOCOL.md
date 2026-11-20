# Hourly Market Probing for Topology Discrimination: Protocol

**Status: DRAFT v3, 2026-11-19, revised 2026-11-20. Not locked.** Nothing has been implemented in Python or run on real data. Settings marked *open* need a decision before the lock commit.

v3 replaces v2 after two changes of premise:
- **Only controls that exist in today's LV networks.** Reactive power is not dispatched, because household inverters run local volt-var by default and remote reactive set-points exist only in trials. Active power is bought from an aggregator of household batteries and EV chargers.
- **Topology is the only goal.** R and X are nuisance parameters: refitted every hour, never targeted. The pre-check (PRECHECK Finding 7) found that targeted P-only probes at κ = 1.5–3 confirmed the truth in every run without any R/X calibration, while scoring with the frozen passive fit caused wrong confirmations.

What changed from v2:

| Item | v2 | v3 |
| --- | --- | --- |
| Products | P and Q blocks | P blocks and zero-sum P swaps only |
| Calibration phase | 3 h of Q-only PRBS, then targeted | Removed |
| Resources | Repository library, trust region κσ | Scenario R1: enrolled household batteries and EVs (§2) |
| Stop rule | Every rival ≥ ln 19 behind for 2 h | Leader posterior ≥ 0.95 for 2 h (*open*: pairwise) |
| Mismatch handling | Variance-ratio alarm, floor-KKT test | No-progress rule, score-test guard, optional broad check arm |
| Negative control | v1 planner and rule | Frozen-fit scoring on the main arm's exact probes |
| Calibration decision (C3) | Calibration pays | Replaced by "the broad check pays" |

The method is written up in [MANUSCRIPT.md](MANUSCRIPT.md); equation numbers below refer to it.

## 1. Question

A low-voltage distribution grid (LVDG) operator holds a candidate bank from `fit`. When the bank cannot separate the leading topologies, the operator buys hourly active-power blocks at chosen nodes from an aggregator. After each hour it refits every candidate, rescores it and decides the next block.

| ID | Comparison | Question |
| --- | --- | --- |
| C0 (Stage 0, offline) | BIC refit scorer vs the current validation-MSE selector, on existing frozen banks | Does a complexity-penalised refit pick the true topology more often before any probing? |
| C1 (primary) | `t_opt_P` vs `passive_bic` | Does P-only probing raise the confirmed-correct rate, or shorten the time to confirmation, without adding wrong confirmations? |
| C2 | `t_opt_P` vs `random_matched_P` | Does the design matter at the same hours and cost? |
| C3 | `check_then_t_opt` vs `t_opt_P` | Does a broad zero-sum check lower wrong confirmations when the truth is missing, and at what cost? |
| C4 (negative control) | `frozen_replay` vs `t_opt_P` | Does scoring with the frozen passive fit cause wrong confirmations on the same data? |

## 2. Market, resources and timing

- **Product:** one hourly active-power block per node, constant over the four quarter-hour slots of the hour, plus zero-sum swaps between node sets. It is signed and delivered as `a·u`, with a fixed delivery gain per node for the day, `a ∈ [0.9, 1]`, learned only from telemetry already received. No reactive set-points are bought.
- **Interpretation:** a contract with an aggregator that dispatches enrolled devices one by one (a VPP), or a local flexibility market. It is not a transmission spot market.
- **Resources, scenario R1 (main):** each node holds `N_i = p̄_i / 0.8 kW` household equivalents.
  - 25% of households own a 5 kW / 13.5 kWh battery.
  - 20% own an EV with a 7 kW charger, plugged in 18:00–08:00.
  - 25% of device owners are enrolled.
  - This gives an enrolled headroom of κ_A ≈ 1.2–1.5 by day and 2.5–3.3 at night (upward, with EVs). Batteries keep 20–80% state of charge.
  - The per-node allocation file `resources/r1_allocations.json` must be built from these rules before the lock.
- **Reactive power:** household inverters stay in their local volt-var mode (AS/NZS 4777.2:2020 Australia A defaults, zero Q between 220 and 240 V). Their Q is metered and enters the fit.
- **Sensitivities:**
  - R0, the repository library;
  - R1 with 10% batteries and 5% EVs (κ ≈ 0.5–1.1);
  - R2, R1 plus one operator-owned community battery at a mid-feeder node (size and location *open*);
  - impedance scale 2 with regenerated banks;
  - service X/R 0.25.
- **Price:** the activation prices in `resources/library.json`, using the prepaid quadratic contract with the enrolled headroom as `A_i`. Root energy is settled separately.
- **Caps (*open*):** 5 CNY per hour and 50 CNY per feeder-day.
- **Gate lead (*open*, default 1 h):** the block for hour *h+1* is sealed at the start of hour *h*, using data up to the end of hour *h−1*. The pre-check used 0 h, so Stage D must report the effect of the 1 h lead.

## 3. Inputs and strata

The source banks are the existing native frozen banks and their training/validation splits:

- `history288_pq_noise_20261115`: 192/96 points, new day = day 4;
- `flexibility_grid_20261115`: 192/96, 96/48 and 48/24 points.

Truth is used only to assign strata and to evaluate.

Strata are assigned with the v3 scorer (§4) on training plus validation data:

| Stratum | Rule | Role |
| --- | --- | --- |
| S1 ambiguous | The true internal topology is in the bank, and the BIC runner-up is < ln 19 behind or the BIC leader is wrong | Primary evidence |
| S2 confident | The BIC leader is correct and every rival is ≥ ln 19 behind | Cost of false triggers |
| S3 truth missing | The true internal topology is not in the bank | Wrong-confirmation, no-progress and guard control |

- Take up to 8 inputs per stratum, ascending by input ID, with at most 4 per network. If fewer qualify, use all and report the count.
- Each input gets 8 paired paths covering forecast error, meter error and delivery gain. Seed: `2026111901`.
- Stratum assignment uses known truth, so the results do not estimate population rates.

## 4. Scoring

After every hour, refit every retained candidate *k* on its fixed support, warm-started from its previous fit (initially the passive identifier's fit). The data are all of training, validation and the new day so far.

**Bounds and resistance floor (manuscript (16)).** The box bounds are those of `QPFitter.fit` (0 ≤ r_C, x_C ≤ 2). In addition, every non-singleton support (internal branches and the common upstream segment) carries the floor `r_C ≥ r_min`, with `r_min = 2·R_min/Z_base`:
- `R_min` is the resistance of the shortest internal segment that the operator's records allow. The default is 0.003 Ω, the identifier's recommended floor (`impedance_floor_20261116`), which gives `r_min = 0.0075` at `Z_base = 0.8 Ω` (*open*, §11).
- The floor is a bound inside the refit, not a deletion. A branch held at its floor stays in the candidate, so a candidate with one extra line remains the separate hypothesis "this line exists and is at least `R_min`". Output contraction is not applied during the day, because deleting the branch would silently turn the candidate into another one.
- Reactance has no floor: P-only probes inform only the resistance block.
- At the first gate, passive fits below the floor are projected onto it. Coordinates held at a bound are left out of the covariance (manuscript (18)).

Manuscript Proposition 6 explains why the floor matters. Against a rival that is the leader plus one extra line, evidence comes from the complexity penalty and from the floor, not from probing.

**Loss compensation (manuscript (1), (16)).** Before each refit, the candidate's branch-flow loss term is computed from its previous fit, the metered P/Q and the metered root voltage, and subtracted from the voltage drops. The refit itself is unchanged: the same QP on the same parameters. One fixed-point step is used per hour, and three at the first gate. Only metered quantities and the candidate's own parameters enter. Then

$$
J_k=\frac{\mathrm{SSE}_k}{2\hat\sigma^2}+\tfrac12\,p_k\ln N,\qquad
\hat\sigma^2=\min_k\frac{\mathrm{SSE}_k}{N-p_k},
$$

where *N* is the number of scalar residuals and `p_k` is twice the number of supports. The leader is `argmin J`. The evidence against rival *j* is `E_j = J_j − J_L`.

The refitted R and X are reported for the confirmed topology but are not targets.

## 5. Feedback and planner

**Trigger, stop and progress:**
- **Trigger:** probing is armed when a rival that is not operationally equivalent to the leader has `E_j < ln 19`. Equivalent means refitted voltage predictions within the meter resolution over the forecast day (manuscript (19)).
- **Confirmation:** the leader's posterior, over non-equivalent rivals, must stay at or above 0.95 for two consecutive hourly checks (manuscript (20)). Probing then stops; a change of leader re-arms it. *Open:* the pairwise rule `min_j E_j ≥ ln 19`, which the pre-check used.
- **Planning horizon:** each gate plans the next `T_r = min(H_c, hours left in the day)` hours, with confirmation target `H_c = 2 h` (*open*), and executes only the first block.
- **Reach test (manuscript Section V-C):** before the active set is formed, each non-equivalent rival is tested for whether the dictionary can still resolve it today.
  - Its reach `Q̄_j` sums, over the remaining hours of the day, the frozen-form `q_j` (Σ = 0) of the best dictionary block in each hour. The block is taken at that hour's own ceiling (headroom, voltage, transformer, unbalance, linearisation; energy and caps dropped) or at zero, whichever gives more.
  - This bounds `q_j` for every dictionary plan of the day, even one that changes shape every hour.
  - If `Q̄_j < η_j` at the most lenient share `δ_j = δ`, rival *j* is **beyond probing reach**. It leaves the candidates for the active set, and its term `c_j` is added to `R_tail`. If these terms alone exceed `δ/2`, the gate buys nothing.
  - Beyond-reach rivals stay in the stopping rule, so their evidence can come only from the penalty and passive data. They are logged every hour.
  - The typical case is a rival that is the leader plus one extra branch held at its floor (manuscript Proposition 6).
- **Active competitors:** the non-equivalent rivals within reach, in increasing `E_j`, until the rest satisfy `R_tail = Σ √19·e^(−E_j/2) ≤ δ/2` (manuscript (10), (12)). The miss probability is `δ = 0.1` (*open*); what remains after `R_tail` is split equally, so each active rival gets `δ_j` between 0.017 and 0.1 when one to three are active. This sets its required level `η_j = 8γ_j`. The number of active rivals is logged.
- **No-progress rule:** if the closest active rival's `E_j` has risen by less than 1 over the last 6 probed hours, buying stops for the day. The input is recorded as "unresolved, possible model mismatch". *Open:* window and threshold.

**Targeted P design (all probing arms except `random_matched_P`):**
The planner uses the same topology condition as the full program: each active competitor needs `q_j ≥ η_j` over the planned hours (manuscript (15a)).
1. Build a dictionary of device-feasible shapes: single-node ±P blocks, plus, for each active competitor, a group ±P block on the nodes whose upstream path differs from the leader's.
2. For each shape, compute its window `κ_max` (manuscript (25)–(28)): headroom, energy, robust voltage, transformer, linearisation and the caps, over all planned hours.
3. Hold the shape at one level in every planned hour, and find the smallest level `κ_need` at which `q_j ≥ η_j` for every active competitor (manuscript (22)). Use a grid on `[0, κ_max]` and refine the first crossing by bisection. A shape with `κ_need > κ_max`, or with no crossing, is not admissible.
4. Pick the admissible shape with the lowest cost `κ_need² c(s̄)` (manuscript (23)), and execute its first block.
5. Execute zero action if the forecast background alone meets the condition, or if the reach test leaves no active competitor. If no shape is admissible, execute the shape with the largest `min_j q_j/η_j` at its ceiling `κ_max`. All active competitors are within reach, so this fallback buys only for rivals that the rest of the day can still resolve.
6. *Optional, open:* refine the plan with the receding-horizon program (manuscript (9)). Its only nonconvex part, the topology condition, is replaced by the convex inner approximation (15c), and the program is solved as a sequence of convex programs (manuscript Section IV-D):
   - starts: the closed-form core plan (15e) of each active competitor, the dictionary plan from step 4, and the previous plan shifted by one hour. The core and shifted plans are first scaled along their rays to the smallest level meeting the condition, and dropped if that level breaks a limit; the dictionary plan is already at that level. The cheapest result is kept;
   - stop at a relative cost change below 1e-4 or after 20 iterations; if no start survives, penalised slacks give the most-progress plan;
   - each refined block records its gap to the closed-form lower bound (15f);
   - the refinement is used only if it passes every check below; otherwise the first block of the dictionary plan is executed.

This needs only NumPy/SciPy, plus an open-source second-order cone solver if step 6 is adopted. The lower bound needs only an n × n eigenproblem per hour and competitor. Gurobi is not required; where it is available, its NonConvex=2 bound on the frozen program may be used as a cross-check on small cases.

The dictionary plan and the refined plan meet the same requirement, so whenever the dictionary plan is admissible the refined plan costs no more over the planned hours. Its first block may still cost more. The requirement is stronger than the pre-check's planner, which aimed for an expected evidence of `ln 19 − E_j` within two hours rather than a guaranteed one. In the frozen single-rival case it asks for 5–18 times that information, so levels are 2.2–4.3 times higher, and under R1 many hours may run at the ceiling or hit the caps. Of this, the conservatism of the bound accounts for a factor 1.5–2.2 over an exact chance constraint (manuscript Section IV-D). Stage D must report how often the window is empty and the cost per confirmation.

**Broad check (arm `check_then_t_opt` only):** the first `H_chk` triggered hours buy a zero-sum block. Signs are random, buys and sells balance, and each node runs at `ρ` times its headroom. The defaults are `H_chk = 2` and `ρ = 0.5`, about a quarter of the pre-check's cost per hour (*open*). Confirmation waits until these hours are done.

**Constraints on every block:**
- device power, energy and delivery-gain limits; the hourly and day caps;
- **robust voltage band:** [0.96, 1.035] pu. The lower edge is 0.95 pu plus a 0.01 pu margin; the upper edge keeps household volt-var inside its deadband. The band must hold for the forecast background ± its 2σ error, under every candidate with BIC weight ≥ 0.01, using each candidate's refitted resistances and their covariance (manuscript (14));
- **root loading:** root apparent power ≤ the transformer rating (125 kVA Soumalas, 200 kVA Pengwah); same-sign blocks at more than one node are not allowed;
- **unbalance:** single-phase blocks are checked on their own phase with a phase-voltage sensitivity of up to 6 times the balanced value, until a three-phase model is available;
- **linearisation remainder:** at every terminal and sample, the part of the loss term that planning leaves out must stay below the voltage-drop noise σ_e (manuscript (27)). This part includes the cross term of the probe with the 2σ forecast-error band of each branch flow, not only the probe's own square. Planning uses the operating-point sensitivity `R_k + J_k` of each candidate;
- a block that fails any check is scaled down until it passes, or replaced by zero action. Every executed block is checked post hoc against AC, and any AC violation is a reported failure.

**Guard (*open*, must be validated in Stage D).** For each nested local extension of the leader (a new support, or a split of an existing one), compute the Rao score statistic (manuscript (21)). If it exceeds χ²₂,₀.₉₉ = 9.21 for 2 consecutive hours, record "re-identification needed" and pause probing. The variance-ratio alarm of v2 is dropped, because it never fired in the pre-check.

## 6. Arms

| Arm | Probing | Scorer |
| --- | --- | --- |
| `passive_bic` | None | Refit + BIC |
| `t_opt_P` (main) | Targeted P design | Refit + BIC |
| `frozen_replay` | The exact blocks executed by `t_opt_P` on the same path | Frozen passive fit, no refit; loss term evaluated at the frozen parameters, so only the freezing differs (negative control) |
| `check_then_t_opt` | Broad check, then targeted P design | Refit + BIC |
| `random_matched_P` | Same hours and realised hourly cost as `t_opt_P`; random single-node P blocks within the headroom | Refit + BIC |

## 7. Physics and sealing

- The physical day is the source new day with the true topology fixed; there are no switching events.
- Executed blocks are added to background P. `physical.backend` computes balanced AC voltages, and the declared meter errors are applied. Stage D also runs the three-phase evaluator `unbalanced.py` (§9).
- Household volt-var is modelled in the AC evaluation if any background or probed voltage leaves 0.953–1.039 pu. Any such hours are reported.
- Every round seals its plan and information cutoff before the evaluator releases that hour's data. Partial trajectories are never restarted with a fresh budget.
- Planning is capped at 60 s per round. Solver and AC times are reported separately.

## 8. Outcomes and decision rules

**End-of-day outcomes:**

- exact internal and full-support recovery of the leader;
- correct confirmation, wrong confirmation, undecided, unresolved (no progress), unresolved (beyond probing reach);
- **ever-confirmed-wrong** (a wrong leader confirmed at any hour), in total and **by type relative to the truth**:
  - *extra line:* the wrong leader's supports contain the truth's, plus at least one more;
  - *missing line:* its supports are contained in the truth's;
  - *wrong placement:* neither, for example a moved or swapped branch;
- for each confirmation, the rivals that were beyond probing reach at the last gate, whose evidence came only from the penalty and passive data;
- hours to first correct confirmation;
- probed hours, cost, root energy cost;
- guard flags in S3;
- R and X relative errors of the confirmed topology (reported, not targets);
- AC voltage violations;
- per block: the planned `q_j` against the realised change in `E_j`, the number of active competitors, the rivals beyond probing reach with their reach `Q̄_j`, and whether the window was empty (best shape run at its ceiling);
- **AC bias ratio per probed block** (evaluation only; the truth enters only here). From `physical.backend`, take the noise-free AC drops for the executed hour on the true network, with and without the block. Let `e` be the leader's prediction error on these drops, using the gate's fit and the realised injections, and `d_j` its predicted difference from active competitor *j* over the same hour. Report `b/J = 2⟨e, d_j⟩/‖d_j‖²` (equal to `2‖e‖/‖d_j‖·cos θ`, as in `flexibility_grid_20261115/RESPONSE_BIAS.md`). Report it for the whole response and for the probe part alone (the AC difference with and without the block, minus the model's probe response), each with and without the loss term. At `b/J = −1` the bias cancels the evidence; `|b/J| > 0.5` flags the block. It is reported by rival type relative to the truth (extra line, missing line, wrong placement), with extra-line rivals shown separately, because an extra line is the rival that a systematic error can make win (manuscript Proposition 6(b));
- if refinement is on: iterations, the winning start, the gap to the lower bound (15f) per refined block, and the plan-cost difference against the dictionary plan (never positive when the dictionary plan is admissible).

All comparisons are paired on input and path. Gains, losses and McNemar counts are reported per input and pooled. Every wrong-confirmation rate, including those in the decision rules below, is also reported by type. The decision rules use the totals.

**Predeclared decisions** (S1 = number of S1 paths, S3 = number of S3 paths):

| Decision | Supported when |
| --- | --- |
| BIC scorer adopted (C0) | It picks the true internal topology on ≥ 10% more Stage 0 inputs than validation MSE, with no more spurious-branch picks |
| Probing helps (C1) | `t_opt_P` gains ≥ 0.10·S1 in confirmed-correct, **or** cuts median hours-to-confirmation by ≥ 2 h; **and** its ever-confirmed-wrong rate in S2 ∪ S3 is no higher than `passive_bic`'s |
| Design matters (C2) | `t_opt_P` beats `random_matched_P` by ≥ 0.10·S1 in confirmed-correct, or by ≥ 2 h in median time |
| Broad check pays (C3) | `check_then_t_opt` lowers ever-confirmed-wrong in S3 by ≥ 0.15·S3 relative to `t_opt_P`, with no loss in S1 confirmed-correct; the extra CNY per avoided wrong confirmation is reported |
| Frozen scoring fails (C4) | `frozen_replay` has ever-confirmed-wrong ≥ that of `t_opt_P` + 0.10·S1 in S1 ∪ S2 |

Null results are reported as null. Thresholds, caps and settings are not tuned after the lock.

## 9. Stages

0. **Stage 0, offline re-scoring:** apply the v3 scorer to the frozen banks of `wzzt_long_grid_20261117_v2` and `w_importance_20261117_v1` using their saved training and validation data. The loss term is evaluated at the saved coefficients, so there are no new fits and no Gurobi. Stage 0 reports the scores with and without the loss term, and assigns the strata with it.
1. **Unit tests:**
   - an hourly block is constant over its four slots, and no reactive set-point is ever issued;
   - with `r_min = 0`, the exact bounded LS matches `QPFitter` within 1e-9 in relative SSE; with `r_min > 0`, every non-singleton `r_C ≥ r_min`, a branch held at its floor stays in the candidate, and its coordinate is left out of `Σ_k`;
   - on each case feeder, at the true parameters and topology, the loss-compensated model cuts the gap to the `physical.backend` AC drops by at least 90% relative to the linear model, and `J_k` matches a finite difference of the loss term;
   - the remainder `ρ_j` (manuscript (27)) matches the AC second-order remainder of a probe within 20% on a single-node test block;
   - with every node's households split equally over the phases, the three-phase evaluator `unbalanced.py` reproduces `physical.backend` within 1e-6 pu;
   - on a linear case with one extra-line rival and white noise, the share of 1,000 seeds with `E_W < ln 19` stays within `19/N` plus its Monte Carlo error. With an injected error aligned with the extra branch at twice `β*`, W leads by ln 19 in most seeds (manuscript (33)–(34));
   - a targeted block on an analytic three-terminal pair lands on the node whose upstream path differs;
   - the dictionary plan meets `q_j ≥ η_j` for every active competitor at `κ_need`, and fails it one bisection step below; on a frozen one-competitor case, `q_j` matches the closed form `‖d_j + D_j u‖²/σ̂²`;
   - the active set leaves `R_tail ≤ δ/2`;
   - the reach `Q̄_j` is never below the frozen `q_j` of any dictionary plan for the rest of the day (brute force on a small case). On a constructed case where the closest rival is the leader plus one extra branch held at its floor, that rival is found beyond reach and nothing is bought;
   - with no new data, the scorer reproduces the bank's in-sample SSE;
   - headroom, energy, voltage band and caps hold;
   - a zero-sum check block sums to zero within its headroom;
   - `frozen_replay` executes exactly the blocks of `t_opt_P`;
   - the no-progress rule stops buying on a constructed truth-missing case;
   - each plan is sealed before the reveal;
   - matched arms reproduce `t_opt_P`'s hourly cost;
   - if refinement is on: the minorant (15b) equals `q_j` at `w*` and matches its gradient by finite differences; every iterate satisfies the exact condition (manuscript (11)); the cost never increases; with Σ = 0, (15c) is the tangent half-space of (13); the core plan (15e) matches a brute-force search on a small one-rival frozen case with equal up and down headroom; the lower bound (15f) never exceeds the returned cost; the returned plan never costs more than an admissible dictionary plan.
2. **Stage D, development wiring:** 4 S1 inputs and 2 S3 inputs × 2 paths × 5 arms, with AC. This stage checks plumbing, gate lead, the no-progress rule, the guard and the reach test. It also reports the AC bias ratio (§8) for every probed block. Two variants test the extra-line risk (manuscript Proposition 6):
   - **Unbalanced AC evaluation.** Every path is run a second time on an unbalanced physical day:
     - each household equivalent sits on one phase, drawn with the seed, and each block is delivered on the phases of the devices that supply it;
     - each terminal's voltage is read on the phase of a seeded representative household, and its P and Q are summed over phases;
     - the fit and the planner stay balanced.

     `physical.backend` is balanced, so this needs a three-phase evaluator (`unbalanced.py`): a backward–forward sweep on the true network, with self and mutual impedances built from the positive-sequence line library, `Z0/Z1 = 3` (*open*) and the neutral earthed at the transformer. Report b/J and wrong confirmations by type, balanced against unbalanced, with extra-line rivals separately.
   - **Short-line set.** For each Stage D input:
     - shorten one internal segment of the true network, chosen with the input seed among those with at least two metered terminals downstream, so that its resistance is `1.5·R_min`;
     - regenerate the candidate bank with the passive identifier, and run the five arms on the balanced evaluation;
     - report whether the regenerated bank still contains the true topology, and the confirmations by type.

     This measures both errors the floor trades: spurious extra lines, and real short lines that are missed.

   **Amendment triggers.**
   - (i) More than 10% of probed blocks are flagged with the loss term in place. The amendment then adds, for example, a multiplicative sensitivity-uncertainty term to the predictive covariance.
   - (ii) In the unbalanced runs, more than 10% of blocks are flagged against extra-line rivals, or an extra-line confirmation occurs that the paired balanced run does not have. The amendment must then address unbalance before Stage M, for example with a per-phase fit where phase connections are known.

   Stage D is not evidence. One protocol amendment is allowed afterwards, and it must address every trigger that fired.
3. **Stage M, main batch:** up to 24 inputs × 8 paths × 5 arms = 960 trajectories under R1.
4. **Stage S, sensitivities:** on S1 and S3 inputs only:
   - the R0, low-adoption R1 and R2 scenarios;
   - the short-line set at `3·R_min`;
   - the records-based floor (§11).
5. **Stage C, confirmation** (separate lock): independent physical dates.

## 10. Records

- Output: `outputs/hourly_probing_20261118_v*`, ignored by Git.
- Store hashes of source, protocol, banks, resource allocation and environment.
- Store per-round seals, blocks, device states, refit parameters, scores, costs, AC results and completion markers.
- An independent audit recomputes the scores and costs.
- The commit holds only code, this protocol, `PRECHECK.md`, the pre-check simulator, tests and summaries.

## 11. Open decisions before lock

| Setting | v3 draft | Alternative | Pre-check note |
| --- | --- | --- | --- |
| Confirmation rule | Posterior ≥ 0.95 for 2 h | Every rival ≥ ln 19 for 2 h | The pre-check used the pairwise rule |
| Broad check | `ρ` = 0.5, `H_chk` = 2 | `ρ` = 1, `H_chk` = 3 | At ρ = 1, 3 h: wrong confirmations in S3 fell from 35–40% to 5–20%, at 32–151 CNY a day |
| No-progress rule | ΔE < 1 over 6 probed hours | ΔE < 2 over 8 hours | Truth-missing runs kept buying for about 20 h |
| Confirmation target `H_c` | 2 h | 4 h | The pre-check's quota also aimed at 2 h |
| Miss probability δ | 0.1 | 0.2 | Untested; it sets the level through `η_j` |
| Program refinement | Off (dictionary only) | On: convex sequence from the core plans, the dictionary plan and the shifted plan; gap to the closed-form bound reported; used only if it passes all checks; never dearer than an admissible dictionary plan | Untested |
| Hourly / day cap | 5 / 50 CNY | No cap (headroom only) | Targeted P cost 2–4 CNY a day uncapped |
| Gate lead | 1 h | 0 h | The pre-check used 0 h |
| Guard | Score test on nested extensions | None | Untested |
| Resistance floor `R_min` | 0.003 Ω, the identifier's recommended floor | Records-based: half the shortest backbone segment, about 0.01 Ω on these feeders (run in Stage S) | The pre-check's dominant wrong rivals were splits at the floor. By manuscript Proposition 6, the floor sets what probing can add against them: at 0.003 Ω, extra-line rivals are mostly beyond reach; at 0.01 Ω, 20 kW blocks can resolve them in 2–4 h |
| Unbalanced evaluator | `Z0/Z1 = 3`; terminal voltage on a representative household's phase | `Z0/Z1 = 1`; household-weighted mean of phase voltages | Needs a new three-phase evaluator |
| R2 battery | Size and node *open* | — | A battery at the transformer gives no topology information |

**Decided:**
- **2026-11-19, impedance scale.** The main batch keeps the repository's impedance scale 1: feeders of at most 186 m, with a background voltage spread of about 2%. Impedance scale 2 is run with candidate banks regenerated by the passive identifier, and service X/R 0.25 is a further sensitivity. Neither changes the main-batch decision rules in §8.
- **2026-11-19, controls.** Probing uses active power only, and R/X calibration is not a goal.
- **2026-11-19, model bias.** Scoring and planning use the loss-compensated branch-flow model, and the linearisation remainder includes the probe's cross term with the forecast error. Every probed block reports the AC bias ratio b/J. A multiplicative sensitivity-uncertainty term is kept in reserve for the Stage D amendment.
- **2026-11-19, planner condition.** The dictionary planner uses the program's guaranteed condition `q_j ≥ η_j` in place of the pre-check's expected-evidence quota, so the planner and the optional refinement pursue the same requirement. The pre-check therefore does not test the main arm's planner.
- **2026-11-20, extra-line rivals.**
  - The refit holds a resistance floor `r_C ≥ r_min` on non-singleton supports, as a bound rather than a deletion.
  - The planner drops rivals that probing cannot reach before the end of the day, and buys nothing if those alone block the requirement.
  - Wrong confirmations and b/J are reported by type: extra line, missing line, wrong placement.
  - Stage D adds a short-line set and an unbalanced AC evaluation, with an amendment trigger on extra-line confirmations.
