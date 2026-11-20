# Pre-check of the Hourly Probing Design

2026-11-18. This is a design check, not study evidence. It tested the draft [protocol](PROTOCOL.md) (v1) in a simplified simulator and led to the revised v2.

## What was simulated

The simulator is [`precheck/precheck.html`](precheck/precheck.html). It is self-contained JavaScript and runs in a browser, because this PC has no Python or Gurobi.

| Item | Pre-check setting | Real study |
| --- | --- | --- |
| Networks | Soumalas11 and Pengwah18 tables from `physical/cases/` | Same, plus Paper15 and Flynn16 |
| Physics | The repo's linear response `Y = RP + XQ`, with `r_C = 2 r_Ω / Z_base`, measured through V and root meters | AC through `physical.backend` |
| Loads | The `generate_synthetic` load law; 2 days of training, 1 day of validation, 1 new day | FeederBW profiles |
| Noise | P/Q 0.5% multiplicative; V and root 0.05%; delivery gain U[0.9, 1] | As in the source cells |
| Candidate bank | Truth plus every laminar local perturbation of it (drop, move a leaf out or in, swap leaves, split off a pair). That gives 40 candidates (Soumalas) or 58 (Pengwah) | Native RNJ/MIQP banks |
| Branch floor | Internal resistance ≥ 0.08 × the median star-model diagonal, standing in for the output contraction | `output_floor.py` |
| Fitting | Exact bounded least squares (active set) | `QPFitter` |
| Prices | 0.8 CNY/kWh for P and 0.4 CNY/kvarh for Q, with the quadratic contract and headroom = κσ | `resources/library.json` |
| Timing | Gate lead 0 h, 24 hourly blocks, 20 seeds per configuration | As locked |

To reproduce, set `RUN` at the top of the file, open the file in a browser, and read the JSON once the page shows DONE. The options are:
- `DIAG`;
- `A`–`D` for the v1 design;
- `E` for v2;
- `F` for the realistic P-only design, which is the current default.

On this PC, the full v1 set (`ALL`) took 614 s, `E` took 298 s and `F` took 415 s.

The first run used coordinate descent. It left SSE errors of up to 0.55%, which are as large as the evidence gaps between candidates, so that run was discarded. All results below use the exact solver.

## Finding 1: correcting R/X needs broadband probing that includes Q

Relative error of the true support's R/X after one day of refitting (Soumalas11, 48 training / 24 validation points, 20 seeds):

| Probe design (24 h) | κ | X error | R error | Cost (CNY/day) |
| --- | ---: | ---: | ---: | ---: |
| Frozen, training only | — | 1.07 | 0.30 | 0 |
| One passive day | 0 | 0.79 | 0.21 | 0 |
| PRBS, P and Q at every node | 1 / 3 / 10 | 0.32 / 0.14 / 0.045 | 0.09 / 0.04 / 0.014 | 329 / 988 / 3,294 |
| PRBS, P only | 3 / 10 / 30 | 0.28 / 0.22 / 0.20 | 0.05 / 0.016 / 0.006 | 836 / 2,788 / 8,363 |
| PRBS, Q only | 3 / 10 | 0.16 / 0.05 | 0.085 / 0.079 | 152 / 506 |
| One node per hour, P and Q | 10 / 30 | 0.21 / 0.16 | 0.06 / 0.04 | 295 / 884 |

- **Your point is confirmed:** passive data barely corrects R/X. With 192 training points, one passive day leaves X error at 0.65.
- **X needs Q excitation.** P-only probes stall at an X error of about 0.2 however large they get.
- Pengwah18 shows the same pattern: PRBS P+Q at κ = 3 brings X error to 0.10 for 1,804 CNY/day.

## Finding 2: the draft scoring and feedback confirm wrong topologies under strong probing

With the draft rules, the hourly residual is predicted *before* the probe direction has been learned, from validation-MSE or prequential scores without a complexity penalty. With large probes, that residual mostly measures parameter error. Wrong structures then win by chance and get confirmed. This is the same mechanism as the frozen-X failure in `flexibility_grid_20261115/RESPONSE_BIAS.md`.

Soumalas11, 48/24 points, κ = 10, hourly cap 5 CNY:

| Arm (v1 design) | Correct at end of day | Confirmed wrong | Cost (CNY) |
| --- | ---: | ---: | ---: |
| Passive, frozen | 0% | 0% | 0 |
| Probing, frozen R/X (planner P1, rule F1) | 0% | 65% | 5 |
| Probing, prequential refit (P1, F1) | 0% | 55% | 14 |
| Refit-aware planner, calibrated F2 rule | 10% | 50% | 56 |
| Random probes at matched cost | 5% | 30% | 39 |

At κ = 3 the wrong confirmations fall to 5–15%, but correct identification only reaches 15–35%. Pengwah18 at κ = 10 is worse: up to 85% wrong confirmations. The draft's "confirmed correct" rate was 0% in almost every configuration.

## Finding 3: a complexity penalty matters more than probing

The dominant rivals are *splits*: the true structure plus one spurious branch at the floor. Ranking by MSE alone cannot reject them, because the extra branch fits noise. The table shows how often the truth leads before the new day:

| Rule on training + validation data | Soumalas 48/24 | Soumalas 192/96 | Pengwah 48/24 |
| --- | ---: | ---: | ---: |
| Validation MSE (current repo selector) | 5% | 0% | 0% |
| Refit, in-sample SSE, no penalty | 0% | 5% | 0% |
| Refit, in-sample SSE + BIC penalty | **65%** | **100%** | **100%** |
| BIC, already confirmed (log-odds ≥ ln 19) | 35% | 100% | 45% |

This also bears on the main identifier: validation MSE with internal count as a tie-breaker only may favour spurious extra branches. It can be checked offline on the existing frozen banks (protocol Stage 0).

## Finding 4: the revised design works in this simulator

Revised rule F3: refit every candidate on all data, score with BIC, take the noise scale from the best in-sample residual variance, and confirm when every rival is ≥ ln 19 behind for two consecutive hours. Results (20 seeds each):

| Network, κ, hourly cap | Arm | Confirmed correct | Hours to confirm | Cost (CNY) | X error |
| --- | --- | ---: | ---: | ---: | ---: |
| Soumalas, 10, none | Passive | 85% | 7.8 | 0 | 0.77 |
| | T-optimal probes only | 100% | 3.0 | 1.5 | 0.76 |
| | Two-phase (3 h PRBS, then T-optimal) | 100% | 4.0 | 412 | 0.20 |
| | PRBS only | 100% | 4.0 | 549 | 0.14 |
| | Random, matched hours and cost | 95% | 3.9 | 18 | 0.57 |
| Soumalas, 10, 20 CNY | Two-phase | 100% | 4.0 | 60 | 0.34 |
| Soumalas, 3, none | T-optimal / two-phase | 100% / 100% | 4.3 / 4.0 | 2.2 / 124 | 0.78 / 0.39 |
| Pengwah, 10, none | Passive / T-optimal / two-phase | 95% / 100% / 100% | 4.4 / 2.9 / 4.0 | 0 / 1.4 / 752 | 0.58 / 0.57 / 0.16 |

- **Topology decision:** refit-aware (T-optimal) probes confirm faster and more often than passive data, for about 1.5–2 CNY a day. The matched random probe is slightly worse.
- **R/X accuracy:** a separate objective, handled by the calibration phase. It costs 60–750 CNY a day in this price model.
- **Planner solver:** the planner no longer needs a nonconvex QCQP. It scores a finite dictionary of device-feasible probes with exact bounded least-squares refits, using only NumPy/SciPy.

## Finding 5: the mismatch alarm is still not good enough

With the true topology removed from the bank (Soumalas, κ = 10), the share of runs that ever confirmed a wrong topology was:

| Passive | T-optimal | PRBS | Two-phase | Random |
| ---: | ---: | ---: | ---: | ---: |
| 35% | 35% | 25% | **5%** | 15% |

The in-sample variance-ratio alarm (> 1.3 for 2 h) never fired. The wrong leaders were near-supersets of the truth, so they fit almost as well. A better alarm still has to be designed and tested; one candidate is flagging when an internal branch is pinned at its floor with a significant KKT multiplier.

## Finding 6: probe shape decides voltage and transformer safety

κ scales each node's probe to its own historical variation: `|u_i| ≤ κ·σ_i`. In these networks, σ_P is 0.7–2.7 kW and σ_Q is 0.3–0.8 kvar per node, so κ = 10 means up to 7–27 kW per node. That is more than the node's nominal load of 4.6–11.8 kW, and far above the roughly 2 kW per feeder enrolled in `resources/library.json`.

Worst case over 10 seeds and the whole day, using the linear model and a root voltage near 1.02 pu:

| Network | Probe, 24 h | V min | V max | Max root S (kVA) | X error | Cost (CNY/day) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Soumalas | None | 1.005 | 1.022 | 75 | 0.79 | 0 |
| | All nodes export, κ = 10 | 1.021 | **1.0505** | 137 | — | — |
| | All nodes consume, κ = 10 | 0.970 | 1.012 | **238** | — | — |
| | PRBS P+Q, κ = 10 | 0.982 | 1.038 | 157 | 0.05 | 3,374 |
| | **Q only, κ = 10** | **0.998** | **1.024** | **93** | **0.06** | 509 |
| | **Zero-sum P+Q pairs, κ = 10** | 0.984 | 1.033 | **77** | 0.06 | 2,131 |
| Pengwah | None | 1.000 | 1.020 | 137 | 0.59 | 0 |
| | All nodes export, κ = 10 | 1.026 | **1.0505** | 236 | — | — |
| | All nodes consume, κ = 10 | **0.961** | 1.004 | **424** | — | — |
| | PRBS P+Q, κ = 10 | 0.961 | 1.045 | **271** | 0.04 | 6,066 |
| | **Q only, κ = 10** | **0.988** | **1.024** | 172 | **0.04** | 967 |
| | **Zero-sum P+Q pairs, κ = 10** | 0.966 | 1.040 | **143** | 0.045 | 4,166 |

- Same-direction probes at κ = 10 break the 1.05 pu limit when exporting. When consuming, they push root power to 2–3 times the base load, which overloads the transformer if its rating is near the 200 kVA per-unit base. Random-sign PRBS stays within the voltage band, but it still reaches 157–271 kVA.
- **Q-only probes** (smart-inverter reactive power) recover X as well as full PRBS, barely move the voltage, and cost the least. They do not improve R much: R error is 0.06–0.075.
- **Zero-sum P pairs** (buy at some nodes and sell the same amount at others in the same hour) keep the root power at base level and give the best R. They need more voltage margin than Q-only probes.
- A single-node probe at κ = 10 stays within 0.987–1.034 pu.

The linear model omits losses, and real LV feeders are unbalanced: a single-phase block moves its own phase voltage more than this balanced model shows. The AC check in Stage D must include this.

## Finding 7: realistic P-only probing identifies the topology without any R/X calibration

Run `F`, 2026-11-19. This run tests the revised plan, which differs from the earlier runs in three ways:
- **Only P is dispatched.** Household inverters keep their local volt-var control, so there is no Q probing at all.
- **Levels are realistic.** κ = 1.5 and 3 match the per-node headroom of household batteries and EVs enrolled through a VPP.
- **Topology is the only goal.**

Other settings: hourly blocks with |u_i| ≤ κσ_i, no hourly cap, 20 seeds, 48/24 points and a gate lead of 0 h.

The arms are:
- **Passive:** refit and BIC scoring, as in Finding 4.
- **Passive, frozen fit:** no probes, scored with the passive training fit.
- **Targeted P:** refit-aware probes from a P-only dictionary, scored with refits.
- **Targeted P, frozen scoring:** replays the exact probes of the targeted arm, but scores every candidate with its frozen training fit (the "former R/X"). The two arms see identical data and differ only in scoring.
- **Broad check, then targeted:** 3 h of zero-sum random-sign P at every node, then targeted P.
- **Random, matched cost:** random single-node P blocks at the targeted arm's hourly cost.

**Truth in the bank.** The passive arms do not depend on κ, so they are shown once per network. Mean hours count only the runs that confirmed correctly.

| Network, κ | Arm | Confirmed correct | Ever confirmed wrong | Mean hours to confirm | Confirmed by hour 6 | Cost (CNY/day) | X error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Soumalas | Passive | 85% | 0% | 7.8 | 45% | 0 | 0.77 |
| | Passive, frozen fit | 85% | 10% | 6.2 | 50% | 0 | 1.07 |
| Soumalas, 1.5 | Targeted P | 100% | 0% | 4.8 | 70% | 4.3 | 0.77 |
| | Targeted P, frozen scoring | 100% | 15% | 5.5 | 70% | 4.3 | 1.07 |
| | Broad check, then targeted | 100% | 0% | 5.0 | 80% | 32 | 0.68 |
| | Random, matched cost | 95% | 0% | 7.4 | 50% | 3.5 | 0.78 |
| Soumalas, 3 | Targeted P | 100% | 0% | 4.1 | 80% | 3.0 | 0.76 |
| | Targeted P, frozen scoring | 100% | 15% | 3.9 | 80% | 3.0 | 1.07 |
| | Broad check, then targeted | 100% | 0% | 4.1 | 100% | 61 | 0.59 |
| | Random, matched cost | 90% | 0% | 7.2 | 45% | 2.6 | 0.76 |
| Pengwah | Passive | 95% | 0% | 4.4 | 75% | 0 | 0.58 |
| | Passive, frozen fit | 95% | 0% | 4.8 | 70% | 0 | 0.82 |
| Pengwah, 1.5 | Targeted P | 100% | 0% | 3.3 | 95% | 2.2 | 0.58 |
| | Targeted P, frozen scoring | 100% | 0% | 3.5 | 95% | 2.2 | 0.82 |
| | Broad check, then targeted | 100% | 0% | 4.5 | 95% | 64 | 0.47 |
| | Random, matched cost | 95% | 0% | 4.1 | 80% | 1.8 | 0.58 |
| Pengwah, 3 | Targeted P | 100% | 0% | 2.9 | 95% | 2.2 | 0.57 |
| | Targeted P, frozen scoring | 100% | 0% | 3.9 | 90% | 2.2 | 0.82 |
| | Broad check, then targeted | 100% | 0% | 4.2 | 100% | 126 | 0.39 |
| | Random, matched cost | 95% | 0% | 3.6 | 80% | 1.7 | 0.58 |

**Truth missing** (Soumalas). Share of runs that ever confirmed a wrong topology:

| κ | Passive | Passive, frozen fit | Targeted P | Targeted P, frozen scoring | Broad check, then targeted | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 35% | 40% | 35% | 60% | 20% | 35% |
| 3 | 35% | 40% | 40% | 55% | 5% | 40% |

In this stratum the targeted arm kept probing for about 20 h and spent 39–70 CNY without being able to confirm. The broad arm spent 71–151 CNY.

What this shows:
- **P alone is enough to identify the topology at realistic levels, with no Q and no R/X calibration.** Targeted P confirmed the truth in 100% of runs, sooner than passive data: 4.8 h against 7.8 h on Soumalas at κ = 1.5, and 3.3 h against 4.4 h on Pengwah. X error stayed at 0.57–0.77 throughout.
- **κ = 1.5 is almost as good as κ = 3:** 4.8 h against 4.1 h on Soumalas, and 3.3 h against 2.9 h on Pengwah.
- **Scoring with the frozen former fit is unsafe.** With the same probes, it confirmed a wrong topology in 15% of Soumalas runs (and in 10% even without probes). When the truth was missing, it raised wrong confirmations from 35–40% to 55–60%. This is milder than at κ = 10 (65%, Finding 2) but points the same way. With 20 seeds, 3 of 20 against 0 of 20 is not significant on its own.
- **The design matters on Soumalas.** Random blocks at the same cost confirmed 90–95% of runs, in 7.2–7.4 h. On Pengwah the gap is small.
- **Only the broad check reduced wrong confirmations when the truth was missing:** to 20% at κ = 1.5 and 5% at κ = 3. It costs 10–50 times as much as targeted probing, though, and it delays the earliest confirmation until its 3 broad hours are over. A lower broad level or fewer hours might cut the cost; this is untested.
- **When the truth is missing, targeted probing keeps buying without progress,** so a no-progress stop rule is needed.
- **The variance-ratio alarm again never fired.**

The run also has limits:
- The truth-in-bank cells were not selected for ambiguity. Passive data already confirm 85–95% of them, so most of the gain shown here is speed.
- Headroom is a uniform κσ_i box, not the node-specific envelope of the resource scenarios.

## Limits

- Linear physics only: no AC remainder, no device state of charge, no voltage limits.
- Synthetic loads, and a constructed bank of local perturbations. Native banks may contain non-nested errors that BIC does not resolve.
- Prices and headroom are assumptions.
- Gate lead was 0 h.
- 20 seeds per configuration on two networks.

These results justify the v2 changes. They do not estimate field performance.
