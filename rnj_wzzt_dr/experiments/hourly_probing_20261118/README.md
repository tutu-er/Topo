# Hourly Market Probing

**Status: draft protocol v3, a design pre-check, and first development code with pre-lock development runs on regenerated synthetic inputs ([DEVELOPMENT_RUNS.md](DEVELOPMENT_RUNS.md), 2026-11-20). Nothing has run on real data. Protocol v4 (2026-11-23) decides every setting and locks the block selector; one item, the Stage M input source (a real surveyed feeder or fresh synthetic seeds), stays open before the lock commit (PROTOCOL §9, §11).**

The development runs found that probing is safe and cheap but adds little under scenario R1. They also found that voltage-meter bias and short lines push the true topology out of the passive bank, after which wrong topologies are confirmed. Read them before settling the open decisions. A second round (2026-11-20/21: single phase only, typical meter bias 0.05%, all four case feeders) found that a revised step 1 keeps the truth in the bank (64 of 64 against 39 of 64). It also found that targeted probing then helps only with a best-effort planner (ambiguous inputs 89% → 100% correct, 8 h → 4 h), because the v3 planner's guarantee leaves it idle; see DEVELOPMENT_RUNS Sections 10–12. A pre-registered head-to-head (Section 13, 2026-11-22) found that targeting beats random excitation, but that a certified planner ([`PLANNER_REVISION.md`](PLANNER_REVISION.md)) does not beat a simple query-by-committee rule. A confirmation on fresh seeds (Sections 14–15, 2026-11-23) found that the refined pipeline (revised step 1, per-meter offsets, GLS scoring for the shared root-meter error) confirms the true topology in 100% of main-case runs and 78% of short-line runs, against 48% and 27% for the first-round pipeline. Probing added a non-significant 5 to 1 there. A hard-case test at realistic demand-response prices (Section 16, 2026-11-23) then found that, on inputs where one day of passive data is not enough, targeted probing raises correct confirmations from 29% to 40% (62 to 2) and beats random blocks (55 to 6), at 17–27 CNY a day. A sweep of the price of certainty V (Section 17) found every V from 100 to 2,000 CNY about equally good (37–39% correct against 29% passive, with all fixes). It also showed that a battery-state defect had inflated Section 16's accuracy by about 3 points. The fresh-seed lock test (Section 18) replicated the benefit (43% against 32% correct, 61 to 2; random blocks beaten 47 to 8) but failed its registered safety test with the truth removed: probing confirmed a near-twin of the missing truth more often than passive (64% against 60%, 37 to 17). The trust gate could not tell those banks apart. On the same fresh seeds V = 2,000 beat V = 100 (38 to 14), so Section 17's flat V curve was underpowered; V = 100,000 added nothing. On the 62 inputs whose truth the bank missed by itself (Section 19), probing changed little (45% against 44% wrong), because the gate's closure check blocked probing on 51 of them; all of the truth-removed test's harm came from completed banks, a case that arose in 2.4% of probe-eligible inputs. **On 2026-11-23 the user locked the block selector** (the strong greedy at V = 2,000 CNY, linear DR prices, GLS scoring; `config.locked_settings()`), departing from the registered rule and disclosing the failed safety test with its estimated size (about 0.1 point of extra wrong confirmations against 11 points of extra correct ones).

This study tests whether an LVDG operator can resolve an ambiguous topology bank by buying hourly **active-power** blocks at chosen nodes from an aggregator of household batteries and EV chargers, then rescoring the candidates after each hour. Two premises shape the design:
- **Only controls that exist in today's LV networks.** Reactive power is not dispatched, because household inverters run local volt-var by default and remote reactive set-points exist only in trials.
- **Topology is the only goal.** R and X are nuisance parameters: refitted every hour, never targeted.

See the [protocol](PROTOCOL.md) (v3) for arms, strata, decision rules and the open settings that must be fixed before the lock commit.

The [pre-check](PRECHECK.md) ran a linear-model simulator ([`precheck/precheck.html`](precheck/precheck.html), opens in any browser) on Soumalas11 and Pengwah18. It found that:
- the v1 scoring and feedback confirm wrong topologies under strong probing;
- a refit + BIC scorer is decisive: it lets the truth lead before probing in 65–100% of runs, against 0–5% for validation MSE;
- realistic P-only probing (κ = 1.5–3, no Q) confirmed the truth in every run, sooner than passive data, for 2–4 CNY a day, with X left uncalibrated (Finding 7);
- scoring the same probes with the frozen passive fit caused wrong confirmations;
- when the truth is missing from the bank, 35–40% of runs still confirm a wrong topology. A broad zero-sum P check lowered this to 5–20%, but at 10–50 times the cost.

The [manuscript draft](MANUSCRIPT.md) writes up the method in paper style. Its resource scenarios are:

| Scenario | Resources | Day κ_A | Night κ_A |
| --- | --- | ---: | ---: |
| R0 | Original library | < 1 | < 1 |
| R1 (main) | 25% of households own batteries and 20% EVs; 25% of owners enrolled with an aggregator | 1.2–1.5 | 2.5–3.3 |
| R1, low adoption | 10% batteries, 5% EVs | 0.5–0.6 | 0.8–1.1 |
| R2 | R1 plus one operator-owned community battery at a mid-feeder node | R1, plus one strong node | |

Each hour's dispatch is one program (manuscript Section IV-C/D):
- it minimises procurement cost, subject to a Bayesian topology condition and voltage, transformer, device and budget limits, over a rolling 2-hour confirmation target;
- the topology condition is a Chernoff bound on the posterior with R/X integrated out. With frozen R/X it reduces to the quadratic constraint of the earlier procurement designs;
- after a variational step it is a convex quadratic constraint.

The response model keeps the branch losses. The linearised model drops them, which leaves a load-dependent sensitivity error of about 1–6% on the head segment that would otherwise accumulate as false evidence. Each refit computes a candidate's loss term from its previous fit and the metered flows and subtracts it, so the refit is still the same QP. Planning uses each candidate's operating-point sensitivity, and the linearisation limit counts the probe's cross term with the forecast error. Every probed block reports the AC bias ratio b/J of [`RESPONSE_BIAS.md`](../flexibility_grid_20261115/RESPONSE_BIAS.md) against each competitor (PROTOCOL §8).

A wrong topology that is the truth plus one extra line needs separate treatment (manuscript Proposition 6):
- **Why probing alone fails here.** Without a floor, that topology reproduces the truth for any probe, so probing can't separate them. Only the BIC penalty rejects it: by noise, it stays within ln 19 of the truth with probability at most 19/N.
- **Why systematic error matters.** That protection fails if a systematic error lines up with the extra branch. An aligned 5–10% of the meter noise is enough, which is why losses are compensated and b/J is reported by rival type.
- **The floor.** The refit keeps a resistance floor `r_C ≥ r_min` on non-singleton supports as a bound, not a deletion; the default is 0.003 Ω. At this floor, probing adds too little to resolve such a rival within the day.
- **Reach test.** The planner tests each rival's reach and doesn't buy against rivals it can't resolve.
- **Stage D.** It adds an unbalanced AC evaluation and a short-line set (a real segment shortened to 1.5·R_min), so that both spurious extra lines and missed short lines are measured. Wrong confirmations are reported as extra line, missing line or wrong placement.

Section IV-D of the manuscript lists every source of nonconvexity and how it is handled:
- modelling removes most of them: the topology bank is enumerated, R/X are integrated out, the power flow uses the loss-compensated branch-flow model and is checked in AC, and device envelopes are continuous;
- the topology condition is the only one that needs an algorithm. Dropping a non-negative log-determinant term and applying the quadratic transform gives a convex inner approximation that touches the condition at the current plan;
- a sequence of such convex programs keeps every iterate feasible and never raises the cost. It reaches a local optimum, not a certified global one;
- the single-rival frozen core of the program has a closed-form global solution (one small eigenproblem per hour). It supplies the starting plan and a lower bound that reports how far from global the plan can be;
- the larger, predictable loss is in the guarantee itself: in the frozen single-rival case it asks for 1.5–2.2 times the information of an exact chance constraint.

The default planner of the main arm is the program restricted to a dictionary of shapes: each shape is held at the smallest level that meets the same topology condition, and the cheapest admissible shape is executed. The optional refinement starts from that plan, so it can only lower the plan's cost. The planner and the program are derived but untested.

The pre-check used an earlier planner that aimed for an *expected* evidence of ln 19 − E_j within two hours. The current condition asks for that evidence with probability 1 − δ, which in the frozen single-rival case means 5–18 times the information. Under R1, many hours may therefore run at the headroom ceiling or hit the caps. The confirmation target and δ are open settings (PROTOCOL §11).

Both feeders are synthetic and short (at most 186 m). The main case keeps impedance scale 1 (decided 2026-11-19). Impedance scale 2, with regenerated candidate banks, and service X/R 0.25 are sensitivities.

A paper-style LaTeX version of the method and formulas is in [`paper/probe_method.tex`](paper/probe_method.tex). It is meant to be compiled with pdfLaTeX, for example on Overleaf (it uses TikZ for the outline figure). Only the LaTeX file has the derivation outline (Section II: flow figure, assumptions A1–A7, and a table that traces every term of the final convex subproblem to its source formula) and the appendix with the proofs specific to this method.

The earlier [method note](METHOD.md) documents the v2 reasoning (κ formulas, the PROBE components and the Q-based calibration). The manuscript and protocol v3 supersede it.

## How it differs from earlier studies

| Study | Decision step | Response | Inputs | What it found |
| --- | --- | --- | --- | --- |
| [`rolling_flexibility_20261115`](../rolling_flexibility_20261115/README.md) | 15 min | Frozen | 3 known inputs | Zero procurement in all 1,152 rounds |
| [`flexibility_grid_20261115`](../flexibility_grid_20261115/README.md) | Day-ahead | Frozen | 36 cells | No overall gain; a Pengwah gain and a Soumalas loss caused by frozen-X bias |
| **This study (v3)** | 1 h active-power block, triggered; targeted, with a no-progress stop | Refit on all data, BIC scoring | Selected for ambiguity, plus two control strata | Pre-check only, see `PRECHECK.md` |

## Code layout

The development code (2026-11-20) reuses the maintained package and does not modify it. It is a compact first version of the plan below:

| File | Contents |
| --- | --- |
| `config.py` | v3 draft settings |
| `scenario.py` | Multi-day load law, meter model (noise, gain bias, quantisation), R1 resources, prices |
| `unbalanced.py` | Three-phase backward–forward sweep (Z0/Z1, seeded phases) |
| `model.py` | Candidate model with the loss term, bounded refit with the resistance floor, Σ_k |
| `scores.py` | BIC, equivalence (19), posterior rule (20), rival types |
| `planner.py` | Reach test, active set, dictionary planner, windows (25)–(28), check and random blocks |
| `run.py` | Seal-then-reveal hourly runner for the five arms, b/J, outcomes |
| `bank.py` | Regenerated inputs: identifier bank plus one-edit neighbours, strata, short-line set |
| `experiments.py`, `report.py`, `noise_check.py`, `identifier_remedies.py` | Development batches, summaries, the Proposition 6 noise check and the meter-bias remedy test |
| `coverage.py` | Step-1 truth coverage on all four case feeders: current bank against the revised step 1 (per-meter offsets, pre-contraction paths, BIC neighbourhood closure) |
| `test_hourly_probing.py` | PROTOCOL §9 unit tests the code covers |

Not yet implemented: the score-test guard, program refinement (step 6), the `resources/r1_allocations.json` file, and the independent audit.

The original plan:

| New file | Reuses |
| --- | --- |
| `config.py` | Frozen settings from PROTOCOL §2, §5–§7 |
| `resources.py` | Builds `resources/r1_allocations.json` (enrolled battery and EV headroom per node) from PROTOCOL §2 |
| `strata.py` | Bank state at validation and truth coverage, for strata only |
| `scores.py` | BIC refit scorer (PROTOCOL §4); frozen-fit scoring for `frozen_replay` |
| `refit.py` | The bounded least squares of `identification.optimization.QPFitter.fit` on fixed supports, warm-started, after subtracting each candidate's branch-flow loss term. `QPFitter` takes only scalar bounds, so the resistance floor on non-singleton supports calls `scipy.optimize.lsq_linear` with per-coordinate bounds. Also computes the derivative `J_k` for planning |
| `planner.py` | Device domain from `rolling_flexibility_20261115.domain`; the P-only targeted dictionary, the reach test and the zero-sum check (NumPy/SciPy only) |
| `feedback.py` | Trigger, confirmation, no-progress rule and score-test guard, in hours |
| `run.py` / `audit.py` / `report.py` | Seal-then-reveal runner, AC evaluation through `physical.backend`, the per-block AC bias ratio b/J, wrong confirmations by type, independent audit |
| `unbalanced.py` | New three-phase backward–forward sweep for the Stage D unbalanced evaluation; line data from `physical/cases`. `physical.backend` is balanced |
| `shortline.py` | Builds the short-line set: shortens one internal segment of the true network and regenerates the bank with the passive identifier |

## Before running

*Updated 2026-11-23 for protocol v4.* The code, the unit tests and Stage D (the development runs) are done; the open settings are decided; the R1 allocations come from `scenario.resources`; Stage 0 is deferred until the frozen banks' `data/` is restored.
1. Choose the Stage M input source (PROTOCOL §9): a real surveyed feeder with smart-meter data (M-real), or fresh synthetic seeds 129–160 (M-syn).
2. Commit the protocol with that choice on its own (the lock commit) on branch `exp/hourly_probing_20261118`. See [GIT_WORKFLOW.md](../../../GIT_WORKFLOW.md).
3. Run Stage M from the lock commit with `config.locked_settings()`, then Stage S.
