# Development protocol

What is run, on which cases, how it is measured, and the rules that keep the results honest. Everything here is
**development**: no claim in this directory is a registered result. (The one registered result, the sealed test of the
predecessor method, is summarised in RESULTS.md, section 1; its test set is spent.)

## 1. Rules

1. **Cases are fixed.** The case generator (`rnj_wzzt_dr/experiments/case_library_20261125/`: `conductors.py`,
   `feeders.py`, `loads.py`, `physics.py`, `build.py`) was set from real-feeder statistics and committed before any
   identifier ran on its cases. Nothing in it changes because of a method's result. This directory does not contain
   the generator; it **reads** the cases, and every development case file is checked against its committed SHA-256
   (`ptid/cases/manifest_dev.json`).
2. **The sealed test set is never opened again.** Its 240 cases were used once, by the registered protocol of the
   predecessor (2026-11-26). No identifier in this directory reads them; `library.open_case` refuses any name not in
   the development manifest.
3. **The inputs built for Stage M of the earlier study** (46 large real networks) are not used without the user's word.
4. **Every method change is followed by a rerun and a comparison** on the same cases (`identify_cases compare`), and
   RESULTS.md records the commit of every run.
5. **Gurobi for every QP, bounded least squares and MIQP**; one thread per case; fixed seeds.
6. **Criteria first.** The pass criteria of a test are written here or in PROPOSALS.md before the test runs; when a
   criterion is added after a run has started, it says so.
7. **Outputs stay out of Git** (`outputs/topology_study/`); the numbers that matter go into RESULTS.md.

## 2. Cases

### 2.1 Development library (288 cases)

| Kind | Feeders | Meters | What it is |
| --- | --- | --- | --- |
| star | 4 named × 6 load draws | 11-18 | the four old cases (`soumalas11`, `paper15`, `flynn16`, `pengwah18`), lines unchanged |
| street | the same 4 × 6 | 11-18 | the same meters, customers and services, rearranged along one street |
| small, medium, large, huge | 6 random feeders each | 10-15, 16-25, 26-40, 41-60 | new street feeders from real-feeder statistics |

- **Twins.** Each feeder comes as four twins with identical loads, truth and meter errors: no unmetered load, or one at
  the head (just after the trunk), the middle or the end. The load is 2, 5 or 10% of the metered power and is street
  lighting, three unmetered homes, or a pump/charger in blocks. Differences between twins come from the load alone.
- **Physics.** AC power flow (balanced single-phase, constant power) at 15 minutes; the identifiers see hourly
  averages; their linear model is never used to make data.
- **Meters.** Voltage gain error 0.05% (fixed) and noise 0.05%; P and Q gain error 0.3% (0.2% at the PCC) and noise
  0.5%. The PCC's V, P and Q are metered.
- **History.** 8 days hourly (192 rows) for identification; day 9 for a simulated probing day.
- **Truth.** The identifiable truth: lines of at least 3 milliohm (shorter sections are merged).
- Full design and the audit of the stored cases: `rnj_wzzt_dr/experiments/case_library_20261125/DESIGN.md`.

### 2.2 One twin per feeder (72 cases)

For a faster pass, one twin of each feeder (`library.one_twin_per_feeder`): within each kind the feeders, sorted by
name, take no load, head, middle and end in turn. Star and street give 6 per location; each random size gives 2, 2, 1, 1.
In all: no load 20, head 20, middle 16, end 16. The other 216 twins can be added later without rerunning these.

### 2.3 Real EU LV feeders (66 development inputs)

33 feeders of the European LV test data set with real topologies (8 to 24 meters), real customer load profiles, the
same physics and meters, 8 days hourly, **start days 0 and 6**. They have **no PCC power**, so no unmetered load is
modelled. The truth is their identifiable truth. (The same 33 feeders on start days 1-5 and 7-11, 330 inputs, served as
a confirmation set in the earlier study; they are not used here.) The 84 real feeders that set λ and the generator
include these 33, so the real set is not independent of the method's settings.

### 2.4 Not used

- The sealed test set (240 cases): spent.
- The Stage M inputs: 17 whole transformer networks (26-77 meters) and 6 large feeders, start days 0 and 6 (46 inputs).
  They are the most realistic test of the solver at scale; using them needs the user's word.

## 3. Settings (fixed for the development runs)

| Setting | Value |
| --- | --- |
| Gate ς₀ | 1% of the metered mean power |
| λ; m; bound on r, x | 1.96; 4.6 (ln 100, half-BIC units); 2 pu |
| K; anchored searches R | 9 units or children; at most 2 |
| Tree MIQP | 120 s up to 35 meters, 240 s beyond; started from the converged local-search tree; coefficients scaled to their columns; Gurobi's default tolerances; per-meter bounds on (a quarter of the limit at most) |
| Flip test | cutoff H(T) + m; local flips first; same limit as the tree MIQP |
| Local search | a first pass of 20 s per start (40 s beyond 35 meters); the best start, and the MIQP's tree when better, to convergence |
| Pricing | enumeration up to K children, node MIQP beyond (60 s); at most 10 additions per round |
| Load evidence | the no-load model over the pool's meter groupings and the load model over the pool, same limits |
| Probing | prices 1.2 / 4 CNY per kWh up / down; caps 20 CNY per hour, 100 per day; band [0.96, 1.035] pu, z = 2.33; trust gate 1.2; σ_v = 0.05%; blocks at one meter in 4 sizes; ψ = placeholder |

## 4. Identification runs (`ptid.runs.identify_cases`)

What is recorded per case: the tree, labels, rounds, stop reason, certificate scope, timing, and the measurements the
method never sees (truth, F1, the truth of each label, test S1's adequacy F, test S2's completeness).

Reported (`summary`), by kind:

| Measure | Meaning |
| --- | --- |
| F1, precision, recall | the tree's groupings against the identifiable truth |
| decided, wrong | groupings of the tree labelled decided; of those, how many are not true |
| open, unsettled | groupings of the tree labelled open or unsettled |
| solver outcomes | anchored MIQPs and flip tests: optimal, cut off, time limit, numerical failure |
| stop | converged (Lemma 3) or round limit |
| S1, S2 | true and found trees flagged by the adequacy F test; pricing complete |
| time | median and largest per case |

`compare OLD NEW` gives, on the common cases: same tree, F1 old → new, wrong decided old → new, median time.

**Criteria for the v15/v16 solver fix** (written 2026-11-27, after the runs had started and before any of their results
were read):
- no solver claim that contradicts a known tree is used (counted as numerical failures, never as proofs);
- decided groupings wrong in at most 2% of decided groupings, over the development cases run;
- on the cases below 26 meters, F1 not lower than the v13 run on the same cases by more than 0.01 on the mean.

## 5. Probing runs (`ptid.runs.probe_cases`)

A simulated day 9 on a development case: Algorithm 1 on the 8-day history, then Algorithm 2 with the case's world
(`ptid/cases/world.py`: the case's true loads and unmetered load, AC power flow, the same meters with new noise;
batteries and EV chargers of scenario R1). Reported per day: F1 before and after, blocks bought, money spent, targets
and their final labels, wrong decided targets, stop reason. Variants T0, T1, T2 of the target rule and the start rule
are tested as PROPOSALS.md describes. ψ is chosen at the knee of the development cost-coverage curve (PLAN.md).

## 6. Toward a registered test

Development results do not carry over to a registered claim. A registered test of this method needs a new sealed set
(the generator with new feeder, customer and load random numbers), a protocol with hypotheses and analysis written
before the set is built, and one run. PLAN.md lists this as the last step.
