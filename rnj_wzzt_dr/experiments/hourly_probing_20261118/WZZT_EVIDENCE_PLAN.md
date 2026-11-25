# WZZT: does the MIQP earn its place, and how to show the method beats others (experiment plan, 2026-11-24)

**Questions.**
1. Does WZZT's MIQP actually do useful work in each stage, or could a set of heuristics replace it? If it can be replaced, what is novel about WZZT?
2. Which cases and comparisons would show that WZZT, or RNJ+WZZT, beats other methods?

Phase 1 (experiments A, B, F) has run; results and decisions are in [wzzt_evidence_20261124/RESULTS.md](../wzzt_evidence_20261124/RESULTS.md). In short: E's MIQP is replaceable as the solver and moves to the certificate role (D-E); W's MIQP stays (D-W); and Gurobi's default "optimal" stamps are sometimes false, so certificates need the robust formulation and a confirmation solve. The rest is still a plan, building on [WZZT_REVISION.md](WZZT_REVISION.md) (findings 1–7, proposals S1–S2, R0–R4).

Phase 2 has run: the revised WZZT (WZZT-R, `rnj_wzzt_dr/identification/revised.py`) passed all five freeze rules of D2 on the 130 development inputs and is frozen ([RESULTS_PHASE2.md](../wzzt_evidence_20261124/RESULTS_PHASE2.md)). Real supported recall 90% against 53% (`O_dm`) at equal precision; synthetic exact 32/64 against 19; truth in the study's bank 84 against 75. The ablations show the score transform (meter offsets and GLS) and the whole history carry most of the gain; the generators matter on real feeders; R3 is refuted; the BIC price trades synthetic exactness for real recall. Phase 3, the confirmation on unseen inputs, is registered ([PROTOCOL_PHASE3.md](../wzzt_evidence_20261124/PROTOCOL_PHASE3.md)) and running. It covers WZZT-R against the original only; the literature baselines and the random-feeder benchmark (§5) are not yet built.

## 1. What the existing evidence says

| Stage | What the MIQP does | Evidence so far | Verdict |
| --- | --- | --- | --- |
| RNJ guidance | None (distance tree) | The pool holds 48–50% of true segments on real feeders and 86–88% on synthetic ones (finding 7) | Weak on real feeders |
| W, pool trials | None (fixed-support QPs, greedy) | W ablation, 160 synthetic inputs: full W 132 exact, pool trials only 129, no W 112 (`w_importance_20261117`) | Carry most of W's value |
| W, unknown-support MIQP | Finds the single best support to add over *all* compatible terminal sets | Net +3 of 160 (+7/−4) over pool trials only, at about 4× the run time (median paired ratio 0.235). In the old 240-input study, 13 of the 120 correct outputs contained a true support outside the RNJ pool that only this MIQP could have proposed. Without RNJ guidance, 1,033 of 1,034 calls improved on their start; after strong RNJ starts, 225 of 257 returned the start unchanged; on 8 easy long-history inputs with a 10 s step cap, 58 of 58 hit the limit with no gain (`identification_contribution_20261116`) | Large without guidance, small with good guidance |
| E, finite-pool MIQP | Re-selects the whole set of branches jointly (it can drop earlier choices) | Old 240-input study, short history: R+W+E 120/240 internal exact against R+W 45, RNJ15 55, RNJ1 46. In 58 inputs no RNJ15 candidate was right but R+W+E was | The main contributor on hard inputs |
| T | Joint rebuild of a small neighbourhood | Never isolated | Unknown |
| All, easy inputs | — | Long history, 8 inputs: RNJ1 plus contraction already exact 8/8; W and E added nothing | Not needed |
| All, real feeders | — | Gurobi stamped every MIQP "optimal", but Phase 1 found some of those stamps false (2 of 255 E solves; [RESULTS.md](../wzzt_evidence_20261124/RESULTS.md) §1). The search reaches the optimum on 128–131 of 131 real instances; the limit is the criterion and the resolution floor (findings 2–4, 7) | Search is not the bottleneck; certificates need the robust formulation |

**What has never been tested.** Every comparison so far changed the pipeline (removed W, removed E, swapped the start). None solved the *same* combinatorial problem with an exact method and with good heuristics. So we know that joint re-selection (E) matters on hard inputs, but not whether *exactness* matters. The study "R+E without W" on current code is also still open.

## 2. The problem WZZT solves, stated precisely

Data: T time slots of metered P and Q and squared-voltage drops Y at n customer meters. Hidden internal nodes are not observed. Find a laminar family F of supports (any two are nested or disjoint; singletons and the whole feeder always included) and nonnegative parameters minimising

$$
\big\|W\big(Y-\textstyle\sum_{C\in F} z_C z_C^\top (r_C P_t + x_C Q_t) - s_t\mathbf 1 - o - \lambda_F\big)\big\|^2/\hat\sigma^2 \;+\; \ln N\cdot k(F).
$$

- **z_C**: the meters downstream of branch C. **r_C, x_C ≥ 0** are nuisance parameters.
- **k(F)**: the number of *identifiable* parameters (one per support under a fixed power factor, two otherwise). The price per identifiable parameter is sample-size-aware, as in R1.
- **s_t, the extra source term**: I read this as the unknown source-bus (transformer LV side) squared voltage in each slot. Please correct me if you meant another term.
  - It is common to all meters, so it can be removed exactly by subtracting the across-meter mean of each slot's residual (a projection I − 11ᵀ/n). The MIQP keeps its form and size, and the QR compression still applies.
  - It makes identification independent of a root meter. It also absorbs root-meter noise or bias and MV-side voltage swings, which is why it can replace root whitening.
  - Cost: the whole-feeder support and the source impedance become unidentifiable (both are common-mode). Topology does not need them; every internal support stays identifiable.
- **o**: meter offsets, one per meter.
- **λ_F**: the second-order loss term, evaluated at the previous fit as in the bank refit. It has the same support structure, so it is exact for the true family.
- **W**: whitening, if a measured root is kept.

**How hard it is.** This is best-subset selection under a laminar constraint, which is NP-hard in general.
- E solves it exactly over a finite pool of supports.
- W solves exactly the one-support extension over every compatible terminal set (z is a binary vector, and the z_i z_j products are linearised).

**What any method can and cannot carry.**
- Any regression-based method can carry s_t, o and λ, and RNJ can too: with a symmetric fit, R is identified up to the whole-feeder support, and RNJ's path distances do not depend on that.
- Distance methods applied to raw voltage correlations cannot carry them.
- What only a structural search provides is the *joint* choice of structure and parameters under the laminar constraint and the identifiability-aware price, plus a *certificate*. So the extra terms strengthen the formulation, but they are not by themselves an argument for the MIQP.

## 3. Three roles for the MIQP, and what novelty rests on

1. **Solver**: finds the optimum (the current role).
2. **Certificate**:
   - E's bound proves that a family is optimal over the pool.
   - W's bound proves that *no* compatible support can be added with a gain: a "no-gain certificate". That is exactly the statement a resolution certificate (R4) needs: the data supports no further branch anywhere, not just none in the pool.
3. **Escalation**: the MIQP runs only when a heuristic's answer cannot be certified.

**Proposed design: heuristic first, then certify, and escalate only when needed.** A heuristic does the search. The MIQP bound checks the answer: if the gap is closed, stop; if not, the MIQP finishes the job. This keeps exactness where it pays and speed elsewhere.

**Novelty then rests on four things, none of which is "we use an MIQP":**
- (a) The formulation: identifiability-aware laminar regression for feeders with hidden nodes, with source, offset and loss terms and a sample-size-consistent price.
- (b) The move set: RNJ guidance, unknown-support extension (W), local rebuild (T) and revocable joint re-selection (E).
- (c) Exact certificates: optimality over the pool, the no-gain stop, and the resolution report. Phase 1: these hold only with the SVD-compressed formulation, `NumericFocus=2`, and a second, cold-start solve that confirms the first (RESULTS.md §1).
- (d) Cost that does not grow with history length (S1, S2).

**If the MIQP were dropped entirely,** (c) disappears, and the claim rests on (a), (b), (d) and the benchmark. That is a weaker position: it is close to "a good heuristic for a well-posed regression", and I do not recommend it. If experiment A shows heuristics always reach the optimum, the MIQP stays as the certifier, and no baseline offers that.

## 4. Experiments for question 1: does the MIQP matter, and where?

Inputs for development: the 66 real pilot inputs and 66 synthetic inputs (`lock_short01`, `lock_noisy`). The old 240-input benchmark is not on this computer.

**A. Same problem, different solvers (E stage).**
- **Design.** Freeze each input's E-stage instance from a `111` run: data, pool, constraints, start point and objective. Solve the identical instance with:
  - (1) the MIQP (certified gap);
  - (2) greedy forward addition;
  - (3) forward–backward stepwise with add, drop and swap moves;
  - (4) beam search, widths 5 and 20;
  - (5) multi-start local search or simulated annealing, at equal time;
  - (6) a convex relaxation: a nonnegative group-lasso path over the pool supports, laminar repair, then a refit;
  - (7) a one-edit neighbourhood search like the BIC closure.
- **Metrics.**
  - The share of instances where each heuristic reaches the certified optimum, and its objective gap otherwise.
  - When a heuristic misses the optimum: whether the optimum is the truth, or closer to it (recall, precision).
  - Time.
  - Which instance features predict failure: pool size, number of crossing pairs, n, signal-to-noise.
- **Decision rule (fix before running).** If the best heuristic reaches the certified optimum in at least 99% of instances and never loses a correct recovery, the MIQP moves to the certificate role in E. Otherwise it stays as the solver on the instance class where heuristics fail.

**B. W: exact unknown-support search against heuristic support generators.**
- **Generators:**
  - (1) the RNJ pool only;
  - (2) set algebra on the pool: unions of siblings and differences of nested pairs;
  - (3) clustering of meters by how their residuals correlate with each meter's downstream power;
  - (4) RNJ re-run on the residual sensitivities;
  - (5) every compatible support up to size 4.
- **Truth-removed pools.** Rerun each input with the true supports deleted from the RNJ pool. This isolates the capability that only W's MIQP has: proposing a support nobody suggested.
- **Metrics.**
  - Whether the MIQP's best extension is in each generator's set, and the gain lost when it is not.
  - Correct supports found outside the pool (on the real feeders too).
  - How often the no-gain certificate is reached, and its cost when run once at the end instead of at every step.

**C. Component ladder, end to end.**
- **Methods**, all on the same inputs, split and output processing:
  - NJ, RNJ1, RNJ15 (validation-chosen);
  - RNJ+E (no W), RNJ+W (pool trials only)+E, RNJ+W+E (`101`), `111`, W alone (`000`);
  - the heuristic replacements from A and B;
  - the heuristic-first, certify-and-escalate version.
- **Metrics**:
  - exact recovery of the identifiable topology;
  - segment F1 and normalised Robinson–Foulds distance (a standard tree-comparison measure);
  - recall of data-supported segments against the ceiling;
  - false segments per feeder;
  - voltage error on held-out days;
  - time.

**D. Where exactness pays: a hardness map.**
- **Factors** (fractional factorial):
  - history length (2, 4, 8, 18 days);
  - meter noise (0.05%, 0.1%, 0.2%);
  - short-segment length (0.005, 0.01, 0.03 Ω);
  - meters per feeder (8–60; needs the generator in section 6);
  - power-factor diversity (fixed or varying);
  - load correlation (shared or diverse daily profiles);
  - PV back-feed.
- **Output**: the gain of exact over best heuristic, and of RNJ+WZZT over RNJ15, per cell.
- **Hypothesis to state in advance**: gains appear in the middle signal-to-noise range and on feeders with many interacting short branches. Easy cases tie, and hopeless cases tie.

**E. The source term and other extra terms.**
- **Variants:**
  - current (measured root, whitening);
  - source term s_t;
  - s_t with offsets;
  - each with and without the loss term λ.
- **Scenarios**: root exact; root noisy (0.1%, 0.5%); root biased; root absent; MV-side voltage swings (±3%).
- **Metrics**: recovery and false branches against the variants and against RNJ15.
- **Expected**: the source-term version is unaffected by root problems, and the others degrade. The source-term version loses a little efficiency when the root is perfect; measure how much.

**F. Speed and scaling.** After S1 and S2: solve time against history length (8, 18, 36 days) and meters per feeder (8–60), for each stage. Report how often the time budget binds.

## 5. Experiments for question 2: showing RNJ+WZZT beats other methods

**Baselines.** All get the same terminal-only data, noise, split and output format (a laminar family). Their hyperparameters are tuned on a development set kept apart from the test set. References are to verify in a literature search before citing.
- **Distance and latent-tree methods:**
  - neighbour joining (NJ);
  - RNJ1 and RNJ15;
  - recursive grouping and Chow–Liu recursive grouping (latent-tree learning, Choi et al. 2011);
  - the terminal-node methods of Deka, Backhaus, Chertkov and of Park, Deka, Chertkov (LinDistFlow distances with hidden nodes);
  - hierarchical clustering of voltage correlations (UPGMA).
- **Sparse regression**: a nonnegative group lasso over a candidate support dictionary (in the spirit of Liao et al. 2019), and thresholded least-squares sensitivities followed by tree building.
- **References, not competitors:**
  - the truth refit (best possible fit);
  - the data-supported ceiling (segments the BIC supports);
  - correlation-plus-spanning-tree methods (Bolognani et al. 2013; Weng et al. 2017), only on fully metered variants, because they need the internal nodes.

**Case families.**

| Family | What it tests | Where we expect WZZT to win, and why | Where we expect a tie |
| --- | --- | --- | --- |
| B1. Four bundled synthetic feeders (`paper15`, `soumalas11`, `flynn16`, `pengwah18`), study settings | Comparability with earlier work | Short histories: joint re-selection | Long histories: RNJ1 already exact |
| B2. 33 real EU LV feeders, 8 days hourly, 12 start days | Realism; the identifiable truth and the ceiling | Recall of data-supported segments at equal precision | Segments below resolution: no method |
| B3. Random feeders, 10–60 meters, real cable data | Scaling claim | Larger n with many short segments | — |
| B4. Multifurcations: many services at one pole | Structure: NJ forces binary trees and needs a threshold to contract them; WZZT prices each branch by evidence | Fewer false branches | — |
| B5. Root voltage absent or noisy | Source term | Robust where root-based methods degrade | Root perfect |
| B6. Varying power factor, PV back-feed | Excitation | R and X both identifiable, stronger evidence | — |
| B7. Hard cases from D | Exactness | The cells where D shows a gain | Easy cells |

**Metrics and statistics.**
- **Primary endpoint** (fixed in advance): exact recovery of the identifiable topology, compared in pairs (McNemar).
- **Secondary**:
  - normalised Robinson–Foulds distance (Wilcoxon signed-rank);
  - recall of data-supported segments and false segments per feeder;
  - voltage error on held-out days;
  - time.
- **Uncertainty**: bootstrap confidence intervals over feeders.
- **Splits**: development and confirmation are kept separate, as this project already does. Real feeders are split by feeder, not by start day, so that no feeder appears in both.
- **Reporting**: every claim is reported per regime ("on short histories with short segments, X more exact recoveries than the best baseline"), and so is every tie, including where RNJ1 is enough and much cheaper.

**Hypotheses to register before confirmation:**
- H1. RNJ+WZZT (revised) beats the best baseline in exact recovery and RF distance on hard strata (B1 short history, B4, B7).
- H2. Fewer false branches than NJ and RNJ at equal recall.
- H3. On real feeders, higher recall of data-supported segments at equal precision.
- H4. The certified E equals or beats the best heuristic; the heuristic failure rate is reported.
- H5. With the source term, accuracy holds when the root voltage is absent or noisy, while the baselines degrade.
- H6. With compression, the time does not grow with history length, and 60-meter feeders fit the budget.

## 6. Order, cost and decision points

| Phase | Work | Compute (14 workers) | Decision at the end |
| --- | --- | --- | --- |
| 0. Build | S1, S2; source term and loss term as options; a harness that freezes E and W instances; the heuristics; the baselines; a random-feeder generator (10–60 meters, cable data from the EU LV set) | Code, a few days | — |
| 1. Does the MIQP matter? **Done** | A, B, F on the 66 real and 64 synthetic inputs, plus two re-certifications | About 6 h in all | D1 (taken): E — heuristic search, MIQP certificate with a confirmation solve; W — MIQP stays; compression is not a speed result |
| 2. Revised WZZT **Done (E pending)** | R0–R3 and the source term; C and E | C: about 4 h (ablations 3 h more); E: about 2 h | D2 (taken): freeze WZZT-R; R3 refuted; certificate needs tightened bounds (deviation D1) |
| 3. Benchmark **(WZZT-R vs the original: running)** | Tune the baselines on development inputs, then register the protocol; confirmation on fresh synthetic seeds and on held-out real feeders; D; scaling | Building the real inputs: about 4.5 h; D: about 5 h; methods: about 6 h | Claims per regime |
| 4. Back to the probing study | Re-run the synthetic development checks (strata, T1/N1, L1) on the revised stage 0 | As in §18–19 | Then decide Stage M |

**Limits.**
- The resolution floor (finding 4) caps every method on real feeders, so the real-feeder comparison is about recall at equal precision against the ceiling, not exact recovery.
- The literature baselines must be reimplemented. Faithfulness to each paper must be checked on that paper's own example where possible.
- The novelty claim still needs the systematic literature search noted in MANUSCRIPT.md.
