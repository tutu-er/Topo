# Experiment M: where WZZT's exact search is critical (design, registered 2026-11-24, before any run)

**Question (user, 2026-11-24).** Not to raise accuracy further, but to show what the MIQP gives that heuristics cannot: find the circumstances in which WZZT's exact search is critical, and add components where needed while keeping the method's novelty (RNJ-guided, exact structural search over unknown supports).

## What is already known

- On the 130 development inputs (at most 24 meters) the cheap generators found every W step, and the unknown-support MIQP, run once at the end, proved 130 times that no further branch adds evidence (Phase 2 §4). Growing W by MIQP (`R_wmiqp`) was no more accurate and was slower (§6). Phase 1: generators reached 96.9% of the original W's steps; their misses were large supports (7–19 meters).
- The E MIQP never improved the heuristic portfolio's family in the final Phase 2 run; it certified it (122/130).
- W is greedy: it adds one branch at a time, the one that helps most, and never revisits that choice; E then re-selects the whole set from a finite pool.

So on small feeders the MIQP's value is the proof, not the search. Three circumstances should change that. Each is a mechanism, with a test that runs the exact and the heuristic search on the same instance.

## M1. Search space too large to enumerate (scale)

At a node with k child blocks there are 2^k − k − 2 compatible new supports. WZZT-R's screening enumerates them only for k ≤ 20 (and ranks them by an approximate gain with the old coefficients fixed); the RNJ, residual-RNJ and residual-cluster generators propose O(k) sets each. The MIQP searches all of them exactly. At the first step every meter is a child of the root, so k = n.

- **Instances.** The whole LV network of each EU LV transformer with 26–77 metered connections (17 transformers; all feeders of a transformer under its LV bus, normally-open fuses open), plus the 6 single feeders with 26–44 connections; start days 0 and 6, as the development inputs (46 inputs). Same physics, meters and history as `eulv` (8 days hourly). Identifying the network under a transformer, including which customer is on which feeder, is a real utility task.
- **Measurement (anatomy of W).** WZZT-R's W in generator mode, with the unknown-support MIQP run at *every* step on the same current family. Per step: the best generator support and its exact gain, the MIQP's support and its exact gain, the node width k, the support size, whether each support is a true segment. A step is *MIQP-critical* if the MIQP's exact refit beats every generator candidate by more than the gain tolerance. Also the final family of W by generators against W by MIQP (`w="miqp"`), each followed by the same E.
- **Hypothesis H1.** MIQP-critical steps are rare on the development inputs (k ≤ 24) and common on the transformer networks, concentrated at wide nodes (k > 20) and large supports; the MIQP's supports there are mostly true segments (feeder-level and street-level groups).

## M2. Interacting branches (pairs), induced by the source term

With the slot term (GLS strength c, or a free source c = 1), the transformed columns of two disjoint supports A, B are negatively correlated: ⟨w_A, w_B⟩ = −κ|A||B|/n per slot, κ = 2c − c², times the correlation of their (demeaned) flows. For two unit-norm columns with correlation ρ < 0, the joint gain 2 + 2ρ exceeds the sum of the single gains 2(1 + ρ)² by −2ρ(1 + ρ) > 0: gains are super-additive. Two branches that each fall below the price can pay together. Every one-at-a-time method misses such a pair: the generators, beam and local search, and the single-support MIQP and its no-gain certificate.

- **New component W2 (exact joint extension).** `extend_k_miqp(k=2)`: two unknown supports chosen jointly, each compatible with the family and with each other (nested or disjoint), each charged the price; the same exact linearisation and SVD-compressed rows as the single-support MIQP. Searching all pairs of the exponential support space is only practical this way.
- **Measurement.** From WZZT-R's selected family on every development input and on the transformer networks, with c = γ (as selected), and on the development inputs also with c = 0 and c = 1: does W2 find a pair that lowers the objective while the single-support certificate holds? Is the pair true, and does the truth's score improve?
- **Hypothesis H2.** Improving pairs exist after the single-support certificate, more often as c rises, and they are mostly true segments.

## M3. Proofs: certified resolution and data requirement

A heuristic gives a lower bound on the best possible improvement (the best it found); only the exact search gives an upper bound. That upper bound is what can say "the data cannot resolve this", and how much more data would be needed.

- **New component (R4 of WZZT_REVISION.md): certified resolution report.** For each internal node v of the selected family with at least three child blocks, the node-restricted unknown-support MIQP (`extend_k_miqp(within=v)`) proves the largest evidence ΔJ_v (BIC units) that any single compatible split inside v could add. Gap: price − ΔJ_v. Since the evidence of a split grows about linearly with the history length while the price grows with ln N, the certificate gives a data requirement: v cannot be split by a single support with fewer than m_v ≈ price/ΔJ_v times the history (at similar load variability).
- **Validation with longer histories.** The 33 real feeders with 16 days of history (14 + 2) from start day 0 (the recorded 20 days allow it), compared with the 8-day development input of the same start day: nodes certified as needing m_v > 2 should stay unsplit at 16 days (a falsifiable, MIQP-only prediction); nodes with m_v ≤ 2 are predicted splittable and should be split.
- **Hypothesis H3.** At 16 days, WZZT-R splits fewer than 10% of nodes certified m_v > 2, and most nodes certified m_v ≤ 2 (if any).

## M4 (later): certified bank completeness for the probing study's trust gate

The probing study's lock failed T1 (wrong confirmations when the truth is missing from the bank; DEVELOPMENT_RUNS §18), and no logged statistic separated truth-missing inputs (AUC 0.45–0.64). From the bank leader, one exact MIQP can test whether any one-support edit outside the bank beats the leader by more than the gate's margin. Planned after M1–M3, on the §18 truth-removed inputs; not part of this run.

## Amendment A (2026-11-24, after feasibility probes, before any experiment M run)

- **The exact single-support MIQP does not scale to the transformer networks in its present form.** On the 44-meter network `eulv_td400291_all` (start day 0), the first W step (the root has 44 child blocks) found no feasible solution and no bound within 300 s. The SVD-compressed model has 1,938 dense rows over 2,070 columns (about 4 million coefficients), and only 967 singular values exceed 0.1% of the largest, because active and reactive flows move almost together on these feeders. So M1 as designed is not testable above about 34 meters. That is a finding in itself: the circumstance in which exact search is most needed (wide nodes) is where the current formulation stops. A scalable exact formulation (block-aggregated per node, reparametrised to the identifiable impedance, with tightened bounds) is a separate piece of method work, not part of this run.
- **Scope of this run, therefore.** M1 (anatomy) on the development inputs and on the transformer networks and large feeders with at most 34 meters (where the root still has more than 20 child blocks, so screening is off at the first steps). WZZT-R and WZZT-R with MIQP-grown W run on all 46 large inputs (with W's MIQP time-limited, it falls back to the pool candidates; reported). M3 certificates skip nodes with more than 34 meters (reported as skipped).
- **W2 is borderline** at about 19 meters (one 300 s solve ended without a bound; on 18 meters it was proved in 78 s). M2 runs on the development inputs only, with 300 s per solve; solves without a proof are reported and excluded from the "interaction-critical" count.
- **Order.** M3 on the development inputs and the 16-day inputs first (cheap: 9–175 s per input), then M1, the large fits, M3 on the large inputs, and M2 last.
- **Code change before the run.** `extend_k_miqp` builds variables only for the meters a new support may contain (a node's meters for certificates), so a certificate on a node of m meters costs O(m²) variables, not O(n²). Tested against brute force, including a node with an inner support (8 tests).

## Amendment B (2026-11-24, user's framing; before any run of M5): the R/X map, and a scalable formulation

**The essential difference (user).** The MIQP works directly on the whole R/X map and its fitting error; NJ/RNJ works on the elements of the best-fitting matrix. Precisely: with the transformed fitting error written on the map Θ = [R; X] (prediction Y^ = P R + Q X, R = Σ r_C z_C z_Cᵀ), and Θ̂ the unconstrained least-squares map,

  SSE(Θ) = SSE(Θ̂) + (vec Θ − vec Θ̂)ᵀ H (vec Θ − vec Θ̂),   H = S_c² ⊗ ΦᵀΦ,  Φ = [P Q].

So WZZT's MIQP is the projection of Θ̂ onto the tree-structured (laminar, non-negative) maps **in the metric H**, the information the data carry about every entry and their correlations, with the BIC price. RNJ is a plug-in method: it treats the entries of Θ̂ (or of a P-only fit) as data and clusters them in a fixed metric. It cannot use how precise each entry is, how entries' errors are correlated, which combinations are identifiable (under a fixed power factor only r + x·tanφ is), or the structure while estimating.

**M5 (new mechanism): where the plug-in fails.** The difference matters where H is far from a multiple of the identity on the entries. Above all it matters when the history is short relative to the network: the unconstrained map has n(n+1) entries to estimate from T·n readings (ratio T/(n+1)), while the structured fit has O(n) parameters. At fixed history, RNJ's input therefore degrades as networks grow. That is the regime where the MIQP must scale. Also: uneven load variability across meters, correlated customer profiles, and P/Q collinearity.
- *Measurement.* RNJ's guidance quality (truth recall of its pool and its reference tree) and WZZT-R's accuracy against T/(n+1): the development feeders (8 days, T/(n+1) ≥ 7.7), the 16-day histories (≥ 15), and the transformer networks (down to 2.5 at 77 meters).
- *Hypothesis H5.* The reference tree's recall falls with T/(n+1) faster than that of the structured search (W + E with the MIQP certificate).
- *Added before its run: the same feeders with shortened histories* (`run_miqp history`). The 66 development real inputs truncated to their last 48 and 96 hourly slots (2 and 4 days) and run in full (192), plus the 16-day inputs (384) from the fit stage. Recall and precision against the identifiable truth, for the reference tree and for WZZT-R. This varies T/(n+1) on fixed networks from about 2 to 45. Already observed in Phase 3 (fixed T = 192, larger feeders): the reference tree's supported recall falls from 67% (8–12 meters) to 27% (21–24) while WZZT-R stays at 89–96%.

**A time-free formulation (the scalable MIQP).** In the same terms, the objective is Σ_m ‖ỹ_m − Φ θ̃_m‖², with θ̃_m the m-th column of Θ S_c: it needs only ΦᵀΦ (2n × 2n), ΦᵀỸ (2n × n) and ‖Ỹ‖². The history length drops out, and the quadratic part becomes block-diagonal over meters, O(n³) nonzeros, instead of the dense SVD-compressed rows of every pair column, O(n⁴) (about 4 million at 44 meters). `miqp_ext.extend_tf_miqp` implements it for k = 1 or 2 supports, node-restricted and forced. It has the same model and the same optimum: tested against brute force and against the SVD form for c = 0, 0.7, 1, with and without offsets, for pairs and node certificates. Benchmarks against the SVD form on the first W step (the widest node) at 24–44 meters decide whether M1 can be extended beyond 34 meters.

## Amendment C (2026-11-24, before the heavy runs): what limits the exact search, and the scalable components

Diagnosed with Gurobi logs on the 26-meter transformer network `eulv_td401346_all` (first W step, the root has 26 child blocks):
1. **Partial MIP starts.** A start holding only the old coefficients (no branch choice) makes Gurobi complete it by a sub-MIP as hard as the problem: the whole 90 s went into "processing user MIP start". The frozen `revised.extend_miqp` sets such a start too (its W no-gain check), which explains part of its slowness on large feeders. In the new components a start is given only with a branch choice.
2. **A slow root relaxation** (64 s, 27,000 simplex iterations, coefficient range 1e-6 to 1e3) for the global model.
3. **A weak relaxation at wide nodes.** The root bound was 0.94 against an incumbent of 11.1 (gap 91%). Choosing the best subset of meters behaves like a boolean quadratic program: with fractional membership and a large coefficient bound, the relaxed new support can shape almost any non-negative map. Solver settings (barrier root, NumericFocus) did not change this.

**Scalable components (exact, tested against brute force; 18 tests).**
- *Time-free form* (Amendment B): T-independent, O(n³) block-sparse instead of O(n⁴) dense.
- *A tighter exact linearisation* of the new support's map: off-diagonal entries bounded through the diagonal ones (u_ij ≤ d_i, u_ij ≤ d_j, u_ij ≥ d_i + d_j − r, with d_i = r z_i), with no z_i z_j variables.
- *Exact node decomposition with block aggregation* (`certify_by_nodes`). Every compatible new support is a union of two or more child blocks of exactly one node. So the best single extension (and the no-gain certificate) is the best over the nodes, and each node's MIQP has one binary per child block and, compressed, about 2K + k(k+1) continuous variables whatever the number of meters. After W and E have built a tree, nodes have 2–6 blocks, so certificates scale with the widest node, not with n. The hard case that remains is a first W step at a very wide root (no structure yet), which the generators handle and where the MIQP gives a certified bound if it cannot close the gap.

**Run changes (before the heavy stages; the light stages switch for the jobs not yet run, recorded per row as `form`).** M3 certificates use one block-level time-free MIQP per node (no size cap). M1 anatomy solves each W step over the nodes with the generators' best score as cutoff: a node that cannot beat the generators is closed by a proof, and a node that beats them marks an MIQP-critical step. The 34-meter cap for anatomy is revisited from the benchmarks below.

## Amendment D (2026-11-25, before the 16-day and transformer-network fits are examined line by line): root lines and terminal lines, against NJ, RNJ and RG

**Design idea (the user's, from the start of WZZT):** distance-based tree building handles the lines near the terminals; the whole-map fit and its MIQP handle the lines near the root; the method is their combination. The current procedure is written out in `METHOD_WZZT_R.md`.

**Exploratory lead (already seen, so not a confirmation).** On the Phase 3 real feeders, of 84 data-supported short lines near the root, the RNJ reference found 7 and WZZT-R 83; at matched whole-map evidence (score loss when the line is removed) RNJ's recall depended on the line's length and WZZT-R's did not. The synthetic feeders contain no such lines. The hypothesis was stated by the user after these data existed.

**Baselines, all built from the same best-fitting matrix.** The ordered symmetric fit of R and X to P and Q (the original guidance's QP, `trees.fit_sensitivity`), on the slots WZZT-R searches (training and validation, per-meter demeaned), solved with Gurobi (`solver="gurobi"`; equal to SLSQP within 1e-7 relative objective in `test_tree_builders.py`). From it:
- **RNJ** (`trees.rnj`), **NJ** (`trees.nj`, Q-criterion with the root as an anchor leaf), **RG** (`trees.rg`, recursive grouping of Choi et al. 2011, new; recovers exact trees, including meters at internal nodes, in tests).
- Each at R/X weights 1, 0.75, 0 and tolerances 0, 0.04, 0.08, 0.16, 0.24 (relative to the median root-to-meter depth): 15 settings per builder. Reported: the default setting (0.75, 0.16) and, per builder and input set, the setting with the best pooled F1 against the truth (an after-the-fact choice that favours the baselines).

**Line classes.** For every internal line C of the identifiable truth: position f = |C|/n (near the root: f > 1/2; terminal side: f ≤ 1/2); relative length ρ = r_C / median root-to-meter resistance (short: ρ < 0.08); whole-map evidence dJ, the study judge's score loss when C is removed from the truth (the Phase 3 ceiling routine with the real-feeder settings).

**Inputs.** Confirmatory: `eulv_16d` (33 feeders, 16-day histories; WZZT-R fits finished, not examined) and `eulv_large` (17 transformer networks and 6 large feeders, when their fits finish). Exploratory: all 520 Phase 3 inputs (reported separately, labelled as such).

**Hypotheses (one-sided, α = 0.05, Holm over the three baselines within a hypothesis).**
- **HR1, root lines.** On lines near the root, WZZT-R's recall exceeds each builder's at its default setting and at its best setting. Test: per-input recall difference, Wilcoxon signed-rank over inputs with at least one such line.
- **HR2, terminal lines (division of labour).** On long terminal-side lines (f ≤ 1/2, ρ ≥ 0.08) the best distance-based builder comes within 10 points of WZZT-R's recall. Descriptive with a bootstrap 95% interval over inputs; supported when the interval's lower end is above −10 points.
- **HR3, mechanism.** Within dJ tertiles, each builder's recall is lower on short lines than on long ones, while WZZT-R's short-minus-long difference stays within ±10 points. Descriptive, bootstrap intervals over inputs.
- Precision (true lines among those output) and false lines per input are reported for every method, so that a recall gain cannot come from over-splitting.

**Not registered here:** an explicit combination (terminal blocks from a distance-based builder, root structure by the MIQP over the contracted blocks). It is a new component and would need its own registration and fresh inputs.

## Status and limits

- **Exploratory characterisation, not a confirmation.** No accuracy claim for WZZT-R rests on experiment M; any new component that is to be claimed (W2, the resolution report) needs its own confirmation on fresh inputs afterwards. The transformer networks and the 16-day histories are new inputs never used in development.
- **Cost.** The MIQP has O(n²) continuous variables; on 60–77 meters each solve is large, so M1 runs with fewer parallel workers and per-solve limits (W steps 300 s, node certificates 120 s), and reports time-limited solves as such (a time-limited MIQP still gives a valid bound).
- **Runs after Phase 3**, so that Phase 3's registered time endpoint is not disturbed.
- **Code.** `rnj_wzzt_dr/identification/miqp_ext.py` (W2, node-restricted MIQP; tested against brute force and against `revised.extend_miqp`), `wzzt_evidence_20261124/build_large.py` (transformer networks and long histories), `run_miqp.py` (anatomy, pairs, resolution), `analyse_miqp.py`.
