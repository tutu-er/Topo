# Stage 0 (WZZT): what limits it on real feeders, and a revision that keeps its novelty (proposal, 2026-11-23)

**Why.** On the 33 real feeders (REAL_NETWORK_CASE.md), the true topology entered the candidate bank in only 18 of 66 pilot inputs, and the bank's closure hit its cap in 56. This note checks whether the passive identifier WZZT (RNJ guidance → W → T → E → output contraction, `rnj_wzzt_dr/identification`, method `111`) is the cause, and proposes changes. Nothing here has been adopted. Scripts are in `dev_scripts/` (rows marked "WZZT revision"). The experiments that would test whether the MIQP is needed, and how to benchmark against other methods, are planned in [WZZT_EVIDENCE_PLAN.md](WZZT_EVIDENCE_PLAN.md); Phase 1 results are in [wzzt_evidence_20261124/RESULTS.md](../wzzt_evidence_20261124/RESULTS.md).

**Status (2026-11-24).** S2, R0, R1 and R2 (as "no output floor") were implemented as WZZT-R (`rnj_wzzt_dr/identification/revised.py`) and frozen after the Phase 2 development test ([RESULTS_PHASE2.md](../wzzt_evidence_20261124/RESULTS_PHASE2.md)). R3 was tested and refuted: it lets the closure finish but drops the truth from the bank. R4 is not built. The outcome against the expectations below is under "Outcome of Phase 2" at the end; the confirmation on unseen inputs (Phase 3) is running.

## What WZZT does now (for reference)

- **RNJ guidance**: a pool of candidate branches from Recursive Neighbor Joining on fitted sensitivities.
- **W (greedy forward step)**: starting from a model with no internal branches, each step adds the one branch (support) that improves the training score most. The branch is found by an exact MIQP over all compatible terminal sets, not only the RNJ pool. "Greedy" means it commits to the best single addition at each step and stops when no single addition pays for itself.
- **T**: a joint rebuild of a small neighbourhood.
- **E**: a finite-pool MIQP that re-selects the whole set of branches, so it can drop earlier choices.
- **Score**: training MSE divided by a reference tree's MSE, plus a fixed price α = 0.003 per internal branch.
- **Output contraction**: removes branches whose response is below 8% of the median root-to-terminal response.
- **Selection**: validation MSE.

The hourly-probing pipeline runs WZZT twice, on levels and on demeaned data, and then extends the result by a BIC closure (one-edit neighbours within 6 BIC units, at most 4 rounds, about 1,500 families).

## Findings

**1. WZZT itself never produced the real-feeder truth** (`wzzt_meta.py`). Its output contained the identifiable truth in 0 of 66 real inputs, against 13 and 6 of 66 on two synthetic sets. Everything that did reach the bank came from the closure.

**2. WZZT is fast, accurate as a solver, and too conservative** (`wzzt_profile.py`, seven real feeders, demeaned run):
- Every MIQP solved to optimality, and a whole run took 6–80 s at 8 days of history.
- But WZZT selected only 1–4 of the truth's 2–11 segments, with few false ones: high precision, low recall.
- W usually stopped early ("no gain"): no single extra branch paid the price.

**3. The score and the floor throw away segments the data supports** (`wzzt_criterion.py`, `wzzt_crosstab.py`; 273 segments of the identifiable truth on 33 feeders, start day 0):

| Criterion | Segments supported |
| --- | ---: |
| Study BIC (GLS whitening, meter offsets, identifiable parameters), 8 days | 164 (60%) |
| Study BIC, 18 days | 178 (65%) |
| WZZT score on levels (6 training days) | 154 (56%) |
| WZZT score on demeaned data | 132 (48%) |
| Kept by the 0.08 output floor (demeaned) | 147 (54%) |

Of the 164 segments the data supports, WZZT's demeaned score or its floor discards 65 (40%); the floor alone discards 51. Three design choices cause this:
- **A fixed price per branch gets stricter as data grows.** WZZT keeps a branch only if it lowers the training sum of squares by at least 0.003·N reference variances, where N is the number of readings. That bar is 6.5 for a 15-meter feeder with 6 training days, and 16.5 for a 17-meter feeder with 13.5 days. A consistent information criterion asks for about ln N ≈ 8 per identifiable parameter. So more history makes WZZT *more* reluctant to add branches, which is the opposite of what a longer history should do.
- **Plain MSE.** There is no whitening for the shared root reading, and no meter offsets in the levels run. The downstream score has both, so WZZT ranks families differently from the score that later judges them.
- **A scale rule, not a noise rule.** The 8% floor is relative to the feeder's root-to-terminal response. Real urban feeders are long with short segments, so it removes segments that the meters resolve.

**4. A resolution floor that no search can beat.** At 8 days, 109 of the 273 segments (40%) are not supported by any criterion. Going from 8 to 18 days adds only 22 and loses 8. These segments change voltages by less than the meters resolve.

**5. Where the time goes** (`build_profile.py`, two real feeders, one core). The builds took 146 s and 448 s:
- WZZT (two runs): 66 s and 400 s.
- Closure: 76 s and 42 s. It overshot its cap (1,825 and 4,246 families) with 95 and 325 candidates still unexpanded.

WZZT's MIQP has one residual variable per reading, so its solve time grows with history: 3× from 8 to 18 days in `wzzt_profile.py`.

**6. An exact speed-up works** (`wzzt_compress.py`). The data rows can be compressed once by a QR factorisation, F = QR, using ‖Fθ − y‖² = ‖Rθ − Qᵀy‖² + constant. The E-stage MIQP then has as many residual variables as coefficients, whatever the history length.

| Feeder (meters), days | Original | Compressed | Same optimum and family |
| --- | ---: | ---: | --- |
| feeder159 (15), 8 / 18 | 20.5 s / 93.5 s | 6.0 s / 5.1 s | yes / yes |
| feeder92 (17), 8 / 18 | 19.8 s / 32.7 s | 4.5 s / 4.2 s | yes / yes |
| feeder19 (12), 8 / 18 | 5.5 s / 43.6 s | 1.9 s / 2.2 s | yes / yes |

**7. RNJ guidance and WZZT's parameters** (`rnj_wzzt_sweep.py`, `rnj_wzzt_sweep_analyse.py`). The inputs are the 33 real pilot inputs with start day 0 (8-day history, identifiable truth, 242 internal segments, of which the study BIC supports 145, or 60%) and the first 33 S1 inputs of `lock_short01` (synthetic, exact truth, 210 internal segments). Both use the demeaned training and validation split. "Internal" means below the full feeder; finding 3 also counted the 31 root segments.

*RNJ guidance alone* (recall = true segments found; precision = share of proposed segments that are true):

| Sensitivity fit | Real: pool recall, precision (size) | Real: best single tree | Synthetic: pool recall, precision (size) | Synthetic: best single tree |
| --- | --- | --- | --- | --- |
| Current: R and X jointly from P and Q, 3 weights × 6 tolerances | 50%, 10% (1,219) | 29% (weight .75, tolerance 0) | 86%, 17% (1,057) | 82% (weight .75, tolerance 0) |
| P-only (the identifiable combination when the power factor is fixed), 6 tolerances | 48%, 26% (446) | 45% (tolerance 0) | 88%, 36% (507) | 87% (tolerance 0) |

- P and Q move almost together, so the joint fit splits R from X along a direction the data hardly sees. Both estimates are noisy, and so is every tree built from them; the X-only tree is the worst (32% on synthetic).
- The P-only fit estimates what the data does identify, R + X·(Q/P). Its single tree beats the current fit's best tree on both sets, and its pool has the same recall at 2.4–2.7× fewer candidates.
- Tolerances 0–0.08 give almost the same P-only tree (real 45–43%, synthetic 87–85%). Tolerances 0.16 and 0.24 lose recall on real feeders (31%, 25%).

*WZZT (`111`) under parameter variants.* "Selected" is the chosen family. "Output" is everything WZZT passes to the bank: its frozen candidates plus the uncontracted path.

| Variant | Real: selected recall / precision | Real: output recall / precision | Real: exact | Synthetic: selected recall / precision | Synthetic: exact | Seconds, median (real / synthetic) |
| --- | --- | --- | ---: | --- | ---: | --- |
| Current (α 0.003, floor 0.08) | 32% / 88% | 44% / 78% | 2 | 83% / 92% | 6 | 75 / 135 |
| Floor 0 | 38% / 82% | 44% / 78% | 2 | 83% / 89% | 6 | 82 / 136 |
| α 0.001 | 33% / 86% | 52% / 68% | 1 | 84% / 73% | 2 | 93 / 147 |
| α 0.001, floor 0 | 44% / 76% | 51% / 67% | 1 | 85% / 68% | 2 | 92 / 132 |
| P-only RNJ, floor 0 | 38% / 81% | 43% / 77% | 2 | 84% / 89% | 6 | 57 / 113 |

Exact counts are out of 33 per set.

- **The floor** changes only the selected family; what reaches the bank is the same. Floor 0 adds 6 points of selected recall on real feeders, at a 3–6 point precision cost, and leaves synthetic exact matches unchanged (6 of 33).
- **A lower constant price is not the fix.** α = 0.001 raises real output recall from 44% to 52%, but synthetic precision falls from 92% to 73% and exact matches from 6 to 2 of 33. The right price depends on the data, which is the case for R1 rather than for a new constant.
- **P-only guidance** gives the same accuracy as floor 0 alone on both sets and runs 17–30% faster, because the E-stage pool is smaller.
- **The time budget binds on a few long real feeders.** Real runs took up to 780 s (the 900 s budget less the 120 s output reserve), and the 90th percentile was 380–540 s. Synthetic runs stayed under 400 s. S1 targets exactly these feeders.
- **The remaining gap is the criterion.** WZZT's output holds 44% of the real truth's internal segments, against the 60% the data supports. Every variant above is still plain MSE without whitening or offsets (R1).

## Proposed revision

The novel core stays as it is: RNJ guidance, W's exact unknown-support MIQP, T, E's laminar finite-pool MIQP, the output contraction step and the frozen bank for DR scoring. What changes is how they score and how fast they run.

**Exact speed-ups (identical outputs; can be adopted without re-running any evidence):**
- **S1. Compressed MIQPs.** QR or Gram compression in T and E (shown above). For W, compress per terminal (each terminal's rows against [its branch columns, P, Q]). This makes MIQP size independent of history length, so weeks of history cost no extra solver time.
- **S2. Sufficient-statistics refits in the closure.** Every fixed-family fit needs only the cross-products of branch flows (Σₜ a_C a_D), which can be computed once. Whitening and demeaning enter analytically (an |C∩D| − c|C||D|/n weight). A refit then costs O(S³) instead of O(T·n·S²), so the closure's 1,500–4,000 refits take seconds.

**Statistical changes (a new WZZT variant; the original `111` stays as the published method and an ablation):**
- **R0. Identifiable RNJ fit.** Fit each meter's R on P and X only on the part of Q that P does not explain (Q orthogonalised against P). Build the trees from the R geometry, and add the X geometry only when that independent part of Q is large enough to measure. Under a fixed power factor this is the P-only fit tested in finding 7: same WZZT accuracy, a 2.4–2.7× smaller pool and 17–30% faster. Drop tolerance 0.24, and keep the reference tree at 0.16 unless a test says otherwise.
- **R1. Score consistent with the downstream judge.** Use GLS-whitened, offset-demeaned residuals inside W, T and E. These are linear row transforms, so the MIQPs stay exact. Charge a sample-size-aware price per *identifiable* parameter (BIC-type: σ̂²·ln N per parameter in the normalised units) instead of a fixed 0.003. One run then replaces the levels and demeaned pair, which halves WZZT's time.
- **R2. Evidence-based contraction.** Keep the contraction step, but remove the weakest branch only while the evidence does not support it (likelihood ratio or BIC), instead of the fixed 8% floor. On the real feeders this keeps the 51 supported segments the floor now drops, and still removes the unsupported ones. Finding 7 limits the effect: the floor changes WZZT's selected family (+6 points of real recall with no floor), but not what it passes to the bank.
- **R3. Closure modulo equivalence.** Expand one representative per voltage-equivalence class (fitted voltages within 0.05%), not every member. On flat landscapes the closure can then finish, so the trust gate means what it says.
- **R4. Resolution certificate (new output).** Report the supported tree plus the groups whose inner structure the data cannot resolve, each with its detectability gap: how much more signal a split needs. Stage 1 can then target exactly those groups, and evaluation can use a resolution-limited truth defined from the true network, loads and meter class (expected evidence ≥ ln N), not from the estimated data. This changes the evaluation target once more, so it needs its own decision and must report the exact and identifiable truths alongside.

**Novelty after the revision.** On top of RNJ-guided exact structural search, the method becomes an identifiability-aware, sample-size-consistent structural search whose cost does not grow with the history length, and which reports the resolution it achieved. Both strengthen the method's position against generic sparse regression or distance-tree methods.

## Expected effect and limits

- WZZT's recall should rise toward what the data supports: up to about 60% of the identifiable truth's internal segments on real feeders, from 32% selected and 44% output now (finding 7). The closure should start near the BIC optimum and finish more often.
- Parameters that should *not* change: the price as a constant (α 0.003; lowering it costs synthetic precision and exact matches) and the RNJ weights beyond what R0 makes redundant.
- Build time: WZZT about 2× faster from one run, plus 3–20× on its MIQPs; closure fits about 100× faster.
- **What will not change:** the resolution floor (finding 4). Exact recovery on real urban feeders at hourly resolution will stay rare for any method; R4 is how to report it honestly.
- **Cost to the protocol:** the selector lock and §11–19 used the current stage 0. S1–S2 change nothing, but R1–R3 change the inputs, so the synthetic development checks (strata, T1/N1 safety, L1 efficacy) must be re-run on the revised stage 0 before Stage M.

## Proposed order and test

1. Implement S1 and S2. Check that the outputs match the current code exactly on stored inputs (families, coefficients, strata), and report the speed-up.
2. Implement R0–R3 as a flagged variant. Development test on the 66 real pilot inputs plus 66 synthetic inputs from `lock_short01` and `lock_noisy`. Metrics: WZZT recall and precision against the identifiable truth, truth in bank, closure finished, build time; then passive and strong-greedy outcomes. Synthetic results must not get worse.
3. Decide on R4 and the resolution-limited target after step 2.

## Outcome of Phase 2 (development inputs, 2026-11-24)

- **Recall rose close to the expected ceiling.** WZZT-R's selected family holds 90% of the data-supported segments on the real feeders (the original's better run: 53%) at 85% precision, and 54% of the identifiable truth's segments (expected: up to about 60%). Synthetic exact recovery rose from 19 to 32 of 64.
- **The closure starts nearer the BIC optimum and finishes more often:** finished 66 against 50 of 130; truth in bank 84 against 75. On the real feeders truth in bank rose only from 18 to 23 of 66: the resolution floor still binds there, as expected.
- **Time.** One run of WZZT-R takes about half the time of the levels and demeaned pair (median 85 s against 168 s). The MIQPs became certificates: W's unknown-support MIQP proves that no further branch helps (all 130 proofs complete), and E's MIQP certifies the search result (122 of 130 confirmed, with tightened bounds).
- **The price.** The expectation that the constant 0.003 should not change is half right. The BIC price is higher than 0.003 on the synthetic inputs (two identifiable parameters per branch) and similar on the real ones (one). Against the constant, it gains real supported recall (228 against 217 segments) but loses synthetic exact recovery (32 against 41 of 64). Because the BIC is also the probing study's judge, WZZT-R keeps it; whether the downstream bank gains from also seeding the lower-price family is checked separately (`explore_price.py`).
- **R3 is refuted** (above). R4 and the resolution-limited truth remain open, and the Stage 0 re-checks of §18–19 (strata, T1/N1, L1) are still needed before Stage M.
