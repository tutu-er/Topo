# Phase 2 protocol: the revised WZZT (registered 2026-11-24, before the full Phase 2 runs)

Phase 2 of [WZZT_EVIDENCE_PLAN.md](../hourly_probing_20261118/WZZT_EVIDENCE_PLAN.md): build the revised WZZT (proposals R0–R3, S2 and the source term of [WZZT_REVISION.md](../hourly_probing_20261118/WZZT_REVISION.md)), apply the Phase 1 decisions (E: heuristic search with a confirmed MIQP certificate; W: the unknown-support MIQP stays), and test it on the Phase 1 inputs.

**Status: development.** The inputs are the same 130 development inputs as Phase 1. Development may iterate; every change after this registration is recorded under "Deviations" with its reason, before the run it affects. The claims of the paper need Phase 3: confirmation on inputs that WZZT development has never seen (synthetic `lock_*` seeds 73–96; real feeders on start days other than 0 and 6, disclosed as not feeder-held-out because all 33 usable feeders are in development).

Only a smoke test (three inputs, `R` only, plus a certificate-gap timing check on two inputs) ran before this registration.

## The method under test: WZZT-R (`rnj_wzzt_dr/identification/revised.py`)

The original `algorithm.fit` (method `111`) is unchanged and stays the published method and the reference.

- **Model.** y_t = Σ_C z_C z_Cᵀ(r_C p_t + x_C q_t) + s_t·1 + o + e_t. One linear residual operator M = D ⊗ S_c removes the nuisance terms:
  - D: free meter offsets o (per-meter time demeaning);
  - S_c = I − c·11ᵀ/n in each slot: GLS for the root-meter error shared by every y_i (c = γ from the root reading, "gls"), a free source-bus term s_t (c = 1, no root reading needed, "free"), or nothing (c = 0).
- **Score (R1).** J(F) = ‖M(y − Fθ)‖²/σ̂² + ln N · p_s · k(F).
  - σ̂²: the saturated symmetric linear model (every R_ij and X_ij) under M.
  - k(F): internal supports. p_s: identifiable parameters per support, 1 when Q moves with P within the meter class (pq_sigma 0.005), else 2.
  - This is the hourly-probing study's BIC without its loss term and resistance floor. One run on the whole history (training + validation slots) replaces the levels/demeaned pair and the validation pick.
- **R0 guidance.** A P-only ordered sensitivity fit under M; RNJ trees at tolerances 0, 0.04, 0.08, 0.16; the reference tree at 0.16.
- **W.** Generator-first, one support per step: the RNJ pool plus G2 (sibling pairs), G3 (RNJ on the residual), G4 (residual clusters) and G5 (exact screening of every union of a node's children, top 50), each refit exactly. When no generator candidate gains, the unknown-support MIQP (SVD rows, `NumericFocus=2`, 120 s) runs as the no-gain check; a support it finds is accepted and W continues.
- **E.** The finite pool is the RNJ pool, every path support, every frontier candidate of the final W family (cuts of width ≤ 5) and every pair. Search: beam (width 20), local search from the W result and from each RNJ tree, local search from 20 random laminar starts (budget 120 s). Certificate: the SVD-row MIQP with `NumericFocus=2`, warm-started at the search result; confirmation: a cold start with Seed 7 (each capped at 120 s, MIPGap 1e-4 relative, at most about 0.25 BIC units here; the pre-registration timing check found 1e-4 and 1e-5 equally slow, about 85–115 s per solve on 16-meter synthetic inputs, and Phase 1 validated 1e-4 for this certifier). A certificate incumbent that is better than the search replaces it.
- **R2.** No output floor: the BIC is the contraction rule.
- **Output to the bank.** The W path, the E result, and up to 20 families from the E search within 6 BIC units.
- **Budget.** 900 s in all; W gets up to half.
- **Fits.** All fixed-family fits are exact bounded least squares from Gram statistics computed via the Kronecker structure of the transformed columns (S2). Tests (`test_phase2.py`) check the transformed columns, the Gram statistics, the fits against explicit and package fits, and both MIQPs against brute force.

## Inputs and targets

The 130 Phase 1 inputs: the 66 real pilot inputs (`eulv`) and 32 each of `lock_short01` and `lock_noisy`. The target is the input's truth: the identifiable truth on real feeders (segments below 0.003 Ω merged), the exact truth on synthetic ones. On real feeders the exact truth is also reported.

**Data-supported segments (the ceiling).** A true internal segment is *supported* when removing it from the truth raises the study's BIC (with the condition's offsets, GLS, loss term, resistance floor and identifiable-parameter count) on the whole history. Supported: real 254/484 (52.5%), `lock_short01` 175/184, `lock_noisy` 167/184.

## Experiment C: component ladder

**Variants.** `R` (full WZZT-R, source "gls") and its ablations:

| Variant | Change from `R` | Tests |
| --- | --- | --- |
| `R_pq` | original 15-tree guidance | R0 |
| `R_alpha` | α = 0.003 per support, reference-tree MSE as the unit | R1 price |
| `R_plain` | no offsets, no slot term (plain least squares) | R1 residual operator |
| `R_c0` | offsets, no slot term | GLS |
| `R_src` | free source term (c = 1) | source term |
| `R_wmiqp` | pool trials + the unknown-support MIQP at every W step | W design |
| `R_wpool` | pool trials only | W reach beyond the pool |
| `R_emiqp` | E solved by the MIQP from the W result (no heuristics), plus the cold confirmation | E design |
| `R_train` | search on training slots only | use of the whole history |

**Reference.** The original `111` on levels (`O_lv`) and demeaned data (`O_dm`), from the Phase 1 capture (same inputs, same code as the study's stage 0).

**Metrics.** For the selected family: exact, recall and precision of internal segments, recall of supported segments. For the output (every family passed on): recall, and whether the truth family is among them. Time. Certification.

## Experiment B (downstream): stage 0 of the probing study

The study's stage 0 runs the BIC closure (one-edit neighbours within 6 BIC units, ≤ 4 rounds, ≤ 1,500 families), prunes to 30 BIC units and 150 families, and assigns strata. Run from:
- `orig`: the original identifier's families (both Phase 1 runs: every path family and the selected family);
- `R`: WZZT-R's output.

Each with the study's closure and with **R3**, the closure modulo voltage equivalence (a candidate whose fitted voltages on the history are within 0.05% of an already expanded candidate's is not expanded). Metrics: truth in bank, closure finished (no cap), leader correct, strata, closure time.

## Experiment E: the source term

Root-reading scenarios applied to the stored readings (y = V0′² − V²): `exact` (as stored), `noisy01` and `noisy05` (root reading with an extra 0.1% or 0.5% random error per slot), `biased` (+0.5%), `absent` (V0′ = 1 p.u.). Variants `R` (gls), `R_src` (free) and `R_c0`. Metrics as in C.

**Expected (stated in advance).** `R_src` is unaffected by the root scenario; `R` and `R_c0` degrade under `noisy05` and `absent`; with an exact root, `R_src` loses a little against `R`.

**Variant detail (registered with D1, before any experiment E run).** Experiment E runs `R_h`, `R_src_h` and `R_c0_h`: `R`, `R_src` and `R_c0` with the search result and no certificate, for all five scenarios including `exact`. Certificates are the subject of experiment C. In the first two `R` ladders they changed no selected family, and they would double the cost of the 1,950 runs.

## Decision rule D2 (freeze the revised WZZT)

WZZT-R is frozen as the stage-0 identifier for Phase 3 and the probing study if all of the following hold on the development inputs:

1. **Synthetic exact recovery** (64 inputs): `R` against `O_lv` and against `O_dm`, paired: inputs where only `R` is exact ≥ inputs where only the original is exact.
2. **Real feeders** (66 inputs): recall of supported segments in the selected family at least as high as `O_dm`'s and `O_lv`'s in total, with precision (against the identifiable truth) no more than 5 points lower.
3. **Downstream:** truth in bank with `R` (best of the two closures) ≥ with `orig` (same closure), over all 130 inputs; closure finished at least as often.
4. **Certification:** E certified and confirmed on ≥ 90% of inputs within the budget.
5. **Time:** median wall time of `R` ≤ the median of `O_lv` + `O_dm` (the pair it replaces).

If a rule fails, the failing part is diagnosed and revised as a development deviation; the source-term option for Phase 3 is chosen from experiment E (the default stays "gls" unless "free" is at least as accurate with an exact root).

## Deviations

**D1 (registered 2026-11-24 after the first `R` ladder, before its re-run): a tightened, cold, cutoff certificate.**
- *Observed in the first run* (kept in `phase2/run1/`): accuracy as expected, but E was confirmed on only 101 of 130 inputs (rule 4 needs 117). About 26 instances, mostly 15–24 meters, hit the 120 s caps in both solves. On 3 inputs with 24 meters, the cold solve stamped "optimal" with a lower bound about 10 normalised units *above* a known feasible family (the false-stamp problem of Phase 1, §1). The certificates never improved on the search.
- *Diagnosis.* The big-M constant is the coefficient bound 2, while fitted coefficients are 0.02–0.07 (median) and at most about 0.4. The relaxation then carries almost none of the support penalty.
- *Change.* Optimality-based bound tightening (`revised.response_bounds`): for every meter, the largest R_ii and X_ii of any coefficient vector whose residual objective is at most the best known value (a convex QCQP per meter and block, with a 5% safety margin). Every family at least as good as that value satisfies them, so the optimum is unchanged. They set each support's big-M to the smallest bound over its meters, and add the per-meter sums as valid constraints. The certificate is now one cold solve (Seed 7, no warm start) with these bounds and the best known value as the solver cutoff, capped at 240 s instead of two 120 s solves. It must rediscover a family at least that good and prove it ("confirmed"). A bound above the known value, or "infeasible under the cutoff", is a detected false bound ("contradicted"), which a warm-started solve could not reveal.
- *Evidence before adopting it* (5 hard instances, 300 s per solve, under load): the false stamps disappeared (valid bounds 2–4% below the best value on the two 24-meter feeders); `flynn16_s65` and `pengwah18_s65` were proved optimal in 150–246 s, where the untightened solves timed out; on `eulv_td400291_feeder10_s0` (the instance Phase 1 could not settle) both tightened solves found a family better than the heuristic search. The 21–24-meter feeders still need more than 300 s.
- *Effect.* `R`, `R_emiqp` and the downstream `R` closures are re-run; the ablations run once, with D1. Rule 4 now counts confirmed certificates of this form.
- *Second run with D1 (kept in `phase2/run2/`) and two corrections* (registered before the third run):
  - The second run confirmed all 64 synthetic inputs and 52 of 66 real ones (116 of 130); no selected family changed. Eight real inputs were "contradicted".
  - **A bug in the tightening:** each meter's bound was capped at the coefficient bound 2, but R_ii and X_ii *sum* every support containing the meter and can exceed 2 (X_ii = 2.30 on `feeder113_s0`, where the cap excluded the known family). Fixed: the trivial bound is 2 × the number of supports containing the meter; a test now covers a case where the sums exceed the bound. Because a too-tight bound can also make a certificate look confirmed, the whole `R` ladder is re-run.
  - **Gap-consistent verdicts:** on `feeder44_s0` and `feeder160_s0` the solver reported "nothing at or below the cutoff" with a bound 2e-6 above the held value, inside its 1e-4 relative gap; `NumericFocus=3` confirmed `feeder44_s0`. The verdict is now stated in the units of the gap g = 1e-4: the solver cutoff is h + g·|h|; *confirmed* when the solve is optimal and its bound is within g of the held value h (or of a better family it finds); *contradicted* when the bound exceeds h, or the solver reports nothing at or below the cutoff although h is attained. A contradiction is escalated once with `NumericFocus=3`, from a cold start, within the same 240 s cap.
  - For the record: without a cutoff, the cold tightened solve on `feeder44_s0` stamped a family 0.25 units worse "optimal", with a bound 0.91 above the known value, a gross false stamp of the Phase 1 kind.

**D3 (registered 2026-11-24, before any experiment E run): a bug fix and a data-consistent GLS variant for experiment E.**
- *Bug.* `revised.fit` with `e="heuristic"` (the setting of every experiment E variant) raised `UnboundLocalError` (the certificate verdict was initialised only inside the certificate branch). Fixed by initialising it before the branch. The default `e="certified"` (the frozen method, Phase 2 results and Phase 3) does not take this path, so no result changes. Regression test: `test_heuristic_e_runs_without_certificate`.
- *What the free source term implies for the data, checked.* With a free source term (c = 1) every per-slot common term is projected out, so the root reading cannot affect the fit: rebuilding y = V0′² − V² with any other root reading (or with V0′ = 1, no root meter) gives the identical selection, noise estimate and objective (`test_free_source_fit_ignores_the_root_reading`; on `soumalas11_s67` identical to 10 digits under `exact`, `noisy05` and `absent`). The whole-feeder support drops out at the same time (its transformed column is zero), and the noise degrees of freedom lose one per slot. So no separate data adjustment is needed for the free source term, and the Phase 2 `R_src` rows are already the result *without any root reading*. `R_src_h` is kept in experiment E as a check that its five scenarios agree.
- *What was not consistent: the GLS strength.* `source="gls"` sets γ from the root reading assuming the root meter is as accurate as a household meter. When the root reading is noisier or absent, the common per-slot error is larger and γ should be larger; the registered `R_h` keeps γ ≈ 0.71–0.78 in every scenario. New option `source="estimated"` (`revised.estimated_gamma`): ρ is estimated from the residual of the saturated symmetric model fitted with offsets and no slot term, as the slot means' variance against the within-slot spread, and γ = 1 − 1/√(1 + nρ). Checks: with the stored root it recovers ρ ≈ 1 (0.91–1.19 on three inputs), the GLS model's assumption; `noisy01` raises ρ to 1.8–4.8, `noisy05` to 12–26, a constant bias leaves it unchanged (absorbed by the offsets); `absent` raises it only to 1.0–2.4, because the saturated model absorbs the part of the source variation that follows the loads. Unit test: `test_estimated_gamma_follows_the_common_error`.
- *Change to experiment E.* A fourth variant `R_gest_h` (`source="estimated"`, no certificate) runs in all five scenarios after the three registered ones.
- *Reference sets.* The data-supported segments (ceiling) and the study's judge are computed with the stored root reading and the study's loss term. For variants without a root reading they are a reference, not the model those variants fit; experiment E therefore reports exact recovery, recall and precision against the truth first.

**D4 (2026-11-25, at the user's request; after Phase 3, before the transformer-network fits): the ordered matrix fits use Gurobi.**
- *Change.* WZZT-R's two ordered matrix fits, the guidance (P-only fit under M, then RNJ at four tolerances) and the residual-RNJ generator in W, are solved by Gurobi's barrier (`trees.ordered_qp_gurobi`) instead of SciPy SLSQP: new `RevisedConfig.qp_solver`, default `"gurobi"`. The QP itself (objective, order constraints, margins) is unchanged. `qp_solver="slsqp"` reproduces the frozen runs. The original WZZT baselines (`O_lv`, `O_dm`) and `miqp_ext.setup` (used by the running experiment M anatomy stage) keep SLSQP.
- *Reason.* SLSQP's dense iterations dominate WZZT-R's set-up on large networks (guidance 11 s at 34 meters, 73 s at 44; whole set-up 22 min at 63, 78 min at 73, 3.4 h at 77); Gurobi solves the same QP in seconds.
- *Equivalence check* (163 inputs: the 130 development inputs and the 33 sixteen-day inputs; `outputs/wzzt_evidence_20261124/equiv_qp.jsonl`). Both solvers reach the same optimal value on every input, for the guidance and for the residual fit at the first W step (largest relative difference 1.4e-9; SLSQP success and Gurobi status OPTIMAL on all). The QP has a flat optimum (P-only features are nearly collinear), so the matrices can differ in directions that do not change the fit (relative difference median 2e-7, up to 0.98), and RNJ at tolerance 0 breaks such ties differently: the guidance pool is identical on 141 of 163 inputs, the reference tree on 150, the residual-RNJ candidates of the first W step on 47. Every candidate is still refitted exactly and judged by the same score, so results can change only where a tie-broken candidate decides a step.
- *Consequence.* Results already reported (Phase 2, Phase 3, experiment E) stand as run with SLSQP. Runs started after this deviation record `qp_solver`. The impact on the selected trees is measured on the development inputs before any confirmatory use (WZZT-R with each solver, same inputs).
- *D4 impact, measured (2026-11-25, `outputs/wzzt_evidence_20261124/d4/dev_R_gurobi.jsonl`).* WZZT-R with the Gurobi guidance on the 130 development inputs selects the same tree as the stored SLSQP run on 128 of 130 (the other two, `eulv_td400858_feeder59` on days 0 and 6, end on different trees; the Gurobi run scores 1.6 and 3.5 half-BIC units lower and neither of these two Gurobi runs is certified (the SLSQP runs were not checked); cause not established: different candidates or a time-limited search under another machine load); synthetic exact 32 against 32, real supported lines 228 against 228 of 254, precision 91.3% against 91.3%; no errors; median wall time 68 s against 85 s (different machine load). The accuracy counts on these inputs are unchanged; two selected trees differ.
