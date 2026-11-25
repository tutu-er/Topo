# Phase 3 protocol: confirmation of the revised WZZT (registered 2026-11-24, before any Phase 3 method run)

Phase 2 froze WZZT-R on the 130 development inputs (rule D2 passed; [PROTOCOL_PHASE2.md](PROTOCOL_PHASE2.md), results in RESULTS_PHASE2.md). Phase 3 tests whether its gains over the original WZZT (`111`) hold on inputs that WZZT development has never used. Only the input building ran before this registration (`build_phase3.py`; it runs no identifier). Nothing in this file is changed after the Phase 3 runs start; anything added later is labelled exploratory.

## Method under test (frozen)

WZZT-R exactly as frozen by D2: `rnj_wzzt_dr/identification/revised.py`, `RevisedConfig()` defaults, i.e. source "gls" (the measured root reading), offsets, BIC price, P-only guidance, generator-first W with the MIQP no-gain check, the E portfolio with the D1 certificate (tightened bounds, one cold solve with the cutoff, 240 s, `NumericFocus=3` escalation), the whole history, a 900 s budget. Run by `run_phase3.py`.

The source-term variant (experiment E of Phase 2) is not part of this confirmation; it runs on the development inputs afterwards and needs its own confirmation if it is to be claimed.

## Inputs (never used in WZZT development)

- **Synthetic:** `lock_short01` and `lock_noisy`, seeds 73–96 of each of the four networks: 96 + 96 inputs. (Phase 1–2 used seeds 65–72. These inputs were built by the hourly-probing study and used in its §18 block-selector confirmation, never by WZZT development.)
- **Real:** the 33 EU LV feeders on start days 1–5 and 7–11 (330 inputs; 8-day hourly history, 6 training and 2 validation days, identifiable truth). Disclosed limitation: these are the same 33 feeders as in development (all usable feeders of the data set are in development); the histories are different days, partly overlapping in time with the development windows. So the real-feeder confirmation is about new load data on known networks, not new networks.

## Comparators

- `O_dm`, `O_lv`: the original `111` on demeaned and levels data (`SearchConfig(total_seconds=900, per_mip_seconds=300, output_reserve_seconds=120)`, as in the study's stage 0 and Phase 1).
- Reported, not tested: RNJ1 (the original reference tree) and the R0 reference tree (P-only RNJ at tolerance 0.16).

## Endpoints and decision

**Primary, each tested one-sided at α = 0.05; both must hold for the confirmation claim:**
- **P1 (synthetic, 192 inputs):** exact recovery of the selected family, `R` against `O_dm`, exact McNemar test (inputs where only `R` is exact against inputs where only `O_dm` is exact).
- **P2 (real, 330 inputs):** recall of data-supported true segments in the selected family, `R` against `O_dm`: Wilcoxon signed-rank test on per-input differences in supported segments found, *and* non-inferior precision against the identifiable truth (pooled precision of `R` ≥ that of `O_dm` − 5 points).

The same two tests against `O_lv` are reported as secondary (`O_dm` is the stronger original run in development).

**Secondary (reported with their counts):**
- S1 judge: the share of inputs where the study's own BIC rates `R`'s selected family at least as well as the truth (whole-feeder support scored with and without), against `O_dm` and `O_lv`.
- S2 stage 0 downstream: truth in bank and closure finished after the study's closure, from `R`'s families against the original identifier's (both runs); BIC leader correct; strata.
- S3 time: median wall time of `R` against `O_lv` + `O_dm`.
- S4 certificates: E confirmed, contradicted after escalation, time limit; certificates that found a better family than the search.
- Recall, precision and exact against the exact truth on real feeders; output recall; truth among the output families.

**Interpretation rules.**
- Confirmed: P1 and P2 hold. Partially confirmed: one holds (reported as such, per condition). Not confirmed: neither.
- Results are reported per condition (`lock_short01`, `lock_noisy`, real), including where `R` is not better.
