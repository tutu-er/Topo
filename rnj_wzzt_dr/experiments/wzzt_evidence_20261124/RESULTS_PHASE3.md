# WZZT evidence, Phase 3: confirmation of the revised WZZT on unseen inputs (results, 2026-11-24)

Registered protocol: [PROTOCOL_PHASE3.md](PROTOCOL_PHASE3.md) (written before any Phase 3 method run). Method: WZZT-R frozen by D2 (`revised.py`, `RevisedConfig()` defaults). Comparators: the original `111` on demeaned (`O_dm`) and levels (`O_lv`) data. Inputs never used in WZZT development: 192 synthetic (`lock_short01`, `lock_noisy`, seeds 73–96 of the four networks) and 330 real (the 33 EU LV feeders on start days 1–5 and 7–11). Disclosed limitation: the real inputs reuse the development feeders, on different days.

## Decision

**Confirmed.** Both primary endpoints hold (one-sided, α = 0.05).

| Endpoint | Result | Verdict |
| --- | --- | --- |
| P1, synthetic exact recovery, `R` vs `O_dm` (192 inputs) | `R` exact on 92, `O_dm` on 58; only `R` exact on 40 inputs, only `O_dm` on 6 (exact McNemar p = 1.6e-7) | holds |
| P2, real recall of data-supported segments, `R` vs `O_dm` (330 inputs) | 1,167 against 668 of 1,272 supported segments; `R` finds more on 213 inputs, fewer on 4 (Wilcoxon p = 5e-37); pooled precision 86.1% against 82.2% (non-inferiority margin −5 points) | holds |

- **Two failed `R` runs, counted against `R`.** WZZT-R raised a numerical error (`LinAlgError`: LAPACK's SVD did not converge inside the bounded least-squares refit of the E search) on 2 of the 330 real inputs (`feeder59_s3`, `feeder143_s11`). They are counted as `R` finding nothing (P2 numbers above); excluding them instead gives 1,167 against 665 (more on 213, fewer on 2, p = 1e-37). It is a robustness defect of the frozen method (a fallback for the failed factorisation is missing), fixed after the confirmation and not used here.
- **Against `O_lv`** (secondary): only `R` exact on 69 synthetic inputs, only `O_lv` on 3 (p = 1e-17); real supported segments 1,167 against 380, precision 86.1% against 38.0%.

## Accuracy per condition

| Condition | Method | Exact | Recall | Precision | Supported recall | Output recall | Truth among output families | Median time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real (328 with all methods) | `O_lv` | 1 | 19% | 38% | 30% | 26% | 6 | 72 s |
| | `O_dm` | 6 | 31% | 82% | 53% | 41% | 16 | 50 s |
| | **`R`** | **13** | **54%** | **86%** | **92%** | **80%** | **60** | **45 s** |
| `lock_short01` (96) | `O_lv` | 20 | 86% | 79% | 89% | 87% | 20 | 102 s |
| | `O_dm` | 42 | 87% | 93% | 91% | 88% | 42 | 109 s |
| | **`R`** | **59** | **92%** | **98%** | **97%** | **99%** | **89** | 119 s |
| `lock_noisy` (96) | `O_lv` | 6 | 52% | 55% | 56% | 55% | 6 | 114 s |
| | `O_dm` | 16 | 65% | 78% | 70% | 69% | 20 | 123 s |
| | **`R`** | **33** | **77%** | **90%** | **85%** | **91%** | **64** | 120 s |

- The real-feeder gains of development replicate almost exactly: supported recall 92% (development 90%) against 53% (53%), at higher precision.
- Real feeders, against the exact truth: recall 46% (`R`) against 26% (`O_dm`) and 16% (`O_lv`), precision 92% against 88% and 41%. Exact recovery of the exact truth stays at 0–1 for every method (resolution floor); 29 of the 328 identifiable truths have every segment supported.
- The guidance alone: the R0 reference tree (P-only RNJ at tolerance 0.16) reaches 46% real supported recall at 38% precision, the original's RNJ1 trees 24–29% at 18–22%. Most of `R`'s accuracy comes from the structured search on the R/X map, not from the RNJ tree it starts from (see experiment M, Amendment B).

## Secondary endpoints

- **S3, time.** Median wall time of `R` 65 s against 181 s for `O_lv` + `O_dm` (real 45 s against 136 s; synthetic 119–120 s against 221–252 s). Longest `R` run 732 s (a 24-meter feeder).
- **S4, certificates.** E confirmed on 475 of 520 inputs (91%): all 345 inputs with at most 15 meters, 129 of 146 with 16–20, but **only 1 of 29 with 21 or more meters** (43 time limits in all, 2 contradicted after escalation). On 4 inputs the certificate found a better family than the heuristic search. The certificate's cost grows steeply with the number of meters, which motivates the scalable formulation of experiment M (Amendment B).
- **S1, the study's judge.** WZZT-R's family scores at least as well as the truth on the study's own BIC on 327 of 328 real inputs (O_dm 148, O_lv 33), 89 of 96 `lock_short01` (O_dm 64) and 76 of 96 `lock_noisy` (O_dm 38). On the real feeders the truth itself is rarely the best-scoring family: WZZT-R is exact on 13 and finds a different, better-scoring family on 314 (identifiability limits, as in REAL_NETWORK_CASE.md).
- **S2, stage 0 of the probing study** (the study's BIC closure seeded with each method's output): truth in bank R 106/328 against the original's 90/330 on the real feeders, 94/96 for both on `lock_short01`, 77/96 against 66/96 on `lock_noisy`; paired, R only 34, original only 7 (one-sided p = 1.3e-5). Closure finished 69 against 52 (real), 91 against 87, 43 against 30. Strata with R: real 94 S1 / 12 S2 / 222 S3; `lock_short01` 45 / 49 / 2; `lock_noisy` 66 / 11 / 19. Median closure 58 s (real). Two bank jobs failed (`KeyError 'R'`, the inputs where WZZT-R had crashed).

## Files

- Outputs (not in Git): `outputs/wzzt_evidence_20261124/phase3/`: `ceiling.jsonl`, `orig.jsonl`, `R.jsonl`, `bank.jsonl`, `judge.jsonl`; inputs in `phase3/inputs/eulv/`.
- Code: `build_phase3.py`, `run_phase3.py`, `analyse_phase3.py`.
