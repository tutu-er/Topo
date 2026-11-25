# WZZT evidence, Phase 2: the revised WZZT (results, 2026-11-24)

Design, variants, decision rule D2 and deviation D1: [PROTOCOL_PHASE2.md](PROTOCOL_PHASE2.md). The method is `rnj_wzzt_dr/identification/revised.py` (WZZT-R); the original `algorithm.fit` (`111`) is unchanged. Inputs: the 130 development inputs of Phase 1 (66 real `eulv`, 32 `lock_short01`, 32 `lock_noisy`). These are development results; the confirmation on unseen inputs is Phase 3 ([PROTOCOL_PHASE3.md](PROTOCOL_PHASE3.md)).

## Decision

**D2: freeze WZZT-R.** All five registered rules pass:

| Rule | Result | Verdict |
| --- | --- | --- |
| 1. Synthetic exact recovery, paired | `R` only exact 31 vs `O_lv` only 3 (p < 0.001); `R` only 16 vs `O_dm` only 3 (McNemar p = 0.004) | pass |
| 2. Real recall of supported segments, precision within 5 points | 228 vs 135 (`O_dm`) and 100 (`O_lv`) segments (Wilcoxon p < 1e-7); precision 85.4% vs 83.5% and 51.7% | pass |
| 3. Truth in bank and closure finished (study closure, 130 inputs) | truth in bank 84 vs 75; closure finished 66 vs 50 | pass |
| 4. E certified and confirmed ≥ 90% | 122/130 (94%) | pass |
| 5. Median time ≤ `O_lv` + `O_dm` | 85 s vs 168 s | pass |

## 1. Accuracy (experiment C)

| Condition | Method | Exact | Recall | Precision | Supported recall | Output recall | Truth among output families | Median time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real (66) | `O_lv` | 1 | 25% | 52% | 39% | 35% | 1 | 68 s |
| | `O_dm` | 3 | 31% | 84% | 53% | 43% | 3 | 48 s |
| | **`R`** | 2 | **54%** | **85%** | **90%** | **81%** | **16** | 42 s |
| `lock_short01` (32) | `O_lv` | 3 | 82% | 74% | 83% | 85% | 3 | 98 s |
| | `O_dm` | 13 | 89% | 91% | 92% | 90% | 15 | 91 s |
| | **`R`** | **21** | **94%** | **100%** | **99%** | **99%** | **31** | 120 s |
| `lock_noisy` (32) | `O_lv` | 1 | 55% | 56% | 59% | 61% | 2 | 117 s |
| | `O_dm` | 6 | 65% | 77% | 70% | 71% | 9 | 132 s |
| | **`R`** | **11** | **81%** | **93%** | **89%** | **95%** | **21** | 110 s |

- "Supported" segments are those whose removal from the truth raises the study's BIC (real 254 of 484 true segments, `lock_short01` 175/184, `lock_noisy` 167/184). The real-feeder truth is the identifiable truth (segments below 0.003 Ω merged).
- Against the exact real truth, recall is 47% for `R` against 26% (`O_dm`) and 21% (`O_lv`); no method is exact there.
- On the real feeders `R` selects slightly more segments: 0.68 false segments per input against 0.45 (`O_dm`) and 1.71 (`O_lv`), at the same precision as `O_dm` and with 1.7 times its supported recall.
- **Real exact recovery stays rare for every method,** because about half of the true segments are below what eight days of hourly data can resolve: only 4 of the 66 real truths have every segment supported (`R` exact on 2, `O_dm` on 3).

## 2. Is the search or the data the limit? (the study's own judge)

Each selected family and the truth were scored with the hourly-probing study's BIC (offsets, GLS, loss term, resistance floor, identifiable parameters; the whole-feeder support scored with and without, since the synthetic truths lack it). `judge_phase2.py`.

| Condition | Selected family rated at least as well as the truth: `O_lv` | `O_dm` | `R` |
| --- | --- | --- | --- |
| Real (66) | 12 | 30 | **66** (64 of them a different family rated *better* than the truth) |
| `lock_short01` (32) | 5 | 17 | **30** (21 exact, 9 different and better) |
| `lock_noisy` (32) | 2 | 13 | **28** (11 exact, 17 different and better) |

So on 124 of 130 inputs WZZT-R either finds the truth or finds a family that the downstream judge prefers to the truth. Its remaining misses are limits of the data and the criterion, not of the search. The original runs fall short of the truth's score on 70 (`O_dm`) and 111 (`O_lv`) inputs.

## 3. Stage 0 of the probing study (downstream)

The study's closure (one-edit neighbours within 6 BIC units, ≤ 4 rounds, ≤ 1,500 families), then pruning to 30 BIC units and 150 families:

| Condition | Identifier | Truth in bank | Closure finished | BIC leader correct | S1 / S2 / S3 |
| --- | --- | --- | --- | --- | --- |
| Real | original (both runs) | 18/66 | 10 | 2 | 16 / 2 / 48 |
| | **WZZT-R** | **23/66** | **16** | 2 | 21 / 2 / 43 |
| `lock_noisy` | original | 25/32 | 12 | 11 | 22 / 3 / 7 |
| | **WZZT-R** | **29/32** | **18** | **13** | 25 / 4 / 3 |
| `lock_short01` | original | 32/32 | 28 | 23 | 16 / 16 / 0 |
| | **WZZT-R** | 32/32 | **32** | 23 | 16 / 16 / 0 |

The closure rerun from the Phase 1 identifier families reproduces the stored inputs (real 18/66, `lock_short01` 32/32; `lock_noisy` 25 against 27, because the contracted copies of the original families are not stored).

**R3 is refuted.** Expanding only one family per voltage-equivalence class (fitted voltages within 0.05% everywhere) lets the closure finish almost always, but loses the truth: truth in bank falls to 5/66 (real), 13/32 and 16/32 from the original families, and to 21/66, 20/32 and 29/32 from WZZT-R's. Families that agree to within meter resolution on every reading still differ in BIC over the whole history, so voltage equivalence at reading resolution is the wrong rule for pruning the closure. R3 is dropped.

## 4. How the stages work in WZZT-R

- **W.** Every accepted step came from the cheap generators: RNJ pool 425, exact screening 156, sibling pairs 41, residual clusters 32, residual RNJ 13. The unknown-support MIQP ran once per input as the final no-gain check: 130 calls, all proved optimal, none found a further gain. It is now a certificate ("no compatible support anywhere adds evidence"), which is the role Phase 1 suggested.
- **E.** The heuristic portfolio found the final family on every input. The certificate never found a better one in the final run (in the D1 tests it did once, on `feeder10_s0`, with a longer solve).
- **Time.** Median wall time 42 s real, 110–120 s synthetic; E (mostly the certificate) dominates; W takes 2–7 s.
- **Score.** p_s = 1 identifiable parameter per support on every real input (fixed power factor) and 2 on every synthetic one. The GLS strength c ranged 0.67–0.81.

## 5. Certificates (deviation D1)

| Run | Certificate | Confirmed | Notes |
| --- | --- | --- | --- |
| First (registered) | warm solve + cold confirmation, 120 s each, coefficient bound 2 as big-M | 101/130 | 26 time limits; 3 gross false "optimal" stamps on 24-meter feeders (bound about 10 units above a known family) |
| D1, second | tightened big-M, one cold solve with the search value as cutoff, 240 s | 116/130 | 8 "contradicted", 6 time limits; a bug capped per-meter bounds at 2 |
| **D1, final** | bug fixed; gap-consistent verdict; `NumericFocus=3` escalation | **122/130** | 7 time limits (16–24 meters, gaps 0.4–4%); 1 contradicted after escalation (`feeder59_s0`, 24 meters) |

- **Why tightening was needed.** The big-M constant was the coefficient bound 2, while fitted coefficients are 0.02–0.07 (median) and at most about 0.4, so the relaxation carried almost none of the support penalty.
- **Why it is valid.** For every meter, the largest diagonal responses R_ii and X_ii of any coefficient vector whose residual is within the best known objective (a convex QCQP per meter, with a 5% margin) bound every family at least that good. So the optimum is unchanged.
- **Why one cold solve with a cutoff.** It must rediscover a family at least as good as the search's and prove it. A false bound then *shows*: the solver claims nothing reaches a value we hold. A warm-started solve can never reveal that. One such false bound (the cold, untightened solve on `feeder44_s0` stamped a family 0.25 units worse "optimal") was caught this way during the tests.

## 6. Ablations (experiment C)

Each variant changes one component of `R` and runs all 130 inputs (no errors). Synthetic: exact recovery (64 inputs); real: data-supported segments found (66 inputs, 254 supported) and pooled precision. Paired against `R`, two-sided (exploratory): McNemar on exact recovery, Wilcoxon on supported segments found.

| Variant (one change) | Synthetic exact | `R` only / variant only (p) | Real supported found | p | Real precision | Median time real / synthetic |
| --- | --- | --- | --- | --- | --- | --- |
| **`R`** | **32** | – | **228** | – | 85.4% | 42 / 116 s |
| `R_plain`: no offsets, no GLS (c = 0) | 8 | 28 / 4 (< 0.001) | 162 | < 0.001 | 52.7% | 67 / 108 s |
| `R_c0`: offsets, no GLS | 10 | 22 / 0 (< 0.001) | 193 | < 0.001 | 81.1% | 44 / 100 s |
| `R_train`: training days only | 17 | 15 / 0 (< 0.001) | 210 | 0.005 | 85.1% | 42 / 103 s |
| `R_wpool`: W uses only the RNJ pool | 33 | 1 / 2 (1.0) | 186 | < 0.001 | 71.6% | 32 / 145 s |
| `R_wmiqp`: W grows by MIQP, not generators | 33 | 0 / 1 (1.0) | 208 | 0.07 | 79.2% | 51 / 184 s |
| `R_emiqp`: E by one MIQP, no portfolio | 32 | 0 / 0 | 227 | 0.32 | 84.7% | 24 / 110 s |
| `R_pq`: P and Q guidance | 30 | 2 / 0 (0.5) | 226 | 0.32 | 85.0% | 53 / 140 s |
| `R_src`: free source term (c = 1) | 28 | 4 / 0 (0.13) | 232 | 0.10 | 86.1% | 59 / 167 s |
| `R_alpha`: fixed price α = 0.003 (the original's) | **41** | 4 / 13 (0.049) | 217 | 0.035 | 85.9% | 41 / 138 s |

What each component is worth:
- **The score transform carries most of the gain.** Without meter offsets and GLS (`R_plain`) WZZT-R falls back to roughly the original's level (real precision 53%, synthetic exact 8). Offsets without GLS (`R_c0`) recover part of it; GLS whitening is the larger single component.
- **The whole history helps** (`R_train`): fitting on the 6 training days only loses 15 synthetic exact recoveries and 18 supported real segments.
- **The generators matter on real feeders.** W with only the RNJ pool (`R_wpool`) finds 42 fewer supported segments at 72% precision; growing W by MIQP instead of generators (`R_wmiqp`) is no more accurate on synthetic, worse on real precision (79%), and slower.
- **Neutral components.** The E portfolio and a single E MIQP select the same families (`R_emiqp`), but the portfolio supplies the alternatives: truth among the output families 68 against 36. P-and-Q guidance (`R_pq`) changes nothing and is slower. A free source term (`R_src`) neither helps nor hurts significantly when the root reading is accurate, and is slower; its use case (a poor root reading) is experiment E.
- **The price is a trade-off, not a free gain.** The BIC price is α = p_s ln N / N: 0.0018–0.0048 on the real inputs (one parameter per support) but 0.006–0.009 on the synthetic ones (two parameters per support), two to three times the original's fixed 0.003. With the fixed price, synthetic exact recovery rises from 32 to 41 (13 against 4 inputs; on `lock_noisy` 18 against 11, p = 0.016), while real supported recall falls from 228 to 217 segments (p = 0.035) at the same precision. The BIC is the study's own judge (section 2), so `R` selects the family the judge prefers; the lower price selects families the judge rates worse but that are closer to the synthetic truth. Under the BIC, only 23 (`lock_short01`) and 18 (`lock_noisy`) of the 32 synthetic truths have every segment supported, so the BIC itself caps exact recovery there. This is a property of the criterion, not of the search, and it is the one ablation that does not favour `R`. The price stays as frozen for Phase 3; a lower price, or an output that also carries the lower-price family, would be a new variant needing its own confirmation (see section 8).

## 7. The source term (experiment E)

*Runs after Phase 3; to be filled in.*

## Files

- Outputs (not in Git): `outputs/wzzt_evidence_20261124/phase2/`: `ceiling.jsonl`, `ladder.jsonl`, `bank.jsonl`, `judge.jsonl`, `root.jsonl`; earlier runs in `run1/` (registered certificate) and `run2/` (D1 before the fix).
- Code: `run_phase2.py`, `analyse_phase2.py`, `judge_phase2.py`, `test_phase2.py` (14 tests).
