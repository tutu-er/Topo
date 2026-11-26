# Identification: which way for the paper, and the experiments (2026-11-25, development stage)

All numbers here are **development** results, most of them partial (the runs go on); the sealed test set has not been
opened by any identifier. Numbers are percentages of the *identifiable* true lines (lines of at least 3 milliohm).
"Supported" lines are those a BIC judge finds in the data; this label has the same form as our score and favours it, so
recall, precision and F1 are always shown next to it.

## 1. The way: WZZT-C with the located source (S2), exact and certified

| | Claim | Evidence so far (development) |
| --- | --- | --- |
| N1 | Deciding on the whole voltage map instead of a fitted distance matrix keeps the short root-side lines | random feeders, no source, 25 meters and fewer (7 cases): F1 74 against 63 (RNJ best tolerance in hindsight) and 59 (NJ); 15 of 15 supported root-side lines against 13 and 13. Medium and large random feeders, source-blind (6 and 5 cases): F1 72 and 69 against 45 and 44 (RNJ), 43 and 39 (NJ); 14 of 14 and 29 of 29 supported root-side lines against 7 and 16 |
| N2 | **An unmetered load is one more leaf with a known power and no voltage; the same exact MIQP chooses tree and path together** | with a source, S2 against the source-blind method: random F1 75 against 69, street 84 against 79; source placed correctly among the returned lines 101 of 112; false alarms without a source 0 of 37. Distance methods cannot place a leaf that has no voltage row |
| N3 | Every tree comes with a certificate (S1: node MIQP; S2: exact enumeration of one more line, `source_cert.py`), and every line with an evidence margin | S1 certified as in the old study; S2 certificate in ablation B (running) |
| N4 | Margins rank how far to trust a line | ablation C (running): against RNJ/NJ bootstrap support |

### Ablation, complete (240 development cases, 25 meters and fewer, no errors)

The interim reading (first 58 cases) was too favourable and is replaced by this one.

- **A. Search, no source (60 cases, the same candidate lines, half-BIC units above the exact optimum).** Local search (six starts) is worse in 2 of 60 (at
  most 3.3), plain greedy in 14 of 60 (up to 15.9). Recall, precision and F1 of the resulting trees: exact 76.3, 98.8, 86.1; local
  75.9, 98.5, 85.8; greedy 75.9, 96.0, 84.8. Pool MIQP proved optimal in 58 of 60.
- **A. Search, S2 (tree and source path together, 240 cases).** Local search is worse in 8 (at most 1.7); greedy in 64 (up to 100, when
  it fixes a wrong path early). **The MIQP was not warm-started here and proved the optimum in only 222 of 240** (time limit 240 s, 6.3 s
  median, 29 s mean; local 13 s median, 17 s mean); in the other 18 local search found better trees (hence the negative mean
  gap of local, -0.97). So the exact MIQP is **not** faster and not always proved at this size; the method itself
  warm-starts it. The earlier "2.6 s on every pool" was true only of the smallest cases.
- **B. S2 certificate by enumeration.** Certified in 229 of 240; in 9 cases one more line would have improved the tree, so
  the completion step (`source_cert.complete`) belongs in the method; 3 cases had a node with more than 12 child blocks and cannot be certified by
  enumeration.
- **C. Belief (share of true lines found at 95% precision, threshold in hindsight for each method).**

  | | WZZT-C margin | RNJ bootstrap support | NJ bootstrap support |
  | --- | --- | --- | --- |
  | Star | 100.0 | 98.6 | 97.1 |
  | Street | 71.7 | **77.5** | 77.2 |
  | Random small | **61.0** | 34.1 | 24.4 |
  | Random medium | **58.4** | 28.6 | 16.9 |

  On the random feeders the evidence margin finds about twice as many true lines at 95% precision as bootstrapped RNJ; on the
  binary street feeders bootstrapped RNJ is better. Ranking of true over false lines: 0.97 for the margin (5 false lines, little power), 0.94 for
  RNJ and 0.92 for NJ (161 and 232 false lines).

**Conclusion for the paper.** The MIQP is not claimed as a better search: against a multi-start local search it gains at most 3.3 units, gives
a different tree in 3 of 60 no-source cases and leaves F1 unchanged (86.1 against 85.8). Plain greedy (add the best line, never
revise) is clearly insufficient, so a method that only adds lines is not enough. What is claimed is (i) the
formulation that takes the source as a leaf, (ii) the certificate with completion, (iii) the evidence margin, whose advantage over bootstrapped
RNJ and NJ is large on random feeders and reversed on full binary trees.

### WZZT-R against WZZT-C on the development library (60 no-source cases, at most 25 meters; complete)

Same tree on 56 of 60. Of the four that differ, WZZT-C's tree scores better on two (6.2 and 5.5 half-BIC units, i.e.
above m = 4.6) and WZZT-R's on two (1.9 and 0.3). Pooled: recall 76.3 against 75.6, precision 98.5 against 98.3, F1 86.0
against 85.4, supported lines 400 against 397 of 404. Median time 2.8 s against 165 s, on a machine shared with other
runs, so this is **not** a controlled comparison. WZZT-R was not run on cases above 25 meters.

### Why RNJ is good on the street cases, and what can be absorbed (price study, development inputs, no heuristic added)

`price_study.py`: the same candidate lines for every setting; the tree is the best of six local-search starts of
0.5 SSE / sigma^2 + sum of line prices. The method's price is BIC, 0.5 p_s ln N per line.

- **Cause.** RNJ with tolerance 0 resolves every node into two-way nodes. On the street feeders 88% of the nodes are two-way (real feeders:
  87% overall, 79 to 87% by size; the old star cases 7%), and 56% of the extra lines RNJ returns over WZZT-C are true; on random small 27%, medium
  9%, star 0%. It is not a better search: it is a smaller price for a line.
- **Not a property of real feeders.** On the 66 real development inputs (EU LV, days 0 and 6) RNJ gets F1 41 at its best tolerance and 39 at
  tolerance 0, against 62.5 for the BIC price of our method.
- **One number, from real-feeder statistics.** A geometric prior on the number of children above two makes every line worth lambda = ln(1/rho)
  in log prior odds, where rho is the odds of a node having more than two children: rho = 0.14 over all 84 real feeders (0.15 to 0.22 by
  size), so lambda = 1.5 to 2.0 half-BIC units. With price = BIC - 2 (F1 with BIC in brackets): star 100 (100), street 85.4 (83.2), random small
  76.5 (74.6), medium 73.3 (69.9), real 66.7 (62.5). No loss on any set. The sum over nodes of (children - 2) is linear in the number of lines, so the pool
  MIQP, the node certificate and the enumeration certificate are unchanged except for alpha.
- **The real-feeder statistic is not independent of the cases:** the 84 feeders include the 33 development ones, and the random feeders were calibrated
  on the same statistics. Estimated gains may shrink on other networks.
- **Full marginal likelihood with a physical prior on r and x** (price 1.6 half-BIC units, the spread of real section resistances) is not adopted: star
  precision falls to 80, F1 medium 68 (BIC 70), and the gain on street (93) is at the cost of false lines on star.
- **The price as an operating point (precision-recall curve over a factor of BIC).** Our curve dominates RNJ's and NJ's on every set: for
  example street, factor 0.1: recall 93 at precision 94.5 (RNJ tolerance 0: 94 at 83); real, factor 0.25: recall 65 at precision 74 (RNJ at its best tolerance
  0.04 gets recall 49 at precision 35). The curve is the paper's way to show this; the single reported setting stays BIC with lambda.

### Source-blind comparison, complete (288 development cases, 0 errors)

| No unmetered load | Cases | WZZT-C F1 (precision) | RNJ best tolerance | NJ best tolerance |
| --- | --- | --- | --- | --- |
| Star | 24 | 100 (100) | 95.8 | 95.5 |
| Street | 24 | 83.2 (99) | **88.1** | 86.6 |
| Random small | 6 | 74.6 (96) | 66.0 | 61.4 |
| Random medium (16-25) | 6 | 72.0 (94) | 45.4 | 42.9 |
| Random large (26-40) | 6 | 66.1 (96) | 43.1 | 35.5 |
| Random huge (41-59) | 6 | 63.9 (**83**) | 29.1 | 24.3 |

- **Two limits found here.** (i) Precision falls with size: 83% at 41-59 meters (3.3 false lines per case), lower again with an ignored source
  (76% at 41-59 meters). The margins have not been checked at this size. (ii) **Time:** median 2 s at 10-15 meters, but 219 s at 16-25, 509 s at
  26-40 and 843 s at 41-59 meters, because pool and root MIQPs reach their 240 s limits (all runs share the machine with other jobs, so
  these are upper bounds). The "2.8 s" of the WZZT-R comparison describes cases up to 18 meters only and must not be quoted for the method.
- All trees were certified (S1 sense) except 2 of the 18 huge cases with a source.

### Development verification of the final method, first run (2026-11-25; a regression was found)

72 no-source development cases, `final_blind.jsonl`, with the structural prior (bonus, lam = 1.96) and without it, in the same pipeline
(0 errors):

| | Bonus | Plain | WZZT-C |
| --- | --- | --- | --- |
| Star (24) | 100 | 100 | 100 |
| Street (24) | **85.6** | 83.2 | 83.2 |
| Random small (6) | **76.5** | 74.6 | 74.6 |
| Random medium (6) | **73.3** (precision 88.9) | 72.0 (93.8) | 72.0 |
| Random large (6) | **69.6** | 67.0 | 66.1 |
| Random huge (6) | 54.0 (precision **56.4**) | 51.8 (54.5) | **63.9** (83.5) |

- **Bonus, the user's condition (no loss at 41-59 meters):** against plain BIC in the same pipeline the bonus gains on the mean at every class (F1 per case
  +2.5 street, +1.8 small, +1.9 medium, +2.4 large, +2.3 huge; huge better 4, worse 2; medium better 2, worse 2, with precision
  93.8 to 88.9). So the condition as written holds at the huge size for the mean, not case by case.
- **Regression found:** at 41-59 meters the first version of the final pipeline was far below WZZT-C (F1 54 against 64, 13 false lines per case
  against 3.3). Cause: at 58 and 60 meters the warm-started pool MIQP found no solution in 120 s and the fallback was a weak
  local search (the true tree scored 400 to 1,100 half-BIC units better than the returned one). On the other four large
  cases the two pipelines' scores differ by at most 17 half-BIC units. `final.py` now runs a cold, longer MIQP beyond 35 meters and keeps the
  better of the MIQP's and the local search's trees. **This has not been verified yet** (`final_blind_v2.jsonl` is running).
- Time for the first version: 136 s median at 16-25 meters, 342 s at 26-40, 544 s at 41-59, so the pipeline is not faster than WZZT-C at
  small sizes (5 against 3 s) and only about a third faster at 26-59.

### S2 on 26 to 60 meters (development, 48 cases, 0 errors; `source.fit`, without the bonus and without the completion step)

| | S2 | Blind WZZT-C | RNJ best tolerance | NJ best tolerance |
| --- | --- | --- | --- | --- |
| 26-40 meters, source (18): F1 (precision) | **65.2** (95) | 58.0 (90) | 44.4 | 39.0 |
| 41-59 meters, source (18) | **62.4** (84) | 55.2 (76) | 27.7 | 24.0 |
| 26-40 meters, no source (6) | 65.4 | 66.1 | 44.0 | 38.4 |
| 41-59 meters, no source (6) | 61.9 (84) | 63.9 (84) | 26.7 | 23.9 |
| Root-side supported lines, source | 92/93 and 86/102 | 79/93 and 74/102 | 54/93 and 24/102 | 33/93 and 13/102 |

- Source: no false alarm in 12 no-source cases (largest evidence 2.1); evidence above 4.6 in 30 of 36 source cases; the exact place is right in 26
  of 36; the true place is within 4.6 of the best in **36 of 36**.
- **Not proved:** the S2 pool MIQP reached optimality in 8 of 48 cases only (the 240 s limit), so at these sizes S2 is a good search and not a
  proved optimum; S1's tree was certified in 47 of 48 (one huge source case not). Time: median 761 to 1,087 s per case (shared machine).
- Precision falls from 95% at 26-40 meters to 84% at 41-59.

### Repaired pipeline, second run (v2: cold MIQP beyond 35 meters, better of two searches; 12 no-source cases, 0 errors)

| | Bonus | Plain | WZZT-C |
| --- | --- | --- | --- |
| Large (6): F1 (precision) | **69.6** (96) | 67.6 (95) | 66.1 (96) |
| Huge (6): F1 (precision) | 62.6 (**73**) | 65.2 (84) | 63.9 (84) |

- The repair restores the plain pipeline (F1 65.2 against 63.9 for WZZT-C; it was 51.8 in v1).
- **The user's condition fails in this run:** at 41-59 meters the bonus loses 2.6 F1 points and 11 points of precision (3.3 false lines per case to
  6.7), better 3, worse 2. **The loss is one feeder** (`dev_huge_004`, 60 meters): under the bonus's own objective its returned tree is 763 half-BIC
  units worse than the plain tree, so the search failed there; the other five gain or tie. A price continuation (plain price first, then the bonus starting
  from that tree) was added and is being rerun (v3). Until it is checked the bonus is not adopted.
- **Worked example `dev_medium_003` (25 meters, urban, 14 kW mean, largest drop 1.6%).** Not a search failure and not a meter-error effect (removing the P/Q
  gain and noise changes nothing): the truth scores 31 half-BIC units worse than the returned tree because 7 of its 11 lines are below the price
  (the truth pays for 11 lines, the returned tree for 5). The data cannot show these lines; the returned lines include
  two or three false ones that fit almost as well.

### Repaired pipeline with the price continuation (v3, 6 no-source cases of 41-59 meters): the bonus is adopted

| | F1 | Precision | Root-side lines | False lines per case |
| --- | --- | --- | --- | --- |
| Bonus (v3) | **69.6** | 88 | 33/34 | 2.5 |
| Plain (v3) | 65.0 | 83 | 32/34 | 3.5 |
| Bonus without the continuation (v2) | 62.6 | 73 | 30/34 | 6.7 |
| WZZT-C | 63.9 | 84 | 32/34 | 3.3 |

Per feeder the bonus is better than plain on 4, equal on 1 and worse on 1 (by 1.3 F1 points): **no loss at 41-59 meters in the mean or in precision, so the user's
condition of 2026-11-25 is met and the bonus (lam = 1.96) is in the method** (`Settings.lam`, with `continuation` beyond 35 meters). It rests on
six feeders, and the plain run differs from v2 by 0.2 F1 points because the MIQPs stop at their time limits. **Time: 1,913 s median (2,205 s maximum)
at 41-59 meters**, about twice the plain pipeline, since the continuation solves the tree twice. Not solved.

### Robustness to less and poorer data (I5, complete: 840 runs, 0 errors; `robustness.py`)

Development twins of at most 25 meters (24 star, 24 street, 12 random feeders), without a source and with one at the end; one change at a time
from the baseline (hourly, 6 + 2 days, voltage 0.05%, P/Q 0.3% gain and 0.5% noise). The final method ran as it will be frozen: the rows
started after the v2 commit, and the later two changes act only beyond 35 meters (continuation) or on a solver failure (fallback, not triggered).
F1 against the identifiable truth (all segments of at least 3 milliohm), pooled over cases. **RNJ and NJ get their best tolerance in hindsight
in every cell** (optimistic for them).

| Condition | Star: S2 / RNJ | Street, no source: S2 / RNJ | Street, end source: S2 / RNJ | Random, no source: S2 / RNJ | Random, end source: S2 / RNJ |
| --- | --- | --- | --- | --- | --- |
| Baseline | **100** / 95.5 | 85.6 / **88.8** | 86.2 / **87.8** | **72.7** / 49.2 | **77.1** / 49.8 |
| 3 + 1 days | **99.6** / 88.0 | **81.8** / 79.9 | **83.2** / 80.3 | **65.6** / 35.3 | **68.4** / 41.1 |
| 1.5 + 0.5 days | **94.2** / 65.8 | **76.4** / 70.4 | **77.8** / 71.4 | **58.0** / 20.4 | **58.4** / 23.8 |
| Quarter-hourly, 2 days | **100** / 98.6 | 87.5 / **89.8** | **90.1** / 89.8 | **73.9** / 57.3 | **74.6** / 58.7 |
| Voltage 0.10% | **98.5** / 79.2 | 75.2 / **77.9** | 77.7 / **78.1** | **55.8** / 27.5 | **59.1** / 26.7 |
| Voltage 0.20% | **76.3** / 50.3 | **58.5** / 57.1 | **60.7** / 57.5 | **30.9** / 15.1 | **35.3** / 18.5 |
| P/Q 1.0% (PCC too) | **100** / 94.5 | 85.6 / **89.1** | 86.4 / **87.8** | **73.7** / 49.4 | **76.5** / 51.9 |

(Star column: the no-source cases. With a source at the end, star S2 is 97 to 100 in every row except 79.7 at 0.20% voltage error; RNJ 66 to 98, and 52.)

- **Our advantage grows as the data get worse.** With 2 days of hourly data the star F1 is 94 against 66, random 58 against 20. On the street
  feeders RNJ, with its best tolerance, leads by 0.4 to 3.5 points when the data are as informative as the baseline (baseline, quarter-hourly,
  1% P/Q, 0.10% voltage; the one exception is quarter-hourly with a source, +0.3 for us). We lead by 1.4 to 6.4 points with 4 days or fewer
  and at 0.20% voltage error.
- **Precision of S2:** at least 92% on star and street in every condition, and at least 94% except at 0.20% voltage error; 86-93% on random feeders,
  and 74-77% at 0.20% voltage error. When information is lost the method drops lines; it rarely invents them.
- **Voltage accuracy is what matters.** 1% P/Q error changes nothing. Doubling the voltage error costs 2 points on star, 8-10 on street and
  17-18 on random; quadrupling it costs 20 to 42. Quarter-hourly readings over 2 days are as good as hourly ones over 8.
- **Source (end):** placed within the 4.6-unit set in 418 of 420 runs (the two misses: one street run at 1% P/Q, one random run with 2 days).
  Evidence above 4.6 in 403 of 420; it falls at 0.20% voltage error (51 of 60). **False alarms without a source: 0 of 420.**
- **S2 against S1:** with a source, S2 is better in 17 of 21 cells (by up to 9.5 points), equal in 3 (star at 100) and worse in one (by 0.7).
  Without a source the two agree within 0.4 points or S2 is better (up to 3.6 on star).

### Worked examples (case study, 6 development cases; `case_study.py`, figure `outputs/case_library_20261125/case_study.png`)

Chosen to show both strengths and the failure mode: three street twins of the old feeders, two random feeders where we do well, and the
worst case of the development library. Same method version as the robustness study (the cases have at most 25 meters).

| Case | Meters | Source | True lines | Ours: found / false | RNJ 0.08: found / false | Root-side lines, ours / RNJ |
| --- | --- | --- | --- | --- | --- | --- |
| paper15, street, load 1 | 15 | end | 13 | 9 / 0 | 8 / 0 | 4 / 3 of 5 |
| flynn16, street, load 2 | 16 | none | 13 | 11 / 0 | 10 / 1 | 1 / 0 of 1 |
| pengwah18, street, load 3 | 18 | end | 13 | 11 / 0 | 9 / 2 | 5 / 3 of 5 |
| dev_small_002 | 14 | end | 9 | 7 / 0 | 6 / 1 | 5 / 3 of 5 |
| dev_medium_005 | 16 | end | 11 | 7 / 0 | 7 / 7 | 2 / 2 of 2 |
| dev_medium_003 (worst case) | 25 | none | 11 | 2 / 3 | 2 / 16 | 2 / 2 of 3 |
| **Total** | | | **70** | **47 / 3** | **42 / 27** | **19 / 13 of 21** |

- **What the examples show.** On the street twins, which favour RNJ on average, our method finds more lines and none false, and it finds the
  lines near the transformer (root side) that a utility needs to map the trunk: 19 of 21 against 13 (RNJ misses 8 root-side lines, we miss 2).
  RNJ's false lines are mostly small wrong groups (18 of its 27 are pairs or triples) that cut across the true groups.
- **The source path is right in all four examples with a source:** every line of the returned path is on the true path (29 of 29, drawn in
  green), and the place is the smallest returned line on it. The two examples without a source have evidence below zero (no false alarm).
- **The failure (`dev_medium_003`)** is low information, not search: the true tree scores 31 half-BIC units worse than the returned one, because 7 of
  its 11 lines are below the price (see above). Our method returns 5 lines, 3 of them false; RNJ returns 18, 16 of them false.

### The final method without the PCC on all 72 no-source development cases (repeated inside the likely-lines check)

Per-case F1 against the identifiable truth, the protocol's own statistic (one case per feeder here), and precision over all lines:

| No unmetered load | Final method (bonus, v3): F1 (precision) | WZZT-C | RNJ at its registered tolerance 0.04 |
| --- | --- | --- | --- |
| Star (24) | 100 (100) | 100 | 69.0 |
| Street (24) | 85.2 (99) | 82.7 | **87.0** |
| Random small (6) | **76.1** (96) | 74.4 | 62.2 |
| Random medium (6) | **70.5** (89) | 68.6 | 44.0 |
| Random large (6) | **68.7** (96) | 65.2 | 41.8 |
| Random huge (6) | **69.9** (88) | 63.8 | 29.4 |

- 66 of the 72 trees are identical to the first run (all but the six of 41-59 meters, where v2 and v3 changed the pipeline).
- **Run to run:** against the stored v3 run, 4 of the 6 huge trees are identical; one has one more true line (F1 70.8 to 73.5) and
  one has one line swapped (F1 unchanged). The searches stop at wall-clock limits, so beyond 35 meters a run on a busier machine can
  end elsewhere. The sealed test is one run, and this variation is part of its result.
- RNJ's tolerance 0.04 is the one chosen on the random feeders, the only kind in the sealed set; on star and street feeders a
  higher tolerance suits it better (section 3 of PROTOCOL_ID.md).

### The final method with the PCC on all 288 development twins (after the completion fix; `final_source.jsonl`, 0 errors)

The 39 twins affected by the completion defect were rerun with the fix (PROTOCOL_ID.md section 7). The other 249 twins take the
same code path and keep their first-run results. Per-case F1, with precision in brackets:

| With an unmetered load | S2, final | S1, final | S2 before (no bonus, no completion) | Final method without the PCC | RNJ, projected, 0.08 | Place covered | Time, median / 90% / max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Star (72) | 100 (100) | 100 | 100 | 94.7 | 79.8 | 71 of 72 | 8 / 12 / 15 s |
| Street (72) | **85.8** (99) | 84.1 | 84.0 | 76.1 | 84.1 | 71 of 72 | 10 / 19 / 31 s |
| Random small (18) | **77.7** (96) | 76.1 | 76.1 | 69.9 | 62.9 | 18 of 18 | 7 / 28 / 38 s |
| Random medium (18) | **70.8** (86) | 67.5 | 68.8 | 59.3 | 42.9 | 18 of 18 | 260 / 561 / 761 s |
| Random large (18) | **71.5** (96) | 65.9 | 64.5 | 57.2 | 41.8 | 18 of 18 | 651 / 1,326 / 1,621 s |
| Random huge (18) | **69.5** (85) | 65.4 | 62.5 | 54.3 | 26.3 | 18 of 18 | 2,351 / 2,862 / 2,880 s |

- **Without an unmetered load** (72 twins), S2 is as good as the blind tree or better: star 100, street 85.2, random 76.1, 68.4,
  70.3 and 68.6 by size. The source evidence is above 4.6 on 0 of the 72 (false alarms). With a source, it is above 4.6 on 53 of 72
  at the head, 66 of 72 in the middle and 71 of 72 at the end.
- **Certificates:** all 288 twins are certified, and no completion step is stuck. Before the fix, 31 cases of 16 meters or more
  ended uncertified after 10 rounds. Completion rounds per case:
  - huge: 1 to 5;
  - large: 1 to 5, and 10 of 24 need one round;
  - medium: 1 to 3;
  - small, star and street: always 1.
- **What the fix changed** (35 twins with results before and after; 4 twins of `dev_huge_004` had not finished before):
  - S2 F1 rose from 69.7 to 74.3: better on 27 twins, equal on 3, worse on 5.
  - Certified twins went from 4 to 35.
  - The median time fell from 2,041 s to 1,132 s, and the maximum from 4,341 s to 2,878 s.
  - S1 runs the same code as before and is identical on 28 of the 35 twins; the other 7 are wall-clock differences beyond 35
    meters.
  - The fix was kept before the rerun was read (PROTOCOL_ID.md section 7).
- **Sealed-test duration:** a huge case needs about 2,350 s with the PCC plus about 1,900 s blind, so 70 minutes or more. With
  the large and medium cases, the estimate is about 110 CPU hours, or roughly 10 hours on 12 workers.

### Likely lines (development check of the secondary output; `identification/likely.py`, `likely_dev.py`; 0 errors)

The candidate lines left out of a returned tree whose evidence misses the line price by less than 4.6 half-BIC units (the line swapped
into the tree, the tree lines it crosses removed). Computed after the tree; it never changes it.

| Feeders | Blind trees (72 no-source cases): listed per case, share true | S1 trees (all 288 twins, after the completion fix): listed per case, share true |
| --- | --- | --- |
| Star | 1.1, 0% (0/26) | 1.4, 0% (0/131) |
| Street | 2.0, 81% (38/47) | 2.1, 82% (163/199) |
| Random small | 3.5, 33% (7/21) | 3.4, 29% (24/82) |
| Random medium | 6.0, 31% (11/36) | 5.5, 38% (50/132) |
| Random large | 6.3, 66% (25/38) | 6.3, 64% (96/151) |
| Random huge | 8.0, 46% (22/48) | 8.8, 40% (84/211) |
| **All** | **3.0, 47.7% (103/216); holds 103 of the 259 missed true lines** | **3.1, 46.0% (417/906); holds 417 of 1,101** |

The S1 column first covered 277 twins: 2.8 lines per case, 47.5% true. The 40 twins that the completion rerun touched or that were
missing were recomputed on their final S1 trees.

- **Closeness to the price is what counts.** Within 2 units of the price, 66% of the listed lines are true on blind trees (25/38)
  and 71% on S1 trees (117/165). From 2 to 4.6 units: 44% (78/177) and 41% (298/724). Lines that would improve the score, where the
  search stopped short (huge feeders only), are 0 of 1 and 2 of 17 true.
- **The source does not matter:** 45 to 48% true at every source place (none, head, middle, end).
- **Star trees miss nothing**, so every listed line there is false (about one per case). The list reads "check these", not "these
  are there": on street feeders and large random feeders it holds many of the lines the tree misses.

## 2. Experiments, in order

| # | Experiment | State | What it decides |
| --- | --- | --- | --- |
| I1 | WZZT-C confirmation on the old inputs | **done: not confirmed** (C1a, C2a-c hold; C1b on the 13 clean inputs not shown, decided by one failed job; `wzzt_evidence_20261124/RESULTS_WZZT_C.md`) | non-inferiority to WZZT-R on the old cases |
| I2 | **Ablation A:** the same candidate lines, exact MIQP against local search and greedy; B: S2 certificate; C: belief | **complete** (240 cases), development, 25 meters and fewer | whether exactness changes the trees; whether the certificate is needed |
| I3 | Comparison on the development library: S2, S1, blind, RNJ, NJ, RG (each at all tolerances) with and without a source; n up to 59 | **complete** (288 blind; with the PCC 240 cases of at most 25 meters and 48 of 26-59) | the numbers in N1 and N2 |
| I4 | Size: 26 to 60 meters, time, certified share | **complete** (blind and S2; time 1,913 s median at 41-59 meters with the bonus) | the scaling claim |
| I5 | Resolution and data: hourly against quarter-hourly, history 2/4/8 days, meter classes | **complete** (840 runs; section above) | robustness |
| I6 | Source size: 2, 5, 10%; kind; place (in I3) | **complete** (in I3) | when is a source locatable |
| I7 | **Registered test on the sealed set** (`test`, 240 cases, 60 new feeders), opened once | not started | the paper's numbers |

### What I7 will contain (to be registered before the sealed set is opened)

- Frozen: S2 with the completion step (`complete` of `source_cert.py`), settings of WZZT-C, noise and grid as in
  `build.py`. The baselines: RNJ, NJ, RG at tolerances chosen on the development library, with source-projected input in
  the source-aware part, and their best tolerance in hindsight shown as a secondary, optimistic line.
- Primary endpoints, per case and then averaged by feeder (60 clusters, bootstrap over feeders):
  (a) F1 of S2 minus the best baseline, no source; (b) the same with the source; (c) share of cases with the source
  placed correctly among the returned lines; (d) false-alarm rate of the source evidence without a source.
- Secondary: root-side supported lines, exact trees, precision, recall, size classes, time, certified share, all three
  source places.
- What would change the plan: if ablation A shows no difference to local search, the paper's claim is the formulation and
  the certificate (N2, N3, N4), and the search is stated as "found by the same heuristic, then certified".

## 3. What is missing before I7

1. S2 with the completion step and the bonus (the final method) on all 288 development twins, with times: **done**.
   - A defect of the completion step was found: the MIQP stopped at its time limit and returned the tree without the paying line,
     so 31 of 67 cases of 16 meters or more never completed. It was fixed, and the 39 affected twins were rerun.
   - All 288 twins are now certified, with 0 errors (section "The final method with the PCC" above).
2. A fixed rule for the source evidence threshold: **fixed**, 4.6 half-BIC units = Bayes factor 100, as for lines.
3. The tolerance choice of the baselines on development: **filled** (PROTOCOL_ID.md section 3, from `compare_dev` / `compare_source`).
4. The protocol text: drafted (PROTOCOL_ID.md). **The user's go-ahead: given on 2026-11-25** ("do the first and second": adopt the bonus if it
   costs nothing at 41-59 meters, then freeze, register and run the sealed test once). Item 1 is done.
5. WZZT-R as a secondary comparator on the 60 no-source sealed cases: added after the WZZT-C confirmation ended "not confirmed"
   (PROTOCOL_ID.md sections 3 and 7). Dry run done: its trees are identical to the development WZZT-R run.
