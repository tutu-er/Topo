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

## 2. Experiments, in order

| # | Experiment | State | What it decides |
| --- | --- | --- | --- |
| I1 | WZZT-C confirmation on the old inputs (running, unread) | running | non-inferiority to WZZT-R on the old cases |
| I2 | **Ablation A:** the same candidate lines, exact MIQP against local search and greedy; B: S2 certificate; C: belief | running, development, 25 meters and fewer | whether exactness changes the trees; whether the certificate is needed |
| I3 | Comparison on the development library: S2, S1, blind, RNJ, NJ, RG (each at all tolerances) with and without a source; n up to 59 | running | the numbers in N1 and N2 |
| I4 | Size: 26 to 60 meters, time, certified share | running (blind), S2 still to do | the scaling claim |
| I5 | Resolution and data: hourly against quarter-hourly, history 2/4/8 days, meter classes | not started | robustness |
| I6 | Source size: 2, 5, 10%; kind; place (in I3) | running | when is a source locatable |
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

1. S2 with the completion step, run on development (2 to 59 meters) with times.
2. A fixed rule for the source evidence threshold (4.6 half-BIC units = Bayes factor 100, as for lines).
3. The tolerance choice of the baselines on development.
4. The protocol text, and the user's go-ahead.
