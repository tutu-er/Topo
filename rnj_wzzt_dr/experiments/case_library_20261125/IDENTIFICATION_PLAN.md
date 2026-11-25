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

### Ablation, interim (58 of 240 cases, at most 15 meters; the run continues)

- **A. Search.** On the same candidate lines, local search with six starts found the exact optimum on all 14 no-source cases and
  on 56 of 58 S2 pools (the other two were worse by at most 0.5 half-BIC units); plain greedy was worse on 1 of 14 and 9 of 58
  (up to 7.7 units). The MIQP found and proved the optimum on every pool in 2.6 s on average, against 3.9 s for the local
  search. So exactness has **not** changed the returned trees against a good local search; it replaces a search with
  several starts by one proved answer. Greedy alone is not enough.
- **B. S2 certificate.** 58 of 58 trees certified, none needed a further line. No node had more than 12 child blocks.
- **C. Belief.** Bootstrap support of RNJ is a strong rival: its top half of lines (100% precision in hindsight) keeps
  66% recall; WZZT-C keeps 76% recall at 98% precision with no threshold, and 70% at 100% precision. The paper must
  show RNJ with bootstrap as a main baseline. Only 1 false line exists in the 59 WZZT-C lines, so the margin's ranking
  is untested yet.

### Source-aware run, complete for 25 meters and fewer (240 development cases, 0 errors)

| | S2 | S1 | WZZT-C blind | RNJ, best tolerance (hindsight) | NJ, best tolerance |
| --- | --- | --- | --- | --- | --- |
| Star, source (72): F1 | **100** | 100 | 93.9 | 95.9 | 95.0 |
| Street, source (72): F1 | 84.4 | 82.7 | 77.0 | **88.1** | 87.2 |
| Street, source: root-side supported | **226/228** | 219 | 197 | 213 | 195 |
| Random, source (36): F1 | **73.1** | 71.2 | 65.0 | 50.9 | 48.3 |
| Random, source: root-side supported | **81/81** | 78 | 76 | 54 | 56 |
| Random, no source (12): F1 | 71.9 | 71.9 | **72.9** | 50.7 | 47.8 |
| Precision of S2 | 99-100 (star, street), 94 (random) | | 88-96 | 83-92 (star, street), 47 (random) | 88-91, 42 |

- The source costs the blind method 6 to 8 F1 points (street 83 to 77, star 100 to 94, random 73 to 65), and S2 recovers almost all of it. Where
  there is no source S2 is within one line of the blind method on the random cases (71.9 against 72.9).
- Source existence: evidence above 4.6 in 0 of 60 no-source cases (largest 2.1) and in 160 of 180 source cases: 45 of 60 at
  2% of the load, 58 of 60 at 5%, 57 of 60 at 10%; by kind 57 of 60 (lamp), 43 of 60 (house) and 60 of 60 (block).
- Source place: exactly right among the returned lines in 163 of 180. **The true place is within 4.6 half-BIC units of the
  best in 178 of 180 cases, with 1.4 places in the set on average**: the set is the honest output; a single place is not.
- 13 of the 17 wrong places are 2% sources, 4 are 5% sources, none is 10%. Five were inspected: in `dev_medium_002_end` and
  `dev_medium_003_end` the returned tree itself has a false line on the source's path; in `soumalas11_street_l4_middle`,
  `dev_small_003_middle` and `dev_medium_006_middle` the place is one line off and the true one is within 3 half-BIC units
  of the best. The other 12 were not inspected one by one.

Two limits to state in the paper:
- **Full binary trees.** On the named street feeders (two lines out of every node) RNJ with tolerance 0 gets F1 90
  against S2's 84, because it always returns a full binary tree and the true trees are full binary. S2 does not add lines
  the data do not support. On random feeders, where nodes are wider, this advantage of RNJ is gone.
- **Exactness of the search** is not what we claim: ablation A finds no difference to local search so far. What the MIQP gives is the
  certificate, the evidence margin, and the one formulation that takes the source leaf.

### WZZT-R against WZZT-C on the development library (60 no-source cases, at most 25 meters; complete)

Same tree on 56 of 60. Of the four that differ, WZZT-C's tree scores better on two (6.2 and 5.5 half-BIC units, i.e.
above m = 4.6) and WZZT-R's on two (1.9 and 0.3). Pooled: recall 76.3 against 75.6, precision 98.5 against 98.3, F1 86.0
against 85.4, supported lines 400 against 397 of 404. Median time 2.8 s against 165 s, on a machine shared with other
runs, so this is **not** a controlled comparison. WZZT-R was not run on cases above 25 meters.

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
