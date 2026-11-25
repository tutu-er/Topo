# Identification: which way for the paper, and the experiments (2026-11-25, development stage)

All numbers here are **development** results, most of them partial (the runs go on); the sealed test set has not been
opened by any identifier. Numbers are percentages of the *identifiable* true lines (lines of at least 3 milliohm).
"Supported" lines are those a BIC judge finds in the data; this label has the same form as our score and favours it, so
recall, precision and F1 are always shown next to it.

## 1. The way: WZZT-C with the located source (S2), exact and certified

| | Claim | Evidence so far (development) |
| --- | --- | --- |
| N1 | Deciding on the whole voltage map instead of a fitted distance matrix keeps the short root-side lines | random feeders, no source, 25 meters and fewer: F1 74 against 63 (RNJ best tolerance in hindsight) and 59 (NJ); 15 of 15 supported root-side lines against 13 and 13. Medium and large random feeders, source-blind: F1 72 and 69 against 45 and 44 (RNJ), 43 and 39 (NJ); 14 of 14 and 29 of 29 supported root-side lines against 7 and 16 |
| N2 | **An unmetered load is one more leaf with a known power and no voltage; the same exact MIQP chooses tree and path together** | with a source, S2 against the source-blind method: random F1 75 against 69, street 84 against 79; source placed correctly among the returned lines 101 of 112; false alarms without a source 0 of 37. Distance methods cannot place a leaf that has no voltage row |
| N3 | Every tree comes with a certificate (S1: node MIQP; S2: exact enumeration of one more line, `source_cert.py`), and every line with an evidence margin | S1 certified as in the old study; S2 certificate in ablation B (running) |
| N4 | Margins rank how far to trust a line | ablation C (running): against RNJ/NJ bootstrap support |

Two limits to state in the paper:
- **Full binary trees.** On the named street feeders (two lines out of every node) RNJ with tolerance 0 gets F1 90
  against S2's 84, because it always returns a full binary tree and the true trees are full binary. S2 does not add lines
  the data do not support. On random feeders, where nodes are wider, this advantage of RNJ is gone.
- **Exactness of the search** is not what we claim unless ablation A shows it (see below). What the MIQP gives is the
  certificate, the evidence margin, and the one formulation that takes the source leaf.

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
