# Overall procedure check and plan (2026-11-25; revised the same day after an audit)

Sections 1 to 3 are the three questions. Section 4 lists what was wrong in the first version and is now corrected.
Nothing here is committed.

## 1. Cases

### Your old cases

The four synthetic feeders used throughout are `paper15`, `soumalas11`, `flynn16` and `pengwah18` (15, 11, 16, 18 meters).
There is no `flynn17` in either local checkout, in file names or in text (`flynn16` is the nearest). The branch files of
all four are identical in `repo` and `Topo-English`. Feeder 65037 and the source-term code of the October work are not on
this machine, so the cases were never compared with those.

**Impedance scale.** The case metadata record an original impedance scale of 4.5 to 5.2; the files use 1 (recorded
decision of 2026-11-19: feeders of at most 186 m, voltage spread about 2%). It is not known which scale the October runs
used. It matters for the AC question (section 4).

### What exists and what it covers

| Set | Inputs | Trees | Independent of development? |
| --- | --- | --- | --- |
| Synthetic, all seeds (33-96 earlier; confirmation 161-184) | 192 in the confirmation | 4, the same in every set | new loads and noise only |
| Real `eulv` (33 feeders, 8-25 meters, days 0-11) | 396 | 33 | development and Phase 3 share the feeders |
| Real 16-day (`eulv_16d`) | 33 | 33, all 33 identical to development trees | no |
| Real large (`eulv_large`) | 46 | 23 networks | 36% of meters are development meters; 13 inputs (7 networks) have none |
| Root-reading scenarios (experiment E) | 130 x 5 | as development | run, not analysed |

So there is **no independent topology set** in the study apart from the 7 clean networks (13 inputs), and no set has a
topology that a method was not developed on otherwise.

### Source term: not covered, in either direction

Every load is metered; the PCC records only voltage. All cases are "without source". The free-source variant and
experiment E test an unknown **root voltage**, which equals an unmetered load only at the feeder head.

**To add: paired twins of each input, with and without one unmetered load** (AC power flow; PCC P and Q with 0.5% error;
the methods receive PCC minus the meters minus estimated losses). Grid to decide: location (root, middle, terminal),
size (2, 5, 10% of mean load), profile (intermittent, constant, PV-shaped). Losses are second order in the load, so AC
data are needed here.

### Update, same day: the case library is built

`experiments/case_library_20261125/DESIGN.md`: the four old feeders as star and street twins, random street feeders of
10 to 60 meters, real customers' loads, AC physics, and every case with and without one unmetered load at the head,
middle or end. 288 development cases and 240 sealed held-out cases (60 new feeders). The generator was fixed and committed
before any identifier ran on it. The old cases had no line with more than half of the meters downstream (real feeders
of 13-25 connections: median 5), so they could not test the root lines.
This answers decisions 1 (no impedance scale: physical lengths and conductors; linear misfit at most 9% of the meter
noise in the largest case), 3 (a default source grid, still the user's to change) and 4 (built) of section 5.

### The gap that matters most: topology diversity

New seeds do not give new trees. Two sources to add: a random feeder generator (topology, segment lengths and cable
types drawn from the EU LV set, 10 to 60 meters), kept in a separate set never used for tuning; and a second real
dataset, if one can be found.

## 2. Identification

### What is established, and on what

- WZZT-R beats the original WZZT on new data from the same feeders and topologies (Phase 3). It beats NJ, RNJ and RG on
  root lines (Amendment D), on a set that shares all 33 feeders and trees with development: call that a replication
  on new days, not an independent confirmation.
- The certificate scales: every node MIQP proved optimal up to 77 meters, at most 23 s.
- Evidence margins rank lines by truth on the development data (67% true below 2, 100% at 50 or more).
- WZZT-C is within 2% of WZZT-R on the development inputs. Its confirmation is running.

### What is not established

- **That the exact MIQP finds better trees.** In WZZT-C's root step it improved on a plain local search in 1 of 127
  development inputs, by 1.7 half-BIC units (below the threshold m). Phase 1 found WZZT-R's pool search replaceable by
  heuristics as well. The MIQP's demonstrated role is the proof, not the finding.
- **That WZZT-C has no greedy step.** Its repair adds the best line and keeps it (used on 13 of 130 development inputs;
  pair moves once; the re-selection changed the tree on 6).
- **That "certified" compares with WZZT-R.** WZZT-C's certificate says no single compatible line pays; WZZT-R's says no
  family in its finite pool scores better. They are different statements and should not share a column.
- **That WZZT-C is 14 times faster.** The runs differed in machine load and in solver settings, and WZZT-R also
  confirms its certificate with a cold start, which WZZT-C does not.

### Which way for the paper

**Recommendation: WZZT-C as the framework, with the MIQP claimed as a proof and a measure of belief, not as a better
search.** Headline claims, in order of evidence: (1) deciding on the whole voltage map rather than on the entries of a
fitted matrix keeps the short lines near the root; (2) every returned tree has a certificate and every line an
evidence margin that ranks how far to trust it; (3) both scale to 77 meters. WZZT-R is the reference. If WZZT-C fails
its test, the paper keeps WZZT-R. The decision on whether the exact MIQP helps finding is made by experiment I2.

### Experiments, in order

| # | Experiment | Purpose |
| --- | --- | --- |
| I1 | WZZT-C confirmation (running): synthetic set, the 13 clean real inputs | non-inferiority to WZZT-R, better than baselines; "new data on known topologies" |
| I2 | **Ablation: same candidates, MIQP against a heuristic search** at steps 3 and 5, on development inputs | does exactness change trees, or only prove them |
| I3 | **New topologies:** a random-feeder set (30 or more trees, 10-60 meters, drops 0.5-8%), generated once, held out | generalisation, the claim the present sets cannot support |
| I4 | **Belief:** margin against truth on I3, with the budget "lines to check in the field to catch 90% of errors", against bootstrap support of RNJ | is the margin useful, not just ordered |
| I5 | Source term: with/without twins on I3 | source-aware against source-blind, and locating the source |
| I6 | Root scenarios (experiment E), history length: analyse the runs already done | robustness |

## 3. Probing

### Is the greedy still good? Unknown, and the first version of this plan guessed wrongly.

"Greedy" here: each hour the selector buys the one demand-response block that most lowers the risk of naming the wrong
topology for its price, and does not plan ahead. All its evidence comes from banks of the original WZZT, with the safety
test T1 failed and no benefit seen on real feeders.

I had written that a better passive stage leaves fewer and harder ambiguous inputs. The data say otherwise (Phase 3,
same 520 inputs): inputs the gate would allow to be probed (truth in the bank, leader ambiguous, closure finished) rise
from 89 with the original bank to 120 with WZZT-R's (real 32 to 47, short segment 38 to 41, meter stress 19 to 32). The
dangerous inputs (truth missing, closure finished, so the gate would be open) are 11 against 12. So the opportunity
grows by 35% and the risk does not shrink. Whether the greedy uses it has not been tested.

### Candidate pool

- *Current:* the identifier's output families plus the BIC one-edit closure (margin 6, up to 8 children). Truth in the
  bank, WZZT-R: 106 of 328 real, 94 of 96, 77 of 96.
- *Proposed:* from WZZT-C, fix lines with margin >= m and take every combination of the uncertain lines' alternatives
  within 6 units, then the same closure. Untested. Advantages to verify: smaller pool, the same parameter as the
  block rule, probes aimed at exactly the uncertain lines. Risk to check: a line fixed by a wrong margin (16% to 18% of
  lines with margin 4.6 to 10, and 6% of those with 20 to 50, were not true on the development data) removes the truth from the pool for good.

### Experiments, in order

| # | Experiment | Purpose |
| --- | --- | --- |
| P0 | Pools on the same 520 inputs: current closure, evidence-based, union (truth in pool, size, closure finished, probe-eligible inputs, truth-missing-with-open-gate) | choose the pool on both opportunity and risk |
| P1 | Locked selector re-run with the chosen pool: greedy, passive, random-matched, certified; with truth-removed copies | is the lock's premise still true |
| P2 | Stage M as registered (v4): fresh seeds 129-160; four synthetic trees only, so the same caveat as I1 | main probing result |
| P3 | Stage S sensitivities; impedance scale 2 (registered) and the original 4.5-5.2 scale | robustness |

## 4. Corrections to the first version of this plan and to earlier statements

1. The large real inputs were called fresh; 36% of their meters are development meters (protocol correction).
2. The "192 fresh synthetic inputs" are new data on the same four trees, not new topologies.
3. Amendment D was called confirmatory; its set has the same 33 feeders and true trees as development.
4. "No greedy step", "the MIQP gives the method a real search role" and "the exact root search does real work" are not
   supported (section 2).
5. "Certified" columns mixed two different certificates; "14 times faster" was not a controlled comparison.
6. "A better passive stage leaves fewer ambiguous inputs" is contradicted by the data (section 3).
7. The AC answer (linear misfit at most 1% of meter noise) holds at impedance scale 1, the scale the files use. At the
   metadata's original scale (4.5 to 5.2) the largest drop is 8.6 to 12.7% and the misfit 9 to 20% of the noise
   (one seed, four feeders), up to about 40 half-BIC units at N = 2,592, more than m = 4.6. The decision to keep the
   linear model was made on the scale-1 numbers; it should be re-made if the original scale is the one to use.
8. D4: the cause of the two inputs on which the Gurobi and SciPy runs differ is not established.

## 5. Your decisions

1. Which impedance scale do the old cases use: 1 (the files) or the original 4.5 to 5.2? If the latter, decide again
   whether the identifiers keep the linear model.
2. Is `flynn17` really `flynn16`, or a case that is not on this machine?
3. The source-case grid of section 1.
4. Whether to build the random-feeder set (I3), the largest gap in the study.
