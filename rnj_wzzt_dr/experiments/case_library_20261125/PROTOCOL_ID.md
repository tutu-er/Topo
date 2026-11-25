# Protocol: the sealed identification test (DRAFT, not registered, not run)

Status: draft of 2026-11-25. It becomes binding only when (i) the development verification of `final_dev.py` is finished,
(ii) the blanks marked [FILL] are filled from development data only, and (iii) the file is committed and pushed. Until
then nothing of the sealed set (`test`, 240 cases, 60 feeders) has been opened by any identifier.

## 1. Method, frozen

`identification/final.py` at the commit this protocol is registered in, default `Settings` (lam = 1.96, m = 4.6, k_max = 9, pool and
root MIQP 120 s, node MIQP 60 s, S2 MIQP 120 s, completion up to 10 rounds). It is not changed afterwards. The
structural prior is in the method only if the development verification shows no loss at 41-59 meters (the user's
condition of 2026-11-25); the check and the decision are recorded in section 7 of this file.

## 2. Cases

All 240 sealed cases (`manifest_test.json`: 60 random street feeders, 15 per size class; four twins each: no source, head,
middle, end). Readings by `build.to_input` with its defaults: hourly, 144 + 48 values, meter seed 0. Analyses follow the
feeder: 60 clusters.

## 3. Comparators

On the same readings, all on the P-only matrix of the same pipeline: RNJ, NJ and RG, with and without the source projected
out (the projected matrix for the source-aware comparisons). Tolerances are the development-best per builder over all
development cases of at least 10 meters: RNJ [FILL], NJ [FILL], RG [FILL]. Every tolerance in {0, 0.04, 0.08, 0.16, 0.24}
is also run and reported as secondary, and each builder's after-the-fact best. The final method without the structural prior
(`lam = 0`) is run on the no-source cases only, as a secondary ablation.

## 4. Failure rule

A method that crashes or returns nothing gets an empty tree for that case, for every metric. The log keeps the trace. A
case where the comparators crash counts as an empty tree for them. Nothing is removed.

## 5. Primary endpoints

Per case, the F1 of the returned inner lines against the identifiable truth (segments of at least 3 milliohm); differences
are paired per case, averaged within a feeder, then bootstrapped over feeders (10,000 resamples, 95% percentile interval,
two-sided level 5%). Pooled over the four size classes (equal weight per class).

- **H1, no source:** F1 of the final method without PCC minus RNJ (development-best tolerance), 60 no-source cases: the interval is
  above 0.
- **H2, with a source (180 cases):** F1 of S2 (PCC) minus RNJ on the projected matrix (development-best tolerance): the interval is above 0.
- **H3, source awareness (180 cases):** F1 of S2 minus F1 of the final method without PCC: the interval is above 0.
- **H4, source found (180 cases and 60 no-source cases):** the true place is within 4.6 half-BIC units of the best in at
  least 90% of the source cases (lower end of the cluster-bootstrap interval above 85%), and the share of no-source cases with
  source evidence above 4.6 is at most 5% (upper end of the interval at most 10%).

Holm correction over H1 to H4. The claim of the paper is "better than the distance methods, and better because it knows the
unmetered load" only if H1 to H3 all hold; H4 is the location claim.

## 6. Secondary

Root-side supported lines, precision, recall, exact trees, by size and by source place, size of the source and its kind; S1
against S2; the share of certified trees; time (median, 90th percentile, maximum) on an idle machine; every tolerance and the
after-the-fact best of each builder; the structural prior on and off (H1 cases). The margin's calibration: share of true
lines by margin class. No decision rests on them.

## 7. Log

(development verification and decisions are recorded here before registration)
