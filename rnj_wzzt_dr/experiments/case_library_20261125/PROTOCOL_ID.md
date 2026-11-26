# Protocol: the sealed identification test (DRAFT, not registered, not run)

Status: draft of 2026-11-25. It becomes binding only when (i) the development verification of `final_dev.py` is finished and
recorded in section 7, (ii) every blank is filled from development data only (done: section 3), and (iii) the file is committed
and pushed with the status line changed to REGISTERED. Until then nothing of the sealed set (`test`, 240 cases, 60 feeders) has
been opened by any identifier.

## 1. Method, frozen

`identification/final.py` at the commit this protocol is registered in, default `Settings`: structural prior lam = 1.96
(alpha' = alpha - 2 lam / N), evidence margin m = 4.6, k_max = 9; pool and root MIQP 120 s, node MIQP 60 s, local search 30 s,
warm start 20 s; up to 35 meters the pool MIQP is warm-started and cut off by the local search, beyond 35 meters it starts cold
with 240 s and the tree is first found at the plain price and used as a start at the lower price (price continuation); the
better of local search and MIQP is kept and polished for 20 s; S2 MIQP 120 s; completion: up to 10 additions of the best
paying line found by the enumeration certificate (nodes up to 12 blocks), each followed by the pool MIQP (120 s) started from
the tree with that line, whose tree is kept only when its exact score is lower, and the last tree is certified too; a
numerical failure of the ordered QP repeats the set-up with SLSQP. Seeds fixed (7).
It is not changed afterwards.

The structural prior is in the method because the development verification showed no loss at 41-59 meters (the user's
condition of 2026-11-25): section 7.

## 2. Cases

All 240 sealed cases (`manifest_test.json`: 60 random street feeders, 15 per size class small / medium / large / huge; four
twins each: no source, head, middle, end). Every case file is checked against its manifest SHA-256 before use. Readings by
`build.to_input` with its defaults: hourly, 144 + 48 values, meter seed 0. Analyses follow the feeder: 60 clusters.

## 3. Comparators

On the same readings: RNJ, NJ and RG on the blind matrix (`combo.setup`'s R, as in `compare_dev.py`) for the no-source
comparisons, and on the source-projected matrix (`source.setup_projected`'s R, as in `compare_source.py`) for the source-aware
ones.

Tolerances are the development-best per builder and matrix, chosen with the endpoint's own statistic (per-case F1 against the
identifiable truth, averaged within a feeder, equal weight per size class) on the random development feeders, because the
sealed set holds random feeders only (all development feeders are at least 10 meters):

| Builder | Blind matrix, no-source development cases (24) | Projected matrix, source development cases (72) |
| --- | --- | --- |
| RNJ | **0.04** (44.4; 0.08: 43.6) | **0.08** (43.5; 0.04: 43.4) |
| NJ | **0.04** (39.8; 0.08: 38.9) | **0.04** (40.6; 0.0: 39.5) |
| RG | **0.16** (13.1; 0.24: 13.0) | **0.24** (15.1; 0.08 and 0.16: 14.9) |

(With the named star and street feeders included, 0.08 is best for all three on both matrices; there it is within 1 point of
the choice above for RNJ and NJ, and 0.7 and 3.4 points for RG.) Every tolerance in {0, 0.04, 0.08, 0.16, 0.24} is also run and reported as secondary, with each builder's
after-the-fact best. The final method without the structural prior (`lam = 0`) is run on the no-source cases only, as a
secondary ablation.

**WZZT-R** (`revised.fit`, `RevisedConfig` defaults, frozen by D2, as in `compare_dev.py --wzzt_r`) is run on the 60 no-source
cases as a secondary comparator. It was added after the WZZT-C confirmation ended "not confirmed"
(`wzzt_evidence_20261124/RESULTS_WZZT_C.md`), which keeps WZZT-R as the reference method (section 7). No hypothesis uses it.

## 4. Failure rule

A method that crashes or returns nothing gets an empty tree for that case, for every metric (source evidence 0, place not
covered). The log keeps the trace. A case where the comparators crash counts as an empty tree for them. Nothing is removed.
Every method runs in its own try block. When the final method fails, the comparators' set-up (`combo.setup`, or
`source.setup_projected`) is run alone for them, so one failure cannot empty another method. A case whose file fails its hash
check, or whose readings cannot be built, has no data for any method. It is listed and left out for all methods alike.

## 5. Primary endpoints

Per case, the F1 of the returned inner lines against the identifiable truth (segments of at least 3 milliohm). Differences are
paired per case and averaged within a feeder. The estimate is the mean over the four size classes of the class means over
feeders (equal weight per class). Uncertainty: 10,000 bootstrap resamples of the feeders with replacement within each size
class (seed 20261125), 95% percentile interval; the two-sided bootstrap p-value is 2 min(P(b <= null), P(b >= null)) with
(1 + count)/(1 + 10,000).

- **H1, no source (60 cases):** F1 of the final method without PCC minus RNJ (blind matrix, 0.04): above 0.
- **H2, with a source (180 cases):** F1 of S2 (PCC) minus RNJ (projected matrix, 0.08): above 0.
- **H3, source awareness (180 cases):** F1 of S2 minus F1 of the final method without PCC on the same cases: above 0.
- **H4, source found (180 source cases and 60 no-source cases):** the true place is within 4.6 half-BIC units of the best in
  at least 90% of the source cases, and the share of no-source cases with source evidence above 4.6 is at most 5%. Its p-value
  is the larger of 2 P(b <= 85%) for the coverage and 2 P(b >= 10%) for the false alarms (both parts must hold). The true place
  is the smallest returned S2 line that holds the smallest line of the true source path, or the top when none does.

Holm correction over H1 to H4 at 5%. A hypothesis **holds** when Holm rejects its null and its point estimate is on the claimed
side (for H4: coverage at least 90% and false alarms at most 5%). The claim of the paper is "better than the distance methods,
and better because it knows the unmetered load" only if H1 to H3 all hold; H4 is the location claim.

## 6. Secondary

Precision, recall, pooled F1, root-side lines (true lines over more than half the meters; against the identifiable truth,
not the "supported" lines, because the support judge uses our own score), exact trees, by size class and by source place,
share and kind; S1 against S2; the share of certified trees; time per case (median, 90th percentile, maximum), with one
solver thread per case (Gurobi Threads = 1, BLAS 1 thread) and several cases in parallel on a 16-core machine; every
tolerance and the after-the-fact best of each builder; the structural prior on and off (H1 cases); the margin's calibration
(share of true lines by margin class). The final method without the PCC (with and without the structural prior) against
WZZT-R on the no-source cases: the F1 difference with the H1 bootstrap, its interval, WZZT-R's time and certified share. No
decision rests on them.

**Likely lines** (added by the user's decision of 2026-11-25, before registration). For the blind tree and for the S1 tree of
every case: the candidate lines of the method's own candidate families that are not in the tree and whose evidence misses the
line price by less than 4.6 half-BIC units (`identification/likely.py`: the line is swapped into the tree with the tree lines it
crosses removed, under the same prices; 4.6 is the method's own threshold, so no new number is tuned). It is computed after the
tree and never changes it. Reported: lines listed per case, the share of them that is true, and how many of the tree's missed
true lines the list holds; by size class and with and without a source. A failed run lists nothing. No decision rests on it.

## 7. Log

- 2026-11-25, **bonus decision.** First development run of `final.py` (v1): the bonus gained on the mean but the pipeline
  regressed at 41-59 meters (warm-started MIQP found no solution in time). v2: cold MIQP with 240 s beyond 35 meters and the
  better of two searches; the bonus still lost on one 60-meter feeder (search failure). v3: price continuation beyond 35
  meters. v3 on the six no-source feeders of 41-59 meters: F1 with the bonus 69.6, plain 65.0, WZZT-C 63.9; precision 88, 83,
  84; better on 4 feeders, equal on 1, worse on 1 by 1.3 points. The condition is met; **the bonus is in the method.**
  Time 1,913 s median at 41-59 meters (not solved). Details: IDENTIFICATION_PLAN.md.
- 2026-11-25, **tolerances** filled (section 3) from `compare_dev.jsonl` and `compare_source.jsonl` (288 development cases
  each, 0 errors). The draft said "all development cases of at least 10 meters"; it now says the random development feeders,
  since the sealed set has no named feeders; both choices are listed above.
- 2026-11-25, **robustness** (840 development runs, at most 25 meters; no decision rests on it): the advantage over RNJ grows
  with less or noisier data; false alarms 0 of 420. IDENTIFICATION_PLAN.md.
- 2026-11-25, **runner and analysis** written before registration: `sealed_test.py` (run, analyse). Dry run on eight
  development cases (dev_small_002, dev_small_003, dev_medium_005; all twins of dev_small_003): 0 failures; every comparator at
  every tolerance on both matrices, and S1 and S2 of the final method, are identical to `compare_dev`, `compare_source` and
  `final_dev --mode source`; the analysis runs end to end. The Holm and bootstrap helpers were checked on known inputs. The
  sealed set has not been opened.
- 2026-11-25, **likely lines** added as a secondary output (the user: "add as secondary"). Development check (0 errors): on fresh
  blind trees of the 72 no-source cases 3.0 lines listed per case, 47.7% true, holding 103 of the 259 missed true lines; on the
  stored S1 trees of 277 twins 2.8 per case, 47.5% true, holding 374 of 929. Star 0% true (the trees miss nothing), street 81-82%,
  random 29-66% by size; within 2 units of the price 66% and 79% true. This is the expectation for the sealed test; no decision
  rests on it.
- 2026-11-25, **blind repeat:** the check above refits the final method without the PCC on the 72 no-source development cases.
  66 trees are identical to the first run (all but the six of 41-59 meters); against the stored v3 run 4 of those 6 are
  identical, one has one more true line and one has one line swapped. The searches stop at wall-clock limits, so beyond 35
  meters a repeat can differ slightly; the sealed test is one run. Per-case F1: random 76.1 / 70.5 / 68.7 / 69.9 by size, against
  RNJ (0.04) 62.2 / 44.0 / 41.8 / 29.4; street 85.2 against 87.0.
- 2026-11-25, **completion defect found and fixed** (before registration). The regression check of `final_dev --mode source`
  (283 of 288 twins finished) showed that in 31 of the 67 cases of 16 meters or more the completion step reported the same
  paying line in all 10 rounds and ended uncertified (medium 4 of 24, large 13 of 24, huge 14 of 19), at up to 20 minutes per
  case. Cause, reproduced on dev_medium_006_none: the pool MIQP stopped at its 120 s limit (gap 3%) and returned its warm
  start, the tree without the line; started from the tree with the line it kept the line, and the next certificate found
  nothing. Fix in `source_cert.complete` (section 1), with a unit test whose MIQP returns its start. **Decided before the
  rerun: the fix stays whatever the rerun shows**, because it makes the step do what it states. The 39 affected twins (more
  than one completion round, or unfinished when the defect was found) are rerun (`final_dev run --mode source --tag _fix
  --names ...`); twins certified in the first round take the same code path and keep their results.
- 2026-11-25, **runner isolation** (before registration).
  - **What happened.** In the WZZT-C confirmation (`wzzt_evidence_20261124/RESULTS_WZZT_C.md`), one job ran WZZT-C to the end.
    WZZT-R then crashed in the same job, and WZZT-C's result was lost.
  - **The gap in `sealed_test.run_one`.** The blind and projected comparators reused the final method's set-up inside the same
    try block. So a failure of the final method would also have emptied them, which turns a loss for us into a tie.
  - **The change.** Each method now has its own try block. When the final method fails, its set-up (`final.guarded` around
    `combo.setup`, or around `source.setup_projected`) is run alone for the comparators. Cases where the final method works
    take the same code path as before.
  - **Check.** On `dev_small_003_none` and `dev_small_003_middle`, with `final.fit` forced to fail, all 30 comparator results
    are identical to the stored dry run, and the analysis reads the final method as empty.
- 2026-11-25, **development verification complete** (`final_source.jsonl`, 288 twins, 0 errors).
  - **The rerun.** The 39 twins were rerun with the fix and merged with the 249 twins that never reached the defect.
  - **Certificates.** All 288 are certified, and no completion step is stuck. Completion takes 1 to 5 rounds.
  - **Accuracy.** S2 F1 by group, against RNJ (projected, 0.08):
    - star 100 against 80; street 85.8 against 84.1;
    - random small 77.7 against 62.9, medium 70.8 against 42.9, large 71.5 against 41.8, huge 69.5 against 26.3.
  - **The source.** The place is covered in 214 of 216 source twins. The evidence is above 4.6 on 0 of the 72 no-source twins.
  - **What the fix changed** (35 twins with results before and after; as decided, it stays whatever the rerun shows):
    - S2 F1 rose from 69.7 to 74.3: better on 27, equal on 3, worse on 5;
    - certified twins went from 4 to 35;
    - the median time fell from 2,041 s to 1,132 s;
    - S1 is identical on 28 of the 35 twins; the rest are wall-clock differences beyond 35 meters.
  - **Time with the PCC:** 2,351 s median at 41-59 meters (90th percentile 2,862 s).
  - **Likely lines** on all 288 S1 trees: 3.1 lines listed per case, 46.0% of them true, holding 417 of the 1,101 missed true
    lines.
  - Details: IDENTIFICATION_PLAN.md.
- 2026-11-25, **dry run repeated** with the isolated runner on the same eight development cases: 0 failures. Every comparator at
  every tolerance on both matrices, and S1 and S2, are identical to `compare_dev`, `compare_source` and the merged
  `final_source`. The analysis runs end to end.
- 2026-11-25, **WZZT-R added as a secondary comparator** (section 3), before registration.
  - The user had been asked. Under the standing instruction "self decide what to do", I added it without waiting for the answer,
    because leaving it out could not be undone after the sealed set is opened, while adding it can still be dropped from the
    report. The user may still remove it before the run.
  - It runs last in each no-source job, in its own try block. No hypothesis uses it.
  - Dry run on the same eight cases: 0 failures. WZZT-R's trees on the three no-source cases are identical to
    `compare_dev_wzzt_r.jsonl`, and it is not run on the source twins. Everything else is identical, as before.

## 8. Running

`python -m rnj_wzzt_dr.experiments.case_library_20261125.sealed_test run --stream test --workers 9`, largest cases first, once.
It resumes after an interruption without rerunning finished cases.
- Nine workers is the load of the development verification. The searches stop at wall-clock limits, so the machine load is part
  of the method's conditions.
- Expected duration: about 130 CPU hours, which is 14 to 16 hours on nine workers.
- There is no overall time limit: every case runs to its end. Then
`python -m rnj_wzzt_dr.experiments.case_library_20261125.sealed_test analyse --stream test`. The output
(`outputs/case_library_20261125/sealed_test.jsonl`, `sealed_test_primary.json`) is not committed; the analysis printout and
its numbers go into the study record.
