# Algorithm 1 of draft v12 on the development twins (2026-11-26)

**Run.** `dev_v8.py run --max-n 25`: the 240 development twins with at most 25 meters, with the PCC (the gate
decides). Code frozen at ef9ec75 (a separate checkout, so later edits could not touch the run). Eight workers, about
1 h 45 min, 0 errors. The 48 twins with 26 to 60 meters are running now; this note covers the 240.

## Against the registered predecessor (same cases, both with the PCC)

The predecessor's numbers are S2 of `final_source.jsonl`. The truth is the identifiable truth.

| Cases | F1, v12 | F1, predecessor | v12 better / equal / worse |
| --- | --- | --- | --- |
| named feeders, 4 x 48 | 0.924 to 0.931 | 0.926 to 0.933 | 0 / 189 / 3 |
| random small and medium, 4 x 12 | 0.729 to 0.778 | 0.715 to 0.778 | 3 / 45 / 0 |
| all 240 | 0.890 | 0.890 | 3 / 234 / 3 |

- The trees are the same in 234 of 240 cases.
- What v12 adds is the labels and the load gate.
- It is slower: median 57 s per case against 9 s, maximum 4,040 s, because of the flip tests.
- **Gate:** it modelled the load in 180 of the 180 cases with a load and in 0 of the 60 without.
- **Stops:** 220 runs stopped by the stopping rule (Lemma 3) and 20 at the round limit.

## Labels

| Groupings of the tree | True | False |
| --- | --- | --- |
| decided | 1,475 | 3 (0.2%, in 3 cases) |
| open | 137 | 35 |
| unsettled (flip test at its time limit) | 82 | 6 |

| Open groupings outside the tree | True | False |
| --- | --- | --- |
| nested in the tree (one more line) | 235 | 350 |
| crossing the tree | 0 | 9 |

- **Decided labels are reliable:** 0.2% of them are wrong, against about 1% if read as a Bayes factor of 100. This is
  much better than the predecessor's local margins on the sealed set: only 78% of lines with a margin of 4.6 to 10
  were true (`RESULTS_SEALED.md`).
- **Open is not a "no":** 80% of the open groupings inside the tree are true lines.
- **The trees miss 434 true groupings.** 235 of them (54%) are labelled open, as lines nested in the tree.

## What this says for the proposals (`PROPOSALS_UNDER_TEST.md`)

- **SR1 passes** on these 240: the true tree is flagged by the S1 F-test in 0 cases (criterion: at most 2%). The leader
  is flagged in 1.
- **SR3, start rule:**
  - "No target" (every grouping decided, nothing open outside) happened in 25 cases.
  - None of the 25 has a wrong decision; 22 have a perfect tree.
  - The other 3 are the three twins of `flynn16_street_l6`. Each misses 4 true lines with no open alternative,
    probably lines the data cannot see; this still needs a look.
  - Decided groupings are wrong in 0.2% overall (criterion: at most 2%).
- **SR4:** pricing was complete in 240 of 240 cases, so S2 never blocks here and this set cannot test it.
- **TR2 and TR4, target rule:**
  - Open nested groupings average 2.4 per case, and 40% of them are true lines the tree lacks: 235 lines in 240 cases.
  - The TR4 criterion for T1 ("at most 1 such line per 50 cases") fails by far. So T1 would drop the targets most
    worth probing, unless probing cannot recover them under T0 or T2 either. That needs probing runs.
  - The simulated day on `dev_small_003_head` already showed one: (8, 9), true and nested, was T0's first target.

## The rewritten builders (`identification_v8/builders.py`, 2026-11-26)

The nine starting trees were rewritten inside the package, replacing `candidates.py` and its use of `identification/`.
On all 240 twins with at most 25 meters, with the load modelled exactly when the gate says so, every one of the nine
trees is identical to the predecessor's.

- One recursive-grouping tree differed at first, from a rounding difference of 1e-16 in how the load's profiles were
  projected out. The rewrite now does those operations in the predecessor's order.
- Algorithm 1's whole output on two cases is identical to the frozen ef9ec75.
- The 48 larger twins (26 to 60 meters) give identical trees too, so all nine builders match the predecessor on all
  288 twins. The rewrite is also faster there: 441 s against 788 s in total.
