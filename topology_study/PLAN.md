# Plan (2026-11-27)

## 1. Where we are

| Part | State |
| --- | --- |
| Method | Draft v16 (`paper/procedure.tex`), code `ptid/method/`. Algorithm 1 implemented and run on development cases; Algorithm 2 implemented, run on simulated days only. |
| Solver | Fixed 2026-11-26/27: on 47-60 meter feeders the MIQP gave false proofs and the local search stopped far from a local optimum (RESULTS.md, section 3). Now: converged local search, scaled coefficients, default tolerances, every claim checked against known trees. |
| Development runs | Done with the fixed code (RESULTS.md, section 4): one twin per feeder (72 cases) and the 66 real-feeder inputs. 0 of 367 and 0 of 162 decided labels are wrong (not physical lines); 0 numerical failures; below 26 meters the same trees as before in 59 of 60 cases. On 26-60 meter feeders few labels are decided and the search is weaker (step 4). |
| Registered result | The predecessor's sealed test holds (RESULTS.md, section 1); its set is spent. Nothing here is registered. |
| Proposals | Start rule, target rule, stop rules, κ alternative: recorded, not adopted (PROPOSALS.md). |

## 2. Next steps, in order

1. **The v16 identification runs** (done 2026-11-27): all three criteria of PROTOCOL.md §4 are met (RESULTS.md §4).
2. **The other 216 twins** (decision D1): not needed for the feeders below 26 meters (the trees are unchanged in 59 of
   60 comparable cases). Worth running for the 26-60 meter kinds only after step 4, to measure its effect.
3. **The 46 large real networks of Stage M** (decision D2): the most realistic test at scale. They have 26-77 meters,
   so they would meet the limits of step 4; better run after it, or now as a baseline.
4. **Search and labels on feeders of 26 to 60 meters** (found 2026-11-27, RESULTS.md §4.2). With honest proofs, few
   groupings are decided there, the anchors found in round 0 vanish when the pool grows, the anchored search then
   offers no new lines, and one huge twin ended 70.5 units worse than the old (falsely anchored) run, with a paying
   single line left. Options, all keeping every claim exact:
   - (a) **price until no single line pays** before the output, instead of at most 10 lines per round and then the
     round limit. Exact and cheap; on `dev_huge_004_end` it gains at least the 17.6 units left.
   - (b) **widen the pool from the tree itself:** offer the unions of 2 to k-1 children at every node of the current
     tree with at most K children, as the anchored universe does at anchors, but without requiring anything. The
     MIQP then also sees pairs of lines that pay only together (Remark 1). No label is needed for it, so it works when
     nothing is decided.
   - (c) measure how often a second anchored round changed anything (the "stop when nothing changes" proposal,
     PROPOSALS.md §6); on the large and huge twins it saves about a third of the time.
   - (d) ablation V5: per-meter bounds on and off. If they do not help, remove them (simpler code).
   Recommendation: (a) and (b), then rerun the 12 large and huge twins of the 72 and compare.
5. **The scoring truth on real feeders** (done 2026-11-27, RESULTS.md §4.1): test S1 flagged the true tree on one real
   feeder (both days), and 4 of 162 decided groupings were not in the scoring truth. All of it came from the
   3 milliohm floor of the scoring truth: the 4 groupings are physical lines below it, and the exact truth fits.
   Left to decide: whether the paper scores real feeders with the exact truth, the identifiable truth, or both (now
   both are reported).
6. **Probing:**
   - development probing days under the target variants T0, T1, T2 (tests TR3, TR4);
   - the ψ knee: the cost-coverage curve over ψ;
   - the κ alternative, P(Z ≥ δ), on the same days;
   - the start rule's tests SR2 to SR4.
7. **Open items:**
   - `flynn16_street_l6`: three twins miss 4 true lines each with no open alternative;
   - true lines fitted at r = x = 0 (invisible to the data).
8. **Paper:** fill the results into draft v16 as steps 1 to 6 finish; compile on Overleaf; figures.
9. **A registered test of this method:**
   - a new sealed set (the generator with new feeder, customer and load random numbers);
   - a protocol whose hypotheses and analysis are written before the set is built;
   - one run.

## 3. Decisions needed from you

| | Question | Recommendation |
| --- | --- | --- |
| D1 | Run the other 216 development twins? | not now; the large and huge ones after step 4 |
| D2 | Run Algorithm 1 on the 46 Stage M inputs (large real networks)? | after step 4 (or now, as a baseline) |
| D3 | Flip tests on 47-60 meter feeders end unsettled after the full MIQP limit. Keep the limit (your choice of 2026-11-26), or shorten it there and report it? | keep until step 4 measures the effect |
| D4 | Adopt "stop when a round changes nothing"? | decide after its measurement in step 4 |
| D5 | Start and target rules (PROPOSALS.md)? | after their tests (step 6) |
| D6 | Which search improvements for 26-60 meter feeders (step 4)? | (a) and (b) |
| D7 | Score real feeders against the exact truth, the identifiable truth, or both (step 5)? | both |

## 4. Time on this machine (8 cores, one case per core)

| Run | Time |
| --- | --- |
| 72 development twins (one per feeder) | 4.6 hours (measured; the six huge twins set the pace) |
| the other 216 twins | about 12 hours |
| 66 real inputs | 1.1 hours (measured, 2 workers beside the other run) |
| 12 large and huge twins after step 4 | about 5 hours |
| 46 Stage M inputs (26-77 meters) | not measured; likely 10-20 hours |
| a probing day | minutes (small feeders) to hours (large) |
