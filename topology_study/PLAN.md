# Plan (2026-11-27)

## 1. Where we are

| Part | State |
| --- | --- |
| Method | Draft v16 (`paper/procedure.tex`), code `ptid/method/`. Algorithm 1 implemented and run on development cases; Algorithm 2 implemented, run on simulated days only. |
| Solver | Fixed 2026-11-26/27: on 47-60 meter feeders the MIQP gave false proofs and the local search stopped far from a local optimum (RESULTS.md, section 3). Now: converged local search, scaled coefficients, default tolerances, every claim checked against known trees. |
| Development runs | Running with the fixed code: one twin per feeder (72 cases) and the 66 real-feeder inputs. The 72-case run uses c5c4aae, which equals `ptid` except two refinements (CHANGES.md, section 3): probing's check, which identification never uses, and the node MIQP's scaling, an exact change of variables that gave the same results wherever it was compared (RESULTS.md, sections 4 and 5.1). The real-feeder run uses `ptid` itself. |
| Registered result | The predecessor's sealed test holds (RESULTS.md, section 1); its set is spent. Nothing here is registered. |
| Proposals | Start rule, target rule, stop rules, κ alternative: recorded, not adopted (PROPOSALS.md). |

## 2. Next steps, in order

1. **Finish the v16 identification runs** (72 development twins, 66 real inputs) and report them in RESULTS.md: F1
   against v13 on the same cases, wrong decided groupings, numerical failures, time, and the first real-feeder results.
   Check them against the criteria of PROTOCOL.md, section 4.
2. **The other 216 twins** (decision D1): only if step 1 shows changes on the feeders below 26 meters, or the
   wrong-label rate needs a tighter estimate.
3. **The 46 large real networks of Stage M** (decision D2): the most realistic test of the solver at scale.
4. **Speed at scale**, from the rows of step 1:
   - how often a second anchored round changed anything (the "stop when nothing changes" proposal, PROPOSALS.md §6);
   - ablation V5: per-meter bounds on and off on the medium twins. If they do not help, remove them (simpler code).
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
| D1 | Run the other 216 development twins? | after step 1, only if needed (see step 2) |
| D2 | Run Algorithm 1 on the 46 Stage M inputs (large real networks)? | yes, after step 1 |
| D3 | Flip tests on 47-60 meter feeders end unsettled after the full MIQP limit. Keep the limit (your choice of 2026-11-26), or shorten it there and report it? | keep until step 4 measures the effect |
| D4 | Adopt "stop when a round changes nothing"? | decide after its measurement in step 4 |
| D5 | Start and target rules (PROPOSALS.md)? | after their tests (step 6) |

## 4. Time on this machine (8 cores, one case per core)

| Run | Time |
| --- | --- |
| 72 development twins (one per feeder) | about 4 hours (the six huge twins set the pace) |
| the other 216 twins | about 10 hours |
| 66 real inputs | about 1 hour |
| 46 Stage M inputs (26-77 meters) | not measured; likely 10-20 hours |
| a probing day | minutes (small feeders) to hours (large) |
