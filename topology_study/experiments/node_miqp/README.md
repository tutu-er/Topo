# Tests of the node MIQP (2026-11-28)

The paper's claim is now the node MIQP: at a junction with d children of the current tree, d binary variables encode
all 2^d - d - 2 ways to add one shared line, and one solve finds the best of them (with a lower bound) and refits every
line. It can propose a line that no candidate pool contains. These tests ask four questions:

| Test | Question | Script (`ptid/runs/`) | Rows |
| --- | --- | --- | --- |
| N1 | Is it exact, and how does it scale against enumeration? | `node_exact.py` | `exact.jsonl` |
| N2 | Does it find hidden shared lines that the nine tree builders miss, and how often does it propose a wrong one? | `node_recovery.py` | `recovery.jsonl` |
| N2b | What price should a line found among 2^d - d - 2 unions pay? (post hoc, from the N2 and N1 rows) | `node_price.py` | (none) |
| N3 | What does it change in Algorithm 1, end to end? | `node_pipeline.py` | `pipeline.jsonl` |
| N4 | When the node question stays open, can probing settle it? | `node_probe.py` | `probe.jsonl` |
| N5 | Root lines (short lines near the substation that carry many homes): does the node MIQP find them where the builders and a local search do not? | `node_roots.py` | `roots.jsonl` |
| N7 | End to end on short root lines: does Algorithm 1 with the node MIQP recover the root lines that every builder misses, and without it not? (the EU LV comparison with a known truth) | `node_roots_e2e.py` | `roots_e2e.jsonl` (version 1: `roots_e2e_v1.jsonl`) |
| N6 | When a root line stays open, does probing settle it, under noisier meters, other loads and higher caps? | `node_probe_roots.py` | `probe_roots.jsonl` |

Shared pieces (feeder generator, AC simulation, meters, node scan, enumeration) are in `ptid/runs/node_study.py`;
checks of them in `ptid/tests/test_node_study.py`.

## Four additions to the method tested here

1. **The node flip test.** `node_miqp(..., exclude=[S])` adds a no-good cut that forbids the union S. Solved with the
   best union S* excluded, it returns the best *other* union S2 and a lower bound on every other union at the node. A
   hidden line is then *decided present* only when S* beats the flat junction by the margin m = 4.6 **and** beats every
   other union by m (the bound of the second solve); *decided absent* when the first solve's bound shows that every
   union scores at least m worse than the flat junction; *open* otherwise. This is the label of Definition 1 restricted
   to single insertions at one node, over all 2^d - d - 2 unions, which no listing can cover at wide nodes.
2. **Probing a node question.** The competing trees are the flat junction, T + S* and T + S' for every union S' that
   differs from S* in one child (one block's membership each). Blocks are chosen by eq. (15) of the draft, with
   psi = V / m and V = 2000 CNY per resolved question (the value locked in the earlier hourly-probing study). Two block
   rules are compared: the draft's one meter per hour (`greedy1`), and an aggregated order of up to three
   participating homes per hour ranked by expected evidence per CNY (`greedy3`), with the same price caps and voltage
   band, against random orders bought in the same hours (`random3`) and passive operation.
3. **The price of a line found among many.** Theorem 1(b) bounds what one redundant line gains; the best of M
   redundant candidates gains about ln M more (a union bound over the M candidates). The node MIQP searches
   M = 2^d - d - 2 unions, so with the price of one line it adds noise lines at wide nodes. The extended BIC (Chen and
   Chen, 2008, Biometrika 95:759) charges gamma ln M more for a line chosen among M. Option
   `Settings.node_gamma` (`search.node_price`): a line found at a node with c children must improve the score by more
   than gamma ln(2^c - c - 2), and lines are ranked net of it. Default 0 (the draft's price, so nothing else
   changes); N2b measures gamma = 0.5 and 1 post hoc and N3 runs gamma = 0.5 end to end (arm D).
4. **The line swap at a node** (added after N7 version 1). Pricing only adds lines, so a line of the tree that is one
   lateral too wide or too narrow stays whenever adding the right line does not pay its price. `search.node_swap`
   removes each tree line in turn and puts the best union of its parent's children in its place when that lowers the
   score; the number of lines and the node stay the same, so no price enters. Without a root line the root has all
   its laterals as children (10 to 15 here), beyond K = 9: the node MIQP solves those nodes, enumeration the others.
   Option `Settings.node_swap` (default False); N7 version 2 runs it with and without the node MIQP.

## What the results support (for the paper's claim)

1. **One solve replaces 2^d - d - 2 refits, exactly.** Same best union and runner-up as enumeration on 54 of 54
   junctions up to 16 children, faster from about 10 children, 10 times faster at 16 (N1).
2. **It finds lines no candidate pool holds.** 11 of the 31 paying segments that none of the nine builders proposed
   (N2). This is the case the paper should lead with; the earlier wide-feeder test needed a hand-restricted pool.
3. **Its decisions are reliable only with its flip test.** The excluded-union solve certifies the chosen union
   against all other unions at the node: 0 wrong of 240, against 30 wrong on the gain alone (N2).
4. **A line found among M candidates needs ln M more price.** The draft's price lets the best of thousands of noise
   unions pay; the extended BIC (gamma = 0.5) removes the noise additions in the controls at almost no loss (N2b, N3).
   To be registered before it is adopted.
5. **Probing completes it.** When a node question stays open, aggregated orders chosen by eq. (15) decide the exact
   hidden line in 15 of 47 open segments against 4 passively, with no wrong decision (N4).

Limits: one feeder layout (one wide junction, two meters per block); synthetic loads; the flat tree around the
junction is given in N1, N2 and N4; at 20 or more children the node MIQP often stops at its limit; "absent" cannot be
certified at wide nodes with this margin; N3 uses shortened solver limits.

## The earlier EU LV root-line result: can it be the main result? (checked 2026-11-28)

The figure "81% of near-root lines against 17-38% for the builders" is Amendment D of the predecessor's experiment M
(`rnj_wzzt_dr/experiments/wzzt_evidence_20261124/RESULTS_MIQP.md`): WZZT-R against RNJ, NJ and RG on `eulv_16d`
(33 real feeders, 16-day histories, 88 lines feeding more than half of the meters). It cannot carry the node MIQP claim:

1. **The MIQP did not find those lines.** In WZZT-R's own attribution, 58 of its 82 accepted near-root lines came from
   an RNJ pool built on its transformed matrix and the rest from a screen, sibling pairs and residual clusters; the
   whole-map score decided which to keep. Its final MIQP check ran 32 times and never found a better line.
2. **The builders had a weaker matrix.** On WZZT-R's own matrix (`outputs/.../builders/analysis_ponly_20261125.txt`,
   exploratory) RNJ's and NJ's best near-root recall is 57% and 59% (RG 33%), not 17-38%; against RNJ's best the difference
   (13.7 points) is not significant (p = 0.063); against NJ's best, 25 lines are found by WZZT-R only and 6 by NJ only.
3. **The present pipeline shows the same split.** On the 21 verified random development feeders, NJ and RNJ both
   missed 67 lines feeding more than half of the meters; the pipeline recovered 36, 29 of them already in its warm
   start (local search over the nine builders' trees); in the traced cases the node MIQP added none
   (`experiments/root_ablation/README.md`). After the tree MIQP the junctions there are rarely wider than 9 children.
4. `eulv_16d` has been seen, and the transformer networks of `eulv_large` are the Stage M inputs (not used without
   the user's word).

What it does show is the mechanism: distance-based builders lose *short* lines even when their whole-map evidence is
high (short minus long recall -40 to -63 points in the high-evidence tertile, for every builder), the whole-map score
does not. The node MIQP is the exact whole-map search at a junction, so it inherits that advantage exactly where the
two conditions meet: the line is in **no** builder's tree (short, relative length below about 0.08, yet carrying many
homes), and its junction is **wider than 9 children** (a missed root line leaves all its laterals there). N5 builds
that case with a known truth; its first rows (chain layout, 12 laterals, r = 0.003 ohm, relative length about 0.04)
look like the EU LV short near-root row: the builders' pool held 4 of 8 root lines and the best builder's tree 3,
the local search found 6 and the node MIQP all 8. From 0.005 ohm on the builders propose almost every root line, so
shorter lines (0.0015 and 0.002 ohm) were added. The end-to-end form of the EU LV comparison (Algorithm 1 with and
without the node MIQP against each builder's tree, on near-root lines) follows on the same feeders.

## Feeders

One junction of d blocks (laterals of 0.03 to 0.05 ohm), two meters per block (service cables 0.015 to 0.025 ohm), a
trunk of 0.016 ohm, 0.4 kV, 0.2 MVA base. A hidden shared segment of r ohm (x = 0.4 r) feeds a random 2 to d - 2 of the
blocks. Loads: 1.5 kW mean per meter (0.6 to 1.4 times), a common daily cycle, and each meter's own hourly variation
(relative size 0.32, or 0.10 for "weak variation": loads that move together). AC power flow at quarter hours,
hourly readings, meter errors of `ptid.cases.physics.Meters` (voltage gain and noise 0.05%, P and Q 0.3% and 0.5%).
Every hour's meter noise has its own random stream, so paired runs differ only by what the probing changed. The
*true evidence* of a segment is H(flat tree) - H(true tree) on the same data: the most any method could see.

This generator is new and separate from the frozen case library (`case_library_20261125`), whose feeders rarely have
a junction wider than nine children.

## How to run and check

From `topology_study` (Python 3.10, numpy, scipy, gurobipy; on this machine `conda run -n Topo310`):

```
python -m unittest ptid.tests.test_node_study          # 12 checks, about a minute
python -m ptid.runs.node_exact run --workers 8          # N1, about 1.5 hours on 8 cores
python -m ptid.runs.node_recovery run --workers 8       # N2, about 1 hour
python -m ptid.runs.node_pipeline run --workers 8       # N3, several hours
python -m ptid.runs.node_probe run --workers 8          # N4, about 1.5 hours
python -m ptid.runs.node_roots run --workers 8          # N5, about 2 hours
python -m ptid.runs.node_roots_e2e run --workers 8      # N7, several hours
python -m ptid.runs.node_probe_roots run --workers 8    # N6, several hours (only open inputs are probed)
python -m ptid.runs.node_exact summary                  # and likewise for the others: the tables below
```

Each `run` is resumable and appends to the `.jsonl` file named above (in this directory; `--out` changes it). Quick
versions: `--widths 6,8 --seeds 1` (N1), `--widths 10 --seeds 11` (N2), `--seeds 21 --r 0.014` (N4),
`--widths 12 --seeds 51 --r 0.005` (N5), `--settings base --r 0.005 --seeds 61` (N6).
`python experiments/run.py node-exact run` and so on work as well.

## Results (development runs, 2026-11-28; every number below is rebuilt by `python -m ptid.runs.node_report`)

### N1: exact, and faster than enumeration from about 10 children

The flat junction of the true tree; 9 feeders per width (3 seeds x no segment, 0.004 and 0.008 ohm), 8 days of
hourly data. Times include both solves (best union, then the flip test with it excluded), one thread, 300 s limit each.

| children d | unions | node MIQP s (median / max) | both solves proved | enumeration s (median) | same best union | same runner-up |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 6 | 56 | 0.4 / 0.8 | 9/9 | 0.2 | 9/9 | 9/9 |
| 8 | 246 | 1.1 / 1.3 | 9/9 | 1.1 | 9/9 | 9/9 |
| 10 | 1,012 | 3.0 / 4.6 | 9/9 | 6.2 | 9/9 | 9/9 |
| 12 | 4,082 | 10.6 / 11.5 | 9/9 | 32.8 | 9/9 | 9/9 |
| 14 | 16,368 | 23.3 / 73.3 | 9/9 | 217 | 9/9 | 9/9 |
| 16 | 65,518 | 75.6 / 273 | 9/9 | 811 | 9/9 | 9/9 |
| 20 | 1,048,554 | 239 / 604 | 5/9 | not run | - | - |
| 24 | 16,777,190 | 606 / 607 | 1/9 | not run | - | - |

- Up to 16 children the node MIQP and its flip test give the same best union and the same runner-up as enumeration
  on 54 of 54 feeders, with objective differences below 4e-11. Enumeration time grows about 4 times per 2 children,
  the node MIQP's about 3 times; they cross at about 10 children (the default K = 9 is right).
- At 20 children 5 of 9 pairs of solves are proved within 300 s each; at 24 children 1 of 9. A warm start from a
  quick local search over unions (`--warm`, file `exact_warm.jsonl`; it changes no result of a finished solve)
  proved 6 of 9 at 20 children and 3 of 9 at 24, and where solves stop at the limit it finds better unions: 1 of 9 at
  20 children, 5 of 9 at 24. It should be on for nodes beyond about 16 children. Proving optimality there needs a
  stronger formulation (open item); the time-limited result is still a feasible union and a valid bound.
- No "present" decision named a wrong union at any width.

### N2: it finds lines the builders miss; the flip test keeps its decisions right; it also proposes noise lines

240 feeders: d = 10, 14, 18; segment 0 (control), 0.003, 0.005, 0.008, 0.012 ohm; load variation 0.32 and 0.10;
8 seeds; 8 days of data; node MIQP limit 120 s per solve. Rows grouped by the segment's true evidence.

| true evidence | feeders | in builder pool | node adds the right union | ... and it was not in the pool | adds a wrong union | decided right | decided wrong | decided wrong without the flip test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no segment | 48 | - | - | - | 8 | - | 0 | 2 |
| below 0 | 77 | 13 | 0 | 0 | 25 | 0 | 0 | 2 |
| 0 to 4.6 | 21 | 8 | 6 | 3 | 14 | 0 | 0 | 7 |
| 4.6 to 15 | 23 | 13 | 9 | 1 | 14 | 0 | 0 | 14 |
| 15 to 50 | 33 | 26 | 28 | 6 | 5 | 14 | 0 | 5 |
| 50 or more | 38 | 37 | 38 | 1 | 0 | 36 | 0 | 0 |

- **Lines the builders miss.** In 31 feeders the segment pays (true evidence above 0) but none of the nine builders
  proposed it; the node MIQP added the right union in 11 of them. Where the segment is clear (evidence 15 or more)
  it adds the right union in 66 of 71 feeders. The builder that finds segments most often is the clustering of the
  meters' voltage drops (`volt_m`, 82 of 192), not RNJ (41) or NJ (32).
- **The flip test is necessary.** Deciding on the gain over the flat junction alone would have named a wrong union 30
  times (near-twins: the segment plus or minus one block, and 2 noise unions in the controls). With the flip test
  (the runner-up must also be beaten by 4.6) no decision is wrong (0 of 240) and 50 are right.
- **Noise lines.** The best of 2^d - d - 2 unions pays by chance in 8 of 48 controls and adds a wrong union in 58
  feeders with a segment (often a near-twin). In Algorithm 1 such lines enter the tree labelled open: they lower F1,
  not the decided labels. N2b addresses this.
- **Width.** At 18 children the node MIQP reaches its 120 s limit (median 207 to 243 s for both solves). There 46 of
  64 segments pay and the builders propose 32; at 10 children 28 pay and they propose 33.
- **"Absent" is out of reach at wide nodes.** No control was certified absent: that needs every union to score 4.6
  worse than the flat junction, but the best noise union among thousands gains about ln(2^d) (6.9 at 10 children),
  more than the line price of about 6.6 leaves. Controls are therefore judged by wrong decisions and wrong additions.

### N2b: the price of a line found among many (post hoc, no new solves)

Additions recounted with an extra price gamma ln(2^d - d - 2) (6.9, 9.7, 12.5 score units at gamma = 1 for d = 10,
14, 18):

| rows | gamma | right additions | wrong additions | wrong additions in controls |
| --- | ---: | ---: | ---: | ---: |
| N2 (240 feeders, seeds 11-18) | 0 (draft) | 81 of 192 | 66 | 8 of 48 |
| | 0.5 | 78 | 22 | 0 |
| | 1 | 72 | 9 | 0 |
| N1 (72 feeders, seeds 1-3, d = 6 to 24) | 0 (draft) | 27 of 48 | 16 | 6 of 24 |
| | 0.5 | 24 | 6 | 0 |
| | 1 | 23 | 4 | 0 |

gamma = 0.5 removes two thirds of the wrong additions and all of those in the controls, for 3 of 81 right ones; the
same holds on N1's separate seeds and widths. This was chosen after seeing N2, so it is a development finding; N3 runs
it end to end (arm D) and it needs a registered test before it is adopted.

### N3: Algorithm 1 end to end

12 feeders: one junction of 12 or 16 blocks, segment 0 (control), 0.006 or 0.010 ohm, load variation 0.32, seeds 41
and 42; the shortened solver limits of `node_pipeline.SHORT` (tree MIQP 20 s, flip tests 10 s, node MIQP 30 s).

| arm | mean F1, segments (8) | segment in final tree | extra lines | mean F1, controls (4) | extra lines, controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| A tree MIQP only | 0.806 | 6 | 12 | 0.807 | 4 |
| B enumeration up to 9 children | 0.809 | 6 | 11 | 0.807 | 4 |
| C node MIQP beyond 9 (the method) | 0.799 | 6 | 12 | 0.800 | 5 |
| D C with gamma = 0.5 | 0.806 | 6 | 12 | 0.807 | 4 |

- Here every segment was already in the builders' pool (8 of 8) and the first tree solve had 6 of them, so node
  search had nothing to add: all four arms end with the same 6 segments, none decided with 10 s flip tests. The node
  MIQP's value end to end is where the pool lacks the line (N2: 11 of 31 such segments found; N5).
- C adds one noise line in a control that A, B and D do not; D (gamma = 0.5) removes it, as N2b predicts.
- F1 stops near 0.81 in every arm: laterals of two homes are weakly identified and every arm merges some of them.
  N5 and N6 therefore use 3 homes per lateral (and 2 and 4 in N5).

### N4: probing an open node question

First set: 48 inputs, d = 12, 8-day history, weak load variation (0.10), segment 0 (control), 0.006, 0.010,
0.014 ohm, 12 seeds (21-32); one probing day. Added set (after the first set's results, for power, same design):
seeds 33-44 at 0.010 and 0.014 ohm, 24 inputs. Open after the history: 43 of the first set (31 with a segment, 12
controls) and 16 of the added set; 59 together.

All 59 open inputs:

| arm | exact union decided right | existence decided right | any wrong decision | leader right | overlap with the segment | homes probed per day | kWh per day | CNY per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 4 | 21 | 0 | 27 | 0.50 | 0 | 0 | 0 |
| greedy1 (draft: one home per hour) | 10 | 26 | 0 | 36 | 0.65 | 13.6 | 38.0 | 82.8 |
| greedy3 (up to 3 homes, per CNY) | 15 | 30 | 0 | 34 | 0.68 | 23.1 | 61.2 | 83.8 |
| random3 (same hours and homes) | 6 | 23 | 0 | 33 | 0.61 | 21.9 | 36.4 | 84.9 |

"Existence decided" means some union beats the flat junction by 4.6 (a hidden shared line is there) without naming it
exactly; "exact union" also needs every one-block change to lose by 4.6 (the node flip test).

Paired on the same inputs (right with one arm only; two-sided sign test):

| comparison | exact union | existence |
| --- | --- | --- |
| greedy3 vs passive, all 59 | 11 to 0 (p = 0.001) | 9 to 0 (p = 0.004) |
| greedy3 vs random3, all 59 | 9 to 0 (p = 0.004) | 7 to 0 (p = 0.016) |
| greedy1 vs passive, all 59 | 6 to 0 (p = 0.031) | 5 to 0 (p = 0.063) |
| greedy3 vs greedy1, all 59 | 5 to 0 (p = 0.063) | |
| greedy3 vs passive, first set (43) | 4 to 0 (p = 0.125) | 6 to 0 (p = 0.031) |
| greedy3 vs passive, added set (16) | 7 to 0 (p = 0.016) | 3 to 0 (p = 0.25) |

- Among the 47 open inputs with a segment, the exact hidden line was decided by passive operation in 4 (9%), by the
  draft's one-meter probing in 10 (21%) and by aggregated orders in 15 (32%), at about 84 CNY per day, with no wrong
  decision in any arm. Random orders in the same hours (same cost) decided 6: where the orders go matters, not only
  that energy is bought.
- No arm made a wrong decision, and probing did not create one: in the controls every arm stayed open, and with
  greedy3 the best noise union's gain fell further than with passive operation in 8 of 12.
- The limit is the budget: the 100 CNY daily cap is spent in 8 to 10 hours (about 63 kWh of mostly cheap increases),
  and the last block memberships need more evidence than one day buys. Two traces: on seed 28 (0.006 ohm) the exact
  union led from hour 7 to 16 but one block's membership stayed within 0.1 to 0.4 of its rival; on seed 27
  (0.010 ohm) every membership passed 4.6 at hours 2 and 3, buying stopped, and later the membership of a block with
  no participating home slipped back to 3.1. At 0.006 ohm (true evidence -3.5) nothing was decided in a day by any
  arm.
- One meter per hour (the draft's block) gains less: its blocks are 3 kW, and evidence grows with the square of the
  block (a day of 3 kW at one member adds about 8 score units of evidence, three homes at once about 80).

Compared with the earlier hourly-probing study (2026-11-23: strong greedy 40% against passive 29% correct, 62 to 2
discordant on 550 hard runs), the gain here is larger (exact line 9% to 32%, 11 to 0 discordant) and, as there,
probing added no wrong confirmation; the sample is smaller (59 open inputs, one feeder layout). Two differences from
the draft's Algorithm 2 produce it: psi = 2000 CNY / 4.6 per score unit (the earlier study's value per resolved
question) instead of the placeholder 5, and orders of up to three homes ranked by evidence per CNY.

### N7: root lines end to end (version 1: the draft's Algorithm 1)

40 feeders (seeds 51-54; 12 or 16 laterals of 3 homes; one root line or a nested chain at 0.002 and 0.003 ohm; 8
controls), `node_pipeline.SHORT` limits, arms B (enumeration up to 9 children) and D (node MIQP beyond, gamma 0.5)
from one common first solve. Root lines with positive true evidence (46; 31 of them feeding more than half of the
meters, the EU LV class):

| | rnj | nj | rg | volt | volt_m | resid | rrows | upgma | ward | pool | first solve | B final | D final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all 46 | 13% | 4% | 0% | 24% | 41% | 0% | 7% | 0% | 2% | 57% | 48% | 48% | 48% |
| over half the meters (31) | 19% | 3% | 0% | 29% | 55% | 0% | 0% | 0% | 0% | 68% | 65% | 65% | 65% |

- Algorithm 1 found many more root lines than any builder's tree (48% against 0 to 41%; F1 0.92 against 0.51 for the
  best builder), but B and D found the same 22: every one was already in the first tree solve, and no line decided
  by the pool's flip test was wrong.
- The node MIQP added nothing because it was never asked about the root: the first solve always put *some* line
  there, so the root kept 3 to 9 children. Every one of the 24 missed root lines was a **near miss**, a line of the
  final tree one lateral (3 homes) too wide or too narrow. In 17 of the 24 the true tree scores better than the final
  tree (by up to 15.8), and in 16 swapping the near miss for the true line lowers the score; adding the true line
  never pays its price (the near miss already takes most of its evidence), and pricing only adds lines.
- Hence the fourth addition, the line swap at a node (above). Removing the near miss leaves the root with 10 to 15
  children, which only the node MIQP solves exactly. Version 2 runs both arms with the swap (B skips nodes over 9
  children, D solves them) on fresh seeds 55-58 and then on the diagnosed seeds 51-54. Its predictions were written
  in `node_roots_e2e.py` before it ran.
