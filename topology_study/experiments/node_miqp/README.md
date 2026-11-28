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

Shared pieces (feeder generator, AC simulation, meters, node scan, enumeration) are in `ptid/runs/node_study.py`;
checks of them in `ptid/tests/test_node_study.py`.

## Three additions to the method tested here

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
python -m unittest ptid.tests.test_node_study          # 8 checks, under a minute
python -m ptid.runs.node_exact run --workers 8          # N1, about 1.5 hours on 8 cores
python -m ptid.runs.node_recovery run --workers 8       # N2, about 1 hour
python -m ptid.runs.node_pipeline run --workers 8       # N3, several hours
python -m ptid.runs.node_probe run --workers 8          # N4, about 1.5 hours
python -m ptid.runs.node_exact summary                  # and likewise for the others: the tables below
```

Each `run` is resumable and appends to the `.jsonl` file named above (in this directory; `--out` changes it). Quick
versions: `--widths 6,8 --seeds 1` (N1), `--widths 10 --seeds 11` (N2), `--seeds 21 --r 0.014` (N4).
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
- At 20 children 5 of 9 pairs of solves are proved within 300 s each; at 24 children 1 of 9. The lower bound, not the
  incumbent, is what is slow: a warm start from a quick local search over unions (`--warm`, file `exact_warm.jsonl`)
  proved 6 of 9 at 20 children at the same median time and improved one time-limited incumbent; it changes no
  result. Wider nodes need a stronger formulation (open item).
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

(running; see TABLES.md when it finishes)

### N4: probing an open node question

48 inputs: d = 12, 8-day history, weak load variation (0.10), segment 0 (control), 0.006, 0.010, 0.014 ohm, 12 seeds
(21-32); one probing day. 43 were open after the history (31 with a segment, 12 controls):

| arm | exact union decided right | existence decided right | any wrong decision | leader right | overlap with the segment | homes probed per day | kWh per day | CNY per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 3 | 15 | 0 | 21 | 0.53 | 0 | 0 | 0 |
| greedy1 (draft: one home per hour) | 5 | 17 | 0 | 27 | 0.66 | 14.1 | 39.5 | 83.9 |
| greedy3 (up to 3 homes, per CNY) | 7 | 21 | 0 | 25 | 0.69 | 23.7 | 62.9 | 86.8 |
| random3 (same hours and homes) | 2 | 15 | 0 | 24 | 0.62 | 22.8 | 37.5 | 85.6 |

"Existence decided" means some union beats the flat junction by 4.6 (a hidden shared line is there) without naming it
exactly; "exact union" also needs every one-block change to lose by 4.6.

- Paired on the same inputs, greedy3 decides existence where passive operation does not in 6 inputs and the reverse
  in none (sign test p = 0.031); against random orders bought in the same hours, also 6 to 0 (p = 0.031). Exact
  unions: 4 to 0 against passive (p = 0.125), 5 to 0 against random (p = 0.0625).
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
discordant on 550 hard runs), the direction and the absence of wrong confirmations are the same; the sample here is
small (43 open inputs). 24 more inputs (seeds 33-44 at 0.010 and 0.014 ohm) were added after these results for power;
they are reported separately below.
