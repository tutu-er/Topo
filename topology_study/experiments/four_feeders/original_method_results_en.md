# What Node MIQP adds to Tree MIQP in feeder topology identification

## Paper claim

> We propose a node mixed-integer quadratic program (Node MIQP) that can identify a shared feeder line even when its downstream meter group is missing from the candidate pool. The Tree MIQP first fits a radial tree by selecting compatible groups from that pool. At a junction of this tree, the Node MIQP selects which child subtrees share an upstream line. Their combined active and reactive power produces a voltage drop observed by meters in each selected subtree. The model fits this response while re-estimating the impedances of the new and existing lines. The Tree MIQP then reselects the feeder topology with the new group available.

The manuscript source is **`paper/procedure.tex`, draft v19**. The abstract introduces the two models, the introduction states the contribution, and the methods section gives the node formulation. Scheduled probing appears later as a way to distinguish competing groupings. [Abstract](../../paper/procedure.tex#L97) · [Contribution statement](../../paper/procedure.tex#L131) · [Node formulation](../../paper/procedure.tex#L308) · [Scheduled probing](../../paper/procedure.tex#L586)

## Direct comparison on the same starting tree

Consider a junction with $d$ child subtrees. The Tree MIQP has a selection variable for each candidate line supplied to that solve. The Node MIQP has one membership variable per child. Selecting two through $d-1$ children describes $2^d-d-2$ possible shared lines. At a 22-child junction, 22 binary choices represent 4,194,280 possible single-line insertions. A fixed-pool Tree MIQP would need to list each group to offer the same choices. [Tree selection](../../ptid/method/miqp.py#L105) · [Node selection](../../ptid/method/search.py#L58)

The existing branches stay in place during the node solve, and its bound applies to the best single insertion at that junction. On fixed junctions with 8, 10 and 11 children, exhaustive enumeration refitted 3,293 legal groups in total. It selected the same group as Node MIQP at each junction, with matching objectives to solver tolerance. [Enumeration check](evidence.json) · [Tree re-solve](../../ptid/method/identify.py#L132)

The paired experiment used 24- and 36-meter feeders. Both arms received the same measurements and knew which meters belonged to each of 12 neighborhoods. The shared connection upstream of seven neighborhoods was withheld. Both arms started from the same anchored Tree MIQP solution and enumerated additions at junctions with up to nine children. The second arm also ran Node MIQP at wider junctions. The per-solve limit was 60 seconds.

| Feeder | Children at tested junction | Groups recovered by anchored Tree MIQP | After narrow-node search | After enabling wide-node MIQP and re-solving the tree | New group first found by Node MIQP |
|---|---:|---:|---:|---:|---|
| 24 meters | 16 | 8/13 | 8/13 | **13/13** | 14 meters sharing one line |
| 36 meters | 22 | 7/13 | 7/13 | **13/13** | 21 meters sharing one line |

Both anchored Tree MIQP solves **reached their specified optimality tolerance**. The missing line was absent from their respective pools of 12 and 33 candidates. The first Node MIQP solve found the true shared line in 7.38 and 39.34 seconds, respectively. With the enlarged pools, local search produced Tree MIQP warm starts containing all 13 true groups and zero false additions. The subsequent Tree MIQP retained both trees. At the earlier four-second limit, the 36-meter case had stayed at 7/13 even with wide-node search enabled. [Paired results](../wide_feeders/README.md) · [Fixed-node bounds and tree solutions](../wide_feeders/time_limits.json)

With ordinary initialization, neighbor joining (NJ) and rooted neighbor joining (RNJ) already found the shared line on both feeders. In the paired controls without that extra segment, both refinement settings recovered all 12 true groups and added zero false groups at the four-second limit. [Initialization and controls](../wide_feeders/results.json)

## Performance under ordinary initialization

Grouping F1 compares the sets of meters downstream of the reference and estimated branches after merging segments below 3 mΩ. The means give each feeder equal weight. An exact match recovers every scored group without any false additions.

| Development feeders | Full pipeline | RG | NJ | RNJ | Park | Exact matches by full pipeline |
|---|---:|---:|---:|---:|---:|---:|
| Soumalas11, Paper15, Flynn16, Pengwah18 | **1.000** | 0.914 | 0.727 | 0.691 | 0.087 | 4/4 |
| 21 verified random feeders, small to huge | **0.622** | 0.129 | 0.340 | 0.399 | 0.046 | 0/21 |

On the four named feeders, multistart local search recovered all 23 true groups before the first Tree MIQP solve. On the random feeders, mean F1 rose from 0.608 at that warm start to 0.622 at the end. Eight feeders improved, eleven tied, and two declined. Wide-node MIQP ran ten times on two feeders and proposed no true additional group. Of seven branches within two physical edges of the substation missed by both NJ and RNJ, the pipeline recovered four, all already in its warm starts. Three of the 24 planned random feeders were excluded because their input checksums differed from the archive. [Named feeders](offline.md) · [Random feeders](random_time_limits.json) · [Near-substation branches](../root_ablation/README.md)

The two stages can also move physical accuracy in opposite directions. On a separate hidden-load variant, `medium_003_end`, the raw Tree MIQP solution found an eight-meter branch missed by NJ, RNJ and the warm start, and removed two false groups. F1 rose from 0.250 to 0.400. A subsequent Node MIQP improved the fitting objective by adding a false 13-meter group. Final F1 was 0.375. [Recorded integer solutions and branch changes](../root_ablation/README.md)

## Scheduled probing and remaining evidence

Across 28 simulated policy trajectories, no dispatch was purchased. In the main experiment, the proposed policy had feasible offers at all 74 eligible decisions. The cheapest offer with positive predicted gain cost CNY 43.04 per score unit, against a preset value of CNY 5. Of five initially unresolved groups, further passive measurements ruled out one within five hours. Four remained unresolved after 24 hours. [Probe results](probe_results.md)

1. **Estimate the incremental recovery rate on fresh feeders.** With identical measurements, warm starts, required groups and solver limits, compare Tree MIQP alone, Tree MIQP with narrow-node enumeration, and the same procedure with wide-node MIQP enabled. Count true and false additions at each stage. For each missing true line, record whether it was absent from the Tree MIQP pool, could be formed from intact children, was proposed by Node MIQP and survived tree reselection.
2. **Separate search limits from errors in the fitting score.** All 21 random feeders returned feasible trees after relaxing the time limits. Six final Tree MIQP solves reached the prescribed gap within 30 seconds. For a missed group that the model can represent, compare feasible scores and bounds with the group required and excluded in the same pool. This distinguishes an unfinished solve from a case where the fitted objective favors the wrong topology. [Solver records](random_time_limits.json)
3. **Measure the benefit of an executed probe.** Pair active and passive trajectories on the same unresolved feeder, with the same initial observations and independently set procurement prices. Measure paid cost, final grouping accuracy and time to a correct decision that holds through the end of the window.

## Which implementation produced these results?

The experiments used the reviewed `topology_study` identification and probing routines. The passive benchmarks used one anchored round and up to two line additions. The reference commit before the line-exchange experiment was `22907e5b`. The entire `ptid/method` directory is identical at that commit and at the exchange experiment's commit `a337fabc`. The first imported version, `10623c45`, predates fixes to candidate handling and line scoring. On the 21 random feeders, **0.6217** is the mean F1 for the full pre-exchange pipeline. **0.6148** is the result of the later single-step experiment allowing a line exchange from a saved tree. [Passive runner settings](https://github.com/tutu-er/Topo/blob/22907e5b22d06e5a338092a2dd043d74d740bf80/topology_study/ptid/runs/four_offline.py#L41) · [Probe policy call](https://github.com/tutu-er/Topo/blob/22907e5b22d06e5a338092a2dd043d74d740bf80/topology_study/ptid/runs/four_probe.py#L302) · [Exchange results](../line_exchange/README.md)
