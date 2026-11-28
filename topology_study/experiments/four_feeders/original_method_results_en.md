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

## Code changes by stage

The first study snapshot already contained `tree_miqp()`, `node_miqp()` and `choose_block()`. When the code moved from `identification_v8`, the Node MIQP's continuous variables were rescaled by their design-column norms to improve numerical conditioning. This preserves the set of possible line additions. The probing code also began checking a flip for a group outside the tree against a known feasible alternative. [Migration record](../../CHANGES.md) · [First study snapshot](https://github.com/tutu-er/Topo/tree/10623c45a116cfda6fbb61cd160e86056f718085/topology_study/ptid/method)

The later changes trace a line from the warm start to the integer solution, record which alternatives the solver checked and isolate the effect of wide-node search. The nine tree builders and the defaults `K = 9`, `m = 4.6` and `lam = 1.96` are the same in the imported and reviewed versions.

| Stage | Functions in `ptid/method/` | Change and practical effect | Commits |
|---|---|---|---|
| Node MIQP and its ablation | `search.py` · `node_miqp`, `best_addition`. `identify.py` · `Settings`, `solve`, `price` | Record the feasible score and any available lower bound for a single insertion at a wide junction. From the same first tree, A turns off `node_refinement`, B turns off only `wide_node_miqp` and retains enumeration up to nine children, and C enables both. The run records the warm tree, raw integer tree and retained tree separately. | [Node bounds](https://github.com/tutu-er/Topo/commit/041f2ee7f3852db3bee09f6939eabd92a21f4e33) · [Tree-only switch](https://github.com/tutu-er/Topo/commit/9979e501b23033611601684de065dee9f5fe566b) · [Wide-node switch and paired first solve](https://github.com/tutu-er/Topo/commit/4cf1c29644ae1ee90f695fbbd16daa49164dcf3b) |
| Tree MIQP and line-flip scoring | `identify.py` · `refine`, `improving_flip`. `miqp.py` · `tree_miqp`. `labels.py` · `flip_test` | Keep proposed lines in the candidate pool even when the current tree omits them. Tie each bound to the candidate set and required groups actually solved. Refit and check a tree that reverses a grouping. A better feasible tree can supply a new start or become the final tree when it satisfies the relevant anchors. | [Candidate pools and score bounds](https://github.com/tutu-er/Topo/commit/041f2ee7f3852db3bee09f6939eabd92a21f4e33) · [Reuse of feasible alternatives](https://github.com/tutu-er/Topo/commit/9979e501b23033611601684de065dee9f5fe566b) |
| Line penalty in controlled comparisons | `model.py` · `Model`. `identify.py` · `Settings.p_s` | An experiment can fix the number of parameters counted in the penalty for each line to one or two, so both arms use the same penalty. The default estimates that number from active- and reactive-power measurements. | [Penalty control](https://github.com/tutu-er/Topo/commit/9979e501b23033611601684de065dee9f5fe566b) |
| Scheduled Probe | `probe.py` · `placement_rival`, `band_interval`, `feasible_blocks`, `choose_block` | Score an alternative unmetered-load location against the current fitted tree. Reject an offer if even a meter unaffected by the action would lie outside the voltage band. The policy and experiment runner now draw on the same feasible offer generator. | [Placement and voltage checks](https://github.com/tutu-er/Topo/commit/9979e501b23033611601684de065dee9f5fe566b) · [Shared offers and dispatch replay](https://github.com/tutu-er/Topo/commit/879b7b836b5723aef34b5b0ef0e02c9c1bca42b3) |

The regression tests cover candidate retention after a tree re-solve, agreement between node bounds and exhaustive enumeration, feasible alternatives for a line flip, and voltage and placement checks for probing. [Pool and node tests](../../ptid/tests/test_pool_pricing.py) · [Score-bound tests](../../ptid/tests/test_score_bounds.py) · [Probe tests](../../ptid/tests/test_probe_rivals.py)

### Experiment entry points

From `topology_study`, run `python -B experiments/run.py --help` to list the commands. The following names select the corresponding files in `ptid/runs/`. [Tracked experiment dispatcher](https://github.com/tutu-er/Topo/blob/a337fabcc5d2ce5a1a52e53da1c47ca2261fc37e/topology_study/experiments/run.py)

| Experiment command | Implementation | What it checks |
|---|---|---|
| `node`, `wide-feeders` | `node_mechanism.py`, `wide_feeders.py` | Compare the best single insertion with exhaustive enumeration. On the 24- and 36-meter feeders, compare wide-node search off and on from the same anchored Tree MIQP output. [Fixed-node check](https://github.com/tutu-er/Topo/commit/4cf1c29644ae1ee90f695fbbd16daa49164dcf3b) · [Wide feeders](https://github.com/tutu-er/Topo/commit/e38d6115fc1ad765e0be64c3f6191daa35a99fdf) |
| `offline`, `root-ablation`, `random-budget` | `four_offline.py`, `root_ablation.py`, `random_budget.py` | Compare the full pipeline with the passive baselines, trace where a recovered branch first appeared and rerun the same verified random-feeder inputs with longer solver limits. [Named feeders](https://github.com/tutu-er/Topo/commit/a8d140d29611d7e2674331ec29dc36bf17fa472a) · [Branch attribution](https://github.com/tutu-er/Topo/commit/487120452668041c43bd840fe14f93dc6e1a3573) · [Time limits](https://github.com/tutu-er/Topo/commit/22907e5b22d06e5a338092a2dd043d74d740bf80) |
| `probe`, `probe-summary` | `four_probe.py`, `four_probe_summary.py` | Run one- or two-day dispatch policies, deliver the ordered action in the AC simulation and compare the observed outcomes with passive operation. The rolling policy calls the original `choose_block()` routine. [Dispatch runner](https://github.com/tutu-er/Topo/commit/879b7b836b5723aef34b5b0ef0e02c9c1bca42b3) |
| `line-exchange`, `exchange-reference` | `line_exchange.py`, `exchange_cases.py`, `exchange_reference.py` | In a separate experiment, contract one non-anchored line and test a replacement at its parent junction. Compare greedy and exact searches and check a fixed junction against enumeration. The original identification path uses single-line addition. [Exchange implementation](https://github.com/tutu-er/Topo/commit/33c48000d21a837d9b5dc2cf4ff8a3b80a203900) · [21-feeder comparison](https://github.com/tutu-er/Topo/commit/a337fabcc5d2ce5a1a52e53da1c47ca2261fc37e) |
