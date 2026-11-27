# Small candidate-pool audit

Four synthetic inputs test how the candidate pool affects a meter grouping’s identification. All four use the same four metered leaves, 96 hourly rows, two physical meter groupings, and seed 20261127. The paired variants switch an unmetered load on or off and scale the independent fluctuations of active and reactive power by 1 or 0.15. The daily component and voltage noise draws are shared. The data come from the linear voltage model, with PCC power constructed so that the model's balance formula recovers the simulated load. These cases test computation and sensitivity to excitation.

| Independent fluctuation scale | Unmetered leaf | Candidate supports, initial / final / all | Compatible trees, initial / final / all | Initially decided inside the first tree → open in full pool | Initially uncovered meter groups → open in full pool |
| --- | --- | --- | --- | --- | --- |
| 1 | no | 2 / 2 / 10 | 4 / 4 / 26 | 0 / 2 | 0 / 8 |
| 1 | yes | 9 / 9 / 25 | 32 / 32 / 236 | 0 / 2 | 0 / 8 |
| 0.15 | no | 6 / 10 / 10 | 13 / 26 / 26 | no inside groups | 4 / 4 |
| 0.15 | yes | 15 / 25 / 25 | 96 / 236 / 236 | no inside groups | 5 / 5 |

Under stronger excitation the two groupings in each initial tree stayed decided in the full pool (4/4 case–group pairs). Under weaker excitation the star was best even in the full pool. The initial pools already covered both true meter groupings in all four cases. In the weak pair the remaining nine initially uncovered case–group pairs each admitted a competitive tree after expansion. An uncovered alternative is recorded separately from an excluded one. This experiment uses one metered topology and one shared noise draw; it does not estimate population error rates or the contribution of individual builders.

In the weak cases the generating tree scored 4.5942 H units above the full-pool minimum without the unmetered leaf and 4.7898 above it with that leaf. Each true group individually had a competitive supporting tree in both cases. In the latter input those separate witnesses therefore did not certify that the generating augmented tree as a whole was within the 4.6-unit margin.

Among the 236 augmented trees in each load-present case, the full-domain minimum plus 4.6 admitted 5 trees under stronger excitation (1 meter-grouping set, 4 hidden internal-support paths) and 27 under weaker excitation (12 meter-grouping sets, 15 paths), with zero trees on the numerical boundary. A path records the metered descendants of each internal support carrying the hidden leaf, in root-to-leaf order, so attachment positions with the same sequence share a path; the JSON gives two concrete trees with identical meter groupings and different paths for each input with a hidden leaf.

## Audit and verification

1. Run the identifier with its nine builders. Save its first feasible tree, the initial pool, the final cumulative pool and the final labels. The gate uses the measured inputs in both load branches. The first solve uses the normal empty and builder starts, the winning local tree, and the tree MIQP. For these small inputs the initial per-start limit is 2 seconds and each tree/flip MIQP has a 10-second limit. Load-evidence estimation is disabled because it does not affect the topology or labels.
2. Enumerate every compatible family of internal supports over 4 or 5 total leaves, including the unmetered leaf when present. Singleton supports and the root are common to every family. Directly construct its time-centred, meter-weighted design and fit box-constrained least squares with SciPy BVLS. This shares the identifier's estimated variance and penalty, and evaluates the tree independently of its Gram-matrix factorization and mixed-integer search.
3. For each fitted tree, evaluate the score **H** at its feasible coefficients as an upper bound. Minimise the tangent plane of the convex quadratic over the same box to obtain a lower bound. For a given pool take the minima of these bounds over all admissible families, separately with and without each grouping. Groupings refer to the metered leaves; both possible positions of the unmetered leaf count. All domains are evaluated free of anchor constraints.
4. Hold the first tree and its threshold **H(first) + 4.6** fixed across the three pools. A feasible opposite tree below the threshold (allowing for the reference fit interval) establishes `open`. A valid opposite lower bound above the threshold establishes `decided`. Use a numerical guard of 1e−6 in H units and retain any ambiguous case as `unsettled`. These are independent audit verdicts. The identifier itself only reports selected outside groupings. Its actual final labels are also checked against enumeration relative to its own final tree.

All 524 tree fits completed. The widest fixed-tree interval was below 4e−10 in H units. The largest difference between independent and production scores of a first or final tree was below 2e−12. All 24 conclusive final labels agreed with the final-pool enumeration; no final label was unsettled. The four first and four anchored tree solves reported optimal status. These sizes did not exercise the wide-node MIQP. Detailed feasible scores, lower bounds, domain membership, presence/absence minima, seeds, timings and normalized source hashes are in [results.json](results.json).

## Reproduce

From `topology_study`, run:

```powershell
python -B -m ptid.runs.pool_audit --out pool_audit_rerun.json
```

The recorded run used Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3 and Gurobi 13.0.1. It requires a usable Gurobi license for the production identifier. No external case files are needed. Compare the per-case bounds, pool membership and grouping verdicts when rerunning; timings depend on the machine.
