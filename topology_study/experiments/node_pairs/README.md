# Paired node refinement on one synthetic feeder

Node refinement improved the penalised voltage fit in one of four matched comparisons. The added line was absent from both candidate domains, remained in the final tree and did not belong to the generating feeder. The other three comparisons returned identical trees with refinement on and off.

The [development protocol](../../PROTOCOL.md) specifies the design. One linear-model feeder has 11 meters and one internal group, `{2,8}` (zero-based), giving the physical junction ten children. The 96-hour inputs share the periodic load, random draws and voltage noise. The independent load fluctuations have scale 1 or 0.15. The nine builders and the identifier see the same P, Q and voltage inputs in both arms. Truth is used to produce the physical measurements and to evaluate returned trees. The historical 11-child development case files were unavailable in this workspace.

| Fluctuation scale | Fixed penalty dimension | Final true / extra groups, off → on | Final score H, off → on | Identification time (s), off → on |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 / 3 → 1 / 3 | 509.0674 → 509.0674 | 4.26 → 4.49 |
| 1 | 2 | 1 / 0 → 1 / 1 | 515.6148 → 515.5510 | 3.70 → 5.17 |
| 0.15 | 1 | 1 / 3 → 1 / 3 | 508.2855 → 508.2855 | 4.78 → 4.90 |
| 0.15 | 2 | 1 / 0 → 1 / 0 | 515.3898 → 515.3898 | 4.19 → 4.97 |

Both inputs selected `p_s=2` under the automatic P/Q rule. Fixing it to 1 yielded three additional spurious groups in each input. Scores are comparable within rows. Within each pair the code verifies identical input hashes, builder pools, converged local-search starts, first Tree MIQP trees and starting scores. The pools contained 39 and 40 lines and both already covered the one true group.

With strong excitation and `p_s=2`, the Node MIQP at the ten-child root found `{0,1,4,6,9,10}`. The feasible score fell by 0.06387 H after the tree re-solve. Physical-group F1 fell from 1 to 2/3. The extra line was open at the final flip test. The re-solve's lower bound was 515.54258 in its anchored domain. With weak excitation and the same penalty the wide-node scan excluded a single-line gain above tolerance. With `p_s=1` the fitted tree had no junction wider than four children and used enumeration. No better flip witness was recorded in the eight runs.

## What happened when the early anchor was released

Only the strongly excited input produced an initial anchor, the true group `{2,8}`, which stayed anchored in both arms at each penalty. A separate diagnostic removed the anchor from the common first tree, kept the other lines and fitted every child union at its original parent that crossed the old anchor.

| Fixed penalty dimension | All crossing replacements fitted | Outside the initial pool | Best outside-pool H increase over the first tree |
| --- | --- | --- | --- |
| 1 | 6 / 6 | 5 | 108.441 |
| 2 | 1,022 / 1,022 | 1,018 | 132.063 |

None reached the 4.6-unit margin. This check covers one replacement at the parent with the remaining branches fixed. The two weak-input runs had no early anchor to release. These observations concern one feeder and one noise draw. Recovery of a genuinely missing physical group remains open.

## Reproduction and checks

From `topology_study`, with NumPy, SciPy and a licensed Gurobi installation:

```powershell
python -B -m ptid.runs.node_pairs --out node_pairs_rerun.json
```

All eight arms finished under the same 120 s process limit (4.49–5.98 s per arm including startup and independent scoring). Each arm used one anchored round, up to two additions, one second per first local-search pass and four seconds per tree, flip or wide-node solver call. The comparison uses the same maximum budget, with actual times shown above. Every first and anchored tree solve reported optimal to solver tolerance. All four enabled final node scans excluded a single-line gain above the pricing tolerance. The status of an earlier accepting scan is not retained by the identifier. The independent SciPy box fit on the explicit voltage design agreed with all eight final scores to within 1.1e−11 H units. The post-decision anchor checks had a separate 60 s allowance per anchor and both finished.

[Per-arm bounds, case and source hashes, labels, anchors and witnesses](results.json)
