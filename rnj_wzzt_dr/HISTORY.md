# Research History

The current CLI is described in the [project README](../README.md). Each older study used its own input catalog, data splits, candidate-generation code, and scoring rule. The original records for this edition are pinned to the November 18 source snapshot.

- [Original research index](https://github.com/tutu-er/Topo/blob/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/HISTORY.md) covers the earlier topology, probability, device, and procurement studies.
- [Identification contribution audit](https://github.com/tutu-er/Topo/tree/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/experiments/identification_contribution_20261116) compares fixed-RNJ baselines and earlier joint-search banks under shared output processing. It also examines the actual warm start, solver return, and refitted candidate.
- [Evaluation scope](evaluation_scope.json) defines the 240, 40, and 16 active benchmark entries and the retired 160-entry pressure cohort.
- [Latest loss study](experiments/wzzt_long_grid_20261117/README.md) compares MSE and MAE under the same validation rule. [Latest W ablation](experiments/w_importance_20261117/README.md) compares the full W stage, pool-only trials, and direct E.

The repository retains the earlier experiment code and documents in their original locations. Local data and full search trajectories are available through the [archive layout](ARCHIVE.md).
