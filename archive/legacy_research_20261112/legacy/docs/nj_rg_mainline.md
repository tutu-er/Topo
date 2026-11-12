# NJ/RG hidden-tree mainline

The active topology path is now:

1. Generate noisy terminal P/Q/V with the AC power-flow solver.
2. Fit symmetric nonnegative reduced R/X matrices over one or more scenarios.
3. Convert R/X to terminal additive distances.
4. Reconstruct an unrooted hidden tree with neighbor joining (NJ) or recursive grouping (RG).
5. Evaluate hidden topology by terminal splits. Hidden node identifiers are never compared directly.
6. Perturb the already noisy measurements, refit NJ and RG, and retain only base-data splits that persist under both algorithms.
7. Contract small, disjoint, consensus-stable split sides into pseudo nodes, aggregate P/Q and mean squared voltage, and reconstruct the pseudo layer again.

`terminal_equivalent_minimum_distance_tree` is intentionally disabled. It does not create hidden nodes and is not part of this mainline.

## Interfaces

- `terminal_case33.graph.latent_tree.neighbor_joining(distance, terminals)`
- `terminal_case33.graph.latent_tree.recursive_grouping(distance, terminals, tolerance)`
- `terminal_case33.graph.latent_tree.terminal_splits(edges, terminals)`
- `terminal_case33.estimation.multiscenario.fit_projected_sensitivity(scenarios, alpha)`
- `terminal_case33.estimation.multiscenario.distance_candidates(R, X)`
- `python -m experiments.run_latent_tree_stability_aggregation`

The pseudo-node voltage is currently the square root of the mean terminal squared voltage. This is a proxy, not an exact hidden-parent voltage. A service-drop or subtree-drop reconstruction should be introduced before treating aggregation gains as a final algorithmic result.

## Output interpretation

- `oracle_split_f1`: reconstruction from the exact additive distance; this checks the NJ/RG implementation.
- `baseline_split_f1`: reconstruction from measured noisy AC scenarios.
- `aggregated_split_f1`: split score after consensus-stable pseudo aggregation and expansion.
- `stable_cluster_precision`: diagnostic available only in simulation; it must not be used for parameter selection on real data.
- `base_forced_merges`: RG fallback count when the estimated distance is not sufficiently additive at the configured tolerance.
