# Frozen-waveform electrical location ablation: independent evaluation

Completed: 12 independent histories (989201–989212), two frozen primary methods, 24 paired method conditions. All conditions and all four candidate topologies are retained. The parent AC/operations and QP reviews passed before execution; the ablation's `review_full.json` independently replayed 432 saved fit states, including all 152 new candidate fits, and 48 true physical states. All 75 bound sources and parent input hashes matched. No selector was rerun.

For each frozen action, the comparison puts its same per-slot total EV grid power at initial site 0. Background, source interface, R/X, root voltage and white meter noise remain paired. Root P/Q and losses are recomputed. Future EV is added once. Here `M = min_wrong SSE - true_candidate SSE`, and `delta_M = M_original_location - M_site0`; positive values favor the selected spatial placement.

| Primary method | Actual delta_M mean / median | Actual + / zero / − | Noiseless-future delta_M mean / median | Noiseless + / zero / − |
|---|---:|---:|---:|---:|
| Refined mean | 9.2807 / 1.7160 | 6 / 2 / 4 | 12.8282 / 3.7711 | 8 / 2 / 2 |
| Cap20 | 60.8058 / 12.0570 | 9 / 2 / 1 | 56.7081 / 21.4441 | 10 / 2 / 0 |

Each row has n=12 histories. The two methods share histories; the 24 method conditions are not independent samples. Refined mean actual differences range from −161.8485 to 153.3339; Cap20 differences range from −9.6473 to 178.7127. The negative cases remain in `paired_locations.csv`. The noiseless branch removes future meter noise only; historical observations stay unchanged.

In the true candidate model, mean relative R/X errors for original versus site-0 placements are: refined R .02840 versus .17023, X .65066 versus .66638; Cap20 R .03079 versus .17634, X .60482 versus .63074. These are finite local fitted values. X remains weakly estimated.

Four conditions are stationary and have exact zero differences. A fifth cached condition shares a moving remap curve between the two methods. New work comprised 152 candidate fits, 4,773 fit-internal AC calls and 19 physical forward calls (4,792 total counted AC calls). All new fits report success; maximum reported optimality is .0380536. Independent residual replay passed. These software stops and repeated starts do not certify global MLEs.

Both locations use the saved H-fit line/source warm starts, two starts, max_nfev=250, tol=1e-10 and exact linear algebra; noisy seeds are policy_seed+802+candidate_index and noiseless seeds policy_seed+1202+candidate_index. Original fits are reused; remap future states initialize from their corresponding changed meters. The comparison therefore uses matched local optimization rules, with direct physical voltage/root responses saved separately to show the electrical change independently of fitted costs.

The 20 moved method conditions produce maximum per-condition voltage changes between .00228680 and .00350574 pu. Root P changes range from .000592849 to .0217776 kW and root Q changes from .000500548 to .00533148 kvar. Stationary differences are exactly zero. Grid net energy and throughput match within 8.9e-16 kWh. Canceling driving leaves about .2 kWh extra terminal battery energy in each moved condition: this is an electrical location isolation, not an equal-terminal-service equipment comparison.

This modified process has no calibrated rejection rule. Original policy thresholds are not applied, and no rejection-risk or topology-probability claim is attached to this ablation. Per-candidate future P/Q/V/root channel SSE and historical penalties are preserved in `candidate_channel_costs.csv`; direct physical responses and fit stopping metadata are in the two reviewer CSV files.

Parent policy: `033d4f3bd0077ed68216268adbd3259110f6c1b1bdb67993ad7ea44fa19ad42f`. Parent freeze SHA256: `6e746d1d4284904c9c179d6607bc4dbde83c276681d125e1685bfe1b5d79e762`. Exact source, input and parent-review bindings are in `freeze.json`; review scope and evidence hashes are in `review_full.json`.
