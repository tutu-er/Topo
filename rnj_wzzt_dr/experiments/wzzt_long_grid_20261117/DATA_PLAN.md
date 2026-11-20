# Data Sources and Preparation

For the final task grid and budgets, see the [executed design](PROTOCOL.md). The [original planning record](https://github.com/tutu-er/Topo/blob/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/experiments/wzzt_long_grid_20261117/DATA_PLAN.md) retains the earlier grid proposals, source hashes, and historical 48-input loss comparison.

The study uses signed net active and reactive power from the same FeederBW meter and timestamp for each source feeder. The feeder profiles are load inputs to the four reference networks. Voltages come from AC simulation on those reference networks. The P/Q archive has 18 source feeders. Each observed terminal uses its assigned feeder and the same fixed positive scale for both power channels. The source archive manifest identifies FeederBW v1.1, DOI `10.5281/zenodo.19550073`.

## Required files for preparation

Paths are relative to the repository root. The runner checks these files against the locked hashes.

| Path | Contents |
| --- | --- |
| `data/measured_pq_sources_20261114/paired_pq_15min.npz` and adjacent `archive.json` | Synchronized P/Q, timestamps, and valid-minute counts |
| `data/distribution_panel_confirmation_data_20261115_v1/confirmation/<case>__2024-01-04/resource_context.json` | Earlier case mapping, positive scale, and source identity for each of the four networks |
| `data/history10_rx_20261116_v1/worlds.json` and its referenced input files | The existing 1,056-point Soumalas11 and Pengwah18 worlds, full AC states, network responses, and receipts |
| `rnj_wzzt_dr/physical/cases/` | Network branches, buses, and case metadata for the four feeders |

The stage reuses the Soumalas and Pengwah worlds. It computes Paper and Flynn AC voltages for the same window and checks the original and full-state simulators against each other. The window begins January 21, 2024 at 08:00 Central European Time. Each selected quarter-hour bin requires at least twelve valid source minutes. Missing bins cause a qualification failure. The fixed window and source mapping are shared across all methods.

| Split | Row indices within the 1,056-point window |
| --- | --- |
| Training | The last 48, 96, 192, 384, or 768 rows before index 768 |
| Validation | `[768, 960)` |
| New-day scoring | `[960, 1056)` |

The main grid changes both sample count and the history covered by consecutive observations. The evenly spaced controls hold the training interval fixed and change the number of points. References to archived paths may be absolute. Keep the sealed inputs at resolvable original or retained paths, with their stored bytes intact. See the [data and archive layout](../../ARCHIVE.md).
