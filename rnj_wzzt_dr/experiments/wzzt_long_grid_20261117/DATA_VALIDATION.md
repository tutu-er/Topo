# Data Qualification

The November 17, 2026 qualification passed all 192 planned inputs before fitting. It independently reconstructed the saved arrays and checked their sources.

| Check | Recorded result |
| --- | --- |
| Planned inputs | 160 main-grid, 16 root-meter, and 16 fixed-window sampling inputs. The long panel reuses 16 main-grid inputs. |
| Observation arrays | 576 train, validation, and new-day NPZ files. All 1,609,920 stored squared-voltage drops matched the reconstruction. |
| Noise draws | 32 complete streams reconstructed with the recorded split, component, and scaling rules |
| Provenance | 1,027 unique file hashes, 24 runtime or source-snapshot entries, and 18 original inputs checked |
| AC consistency | Maximum KCL error 6.40e-13 pu, branch voltage-drop identity error at most 3.88e-18 pu, and power-balance error at most 4.01e-10 kVA |

Clean simulated bus voltages ranged from 0.998192 to 1.024592 pu for Paper, 0.997606 to 1.024495 for Flynn, 1.004763 to 1.024464 for Soumalas, and 0.997405 to 1.024878 for Pengwah. All lie within 0.95 to 1.05 pu. The preparation used baseline loads and contains no device intervention. The source networks did not specify branch thermal ratings.

The standardized eight-day P/Q designs were full rank across 30, 32, 22, and 36 columns respectively. Per-node P/Q correlations ranged from approximately −0.174 to 0.678. After regressing Q on P, the residual standard deviation was still 73.5% to 99.6% of Q's standard deviation. Full rank permits weak directions, so these checks characterize the inputs to the comparison.

The original P/Q meter error is unknown. The 0% setting refers to additional simulated error. The local evidence lives in the `data/wzzt_long_grid_data_20261117_v1/` completion, protocol, qualification, attempt, and world records.
