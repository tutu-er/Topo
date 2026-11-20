# First Four Completed Loss Pairs

This November 17, 2026 checkpoint covered 32 completed tasks from the long panel. The four completed `101` loss pairs used 96 training points, repeat 0, and the same physical window.

| Input | MSE runtime (s) | MAE runtime (s) | Exact full support |
| --- | ---: | ---: | --- |
| Flynn, 0.5% added P/Q error | 174.25 | 344.14 | Both |
| Flynn, 2% added P/Q error | 152.65 | 349.68 | Both |
| Paper, 0.5% added P/Q error | 72.09 | 180.01 | Both |
| Paper, 2% added P/Q error | 71.39 | 173.67 | Both |

Validation, new-day, and cumulative selection agreed on all four exact recoveries. Total runtime was 470.39 seconds for MSE and 1,047.50 seconds for MAE. Native W accounted for 418.35 and 815.55 seconds respectively. W fixed-support trials accounted for 3.34 seconds across 607 QPs and 114.79 seconds across 616 LPs. E used 46.76 and 102.95 seconds respectively. Both losses made 50 native calls with zero time limits.

The independent evaluator found no score or native-bound conflict. One MSE call logged two Windows temporary-file replacement errors. Its in-memory trace and final artifact survived. The [completed study](README.md) reports the full denominators and final paired comparison.
