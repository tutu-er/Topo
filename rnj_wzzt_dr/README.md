# RNJ/WZZT and Conditional Demand Response

Start with the [project README](../README.md) for installation, input arrays, CLI commands, and the latest study results.

The public Python API exposes `Observation`, `load_observations`, `fit_observed`, and `freeze_history`. Use `prepare_dr` and `zero_screen` to evaluate the frozen candidate bank under a supplied forecast and error model. `load_resources`, `pair_gate`, and `plan` provide access to the device allocation and the single-competitor optimizer.

- [Model and scoring](MODEL.md) explains the fitted response, structural stages, output contraction, and DR probability bound.
- [Validation](VALIDATION.md) describes the tests and the scope of the evidence.
- [Data and archives](ARCHIVE.md) lists local assets and explains how to replay sealed research.
- [Research history](HISTORY.md) points to earlier implementations and records.
