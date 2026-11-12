# Paper-Style Terminal-Only LV Case

This case is a compact synthetic feeder built to sit between the very small
debug LV case and terminalized IEEE case33.

## Motivation

The terminalized case33 feeder is useful because it preserves IEEE case33
loads/topology, but its long backbone creates strong common-path voltage modes.
After removing the root voltage, many downstream squared-voltage curves remain
highly correlated because they share most upstream branch drops.

The paper-style LV case instead follows the topology assumptions in the
terminal-meter literature:

- known root/transformer bus;
- hidden internal branching nodes;
- no load or meters at internal nodes;
- all P/Q/V smart-meter measurements at terminal leaves;
- hidden internal node degree at least three, avoiding unrecoverable degree-2
  chains.

## Topology

- 22 buses total.
- 1 observable root.
- 6 hidden internal zero-injection branching nodes.
- 15 observed terminal loads.
- 21 closed radial edges.
- 3 balanced main laterals from the root.
- Every hidden internal node has degree 4.
- Synthetic line parameters are chosen with R and X in the same order of
  magnitude: overhead `R=0.42 Ohm/km, X=0.34 Ohm/km`; service
  `R=0.55 Ohm/km, X=0.38 Ohm/km`. This differs from the Soumalas-style LV
  service cable setting where service `X/R` can be very small; the balanced
  library is used here to avoid making X sensitivity numerically negligible.
- To match LV engineering voltage-drop assumptions, all synthetic branch
  impedances are multiplied by `5.0`, and the transformer secondary/root voltage
  is generated around `1.02 pu`. In the default stressed run this places terminal
  voltage magnitudes in the intended `0.94-1.00 pu` range.

## Data

The active/reactive profile generator reuses the current heterogeneous curve
logic:

- residential/commercial/industrial demand shapes inspired by End-Use Load
  Profiles / BuildingsBench;
- PV shapes inspired by NSRDB-style irradiance diversity;
- wind shapes inspired by WIND Toolkit site-specific ramps and lulls;
- reactive-power curves from time-varying demand power factor plus DER apparent
  power capability constraints.

The generated assignment deliberately uses mixed DER terminals:

- PV-only;
- wind-only;
- small-generator-only;
- PV + wind;
- PV + small generator;
- wind + small generator.

## Current Output

Run:

```bash
python -m experiments.run_paper_style_terminal_lv_case --output outputs/paper_style_terminal_lv --T 96 --pq-noise-rel 0.005 --v-noise-rel 0.0002 --root-voltage-sigma 0.0008
```

Current metrics:

- observed terminals: 15;
- hidden internal nodes: 6;
- mixed DER component nodes: 10;
- P/Q measurement noise std: 0.500% of instantaneous true P/Q;
- V measurement noise std: 0.020% of instantaneous true voltage magnitude;
- root voltage Gaussian fluctuation std: 0.0008 pu;
- root voltage range: 1.018102 to 1.022112 pu;
- mean off-diagonal P-curve correlation: 0.096;
- mean off-diagonal squared-voltage-drop correlation: 0.188;
- terminal voltage range: 0.939032 to 0.998176 pu;
- all AC power-flow snapshots converged.

Compared with terminalized case33, this new case substantially reduces the
voltage common-path dominance while preserving the hidden-node terminal-meter
setting.
