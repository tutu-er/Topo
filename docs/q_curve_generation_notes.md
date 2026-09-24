# Q-Curve Generation Notes

This note records the current rule used to generate reactive-power curves in
`experiments/export_current_pqv_load_curves.py`.

## Source-informed assumptions

- Building demand diversity follows the idea of using many residential and
  commercial building time series rather than one generic shape. The current
  synthetic profiles are inspired by NLR/NREL End-Use Load Profiles and
  BuildingsBench, which is extracted from the End-Use Load Profiles database.
- PV active-power diversity is based on NSRDB-style solar resource data:
  site-dependent irradiance, temporal resolution, orientation effects, and cloud
  dips. The code uses east/south/west/flat orientation variants and independent
  cloud attenuation per terminal.
- Wind active-power diversity is based on WIND Toolkit-style data: distinct
  modeled sites, meteorological variation, turbine power output, ramps, and
  lulls. The code gives each wind component a different phase, roughness, ramp,
  and stochastic AR component.
- DER reactive behavior follows common distribution-simulation practice:
  demand uses lagging, time-varying power factor; PV/wind inverters and small
  synchronous or CHP generators can inject or absorb reactive power subject to
  apparent-power capability.

## Implemented Q rule

Positive `Q` means reactive load. For each terminal:

```text
Q_net_load(t) = Q_demand(t) - Q_pv_injection(t)
              - Q_wind_injection(t) - Q_generator_injection(t)
```

Demand:

```text
Q_demand(t) = P_demand(t) tan(arccos(pf_demand(t)))
```

with class-dependent, time-varying lagging power factors:

- residential: around 0.955, lower in the evening motor/HVAC period;
- commercial: around 0.945;
- industrial/small-generator host: around 0.925.

DER components:

```text
Q_der_desired(t) = sign * P_der(t) tan(arccos(pf_der))
|Q_der(t)| <= sqrt(S_rating^2 - P_der(t)^2)
S_rating = 1.10 * P_capacity
```

The default signs make most DERs supply vars, while a subset of PV/wind nodes
absorbs vars to avoid all Q curves having the same phase and sign. This is a
reasonable stress-test convention for topology identification because it creates
distinct P/Q excitations while respecting inverter/generator capability limits.

## Why this is preferable to fixed Q/P

A fixed `Q/P` ratio makes every Q curve a scaled copy of P. That is convenient
for regression but unrealistic and can create rank/identifiability artifacts.
The current rule separates:

- demand power-factor behavior;
- PV irradiance and inverter reactive setpoint;
- wind site variability and converter setpoint;
- small-generator dispatch and capability curve.

The resulting `Q` is still reproducible and interpretable, but it is no longer
collinear with `P` at every node.

## References Used

- NLR/NREL BuildingsBench and End-Use Load Profiles:
  https://github.com/NREL/BuildingsBench
- NSRDB API documentation:
  https://developer.nrel.gov/docs/solar/nsrdb/
- WIND Toolkit documentation:
  https://www.nrel.gov/grid/wind-toolkit.html
- IEEE 1547-2018 educational resources:
  https://www.nrel.gov/grid/ieee-standard-1547
- Turitsyn, Sulc, Backhaus, Chertkov, local control of reactive power by PV:
  https://arxiv.org/abs/1006.0160
