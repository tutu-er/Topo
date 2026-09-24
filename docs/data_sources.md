# Data Sources and Citation Notes

This project separates **network topology data** from **time-series profile data**.
The built-in network data are embedded for reproducibility. External open datasets
are recommended as profile templates for load, PV, wind, and small generator
behavior, but they are not downloaded automatically.

## Network Data

### IEEE/MATPOWER case33bw

The base feeder is the MATPOWER `case33bw` distribution system. MATPOWER describes
it as a 33-bus distribution-system power-flow case from Baran and Wu. The source
case stores bus loads in kW/kvar and branch impedances in Ohm before per-unit
conversion.

Use in this project:

- raw 33-bus feeder data are embedded in `terminal_case33/data/case33bw_raw.py`;
- the terminalized case keeps the radial backbone and moves internal-node loads
  to observed terminal leaves;
- optional tie lines are retained as normally open candidate lines.

Recommended citations:

- MATPOWER `case33bw` reference page: https://matpower.org/docs/ref/matpower6.0/case33bw.html
- Baran, M. E., and Wu, F. F. "Network reconfiguration in distribution systems
  for loss reduction and load balancing." IEEE Transactions on Power Delivery,
  4(2), 1401-1407, 1989. DOI: 10.1109/61.25627.

## Load Profile Data

### NREL End-Use Load Profiles

Recommended for `load_only` terminals and for the baseline demand component of
PV/wind/generator prosumers.

NREL's End-Use Load Profiles (EULP) dataset represents major end uses, building
types, and climate regions in U.S. commercial and residential building stock.
The dataset includes one year of 15-minute consumption outputs for building
energy models, with aggregate and individual-building access paths.

Use in this project:

- sample residential/commercial profiles by building type and climate region;
- normalize each profile to unit mean or peak;
- scale by the terminal's original case33 `pd_kw`;
- keep original `Q/P` ratio or apply a configured power-factor model.

Source:

- https://www.nrel.gov/buildings/end-use-load-profiles.html

## PV Profile Data

### NREL Solar Power Data for Integration Studies

Recommended for `pv_prosumer` terminals.

NREL's Solar Power Data for Integration Studies provide synthetic PV power
plant data for the United States. The methodology page describes one year of
5-minute solar power and hourly day-ahead forecast data for about 6,000 simulated
PV plants. Its naming convention distinguishes `UPV` and `DPV`; `DPV` is the
closer template for behind-the-meter distributed PV in this project.

Use in this project:

- choose `DPV` profiles when possible;
- normalize PV output by installed capacity;
- scale by `pv_capacity_kw` in `terminal_resource_assignment.csv`;
- subtract PV generation from terminal load under the load-positive convention.

Source:

- https://www.nrel.gov/grid/solar-power-data.html

Optional alternative:

- NSRDB-derived irradiance profiles can be converted to PV output for location-
  specific studies, but this project currently records them as a future data
  integration path rather than a required input.

## Wind Profile Data

### NREL WIND Toolkit

Recommended for `wind_prosumer` terminals.

The WIND Toolkit provides meteorological conditions and calculated turbine power
for more than 126,000 continental-U.S. sites for 2007-2013. The toolkit includes
meteorological, power, and forecast datasets; the Techno-Economic subset provides
5-minute resolution data for selected points.

Use in this project:

- choose a small number of geographically coherent wind sites;
- use the power dataset or capacity-factor series;
- scale by `wind_capacity_kw`;
- subtract wind generation from terminal load.

Recommended citations from the WIND Toolkit page:

- Draxl, C., Hodge, B. M., Clifton, A., and McCaa, J. "The Wind Integration
  National Dataset (WIND) Toolkit." Applied Energy, 151, 355-366, 2015.
- Draxl, C., Hodge, B. M., Clifton, A., and McCaa, J. "Overview and
  Meteorological Validation of the Wind Integration National Dataset Toolkit."
  NREL Technical Report NREL/TP-5000-61740, 2015.

Source:

- https://www.nrel.gov/grid/wind-toolkit.html

## Small Generator / CHP Profile Data

### EIA Form 923 and PUDL

Recommended for `small_generator` terminals.

EIA Form 923 is the Power Plant Operations Report. EIA publishes monthly and
annual detailed electric-power data. PUDL documents EIA-923 as collecting
generation, fuel consumption, fossil fuel stocks, and receipts at plant and
prime-mover level, including electric and CHP plants. PUDL provides a cleaned
data pipeline and normalized tables that are easier to query than raw EIA
spreadsheets.

Use in this project:

- use EIA-923/PUDL to derive smooth dispatch templates for CHP, gas, diesel, or
  other small dispatchable generators;
- rescale the template to `small_generator_capacity_kw`;
- represent the generator as negative terminal load;
- keep the exact feeder-level dispatch synthetic unless a site-specific
  generator record is available.

Sources:

- EIA-923: https://www.eia.gov/electricity/data/eia923/
- PUDL EIA-923 documentation: https://docs.catalyst.coop/pudl/en/latest/data_sources/eia923.html

## Assignment Artifacts in This Repository

The current terminal resource assignment generated for terminalized case33 is:

- `outputs/resource_assignment_case33/terminal_resource_assignment.csv`
- `outputs/resource_assignment_case33/resource_assignment_summary.csv`
- `outputs/resource_assignment_case33/topology_case33_resource_assignment.png`

The assignment keeps all DER and load variations on observed terminal leaves.
Hidden internal nodes remain zero-injection and unobserved.

Net active load convention:

```text
P_net_load(t) = P_base_load(t) - P_pv(t) - P_wind(t) - P_small_generator(t)
```

For the default load-positive LinDistFlow convention:

```text
Delta V ~= -R Delta P_load - X Delta Q_load
```

local generation should therefore enter the terminal time series as negative
load or, equivalently, as positive net injection after changing conventions.

## Citation Files

BibTeX entries for the above sources are provided in:

- `docs/data_sources.bib`

