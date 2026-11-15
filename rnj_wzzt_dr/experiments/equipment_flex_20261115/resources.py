"""History-registered, explicitly synthetic aggregate equipment schedules.

No candidate, true network, forecast or future observation enters placement or
the declared device baselines. Registered kW is scaled once, then 10% joins the
contract. Fractional aggregate capacity is not a measured integer device count.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json

import numpy as np

from ..measured_pq_flex_20261114.domain import PRICING


KINDS = (('pv', 'PV_systems_kW', 3, .30),
         ('ev', 'EV_chargers_kW', 2, .10),
         ('thermal', 'hot_water_tanks_kW', 2, .15),
         ('battery', 'batteries_kW', 2, .10))


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _window(clock, start, end):
    return ((clock >= start) & (clock < end) if start < end
            else (clock >= start) | (clock < end))


def build_spec(context, labels, start_date, fraction=.1):
    """Select top registered capacities and declare reproducible equipment.

    ``start_date`` is the history start, used for registration eligibility and a
    fixed seasonal solar shape. The synthetic shape is not measured irradiance.
    Device baselines are a decomposition of an existing net profile, never an
    additional network load. All powers below are already in network units.
    """
    labels = list(map(int, labels))
    start = date.fromisoformat(str(start_date)[:10])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Nonempty unique terminal labels required')
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Participation fraction must be in (0,1]')
    mapping = deepcopy(context['node_mapping'])
    if [int(row['label']) for row in mapping] != labels:
        raise ValueError('Resource context and measured terminal order differ')
    if len({int(row['feeder_id']) for row in mapping}) != len(mapping):
        raise ValueError('Each measured terminal needs a distinct source feeder')
    registrations = {}
    for row in context['resource_metadata_at_history_start']:
        feeder = int(row['feeder'])
        if feeder in registrations:
            raise ValueError('Duplicate source registration')
        if date.fromisoformat(str(row['date'])[:10]) > start:
            raise ValueError('Registration after history start is unavailable')
        registrations[feeder] = row
    clock = (8.+(np.arange(96)+.5)/4.) % 24.
    clock_h = (8.+np.arange(24)+.5) % 24.
    # Fixed approximate seasonal daylight, rather than a fitted weather model.
    # 8--16 h range and solar noon 12 local are declared engineering choices.
    doy = start.timetuple().tm_yday
    daylight = 12.+4.*np.sin(2.*np.pi*(doy-80.)/365.25)
    sunrise, sunset = 12.-daylight/2., 12.+daylight/2.
    phase = (clock-sunrise)/daylight
    solar_shape = np.where((phase > 0.) & (phase < 1.),
        np.maximum(0., np.sin(np.pi*phase))**1.5, 0.)
    devices = []
    for kind, field, count, price in KINDS:
        available = []
        for node in mapping:
            feeder, scale = int(node['feeder_id']), float(node['scale'])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Positive finite source-to-network scale required')
            registration = registrations[feeder]
            registered = float(registration.get(field, 0.) or 0.)
            if not np.isfinite(registered) or registered < 0:
                raise ValueError('Registered kW must be finite and nonnegative')
            if registered == 0:
                continue
            scaled = registered*scale
            label = int(node['label'])
            available.append(dict(id=f'{kind}_{label}', physical_device_id=f'{kind}_{label}',
                kind=kind, node_label=label, feeder_id=feeder, metadata_field=field,
                registration_date=str(registration['date']), registered_kw=registered,
                source_to_network_scale=scale, scaled_registered_kw=scaled,
                participation_fraction=float(fraction), power_kw=float(fraction*scaled),
                contract_kw=float(fraction*scaled),
                old_resource_id=f'{kind}_{label}' if kind != 'battery' else None,
                fee_positive_yuan_per_kwh=price, fee_negative_yuan_per_kwh=price,
                count_interpretation='Synthetic participating aggregate; no integer device count inferred'))
        for device in sorted(available, key=lambda row: (-row['scaled_registered_kw'], row['node_label']))[:count]:
            power = device['power_kw']
            p0 = np.zeros(96)
            if kind == 'pv':
                available_p = power*solar_shape
                p0 = -available_p
                device.update(inverter_kva=power, reactive_cap_kvar=.4*power,
                    available_generation_kw=available_p.tolist(),
                    reactive_availability='Only hours whose four available-generation samples are strictly positive',
                    baseline_curtailment_fraction=0.,
                    interpretation='AC gross PV synthetic available power; full available generation at baseline; curtailment only')
            elif kind == 'ev':
                p0[_window(clock, 22, 4)] = .5*power
                device.update(connected_clock_hours=[18, 7], baseline_charging_clock_hours=[22, 4],
                    charging_efficiency=.95, required_battery_energy_kwh=.95*3.*power,
                    v2g=False, interpretation='Ordinary charging only; signed delta shifts an existing feasible charging baseline')
            elif kind == 'thermal':
                p0[:] = .2*power
                p0[_window(clock, 6, 9)] = .4*power
                p0[_window(clock, 17, 22)] = .5*power
                litres = 150.*power
                capacity = litres*4.186/3600.
                initial, ambient = 47.5, 20.
                # SHW300S: 300 L, minimum selectable 2 kW, standby 1.80
                # kWh/day at 65 C. Ambient20 C is our explicit assumption.
                # Linear first-order hourly heat balance: rho=1-UA*dt/C.
                ua_reference = (1.80/24.)/(65.-ambient)
                ua = ua_reference*power/2.
                rho = 1.-ua/capacity
                demand = p0-(1.-rho)*capacity*(initial-ambient)
                if np.any(demand < 0):
                    raise ValueError('Declared constant-temperature baseline requires negative heat demand')
                device.update(cop=1., water_litres=litres,
                    heat_capacity_kwh_per_C=capacity, water_heat_capacity_kJ_per_kg_K=4.186,
                    hourly_temperature_retention=rho, initial_temperature_C=initial,
                    minimum_temperature_C=40., maximum_temperature_C=55., ambient_temperature_C=ambient,
                    heat_demand_kw=demand.tolist(), terminal_temperature_C=initial,
                    equivalent_thermal_loss_coefficient_kw_per_C=ua,
                    thermal_reference=dict(model='STIEBEL ELTRON SHW300S', volume_litres=300.,
                        selected_reference_power_kw=2., standby_kwh_per_24h_at_65C=1.80,
                        reference_water_temperature_C=65., assumed_ambient_temperature_C=ambient,
                        reference_UA_kw_per_C=ua_reference,
                        url='https://www.stiebel-eltron.com/en/home/products-solutions/dhw/small_water_heaterswallandfloormountedcylinders/floorstanding_cylindersfrom200l/shw-s/shw-300-s.html',
                        reading_scope='Manufacturer technical data webpage; aggregate scaling and ambient20C are declared assumptions'),
                    interpretation='Resistive hot-water aggregate; 300 L per 2 kW reference scaling; UA from stated standby loss; hourly Euler thermal balance; constant baseline temperature and declared fixed heat demand')
            else:
                device.update(energy_kwh=2.*power, inverter_kva=power,
                    reactive_cap_kvar=.4*power, charging_efficiency=.95,
                    discharging_efficiency=.95, minimum_soc=.2, maximum_soc=.9,
                    initial_soc=.5, terminal_soc=.5,
                    charging_clock_hours=[10, 16], discharging_clock_hours=[18, 23],
                    reactive_clock_windows=[[10, 16], [18, 23]],
                    baseline_self_discharge_per_hour=0., auxiliary_load_included_in_background=True,
                    interpretation='Independent AC-coupled aggregate; 2 h energy/power; directions prelocked; signed Q only during contracted online windows')
            device.update(baseline_p_kw=p0.tolist(), baseline_q_kvar=np.zeros(96).tolist())
            devices.append(device)
    if not devices:
        raise ValueError('No positive registered capacity for the four equipment classes')
    spec = dict(schema='equipment_flex_v1', expected_terminal_labels=labels,
        history_start=start.isoformat(), participation_fraction=float(fraction),
        time=dict(start_hour=8, hours=24, decision_hours=1., measurement_minutes=15,
                  clock_quarter_midpoints=clock.tolist(), clock_hour_midpoints=clock_h.tolist()),
        devices=devices, pricing=deepcopy(PRICING), reactive_fee_yuan_per_kvarh=.05,
        placement='Top3 PV, top2 EV, top2 hot water, top2 batteries by scaled registered kW, ties by node label; no candidate or truth',
        solar=dict(day_of_year=doy, daylight_hours=float(daylight), sunrise_hour=float(sunrise),
                   sunset_hour=float(sunset), shape_exponent=1.5, peak_available_fraction=1.,
                   source='Declared deterministic seasonal engineering shape; not measured weather, inferred gross PV, or a forecast trained on future observations'),
        baseline_rule='net_new = original_measured_net + sum(device_actual - declared_device_baseline); no extra baseline is added',
        residual_background='Net baseline minus synthetic participating-device baselines; may include uncontracted generation. Forecast-only decomposition diagnostics must be reported.',
        actual_device_availability='Same predeclared deterministic baseline and availability as planning; no hidden weather realization or net-export clipping',
        response='One whole-day eta in [0.9,1] per physical device, shared by its P and Q columns; eta scales increments toward the feasible baseline',
        limitations='Registration provides kW, not inventory counts, inverter kVA, SOC, gross generation, or remote-control authorization; these are explicit synthetic engineering contracts',
        assumptions=dict(inverter_kva_equals_participating_kw=True, Q_fraction_of_kva=.4,
            no_standalone_SVG=True, battery_two_hour_energy=True, device_baseline_measured=False,
            thermal_reference='SHW300S 300L/2kW and standby1.80kWh/day at65C; ambient20C assumed; UA scaled by power; rho=1-UA/C. Temperature40--55C and initial47.5C remain declared service choices'),
        context_content_sha256=content_hash(context))
    spec['spec_content_sha256'] = content_hash(spec)
    return spec
