"""Lock and validate the new resource case; no topology fitting or DR search.

48 linear programs find each hour's coupled upward/downward headroom. Seven
predeclared schedules then pass through the existing original AC routine.
These are deterministic resource checks, not independent recovery trials.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from ... import io
from .model import CustomerCase


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_RUN = ROOT/'outputs/linear_ac_population_20261113_v2'
PROFILE_HOURS = [0, 6, 8, 12, 16, 19, 22, 24]
PROFILE_LOAD_FACTORS = [.45, .65, .75, .60, .65, .90, .60, .45]
AC_CLOCK_HOURS = [12, 19, 1]


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(out, runtime_run):
    began = perf_counter()
    if out.exists():
        raise FileExistsError('Preserve previous checks; choose a new output directory: '+str(out))
    out.mkdir(parents=True)
    case = CustomerCase()
    sources = [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('case.json'),
               Path(__file__).parents[1]/'expected_logodds_admm_20261113/evidence.py']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        id=case.config['id'], purpose='Resource configuration, linear feasibility and deterministic AC checks only',
        sources={str(p.relative_to(ROOT)): io.ref(p)['sha256'] for p in sources}, environment=io.environment(),
        resource_case=case.config, runtime_run=str(runtime_run),
        reference_inflexible_load='Original case nominal load times the locked piecewise-linear diurnal profile',
        profile_clock_hours=PROFILE_HOURS, profile_load_factors=PROFILE_LOAD_FACTORS,
        decisions='08:00 to next 08:00; 24 hourly device changes, each held for four 15-minute observations',
        root_voltage_pu=1.02, predeclared_ac_clock_hours=AC_CLOCK_HOURS,
        ac_schedules='No DFS and both net-demand extrema at each of the three predeclared hours; no result-based replacement',
        numerical_voltage_screen_pu=[.95, 1.05],
        limitations=['Synthetic device placement, reference day and quotations; not empirical calibration.',
                    'Hourly headroom extrema come from separate feasible schedules and are not simultaneously deliverable.',
                    'No candidate generation, frozen-H adjustment, DR-evidence optimization or posterior-recovery study.',
                    'AC checks report terminal voltage and convergence; branch ampacity/transformer thermal limits are not available in this fixture.',
                    'No artificial baseline PV curtailment; restoration is unavailable in the main case.'],
        inputs_to_device_LPs=['resource configuration only; no physical branches, H, truth, voltage or future measured power'])
    io.write(out/'protocol.json', protocol)
    io.write(out/'started.json', dict(pid=os.getpid(), protocol=io.ref(out/'protocol.json')))
    for source in sources:
        dest = out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    zero = np.zeros(case.shape)
    zero_audit = case.audit(zero)
    if not zero_audit['passed']:
        raise ArithmeticError('Baseline is not device feasible')
    rows, plans = [], {'no_dfs': zero}
    failures = []
    for h, clock in enumerate(case.clock):
        row = dict(decision_index=h, clock_hour_start=int(clock-.5), up_net_demand_kw=None,
                   down_net_demand_kw=None, up_witness_service_yuan=None, down_witness_service_yuan=None)
        for name, sign in [('up', -1.), ('down', 1.)]:
            direction = np.zeros(case.shape)
            direction[h] = sign
            try:
                point, receipt = case.linear_optimum(direction, least_fee=True)
                value = float(point[h].sum())
                row[name+'_net_demand_kw'] = max(0., value if name == 'up' else -value)
                row[name+'_witness_service_yuan'] = receipt['resource_audit']['service_fee_yuan']
                plans[f'{name}_{int(clock-.5):02d}'] = point
            except Exception as exc:
                failures.append(dict(stage='device_LP', hour=h, direction=name, error=repr(exc)))
        rows.append(row)
    csv_write(out/'hourly_headroom.csv', rows)
    np.savez_compressed(out/'private_device_schedules.npz', baseline_kw=case.baseline,
                        lower_kw=case.lower, upper_kw=case.upper, **plans)
    ac_rows, physical_arrays, runtime_ref, reference_summary = [], {}, None, {}
    try:
        from ..unified_posterior_dr_20261112.physical import open_runtime
        module, batch, prior_protocol = open_runtime(runtime_run, 'main')
        network = module.load_case(case.config['network_case'])
        terminals = [b for b in network['buses'] if b['is_terminal']]
        labels = tuple(b['bus_id'] for b in terminals)
        if labels != case.labels:
            raise ValueError('Physical terminal order differs; bind the device case explicitly before execution')
        runtime_ref = dict(source=io.ref(runtime_run/'physical/source_runtime/main/data.py'),
                           original_batch=str(batch), network_case=case.config['network_case'],
                           physical_case_files={p.name: io.ref(p) for p in
                             (runtime_run/'physical/source_runtime/main/cases'/case.config['network_case']).glob('*')})
        shape = np.repeat(np.interp(case.clock, PROFILE_HOURS, PROFILE_LOAD_FACTORS), case.repeat)
        # This is a new synthetic inflexible-demand fixture, not an old net
        # observation onto which device baselines may accidentally be added twice.
        p0 = shape[:, None]*np.array([b['pd_kw'] for b in terminals])
        q0 = shape[:, None]*np.array([b['qd_kvar'] for b in terminals])
        bp, bq = case.device_baseline_pq_kw()
        net_p, net_q = p0+bp, q0+bq
        root = np.full(96, protocol['root_voltage_pu'])
        base_value = case.grid_energy_value(net_p)
        reference_summary = dict(network_bus_count=len(network['buses']),
            nominal_inflexible_load_kw=sum(b['pd_kw'] for b in terminals),
            nominal_system_base_kw=network['base_kw'], nominal_voltage_kv=network['base_kv'],
            reference_inflexible_energy_kwh=float(case.dt/case.repeat*p0.sum()),
            reference_net_terminal_energy_kwh=float(case.dt/case.repeat*net_p.sum()),
            reference_loss_free_net_energy_value_yuan=base_value)
        chosen = ['no_dfs']+[f'{s}_{h:02d}' for h in AC_CLOCK_HOURS for s in ('up', 'down')]
        for name in chosen:
            if name not in plans:
                failures.append(dict(stage='AC', schedule=name, error='Predeclared LP schedule unavailable'))
                continue
            u = plans[name]
            dp, dq = case.aggregate(case.delivery(u))
            try:
                voltage, audit = module._ac_voltages(network, case.labels,
                    (net_p+dp)/network['base_kw'], (net_q+dq)/network['base_kw'], root)
                energy = case.grid_energy_value(net_p+dp)
                physical_arrays[name+'_P_kw'] = net_p+dp
                physical_arrays[name+'_Q_kvar'] = net_q+dq
                physical_arrays[name+'_V_pu'] = voltage
                ac_rows.append(dict(schedule=name, status='converged',
                    device_feasible=case.audit(u)['passed'],
                    minimum_voltage_pu=float(voltage.min()), maximum_voltage_pu=float(voltage.max()),
                    terminal_cells_outside_screen=int(((voltage < .95) | (voltage > 1.05)).sum()),
                    maximum_voltage_update=audit['maximum_voltage_update'],
                    peak_terminal_aggregate_apparent_kva=float(np.sqrt((net_p+dp).sum(1)**2+(net_q+dq).sum(1)**2).max()),
                    service_compensation_yuan=case.fee(u),
                    loss_free_net_energy_value_yuan=energy, change_loss_free_net_energy_value_yuan=energy-base_value))
            except Exception as exc:
                failures.append(dict(stage='AC', schedule=name, error=repr(exc), traceback=traceback.format_exc()))
        physical_arrays.update(inflexible_P_kw=p0, inflexible_Q_kvar=q0, root_pu=root)
        np.savez_compressed(out/'private_physical_reference.npz', **physical_arrays)
    except Exception as exc:
        failures.append(dict(stage='physical_runtime', error=repr(exc), traceback=traceback.format_exc()))
    if ac_rows:
        csv_write(out/'ac_checks.csv', ac_rows)
    summary = dict(id=case.config['id'], terminal_count=case.N, resource_count=case.R,
        controlled_terminal_count=len(set(case.node)), controlled_terminal_labels=sorted(set(r['node_label'] for r in case.resources)),
        hourly_decision_count=case.dimension, integer_variables=0,
        counts={k: sum(r['kind'] == k for r in case.resources) for k in ('pv', 'ev', 'thermal')},
        pv_dc_total_kwp=sum(r['dc_peak_kwp'] for r in case.resources if r['kind'] == 'pv'),
        pv_ac_total_kva=sum(r['inverter_kva'] for r in case.resources if r['kind'] == 'pv'),
        ev_charger_total_kw=sum(r['power_kw'] for r in case.resources if r['kind'] == 'ev'),
        ev_daily_battery_energy_kwh=sum(r['battery_energy_required_kwh'] for r in case.resources if r['kind'] == 'ev'),
        ev_daily_grid_energy_kwh=sum(r['battery_energy_required_kwh']/r['efficiency'] for r in case.resources if r['kind'] == 'ev'),
        thermal_electrical_power_kw=sum(r['power_kw'] for r in case.resources if r['kind'] == 'thermal'),
        thermal_water_litres=case.config['thermal']['water_litres'],
        pv_reference_generation_kwh=-case.dt*sum(case.baseline[:, j].sum() for j, r in enumerate(case.resources) if r['kind'] == 'pv'),
        thermal_reference_electric_energy_kwh=case.dt*sum(case.baseline[:, j].sum() for j, r in enumerate(case.resources) if r['kind'] == 'thermal'),
        zero_action_device_audit=zero_audit, lp_schedules_completed=len(plans)-1,
        maximum_hourly_up_net_demand_kw=max((r['up_net_demand_kw'] or 0.) for r in rows),
        maximum_hourly_down_net_demand_kw=max((r['down_net_demand_kw'] or 0.) for r in rows),
        ac_schedules_completed=len(ac_rows),
        ac_schedules_inside_voltage_screen=sum(r['terminal_cells_outside_screen'] == 0 for r in ac_rows),
        failures=failures, limitations=protocol['limitations'],
        actual_recovery_tests=0, global_DR_cost_optimality_claim=False,
        seconds=perf_counter()-began, **reference_summary)
    io.write(out/'summary.json', summary)
    io.write(out/'private_runtime_binding.json', runtime_ref)
    io.write(out/'complete.json', dict(status='passed' if not failures and len(ac_rows) == 7 and
        summary['ac_schedules_inside_voltage_screen'] == 7 else 'complete_with_findings',
        protocol=io.ref(out/'protocol.json'), summary=io.ref(out/'summary.json'),
        solver_scope='Global linear resource headroom only; no DR QCQP cost or posterior result'))
    print(io.read(out/'complete.json')['status'], summary['id'],
          'LP', summary['lp_schedules_completed'], 'AC', summary['ac_schedules_completed'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-run', type=Path, default=DEFAULT_RUNTIME_RUN)
    args = parser.parse_args()
    run(args.output.resolve(), args.runtime_run.resolve())
