"""Independently audit saved P/Q actions and full AC states, without solving.

Only NumPy and the standard library are used. All electrical quantities are
reconstructed from the saved complex bus voltages and the sealed branch table.
No fitting, optimization, AC iteration, or production scoring is called.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def maxabs(value):
    return float(np.max(np.abs(value), initial=0.))


def stats(value):
    x = np.asarray(value, float)
    return dict(minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()))


class Audit:
    def __init__(self):
        self.maxima = defaultdict(float)
        self.files = {}

    def same(self, name, a, b, tolerance=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'{name}: shape or finite-value mismatch {a.shape} {b.shape}')
        error = maxabs(a-b)
        self.maxima[name] = max(self.maxima[name], error)
        if error > tolerance:
            raise ArithmeticError(f'{name}: {error} > {tolerance}')

    def input(self, path, expected=None):
        item = ref(path)
        if expected is not None and item['sha256'] != expected:
            raise ValueError('Changed input '+str(path))
        self.files[item['path']] = item
        return Path(path)


def resource_values(config, action, labels, forecast_p, base):
    """Check scalar resource equations, and independently rebuild nodal deltas."""
    labels = list(map(int, labels)); rows = config['resources']
    if action.shape != (24, len(rows)) or not np.isfinite(action).all():
        raise ValueError('Invalid hourly action')
    dp = np.zeros((96, len(labels))); dq = np.zeros_like(dp)
    lower_violation = upper_violation = energy_violation = final_error = 0.
    devices = []
    for j, row in enumerate(rows):
        value = action[:, j]; node = labels.index(int(row['node_label']))
        unit = config['action_units'][j]; clock = (8+np.arange(24)+.5) % 24
        state = 0.; states = []; price = np.ones(24)
        if unit == 'kvar':
            cap = float(row['capacity_kvar']); lo = np.full(24, -cap); hi = -lo
            hours = row.get('available_clock_hours')
            if hours is not None and hours != list(range(24)):
                aa, bb = hours
                valid = (clock >= aa) & (clock < bb) if aa < bb else (clock >= aa) | (clock < bb)
                lo[~valid] = hi[~valid] = 0.
            price = np.asarray(row.get('hourly_fee_multiplier', price), float)
            cost = price*(row['fee_positive_yuan_per_kvarh']*np.maximum(value, 0)+row['fee_negative_yuan_per_kvarh']*np.maximum(-value, 0))
            dq[:, node] += np.repeat(value, 4)
        else:
            cap = float(row['contract_kw']); kind = row['kind']
            if kind == 'pv':
                export = np.maximum(0., -forecast_p[:, node]*base).reshape(24, 4).min(1)
                lo = np.zeros(24); hi = np.where((clock >= 10) & (clock < 14), np.minimum(cap, export), 0.)
            else:
                valid = (clock >= 18) | (clock < 7) if kind == 'ev' else np.ones(24, bool)
                if kind not in ('ev', 'thermal'):
                    raise ValueError('Unknown resource '+kind)
                lo = np.where(valid, -cap, 0.); hi = -lo
                rho, buffer = (1., 2*cap) if kind == 'ev' else (.98, cap)
                for v in value:
                    state = rho*state+v; states.append(float(state))
                energy_violation = max(energy_violation, maxabs(states)-buffer)
                final_error = max(final_error, abs(state))
            price[(clock < 7)] = .8
            price[((clock >= 7) & (clock < 10)) | ((clock >= 17) & (clock < 22))] = 1.25
            cost = price*(row['fee_positive_yuan_per_kwh']*np.maximum(value, 0)+row['fee_negative_yuan_per_kwh']*np.maximum(-value, 0))
            dp[:, node] += np.repeat(value, 4)
        lower_violation = max(lower_violation, float(np.max(lo-value)))
        upper_violation = max(upper_violation, float(np.max(value-hi)))
        devices.append(dict(id=row['id'], kind=row['kind'], node=int(row['node_label']),
            unit=unit, capacity=cap, absolute_energy=float(np.abs(value).sum()),
            signed_energy=float(value.sum()), maximum_absolute_action=maxabs(value),
            positive_energy=float(np.maximum(value, 0).sum()), negative_energy=float(np.maximum(-value, 0).sum()),
            utilization_fraction=maxabs(value)/cap, activation_yuan=float(cost.sum()),
            materially_active=bool(maxabs(value) > 1e-4), maximum_abs_buffer_kwh=maxabs(states), final_buffer_kwh=float(state)))
    violations = dict(lower=max(0., lower_violation), upper=max(0., upper_violation),
        energy=max(0., energy_violation), terminal_energy=final_error)
    if max(violations.values()) > 1e-7:
        raise ArithmeticError('Resource domain violation '+str(violations))
    return devices, violations, dp/base, dq/base


def electrical(audit, ac, obs, metadata, branches, bus_rows):
    """Ohm law, KCL and complex power balance from stored states, not receipts."""
    base = 1000*float(metadata['base_mva']); kv = float(metadata['base_kv'])
    zbase = kv*kv/float(metadata['base_mva'])
    buses = list(map(int, ac['bus_ids'])); positions = {v: i for i, v in enumerate(buses)}
    root = positions[int(metadata['root_bus'])]
    source = np.array([positions[int(v)] for v in ac['branch_from']])
    sink = np.array([positions[int(v)] for v in ac['branch_to']])
    mapping = {(int(r['from_bus']), int(r['to_bus'])): r for r in branches}
    z = np.array([complex(float(mapping[int(a), int(b)]['r_ohm']), float(mapping[int(a), int(b)]['x_ohm']))/zbase
        for a, b in zip(ac['branch_from'], ac['branch_to'])])
    v = ac['voltage_complex_pu']; magnitude = np.abs(v)
    current = (v[:, source]-v[:, sink])/z
    amps = np.abs(current)*base/(np.sqrt(3)*kv)
    sending = v[:, source]*current.conj()*base
    receiving = v[:, sink]*current.conj()*base
    losses = z*np.abs(current)**2*base
    load = ac['specified_load_kw']+1j*ac['specified_load_kvar']
    out = np.zeros_like(v); incoming = np.zeros_like(v)
    for k, (a, b) in enumerate(zip(source, sink)):
        out[:, a] += current[:, k]; incoming[:, b] += current[:, k]
    net_i = incoming-out-np.conj(load/base/v)
    net_i[:, root] = 0.
    root_power = v[:, root]*out[:, root].conj()*base+load[:, root]
    balance = root_power-load.sum(1)-losses.sum(1)
    terminal = np.array([positions[int(label)] for label in obs['labels']])
    hidden = np.array([positions[int(r['bus_id'])] for r in bus_rows if r['bus_type'] == 'hidden_internal'])
    audit.same('magnitude', magnitude, ac['voltage_magnitude_pu'], 1e-12)
    audit.same('terminal_true_V', magnitude[:, terminal], obs['true_V'], 1e-12)
    audit.same('terminal_cached_V', magnitude[:, terminal], ac['terminal_voltage_pu'], 1e-12)
    audit.same('root_voltage', magnitude[:, root], obs['true_root'], 1e-12)
    audit.same('specified_P', load[:, terminal].real, obs['true_P']*base)
    audit.same('specified_Q', load[:, terminal].imag, obs['true_Q']*base)
    audit.same('hidden_zero_injection', load[:, hidden], np.zeros_like(load[:, hidden]), 1e-12)
    audit.same('branch_complex_I', current, ac['branch_current_complex_pu'], 1e-12)
    audit.same('branch_amps', amps, ac['branch_current_A'], 1e-9)
    for name, calculated in [('branch_sending', sending), ('branch_receiving', receiving), ('branch_loss', losses)]:
        audit.same(name+'_P', calculated.real, ac[name+'_kw'])
        audit.same(name+'_Q', calculated.imag, ac[name+'_kvar'])
    audit.same('root_supply_P', root_power.real, ac['root_supply_kw'])
    audit.same('root_supply_Q', root_power.imag, ac['root_supply_kvar'])
    audit.same('KCL_pu', net_i, np.zeros_like(net_i), 1e-7)
    audit.same('global_balance_kVA', balance, np.zeros_like(balance), 1e-5)
    audit.same('branch_balance_kVA', sending-receiving-losses, np.zeros_like(losses), 1e-8)
    cells = int(magnitude.size)
    metrics = dict(all_bus_voltage=stats(magnitude), terminal_voltage=stats(magnitude[:, terminal]),
        hidden_voltage=stats(magnitude[:, hidden]), root_voltage=stats(magnitude[:, root]),
        measured_terminal_voltage=stats(obs['V']), measured_root_voltage=stats(obs['root']),
        all_bus_voltage_cells=cells, cells_below_095=int((magnitude < .95).sum()), cells_above_105=int((magnitude > 1.05).sum()),
        cells_outside_090_110=int(((magnitude < .90) | (magnitude > 1.10)).sum()),
        maximum_branch_current_A=float(amps.max()), root_P_kw=stats(root_power.real), root_Q_kvar=stats(root_power.imag),
        root_S_kVA=stats(np.abs(root_power)), total_net_load_P_kw=stats(load.real.sum(1)), total_net_load_Q_kvar=stats(load.imag.sum(1)),
        active_loss_kw=stats(losses.real.sum(1)), reactive_loss_kvar=stats(losses.imag.sum(1)),
        active_loss_kwh=float(.25*losses.real.sum()), reactive_loss_kvarh=float(.25*losses.imag.sum()),
        root_signed_kwh=float(.25*root_power.real.sum()), root_import_kwh=float(.25*np.maximum(root_power.real,0).sum()),
        root_export_kwh=float(.25*np.maximum(-root_power.real,0).sum()),
        max_KCL_pu=maxabs(net_i), max_power_balance_kVA=maxabs(balance), max_branch_balance_kVA=maxabs(sending-receiving-losses),
        maximum_voltage_above_root_pu=float(np.max(magnitude-magnitude[:, root, None])),
        maximum_voltage_below_root_pu=float(np.max(magnitude[:, root, None]-magnitude)))
    branch_metrics = [dict(from_bus=int(a), to_bus=int(b), max_current_A=float(amps[:, k].max()),
        max_sending_kVA=float(np.abs(sending[:, k]).max()), max_receiving_kVA=float(np.abs(receiving[:, k]).max()))
        for k, (a,b) in enumerate(zip(ac['branch_from'], ac['branch_to']))]
    bus_metrics = [dict(bus=int(bus), minimum_voltage_pu=float(magnitude[:, k].min()), maximum_voltage_pu=float(magnitude[:, k].max()))
        for k,bus in enumerate(buses)]
    return metrics, branch_metrics, bus_metrics


def summarize(rows):
    complete = [r for r in rows if r['audit_status'] == 'passed']
    result = dict(rows=len(rows), passed=len(complete), failures=len(rows)-len(complete))
    if not complete:
        return result
    numeric = ['maximum_branch_current_A','active_loss_kwh','reactive_loss_kvarh','root_signed_kwh',
        'root_import_kwh','root_export_kwh','max_KCL_pu','max_power_balance_kVA','max_branch_balance_kVA',
        'maximum_voltage_above_root_pu','maximum_voltage_below_root_pu']
    nested = ['all_bus_voltage','terminal_voltage','hidden_voltage','root_voltage','measured_terminal_voltage',
        'measured_root_voltage','root_P_kw','root_Q_kvar','root_S_kVA','active_loss_kw','reactive_loss_kvar',
        'total_net_load_P_kw','total_net_load_Q_kvar']
    for key in numeric:
        result[key] = stats([r['electrical'][key] for r in complete])
    for key in nested:
        items = [r['electrical'][key] for r in complete]
        result[key] = dict(minimum=min(x['minimum'] for x in items), maximum=max(x['maximum'] for x in items),
            mean=float(np.mean([x['mean'] for x in items])))
    for key in ['all_bus_voltage_cells','cells_below_095','cells_above_105','cells_outside_090_110']:
        result[key] = sum(r['electrical'][key] for r in complete)
    result['voltage_excursion_paths'] = sum(r['electrical']['cells_below_095']+r['electrical']['cells_above_105']>0 for r in complete)
    for phase in ('requested','delivered'):
        result[phase] = {}
        for unit, suffix in [('kW','P'),('kvar','Q')]:
            result[phase][suffix] = {key:stats([sum(d[field] for d in r[phase] if d['unit']==unit) for r in complete])
                for key,field in [('absolute_energy','absolute_energy'),('activation_yuan','activation_yuan')]}
            result[phase][suffix]['materially_active_paths'] = sum(any(d['materially_active'] for d in r[phase] if d['unit']==unit) for r in complete)
        names = sorted({d['id'] for r in complete for d in r[phase]})
        result[phase]['devices'] = {}
        for name in names:
            items = [d for r in complete for d in r[phase] if d['id']==name]
            result[phase]['devices'][name] = dict(node=items[0]['node'], kind=items[0]['kind'], unit=items[0]['unit'],
                capacity=stats([d['capacity'] for d in items]), paths_available=len(items), active_paths=sum(d['materially_active'] for d in items),
                **{key:stats([d[key] for d in items]) for key in ['absolute_energy','signed_energy','maximum_absolute_action','utilization_fraction','activation_yuan']})
    branches = sorted({(b['from_bus'], b['to_bus']) for r in complete for b in r['branches']})
    result['branches'] = [dict(from_bus=a,to_bus=b,**{k:max(v[k] for r in complete for v in r['branches'] if (v['from_bus'],v['to_bus'])==(a,b))
        for k in ['max_current_A','max_sending_kVA','max_receiving_kVA']}) for a,b in branches]
    buses = sorted({b['bus'] for r in complete for b in r['buses']})
    result['buses'] = [dict(bus=bus,minimum_voltage_pu=min(v['minimum_voltage_pu'] for r in complete for v in r['buses'] if v['bus']==bus),
        maximum_voltage_pu=max(v['maximum_voltage_pu'] for r in complete for v in r['buses'] if v['bus']==bus)) for bus in buses]
    result['maximum_resource_violation'] = max(max(r[k].values()) for r in complete for k in ['requested_violations','delivered_violations'])
    return result


def run(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve previous audit '+str(destination))
    if not (source/'complete.json').exists():
        raise ValueError('Completed source run required')
    destination.mkdir(parents=True)
    shutil.copyfile(__file__, destination/'operational_audit.py')
    audit = Audit(); original = read(audit.input(source/'protocol.json'))
    entries = read(audit.input(original['inputs']['path'], original['inputs']['sha256']))
    modes = list(original['modes'])
    strategies = ['no_dfs']+[name for mode in modes for name in (mode, mode+'_rotation')]
    if len(strategies) != len(set(strategies)):
        raise ValueError('Duplicate declared strategy')
    if original.get('policies') is not None and original['policies'] != strategies:
        raise ValueError('Policy list differs from declared modes')
    entry_ids = [entry['id'] for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError('Duplicate input identity')
    cases = {entry['case'] for entry in entries}
    if len(cases) != 1:
        raise ValueError('This operational audit requires one physical network per run')
    expected_tasks = {(entry['id'], draw) for entry in entries for draw in range(int(original['forecasts']))}
    expected_paths = {(repeat, strategy) for repeat in range(int(original['repeats'])) for strategy in strategies}
    expected_rows = len(expected_tasks)*len(expected_paths)
    source_complete = read(audit.input(source/'complete.json'))
    main_ref = source_complete['main']
    completion = read(audit.input(main_ref['path'], main_ref['sha256']))
    if not source_complete.get('source_unchanged', False):
        raise ValueError('Completed run reports source drift')
    if completion['total_forecasts'] != len(expected_tasks):
        raise ValueError('Completion forecast denominator differs from protocol')
    completed_tasks = {(r['id'], int(r['forecast_index'])) for r in completion['receipts']}
    failed_tasks = {(r['id'], int(r['forecast_index'])) for r in completion['failed_tasks']}
    if (len(completed_tasks) != len(completion['receipts']) or len(failed_tasks) != len(completion['failed_tasks'])
            or completed_tasks & failed_tasks or completed_tasks | failed_tasks != expected_tasks):
        raise ValueError('Missing, duplicate, or unexpected task in completion arrays')
    if completion['completed_forecasts'] != len(completed_tasks):
        raise ValueError('Completed-forecast count differs from receipt array')
    meta_dir = source/'source_snapshot/topology_identification_standalone/cases'/next(iter(cases))
    metadata = read(audit.input(meta_dir/'metadata.json'))
    with audit.input(meta_dir/'branches.csv').open(encoding='utf-8-sig', newline='') as f:
        branches = list(csv.DictReader(f))
    with audit.input(meta_dir/'buses.csv').open(encoding='utf-8-sig', newline='') as f:
        bus_rows = list(csv.DictReader(f))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=ref(__file__),
        saved_source=ref(destination/'operational_audit.py'), source_run=ref(source/'protocol.json'),
        source_completion=ref(source/'complete.json'), environment=dict(python=sys.version,numpy=np.__version__),
        scope='Stored arrays only; no fitting, optimization, production audit/score, or AC solve. Independently reconstruct Ohm law, KCL, flows, losses, P/Q resources and fees.',
        expected_rows=expected_rows, expected_forecasts=len(expected_tasks), expected_rows_per_forecast=len(expected_paths),
        strategies=strategies, input_count=len(entries), forecasts_per_input=int(original['forecasts']),
        executions_per_forecast=int(original['repeats']), material_action_tolerance_native_units=1e-4,
        voltage_reporting_band_pu=[.95,1.05], wider_voltage_reporting_band_pu=[.90,1.10],
        simulation_model='Balanced positive-sequence three-phase powers, no shunts; hidden buses are zero injection.',
        ratings=dict(base_mva=metadata['base_mva'],base_kv=metadata['base_kv'],base_kw=1000*metadata['base_mva'],
            base_is_not_transformer_capacity=True, transformer_rating_kva=None, branch_ampacity_A=None,
            branch_table_columns=list(branches[0]), bus_table_columns=list(bus_rows[0]), metadata=metadata),
        rating_scope='Original files contain base_mva only and no explicit transformer nameplate, branch ampacity, or transformer model. Report actual current/apparent power; do not compute loading percentages.',
        accounting='P in kWh, Q in kvarh separately; actual activation prices only. No reserve, investment or converter-loss charge. 96 quarter-hour states and 24 hourly commands.')
    write(destination/'protocol.json', protocol)
    records=[]; task_errors=[]
    for identity, draw in sorted(expected_tasks):
        folder = source/'fraction0.10'/identity/f'forecast{draw:02d}'
        try:
            if (identity, draw) in failed_tasks:
                raise RuntimeError('Original task failed; original completion record retained')
            task = read(audit.input(folder/'complete.json'))
            if task['id'] != identity or task['forecast_index'] != draw or task['rows'] != len(expected_paths):
                raise ValueError('Task completion identity or denominator differs')
            config = read(audit.input(folder/'resource_case.json')); planner = arrays(audit.input(folder/'planner.npz'))
            if set(planner) != {'forecast', *strategies}:
                raise ValueError('Planner strategy set differs from protocol')
            forecast = planner['forecast']; combined = read(audit.input(task['results']['path'],task['results']['sha256']))
            keys = [(int(r['repeat']), r['strategy']) for r in combined]
            if len(keys) != len(set(keys)) or set(keys) != expected_paths:
                raise ValueError('Missing, duplicate, or unexpected execution within task')
            if task['failed_rows'] != sum(r['status'] != 'complete' for r in combined):
                raise ValueError('Task failed-row count differs from saved results')
            if any(r['id'] != identity or r['forecast_index'] != draw for r in combined):
                raise ValueError('Execution task identity differs')
            base=1000*float(metadata['base_mva'])
            for saved in combined:
                path = folder/f"execution{int(saved['repeat']):02d}"/saved['strategy']
                record = dict(resource_variant=saved['resource_variant'], strategy=saved['strategy'], forecast=saved['forecast_index'],
                    repeat=saved['repeat'], source=str(path), original_status=saved['status'])
                try:
                    receipt=read(audit.input(path/'receipt.json'))
                    if receipt != saved:
                        raise ValueError('Individual and combined receipt differ')
                    if saved['status']!='complete':
                        raise ValueError('Original execution incomplete: '+saved['status'])
                    physical_ref=saved['physical_state']
                    ac=arrays(audit.input(physical_ref['path'],physical_ref['sha256']))
                    if 'time_slice' in physical_ref:
                        start,stop=physical_ref['time_slice']; points=len(ac['voltage_complex_pu'])
                        ac={k:(value[start:stop] if value.ndim and value.shape[0]==points else value)
                            for k,value in ac.items()}
                    obs=arrays(audit.input(path/'observations.npz'))
                    baseline=arrays(audit.input(path.parent/'no_dfs/observations.npz'))
                    eta=np.asarray(saved['delivery']['multipliers'],float)
                    request=obs['requested_action']; delivered=obs['delivered_action']
                    audit.same('sealed_requested_action',request,planner[saved['strategy']],0.)
                    expected=request*eta[None]
                    for j,res in enumerate(config['resources']):
                        if res['kind']=='pv':
                            col=list(map(int,obs['labels'])).index(res['node_label'])
                            cap=np.minimum(res['contract_kw'], np.maximum(0.,-baseline['true_P'][:,col]*base).reshape(24,4).min(1))
                            expected[:,j]=np.minimum(expected[:,j],cap)
                    audit.same('whole_day_response_and_PV_clip',delivered,expected,1e-12)
                    nodes=len(obs['labels'])
                    requested,rv,dp,dq=resource_values(config,request,obs['labels'],forecast[:,:nodes],base)
                    actual,av,ap,aq=resource_values(config,delivered,obs['labels'],forecast[:,:nodes],base)
                    audit.same('delivered_P_mapping',baseline['true_P']+ap,obs['true_P'],1e-12)
                    audit.same('delivered_Q_mapping',baseline['true_Q']+aq,obs['true_Q'],1e-12)
                    audit.same('planned_fee',np.array(sum(d['activation_yuan'] for d in requested)),np.array(saved['planned_fee_yuan']),1e-8)
                    audit.same('settled_fee',np.array(sum(d['activation_yuan'] for d in actual)),np.array(saved['settled_fee_yuan']),1e-8)
                    for unit, key in [('kW','active_abs_kwh'),('kvar','reactive_abs_kvarh')]:
                        audit.same('requested_'+key,np.array(sum(d['absolute_energy'] for d in requested if d['unit']==unit)),np.array(saved['requested_quantities'][key]),1e-8)
                        audit.same('delivered_'+key,np.array(sum(d['absolute_energy'] for d in actual if d['unit']==unit)),np.array(saved['delivered_quantities'][key]),1e-8)
                    e, bm, vm=electrical(audit,ac,obs,metadata,branches,bus_rows)
                    record.update(audit_status='passed',requested=requested,delivered=actual,requested_violations=rv,
                        delivered_violations=av,electrical=e,branches=bm,buses=vm)
                except Exception as exc:
                    record.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
                records.append(record)
        except Exception as exc:
            task_errors.append(dict(path=str(folder),error=repr(exc),traceback=traceback.format_exc()))
        write(destination/'progress.json',dict(last_forecast=str(folder),rows=len(records),errors=sum(r['audit_status']!='passed' for r in records),task_errors=len(task_errors)))
    grouped=defaultdict(list)
    for r in records:
        grouped[r['resource_variant']+'/'+r['strategy']].append(r)
    report=dict(status='passed' if len(records)==protocol['expected_rows'] and not task_errors and all(r['audit_status']=='passed' for r in records) else 'failed',
        protocol=ref(destination/'protocol.json'), overall=summarize(records),groups={k:summarize(v) for k,v in grouped.items()},
        maximum_absolute_errors=dict(audit.maxima),task_errors=task_errors,records=records,
        original_failed_tasks=completion['failed_tasks'], original_execution_failed_rows=completion.get('execution_failed_rows'),
        original_quote_exceptions=completion.get('quote_exceptions'),
        preserved_scope='Audit verifies saved balanced AC states and declared contracts. Thermal security needs missing branch/transformer ratings. Repeated same background day is not a daily operating population.')
    write(destination/'inputs_manifest.json',list(audit.files.values()))
    write(destination/'report.json',report)
    write(destination/'complete.json',dict(status=report['status'],report=ref(destination/'report.json'),inputs=ref(destination/'inputs_manifest.json'),source_unchanged=ref(__file__)['sha256']==protocol['source']['sha256']))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=run(args.run,args.out)
    print(json.dumps(dict(status=result['status'],rows=result['overall']['rows'],passed=result['overall']['passed'],
        errors=result['overall']['failures'],task_errors=len(result['task_errors'])),ensure_ascii=False))
    raise SystemExit(0 if result['status']=='passed' else 1)
