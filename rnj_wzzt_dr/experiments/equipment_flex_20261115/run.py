"""Four explicit device classes on frozen histories and paired measured worlds.

Only this isolated driver changes resource construction and device execution.
The historical single-target gates, node/hour bands, native scoring and AC
backend are reused. No evaluation context is opened before the planning seal.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import Observation
from ..frozen_rx_guard_20261115 import run as paired
from ..distribution_gate_20261114.run import planning_context
from ..distribution_gate_20261114 import solver as engine
from ..single_target_20261115.evidence import SingleTargetEvidence
from ..node_time_pq_20261115.domain import NodeTimePQCase
from ..flexibility_stability_20261114 import solver as prices


ROOT = paired.old.ROOT
MODES = paired.MODES
BASELINE = ROOT/'outputs/node_time_pq_execution_20261115_v1'
BASELINE_AUDIT = ROOT/'outputs/node_time_pq_audit_20261115_v1/report.json'
RESPONSE_NAMESPACE = 'equipment_flex_whole_device_v1'


def old_p_mapping(case, old_config):
    """Map physical IDs only to genuine old P contracts, never old SVGs."""
    rows = old_config['resources']
    units = old_config['action_units']
    if len(rows) != len(units) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Old resource IDs or units are ambiguous')
    old = {r['id']: (j, r) for j, (r, unit) in enumerate(zip(rows, units, strict=True))
           if unit == 'kW'}
    devices = {}
    for column, resource in enumerate(case.resources):
        device = resource.get('physical_device_id', resource.get('device_id'))
        if not isinstance(device, str) or not device:
            raise ValueError('Every action channel needs a physical_device_id')
        if device not in devices:
            devices[device] = dict(device_id=device, columns=[], old_p_index=None,
                                   old_resource_id=None)
        devices[device]['columns'].append(column)
        if column >= case.P_R:
            continue
        requested = resource.get('old_resource_id', device)
        if resource.get('kind') == 'battery':
            # A new battery is not the old independent-Q compensator at its bus.
            if resource.get('old_resource_id'):
                raise ValueError('New battery cannot inherit an old resource realization')
            continue
        if requested in old:
            index, original = old[requested]
            if (int(original['node_label']) != int(resource['node_label'])
                    or original['kind'] != resource['kind']):
                raise ValueError('Resource ID reused for a different physical device')
            devices[device].update(old_p_index=index, old_resource_id=requested)
    return list(devices.values())


def keyed_multiplier(seed, original_index, repeat, device_id, forecast_index=0):
    """Order-independent, private, whole-day realization for a new device."""
    key = [RESPONSE_NAMESPACE, int(seed), int(original_index), int(forecast_index),
           int(repeat), str(device_id)]
    digest = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()
    words = np.frombuffer(digest, dtype='<u4').tolist()
    value = float(np.random.default_rng(np.random.SeedSequence(words)).uniform(.9, 1.))
    return value, dict(key=key, sha256=digest.hex(), distribution='Uniform(0.9,1)',
                       generator='numpy.default_rng(SeedSequence(SHA256 little-endian uint32 words))')


def response_multipliers(case, old_config, original_eta, *, seed, original_index, repeat):
    original_eta = np.asarray(original_eta, float)
    if (original_eta.shape != (len(old_config['resources']),)
            or not np.isfinite(original_eta).all()
            or np.any((original_eta < .9) | (original_eta > 1))):
        raise ValueError('Expected sealed original whole-day resource multipliers')
    eta = np.empty(case.R)
    receipt = []
    for device in old_p_mapping(case, old_config):
        item = dict(device)
        if device['old_p_index'] is not None:
            value = float(original_eta[device['old_p_index']])
            item.update(source='reused_old_physical_P_device', keyed_stream=None)
        else:
            value, stream = keyed_multiplier(seed, original_index, repeat, device['device_id'])
            item.update(source='new_keyed_physical_device', keyed_stream=stream)
        eta[device['columns']] = value
        item['value'] = value
        receipt.append(item)
    return eta, receipt


def mapped_seeds(case, old_config, original):
    """Keep the old common portfolio lineage without relabeling Q equipment."""
    began = perf_counter()
    records, points = [], {'zero': np.zeros(case.shape)}
    if not case.audit(points['zero'])['passed']:
        raise ValueError('The unchanged net-load baseline must be device/band feasible')
    mapping = old_p_mapping(case, old_config)
    pairs = [(j, d['old_p_index']) for d in mapping for j in d['columns']
             if j < case.P_R and d['old_p_index'] is not None]
    for name, value in original.items():
        value = np.asarray(value, float)
        if value.shape != (case.H, len(old_config['resources'])) or not np.isfinite(value).all():
            raise ValueError('Old seed dimensions/values differ from the sealed catalog')
        candidate = np.zeros(case.shape)
        for new, old in pairs:
            candidate[:, new] = value[:, old]
        audit = case.audit(candidate)
        duplicate = next((key for key, point in points.items() if np.array_equal(point, candidate)), None)
        accepted = bool(audit['passed'] and duplicate is None)
        record = dict(name=name, mapped_name='mapped_'+name, physical_and_band_audit=audit,
                      accepted=accepted, duplicate_of=duplicate,
                      discarded_old_Q_columns=True, new_channels_zero=True,
                      original_array_sha256=io.contract_hash(value),
                      mapped_array_sha256=io.contract_hash(candidate))
        records.append(record)
        if accepted:
            points['mapped_'+name] = candidate
    return points, dict(seconds=perf_counter()-began, records=records,
        mapping=mapping, points=len(points), new_search=False,
        source='Original common pool mapped only by physical P-device ID; new channels zero',
        prior_final_policy_or_oracle_seeds=False,
        probability_feasibility_checked_by_engine=True)


def numeric_tree(tree, prefix=''):
    """Supplement the full JSON receipt with numeric, pickle-free trajectories."""
    result = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            result.update(numeric_tree(value, prefix+'__'+str(key) if prefix else str(key)))
    elif isinstance(tree, (list, tuple, np.ndarray)):
        try:
            value = np.asarray(tree)
        except (ValueError, TypeError):
            return result
        if value.dtype.kind in 'bifu' and value.size and np.isfinite(value).all():
            result[prefix] = value
    return result


def save_states(directory, requested, delivered=None):
    state = dict(requested=requested, delivered=delivered)
    io.write(directory/'device_states.json', state)
    np.savez_compressed(directory/'device_states.npz', **numeric_tree(state))
    return dict(json=io.ref(directory/'device_states.json'), arrays=io.ref(directory/'device_states.npz'))


def snapshot(out, entries):
    from . import domain, resources
    paths = {Path(__file__).resolve(), Path(domain.__file__).resolve(), Path(resources.__file__).resolve(),
             Path(__file__).with_name('PROTOCOL.md'), Path(__file__).with_name('MODEL.md')}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if (path.suffix == '.py' and path.is_relative_to(ROOT) and 'outputs' not in path.parts
                    and not path.name.startswith('test_')
                    and path.name not in ('audit.py', 'report.py', 'plot.py')):
                paths.add(path)
    for name in {e['case'] for e in entries}:
        for filename in ('metadata.json', 'buses.csv', 'branches.csv'):
            paths.add(ROOT/'rnj_wzzt_dr'/'physical'/'cases'/name/filename)
    for path in sorted(paths):
        destination = out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return [io.ref(path) for path in sorted(paths)]


def check_old_snapshots(source, protocol):
    receipts = []
    for ref in protocol['sources']:
        relative = Path(ref['path']).relative_to(ROOT)
        saved = source/'source_snapshot'/relative
        paired.old.checked(dict(path=str(saved), sha256=ref['sha256']))
        receipts.append(io.ref(saved))
    return receipts


def context_planning(entry):
    from .domain import EquipmentCase
    planning = planning_context(entry, entry['base_kw'])
    forecast = paired.arrays(paired.old.checked(entry['forecast_ref']))['forecast00']
    spec = io.read(paired.old.checked(entry['equipment_spec']))
    physical = EquipmentCase(spec, planning['labels'], forecast, planning['base'])
    band_meta = io.read(paired.old.checked(entry['node_time_bandwidth']))
    band_arrays = paired.arrays(paired.old.checked(band_meta['arrays']))
    case = NodeTimePQCase(physical, band_arrays, band_meta)
    old_seal = io.read(paired.old.checked(entry['prior_seal']))
    old_config = io.read(paired.old.checked(old_seal['resource_case']))
    old_seeds = paired.arrays(paired.old.checked(entry['initial_points']))
    return planning, forecast, physical, case, old_config, old_seeds


def prepare(out, *, smoke=False, seconds=1200., preflight=None, workers=3):
    from .resources import build_spec
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve previous results: '+str(out))
    if not np.isfinite(seconds) or seconds <= 0 or not 1 <= workers <= 3:
        raise ValueError('Positive finite quote budget and 1..3 workers required')
    source = io.read(BASELINE/'protocol.json')
    done, audited = io.read(BASELINE/'complete.json'), io.read(BASELINE_AUDIT)
    if (done['status'] != 'complete' or not done['source_unchanged']
            or audited['status'] != 'passed' or audited['audited_rows'] != 144):
        raise ValueError('Latest complete independently audited P/Q-band run required')
    paired.check_refs([source['inputs'], audited['source_protocol'], audited['source_inputs']])
    snapshots = check_old_snapshots(BASELINE, source)
    if not smoke and (preflight is None or io.read(preflight).get('status') != 'passed'):
        raise ValueError('Passed independent integration audit required before formal run')
    entries = deepcopy(io.read(source['inputs']['path']))
    if tuple(e['id'] for e in entries) != tuple(paired.IDS):
        raise ValueError('Keep all three original development settings')
    out.mkdir(parents=True)
    for entry in entries:
        planning = planning_context(entry, entry['base_kw'])
        spec = build_spec(planning['resource'], planning['labels'], entry['start_date'], fraction=.1)
        location = out/'equipment_specs'/entry['id']
        io.write(location/'equipment_spec.json', spec)
        entry['equipment_spec'] = io.ref(location/'equipment_spec.json')
        entry['band_baseline_directory'] = str(BASELINE/'cases'/entry['id'])
        entry['band_baseline_seal'] = io.ref(Path(entry['band_baseline_directory'])/'seal.json')
        previous = io.read(paired.old.checked(entry['band_baseline_seal']))
        paired.check_refs([previous[k] for k in ('planner', 'plans', 'score', 'train', 'validation',
                                                'bandwidth_json', 'bandwidth_npz')])
        for key, expected in (('score', entry['score']), ('train', entry['files']['train.npz']),
                              ('validation', entry['files']['validation.npz'])):
            if previous[key] != expected:
                raise ValueError('Historical bank or split changed')
        entry['node_time_bandwidth'] = previous['bandwidth_json']
        planning, forecast, physical, case, _, _ = context_planning(entry)
        baseline_forecast = paired.arrays(previous['planner']['path'])['forecast']
        if not np.array_equal(forecast, baseline_forecast):
            raise ValueError('Paired forecast differs from latest P/Q-band baseline')
        probe = SingleTargetEvidence(case, planning['score'], planning['val'], forecast,
            planning['base'], mode='normal_fixed', meter_relative_sd=entry['meter']['additional_V_relative_sd'],
            forecast_relative_sd=source['forecast_relative_sd'])
        if int(probe.native_indices[probe.targets[0]]) != entry['planned_history_target_native']:
            raise ValueError('Strict historical target changed')
        io.write(location/'prepared_resource_case.json', physical.config)
        io.write(location/'baseline_device_states.json', physical.device_states(np.zeros(physical.shape)))
        io.write(location/'baseline_decomposition.json', physical.baseline_decomposition(forecast))
        entry['prepared_resource_case'] = io.ref(location/'prepared_resource_case.json')
        entry['baseline_device_states'] = io.ref(location/'baseline_device_states.json')
        entry['baseline_decomposition'] = io.ref(location/'baseline_decomposition.json')
    io.write(out/'inputs.json', entries)
    protocol = dict(created_utc=paired.old.utc(), phase='explicit_equipment_development_replay',
        smoke=bool(smoke), oracle=False, inputs=io.ref(out/'inputs.json'),
        source_run=source['source_run'], source_protocol=source['source_protocol'],
        baseline_run=str(BASELINE), baseline_audit=io.ref(BASELINE_AUDIT),
        baseline_protocol=io.ref(BASELINE/'protocol.json'), baseline_completion=io.ref(BASELINE/'complete.json'),
        baseline_source_snapshots=snapshots, grid_baseline_audit=source['baseline_audit'],
        zero_source='Original sealed grid no_dfs observations and common noise arrays',
        preflight_audit=None if preflight is None else io.ref(preflight),
        seed=source['seed'], forecasts=1, repeats=1 if smoke else 16, workers=1 if smoke else workers,
        solver_seconds=2. if smoke else float(seconds), modes=list(MODES),
        policies=['single_'+m for m in MODES], expected_quotes=9, expected_rows=9*(1 if smoke else 16),
        expected_band_baseline_rows=9*(1 if smoke else 16), expected_zero_baseline_rows=3*(1 if smoke else 16),
        target_rule=source['target_rule'], competitor_rule=source['competitor_rule'], guard_rule=source['guard_rule'],
        objective='Minimize activation fee; requested capacity plus actual activation reported separately',
        reservation_prices=source['reservation_prices'], forecast_relative_sd=source['forecast_relative_sd'],
        pairing='Same forecast, background network-day and noise0..3; old physical P-device eta reused, new device keyed eta; each device P/Q shares eta',
        response_namespace=RESPONSE_NAMESPACE, response_distribution='Uniform(0.9,1), one scalar for a whole physical-device trajectory',
        baseline_scope='Latest node_time_pq band solution; old free-Q and new explicit equipment are different resource domains',
        bandwidth_rule=source['bandwidth_rule'], bandwidth_sources='Exactly the old sealed training-only node/hour arrays',
        zero_rule='Verify zero against the device domain, node/hour band and original probability gate first',
        fallback='No verified point executes zero; quotation failure and probability infeasibility remain explicit',
        seed_rule='Map only old common-pool P-device columns by physical ID; no old final-policy points, no SVG-to-battery mapping; audit every mapped point',
        time_scope='Each quote budget includes shared domain/seed preparation and its gate construction/solver; final rechecks and IO wall time reported; previous common LP and150s anchor are sunk work',
        independence='Known development replay: three noise settings on two network-days, sixteen conditional response/meter paths each',
        equipment_baseline='Declared device schedules are a decomposition of net profiles, never added to the measured net load',
        driver_relative_path=Path(__file__).resolve().relative_to(ROOT).as_posix(), environment=io.environment())
    protocol['sources'] = snapshot(out, entries)
    protocol.update(paired.old._git_receipt([Path(r['path']) for r in protocol['sources']]))
    io.write(out/'protocol.json', protocol)
    return protocol


def execute(entry, protocol, directory, planning, physical, case, forecast, controls, plans, old_config):
    """First future observation/true network read occurs after a checked seal."""
    seal = io.read(directory/'seal.json')
    paired.check_refs([seal[k] for k in ('planner', 'plans', 'score', 'train', 'validation',
        'equipment_spec', 'resource_case', 'shared_initial_points', 'bandwidth_npz', 'baseline_device_states')])
    ctx = paired.old.context(entry)
    if ctx['score'] != planning['score'] or ctx['base'] != planning['base']:
        raise ValueError('Planning and evaluation frozen bank disagree')
    H, C, hist, sigma2 = ctx['H'], ctx['common_H'], ctx['history'], ctx['score']['sigma2']
    predicted = {name: paired.old.pair_components(H, C, hist, sigma2,
        forecast+physical.shift_pq_pu(u, ctx['base'])) for name, u in controls.items()}
    zero_prediction = paired.old.pair_components(H, C, hist, sigma2, forecast)
    prior = Path(entry['prior_directory'])
    band_prior = Path(entry['band_baseline_directory'])
    rows = []
    for repeat in range(protocol['repeats']):
        rd = directory/f'execution{repeat:02d}'
        rd.mkdir()
        source_rd = prior/f'execution{repeat:02d}'
        source_random = io.ref(source_rd/'paired_randomness.npz')
        random = paired.arrays(source_random['path'])
        eta, lineage = response_multipliers(physical, old_config, random['eta'], seed=protocol['seed'],
            original_index=entry['index'], repeat=repeat)
        np.savez_compressed(rd/'paired_randomness.npz', eta=eta, original_eta=random['eta'],
                            **{f'noise{i}': random[f'noise{i}'] for i in range(4)})
        io.write(rd/'randomness_lineage.json', dict(original=source_random, devices=lineage,
            node_and_mode_independent=True, no_new_measurement_noise=True,
            original_eta_is_not_relabelled_Q=True))
        random_ref = io.ref(rd/'paired_randomness.npz')
        zero_folder = source_rd/'no_dfs'
        zero_ob = paired.arrays(zero_folder/'observations.npz')
        zero_receipt = io.read(zero_folder/'receipt.json')
        baseline = paired.arrays(zero_folder/'pair_evidence.npz')
        zero_clean = zero_ob['true_root'][:, None]**2-zero_ob['true_V']**2
        zero_supply = ctx['baseline_state']['root_supply_kw']
        zero_grid_value = paired.old.grid_value(physical, zero_supply)
        for name, request in controls.items():
            folder = rd/name
            folder.mkdir()
            quote = plans[name]
            row = dict(id=entry['id'], case=entry['case'], start_date=entry['start_date'],
                original_id=entry['original_id'], index=entry['index'], forecast_index=0, repeat=repeat,
                strategy=name, parent_strategy=name, mode=quote['mode'], guard_kind='equipment_node_hour_pq',
                is_rotation=False, quote_zero_fallback=quote['zero_fallback'],
                quote_new_constraints_feasible=quote['new_gate_feasible'],
                paired_randomness_source=random_ref, original_randomness_source=source_random,
                randomness_lineage=io.ref(rd/'randomness_lineage.json'),
                baseline_source=io.ref(zero_folder/'receipt.json'),
                band_baseline_source=io.ref(band_prior/f'execution{repeat:02d}'/name/'receipt.json'))
            try:
                delivered, delivery = physical.execute(request, ctx['p']*ctx['base'], eta)
                delivered = physical.array(delivered)
                states = save_states(folder, physical.device_states(request), physical.device_states(delivered))
                requested_domain, delivered_domain = physical.audit(request), physical.audit(delivered)
                if not requested_domain['passed'] or not delivered_domain['passed']:
                    raise ArithmeticError('Requested or delivered action violates the explicit equipment domain')
                pp, qq, state, ac, state_ref = paired.old.physical_state(ctx, physical, delivered, folder/'full_ac.npz')
                v, root = state['terminal_voltage_pu'], ctx['root']
                pm, qm, y, vm, rm = paired.old.meter(pp, qq, v, root,
                    [random[f'noise{i}'] for i in range(4)],
                    pq_sigma=entry['meter']['additional_PQ_relative_sd'],
                    voltage_sigma=entry['meter']['additional_V_relative_sd'])
                if np.count_nonzero(delivered) == 0:
                    for key, value in (('P',pm),('Q',qm),('Y',y),('V',vm),('root',rm)):
                        if not np.array_equal(value, zero_ob[key]):
                            raise ArithmeticError('Paired zero observations differ: '+key)
                    row['reused_action_source'] = io.ref(zero_folder/'observations.npz')
                clean_y = root[:, None]**2-v**2
                observed = Observation(pm, qm, y, ctx['labels'], vm, rm)
                values, ll, new_sse = paired.old.score_observation(ctx, observed)
                components = paired.old.pair_components(H, C, hist, sigma2, np.c_[pp, qq],
                    clean_y, observed, ctx['history_without_common'])
                changes, summary = paired.old.evidence_summary(ctx, baseline, components,
                    zero_prediction, predicted[name], None, None)
                np.savez_compressed(folder/'pair_evidence.npz', **components)
                np.savez_compressed(folder/'pair_changes.npz', **changes)
                np.savez_compressed(folder/'observations.npz', P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=root,requested_action=request,delivered_action=delivered,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,
                    representative_log_probability=ll)
                delta_power = np.c_[pp, qq]-np.c_[zero_ob['true_P'], zero_ob['true_Q']]
                increment_y = clean_y-zero_clean
                K_error = np.mean((increment_y[None]-np.einsum('tc,knc->ktn', delta_power, ctx['means']))**2, axis=(1,2))/sigma2
                repH = np.array([r['H'] for r in ctx['score']['parameters']])
                H_error = np.mean((increment_y[None]-np.einsum('tc,knc->ktn', delta_power, repH))**2, axis=(1,2))/sigma2
                planned_cost = prices.cost_breakdown(physical, request)
                settled_cost = prices.cost_breakdown(physical, request, delivered)
                amounts = physical.amounts(delivered)
                voltage = state['voltage_magnitude_pu']
                row.update(status='complete', **values, delivery=delivery, evidence=summary,
                    physical_state=state_ref, AC=ac, device_states=states,
                    requested_equipment_audit=requested_domain, delivered_equipment_audit=delivered_domain,
                    planned_band=case.band_audit(request), delivered_band=case.band_audit(delivered),
                    planned_cost=planned_cost, settled_cost=settled_cost,
                    planned_fee_yuan=planned_cost['total_yuan'], settled_fee_yuan=settled_cost['total_yuan'],
                    requested_quantities=physical.amounts(request), delivered_quantities=amounts,
                    delivered_abs_kwh=amounts['active_abs_kwh'], delivered_abs_kvarh=amounts['reactive_abs_kvarh'],
                    all_bus_voltage_min=float(voltage.min()), all_bus_voltage_max=float(voltage.max()),
                    voltage_outside_095_105_cells=int(((voltage < .95)|(voltage > 1.05)).sum()),
                    baseline_root_energy_value_yuan=zero_grid_value,
                    root_energy_value_yuan=paired.old.grid_value(physical, state['root_supply_kw']),
                    root_energy_value_change_yuan=paired.old.grid_value(physical, state['root_supply_kw'])-zero_grid_value,
                    actual_increment_response_error_H_over_sigma2_by_representative=H_error.tolist(),
                    actual_increment_response_error_K_over_sigma2_by_representative=K_error.tolist())
            except Exception as exc:
                row.update(status='execution_failed', error=repr(exc), traceback=traceback.format_exc())
            io.write(folder/'receipt.json', row)
            rows.append(row)
        io.write(directory/'results.json', rows)
    return rows


def run_case(entry):
    protocol, out = paired.old.WORK['p'], paired.old.WORK['out']
    directory = out/'cases'/entry['id']
    if directory.exists():
        raise FileExistsError('Preserve old or partial case: '+str(directory))
    directory.mkdir(parents=True)
    began = perf_counter()
    io.write(directory/'started.json', dict(pid=os.getpid(), created_utc=paired.old.utc()))
    shared_tick = perf_counter()
    planning, forecast, physical, case, old_config, original_seeds = context_planning(entry)
    if physical.config != io.read(paired.old.checked(entry['prepared_resource_case'])):
        raise ValueError('Prepared resource configuration changed')
    seeds, seed_receipt = mapped_seeds(case, old_config, original_seeds)
    shared_seconds = perf_counter()-shared_tick
    io.write(directory/'resource_case.json', physical.config)
    shutil.copyfile(paired.old.checked(entry['equipment_spec']), directory/'equipment_spec.json')
    shutil.copyfile(paired.old.checked(entry['baseline_device_states']), directory/'baseline_device_states.json')
    shutil.copyfile(paired.old.checked(entry['baseline_decomposition']), directory/'baseline_decomposition.json')
    band_meta = io.read(paired.old.checked(entry['node_time_bandwidth']))
    shutil.copyfile(paired.old.checked(band_meta['arrays']), directory/'bandwidth.npz')
    io.write(directory/'bandwidth.json', band_meta)
    shutil.copyfile(paired.old.checked(entry['initial_points']), directory/'original_initial_points.npz')
    np.savez_compressed(directory/'shared_initial_points.npz', **seeds)
    seed_receipt.update(original_source=entry['initial_points'], shared_domain_and_mapping_seconds=shared_seconds)
    io.write(directory/'initial_mapping.json', seed_receipt)
    initial_ref = io.ref(directory/'shared_initial_points.npz')
    controls, plans, models = {}, {}, {}
    for mode in MODES:
        name = 'single_'+mode
        tick = perf_counter()
        io.write(directory/(name+'_started.json'), dict(pid=os.getpid(), mode=mode,
            allowed_seconds=protocol['solver_seconds']))
        model = None
        try:
            model = SingleTargetEvidence(case, planning['score'], planning['val'], forecast,
                planning['base'], mode=mode, meter_relative_sd=entry['meter']['additional_V_relative_sd'],
                forecast_relative_sd=protocol['forecast_relative_sd'])
            remaining = max(.001, protocol['solver_seconds']-shared_seconds-(perf_counter()-tick))
            quote, point = engine.plan(case, planning['score'], planning['val'], forecast,
                planning['base'], mode=mode, seconds=remaining, env=paired.old.WORK['env'],
                seed_points=seeds, record_directory=directory/(name+'_diagnostics'),
                meter_relative_sd=entry['meter']['additional_V_relative_sd'],
                forecast_relative_sd=protocol['forecast_relative_sd'], evidence_factory=SingleTargetEvidence)
            if bool(quote['feasible']) != (point is not None):
                raise ValueError('Returned point and quote feasibility disagree')
            if point is not None and not (case.audit(point)['passed'] and model.verify_expansion(point)['feasible']):
                raise ArithmeticError('Final original gate, equipment or band audit failed')
        except Exception as exc:
            quote, point = dict(status='exception', feasible=False, error=repr(exc), traceback=traceback.format_exc()), None
        action = np.zeros(case.shape) if point is None else physical.array(point).copy()
        check = (dict(feasible=False, status='evidence_construction_failed') if model is None
                 else model.verify_expansion(action))
        quote.update(mode=mode, guard_kind='equipment_node_hour_pq', new_gate_feasible=point is not None,
            zero_fallback=point is None, executed_source='zero_fallback' if point is None else 'verified_equipment_node_hour_band',
            executed_probability_check=check, executed_cost=prices.cost_breakdown(physical, action),
            target_native=entry['planned_history_target_native'], planned_band=case.band_audit(action),
            equipment_audit=physical.audit(action), initial_points=initial_ref,
            original_initial_points=entry['initial_points'], initial_mapping=io.ref(directory/'initial_mapping.json'),
            shared_preparation_seconds=shared_seconds, requested_quote_budget_seconds=protocol['solver_seconds'],
            quote_wall_seconds=perf_counter()-tick, total_method_seconds_with_shared_preparation=perf_counter()-tick+shared_seconds,
            prior_shared_work='Original common LP and150s anchor were completed previously, not rerun',
            band_encoding=case.band_encoding,
            raw_bound_scope='Activation objective on new equipment constraints AND encoded original node/hour bands AND unchanged single-target probability gates; not old-domain/AC/total-fee bound')
        controls[name], plans[name], models[mode] = action, quote, model
        np.savez_compressed(directory/(name+'_action.npz'), action=action)
        io.write(directory/(name+'_plan.json'), quote)
        print(entry['id'], name, quote['status'], 'fee', quote.get('cost_yuan'), 'fallback', point is None, flush=True)
    np.savez_compressed(directory/'planner.npz', forecast=forecast, **controls)
    io.write(directory/'plans.json', plans)
    io.write(directory/'prediction_audits.json', {mode:{name:(None if model is None else model.verify_expansion(u))
        for name,u in controls.items()} for mode,model in models.items()})
    io.write(directory/'requested_device_states.json', {name:physical.device_states(u) for name,u in controls.items()})
    io.write(directory/'seal.json', dict(created_utc=paired.old.utc(), planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'), score=entry['score'], train=entry['files']['train.npz'],
        validation=entry['files']['validation.npz'], equipment_spec=io.ref(directory/'equipment_spec.json'),
        baseline_device_states=io.ref(directory/'baseline_device_states.json'),
        baseline_decomposition=io.ref(directory/'baseline_decomposition.json'),
        requested_device_states=io.ref(directory/'requested_device_states.json'),
        resource_case=io.ref(directory/'resource_case.json'), bandwidth_json=io.ref(directory/'bandwidth.json'),
        bandwidth_npz=io.ref(directory/'bandwidth.npz'), sampling_source=entry['files']['observations_sealed.json'],
        shared_initial_points=initial_ref, original_initial_points=io.ref(directory/'original_initial_points.npz'),
        initial_mapping=io.ref(directory/'initial_mapping.json'), protocol=io.ref(out/'protocol.json'),
        prediction_audits=io.ref(directory/'prediction_audits.json'),
        planned_history_target_native=entry['planned_history_target_native'], planner_actual_voltage_or_network_inputs=False))
    rows = execute(entry, protocol, directory, planning, physical, case, forecast, controls, plans, old_config)
    good = [r for r in rows if r['status'] == 'complete']
    result = dict(status='complete', id=entry['id'], rows=len(rows), failed_rows=len(rows)-len(good),
        quote_exceptions=sum(q['status']=='exception' for q in plans.values()),
        feasible_quotes=sum(q['feasible'] for q in plans.values()),
        zero_fallbacks=sum(q['zero_fallback'] for q in plans.values()),
        zero_sufficient_quotes=sum(q['status']=='zero_sufficient' for q in plans.values()),
        reused_ac_rows=sum('reused_action_source' in r for r in good),
        new_ac_rows=sum('reused_action_source' not in r for r in good), seconds=perf_counter()-began,
        shared_preparation_seconds=shared_seconds, results=io.ref(directory/'results.json'), plans=io.ref(directory/'plans.json'))
    io.write(directory/'complete.json', result)
    return result


def run(out):
    out = Path(out).resolve()
    protocol = io.read(out/'protocol.json')
    paired.check_refs([protocol['inputs'], *protocol['sources']])
    if (out/'lock.json').exists() or (out/'complete.json').exists():
        raise RuntimeError('Preserve existing locks and complete runs')
    io.write(out/'lock.json', dict(pid=os.getpid(), created_utc=paired.old.utc()))
    receipts, failures = [], []
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers'], initializer=paired.old.initialize,
                                 initargs=(str(out),)) as pool:
            tasks = {pool.submit(run_case, e): e for e in io.read(protocol['inputs']['path'])}
            for task in as_completed(tasks):
                try:
                    receipts.append(task.result())
                except Exception as exc:
                    failures.append(dict(id=tasks[task]['id'], error=repr(exc), traceback=traceback.format_exc()))
                io.write(out/'progress.json', dict(receipts=receipts, failed_tasks=failures))
        result = dict(status='complete' if not failures else 'incomplete', receipts=receipts, failed_tasks=failures,
            **{key:sum(r[key] for r in receipts) for key in ('rows','failed_rows','quote_exceptions',
                'feasible_quotes','zero_fallbacks','zero_sufficient_quotes','reused_ac_rows','new_ac_rows')},
            source_unchanged=all(io.ref(ref['path']) == ref for ref in protocol['sources']))
        io.write(out/'complete.json', result)
        return result
    finally:
        (out/'lock.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seconds', type=float, default=1200.)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.resume:
        prepare(args.out, smoke=args.smoke, seconds=args.seconds, preflight=args.preflight, workers=args.workers)
    if not args.prepare_only:
        print(run(args.out))


if __name__ == '__main__':
    main()
