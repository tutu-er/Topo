"""Hash-gated four-day physical worker for resource-day module increments.

CLI: --request REQUEST.json --expected-sha SHA256 --output EMPTY_DIRECTORY.
No simulation, identifier, or data read runs on import. REQUEST is a frozen
JSON object with case, source_root, source_metadata={path,sha256,case_name},
labels, root_label, base_kw, phase (planning/test), record_id, path,
replicate_base (>=20000), event_profiles (four indices in 0/1/2),
plan_packet={path,sha256}, protocol={path,sha256}, source_sha256={path:hash},
and rho_by_event=[{resource_id:0/1}, ...] (four entries). response_mode is
independent_resource_day_MC or conditional_gamma_event0. A test request also
needs oracle={path,sha256}.
The protocol and oracle have status='frozen'; the oracle binds
protocol_sha256. Relative references resolve against the request directory.

The trusted, hash-bound plan pickle has resources_by_event, modules_by_event,
amplitudes_by_event: four lists each. Resource/module entries are plain field
dictionaries for ResourceDay/DayModule, not pickled dataclass instances (the
immutable module profiles are not pickleable). A battery entry is None or a
plain Battery field dictionary. The same two resource IDs and their
nodes occur on every day. Each rho entry controls that resource's complete
day, without forcing participation to agree across different days. This is
an explicit synthetic resource-day realization, not an estimated behavior
law. The request hash freezes it before natural future generation.
The worker neither draws participation nor clips/resamples a failed delivery.

source_sha256 must include this worker, day_schedule.py, coded_shapes.py,
physical_bridge.py, end_to_end/physical_case.py, and the seven frozen physical
producer modules listed in PRODUCER_MODULES. Every supplied source hash is
checked before any producer import. The protocol may additionally bind the
same sources in its own source_sha256 field.

Only increments are added to natural terminal totals. Resource baselines are
declared components already contained in those totals; the nonflexible
residual may be negative. This is a synthetic decomposition, not a measured
resource-availability claim. Receipt JSON contains scalar audits only.
packet.pkl is observed-only; physical_truth.pkl and truth_evaluation.json
are evaluator-only private outputs. No identifier is imported or run here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import traceback


PRODUCER_MODULES = (
    'rnj_wzzt.scenario.simulation', 'rnj_wzzt.scenario.profiles',
    'rnj_wzzt.scenario.settings', 'rnj_wzzt.data.paper_style_case_bank',
    'rnj_wzzt.data.paper_style_terminal_lv', 'rnj_wzzt.models.network',
    'rnj_wzzt.models.ac_powerflow',
)
LOCAL_SOURCES = (
    'research_experiments/wzzt_dr_selection/dr_day_modules/day_physical_worker.py',
    'research_experiments/wzzt_dr_selection/dr_day_modules/day_schedule.py',
    'research_experiments/wzzt_dr_selection/dr_competition_bridge/coded_shapes.py',
    'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py',
    'research_experiments/wzzt_dr_selection/end_to_end/physical_case.py',
)
RESOURCE_SCALARS = (
    'rho', 'reduced_kwh', 'rebound_kwh', 'net_kwh', 'incentive_payment',
    'incremental_electricity_bill', 'maximum_violation', 'full_plan_feasible',
    'actual_resource_feasible', 'feasible', 'full_delivery_incentive_payment',
    'baseline_feasible', 'strict_provider_band_pass', 'declaration_source',
    'planned_service_deadline_slot', 'service_deadline_source',
    'service_target_kwh', 'planned_deadline_energy_kwh',
    'actual_deadline_energy_kwh', 'planned_deadline_pass',
    'actual_deadline_pass', 'terminal_energy_kwh',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_bytes(pickle.dumps(value, protocol=4))
    temporary.replace(path)


def _resolve(path, parent):
    path = Path(path)
    return (path if path.is_absolute() else parent/path).resolve()


def _reference(item, parent, name):
    if not isinstance(item, dict) or not {'path', 'sha256'} <= set(item):
        raise ValueError(name+' needs path and sha256')
    path = _resolve(item['path'], parent)
    if sha(path) != item['sha256']:
        raise RuntimeError(name+' hash mismatch')
    return path


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return value


def voltage_noise_binding(request, protocol):
    """Keep original defaults, but inherit a new experiment's locked noise level."""
    value = request.get('voltage_sigma', .0002)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('voltage_sigma must be an explicit numeric source-grid value')
    value = float(value)
    if not math.isfinite(value) or value not in (0., .0002, .0005, .001):
        raise ValueError('voltage_sigma outside the existing source grid')
    if 'voltage_sigma' in protocol and protocol['voltage_sigma'] != value:
        raise ValueError('future voltage/root noise differs from the locked source protocol')
    return value


def load_request(request_path, expected_sha):
    """Check all declared gates before unpickling plans or loading a producer."""
    request_path = Path(request_path).resolve()
    if not expected_sha or sha(request_path) != expected_sha:
        raise RuntimeError('request does not match the frozen expected hash')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if request.get('status') != 'frozen':
        raise ValueError('request must be frozen')
    parent = request_path.parent
    if request.get('phase') not in ('planning', 'test'):
        raise ValueError('phase must be planning or test')
    for key in ('case', 'record_id'):
        if not isinstance(request.get(key), str) or not request[key]:
            raise ValueError(key+' must be a nonempty string')
    request['path'] = _integer(request.get('path'), 'path')
    request['replicate_base'] = _integer(request.get('replicate_base'), 'replicate_base', minimum=20000)
    profiles = request.get('event_profiles')
    if not isinstance(profiles, list) or len(profiles) != 4 or any(
            isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1, 2) for v in profiles):
        raise ValueError('four source-profile indices in {0,1,2} are required')
    labels = request.get('labels')
    if (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in labels)):
        raise ValueError('unique integer terminal labels are required')
    root_label = _integer(request.get('root_label'), 'root_label')
    if root_label in labels:
        raise ValueError('root cannot be a terminal')
    base_kw = float(request.get('base_kw', float('nan')))
    if not math.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('positive finite base_kw required')
    request['base_kw'] = base_kw
    budget = float(request.get('incentive_budget', 1.6))
    if not math.isfinite(budget) or not 0 <= budget <= 1.6:
        raise ValueError('one four-event incentive budget in [0,1.6] is required')
    request['incentive_budget'] = budget
    if request.get('response_mode') not in ('independent_resource_day_MC', 'conditional_gamma_event0', 'exact_active_train_validation_states'):
        raise ValueError('explicit resource-day response_mode required')
    rho_by_event = request.get('rho_by_event')
    if not isinstance(rho_by_event, list) or len(rho_by_event) != 4:
        raise ValueError('four prelocked resource-day participation mappings required')
    for rho in rho_by_event:
        if (not isinstance(rho, dict) or len(rho) != 2
                or any(not isinstance(k, str) or not k for k in rho)
                or any(v not in (0, 1) or not isinstance(v, (int, float, bool)) for v in rho.values())):
            raise ValueError('each rho entry needs binary participation for exactly two resource IDs')
    if any(set(rho) != set(rho_by_event[0]) for rho in rho_by_event):
        raise ValueError('the same two resource IDs must be declared on all four days')
    request['rho_by_event'] = [{key: int(value) for key, value in rho.items()} for rho in rho_by_event]
    protocol_path = _reference(request.get('protocol'), parent, 'protocol')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'frozen':
        raise ValueError('protocol must be frozen')
    request['voltage_sigma'] = voltage_noise_binding(request, protocol)
    if request['phase'] == 'test':
        oracle_path = _reference(request.get('oracle'), parent, 'test oracle')
        oracle = json.loads(oracle_path.read_text(encoding='utf-8'))
        if oracle.get('status') != 'frozen' or oracle.get('protocol_sha256') != sha(protocol_path):
            raise ValueError('test oracle is not frozen under this protocol')
    plan_path = _reference(request.get('plan_packet'), parent, 'plan packet')
    source_root = _resolve(request['source_root'], parent)
    repo_root = _resolve(request.get('repo_root', Path(__file__).resolve().parents[3]), parent)
    expected = request.get('source_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('explicit source_sha256 mapping required')
    bound = {}
    for filename, digest in expected.items():
        path = _resolve(filename, parent)
        if str(path) in bound and bound[str(path)] != digest:
            raise ValueError('conflicting source hash aliases')
        if sha(path) != digest:
            raise RuntimeError('source hash mismatch: '+str(path))
        bound[str(path)] = digest
    required = [repo_root/path for path in LOCAL_SOURCES]
    required += [source_root.joinpath(*name.split('.')).with_suffix('.py') for name in PRODUCER_MODULES]
    if any(str(path.resolve()) not in bound for path in required):
        raise ValueError('source_sha256 omits a worker/resource/physical producer dependency')
    if sha(__file__) != bound[str((repo_root/LOCAL_SOURCES[0]).resolve())]:
        raise RuntimeError('executed worker differs from the declared repository worker')
    if 'source_sha256' in protocol:
        protocol_sources = {str(_resolve(p, protocol_path.parent)): h for p, h in protocol['source_sha256'].items()}
        if any(protocol_sources.get(p) != h for p, h in bound.items()):
            raise ValueError('request sources differ from protocol-bound sources')
    metadata = dict(request.get('source_metadata', {}))
    metadata['path'] = str(_reference(metadata, parent, 'source metadata'))
    request.update(repo_root=str(repo_root), source_root=str(source_root), source_metadata=metadata,
                   checked_source_sha256=bound, plan_path=str(plan_path),
                   protocol_path=str(protocol_path), request_path=str(request_path))
    return request


def node_increment_arrays(schedule, labels):
    """Convert resource-only increments to aligned finite 96-by-n arrays."""
    import numpy as np
    positions = {int(label): index for index, label in enumerate(labels)}
    p, q = np.zeros((96, len(labels))), np.zeros((96, len(labels)))
    if schedule.get('increment_only_payload') is not True:
        raise ValueError('resource evaluator must return increments only')
    for label, value in schedule['node_increments'].items():
        if int(label) not in positions:
            raise ValueError('resource node is not a measured terminal')
        dp, dq = np.asarray(value['delta_kw'], float), np.asarray(value['delta_kvar'], float)
        if dp.shape != (96,) or dq.shape != (96,) or not np.isfinite(dp).all() or not np.isfinite(dq).all():
            raise ValueError('each node increment needs 96 finite P/Q values')
        p[:, positions[int(label)]] += dp
        q[:, positions[int(label)]] += dq
    return p, q


def scalar_resource_report(schedule):
    """Only declared scalar fields enter the public-safe receipt."""
    records = {}
    for key, row in schedule['resources'].items():
        record = {name: row[name] for name in RESOURCE_SCALARS if name in row}
        if any(isinstance(value, (list, tuple, dict)) for value in record.values()):
            raise ValueError('resource scalar report unexpectedly contains an array')
        records[key] = record
    return records


def _load_legacy(repo_root):
    path = repo_root/LOCAL_SOURCES[3]
    spec = importlib.util.spec_from_file_location('_day_original_physical_bridge', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('physical bridge unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plan(request):
    """Unpickle only the trusted local hash-bound plan, then audit every event."""
    import numpy as np
    sys.path.insert(0, request['repo_root'])
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
        Battery, DayModule, ResourceDay, evaluate_schedule,
    )
    with Path(request['plan_path']).open('rb') as stream:
        packet = pickle.load(stream)
    keys = ('resources_by_event', 'modules_by_event', 'amplitudes_by_event')
    if not isinstance(packet, dict) or any(not isinstance(packet.get(k), (list, tuple)) or len(packet[k]) != 4 for k in keys):
        raise ValueError('plan must contain four resource/module/amplitude lists')
    resources_by_event, modules_by_event = [], []
    for event in range(4):
        resources, modules = [], []
        for source in packet['resources_by_event'][event]:
            if not isinstance(source, dict):
                raise ValueError('resource plan entries must be plain constructor dictionaries')
            source = dict(source)
            if source.get('battery') is not None:
                if not isinstance(source['battery'], dict):
                    raise ValueError('battery plan entry must be a plain constructor dictionary')
                source['battery'] = Battery(**source['battery'])
            resources.append(ResourceDay(**source))
        for source in packet['modules_by_event'][event]:
            if not isinstance(source, dict):
                raise ValueError('module plan entries must be plain constructor dictionaries')
            modules.append(DayModule(**source))
        resources_by_event.append(resources)
        modules_by_event.append(modules)
    packet = dict(packet, resources_by_event=resources_by_event, modules_by_event=modules_by_event)
    audits, mapping = [], None
    for event in range(4):
        resources = packet['resources_by_event'][event]
        pairs = {r.resource_id: int(r.node_label) for r in resources}
        if len(resources) != 2 or len(pairs) != 2 or set(pairs) != set(request['rho_by_event'][event]):
            raise ValueError('exactly the same two declared resources are required every day')
        if mapping is not None and pairs != mapping:
            raise ValueError('resource node changes across days')
        mapping = pairs
        audit = evaluate_schedule(resources, packet['modules_by_event'][event],
                                  packet['amplitudes_by_event'][event], rho=request['rho_by_event'][event])
        node_increment_arrays(audit, request['labels'])
        # Independent billing replay on each resource's summed executed offset.
        actual_payment = full_payment = bill = 0.
        for resource in resources:
            row = audit['resources'][resource.resource_id]
            actual, planned = np.asarray(row['executed_delta_kw']), np.asarray(row['planned_delta_kw'])
            actual_payment += .25*float(np.dot(resource.incentive_per_kwh, np.maximum(-actual, 0.)))
            full_payment += .25*float(np.dot(resource.incentive_per_kwh, np.maximum(-planned, 0.)))
            bill += .25*float(np.dot(resource.electricity_price_per_kwh, actual))
            if max(abs(.25*actual.sum()), abs(.25*planned.sum())) > 1e-7:
                raise ValueError('resource schedule does not return daily incremental energy')
        for value, expected in ((actual_payment, audit['total_incentive_payment']),
                                (full_payment, audit['full_delivery_incentive_payment']),
                                (bill, audit['total_incremental_electricity_bill'])):
            if not math.isfinite(value) or abs(value-expected) > 1e-9:
                raise AssertionError('independent resource billing mismatch')
        if actual_payment < -1e-12 or full_payment < -1e-12 or actual_payment > full_payment+1e-9:
            raise ValueError('incentives must be nonnegative and binary delivery must not exceed full delivery')
        audit['independent_billing_pass'] = True
        audits.append(audit)
    total_full = sum(a['full_delivery_incentive_payment'] for a in audits)
    total_committed = sum(a['total_incentive_payment'] for a in audits)
    if total_full > request['incentive_budget']+1e-9 or total_committed > request['incentive_budget']+1e-9:
        raise ValueError('whole four-event incentive commitment exceeds the frozen budget')
    return packet, audits


def run(request, output, receipt):
    import numpy as np
    import pandas as pd
    packet, audits = _load_plan(request)
    legacy = _load_legacy(Path(request['repo_root']))
    metadata = legacy.frozen_source_metadata(request, request['case'])
    if metadata['root_label'] != request['root_label'] or abs(metadata['base_kw']-request['base_kw']) > 1e-12:
        raise ValueError('request root/base disagrees with source metadata')
    receipt.update(protocol_sha256=request['protocol']['sha256'], plan_packet_sha256=request['plan_packet']['sha256'],
                   source_sha256=request['checked_source_sha256'], case=request['case'], phase=request['phase'], voltage_sigma=request['voltage_sigma'],
                   record_id=request['record_id'], path=request['path'],
                   base_kw=metadata['base_kw'], base_kv=metadata['base_kv'], base_mva=metadata['base_mva'],
                   root_label=metadata['root_label'], terminal_labels=request['labels'],
                   incentive_budget=request['incentive_budget'],
                   worst_payment=sum(a['full_delivery_incentive_payment'] for a in audits),
                   committed_payment=sum(a['total_incentive_payment'] for a in audits),
                   participation_model='prelocked synthetic binary resource-day realization, not calibrated',
                   participation_scope='one rho per resource per complete day; no forced agreement across days',
                   response_mode=request['response_mode'],
                   resource_baseline_semantics='declared component already contained in natural terminal total',
                   nonflexible_residual_may_be_negative=True, old_test_or_truth_files_read=False)
    write_json(output/'events_receipt.json', receipt)
    # The physical producer is first imported after request, protocol, plan,
    # metadata, source hashes, resource schedules and total-budget checks.
    sim, runtime = legacy._simulation_runtime(Path(request['source_root']))
    for name in PRODUCER_MODULES:
        source = Path(request['source_root']).joinpath(*name.split('.')).with_suffix('.py').resolve()
        if runtime[name] != request['checked_source_sha256'][str(source)]:
            raise RuntimeError('loaded producer differs from the prelocked source hash: '+name)
    checks_function, checks_source = legacy._physical_check_function(Path(request['repo_root']))
    if sha(checks_source) != request['checked_source_sha256'][str(checks_source.resolve())]:
        raise RuntimeError('loaded physical checks differ from the prelocked source hash')
    runtime[str(checks_source)] = sha(checks_source)
    context = sim._prepare_case_context(request['case'], 96, scenario_suite='reference',
        root_observation='noisy', root_meter_noise_rel=request['voltage_sigma'], root_sigma=.003, impedance_scale=1.)
    legacy.validate_context_metadata(context, metadata, request['labels'])
    context.net.check_terminal_load_only(strict=True)
    training, validation, truth = [], [], []
    receipt['runtime_sha256'] = runtime
    for event, audit in enumerate(audits):
        record = receipt['event_records'][event]
        private_event = dict(event=event, status='started', resource_audit=audit)
        truth.append(private_event)
        record.update(status='started', role='training' if event < 2 else 'validation',
                      replicate=request['replicate_base']+10*request['path']+event,
                      profile_index=request['event_profiles'][event],
                      resource_reports=scalar_resource_report(audit),
                      resource_feasible=bool(audit['resource_feasible']),
                      maximum_resource_violation=float(audit['maximum_violation']),
                      committed_payment=float(audit['total_incentive_payment']),
                      full_delivery_incentive_payment=float(audit['full_delivery_incentive_payment']),
                      actual_payment=None, incremental_electricity_bill=None,
                      independent_resource_billing_pass=True)
        try:
            if not audit['resource_feasible']:
                raise ValueError('declared baseline/full plan/actual resource constraints fail')
            base = sim._build_base_trajectory(context, record['profile_index'], record['replicate'])
            natural_p, natural_q = base.p_base.to_numpy(), base.q_base.to_numpy()
            dp, dq = node_increment_arrays(audit, request['labels'])
            if natural_p.shape != dp.shape or natural_q.shape != dq.shape:
                raise ValueError('natural P/Q dimensions differ from the frozen module terminal order')
            physical_p = natural_p+dp/metadata['base_kw']
            physical_q = natural_q+dq/metadata['base_kw']
            private_event.update(P_pu=physical_p, Q_pu=physical_q,
                root_pu=base.root_true.to_numpy(), natural_P_pu=natural_p, natural_Q_pu=natural_q,
                delta_kw=dp, delta_kvar=dq)
            max_p_error = float(np.max(np.abs((physical_p-natural_p)*metadata['base_kw']-dp), initial=0.))
            max_q_error = float(np.max(np.abs((physical_q-natural_q)*metadata['base_kw']-dq), initial=0.))
            if not np.isfinite(physical_p).all() or not np.isfinite(physical_q).all() or max(max_p_error, max_q_error) > 1e-9:
                raise AssertionError('increment-only physical metering identity failed')
            record.update(increment_only_bridge_pass=True, max_p_increment_error_kw=max_p_error,
                          max_q_increment_error_kvar=max_q_error, profile_seed=int(base.shifted_seed))
            pframe = pd.DataFrame(physical_p, index=base.p_base.index, columns=request['labels'])
            qframe = pd.DataFrame(physical_q, index=base.q_base.index, columns=request['labels'])
            ac, voltage = sim._solve_physical_ac(context, pframe, qframe, base.root_true)
            private_event.update(V_pu=voltage.to_numpy(), ac=ac)
            equations = checks_function(context.net, pframe, qframe, ac, (.9, 1.1), 1e-8)
            measured = sim._observe_terminal_channels(context, base, pframe, qframe, voltage,
                                                       pq_noise_rel=.005, v_noise_rel=request['voltage_sigma'])
            observed = legacy.scene(measured['p_measured'].to_numpy(), measured['q_measured'].to_numpy(),
                measured['voltage_measured'].to_numpy(), measured['root_observed'].to_numpy(),
                request['labels'], 'day_'+request['record_id']+'_e'+str(event), 6+event)
            observed['actual_event_id'] = event
            (training if event < 2 else validation).append(observed)
            converged = int(ac['converged'].sum())
            checks = dict(electrical_equations_pass=bool(equations['electrical_equations_pass']),
                voltage_limits_pass=bool(equations['voltage_limits_pass']),
                voltage_violation_cells=int(equations['voltage_violation_cells']),
                ac_converged_samples=converged, ac_all_96_converged=converged == 96,
                voltage_min_pu=float(voltage.to_numpy().min()), voltage_max_pu=float(voltage.to_numpy().max()),
                max_nodal_p_residual_pu=float(equations['max_nodal_p_residual_pu']),
                max_nodal_q_residual_pu=float(equations['max_nodal_q_residual_pu']),
                max_distflow_identity_error_pu2=float(equations['max_distflow_identity_error_pu2']),
                branch_ampacity_checked=False)
            passed = checks['electrical_equations_pass'] and checks['voltage_limits_pass'] and checks['ac_all_96_converged']
            record.update(checks=checks, status='complete' if passed else 'physical_checks_failed',
                          delivery_status='complete' if passed else 'unresolved')
            if passed:
                record.update(actual_payment=float(audit['total_incentive_payment']),
                              incremental_electricity_bill=float(audit['total_incremental_electricity_bill']))
        except Exception as error:
            record.update(status='physical_failed', delivery_status='unresolved',
                          error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        private_event['status'] = record['status']
        if 'error_type' in record:
            private_event.update(error_type=record['error_type'], message=record['message'])
        write_pickle(output/'physical_truth.pkl', dict(events=truth, partial=len(truth) != 4))
        write_json(output/'events_receipt.json', receipt)
    known = [r['actual_payment'] for r in receipt['event_records'] if r.get('actual_payment') is not None]
    unknown = 4-len(known)
    actual_payment = None if unknown else float(sum(known))
    complete = all(r['status'] == 'complete' for r in receipt['event_records'])
    receipt.update(status='complete' if complete else 'physical_failed', completed_utc=now(),
        actual_payment=actual_payment, known_actual_payment=float(sum(known)), unknown_payment_events=unknown,
        incremental_electricity_bill=None if unknown else float(sum(r['incremental_electricity_bill'] for r in receipt['event_records'])),
        physical_events_complete=sum(r['status'] == 'complete' for r in receipt['event_records']),
        observed_training_days=len(training), observed_validation_days=len(validation),
        training_event_ids=[s['actual_event_id'] for s in training], validation_event_ids=[s['actual_event_id'] for s in validation],
        budget_pass=receipt['worst_payment'] <= request['incentive_budget']+1e-9 and receipt['committed_payment'] <= request['incentive_budget']+1e-9,
        all_operation_checks_pass=complete, quote_qualification_pass=all(a['resource_feasible'] for a in audits))
    observed_packet = dict(training=training, validation=validation, metadata=dict(
        terminal_labels=request['labels'], base_kw=metadata['base_kw'], base_kv=metadata['base_kv'],
        base_mva=metadata['base_mva'], root_label=metadata['root_label'], case=request['case'],
        interval_hours=.25, synthetic_dates_are_order_labels=True, observed_only=True,
        request_sha256=receipt['request_sha256_expected'], protocol_sha256=request['protocol']['sha256'],
        phase=request['phase'], record_id=request['record_id'], path=request['path'], status=receipt['status'],
        training_event_ids=receipt['training_event_ids'], validation_event_ids=receipt['validation_event_ids']))
    write_pickle(output/'packet.pkl', observed_packet)
    evaluation = legacy.evaluate_supports([], context.net)
    write_json(output/'truth_evaluation.json', dict(true_supports=evaluation['missed_blocks'],
        terminal_labels=request['labels'], full_effective_truth_blocks=evaluation['full_effective_truth_blocks'],
        zero_impedance_merged=True, arbitrarily_weak_positive_lines_retained=True,
        use='Evaluator only after identifier outputs are frozen; never planner/fit input',
        request_sha256=receipt['request_sha256_expected']))
    write_pickle(output/'physical_truth.pkl', dict(events=truth))
    for name in ('packet.pkl', 'truth_evaluation.json', 'physical_truth.pkl'):
        file = output/name
        receipt['private_files'][name] = dict(sha256=sha(file), bytes=file.stat().st_size)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        print(json.dumps(dict(status='existing_output_retained', output=str(output))), flush=True)
        return 3
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='started', started_utc=now(), request_sha256_expected=args.expected_sha,
        physical_events_expected=4, physical_event_rows_expected=96, private_files={},
        event_records=[dict(event=e, status='not_attempted', actual_payment=None) for e in range(4)],
        actual_payment=None, known_actual_payment=0., unknown_payment_events=4,
        no_clipping=True, no_resampling=True, automatic_retry=False,
        voltage_sigma=.0002, pq_sigma=.005, public_receipt_contains_trajectories=False,
        identifier_run=False, old_test_or_truth_files_read=False)
    write_json(output/'events_receipt.json', receipt)
    try:
        request = load_request(args.request, args.expected_sha)
        receipt.update(source_sha256=request['checked_source_sha256'],
            protocol_sha256=request['protocol']['sha256'], plan_packet_sha256=request['plan_packet']['sha256'],
            case=request['case'], phase=request['phase'], voltage_sigma=request['voltage_sigma'], record_id=request['record_id'], path=request['path'])
        write_json(output/'events_receipt.json', receipt)
        receipt = run(request, output, receipt)
    except Exception as error:
        receipt.update(status='worker_failed', completed_utc=now(), error_type=type(error).__name__,
                       message=str(error), traceback=traceback.format_exc(), failure_retained=True)
        payments = [r['actual_payment'] for r in receipt['event_records'] if r.get('actual_payment') is not None]
        receipt.update(known_actual_payment=sum(payments), unknown_payment_events=4-len(payments),
                       actual_payment=sum(payments) if len(payments) == 4 else None)
    write_json(output/'events_receipt.json', receipt)
    print(json.dumps({key: receipt.get(key) for key in ('status', 'physical_events_complete',
        'actual_payment', 'known_actual_payment', 'unknown_payment_events', 'worst_payment',
        'budget_pass', 'all_operation_checks_pass')}, allow_nan=False), flush=True)
    return 0 if receipt['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
