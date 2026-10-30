"""Observed-history and post-price-freeze physical DR bridge.

The planner calls load_history/forecast_mean/apply_response only. The physical
generator is lazy and guarded by a caller-supplied hash of a frozen plan file.
No old test NPZ or old truth NPZ is ever loaded. New actual profiles and AC
voltages are generated only inside generate_actual_events after that guard.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROBABILITIES = np.array([0., .5, .75])
CONTRACT = dict(interval_hours=.25, slots_per_day=96, reduction_kw=.2,
                reduction_kvar=.07, reduction_slots=[76, 77, 78, 79],
                rebound_slots=[80, 81, 82, 83], capacity_kw=45.,
                four_event_worst_budget=1.6)
DEFAULT_SOURCE = Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def scene(p, q, v, root, labels, name, day_offset):
    """Match the current observed-only adapter without importing physical code.

    Dates are synthetic ordering labels, not a claim about actual calendar data.
    Days 0..2 are old training; 3..5 old validation; 6..9 new actual events.
    """
    p, q, v, root = (np.asarray(x, dtype=float) for x in (p, q, v, root))
    if p.shape != (96, len(labels)) or q.shape != p.shape or v.shape != p.shape or root.shape != (96,):
        raise ValueError('one full day of 96 aligned P/Q/V/root samples required')
    if not all(np.isfinite(x).all() for x in (p, q, v, root)):
        raise ValueError('nonfinite observation')
    if (v <= 0).any() or (root <= 0).any():
        raise ValueError('nonpositive measured voltage')
    index = pd.date_range(pd.Timestamp('2000-01-01')+pd.Timedelta(days=int(day_offset)),
                          periods=96, freq='15min')
    frames = [pd.DataFrame(a, index=index, columns=labels) for a in (p, q, v)]
    rv = pd.Series(root, index=index, name='V_root_measured')
    target = pd.DataFrame(root[:, None]**2-v**2, index=index, columns=labels)
    return dict(name=str(name), P_terminal=frames[0], Q_terminal=frames[1],
                V_terminal=frames[2], V_root_observed=rv, drop_target=target,
                units=dict(P_terminal='pu', Q_terminal='pu', V_terminal='pu',
                           V_root_observed='pu', drop_target='pu2'))


def _source_entry(row, role):
    if role not in ('train', 'validation'):
        raise ValueError('only historical train and validation may be loaded')
    item = row.get(role)
    if isinstance(item, dict):
        return Path(item['path']), item['catalog_sha256']
    return Path(row[role+'_path']), row[role+'_sha256']


def load_history(index_row, *, expected_case='paper15', base_kw=200.):
    """Read complete old train+validation NPZ after checking catalog hashes.

    index_row accepts either the raw datasets.json row or the public
    scenario_review_v1/source_inventory.json row. N must be exactly 288;
    no sample subsampling or outcome-dependent record filtering occurs here.
    """
    if expected_case != 'paper15' or base_kw != 200.:
        raise ValueError('this first bridge is only validated for paper15 at 200kW base')
    if index_row['N'] != 288 or index_row['case'] != expected_case:
        raise ValueError('outside the declared complete-history case')
    if index_row.get('condition') != 'reference':
        raise ValueError('reference source only; no silent stress substitution')
    result = dict(training=[], validation=[], labels=None, base_kw=float(base_kw))
    receipt = dict(job_id=index_row['job_id'], source_hashes={}, rows={},
                   NPZ_test_loaded=False, truth_loaded=False,
                   units='P/Q pu, voltage pu, squared drop pu2',
                   dates='synthetic ordering labels; source splits are independent replicates')
    for role, destination, offset in (('train', 'training', 0), ('validation', 'validation', 3)):
        path, expected = _source_entry(index_row, role)
        if path.name != role+'_observed.npz':
            raise ValueError('historical role/path mismatch; never substitute a test file')
        if sha(path) != expected:
            raise RuntimeError('old observed hash mismatch: '+str(path))
        receipt['source_hashes'][str(path)] = expected
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'labels', 'P', 'Q', 'V', 'root'}:
                raise ValueError('unexpected observed NPZ fields')
            labels = archive['labels'].astype(int).tolist()
            arrays = {k:np.asarray(archive[k], dtype=float) for k in ('P','Q','V','root')}
        if len(labels) != len(set(labels)) or (result['labels'] is not None and labels != result['labels']):
            raise ValueError('terminal order changed or duplicated')
        result['labels'] = labels
        if any(arrays[k].shape != (3,96,len(labels)) for k in ('P','Q','V')) or arrays['root'].shape != (3,96):
            raise ValueError('old split is not exactly three complete 96-point days')
        for day in range(3):
            result[destination].append(scene(*(arrays[k][day] for k in ('P','Q','V','root')),
                labels, f"{index_row['job_id']}_{role}_{day}", offset+day))
        receipt['rows'][destination] = 288
    receipt['terminal_count'] = len(result['labels'])
    receipt['all_available_history_used'] = True
    result['receipt'] = receipt
    return result


def forecast_mean(history):
    """Same-slot mean of all six available past P/Q days; no V prediction."""
    days = history['training']+history['validation']
    if len(days) != 6:
        raise ValueError('all six historical physical days required')
    return dict(P_pu=np.stack([d['P_terminal'].to_numpy() for d in days]).mean(axis=0),
                Q_pu=np.stack([d['Q_terminal'].to_numpy() for d in days]).mean(axis=0),
                labels=list(history['labels']), base_kw=float(history['base_kw']),
                slots=96, dt_hours=.25, historical_days=6,
                source='observed same-slot mean, synthetic reference archive',
                no_future_values=True, no_true_topology=True)


def validate_plan(plan, n=None):
    a = np.asarray(plan)
    if a.ndim != 2 or a.shape[0] != 2 or (n is not None and a.shape[1] != n):
        raise ValueError('two training price rows with exact terminal order required')
    if not np.isfinite(a).all() or not np.isin(a, [0,1,2]).all():
        raise ValueError('only price 0/1/2 allowed')
    a = a.astype(int)
    # Training rows 0/1 are each repeated in one independent validation event.
    worst_payment = 2*CONTRACT['reduction_kw']*float(a.sum())
    if worst_payment > CONTRACT['four_event_worst_budget']+1e-12:
        raise ValueError('four-event worst payment exceeds 1.6')
    return a


def draw_participation(plan, event, path, dependence='local', *, seed=90310000):
    """One participant per node/event executes all four reduction/rebound slots.

    CRN: event/path streams do not depend on policy or prices. For local law,
    node U values are independent; for common law one U is shared per event.
    The same probability map applies to both laws and validation events.
    """
    a = validate_plan(plan)
    if event not in range(4) or isinstance(path, bool) or int(path) != path or path < 0:
        raise ValueError('invalid event/path')
    if dependence not in ('local', 'common'):
        raise ValueError('unknown participation dependence')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(path), int(event)]))
    u = rng.random(a.shape[1]) if dependence == 'local' else np.repeat(rng.random(), a.shape[1])
    return u < PROBABILITIES[a[event % 2]]


def apply_response(p_pu, q_pu, price_row, participants, *, base_kw=200.):
    """Apply the fixed .2kW/.07kvar whole-event contract without clipping.

    Natural Q stays unchanged apart from the contracted increment. Net exports
    are allowed because DR reduces an underlying flexible demand component.
    The existence of that .2kW component is a synthetic contract assumption.
    """
    p, q = np.asarray(p_pu, float).copy(), np.asarray(q_pu, float).copy()
    prices, z = np.asarray(price_row), np.asarray(participants, bool)
    if p.ndim != 2 or p.shape[0] != 96 or q.shape != p.shape:
        raise ValueError('96 aligned quarter-hour P/Q rows required')
    if prices.shape != (p.shape[1],) or z.shape != prices.shape or not np.isin(prices,[0,1,2]).all():
        raise ValueError('price/participant columns mismatch')
    if np.any(z & (prices == 0)) or base_kw <= 0 or not np.isfinite(base_kw):
        raise ValueError('zero-price participation or invalid power base')
    reduce, rebound = CONTRACT['reduction_slots'], CONTRACT['rebound_slots']
    delta_p, delta_q = CONTRACT['reduction_kw']/base_kw, CONTRACT['reduction_kvar']/base_kw
    p[reduce] -= delta_p*z; p[rebound] += delta_p*z
    q[reduce] -= delta_q*z; q[rebound] += delta_q*z
    # Four slots x.25h = one hour. Payment is on reduction energy, not net energy.
    payment = CONTRACT['interval_hours']*len(reduce)*CONTRACT['reduction_kw']*float(prices@z)
    energy = (p-np.asarray(p_pu)).sum(axis=0)*base_kw*CONTRACT['interval_hours']
    checks = dict(actual_payment=payment, participants=int(z.sum()),
                  net_energy_change_kWh=energy.tolist(),
                  energy_return_pass=bool(np.max(abs(energy), initial=0.) <= 1e-10),
                  max_absolute_net_power_kw=float(np.max(abs(p))*base_kw),
                  capacity_pass=bool((abs(p)*base_kw <= CONTRACT['capacity_kw']+1e-12).all()),
                  nonfinite_values=bool(not np.isfinite(p).all() or not np.isfinite(q).all()),
                  interval_hours=.25, reduction_duration_hours=1.,
                  physical_load_not_clipped=True, flexible_component_empirically_calibrated=False)
    return p, q, checks


def _frozen_plan(lock_path, expected_sha256, policy):
    path = Path(lock_path)
    if not expected_sha256 or sha(path) != expected_sha256:
        raise RuntimeError('price file must match the predeclared frozen hash')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'frozen' or policy not in record.get('plans', {}):
        raise RuntimeError('requested policy missing from frozen plans')
    return validate_plan(record['plans'][policy]), record


def _simulation_runtime(source_root):
    """Use original physical producer code in a clean worker or exact same files."""
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    sim = importlib.import_module('rnj_wzzt.scenario.simulation')
    important = ('rnj_wzzt.scenario.simulation', 'rnj_wzzt.scenario.profiles',
                 'rnj_wzzt.scenario.settings', 'rnj_wzzt.data.paper_style_case_bank',
                 'rnj_wzzt.data.paper_style_terminal_lv', 'rnj_wzzt.models.network',
                 'rnj_wzzt.models.ac_powerflow')
    hashes = {}
    for name in important:
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        expected = source_root.joinpath(*name.split('.')).with_suffix('.py')
        if not expected.is_file() or sha(actual) != sha(expected):
            raise RuntimeError('physical producer differs; use fresh worker with frozen source first: '+name)
        hashes[name] = sha(actual)
    return sim, hashes


def _physical_check_function(repo_root):
    """Load the existing check by exact file, independent of package precedence.

    The frozen source can contain a regular research_experiments package that
    lacks this newer DR module. Importing its fully qualified name could then
    shadow the current repository. The explicit path avoids that ambiguity.
    """
    path = Path(repo_root).resolve()/'research_experiments/wzzt_dr_selection/end_to_end/physical_case.py'
    spec = importlib.util.spec_from_file_location('_dr_existing_physical_checks',path)
    if spec is None or spec.loader is None:
        raise RuntimeError('existing physical check module is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._physical_checks, path


def generate_actual_events(lock_path, expected_lock_sha256, policy, *, path,
                           dependence='local', voltage_sigma=.0002,
                           case='paper15', source_root=DEFAULT_SOURCE,
                           replicate_base=10000, event_profiles=(0,2,0,2),
                           participation_seed=90310000, repo_root=None):
    """Evaluator-only future: gate price hash, generate new natural days, then AC.

    Returns separate observed training/validation days and evaluator-only truth.
    This function does not write or publish any trajectories. Caller must keep
    truth private and never pass it to planner/identifier. No exception is
    interpreted as infeasibility or silently retried; one record per event stays.
    """
    plan, lock = _frozen_plan(lock_path, expected_lock_sha256, policy)
    if voltage_sigma not in (0.,.0002,.0005,.001) or case != 'paper15':
        raise ValueError('first bridge uses original paper15 reference/noise grid')
    if len(event_profiles) != 4 or any(v not in (0,1,2) for v in event_profiles):
        raise ValueError('four predeclared source profile indices required')
    if replicate_base < 1000 or int(path) != path or path < 0:
        raise ValueError('new replicate base and nonnegative path required')
    sim, runtime_sha256 = _simulation_runtime(source_root)
    physical_checks, checks_source = _physical_check_function(Path.cwd() if repo_root is None else repo_root)
    runtime_sha256[str(checks_source)] = sha(checks_source)
    # Network and actual natural trajectories are first constructed after the gate.
    context = sim._prepare_case_context(case,96,scenario_suite='reference',
        root_observation='noisy',root_meter_noise_rel=voltage_sigma,root_sigma=.003,impedance_scale=1.)
    validate_plan(plan, len(context.terminals))
    context.net.check_terminal_load_only(strict=True)
    base_kw = float(context.net.base_mva*1000.)
    output = dict(status='complete', training=[], validation=[], event_records=[],
                  truth=[], physical_network=context.net, actual_payment=0.,
                  worst_payment=float(2*.2*plan.sum()), plan_lock_sha256=expected_lock_sha256,
                  policy=policy, path=int(path), dependence=dependence,
                  runtime_sha256=runtime_sha256, source='new synthetic source-generator replicates',
                  old_test_or_truth_files_read=False, labels=list(context.terminals))
    for event in range(4):
        replicate = int(replicate_base)+10*int(path)+event
        z = draw_participation(plan,event,path,dependence,seed=participation_seed)
        event_payment = .2*float(plan[event%2]@z)
        record = dict(event=event, role='training' if event < 2 else 'validation',
                      replicate=replicate, profile_index=int(event_profiles[event]),
                      price_row=plan[event%2].tolist(), status='started',
                      participants=z.astype(int).tolist(), actual_payment=event_payment)
        # Solver failure does not make already-realized contractual payment zero.
        output['actual_payment'] += event_payment
        try:
            base = sim._build_base_trajectory(context,int(event_profiles[event]),replicate)
            p, q, checks = apply_response(base.p_base.to_numpy(),base.q_base.to_numpy(),
                                          plan[event%2],z,base_kw=base_kw)
            if abs(checks['actual_payment']-event_payment) > 1e-12:
                raise AssertionError('quarter-hour payment accounting mismatch')
            record['checks'] = checks
            physical_p = pd.DataFrame(p,index=base.p_base.index,columns=context.terminals)
            physical_q = pd.DataFrame(q,index=base.q_base.index,columns=context.terminals)
            ac, voltage = sim._solve_physical_ac(context,physical_p,physical_q,base.root_true)
            measured = sim._observe_terminal_channels(context,base,physical_p,physical_q,voltage,
                                                      pq_noise_rel=.005,v_noise_rel=voltage_sigma)
            observed = scene(measured['p_measured'].to_numpy(),measured['q_measured'].to_numpy(),
                measured['voltage_measured'].to_numpy(),measured['root_observed'].to_numpy(),
                context.terminals, f'actual_{policy}_{dependence}_p{path}_e{event}',6+event)
            equations = physical_checks(context.net,physical_p,physical_q,ac,(.90,1.10),1e-8)
            checks.update(ac_converged_samples=int(ac['converged'].sum()),
                          voltage_min_pu=float(voltage.to_numpy().min()),
                          voltage_max_pu=float(voltage.to_numpy().max()),
                          voltage_limits_pass=bool(((voltage>=.90)&(voltage<=1.10)).to_numpy().all()),
                          voltage_violation_cells=int(((voltage<.90)|(voltage>1.10)).to_numpy().sum()),
                          branch_ampacity_checked=False,
                          electrical_equations_pass=equations['electrical_equations_pass'],
                          max_nodal_p_residual_pu=equations['max_nodal_p_residual_pu'],
                          max_nodal_q_residual_pu=equations['max_nodal_q_residual_pu'],
                          max_distflow_identity_error_pu2=equations['max_distflow_identity_error_pu2'])
            output['training' if event < 2 else 'validation'].append(observed)
            output['truth'].append(dict(event=event,P_pu=p,Q_pu=q,
                V_pu=voltage.to_numpy(),root_pu=base.root_true.to_numpy(),
                participants=z,ac=ac))
            record.update(status='complete',participants=z.astype(int).tolist(),checks=checks,
                          profile_seed=int(base.shifted_seed),profile=str(base.profile))
        except Exception as error:
            record.update(status='physical_failed',error_type=type(error).__name__,message=str(error))
            output['status'] = 'physical_failed'
        output['event_records'].append(record)
    output['budget_pass'] = bool(output['actual_payment'] <= 1.6+1e-12 and output['worst_payment'] <= 1.6+1e-12)
    output['all_operation_checks_pass'] = bool(output['status']=='complete' and
        all(all(r['checks'][key] for key in ('energy_return_pass','capacity_pass',
            'voltage_limits_pass','electrical_equations_pass')) for r in output['event_records']))
    return output


def evaluate_supports(predicted_support_labels, net):
    """Evaluator-only full effective family, including singleton/root-adjacent blocks.

    Zero-impedance physical edges are exactly observationally inactive; arbitrarily
    weak positive R or X is retained. Repeated identical descendant supports are
    aggregated. No .16 or other magnitude threshold is applied to true supports.
    Call only after the final algorithm output has been fixed.
    """
    import networkx as nx
    graph = net.to_networkx_graph()
    directed = nx.bfs_tree(graph,net.root_bus)
    labels = set(net.buses.loc[net.buses.bus_type.eq('observed_terminal'),'bus_id'].astype(int))
    truth = set()
    for parent,child in directed.edges:
        edge = graph.edges[parent,child]
        if edge['r_ohm'] == 0 and edge['x_ohm'] == 0:
            continue
        support = tuple(sorted(labels & (nx.descendants(directed,child)|{child})))
        if support:truth.add(support)
    predicted = {tuple(sorted(int(v) for v in support)) for support in predicted_support_labels}
    if any(not support or not set(support)<=labels for support in predicted):
        raise ValueError('predicted support outside exact physical terminal labels')
    missed,extra = truth-predicted,predicted-truth
    return dict(full_effective_truth_blocks=len(truth),full_predicted_blocks=len(predicted),
                missed_blocks=[list(s) for s in sorted(missed)],
                extra_blocks=[list(s) for s in sorted(extra)],
                full_block_errors=len(missed)+len(extra),full_recovery=not missed and not extra,
                true_positive_weak_blocks_retained=True,truth_used_only_for_evaluation=True)


def pure_contract_checks():
    """Small tests for unit/budget behavior; creates no natural/AC/test data."""
    p=np.full((96,3),.02);q=np.full((96,3),.008)
    plan=np.array([[2,0,0],[2,0,0]])
    validate_plan(plan,3)
    responded,reactive,checks=apply_response(p,q,plan[0],np.array([1,0,0],bool))
    assert np.allclose(responded[76:80,0],.019)
    assert np.allclose(responded[80:84,0],.021)
    assert np.allclose(reactive[76:80,0],.008-.07/200.)
    assert np.array_equal(responded[:,1:],p[:,1:])
    assert checks['energy_return_pass'] and abs(checks['actual_payment']-.4)<1e-12
    assert np.isclose(2*.2*plan.sum(),1.6)
    same=np.array([[1,1,0],[1,1,0]])
    for event in range(4):
        z=draw_participation(same,event,0,'common')
        assert z[0] == z[1] and not z[2]
    try:validate_plan(np.full((2,3),2))
    except ValueError:pass
    else:raise AssertionError('overbudget accepted')
    return dict(passed=True,AC_calls=0,natural_profiles_generated=0,test_files_loaded=0,
                input_arrays_only=True)
