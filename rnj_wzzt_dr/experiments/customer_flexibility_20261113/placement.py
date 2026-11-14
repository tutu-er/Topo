"""Bounded development audit of sparse resources against frozen candidates.

Truth is used only to qualify historical development inputs. The planning
target and competitors remain the existing history-only choices. No AC
observations, true impedances or posterior performance enter procurement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from ... import io
from ..residual_guard_ed_20261113.model import calibrate, direct
from .model import CustomerCase
from .validate import ROOT, csv_write


REFERENCE = ROOT/'outputs/linear_ac_population_20261113_v2'
TRUTH = ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json'


def terms(case, A, B, K, forecast, sigma2, base_kw):
    """Same residual-guard quadratic as the owning model, mapped to devices."""
    n = case.N
    D, E = A-B, K-A
    normal = np.sqrt(2*sigma2)
    if hasattr(case, 'response_effect'):
        M = case.response_effect(D, base_kw)/normal
        N = case.response_effect(E, base_kw)/normal
    else:
        M = (D[:, case.node]+D[:, n+case.node]*case.qratio)/(base_kw*normal)
        N = (E[:, case.node]+E[:, n+case.node]*case.qratio)/(base_kw*normal)
    b, e = forecast@D.T/normal, forecast@E.T/normal
    Q = case.repeat*(M.T@M+N.T@M+M.T@N)
    a = b.reshape(case.H, case.repeat, n).sum(1)@M
    a += e.reshape(case.H, case.repeat, n).sum(1)@M
    a += b.reshape(case.H, case.repeat, n).sum(1)@N
    return (Q+Q.T)/2, a


class CandidateEvidence:
    def __init__(self, case, H, target, competitors, forecast, sigma2, base_kw, means=None):
        self.case, self.H, self.target = case, H, target
        self.competitors, self.forecast = competitors, forecast
        self.sigma2, self.base_kw = sigma2, base_kw
        self.nominal = case.evidence(H, target, competitors, forecast, sigma2, base_kw, 2.)
        self.pairs = [(target, j, H[target]) for j in competitors]
        if means is not None:
            for j in competitors:
                self.pairs.extend([(target, j, means[target]), (j, target, means[j])])
        qa = [terms(case, H[i], H[j], K, forecast, sigma2, base_kw) for i, j, K in self.pairs]
        self.Q, self.a = np.array([v[0] for v in qa]), np.array([v[1] for v in qa])

    def required(self, gamma):
        return np.r_[np.full(len(self.competitors), gamma),
                     np.zeros(len(self.pairs)-len(self.competitors))]

    def values(self, u):
        return np.einsum('hi,kij,hj->k', u, self.Q, u)+2*np.einsum('khi,hi->k', self.a, u)

    def direct(self, u):
        shift = self.case.shift_pq_pu(u, self.base_kw)
        return np.array([direct(self.H[i], self.H[j], K, self.forecast, shift, self.sigma2)
                         for i, j, K in self.pairs])

    def power_box_upper(self):
        """Original-domain upper bound, relaxing all energy/time constraints."""
        lower = np.repeat(self.case.lower, self.case.repeat, axis=0)
        upper = np.repeat(self.case.upper, self.case.repeat, axis=0)
        answer = []
        for b, F, scale in zip(self.nominal.base, self.nominal.effect, self.nominal.scales, strict=True):
            positive, negative = np.maximum(F, 0), np.minimum(F, 0)
            lo = lower@positive.T+upper@negative.T
            hi = upper@positive.T+lower@negative.T
            answer.append(float(np.maximum(2*b*lo+lo*lo, 2*b*hi+hi*hi).sum()*scale*scale))
        return np.array(answer)


def solve(evidence, gamma, *, seconds=3., env):
    """Supporting LP initialization followed by the original nonconvex QCQP."""
    import gurobipy as gp
    began = perf_counter()
    case = evidence.case
    required = evidence.required(gamma)
    upper = evidence.power_box_upper()
    if np.any(upper < gamma-1e-8):
        return dict(feasible=False, status='infeasible_by_valid_relaxed_power_bound',
                    infeasibility_proved=True, nominal_gain_upper=upper,
                    seconds=perf_counter()-began, cost_yuan=None, lower_bound_yuan=None), None
    warm, initial, seed = None, None, None
    with gp.Model('customer_supporting_LP', env=env) as m:
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.FeasibilityTol = 1e-9
        u, fee = case.attach(m)
        for a, rhs in zip(evidence.a[:len(evidence.competitors)], required[:len(evidence.competitors)], strict=True):
            m.addConstr(2*(a*u).sum() >= rhs*(1+1e-7)+1e-8)
        m.setObjective(fee)
        m.Params.TimeLimit = min(1., max(.001, seconds-(perf_counter()-began)))
        m.optimize()
        initial = dict(source='nominal_supporting_LP', native_status=int(m.Status),
                       domain_feasible=False, evidence_feasible=False, cost_yuan=None)
        if m.SolCount:
            warm = np.asarray(u.X)
            initial.update(domain_feasible=case.audit(warm)['passed'], cost_yuan=case.fee(warm),
                           evidence_feasible=bool(np.min(evidence.direct(warm)-required) >= -1e-7))
            if initial['domain_feasible'] and initial['evidence_feasible']:
                seed = warm.copy()
    with gp.Model('customer_frozen_H_QCQP', env=env) as m:
        for key, value in dict(OutputFlag=0, Threads=1, NonConvex=2, FeasibilityTol=1e-9,
                               NumericFocus=1, MIPGap=1e-5).items():
            m.setParam(key, value)
        u, fee = case.attach(m)
        for Q, a, rhs in zip(evidence.Q, evidence.a, required, strict=True):
            expression = sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            m.addConstr(expression >= rhs+1e-8)
        if warm is not None:
            u.Start = warm
        m.setObjective(fee)
        m.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        m.optimize()
        native_status = int(m.Status)
        try:
            bound = float(m.ObjBound)
            if not np.isfinite(bound):
                bound = None
        except (AttributeError, gp.GurobiError):
            bound = None
        point, source = seed, 'supporting_LP_initial_point_retained'
        if m.SolCount:
            candidate = np.asarray(u.X)
            if case.audit(candidate)['passed'] and np.min(evidence.direct(candidate)-required) >= -1e-7:
                if point is None or case.fee(candidate) < case.fee(point)-1e-10:
                    point = candidate
                    source = 'native_QCQP_found_feasible' if seed is None else 'native_QCQP_improved_initial_point'
        if point is None:
            return dict(feasible=False, status='native_infeasible' if native_status == 3 else 'unresolved',
                        native_status=native_status, infeasibility_proved=native_status == 3,
                        initial=initial, seconds=perf_counter()-began, cost_yuan=None,
                        lower_bound_yuan=bound, nominal_gain_upper=upper), None
        actual = evidence.direct(point)
        calculated = evidence.values(point)
        if not np.allclose(actual, calculated, rtol=1e-7, atol=1e-8):
            raise ArithmeticError('Original full-H gain and resource QCQP differ')
        cost = case.fee(point)
        withdrawn = bound is not None and bound > cost+1e-7*max(1., cost)
        lower = 0. if withdrawn or bound is None else max(0., min(bound, cost))
        gap = max(0., cost-lower)/max(cost, 1e-12)
        return dict(feasible=True, status='feasible', native_status=native_status, source=source,
                    cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
                    global_optimality_verified=gap <= 1e-5, lower_bound_withdrawn=withdrawn,
                    native_infeasibility_withdrawn=native_status == 3,
                    initial=initial, improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
                    seconds=perf_counter()-began, nominal_gain_upper=upper,
                    nominal_gains=actual[:len(evidence.competitors)],
                    minimum_residual_guard_gain=None if len(actual) == len(evidence.competitors)
                        else float(np.min(actual[len(evidence.competitors):])), resource_audit=case.audit(point)), point


def qualify():
    entries = io.read(REFERENCE/'inputs.json')
    evaluation = {x['id']: x for x in io.read(TRUTH)}
    all_rows, eligible, scores = [], [], {}
    for e in entries:
        if e['request']['case'] != 'paper15':
            continue
        s = io.read(e['scores']['path'])
        if io.ref(e['scores']['path'])['sha256'] != e['scores']['sha256']:
            raise ValueError('Frozen candidate scores changed')
        truth = evaluation[e['request']['id']]
        true = next((i for i, m in enumerate(s['models']) if m['topology_id'] == truth['truth_id']), None)
        p_true = None if true is None else s['models'][true]['probability']
        all_rows.append(dict(id=e['request']['id'], truth_present=true is not None, p_true=p_true,
                             p_map=s['p_map'], historical_MAP_correct=true == e['target'],
                             qualified=true is not None and p_true < .99))
        scores[e['request']['id']] = s
        if all_rows[-1]['qualified']:
            eligible.append(e)
    return all_rows, eligible, scores


def configurations():
    original = CustomerCase().config
    compact = deepcopy(original)
    compact.update(id='paper15_compact_electric_buffer_v2')
    compact['thermal'].update(water_litres=300., electric_buffer_kwh=.85)
    focused = deepcopy(compact)
    focused.update(id='paper15_focused_contracts_v2',
                   placement_rule='Development-case qualification uses truth coverage only; new contracts at 107/102 use frozen-candidate response differences, not true branches or new-day outcomes.')
    pv = deepcopy(focused['resources'][0])
    pv.update(id='pv107', node_label=107)
    hp = deepcopy(focused['resources'][-1])
    hp.update(id='hp102', node_label=102, fee_positive_yuan_per_kwh=.15, fee_negative_yuan_per_kwh=.15)
    focused['resources'].extend([pv, hp])
    focused['sources'].extend([
        dict(id='bodal2022', doi='10.1049/gtd2.12651',
             url='https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.12651',
             used='Power/energy flexibility and rebound; Section 2.1.2 discusses 200 L, 2-3 kW residential electric water heaters.',
             difference='Our 300 L heat-pump buffer, COP=3 and 0.85 kWh electrical contract are explicit synthetic assumptions, not that paper device calibration.'),
        dict(id='sanandaji2014', url='https://arxiv.org/abs/1310.1693',
             used='Definition 1: linear dissipative energy state with power and energy limits.',
             difference='Our single resource represents a continuously controllable contracted group; no general exactness claim for arbitrary on/off devices.')])
    compact['sources'] = deepcopy(focused['sources'])
    return dict(original=original, compact=compact, focused=focused)


def directional_configurations():
    """Same device counts, or two extra contracts, with usable directions.

    The first contrast needs increased demand at 108-110 or reduced demand
    at 107. A PV at 107 offers the wrong one-sided action when already fully
    producing. The earlier failed design remains a development record.
    """
    compact = configurations()['compact']
    relocated = deepcopy(compact)
    relocated.update(id='paper15_directional_contracts_v3',
        placement_rule='Replace PV111 by PV110 and HP114 by HP102, using frozen development candidates and the signed planning-forecast evidence derivative. Same device counts, ratings, availability and quotations; truth only qualified development inputs.')
    for r in relocated['resources']:
        if r['id'] == 'pv111':
            r.update(id='pv110', node_label=110)
        elif r['id'] == 'hp114':
            r.update(id='hp102', node_label=102)
    augmented = deepcopy(compact)
    augmented.update(id='paper15_extra_directional_contracts_v3',
        placement_rule='Keep the original eight contracts and additionally procure PV110 and HP102 from the common pre-existing device catalog.')
    augmented['resources'].extend(deepcopy([r for r in relocated['resources'] if r['id'] in ('pv110', 'hp102')]))
    return dict(compact=compact, relocated=relocated, augmented=augmented)


def run(out, directional=False):
    import gurobipy as gp
    if out.exists():
        raise FileExistsError('Preserve previous audits: '+str(out))
    out.mkdir(parents=True)
    started = perf_counter()
    rows, eligible, scores = qualify()
    configs = directional_configurations() if directional else configurations()
    common_key = 'augmented' if directional else 'focused'
    for key, value in configs.items():
        io.write(out/('case_'+key+'.json'), value)
    source_files = [Path(__file__), Path(__file__).with_name('model.py'),
                    Path(__file__).with_name('case.json'),
                    Path(__file__).parents[1]/'residual_guard_ed_20261113/model.py',
                    Path(__file__).parents[1]/'expected_logodds_admm_20261113/evidence.py']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), workers=1, environment=io.environment(),
        sources={str(p.relative_to(ROOT)): io.ref(p)['sha256'] for p in source_files},
        eligibility='All paper15 frozen 101 libraries containing the complete truth candidate with historical true weight < .99; truth used offline for this qualification only.',
        target='Existing historical MAP; competitors unchanged from the frozen history-only entry.',
        eligible_ids=[e['request']['id'] for e in eligible], budgets=dict(gamma=[2., 4.], seconds_per_QCQP=3.,
            nominal_checks=len(configs)*len(eligible)*2,
            guarded_checks=sum(k != 'original' for k in configs)*len(eligible),
            guarded_gamma=2.),
        data=io.ref(REFERENCE/'inputs.json'), offline_qualification=io.ref(TRUTH),
        common_forecast='Previously sealed day000 planning forecast, rolled -32 slots from midnight to 08:00, plus all common-catalog device baselines. Additional PV/HP are present but uncontracted in the smaller control sets. No future actual P/Q/V used.',
        common_catalog=common_key, directional_comparison=directional,
        evaluation_scope='Planning feasibility, original gain and monetary bounds only. No simulated Y=Htarget P+e and no posterior success-rate experiment.',
        configurations=configs, independent_new_days=0)
    io.write(out/'protocol.json', protocol)
    io.write(out/'lock.json', dict(pid=os.getpid(), protocol=io.ref(out/'protocol.json')))
    for source in source_files:
        dest = out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    csv_write(out/'qualification.csv', rows)
    positions, zero_evidence, results, failures, points = [], [], [], [], {}
    full_case = CustomerCase(configs[common_key])
    bp, bq = full_case.device_baseline_pq_kw()
    contexts = []
    for entry in eligible:
        identity = entry['request']['id']
        s = scores[identity]
        if io.ref(entry['request']['observed']['path'])['sha256'] != entry['request']['observed']['sha256']:
            raise ValueError('Historical observations changed')
        H = np.array([p['H'] for p in s['parameters']])
        _, means = calibrate(s, entry['request']['observed']['path'])
        directory = REFERENCE/'population/jobs'/identity/'day000'
        seal = io.read(directory/'policy_seal.json')
        if io.ref(directory/'planner.npz')['sha256'] != seal['planner']['sha256']:
            raise ValueError('Previously sealed planning forecast changed')
        with np.load(directory/'planner.npz', allow_pickle=False) as z:
            forecast = np.roll(z['forecast'], -32, axis=0)+np.column_stack((bp, bq))/200.
        contexts.append(dict(entry=entry, score=s, H=H, means=means, forecast=forecast))
        for j in entry['competitors']:
            D = H[entry['target']]-H[j]
            distance = float(np.sum((forecast@D.T)**2)/(2*s['sigma2']))
            history_odds = s['models'][entry['target']]['log_probability']-s['models'][j]['log_probability']
            zero_evidence.append(dict(id=identity, competitor=j, historical_log_odds=history_odds,
                nominal_no_DFS_prediction_distance=distance,
                expected_no_DFS_log_odds_under_target=history_odds+distance,
                interpretation='Expected log odds under frozen target response; not mean posterior or AC success probability'))
            for name, q in [('pv_ev', 0.), ('thermal', float(np.tan(np.arccos(.95))))]:
                norm = np.linalg.norm(D[:, :15]+q*D[:, 15:], axis=0)
                for i, label in enumerate(s['labels']):
                    positions.append(dict(id=identity, competitor=j, channel=name, node_label=label,
                        relative_response_difference=float(norm[i]/max(norm.max(), 1e-300))))
    csv_write(out/'node_response_differences.csv', positions)
    csv_write(out/'zero_evidence.csv', zero_evidence)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for context in contexts:
            e, s, H, F = context['entry'], context['score'], context['H'], context['forecast']
            for layout, config in configs.items():
                case = CustomerCase(config, labels=s['labels'])
                modes = ['nominal', 'residual_guards'] if layout != 'original' else ['nominal']
                for mode in modes:
                    evidence = CandidateEvidence(case, H, e['target'], e['competitors'], F, s['sigma2'],
                                                 200., context['means'] if mode == 'residual_guards' else None)
                    for gamma in ([2., 4.] if mode == 'nominal' else [2.]):
                        common = dict(id=e['request']['id'], layout=layout, mode=mode, gamma=gamma,
                                      target=e['target'], competitors=e['competitors'])
                        try:
                            result, point = solve(evidence, gamma, env=env)
                            result.update(common)
                            results.append(result)
                            if point is not None:
                                points[f"{e['request']['id']}__{layout}__{mode}__g{gamma:g}"] = point
                        except Exception as exc:
                            failures.append(dict(common, error=repr(exc)))
                        io.write(out/'checkpoint.json', dict(results=len(results), failures=failures))
    io.write(out/'private_results.json', results)
    np.savez_compressed(out/'private_plans.npz', **points)
    scalar_keys = ['id', 'layout', 'mode', 'gamma', 'feasible', 'status', 'source', 'cost_yuan',
                   'lower_bound_yuan', 'relative_gap', 'global_optimality_verified', 'infeasibility_proved',
                   'minimum_residual_guard_gain', 'improvement_from_verified_initial_yuan', 'seconds']
    table = [{key: r.get(key) for key in scalar_keys} for r in results]
    csv_write(out/'planning_feasibility.csv', table)
    summary = dict(paper15_libraries=len(rows), truth_present=sum(r['truth_present'] for r in rows),
        missing_truth=sum(not r['truth_present'] for r in rows),
        historical_MAP_below99=sum(r['p_map'] < .99 for r in rows), qualified=len(eligible),
        results=len(results), feasible=sum(r['feasible'] for r in results),
        proved_infeasible=sum(r.get('infeasibility_proved', False) for r in results),
        unresolved=sum(not r['feasible'] and not r.get('infeasibility_proved', False) for r in results),
        global_fee_certificates=sum(r.get('global_optimality_verified', False) for r in results),
        worker_failures=failures, independent_new_days=0, seconds=perf_counter()-started)
    io.write(out/'summary.json', summary)
    io.write(out/'complete.json', dict(status='complete_with_failures' if failures else 'complete',
                                      summary=io.ref(out/'summary.json'), protocol=io.ref(out/'protocol.json')))
    print(summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--directional', action='store_true')
    a = p.parse_args()
    run(a.output.resolve(), a.directional)
