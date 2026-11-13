"""Maximize the original common evidence increment over the device domain.

This audits the fixed-gamma infeasibility conclusions of placement.py. There
is no procurement-cost restriction, so the result is a resource-capacity
check, not an estimate of economical or actual AC recovery improvement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from ... import io
from .model import CustomerCase
from .placement import CandidateEvidence, REFERENCE, qualify
from .validate import ROOT, csv_write


def maximum_gain(evidence, env, seconds=2., previous_points=()):
    import gurobipy as gp
    began = perf_counter()
    c = evidence.case
    box_upper = float(evidence.power_box_upper().min())
    initial = np.zeros(c.shape)
    initial_gain, initial_source = 0., 'zero_action'
    for name, old_case, old_point in previous_points:
        mapped = np.zeros(c.shape)
        old_index = {r['id']: j for j, r in enumerate(old_case.resources)}
        for j, resource in enumerate(c.resources):
            if resource['id'] in old_index:
                mapped[:, j] = old_point[:, old_index[resource['id']]]
        if c.audit(mapped)['passed']:
            gain = float(evidence.direct(mapped).min())
            if gain > initial_gain:
                initial, initial_gain, initial_source = mapped, gain, 'verified_'+name+'_schedule'
    with gp.Model('maximum_common_original_gain', env=env) as m:
        for k, v in dict(OutputFlag=0, Threads=1, NonConvex=2, MIPGap=1e-5,
                         FeasibilityTol=1e-9, NumericFocus=1).items():
            m.setParam(k, v)
        u, _ = c.attach(m)
        g = m.addVar(lb=0., ub=max(0., box_upper), name='common_gain')
        for Q, a in zip(evidence.Q, evidence.a, strict=True):
            expression = sum(u[h]@Q@u[h] for h in range(c.H))+2*(a*u).sum()
            m.addConstr(g <= expression)
        u.Start = initial
        g.Start = initial_gain
        m.setObjective(g, gp.GRB.MAXIMIZE)
        m.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        m.optimize()
        point = initial.copy()
        source = 'verified_initial_point_retained'
        candidate_audit, repair = None, None
        if m.SolCount:
            candidate = np.asarray(u.X)
            candidate_audit = c.audit(candidate)
            if not candidate_audit['passed']:
                # Repair a numerical residual on the original linear domain;
                # keep its audit instead of silently replacing it by zero.
                with gp.Model('repair_capacity_resource_residual', env=env) as p:
                    p.Params.OutputFlag = 0
                    p.Params.Threads = 1
                    p.Params.FeasibilityTol = 1e-9
                    p.Params.OptimalityTol = 1e-9
                    v, _ = c.attach(p)
                    p.setObjective(((v-candidate)*(v-candidate)).sum())
                    p.Params.TimeLimit = .2
                    p.optimize()
                    if p.SolCount:
                        adjusted = np.asarray(v.X)
                        repair = dict(status=int(p.Status), l1_adjustment_kw=float(abs(adjusted-candidate).sum()),
                                      passed=c.audit(adjusted)['passed'])
                        if repair['passed']:
                            candidate = adjusted
            if c.audit(candidate)['passed'] and evidence.direct(candidate).min() > initial_gain:
                point = candidate
                source = 'native_QCQP_point_repaired_on_original_domain' if repair is not None else 'native_QCQP_improved_verified_initial_point'
        gains = evidence.direct(point)
        if not np.allclose(gains, evidence.values(point), rtol=1e-7, atol=1e-8):
            raise ArithmeticError('Capacity point fails original full-H gain check')
        lower = float(gains.min())
        try:
            native_upper = float(m.ObjBound)
            if not np.isfinite(native_upper):
                native_upper = None
        except (AttributeError, gp.GurobiError):
            native_upper = None
        withdrawn = native_upper is not None and native_upper < lower-1e-7*max(1., abs(lower))
        upper = box_upper if withdrawn or native_upper is None else min(box_upper, max(lower, native_upper))
        return dict(maximum_gain_feasible_lower=lower, maximum_gain_valid_upper=upper,
            absolute_gap=upper-lower, relative_gap=(upper-lower)/max(abs(lower), 1e-12),
            bound_closed=upper-lower <= 1e-7 or (upper-lower)/max(abs(lower), 1e-12) <= 1e-5,
            native_status=int(m.Status), source=source, native_bound_withdrawn=withdrawn,
            native_candidate_audit=candidate_audit, candidate_repair=repair,
            initialization=dict(source=initial_source, verified_gain=initial_gain,
                                verified_cost_yuan=c.fee(initial), mapping_and_verification_in_timing=True),
            gain_improvement_from_initial=lower-initial_gain,
            witness_service_fee_yuan=c.fee(point), resource_audit=c.audit(point),
            seconds=perf_counter()-began), point


def run(previous, out):
    import gurobipy as gp
    if out.exists():
        raise FileExistsError('Preserve previous capacity checks: '+str(out))
    if not (previous/'complete.json').exists():
        raise ValueError('The placement feasibility audit must be complete')
    prior = io.read(previous/'protocol.json')
    for rel, digest in prior['sources'].items():
        if io.ref(ROOT/rel)['sha256'] != digest:
            raise ValueError('Capacity source differs from the preceding feasibility audit: '+rel)
    out.mkdir(parents=True)
    _, eligible, scores = qualify()
    if [e['request']['id'] for e in eligible] != prior['eligible_ids']:
        raise ValueError('Qualified history set changed')
    configs = prior['configurations']
    io.write(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        previous=io.ref(previous/'protocol.json'), sources=dict(capacity=io.ref(__file__)),
        purpose='Original gain capacity over the complete device domain, no price/budget constraint',
        seconds_per_QCQP=2., environment=io.environment(),
        initialization='Reuse verified earlier-layout schedules by resource ID. Their solves are separately timed in the same batch; mapping and rechecking are included in each subsequent solve. Zero is always available.',
        independent_new_days=0, actual_AC_or_posterior_simulation=False))
    complete_case = CustomerCase(configs[prior.get('common_catalog', 'focused')])
    bp, bq = complete_case.device_baseline_pq_kw()
    results, points = [], {}
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for entry in eligible:
            identity = entry['request']['id']
            score = scores[identity]
            H = np.array([p['H'] for p in score['parameters']])
            file = REFERENCE/'population/jobs'/identity/'day000/planner.npz'
            with np.load(file, allow_pickle=False) as z:
                forecast = np.roll(z['forecast'], -32, axis=0)+np.column_stack((bp, bq))/200.
            previous_points = []
            for layout, config in configs.items():
                case = CustomerCase(config, labels=score['labels'])
                evidence = CandidateEvidence(case, H, entry['target'], entry['competitors'],
                                             forecast, score['sigma2'], 200.)
                result, point = maximum_gain(evidence, env, previous_points=previous_points)
                result.update(id=identity, layout=layout)
                results.append(result)
                points[identity+'__'+layout] = point
                previous_points.append((layout, case, point))
    io.write(out/'private_results.json', results)
    np.savez_compressed(out/'private_capacity_schedules.npz', **points)
    keys = ['id', 'layout', 'maximum_gain_feasible_lower', 'maximum_gain_valid_upper',
            'absolute_gap', 'relative_gap', 'bound_closed', 'native_status',
            'source', 'witness_service_fee_yuan', 'seconds']
    csv_write(out/'maximum_gain.csv', [{key: r[key] for key in keys} for r in results])
    old = io.read(previous/'private_results.json')
    contradictions = []
    for r in old:
        cap = next(x for x in results if x['id'] == r['id'] and x['layout'] == r['layout'])
        if r.get('infeasibility_proved') and r['mode'] == 'nominal' and cap['maximum_gain_feasible_lower'] > r['gamma']+1e-6:
            contradictions.append(dict(id=r['id'], layout=r['layout'], gamma=r['gamma']))
    io.write(out/'complete.json', dict(status='contradiction_requires_correction' if contradictions else 'passed',
        results=len(results), bound_closed=sum(r['bound_closed'] for r in results),
        contradictions=contradictions, source=io.ref(__file__), independent_new_days=0))
    print(io.read(out/'complete.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.previous.resolve(), a.output.resolve())
