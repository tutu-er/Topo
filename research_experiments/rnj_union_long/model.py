"""One laminar selection MILP or the original one-new-support WZZT step."""
from __future__ import annotations
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from research_experiments.local_rebuild.model import canonical
from research_experiments.iterative_exit.zero import release_zero_atoms
from research_experiments.rebuild_benchmark.engine import SCALE, audit_vector, reconstruction_start, any_extension, loss
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.joint_exit.model import build, physical_weights
from research_experiments.wzzt_retune.engine import extension_start
from rnj_wzzt.estimation import laminar_l1_milp as core


class DeadlineBank(FloorBank):
    def __init__(self, data, pool, calibration, *, deadline, seconds=300.):
        self.deadline = deadline
        self.lp_limit = seconds
        super().__init__(data, pool, calibration, seconds=seconds)

    def fit(self, family, *, preserve=False):
        remaining = self.deadline-perf_counter()
        if remaining <= .05:
            raise TimeoutError('Total-method deadline reached before fixed-family closure.')
        self.seconds = min(self.lp_limit, remaining)
        self.lp.model.Params.TimeLimit = self.seconds
        return super().fit(family, preserve=preserve)


def training_candidates(common, full, sparse, nj_family):
    """No validation choices enter candidate construction or warm-start ranking."""
    points = [*common['bank']]
    for record in (full, sparse):
        points.extend(record['trajectory']['path'])
        if record.get('exit'):
            points.extend([record['exit']['point'], record['exit']['raw_point']])
    pool = canonical([*common['initial'], *nj_family, *(C for p in points for C in p['supports'])])
    return pool, points


def clean_point(point, n):
    keep = [i for i, (C, r, x) in enumerate(zip(point['supports'], point['r'], point['x']))
            if len(C) in (1, n) or max(r, x) > 1e-9]
    return dict(supports=[point['supports'][i] for i in keep], r=[point['r'][i] for i in keep],
                x=[point['x'][i] for i in keep], objective=point['objective'])


def criterion(point, n, penalty):
    return point['objective']+penalty*sum(1 < len(C) < n for C in point['supports'])


def solve_native(bank, model, start, *, deadline, mip_gap=.0001, heartbeat=None):
    """Native solve with an absolute wall deadline and an audited feasible start."""
    import gurobipy as gp
    then = perf_counter()
    witness = audit_vector(model, start)
    if not witness['valid']:
        raise RuntimeError('Invalid long-budget MILP start.')
    rows = model.rows.build(model.variables.size)
    A = csr_matrix(rows.A); lo, hi = rows.lb, rows.ub
    eq = np.isfinite(lo) & (lo == hi)
    types = np.where(np.asarray(model.variables.integrality, dtype=bool), gp.GRB.BINARY, gp.GRB.CONTINUOUS)
    events, last_ping = [], [then-30.]
    def callback(m, where):
        now = perf_counter()
        if where == gp.GRB.Callback.MIPSOL:
            events.append(dict(seconds=now-then, objective=float(m.cbGet(gp.GRB.Callback.MIPSOL_OBJ))/SCALE))
        if heartbeat and now-last_ping[0] >= 15. and where == gp.GRB.Callback.MIP:
            bound = float(m.cbGet(gp.GRB.Callback.MIP_OBJBND))/SCALE
            incumbent = float(m.cbGet(gp.GRB.Callback.MIP_OBJBST))/SCALE
            heartbeat(dict(solver_seconds=now-then, incumbent=None if abs(incumbent) >= gp.GRB.INFINITY else incumbent,
                           bound=None if abs(bound) >= gp.GRB.INFINITY else bound))
            last_ping[0] = now
    with gp.Model('long_budget_topology', env=bank.lp.environment) as m:
        for key, value in dict(Threads=1, OutputFlag=0, MIPGap=mip_gap, MIPGapAbs=1e-7,
                              FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9,
                              DualReductions=0, Seed=20261031).items():
            m.setParam(key, value)
        v = m.addMVar(model.variables.size, lb=model.variables.lower,
                     ub=np.where(np.isposinf(model.variables.upper), gp.GRB.INFINITY, model.variables.upper),
                     obj=model.variables.objective, vtype=types)
        for mask, sense, rhs in ((eq, '=', lo), (~eq & np.isfinite(lo), '>', lo), (~eq & np.isfinite(hi), '<', hi)):
            if np.any(mask): m.addMConstr(A[mask], v, sense, rhs[mask])
        v.Start = start
        remaining = deadline-perf_counter()
        if remaining <= .05:
            return start, dict(solve_started=False, status=None, warm_start=witness, audit=witness,
                               objective=witness['objective'], bound=None, events=events, seconds=perf_counter()-then)
        m.Params.TimeLimit = remaining
        m.optimize(callback)
        values = np.asarray(v.X) if m.SolCount else np.asarray(start)
        audit = audit_vector(model, values)
        if not audit['valid'] or audit['objective'] > witness['objective']+1e-8:
            values, audit = np.asarray(start), witness
        bound = float(m.ObjBound)/SCALE
        bound = None if abs(bound) >= gp.GRB.INFINITY else bound
        if bound is not None and bound > audit['objective']+1e-8:
            raise RuntimeError('MILP bound exceeds verified feasible objective.')
        record = dict(solve_started=True, status=int(m.Status), objective=audit['objective'], bound=bound,
                      solver_objective=float(m.ObjVal)/SCALE if m.SolCount else None,
                      relative_gap=float(m.MIPGap) if m.SolCount else None, requested_gap=mip_gap,
                      nodes=float(m.NodeCount), warm_start=witness, audit=audit, events=events,
                      seconds=perf_counter()-then, time_limit=remaining, binary_count=int(np.count_nonzero(model.variables.integrality)),
                      variables=model.variables.size, constraints=model.rows.size)
        return values, record


def joint_rebuild(bank, pool, warm_points, *, penalty, deadline, heartbeat=None):
    then = perf_counter()
    model, meta = build(bank.scaled, pool, penalty=penalty)
    points = [clean_point(p, bank.data.n) for p in warm_points]
    for point in points:
        if abs(loss(bank.data, point)-point['objective']) > 1e-8:
            raise RuntimeError('Warm prediction does not reproduce the current observations.')
    warm = min(points, key=lambda p: criterion(p, bank.data.n, penalty))
    start = reconstruction_start(bank, model, meta, warm)
    values, event = solve_native(bank, model, start, deadline=deadline, heartbeat=heartbeat)
    indices = [*range(len(meta['keep'])), *[len(meta['keep'])+i for i, a in enumerate(values[meta['selection']]) if a > .5]]
    r, rp = physical_weights(values[model.r]); x, xp = physical_weights(values[model.x])
    raw = dict(supports=[meta['all_supports'][i] for i in indices], r=r[indices].tolist(), x=x[indices].tolist())
    raw['objective'] = loss(bank.data, raw)
    raw = clean_point(raw, bank.data.n)
    closed = bank.fit(raw['supports'], preserve=True)
    point = clean_point(closed, bank.data.n)
    value = criterion(point, bank.data.n, penalty)
    if value > event['objective']+1e-8 or value > criterion(warm, bank.data.n, penalty)+1e-8:
        raise RuntimeError('Joint closure lost the feasible incumbent guarantee.')
    if event['bound'] is not None and event['bound'] > value+1e-8:
        raise RuntimeError('Joint bound exceeds closed score.')
    return dict(point=point, raw_point=raw, warm=warm, score=value, penalty=penalty,
                event=event, seconds=perf_counter()-then, coefficient_projection=max(rp, xp),
                pool=pool, internal_candidates=len(meta['candidates']), conflicts=meta['conflicts'],
                cardinality_cuts=meta['cardinality_cuts'], clique_cuts=meta['clique_cuts'])


def pure_search(bank, *, structural_deadline, single_mip=3600., heartbeat=None, checkpoint=None, zero_release=True):
    """Leaf/full basis only; no NJ/RNJ pool, initialization, tail or exit."""
    then = perf_counter(); n = bank.data.n
    family = canonical([*((i,) for i in range(n)), tuple(range(n))])
    current = bank.fit(family, preserve=True)
    path, events, zero_events = [current], [], []
    if checkpoint: checkpoint(dict(path=path, point=current, attempts=events, seconds=perf_counter()-then))
    reason = 'step_limit'
    for iteration in range(3*n):
        if structural_deadline-perf_counter() <= .1:
            reason = 'structural_deadline'; break
        family = canonical(current['supports'])
        new = any_extension(family, n, ())
        if new is None:
            reason = 'structurally_complete'; break
        model = core.build_extension_model(bank.scaled, family, r_upper_bound=2., x_upper_bound=2.)
        start = extension_start(model, family, new, current, bank.lp.p, bank.lp.q, bank.lp.y)
        values, event = solve_native(bank, model, start, deadline=min(structural_deadline, perf_counter()+single_mip), heartbeat=heartbeat)
        C = core._round_and_validate_support(values[model.z], family, n)
        q = bank.fit([*family, C], preserve=True)
        q, cleanup = release_zero_atoms(bank, q, enabled=zero_release, stage='pure_extension')
        before = current['objective']; gate = 1e-10+.001*before
        accepted = before-q['objective'] > gate
        event.update(iteration=iteration, kind='pure_wzzt_extension', support=C, before=before, after=q['objective'],
                     gain=before-q['objective'], threshold=gate, accepted=accepted, elapsed=perf_counter()-then,
                     zero_events=cleanup)
        events.append(event)
        if accepted:
            current = q; path.append(current); zero_events.extend(cleanup)
        if checkpoint: checkpoint(dict(path=path, point=current, attempts=events, seconds=perf_counter()-then))
        if not accepted:
            reason = 'bound_excludes_gain' if event['bound'] is not None and before-event['bound'] <= gate else 'no_gain_unresolved'
            break
    return dict(point=current, path=path, attempts=events, zero_events=zero_events, seconds=perf_counter()-then, stop_reason=reason,
                initial_supports=canonical([*((i,) for i in range(n)), tuple(range(n))]), no_RNJ_NJ_supports=True)
