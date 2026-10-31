"""Matched RNJ-preselection and WZZT-tail interventions; at most one joint tail."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rebuild_benchmark.engine import any_extension, solve
from research_experiments.wzzt_retune.engine import extension_start
from .model import open_small, remainder, joint_fit


def original_step(bank, current, *, deadline, per_mip=2., relative=.001):
    family = canonical(current['supports']); n = bank.data.n
    new = any_extension(family, n, bank.lp.pool)
    if new is None:
        return current, dict(kind='wzzt_extension', accepted=False, structurally_complete=True)
    model = core.build_extension_model(bank.scaled, family, r_upper_bound=2., x_upper_bound=2.)
    warm = current; checked = 0
    for c in bank.lp.pool:
        if c in family or not all(compatible(c, s) for s in family):
            continue
        if deadline-perf_counter() < per_mip+.1:
            break
        q = bank.fit([*family, c], preserve=True); checked += 1
        if q['objective'] < warm['objective']-1e-12:
            warm, new = q, c
    remaining = deadline-perf_counter()
    if remaining <= .05:
        return current, dict(kind='wzzt_extension', accepted=False, time_exhausted=True)
    start = extension_start(model, family, new, warm, bank.lp.p, bank.lp.q, bank.lp.y)
    values, event = solve(bank, model, start, seconds=min(per_mip, remaining))
    best = warm
    if values is not None:
        support = core._round_and_validate_support(values[model.z], family, n)
        q = bank.fit([*family, support], preserve=True)
        if q['objective'] < best['objective']:
            best = q
    gain = current['objective']-best['objective']; gate = 1e-10+relative*current['objective']
    event.update(kind='wzzt_extension', warm_checked=checked, before=current['objective'],
                 decoded_objective=best['objective'], gain=gain, threshold=gate, accepted=gain>gate,
                 added=[s for s in best['supports'] if s not in family],
                 maximum_remainder_before=remainder(family, n),
                 maximum_remainder_after=remainder(best['supports'], n))
    return (best if gain>gate else current), event


def search(bank, initial, *, pre_joint=False, tail_joint=False, seconds=20., pre_seconds=5., per_mip=2., relative=.001):
    then = perf_counter(); deadline = then+seconds; n = bank.data.n
    initial = canonical([*initial, *((i,) for i in range(n)), tuple(range(n))])
    raw = bank.fit(initial, preserve=True); current = raw
    path = [raw]; attempts = []; states = dict(rnj=raw); reason = 'wall_limit'
    if pre_joint and perf_counter() < deadline:
        keep, released = open_small(initial, n)
        current, event = joint_fit(bank, current, keep, seconds=min(pre_seconds, deadline-perf_counter()), stage='rnj_pre_joint')
        event['released'] = released; attempts.append(event)
        if event['accepted']:
            path.append(current)
    states['after_pre'] = current
    for iteration in range(3*n):
        if deadline-perf_counter() <= .05:
            break
        size = remainder(current['supports'], n)
        if tail_joint and size <= 5:
            states['before_tail'] = current
            keep, released = open_small(current['supports'], n)
            current, event = joint_fit(bank, current, keep, seconds=deadline-perf_counter(), stage='wzzt_tail_joint')
            event.update(released=released, iteration=iteration); attempts.append(event)
            if event['accepted']:
                path.append(current)
            reason = ('joint_tail_gap_met' if event.get('status') == 2 else
                      'joint_tail_no_candidates' if event.get('no_candidates') else 'joint_tail_unresolved')
            break
        current, event = original_step(bank, current, deadline=deadline, per_mip=per_mip, relative=relative)
        event['iteration'] = iteration; attempts.append(event)
        if event['accepted']:
            path.append(current)
        else:
            if event.get('structurally_complete'):
                reason = 'structurally_complete'
            elif event.get('time_exhausted'):
                reason = 'wall_limit'
            else:
                bound = event.get('dual_bound'); gate = event['threshold']
                reason = 'bound_excludes_gain' if bound is not None and current['objective']-bound <= gate else 'no_gain_unresolved'
            break
    else:
        reason = 'step_limit'
    states['endpoint'] = current
    return dict(point=current, path=path, attempts=attempts, states=states,
                stop_reason=reason, seconds=perf_counter()-then, pre_joint=pre_joint, tail_joint=tail_joint,
                maximum_remainder=remainder(current['supports'], n), tree_global_optimal=False)
