"""One temporary deletion, with optional exact-support ban, then original steps.

No truth, test observations, new penalty or candidate-ranking heuristic enters
this module. Other old supports remain fixed during the resumed forward phase.
"""
from itertools import combinations
from time import perf_counter
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rebuild_benchmark.engine import hierarchy_children, SCALE, reconstruction_start, loss
from research_experiments.joint_five.model import remainder, open_small, build_joint
from research_experiments.wzzt_retune.engine import extension_start
from research_experiments.rnj_union_long.model import DeadlineBank, solve_native, criterion
from research_experiments.iterative_exit.zero import release_zero_atoms


class MaskBank(DeadlineBank):
    def __init__(self, *args, forbidden=(), **kwargs):
        self.forbidden = set(canonical(forbidden))
        super().__init__(*args, **kwargs)

    def fit(self, family, *, preserve=False):
        if set(canonical(family)) & self.forbidden:
            raise ValueError('A forbidden support reached a fixed-family LP.')
        return super().fit(family, preserve=preserve)


def select_block(point, n):
    """Largest retained proper block (>5 terminals), including zero weights."""
    eligible = canonical(C for C in point['supports'] if 5 < len(C) < n)
    return min(eligible, key=lambda C: (-len(C), C)) if eligible else None


def legal_start(family, n, pool, forbidden):
    present = set(canonical(family)); forbidden = set(canonical(forbidden))
    options = list(canonical(pool))
    for blocks in hierarchy_children(family, n).values():
        options.extend(tuple(sorted((*a, *b))) for a, b in combinations(blocks, 2))
    return next((C for C in options if 1 < len(C) < n and C not in present and C not in forbidden
                 and all(compatible(C, D) for D in family)), None)


def ban_membership(model, block, n):
    """Exact Hamming exclusion; does not ban subsets or supersets of block."""
    target = set(block)
    model.rows.add({model.z.start+i: -1. if i in target else 1. for i in range(n)},
                   lower=1.-len(target))


def original_step(bank, current, *, penalty, deadline, per_mip, heartbeat=None):
    started = perf_counter(); family = canonical(current['supports']); n = bank.data.n
    new = legal_start(family, n, bank.lp.pool, bank.forbidden)
    if new is None:
        return current, dict(kind='extension', accepted=False, structurally_complete=True)
    model = core.build_extension_model(bank.scaled, family, r_upper_bound=2., x_upper_bound=2.)
    activation = model.variables.add('activate_new', 1, upper=1., integral=True, objective=penalty*SCALE)
    for w in (model.new_r, model.new_x): model.rows.add({w.start: 1., activation.start: -2.}, upper=0.)
    for C in bank.forbidden: ban_membership(model, C, n)
    warm = current; checked = 0
    for C in bank.lp.pool:
        if C in family or C in bank.forbidden or not all(compatible(C, D) for D in family): continue
        if deadline-perf_counter() < per_mip+.1: break
        q = bank.fit([*family, C], preserve=True); checked += 1
        i = list(map(tuple, q['supports'])).index(C)
        if max(q['r'][i], q['x'][i]) <= 1e-9:
            q = bank.fit([D for D in q['supports'] if tuple(D) != C], preserve=True)
        if criterion(q, n, penalty) < criterion(warm, n, penalty)-1e-12: warm, new = q, C
    if deadline-perf_counter() <= .05:
        return current, dict(kind='extension', accepted=False, deadline=True)
    start = extension_start(model, family, new, warm, bank.lp.p, bank.lp.q, bank.lp.y)
    start[activation] = float(new in canonical(warm['supports']))
    values, event = solve_native(bank, model, start, deadline=min(deadline, perf_counter()+per_mip), heartbeat=heartbeat)
    best = warm
    if values[activation.start] > .5:
        C = core._round_and_validate_support(values[model.z], family, n)
        assert C not in bank.forbidden
        q = bank.fit([*family, C], preserve=True)
        i = list(map(tuple, q['supports'])).index(C)
        if max(q['r'][i], q['x'][i]) <= 1e-9:
            q = bank.fit([D for D in q['supports'] if tuple(D) != C], preserve=True)
        if criterion(q, n, penalty) < criterion(best, n, penalty): best = q
    before = criterion(current, n, penalty); score = criterion(best, n, penalty)
    gate = 1e-10+.001*current['objective']; accepted = before-score > gate
    event.update(kind='extension', accepted=accepted, before=before, score=score, constant=penalty*sum(1 < len(C) < n for C in family),
                 warm_checked=checked, threshold=gate, added=[C for C in best['supports'] if C not in family],
                 forbidden=list(bank.forbidden), stage_seconds=perf_counter()-started)
    return (best if accepted else current), event


def joint_tail(bank, current, *, penalty, deadline, heartbeat=None,hard=()):
    keep, released = open_small(current['supports'], bank.data.n)
    keep=canonical([*keep,*hard])
    released=canonical(C for C in released if C not in set(canonical(hard)))
    model, meta = build_joint(bank.scaled, keep)
    assert set(canonical(current['supports'])) <= set(meta['all_supports'])
    assert not set(meta['keep']) & bank.forbidden
    for k, C in enumerate(meta['candidates']):
        model.variables.objective[meta['selection'].start+k] = penalty*SCALE
        if C in bank.forbidden: model.variables.upper[meta['selection'].start+k] = 0.
    constant = penalty*sum(1 < len(C) < bank.data.n for C in meta['keep'])
    before = criterion(current, bank.data.n, penalty); point = current
    if meta['candidates'] and deadline-perf_counter() > .05:
        start = reconstruction_start(bank, model, meta, current)
        values, event = solve_native(bank, model, start, deadline=deadline, heartbeat=heartbeat)
        indices = [*range(len(meta['keep'])), *[len(meta['keep'])+k for k, C in enumerate(meta['candidates'])
                   if values[meta['selection'].start+k] > .5 and
                   max(values[model.r.start+len(meta['keep'])+k], values[model.x.start+len(meta['keep'])+k]) > 1e-9]]
        point = bank.fit([meta['all_supports'][i] for i in indices], preserve=True)
        assert criterion(point, bank.data.n, penalty) <= event['objective']+constant+1e-8
    else:
        event = dict(status=None, no_candidates=not meta['candidates'], solve_started=False)
    score = criterion(point, bank.data.n, penalty); gate = 1e-10+.001*current['objective']
    accepted = before-score > gate
    event.update(kind='tail', accepted=accepted, before=before, score=score, threshold=gate, constant=constant,
                 released=released, candidates=meta['candidates'], keep=meta['keep'], forbidden=list(bank.forbidden))
    return (point if accepted else current), event


def replay(bank, initial, *, penalty, deadline, per_mip=3600., heartbeat=None, checkpoint=None,zero_release=False,hard=()):
    current = initial; path = [current]; events = []; reason = 'step_limit';zero_events=[]
    def clean(stage):
        nonlocal current
        current,rows=release_zero_atoms(bank,current,hard=hard,enabled=zero_release,stage=stage)
        zero_events.extend(rows)
        if any(e['accepted'] for e in rows):path.append(current)
    clean('initial')
    for iteration in range(3*bank.data.n):
        if deadline-perf_counter() <= .1: reason = 'deadline'; break
        if remainder(current['supports'], bank.data.n) <= 5:
            current, e = joint_tail(bank, current, penalty=penalty, deadline=deadline, heartbeat=heartbeat,hard=hard)
            e['iteration'] = iteration; events.append(e)
            if e['accepted']: path.append(current)
            previous=len(zero_events);clean('tail_joint')
            if any(e['accepted'] for e in zero_events[previous:]):continue
            reason = 'tail_gap_met' if e.get('status') == 2 else 'tail_no_candidates' if e.get('no_candidates') else 'tail_unresolved'
            if checkpoint: checkpoint(dict(point=current, path=path, events=events, stop=reason))
            break
        current, e = original_step(bank, current, penalty=penalty, deadline=deadline, per_mip=per_mip, heartbeat=heartbeat)
        e['iteration'] = iteration; events.append(e)
        if e['accepted']:
            path.append(current);clean('extension')
        if checkpoint: checkpoint(dict(point=current, path=path, events=events))
        if not e['accepted']:
            reason = 'structurally_complete' if e.get('structurally_complete') else 'deadline' if e.get('deadline') else 'no_gain_unresolved'
            if e.get('bound') is not None and e['before']-(e['bound']+e['constant']) <= e['threshold']+1e-10:
                reason = 'bound_excludes_gain'
            break
    return dict(point=current, path=path, events=events, stop=reason, forbidden=list(bank.forbidden),zero_events=zero_events,zero_release=zero_release)
