"""Explicit candidate responses, binary laminar selection, one coupled L1 fit."""
from __future__ import annotations

from itertools import combinations
from time import perf_counter
import numpy as np

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical, compatible, unions
from research_experiments.rebuild_benchmark.engine import hierarchy_children, solve, reconstruction_start


def remainder(family, n):
    return max((len(c) for c in hierarchy_children(family, n).values()), default=1)


def open_small(family, n, *, limit=5):
    """Release provisional <=5-terminal merges while keeping every new arity <=5.

    An already larger polytomy is left intact for the original WZZT phase.
    Parent boundaries >5 terminals are preserved, as are essential small cuts.
    """
    keep = canonical([*family, *((i,) for i in range(n)), tuple(range(n))])
    released = []
    for support in tuple(keep):
        if not 1 < len(support) <= limit or len(support) == n:
            continue
        children = hierarchy_children(keep, n)
        parent = next(v for v, blocks in children.items() if support in blocks)
        expanded = len(children[parent])-1+len(children[support])
        if expanded <= limit:
            keep = canonical(s for s in keep if s != support); released.append(support)
    return keep, canonical(released)


def build_joint(prepared, keep, *, limit=5):
    if limit != 5:
        raise ValueError('this frozen experiment uses exactly five blocks')
    keep = canonical([*keep, *((i,) for i in range(prepared.n)), tuple(range(prepared.n))])
    if not core.is_laminar_family(keep):
        raise ValueError('coarse skeleton must be laminar')
    local = []
    for parent, blocks in hierarchy_children(keep, prepared.n).items():
        if 3 <= len(blocks) <= limit:
            local.append((parent, blocks, unions(blocks)))
    candidates = canonical(s for _, _, pool in local for _, s in pool)
    if any(s in keep or any(not compatible(s, t) for t in keep) for s in candidates):
        raise RuntimeError('invalid residual candidate')
    all_supports = (*keep, *candidates)
    model = core.build_fixed_model(prepared, all_supports, r_upper_bound=2., x_upper_bound=2.)
    selection = model.variables.add('select', len(candidates), upper=1., integral=True)
    lookup = {s: selection.start+k for k, s in enumerate(candidates)}
    for k, s in enumerate(candidates):
        for weights in (model.r, model.x):
            model.rows.add({weights.start+len(keep)+k: 1., lookup[s]: -2.}, upper=0.)
    conflicts = 0; cuts = 0
    for parent, blocks, pool in local:
        for (_, s), (_, t) in combinations(pool, 2):
            if not compatible(s, t):
                model.rows.add({lookup[s]: 1., lookup[t]: 1.}, upper=1.); conflicts += 1
        for size in range(3, len(blocks)+1):
            for w in combinations(range(len(blocks)), size):
                entries = {lookup[s]: 1. for indices, s in pool if set(indices) < set(w)}
                if entries:
                    model.rows.add(entries, upper=float(size-2)); cuts += 1
    return model, dict(keep=keep, candidates=candidates, all_supports=all_supports, selection=selection,
        regions=[dict(parent=v, blocks=b, candidates=len(p)) for v, b, p in local],
        conflicts=conflicts, cardinality_cuts=cuts, maximum_remainder=remainder(keep, prepared.n),
        skipped_large_regions=[dict(parent=v, blocks=b) for v, b in hierarchy_children(keep, prepared.n).items() if len(b)>limit])


def joint_fit(bank, current, keep, *, seconds=5., relative=.001, stage='joint'):
    then = perf_counter(); model, meta = build_joint(bank.scaled, keep)
    if not set(canonical(current['supports'])) <= set(meta['all_supports']):
        raise ValueError('current tree must be recoverable in the enumerated domain')
    before = current['objective']; point = current
    record = dict(kind=stage, solve_started=False, accepted=False, before=before,
                  candidate_count=len(meta['candidates']), conflicts=meta['conflicts'],
                  cardinality_cuts=meta['cardinality_cuts'], regions=meta['regions'],
                  keep=meta['keep'], candidates=meta['candidates'],
                  skipped_large_regions=meta['skipped_large_regions'],
                  maximum_remainder=meta['maximum_remainder'],
                  scope='one coupled solve over all eligible residual regions')
    remaining = seconds-(perf_counter()-then)
    if meta['candidates'] and remaining > .05:
        start = reconstruction_start(bank, model, meta, current)
        values, event = solve(bank, model, start, seconds=remaining)
        record.update(event, solve_started=True)
        if values is not None:
            active = [*range(len(meta['keep'])), *[len(meta['keep'])+k for k, a in enumerate(values[meta['selection']]) if a>.5]]
            r, x = values[model.r], values[model.x]
            active = [k for k in active if k < len(meta['keep']) or max(r[k], x[k])>1e-9]
            family = [meta['all_supports'][k] for k in active]
            point = bank.fit(family, preserve=True)
            if point['objective'] > event['objective']+1e-8:
                raise RuntimeError('joint incumbent physical refit exceeds its feasible objective')
    elif not meta['candidates']:
        point = bank.fit(meta['keep'], preserve=True)
        record['no_candidates'] = True
    gain = before-point['objective']; gate = 1e-10+relative*before
    record.update(decoded_objective=point['objective'], gain=gain, threshold=gate, accepted=gain>gate,
                  wall_seconds=perf_counter()-then,
                  selected_new=[s for s in point['supports'] if s not in meta['keep']],
                  complete_model=not meta['skipped_large_regions'])
    return (point if gain>gate else current), record
