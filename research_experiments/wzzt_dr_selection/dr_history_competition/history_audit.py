"""Bounded historical competition audit; observed P/Q/V only, no future simulator.

This is an audit of the complete five-terminal laminar model, not a price score.
Zero internal weights allow maximal binary families to represent unresolved
multifurcations and root forests. Every fixed family jointly refits all R/X
weights from zero with no finite upper bound. The actual finite-pool decisions,
when supplied, are replayed as fixed-family LPs under the same unrestricted
continuous domain and are not replaced by a new greedy algorithm.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from research_experiments.wzzt_dr_selection.dr_pool_v2 import engine
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays, normalize, input_hash
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    partial.replace(path)


@lru_cache(None)
def maximal_families(nodes):
    """All rooted binary hierarchies, with unordered root splits enumerated once."""
    nodes = tuple(sorted(nodes))
    if len(nodes) == 1:
        return ((nodes,),)
    first, rest = nodes[0], nodes[1:]
    found = set()
    # The left child contains the smallest leaf; the right child is nonempty.
    for size in range(len(rest)):
        for extra in combinations(rest, size):
            left = (first, *extra)
            right = tuple(i for i in rest if i not in extra)
            for a in maximal_families(tuple(left)):
                for b in maximal_families(right):
                    found.add(engine.canonical([nodes, *a, *b]))
    return tuple(sorted(found))


def all_supports(n):
    return engine.canonical(s for size in range(1, n + 1) for s in combinations(range(n), size))


def combinatorial_check():
    counts = {n: len(maximal_families(tuple(range(n)))) for n in range(1, 6)}
    if counts != {1: 1, 2: 1, 3: 3, 4: 15, 5: 105}:
        raise RuntimeError('unexpected rooted binary hierarchy counts')
    nodes = tuple(range(4)); full = maximal_families(nodes)
    optional = [s for s in all_supports(4) if 1 < len(s) < 4]
    required = [(i,) for i in nodes] + [nodes]
    laminar = covered = 0
    # Exhaust all 2^10 possible internal-subset choices for four terminals.
    for bits in product((0, 1), repeat=len(optional)):
        family = engine.canonical([*required, *(s for s, bit in zip(optional, bits) if bit)])
        if any(not engine.compatible(family[:i], s) for i, s in enumerate(family)):
            continue
        laminar += 1
        if any(set(family) <= set(binary) for binary in full): covered += 1
    if covered != laminar:
        raise RuntimeError('a four-terminal nonbinary hierarchy lacks a binary refinement')
    return dict(binary_family_counts=counts, four_terminal_laminar_families=laminar,
                four_terminal_covered=covered, all_checks_pass=True,
                five_terminal_coverage='Every finite laminar hierarchy admits a binary refinement; assigning zero to new internal supports preserves its matrix. All 105 binary refinements are enumerated.')


def load_history(path):
    data = read(path)
    # Explicit schema rejects simulator truth, future events, and test receipts.
    if set(data) != {'terminal_labels', 'history_days'}:
        raise ValueError('history input must contain only terminal_labels and history_days')
    labels = data['terminal_labels']
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError('bounded audit is explicitly limited to five distinct terminals')
    scenes = []
    for day in data['history_days']:
        if set(day) != {'date', 'p_pu', 'q_pu', 'v_pu', 'root_pu'}:
            raise ValueError('historical day fields must be observed date, P/Q/V and root V only')
        idx = pd.date_range(day['date'], periods=len(day['p_pu']), freq='h')
        frames = [pd.DataFrame(day[key], index=idx, columns=labels) for key in ('p_pu', 'q_pu', 'v_pu')]
        root = pd.Series(day['root_pu'], index=idx)
        scenes.append(make_observed_scenario(*frames, root, str(day['date'])))
    obs, days, counts = normalize(scenes, 'history')
    return labels, scenes, obs, days, counts


def witness(obs, tolerance):
    p, q, y = arrays(obs, 'raw')
    _, singular, vt = np.linalg.svd(np.column_stack((p, q)), full_matrices=True)
    rank_p = int(np.linalg.matrix_rank(p)); rank_pq = int(np.linalg.matrix_rank(np.column_stack((p, q))))
    base = p[:, 0]
    power_sum = float(np.sum(p * p))
    kappa = float(np.sum(p * q) / power_sum) if power_sum else None
    nonzero = bool(np.max(np.abs(base)) > tolerance)
    equal_p = bool(np.max(np.abs(p - base[:, None])) <= tolerance)
    common_q = bool(kappa is not None and np.max(np.abs(q - kappa * p)) <= tolerance)
    h = np.linalg.lstsq(base[:, None], y, rcond=None)[0][0] if nonzero else np.zeros(p.shape[1])
    residual = float(np.max(np.abs(base[:, None] * h - y)))
    applicable = equal_p and common_q and nonzero and kappa >= 0 and np.min(h) >= -tolerance and residual <= tolerance
    result = dict(input_rank_P=rank_p, input_rank_PQ=rank_pq,
                  input_singular_values_PQ=singular.tolist(),
                  input_nullspace_PQ=vt[rank_pq:].tolist(),
                  terminal_P_identical=equal_p, common_Q_over_P=kappa,
                  common_Q_over_P_pass=common_q, inferred_combined_row_sums=h.tolist(),
                  row_sum_max_residual=residual, analytic_singleton_witness_applicable=applicable)
    if applicable:
        r = np.diag(np.maximum(h, 0) / (1 + kappa)); x = r.copy()
        werr = float(np.mean(np.abs(p @ r.T + q @ x.T - y)))
        result.update(singleton_witness=dict(R=r.tolist(), X=x.tolist(),
                      all_nonleaf_coefficients=0.0, objective=werr,
                      strictly_positive_diagonal=bool(np.min(np.diag(r)) > 0),
                      fitted_from='historical P/Q/Y only',
                      parameter_role='one explicit compatible model, not a representative of every optimal face'),
                      uniform_strict_acceptance_upper_bound=0.0,
                      uniform_strict_acceptance_reason='Every enumerated family contains all singletons and admits this same zero-internal-weight model. Under any future inputs generated by this fixed model, singleton fitting remains exact. Nonnegative loss cannot have a strictly positive decrease from zero.',
                      limitation='This rejects a guaranteed positive accepting step over all compatible models. It does not say DR is useless or every compatible model absorbs every DR response.')
    return result


def full_point(lp, family):
    point = lp.fit(family)
    k = len(lp.pool)
    values = np.maximum(np.asarray(lp.weights.X), 0.0)
    # This is called only for a family not previously fitted in this audit.
    active = [lp.lookup[s] for s in engine.canonical(family)]
    return dict(native_family=engine.canonical(family), native_r=values[active].tolist(),
                native_x=values[np.asarray(active) + k].tolist(), point=point,
                role='one LP optimizer point; omitted zero coefficients remain explicitly present in native_r/native_x')


def positive_completion(family, analysis, obs):
    """Existence witness only; never a parameter bound or design prior."""
    if not analysis['analytic_singleton_witness_applicable']:
        return None
    h=np.asarray(analysis['inferred_combined_row_sums'])
    if np.min(h)<=0:
        return None
    internal=[s for s in family if len(s)>1]
    load=np.asarray([sum(len(s) for s in internal if i in s) for i in range(len(h))])
    a=float(np.min(h/(2*load)))
    beta=np.asarray([h[s[0]]-a*load[s[0]] if len(s)==1 else a for s in family])
    coeff=beta/(1+analysis['common_Q_over_P'])
    rr=np.zeros((len(h),len(h)));xx=rr.copy()
    for s,w in zip(family,coeff):
        rr[np.ix_(s,s)]+=w;xx[np.ix_(s,s)]+=w
    p,q,y=arrays(obs,'raw')
    error=float(np.max(np.abs(p@rr.T+q@xx.T-y)))
    if error>1e-8 or np.min(beta)<=0:
        raise RuntimeError('positive completion failed independent arithmetic')
    return dict(native_family=family,combined_weights=beta.tolist(),r=coeff.tolist(),x=coeff.tolist(),
                internal_combined_weight=a,max_residual=error,
                role='one explicit strictly positive completion; not a gain/weight lower bound, model probability, or price prior')


def scan_actual_fit(path, lp, tolerance, max_extra, expected_training_sha, labels):
    source = read(path)
    if source.get('input_sha256', {}).get('training') != expected_training_sha:
        raise ValueError('saved history fit training hash differs from the sanitized historical observations')
    if source.get('terminal_labels') != labels:
        raise ValueError('saved history fit terminal order differs from observed history')
    out = dict(source_path=str(path), source_sha256=sha(path), stage=source.get('stage'),
               status=source.get('status'), bounded_coefficient_domain=source.get('settings', {}).get('coefficient_bound'),
               scans=[], skipped_due_to_budget=[], refit_failures=[])
    extra_start = len(lp.solves)
    for name, trajectory in source.get('searches', {}).items():
        points = trajectory['path']
        for decision in trajectory.get('decisions', []):
            index = int(decision['iteration'])
            current = points[index]
            family = engine.canonical(current['supports'])
            if family not in lp.cache and len(lp.solves) - extra_start >= max_extra:
                out['skipped_due_to_budget'].append(dict(path=name, iteration=index, current_family=family))
                continue
            try:
                base = lp.fit(family)
            except Exception as exc:
                out['refit_failures'].append(dict(path=name, iteration=index, family=family, exception=repr(exc)))
                continue
            row = dict(path=name, iteration=index, current_family=family,
                       saved_current_objective=decision['current'],
                       unbounded_current_objective=base['objective'],
                       full_sweep=decision['full_sweep'], candidates=[])
            for candidate in decision['candidates']:
                block = tuple(candidate['candidate'])
                retained = [s for s in family if engine.compatible([s], block)]
                removed = [s for s in family if s not in retained]
                proposed = engine.canonical([*retained, block])
                if proposed not in lp.cache and len(lp.solves) - extra_start >= max_extra:
                    out['skipped_due_to_budget'].append(dict(path=name, iteration=index, candidate=block))
                    continue
                try:
                    fit = lp.fit(proposed)
                except Exception as exc:
                    out['refit_failures'].append(dict(path=name, iteration=index, candidate=block, exception=repr(exc)))
                    continue
                gain = base['objective'] - fit['objective']
                threshold = 1e-10 + .001 * base['objective']
                row['candidates'].append(dict(candidate=block, removed=removed,
                    saved_removed=candidate['removed'], crossing_receipt_match=set(removed) == set(map(tuple, candidate['removed'])),
                    jointly_refitted_family=proposed, saved_objective=candidate['objective'],
                    unbounded_objective=fit['objective'], gain=gain, actual_acceptance_threshold=threshold,
                    strictly_acceptable=bool(gain > threshold),
                    bounded_unbounded_objective_match=bool(abs(candidate['objective'] - fit['objective']) <= tolerance)))
            out['scans'].append(row)
    out.update(extra_LP_records=len(lp.solves) - extra_start,
               all_candidates_audited=not out['skipped_due_to_budget'] and not out['refit_failures'],
               acceptable_candidates=sum(c['strictly_acceptable'] for row in out['scans'] for c in row['candidates']),
               crossing_candidates=sum(bool(c['removed']) for row in out['scans'] for c in row['candidates']))
    return out


def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sources = {str(Path(__file__).resolve()): sha(__file__), str(Path(engine.__file__).resolve()): sha(engine.__file__)}
    identity = dict(history_sha256=sha(args.observed_history), source_sha256=sources,
                    history_fit_sha256=sha(args.history_fit) if args.history_fit else None,
                    zero_loss_tolerance=args.tolerance, max_extra_LP_records=args.max_extra_solves)
    final = output / 'history_audit.json'
    if final.exists():
        old = read(final)
        if args.resume and old.get('identity') == identity and old.get('status') == 'complete':
            return old
        raise FileExistsError('existing audit; only an unchanged completed audit may resume')
    checkpoint = output / 'history_audit_checkpoint.json'
    if checkpoint.exists(): raise FileExistsError('unfinished audit checkpoint; no implicit restart')
    lock = output / 'history_audit.lock'
    with lock.open('x', encoding='utf-8') as stream: json.dump(dict(pid=os.getpid()), stream)
    started = perf_counter(); lp = None
    result = dict(status='running', identity=identity, scope='observed historical P/Q/V only',
                  truth_input_allowed=False, future_input_allowed=False, families=[], failures=[],
                  environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                   numpy=np.__version__, pandas=pd.__version__))
    try:
        result['combinatorial_check'] = combinatorial_check()
        labels, scenes, obs, days, counts = load_history(args.observed_history)
        result.update(terminal_labels=labels, history_day_counts=counts, historical_rows=len(obs[0]),
                      normalized_input_sha256=input_hash(obs, days),
                      settings=dict(nonnegative_weights=True, continuous_upper_bound='unbounded',
                                    fixed_early_blocks=False, maximal_families=105, candidate_supports=31,
                                    LP_time_limit_seconds=15, no_price_objective=True,
                                    response_model_and_budget='unchanged; this audit consumes no physical DR budget'))
        result['history_input_analysis'] = witness(obs, args.tolerance)
        write(checkpoint, result)
        lp = engine.PoolLP(obs, all_supports(5), 'raw', bound=float('inf'))
        result['environment']['gurobi'] = list(lp.gp.gurobi.version())
        for serial, family in enumerate(maximal_families(tuple(range(5)))):
            try:
                point = full_point(lp, family)
                point.update(index=serial, family_labels=[[labels[i] for i in s] for s in family])
                point['positive_completion_witness']=positive_completion(family,result['history_input_analysis'],obs)
                result['families'].append(point)
            except Exception as exc:
                result['failures'].append(dict(index=serial, family=family, exception_type=type(exc).__name__, exception=str(exc)))
            result['LP_solves'] = lp.solves
            result['elapsed_seconds'] = perf_counter() - started
            write(checkpoint, result)
        if args.history_fit:
            result['actual_history_search'] = scan_actual_fit(args.history_fit, lp, args.tolerance, args.max_extra_solves,
                                                             input_hash(obs, days), labels)
        objectives = [f['point']['objective'] for f in result['families']]
        result.update(status='complete' if not result['failures'] else 'completed_with_failures',
                      planned_families=105, completed_families=len(result['families']),
                      zero_loss_families=sum(v <= args.tolerance for v in objectives),
                      strictly_positive_completion_families=sum(f['positive_completion_witness'] is not None for f in result['families']),
                      min_family_objective=min(objectives) if objectives else None,
                      max_family_objective=max(objectives) if objectives else None,
                      LP_solves=lp.solves, elapsed_seconds=perf_counter() - started,
                      mathematical_conclusion='An explicit common singleton witness, when applicable, prevents a uniformly positive WZZT accepting step over all history-compatible models. Future information value and conditional model-dependent improvements are not ruled out.',
                      next_action='Do not assign probabilities to LP representatives. Keep randomized exploration separate from a claimed history-supported unique price target.')
        write(final, result); write(checkpoint, result)
        return result
    except Exception as exc:
        result.update(status='failed', exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc(),
                      LP_solves=lp.solves if lp else [], elapsed_seconds=perf_counter() - started)
        write(checkpoint, result)
        raise
    finally:
        if lp: lp.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed-history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history-fit', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--max-extra-solves', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    result = run(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in ('status', 'planned_families', 'completed_families', 'zero_loss_families', 'min_family_objective', 'max_family_objective', 'elapsed_seconds')}))
