"""One bounded mathematical gate and existing-summary coverage audit.

No observations/household data, topology fits, physical simulation or new noise
are read/generated. Synthetic decision tables verify computation only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import unittest
import numpy as np
import scipy
from . import joint_response_risk as j
from . import test_joint_response_risk as tests


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def coverage(repo):
    evidence = repo/'research_experiments/wzzt_dr_selection/evidence/flynn_local_phase_pilot_v1'
    plans = json.loads((evidence/'plans.json').read_text(encoding='utf-8'))['plans']
    aggregate = json.loads((evidence/'aggregate_summary.json').read_text(encoding='utf-8'))
    records = aggregate['policy_cost_and_errors']
    by_policy = {r['policy']: r for r in records}
    rows, query_union, per_menu_queries = [], set(), {}
    for name, plan in plans.items():
        p = np.asarray(plan, dtype=int)
        if p.shape != (2, 16) or not np.isin(p, [0, 1, 2]).all() or p.sum() > 4:
            raise ValueError('saved pilot contract differs from frozen scope')
        count = 2*int((p > 0).sum())
        this_menu = set()
        for event in range(4):
            for node, price in enumerate(p[event % 2]):
                if price:
                    query_union.add((event, node, int(price)))
                    this_menu.add((event, node, int(price)))
        per_menu_queries[name] = this_menu
        rows.append(dict(menu=name, node_event_participations=count,
                         reachable_states_at_fixed_background=2**count,
                         saved_realized_cells=1 if name in by_policy else 0,
                         worst_payment=float(.4*p.sum())))
    adapter = repo/'research_experiments/wzzt_dr_selection/dr_pool_v2'
    protocol = json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    matches = {}
    for name in ('adapter.py', 'engine.py', 'selection.py', 'observations.py'):
        path = adapter/name
        source = [value for key, value in protocol['source_sha256'].items()
                  if key.replace('\\', '/').endswith('/dr_pool_v2/'+name)]
        matches[name] = len(source) == 1 and digest(path) == source[0]
    if not all(matches.values()):
        raise ValueError('pilot and current joint-identifier source hashes differ')
    return dict(scope='already_seen_pilot_coverage_only_not_planning_or_new_evaluation',
                menus=rows, menu_state_cells=sum(r['reachable_states_at_fixed_background'] for r in rows),
                saved_cells=sum(r['saved_realized_cells'] for r in rows),
                natural_measurement_paths=1, realized_generation_models=1,
                usable_complete_conditional_planning_risk_tables=0,
                source_hash_matches=matches,
                shared_node_event_price_queries=len(query_union),
                dense_common_type_atoms=2**len(query_union),
                dense_type_gate=65536,
                max_pairwise_queries=max(len(a | b) for a in per_menu_queries.values() for b in per_menu_queries.values()),
                max_pairwise_type_atoms=2**max(len(a | b) for a in per_menu_queries.values() for b in per_menu_queries.values()),
                pair_projection_scope='exact for marginal-only response ambiguity, not arbitrary joint-moment constraints',
                may_expand_old_zero_error_case=False,
                interpretation='State cells are not independent samples. A realized error is not an expected risk.')


def run(protocol_path, output):
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    repo = Path(protocol['repository'])
    for relative, expected in protocol['source_sha256'].items():
        if digest(repo/relative) != expected:
            raise ValueError('source/input hash changed after protocol freeze: '+relative)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        raise FileExistsError('completed gate must not be silently repeated')
    lock = output/'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), protocol_sha256=digest(protocol_path)), stream)
    result = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=digest(protocol_path),
                  environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                   numpy=np.__version__, scipy=scipy.__version__),
                  new_identifier_calls=0, new_fixed_family_lp=0, new_ac_timepoints=0,
                  new_future_or_household_trajectories=0, new_DR_recovery_claim=False)
    instances = []
    started = perf_counter()
    original = j.ExpectationLP

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def maximize(self, values):
            if sum(len(x.receipts) for x in instances) >= protocol['max_probability_lp']:
                raise RuntimeError('whole-gate probability LP limit exhausted')
            if perf_counter()-started >= protocol['wall_seconds']:
                raise RuntimeError('whole-gate wall limit exhausted before next LP')
            return super().maximize(values)

    j.ExpectationLP = Tracked
    try:
        buffer = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        test_result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
        (output/'unit_tests.txt').write_text(buffer.getvalue(), encoding='utf-8')
        result['unit_tests'] = dict(run=test_result.testsRun, failures=len(test_result.failures),
                                    errors=len(test_result.errors), successful=test_result.wasSuccessful())
        if not test_result.wasSuccessful():
            raise RuntimeError('mathematical/negative-case unit checks failed')
        domain = j.binary_domain(('participantA:p1', 'participantB:p1'), ((.5, .5), (.5, .5)))
        losses = np.array([[[1., 0, 0, 0], [0., 1, 1, 0]]])
        table = j.evaluate_regret(Tracked(domain), losses, losses)
        selected = j.select_menu(table, [0., .8], seed=protocol['tie_seed'])
        result['analytical_table'] = dict(source='hand_constructed_decision_losses_not_WZZT',
                                           risk_table=table, selection=selected)
        result['existing_cache_coverage'] = coverage(repo)
        result['status'] = 'complete_computational_gate_only'
        result['scope'] = 'finite decision-table arithmetic; no actual RNJ+WZZT price-effectiveness test'
        result['next_boundary'] = 'No complete history-conditioned joint state loss table exists; do not refill closed flynn pilot.'
    except Exception as exc:
        result.update(status='failed', exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
    finally:
        j.ExpectationLP = original
        receipts = [receipt for instance in instances for receipt in instance.receipts]
        result['probability_lp'] = dict(total=len(receipts), successful=sum(r['success'] for r in receipts),
                                        intentionally_infeasible=sum(r['status'] == 2 for r in receipts),
                                        other_unresolved=sum(r['status'] not in (0, 2) for r in receipts), receipts=receipts)
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        result['elapsed_seconds'] = perf_counter()-started
        write(output/'result.json', result)
        lock.unlink(missing_ok=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.output)
    print(json.dumps(dict(status=result['status'], tests=result.get('unit_tests'),
                          probability_lp=result['probability_lp']['total']), ensure_ascii=True))
    raise SystemExit(0 if result['status'].startswith('complete') else 1)
