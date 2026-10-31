"""Actual bounded budget-reference diagnostic using existing physical/fit APIs.

All inputs and output trajectories remain in the caller's private run root.
No topology identifier, physical model, or support search is implemented here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
from random import Random
import subprocess
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def plain(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _reference(record):
    if isinstance(record, str):
        raise ValueError('input path must have an explicit SHA-256 binding')
    path = Path(record['path'])
    expected = record.get('sha256', record.get('catalog_sha256'))
    if not expected or sha(path) != expected:
        raise ValueError('frozen input hash mismatch: '+str(path))
    return path


def configuration(root):
    root = Path(root).resolve()
    cfg = read(root/'protocol.json')
    repo = Path(cfg['repository']).resolve()
    for path, expected in cfg['source_sha256'].items():
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = repo/candidate
        if sha(candidate) != expected:
            raise RuntimeError('source drift: '+str(candidate))
    return root, cfg, repo


def _setup(repo):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(repo))


def _history(case):
    with _reference(case['history_packet']).open('rb') as stream:
        packet = pickle.load(stream)
    history = packet.get('history', packet)
    for role in ('training', 'validation'):
        if sum(len(s['P_terminal']) for s in history[role]) != 288:
            raise ValueError('full 288/288 history is required')
    return history


def _family(point, labels):
    supports = point.get('support_labels')
    if supports is None:
        supports = [[labels[i] for i in s] for s in point.get('support_indices', point.get('supports', []))]
    family = {tuple(sorted(int(v) for v in s)) for s in supports}
    if any(not s or not set(s) <= set(labels) for s in family):
        raise ValueError('predicted support lies outside terminal labels')
    return family


def _error(family, truth):
    missing, extra = truth-family, family-truth
    return dict(full_errors=len(missing)+len(extra), full_recovery=not missing and not extra,
                missing=[list(s) for s in sorted(missing)], extra=[list(s) for s in sorted(extra)],
                predicted_blocks=len(family), truth_blocks=len(truth),
                prediction_definition='all final support labels, no R/X threshold',
                truth_definition='physical evaluator: exact-zero edges inactive, every positive weak edge retained')


def score_fit(fit, truth_labels, root_label):
    """Apply geometric NJ/RNJ and saved joint outputs to the same observations.

    NJ calls the existing implementation on the already fitted raw front. RNJ
    uses the already recorded fixed RX75/.16 native tree. Neither repeats QP.
    """
    import numpy as np
    from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades

    labels = list(fit['terminal_labels'])
    truth = {tuple(sorted(s)) for s in truth_labels}
    models = {}
    for name in ('multistart_original_rule', 'rnj_bank', 'baseline25'):
        point = fit['models'][name]
        value = _error(_family(point, labels), truth)
        before = point.get('selected_uncontracted')
        value.update(selected_origin=point.get('selected_origin', point.get('origin')),
                     train_mae=point.get('train_mae', point.get('objective')),
                     validation_mae=point.get('validation_mae'),
                     before_contraction_errors=_error(_family(before, labels), truth)['full_errors'] if before else None)
        models[name] = value
    reference = [point for point in fit['bank'] if point.get('config') == 'raw_RX75_0.16']
    if len(reference) != 1:
        raise ValueError('fixed geometric RNJ configuration unavailable')
    rnj_family = {tuple(sorted(labels[i] for i in s)) for s in reference[0]['native_family']}
    models['geometric_RNJ_RX75_0.16'] = _error(rnj_family, truth)
    raw, cal = fit['fronts']['raw'], fit['calibrations']['raw']
    r, x = np.asarray(raw['R']), np.asarray(raw['X'])
    kernel = .75*r/float(cal['mu_r'])+.25*x/float(cal['mu_x'])
    tree = infer_classical_nj(kernel, kernel, labels, root_label, collapse_factor=.16)
    clades = rooted_clades(tree.edges, root_label, labels)
    nj_family = {(label,) for label in labels} | {tuple(sorted(s)) for s in clades}
    models['geometric_NJ_RX75_0.16'] = _error(nj_family, truth)
    return models


def search_audit(fit):
    accepts = crosses = removed = 0
    for search in fit.get('searches', {}).values():
        accepts += max(0, len(search['path'])-1)
        for step in range(max(0, len(search['path'])-1)):
            best = None
            for candidate in search['decisions'][step]['candidates']:
                if best is None or candidate['objective'] < best['objective']-1e-12:
                    best = candidate
            if best and best.get('removed'):
                crosses += 1
                removed += len(best['removed'])
    return dict(accepted_updates=accepts, accepted_cross_updates=crosses, removed_blocks=removed,
                search_stops=fit.get('search_stop_statuses', {}),
                wall_limited=any(s == 'pool_wall_limit' for s in fit.get('search_stop_statuses', {}).values()))


def lp_ledger(folder):
    counts, unknown = {}, []
    for cache in sorted(Path(folder).glob('*')):
        if not cache.is_dir():
            continue
        path = cache/'result.json'
        if not path.exists():
            path = cache/'checkpoint.json'
        if path.exists() and 'LP_solves' in (item := read(path)):
            counts[cache.name] = len(item['LP_solves'])
        else:
            unknown.append(cache.name)
    return dict(total=sum(counts.values()), by_key=counts, unknown_cache_keys=unknown)


def _oracle_guard(root, cfg):
    path = root/'oracle_selected.json'
    receipt = read(root/'oracle_selected.sha256.json')
    oracle = read(path)
    if sha(path) != receipt['sha256'] or oracle['status'] != 'frozen':
        raise RuntimeError('global oracle selection must be frozen before test data')
    if oracle['protocol_sha256'] != sha(root/'protocol.json'):
        raise RuntimeError('oracle was selected under a different protocol')
    for key, expected in oracle['planning_phase_sha256'].items():
        if sha(root/key/'planning'/'phase_ended.json') != expected:
            raise RuntimeError('planning results changed after global selection')
    return oracle


def run_phase(root, case_key, phase):
    root, cfg, repo = configuration(root)
    _setup(repo)
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter, BatchBoundary

    case = cfg['cases'][case_key]
    history = _history(case)
    labels = list(history.get('labels', history['training'][0]['P_terminal'].columns))
    root_label = int(case.get('root_label', 1))
    plan_path = _reference(case['plan_lock'])
    plan_sha = sha(plan_path)
    oracle = _oracle_guard(root, cfg) if phase == 'test' else None
    folder = root/case_key/phase
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'phase_ended.json').exists():
        raise FileExistsError('completed phase is retained, not rerun')
    with (folder/'phase.lock').open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), protocol_sha256=sha(root/'protocol.json'), started_utc=now()), stream)
    case_root = root/case_key
    cache_root = case_root/'fit_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    existing_attempts = sum(p.is_dir() for p in cache_root.iterdir())
    allowed = int(cfg.get('max_new_fits_per_case', 33))
    expected_algorithm = cfg['algorithm_source_sha256']
    if source_hashes() != expected_algorithm:
        raise RuntimeError('loaded algorithm source closure differs from frozen baseline/protocol')
    fitter = SharedFitter(cache_root, sources=expected_algorithm, max_new_fits=max(0, allowed-existing_attempts))
    paths = list(cfg['planning_paths'] if phase == 'planning' else cfg['test_paths'])
    policies = list(cfg['policies'])
    expected_ids = [f'{policy}__path{path}' for path in paths for policy in policies]
    progress = folder/'progress.json'
    rows = read(progress)['rows'] if progress.exists() else []
    recorded = {r['record_id'] for r in rows}
    started = time.perf_counter()
    stop_gate = False
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo)
    task = repo/'research_experiments/wzzt_dr_selection'
    wall = float(cfg.get('phase_wall_seconds', 3000))
    lp_gate = int(cfg.get('LP_stop_gate', 16000))

    def save():
        write(progress, dict(case_key=case_key, phase=phase, planned=len(expected_ids), rows=rows,
                             LP_ledger=lp_ledger(cache_root), elapsed_seconds=time.perf_counter()-started))

    try:
        baseline_dir = case_root/'baseline'
        baseline_result = baseline_dir/'result.json'
        if phase == 'planning':
            configuration(root)
            if source_hashes() != expected_algorithm:
                raise RuntimeError('loaded algorithm changed before baseline')
            baseline_dir.mkdir(exist_ok=True)
            try:
                fit = fitter(history['training'], history['validation'], baseline_dir,
                             root_label=root_label, search_seconds=float(cfg.get('per_path_seconds', 25)))
                write(case_root/'baseline_attempt.json', dict(status=fit['status'], result_sha256=sha(baseline_result)))
            except Exception as error:
                write(case_root/'baseline_attempt.json', dict(status='failed', type=type(error).__name__,
                                                             message=str(error), traceback=traceback.format_exc()))
        elif phase == 'test' and not baseline_result.exists():
            # A failed baseline stays a failure; it is never rerun in test.
            write(case_root/'baseline_missing_at_test.json', dict(status='unresolved', recorded_utc=now()))

        for path in paths:
            for policy in policies:
                record_id = f'{policy}__path{path}'
                if record_id in recorded:
                    continue
                row = dict(record_id=record_id, case_key=case_key, phase=phase, policy=policy,
                           path=int(path), status='started', actual_payment=None, committed_payment=None,
                           truth_read_before_fit=False, models=None, started_utc=now())
                rows.append(row); recorded.add(record_id)
                ledger = lp_ledger(cache_root)
                if stop_gate or time.perf_counter()-started >= wall or ledger['total'] >= lp_gate or ledger['unknown_cache_keys']:
                    row.update(status='unresolved_batch_boundary', LP_ledger_at_boundary=ledger)
                    save(); continue
                location = folder/'records'/record_id
                location.mkdir(parents=True, exist_ok=True)
                physical = location/'physical'
                receipt_path = physical/'events_receipt.json'
                try:
                    configuration(root)
                    if not receipt_path.exists():
                        if physical.exists():
                            raise RuntimeError('partial physical attempt retained; no automatic retry')
                        command = [sys.executable, '-B', str(task/'dr_coded_selection/actual_worker.py'),
                                   '--plan-lock', str(plan_path), '--expected-lock-sha', plan_sha,
                                   '--policy', policy, '--path', str(path),
                                   '--dependence', cfg.get('dependence', 'local'),
                                   '--sigma', str(cfg.get('voltage_sigma', .0002)), '--case', case['case'],
                                   '--output', str(physical), '--repo-root', str(repo),
                                   '--source-root', str(cfg['physical_source_root']),
                                   '--replicate-base', str(case.get('replicate_base', cfg.get('replicate_base', 88000))),
                                   '--participation-seed', str(case.get('participation_seed', cfg.get('participation_seed', 9583101))),
                                   '--event-profiles', *map(str, cfg.get('event_profiles', [0, 2, 0, 2]))]
                        proc = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True,
                                              encoding='utf-8', errors='replace',
                                              timeout=float(cfg.get('worker_timeout_seconds', 120)))
                        physical.mkdir(exist_ok=True)
                        (location/'worker_stdout.txt').write_text(proc.stdout, encoding='utf-8')
                        (location/'worker_stderr.txt').write_text(proc.stderr, encoding='utf-8')
                        row['worker_return_code'] = proc.returncode
                    receipt = read(receipt_path)
                    row.update(physical_status=receipt['status'], physical_receipt_path=str(receipt_path),
                               physical_receipt_sha256=sha(receipt_path),
                               actual_payment=receipt.get('actual_payment'), committed_payment=receipt.get('committed_payment'),
                               unknown_payment_events=receipt.get('unknown_payment_events'),
                               operation_checks_pass=receipt.get('all_operation_checks_pass') is True and receipt.get('quote_qualification_pass') is True,
                               quote_qualification_pass=receipt.get('quote_qualification_pass'),
                               physical_events_complete=receipt.get('physical_events_complete'))
                    if receipt['status'] != 'complete' or row.get('worker_return_code', 0) != 0:
                        row['status'] = 'physical_failed_or_unresolved'; continue
                    packet_path = physical/'packet.pkl'
                    if sha(packet_path) != receipt['private_files']['packet.pkl']['sha256']:
                        raise RuntimeError('observed packet hash mismatch')
                    with packet_path.open('rb') as stream:
                        packet = pickle.load(stream)
                    if packet['metadata'].get('observed_only') is not True:
                        raise ValueError('identifier packet is not observed-only')
                    train = history['training']+packet['training']
                    valid = history['validation']+packet['validation']
                    if sum(len(s['P_terminal']) for s in train) != 480 or sum(len(s['P_terminal']) for s in valid) != 480:
                        raise ValueError('full 480/480 merged rows required')
                    configuration(root)
                    if source_hashes() != expected_algorithm:
                        raise RuntimeError('loaded algorithm changed before fit')
                    fit_dir = location/'fit'
                    fit = fitter(train, valid, fit_dir, root_label=root_label,
                                 search_seconds=float(cfg.get('per_path_seconds', 25)))
                    row.update(status='fit_complete', fit_result_path=str(fit_dir/'result.json'),
                               fit_result_sha256=sha(fit_dir/'result.json'),
                               observed_cache_key=fit['matrix_cache']['cache_key'],
                               cache_reused=fit['matrix_cache']['reused'],
                               training_rows=fit['training_rows'], validation_rows=fit['validation_rows'],
                               search=search_audit(fit))
                except BatchBoundary:
                    row['status'] = 'unresolved_fit_count_boundary'; stop_gate = True
                except Exception as error:
                    row.update(status='failed', exception_type=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc())
                    if receipt_path.exists():
                        receipt = read(receipt_path)
                        row.update(actual_payment=receipt.get('actual_payment'),
                                   committed_payment=receipt.get('committed_payment'),
                                   unknown_payment_events=receipt.get('unknown_payment_events'))
                finally:
                    row['ended_utc'] = now()
                    save()
                    print(json.dumps(dict(event='fit_attempt_recorded', case=case_key, phase=phase,
                                           id=record_id, status=row['status'],
                                           paid=row['actual_payment'], LP=lp_ledger(cache_root)['total']), ensure_ascii=True), flush=True)

        # Freeze every observed-only result before opening ANY evaluator truth.
        write(folder/'fits_frozen.json', dict(protocol_sha256=sha(root/'protocol.json'), rows=rows, frozen_utc=now()))
        truth = None
        for row in rows:
            if row['status'] != 'fit_complete':
                continue
            physical = Path(row['physical_receipt_path']).parent
            receipt = read(row['physical_receipt_path'])
            truth_path = physical/'truth_evaluation.json'
            if sha(truth_path) != receipt['private_files']['truth_evaluation.json']['sha256']:
                raise RuntimeError('evaluator truth hash mismatch')
            current = {tuple(sorted(s)) for s in read(truth_path)['true_supports']}
            if truth is None:
                truth = current
            if current != truth:
                raise RuntimeError('static physical truth changed between menus or paths')
            if sha(row['fit_result_path']) != row['fit_result_sha256']:
                raise RuntimeError('fit changed after freeze')
            row['models'] = score_fit(read(row['fit_result_path']), [list(s) for s in truth], root_label)
            row['status'] = 'evaluated' if row['operation_checks_pass'] else 'evaluated_operation_failed'
            row['evaluation_utc'] = now()
        baseline = None
        if truth is not None and baseline_result.exists():
            b = read(baseline_result)
            if b.get('status') == 'complete':
                baseline = score_fit(b, [list(s) for s in truth], root_label)
        result = dict(status='bounded_phase_finished', case_key=case_key, phase=phase,
                      protocol_sha256=sha(root/'protocol.json'), planned=len(expected_ids),
                      recorded=len(rows), evaluated=sum(r['status'] == 'evaluated' for r in rows),
                      operation_failed_complete=sum(r['status'] == 'evaluated_operation_failed' for r in rows),
                      rows=rows, historical_stop=baseline, truth_signature=hashlib.sha256(plain(sorted(truth)).encode()).hexdigest() if truth is not None else None,
                      LP_ledger=lp_ledger(cache_root), elapsed_seconds=time.perf_counter()-started,
                      unique_new_fits=fitter.new_count, reused_fits=fitter.reused_count,
                      independent_response_paths=len(paths), original_history_rows=[288, 288], merged_rows=[480, 480],
                      finite_sample_reference_not_exact_expected_optimum=True,
                      oracle_selection_sha256=sha(root/'oracle_selected.json') if oracle else None,
                      source_case_unchanged=True, all_failure_and_wall_limit_records_retained=True)
        if phase == 'test':
            planned = read(case_root/'planning'/'phase_ended.json')
            if result['truth_signature'] is not None and planned.get('truth_signature') is not None and result['truth_signature'] != planned['truth_signature']:
                raise RuntimeError('planning/test true topology differs')
            result['frozen_reference_policy'] = oracle['cases'][case_key]['selected_policy']
            result['reference_resolved'] = oracle['cases'][case_key]['selection_resolved']
        write(folder/'phase_ended.json', result)
        save()
        print(json.dumps(dict(event='phase_finished', case=case_key, phase=phase,
                               evaluated=result['evaluated'], planned=result['planned'],
                               LP=result['LP_ledger']['total']), ensure_ascii=True), flush=True)
        return result
    finally:
        (folder/'phase.lock').unlink(missing_ok=True)


def select(root):
    root, cfg, _ = configuration(root)
    if (root/'oracle_selected.json').exists():
        raise FileExistsError('oracle selection already frozen; never retune from test')
    selections, bindings = {}, {}
    for case_key, case in cfg['cases'].items():
        path = root/case_key/'planning'/'phase_ended.json'
        phase = read(path)
        if phase['protocol_sha256'] != sha(root/'protocol.json'):
            raise RuntimeError('planning protocol mismatch')
        bindings[case_key] = sha(path)
        planned_n = len(cfg['planning_paths'])
        n = len(_history(case).get('labels', []))
        if not n:
            n = len(read(_reference(case['plan_lock']))['labels'])
        cap = 4*n
        policies = []
        lock = read(_reference(case['plan_lock']))
        for policy in cfg['policies']:
            rows = [r for r in phase['rows'] if r['policy'] == policy]
            known = [r['models']['multistart_original_rule']['full_errors'] for r in rows if r['status'] == 'evaluated']
            missing = planned_n-len(known)
            plan = lock['plans'][policy]
            expected_fee = .4*sum(c*(0., .5, .75)[c] for row in plan for c in row)
            frozen_fee = cfg.get('expected_fee_by_policy', {}).get(policy, expected_fee)
            if abs(float(frozen_fee)-expected_fee) > 1e-10:
                raise ValueError('predeclared expected fee differs from contract')
            policies.append(dict(policy=policy, denominator=planned_n, known=len(known), unresolved=missing,
                                 mean_error_lower=sum(known)/planned_n,
                                 mean_error_upper=(sum(known)+missing*cap)/planned_n,
                                 expected_fee=expected_fee))
        resolved = all(r['unresolved'] == 0 for r in policies)
        selected, tied = None, []
        if resolved:
            best = min(r['mean_error_lower'] for r in policies)
            tied = [r for r in policies if abs(r['mean_error_lower']-best) <= 1e-12]
            fee = min(r['expected_fee'] for r in tied)
            tied = [r['policy'] for r in tied if abs(r['expected_fee']-fee) <= 1e-12]
            selected = Random(str(cfg.get('oracle_tie_seed', 9583104))+':'+case_key).choice(tied)
        selections[case_key] = dict(selected_policy=selected, selection_resolved=resolved,
                                    all_error_and_fee_ties=tied, policies=policies,
                                    interpretation='known-model finite planning-sample reference; not deployment or exact global upper bound',
                                    unknown_loss_cap=cap)
    output = dict(status='frozen', frozen_utc=now(), protocol_sha256=sha(root/'protocol.json'),
                  planning_phase_sha256=bindings, cases=selections,
                  selection_uses_test_paths=False, expectation_estimated_before_menu_selection=True,
                  unresolved_reference_does_not_cancel_predeclared_test_controls=True)
    write(root/'oracle_selected.json', output)
    write(root/'oracle_selected.sha256.json', dict(sha256=sha(root/'oracle_selected.json')))
    print(json.dumps(output, ensure_ascii=True), flush=True)
    return output


def aggregate(root):
    root, cfg, _ = configuration(root)
    oracle = _oracle_guard(root, cfg)
    groups, differences = [], []
    for case_key in cfg['cases']:
        phase = read(root/case_key/'test'/'phase_ended.json')
        if phase['oracle_selection_sha256'] != sha(root/'oracle_selected.json'):
            raise RuntimeError('test was not run under frozen reference')
        methods = ('multistart_original_rule', 'rnj_bank', 'baseline25',
                   'geometric_RNJ_RX75_0.16', 'geometric_NJ_RX75_0.16')
        by = {(r['path'], r['policy']): r for r in phase['rows']}
        for policy in cfg['policies']:
            rows = [r for r in phase['rows'] if r['policy'] == policy]
            feasible = [r for r in rows if r['status'] == 'evaluated']
            models = {}
            for method in methods:
                models[method] = dict(evaluated=len(feasible),
                                      full_error_sum=sum(r['models'][method]['full_errors'] for r in feasible),
                                      full_recovery=sum(r['models'][method]['full_recovery'] for r in feasible))
            groups.append(dict(case_key=case_key, policy=policy, planned=len(cfg['test_paths']),
                               evaluated=len(feasible), unresolved=len(cfg['test_paths'])-len(feasible),
                               known_fee_sum=sum(r['actual_payment'] for r in rows if r['actual_payment'] is not None),
                               unknown_fee_records=sum(r['actual_payment'] is None for r in rows),
                               models=models, selected_by_planning_reference=oracle['cases'][case_key]['selected_policy'] == policy))
        chosen = oracle['cases'][case_key]['selected_policy']
        if chosen is not None:
            for baseline in ('zero', 'single', 'rotation'):
                paired = []
                for path in cfg['test_paths']:
                    left, right = by.get((path, chosen)), by.get((path, baseline))
                    if left and right and left['status'] == right['status'] == 'evaluated':
                        paired.append(dict(path=path, errors_delta=left['models']['multistart_original_rule']['full_errors']-right['models']['multistart_original_rule']['full_errors'],
                                           fee_delta=left['actual_payment']-right['actual_payment']))
                differences.append(dict(case_key=case_key, comparison='frozen_reference_minus_'+baseline,
                                        planned=len(cfg['test_paths']), paired=len(paired), rows=paired))
    output = dict(status='aggregate_complete', oracle_selection_sha256=sha(root/'oracle_selected.json'),
                  groups=groups, paired_frozen_reference=differences,
                  no_test_best_menu_selection=True, new_price_optimizer_claim=False,
                  independent_paths_per_case=len(cfg['test_paths']), method_counts_are_not_independent_samples=True,
                  all_failures_retained=True)
    write(root/'aggregate.json', output)
    print(json.dumps(output, ensure_ascii=True), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run-phase')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--case', required=True)
    run.add_argument('--phase', choices=('planning', 'test'), required=True)
    for name in ('select', 'aggregate'):
        cmd = sub.add_parser(name); cmd.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run-phase':
        run_phase(args.root, args.case, args.phase)
    elif args.command == 'select':
        select(args.root)
    else:
        aggregate(args.root)
