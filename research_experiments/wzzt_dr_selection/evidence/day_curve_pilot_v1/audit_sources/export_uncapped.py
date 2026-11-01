"""Export completed uncapped historical-model evaluation and audit its algebra.

Reads only frozen planning/results, historical P/Q/Y and fitted matrix pools.
It neither imports the planner nor runs optimization, fitting, AC, true-case
graphs or actual/test-future observations. Old capped results stay untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power, _target

REPO = Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
POLICIES = ('zero', 'rotation', 'optimized', 'daily_optimized')
SAMPLE_FEASIBLE_STATUSES = frozenset(('zero_cost_optimal', 'target_feasible_cost_incumbent'))


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))


def checked(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('frozen input drift: ' + str(path))
    return path


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')
    temp.replace(path)


def snapshot_audit(root, protocol):
    snapshot = root / 'source_snapshot.zip'
    with zipfile.ZipFile(snapshot) as archive:
        for name, digest in protocol['source_sha256'].items():
            path = Path(name)
            member = ('dr_day_curve/' + path.name if path.parent.name == 'dr_day_curve'
                      else path.relative_to(REPO).as_posix())
            if hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError('frozen source archive mismatch: ' + member)
    return dict(snapshot=ref(snapshot), verified_files=len(protocol['source_sha256']),
                current_planner_package_imported=False)


def curves_from_plans(plans, resources):
    curves, prices = {}, {}
    positions = {r['resource_id']: i for i, r in enumerate(resources)}
    for name, plan in plans.items():
        u = np.zeros((len(resources), 96)); c = np.zeros(len(resources)); active = set()
        for module, amplitude, price in zip(plan['modules'], plan['amplitudes'], plan['prices']):
            if len(module['profiles']) != 1:
                raise ValueError('one separately metered resource per module required')
            for rid, profile in module['profiles'].items():
                i = positions[rid]; u[i] += float(amplitude) * np.asarray(profile)
                if float(amplitude) != 0.:
                    if rid in active:
                        raise ValueError('multiple active modules on one resource')
                    active.add(rid); c[i] = float(price)
        curves[name], prices[name] = u, c
    return curves, prices


def metrics(rows, day_ids):
    count = len(rows); required = math.ceil(.9 * count)
    margins = np.array([r['margin_micro'] for r in rows])
    surplus = margins - np.array([r['target_micro'] for r in rows])
    q = float(np.sort(surplus)[::-1][required - 1])
    return dict(denominator=count, required=required, passed=sum(r['passed'] for r in rows),
        passed_fraction=sum(r['passed'] for r in rows) / count,
        positive_exit_margin_count=int((margins > 0.).sum()),
        mean_margin_micro=float(margins.mean()),
        mean_increment_micro=float(np.mean([r['increment_micro'] for r in rows])),
        lower90_surplus_micro=q, feasibility_slack_micro=max(0., -q),
        historical_template_counts={str(i): int(np.sum(np.asarray(day_ids) == i)) for i in range(3)})


def paired(rows, zero):
    a = np.array([r['passed'] for r in rows]); b = np.array([r['passed'] for r in zero])
    return dict(denominator=len(rows), gained_pass_count=int((a & ~b).sum()),
        lost_pass_count=int((~a & b).sum()), both_pass_count=int((a & b).sum()),
        both_fail_count=int((~a & ~b).sum()),
        mean_margin_increment_vs_zero_micro=float(np.mean([r['margin_micro'] - z['margin_micro']
                                                         for r, z in zip(rows, zero)])))


def summarize(rows, zero, days):
    result = metrics(rows, days); result['paired_vs_zero'] = paired(rows, zero)
    result['per_historical_day'] = {}
    for i in range(3):
        selected = [k for k, day in enumerate(days) if day == i]
        if not selected:
            result['per_historical_day'][str(i)] = dict(denominator=0, status='not_drawn')
            continue
        local = [rows[k] for k in selected]; base = [zero[k] for k in selected]
        item = metrics(local, [i] * len(local)); item['paired_vs_zero'] = paired(local, base)
        result['per_historical_day'][str(i)] = item
    return result


def reproduce_draws(seed, samples):
    rng = np.random.default_rng(seed)
    days, z = [], []
    for _ in range(samples):
        days.append(int(rng.integers(3))); z.append(rng.uniform(-1., 1., 2))
    return np.array(days), np.asarray(z)


def status_of(result):
    status = result['status']; training = result.get('training', {})
    sample_pass = training.get('passed', -1) >= training.get('required', math.inf)
    resource_ok = result.get('resource_audit', {}).get('feasible', False)
    return dict(planning_status=status,
        usage='conditional_sample_feasible_candidate' if status in SAMPLE_FEASIBLE_STATUSES
              and sample_pass and resource_ok else 'phase_I_diagnostic_only',
        planning_passed=training.get('passed'), planning_required=training.get('required'),
        planning_slack_micro=training.get('feasibility_slack_micro'),
        planning_resource_feasible=resource_ok,
        primary=result.get('primary'), secondary=result.get('secondary'),
        numerical_polish=result.get('numerical_polish'),
        field_deployment_validated=False, new_AC_validated=False)


def run(run_root, output):
    root = Path(run_root).resolve(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('new export directory required: ' + str(output))
    protocol = read(root / 'protocol.json'); ended = read(root / 'ended.json')
    planning_ended = read(root / 'planning_ended.json')
    if protocol.get('budget', 'missing') is not None or protocol.get('payment_cap_enforced') is not False:
        raise ValueError('explicit uncapped payment protocol required')
    if ended.get('status') != 'complete' or planning_ended.get('status') != 'complete':
        raise ValueError('planning and original model evaluation must complete before export')
    eval_path = checked(ended['model_evaluation']); model_eval = read(eval_path)
    if len(protocol['cases']) != 5 or len(model_eval['cases']) != 5:
        raise ValueError('all five original conditions must remain')
    closure = snapshot_audit(root, protocol)
    frozen_pool_source = REPO / 'research_experiments/wzzt_dr_selection/dr_frozen_distance/frozen_pool.py'
    if sha(frozen_pool_source) != protocol['source_sha256'][str(frozen_pool_source)]:
        raise ValueError('current P/Q/Y reader differs from frozen source')
    refs = []
    for case in protocol['cases']:
        checked(case['pool']); checked(case['history'])
        for item in case['old_plans'].values(): checked(item)
        refs.append(dict(condition=case['name'], result=ref(root / 'cases' / case['name'] / 'result.json'),
                         pool=case['pool'], history=case['history'], old_plans=case['old_plans']))
    output.mkdir(parents=True)
    export_protocol = dict(status='frozen', created_utc=now(), export_script=ref(__file__),
        original_protocol=ref(root / 'protocol.json'), planning_ended=ref(root / 'planning_ended.json'),
        price_freeze=ref(root / 'price_freeze.json'), model_evaluation=ref(eval_path), original_ended=ref(root / 'ended.json'),
        source_snapshot_audit=closure, cases=refs, samples_per_policy_role=256,
        budget=None, payment_cap_enforced=False,
        scope='scalar export and direct-matrix replay of completed historical-template evaluation; no new planning or physical observations',
        no_new_optimization=True, no_new_AC=True, no_new_fitting=True, no_test_future_read=True)
    write(output / 'protocol.json', export_protocol)
    original_cases = {c['condition']: c for c in model_eval['cases']}
    public_cases, audit_cases = [], []
    counters = dict(reference_margin=0, alternative_margin=0, reference_pass=0, alternative_pass=0,
                    baseline_margin=0, target=0, expected_payment=0, hard_payment=0)
    maximum_error = 0.
    for c, source in zip(protocol['cases'], refs):
        original = original_cases[c['name']]; r = read(checked(source['result']))
        item = dict(condition=c['name'], case=c['case'], status=original['status'], candidate_status=status_of(r),
                    budget=None, payment_cap_enforced=False)
        if original['status'] != 'complete' or r.get('mean_u') is None:
            item['retained_failure'] = dict(planning_status=r['status'], message=r.get('message'))
            public_cases.append(item); continue
        pool = read(checked(c['pool']))
        with checked(c['history']).open('rb') as stream: packet = pickle.load(stream)
        history = packet.get('history', packet); labels = pool['labels']; n = len(labels)
        models = {m['candidate_id']: m for m in pool['models']}
        fine = models['refined']; bad = next(v for k, v in models.items() if k != 'refined')
        rf, xf, rb, xb = [np.asarray(x, float) for x in (fine['R'], fine['X'], bad['R'], bad['X'])]
        historical = {}
        for name, model in [('fine', fine), ('bad', bad)]:
            errors = []
            for scene in history['validation']:
                pp, qq = _power(scene, labels); yy = _target(scene, pp.shape)
                errors.append(np.abs(yy - pp @ np.asarray(model['R']).T - qq @ np.asarray(model['X']).T))
            historical[name] = float(np.stack(errors).mean())
        H = .6 * (historical['bad'] - 1.03 * historical['fine']) * 1e6
        plans = {name: read(checked(value)) for name, value in c['old_plans'].items()}
        resources = plans['optimized']['resources']; curves, quotes = curves_from_plans(plans, resources)
        curves['daily_optimized'] = np.asarray(r['mean_u']); quotes['daily_optimized'] = np.asarray(r['prices'])
        item.update(history_days={role: len(history[role]) for role in ('training', 'validation')},
                    history_rows={role: 288 for role in ('training', 'validation')}, summaries={})
        case_error = 0.
        for role in ('training', 'validation'):
            days, z = reproduce_draws(c[role + '_evaluation_seed'], protocol['model_holdout_samples'])
            if len(history[role]) != 3:
                raise ValueError('all three historical days must remain')
            templates = []
            for scene in history[role]:
                pp, qq = _power(scene, labels); yy = _target(scene, pp.shape)
                if pp.shape != (96, n): raise ValueError('complete96-slot historical blocks required')
                templates.append((pp, qq, yy))
            basep, baseq, basey = [np.stack([templates[i][j] for i in days]) for j in range(3)]
            zero_bad = np.abs(basey - basep @ rb.T - baseq @ xb.T).mean(axis=(1, 2))
            zero_fine = np.abs(basey - basep @ rf.T - baseq @ xf.T).mean(axis=(1, 2))
            M0 = H + .4 * (zero_bad - 1.02 * zero_fine) * 1e6 - 1e-6
            target = .2 * np.maximum(-M0, 0.) + .0001
            factors = 1 + z @ np.array([[.15, .05], [.15, -.05]]).T
            item['summaries'][role] = {}
            for policy in POLICIES:
                cell = original['summaries'][role][policy]; u = curves[policy]
                dp = np.zeros_like(basep); dq = np.zeros_like(baseq)
                for j, resource in enumerate(resources):
                    col = labels.index(resource['node_label']); actual = factors[:, j, None] * u[j][None, :]
                    ratio = np.broadcast_to(np.asarray(resource.get('reactive_ratio', .35), float), (96,))
                    dp[:, :, col] += actual / pool['base_kw']; dq[:, :, col] += actual * ratio[None, :] / pool['base_kw']
                pp, qq = basep + dp, baseq + dq
                yf = basey + dp @ rf.T + dq @ xf.T; yb = basey + dp @ rb.T + dq @ xb.T
                predictions = dict(
                    reference_fine=H + .4 * (np.abs(yf - pp @ rb.T - qq @ xb.T).mean(axis=(1, 2))
                        - 1.02 * np.abs(yf - pp @ rf.T - qq @ xf.T).mean(axis=(1, 2))) * 1e6 - 1e-6,
                    alternative_coarse=H + .4 * (np.abs(yb - pp @ rb.T - qq @ xb.T).mean(axis=(1, 2))
                        - 1.02 * np.abs(yb - pp @ rf.T - qq @ xf.T).mean(axis=(1, 2))) * 1e6 - 1e-6)
                scalar_cell = {}
                for hypothesis, values in predictions.items():
                    rows = cell[hypothesis]['rows']; zero_rows = original['summaries'][role]['zero'][hypothesis]['rows']
                    if len(rows) != 256: raise ValueError('all256 model draws must remain')
                    error = float(np.max(np.abs(values - np.array([x['margin_micro'] for x in rows]))))
                    maximum_error = max(maximum_error, error); case_error = max(case_error, error)
                    if error > 1e-7: raise AssertionError('direct frozen-matrix margin mismatch')
                    if not np.array_equal(values >= target - 1e-7, np.array([x['passed'] for x in rows])):
                        raise AssertionError('direct model-target pass mismatch')
                    which = 'reference' if hypothesis == 'reference_fine' else 'alternative'
                    counters[which + '_margin'] += 256; counters[which + '_pass'] += 256
                    scalar_cell[hypothesis] = summarize(rows, zero_rows, days)
                for key, values in [('baseline_margin_micro', M0), ('target_micro', target)]:
                    error = float(np.max(np.abs(values - np.array([x[key] for x in cell['reference_fine']['rows']]))))
                    maximum_error = max(maximum_error, error)
                    if error > 1e-7: raise AssertionError('direct baseline/target mismatch')
                    counters['baseline_margin' if key.startswith('baseline') else 'target'] += 256
                fee = float(quotes[policy] @ (.25 * np.maximum(-u, 0.).sum(axis=1)))
                hard = 1.2 * fee
                if policy == 'daily_optimized':
                    if abs(fee-r['resource_audit']['expected_payment']) > 1e-9 or abs(hard-r['resource_audit']['hard_payment_bound']) > 1e-9:
                        raise AssertionError('daily frozen-quote settlement mismatch')
                else:
                    if abs(fee-plans[policy]['expected_payment']) > 1e-9 or abs(hard-plans[policy]['hard_payment_bound']) > 1e-9:
                        raise AssertionError('old frozen-quote settlement mismatch')
                counters['expected_payment'] += 1; counters['hard_payment'] += 1
                audit = cell['resource_audit']
                if audit.get('payment_cap', 'missing') is not None or audit.get('payment_cap_enforced') is not False:
                    raise ValueError('evaluated resource audit must be explicitly uncapped')
                scalar_cell['resource_and_cost'] = dict(feasible=audit['feasible'], maximum_violation=audit['maximum_violation'],
                    budget=None, payment_cap_enforced=False, frozen_quote_expected_payment=fee,
                    frozen_quote_hard_payment_bound=hard,
                    canonical_repricing_diagnostic_expected_payment=audit['expected_payment'],
                    canonical_repricing_diagnostic_hard_payment_bound=audit['hard_payment_bound'])
                item['summaries'][role][policy] = scalar_cell
        public_cases.append(item); audit_cases.append(dict(condition=c['name'], maximum_error_micro=case_error))
    complete = sum(c['status'] == 'complete' for c in public_cases)
    summary = dict(status='complete' if complete == 5 else 'complete_with_failures', created_utc=now(),
        protocol_sha256=sha(output / 'protocol.json'), budget=None, payment_cap_enforced=False,
        planned_conditions=5, completed_conditions=complete, retained_failed_conditions=5-complete,
        planned_policy_role_cells=40, completed_policy_role_cells=complete*8,
        samples_per_policy_role=256, historical_residual_days_per_condition_role=3,
        response_draw_scope='new latent response factors only; four policies share each role draw set; three old residual days reused',
        independent_new_measurement_days=0, new_AC=0, new_fit_calls=0, new_optimization_calls=0,
        forecast_scope='fixed refined-candidate response hypothesis; fixed competing coarse response reported separately',
        probability_scope='model-template excitation frequencies; neither physical90percent nor topology recovery probability',
        rule='historical3percent/newday2percent; historical:newday weight .6:.4; frozen parameters and candidate outputs',
        cost_scope='no payment cap; fees are minimized and reported; old policies retain their old frozen quotes',
        public_content='scalar metrics only; no modulation, P/Q/V, R/X or latent factor arrays', cases=public_cases)
    write(output / 'summary_public_candidate.json', summary)
    audit = dict(status='passed', created_utc=now(), maximum_error_micro=maximum_error,
        comparison_counts=counters, cases=audit_cases, export_protocol=ref(output / 'protocol.json'),
        source_model_evaluation=ref(eval_path), public_summary=ref(output / 'summary_public_candidate.json'),
        method='direct original-unit matrix response replay, paired pass metrics and original-quote settlement; no planner imports',
        new_AC=0, new_fit_calls=0, new_optimization_calls=0)
    write(output / 'audit_direct_matrix.json', audit)
    write(output / 'ended.json', dict(status=summary['status'], created_utc=now(),
        public_summary=ref(output / 'summary_public_candidate.json'), audit=ref(output / 'audit_direct_matrix.json')))
    print(json.dumps(dict(status=summary['status'], completed_conditions=complete, planned_conditions=5,
        maximum_matrix_error_micro=maximum_error, comparison_counts=counters,
        public_summary=ref(output / 'summary_public_candidate.json')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.run_root, args.output)
