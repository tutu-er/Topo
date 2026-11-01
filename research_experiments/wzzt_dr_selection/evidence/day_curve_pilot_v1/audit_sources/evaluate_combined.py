"""Evaluate already frozen daily curves on historical residual blocks only.

This is a read-only combination of the five original planning records and the
one declared paper15 adapter retry. It imports no solver, fitter, simulator,
case graph, previous actual observation, or test-future observation. The new
random factors are frozen before reading historical packets for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dr_day_curve.day_model import (
    alternative_forecast_summary,
    build_problem,
    draw_scenarios,
    resource_audit,
    scenario_summary,
)

STAGE = Path(__file__).resolve().parent
ORIGINAL = STAGE / 'model_pilot_v1'
RETRY = STAGE / 'paper_adapter_retry_v1'
REPO = Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
POLICIES = ('zero', 'rotation', 'old_optimized', 'daily_model_candidate')
DEPLOYABLE_SAMPLE_STATUSES = frozenset(('zero_cost_optimal', 'target_feasible_cost_incumbent'))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('frozen input drift: ' + str(path))
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')
    temp.replace(path)


def curve_from_plan(plan, resources):
    u = np.zeros((len(resources), 96))
    positions = {x['resource_id']: i for i, x in enumerate(resources)}
    active_prices = np.zeros(len(resources))
    priced_resources = set()
    for module, amplitude, price in zip(plan['modules'], plan['amplitudes'], plan['prices']):
        if len(module['profiles']) != 1:
            raise ValueError('separately priced original one-resource modules required')
        for name, profile in module['profiles'].items():
            i = positions[name]
            u[i] += float(amplitude) * np.asarray(profile, float)
            if float(amplitude) != 0.:
                if name in priced_resources:
                    raise ValueError('multiple active original modules on one resource')
                priced_resources.add(name)
                active_prices[i] = float(price)
    return u, active_prices


def _verify_source_manifest(manifest):
    for name, digest in manifest.items():
        if sha(name) != digest:
            raise ValueError('frozen planning source drift: ' + name)


def _planning_source_audit(manifest, archive_path):
    """Check every original byte against its immutable snapshot.

    The original runner is not imported here. Its sole permitted current-file
    drift is the parent-confirmed later persist/raw-result adapter repair.
    All mathematical/model/solver/adapter source bytes must remain unchanged.
    """
    drifts = []
    with zipfile.ZipFile(archive_path) as archive:
        for name, digest in manifest.items():
            path = Path(name)
            member = ('dr_day_curve/' + path.name if path.parent == STAGE / 'dr_day_curve'
                      else path.relative_to(REPO).as_posix())
            if member not in archive.namelist():
                # The independently archived retry uses numbered basenames.
                matches = [x for x in archive.namelist() if '_' in x
                           and x.split('_', 1)[0].isdigit() and x.split('_', 1)[1] == path.name]
                if len(matches) != 1:
                    raise ValueError('unique frozen source archive member required: ' + name)
                member = matches[0]
            archived_digest = hashlib.sha256(archive.read(member)).hexdigest()
            if archived_digest != digest:
                raise ValueError('archived frozen source hash mismatch: ' + member)
            current_digest = sha(path)
            if current_digest != digest:
                if path != STAGE / 'dr_day_curve/runner.py':
                    raise ValueError('unapproved mathematical or adapter source drift: ' + name)
                drifts.append(dict(path=name, archive_member=member, frozen_sha256=digest,
                    archived_sha256=archived_digest, current_sha256=current_digest,
                    reason='parent-confirmed subsequent raw-result persistence/adapter bridge repair; original planning not rerun',
                    imported_by_this_evaluator=False))
    return dict(archive=ref(archive_path), frozen_source_count=len(manifest),
                all_archived_bytes_match=True, subsequent_current_file_drifts=drifts)


def prepare(output):
    """Freeze completed result references and evaluation seeds; evaluate nothing."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    original_end = read(ORIGINAL / 'planning_ended.json')
    retry_end = read(RETRY / 'ended.json')
    p = read(ORIGINAL / 'protocol.json')
    rp = read(RETRY / 'protocol.json')
    if original_end.get('status') != 'complete' or original_end.get('planned') != 5:
        raise ValueError('the original five-case planning stage has not ended')
    if retry_end.get('status') != 'complete':
        raise ValueError('the declared adapter retry has not ended')
    if len(p['cases']) != 5 or len(original_end['rows']) != 5:
        raise ValueError('all five original conditions must remain')
    original_sources = _planning_source_audit(p['source_sha256'], ORIGINAL / 'source_snapshot.zip')
    retry_sources = _planning_source_audit(rp['source_sha256'], RETRY / 'source_snapshot.zip')
    original_refs = {x['condition']: x['result'] for x in original_end['rows']}
    cases = []
    for c in p['cases']:
        old_ref = original_refs[c['name']]
        old_result = read(checked(old_ref))
        if c['name'] == rp['condition']:
            if old_ref != rp['original_failure'] or old_result.get('status') != 'failed':
                raise ValueError('adapter retry must retain its exact original failure')
            result_ref = retry_end['result']
            checked(retry_end['raw_solver'])
            chosen = read(checked(result_ref))
            if chosen.get('condition') != c['name']:
                raise ValueError('retry condition mismatch')
            source = 'declared_adapter_retry'
        else:
            result_ref = old_ref
            source = 'original_planning'
        # Check hashes now; curve or history data are not evaluated at prepare.
        checked(c['pool'])
        checked(c['history'])
        for item in c['old_plans'].values():
            checked(item)
        cases.append(dict(condition=c['name'], case=c['case'], result=result_ref,
            result_source=source, original_result=old_ref,
            original_status=old_result['status'], pool=c['pool'], history=c['history'],
            old_plans=c['old_plans'],
            training_seed=int(c['training_evaluation_seed']),
            validation_seed=int(c['validation_evaluation_seed'])))
    if len({c['training_seed'] for c in cases} | {c['validation_seed'] for c in cases}) != 10:
        raise ValueError('distinct predeclared response evaluation seeds required')
    sources = [Path(__file__).resolve(), STAGE / 'dr_day_curve/day_model.py']
    output.mkdir(parents=True)
    protocol = dict(status='frozen', id='daily_curve_combined_historical_evaluation_v1',
        created_utc=now(), original_protocol=ref(ORIGINAL / 'protocol.json'),
        original_planning_ended=ref(ORIGINAL / 'planning_ended.json'),
        original_price_freeze=ref(ORIGINAL / 'price_freeze.json'),
        retry_protocol=ref(RETRY / 'protocol.json'), retry_ended=ref(RETRY / 'ended.json'),
        retry_raw_solver=retry_end['raw_solver'], cases=cases,
        source_sha256={str(path): sha(path) for path in sources},
        planning_source_sha256=p['source_sha256'],
        retry_source_sha256=rp['source_sha256'], samples_per_role=256,
        original_source_audit=original_sources, retry_source_audit=retry_sources,
        unused_live_runner_policy='old runner verified from immutable snapshot; current runner never imported and may receive subsequent interface-only repairs',
        preserved_preflight_failure=dict(error_type='ValueError',
            message='frozen planning source drift: dr_day_curve/runner.py',
            phase='first prepare, before output creation or historical packet evaluation',
            disposition='verify immutable planning snapshots and explicitly record later unused runner interface repair'),
        preserved_archive_layout_preflight_failure=dict(error_type='KeyError',
            message='retry snapshot used numbered basenames rather than package paths',
            phase='second prepare, before output creation or historical packet evaluation',
            disposition='resolve exact numbered basenames and verify bytes against original hash; broad suffix attempt was also rejected as ambiguous before evaluation'),
        historical_days_per_role=3, historical_rows_per_role=288,
        policies=list(POLICIES), probability_requirement=.9,
        factor_law='one pair of iid Uniform[-1,1] latent factors per complete day; shared across policies; original D retained',
        day_law='uniform draw among all three declared historical residual blocks; all blocks retained and counts reported',
        forecast='historical P/Q/Y residual template plus fixed refined-candidate response; competing coarse-response sensitivity reported separately',
        price_scope='old policies keep their frozen quoted prices; canonical repricing is diagnostic only',
        candidate_scope='sample-target-infeasible/unresolved incumbents are phase-I diagnostics, not deployable candidates',
        independence_scope='fresh response factors only; historical blocks reused; no new physical independence or field-probability guarantee',
        no_new_optimization=True, no_new_AC=True, no_fitting=True,
        no_old_actual_read=True, no_true_graph_read=True, no_test_future_read=True,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__))
    write(output / 'protocol.json', protocol)
    write(output / 'progress.json', dict(status='prepared', completed=0, planned=5, rows=[]))
    return protocol


def _summary_with_days(problem, scenarios, curve, alternative=False):
    function = alternative_forecast_summary if alternative else scenario_summary
    summary = function(problem, scenarios, curve)
    summary['passed_fraction'] = summary['passed'] / summary['denominator']
    summary['positive_exit_margin_count'] = sum(x['margin_micro'] > 0. for x in summary['rows'])
    summary['positive_exit_margin_fraction'] = summary['positive_exit_margin_count'] / summary['denominator']
    summary['historical_template_counts'] = {str(i): sum(s['day'] == i for s in scenarios) for i in range(3)}
    summary['per_historical_day'] = {}
    for i in range(3):
        subset = [s for s in scenarios if s['day'] == i]
        if not subset:
            summary['per_historical_day'][str(i)] = dict(denominator=0, status='not_drawn')
            continue
        local = function(problem, subset, curve)
        summary['per_historical_day'][str(i)] = dict(
            denominator=len(subset), passed=local['passed'],
            passed_fraction=local['passed'] / len(subset),
            mean_margin_micro=float(np.mean([x['margin_micro'] for x in local['rows']])),
            mean_increment_micro=local['mean_increment_micro'],
            positive_exit_margin_count=sum(x['margin_micro'] > 0. for x in local['rows']))
    return summary


def _compact(summary):
    return {k: v for k, v in summary.items() if k not in ('rows', 'forecast_scope')}


def _resource_scalars(audit):
    return {k: audit[k] for k in ('feasible', 'maximum_violation', 'expected_payment', 'hard_payment_bound')}


def _status(result, problem, curve):
    planning_status = result['status']
    train = result.get('training', {})
    resource = resource_audit(problem, curve, prices=result.get('prices'))
    sample_feasible = (train.get('passed', -1) >= train.get('required', math.inf)
                       and resource['feasible'])
    declared = planning_status in DEPLOYABLE_SAMPLE_STATUSES and sample_feasible
    return dict(planning_status=planning_status,
        planning_sample_passed=train.get('passed'), planning_sample_required=train.get('required'),
        planning_sample_slack_micro=train.get('feasibility_slack_micro'),
        planning_resource_feasible=resource['feasible'],
        usage='conditional_sample_feasible_candidate' if declared else 'phase_I_diagnostic_only',
        field_deployment_validated=False, new_AC_validated=False,
        primary=result.get('primary'), secondary=result.get('secondary'),
        numerical_polish=result.get('numerical_polish'))


def evaluate(output):
    output = Path(output).resolve()
    p = read(output / 'protocol.json')
    _verify_source_manifest(p['source_sha256'])
    runtime_source_audits = {}
    for key, audit_key in (('planning_source_sha256', 'original_source_audit'),
                           ('retry_source_sha256', 'retry_source_audit')):
        archive = checked(p[audit_key]['archive'])
        audit = _planning_source_audit(p[key], archive)
        if any(audit[x] != p[audit_key][x] for x in
               ('archive', 'frozen_source_count', 'all_archived_bytes_match')):
            raise ValueError('frozen source archive audit changed: ' + key)
        runtime_source_audits[audit_key] = audit
    write(output / 'source_audit_receipt.json', dict(status='verified',
        created_utc=now(), audits=runtime_source_audits,
        current_runner_imported=False, mathematical_sources_unchanged=True))
    for key in ('original_protocol', 'original_planning_ended', 'original_price_freeze',
                'retry_protocol', 'retry_ended', 'retry_raw_solver'):
        checked(p[key])
    rows = []
    for c in p['cases']:
        out = output / 'cases' / c['condition'] / 'summary.json'
        if out.exists():
            result = read(out)
        else:
            try:
                r = read(checked(c['result']))
                original_result = read(checked(c['original_result']))
                pool = read(checked(c['pool']))
                with checked(c['history']).open('rb') as stream:
                    packet = pickle.load(stream)
                history = packet['history'] if 'history' in packet else packet
                plans = {name: read(checked(item)) for name, item in c['old_plans'].items()}
                resources = plans['optimized']['resources']
                curves = {}
                quotes = {}
                for name, alias in (('zero', 'zero'), ('rotation', 'rotation'), ('optimized', 'old_optimized')):
                    curves[alias], quotes[alias] = curve_from_plan(plans[name], resources)
                if r.get('mean_u') is None:
                    raise ValueError('selected planning record has no evaluable curve: ' + r['status'])
                curves['daily_model_candidate'] = np.asarray(r['mean_u'], float)
                quotes['daily_model_candidate'] = np.asarray(r['prices'], float)
                if any(u.shape != (len(resources), 96) for u in curves.values()):
                    raise ValueError('all policies must preserve two full-day resource curves')
                summaries = {}
                random_draws = {}
                for role in ('training', 'validation'):
                    problem = build_problem(pool, history, resources, template_role=role)
                    seed = c[role + '_seed']
                    scenarios = draw_scenarios(problem, p['samples_per_role'], seed)
                    random_draws[role] = [dict(day=s['day'], z=s['z']) for s in scenarios]
                    summaries[role] = {}
                    for name in POLICIES:
                        curve = curves[name]
                        summaries[role][name] = dict(
                            reference_fine=_summary_with_days(problem, scenarios, curve),
                            alternative_coarse=_summary_with_days(problem, scenarios, curve, alternative=True),
                            original_quote_resource_audit=resource_audit(problem, curve, prices=quotes[name]),
                            canonical_price_resource_audit=resource_audit(problem, curve))
                training_problem = build_problem(pool, history, resources, template_role='training')
                result = dict(status='complete', condition=c['condition'], case=c['case'],
                    result_source=c['result_source'],
                    original_result_status=c['original_status'],
                    retained_original_failure=(dict(status=original_result['status'],
                        error_type=original_result.get('error_type'), message=original_result.get('message'))
                        if c['result_source'] == 'declared_adapter_retry' else None),
                    candidate_status=_status(r, training_problem, curves['daily_model_candidate']),
                    history_days={role: len(history[role]) for role in ('training', 'validation')},
                    history_rows={role: sum(len(s['p']) if 'p' in s else 96 for s in history[role])
                                  for role in ('training', 'validation')},
                    samples_per_role=p['samples_per_role'], summaries=summaries,
                    response_draws=random_draws, sources=dict(result=c['result'], original_result=c['original_result'],
                        pool=c['pool'], history=c['history'], old_plans=c['old_plans']))
            except Exception as error:
                result = dict(status='evaluation_failed', condition=c['condition'], case=c['case'],
                    result_source=c['result_source'], original_result_status=c['original_status'],
                    error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            write(out, result)
        rows.append(dict(condition=c['condition'], status=result['status'], result=ref(out)))
        write(output / 'progress.json', dict(status='running', completed=len(rows), planned=5, rows=rows))
        print(dict(stage='historical_model_evaluated', completed=len(rows), planned=5,
                   condition=c['condition'], status=result['status']), flush=True)
    cases = [read(checked(x['result'])) for x in rows]
    complete = [c for c in cases if c['status'] == 'complete']
    full = dict(status='complete' if len(complete) == 5 else 'complete_with_failures',
        created_utc=now(), protocol=ref(output / 'protocol.json'), cases=cases,
        planned_conditions=5, completed_conditions=len(complete), failed_conditions=5-len(complete),
        planned_policy_role_cells=5*4*2, completed_policy_role_cells=len(complete)*4*2,
        planned_draws_per_role=5*256, complete_draws_per_role=len(complete)*256,
        physical_independent_paths=0, new_AC=0, new_fit_calls=0, new_optimization_calls=0,
        scope=p['independence_scope'])
    write(output / 'summary_private.json', full)
    public_cases = []
    for case in cases:
        item = {k: case[k] for k in ('condition', 'case', 'status', 'result_source', 'original_result_status')}
        if case['status'] == 'complete':
            item.update(candidate_status=case['candidate_status'], retained_original_failure=case['retained_original_failure'],
                history_days=case['history_days'], history_rows=case['history_rows'], samples_per_role=case['samples_per_role'])
            item['summaries'] = {role: {policy: dict(
                reference_fine=_compact(s['reference_fine']), alternative_coarse=_compact(s['alternative_coarse']),
                original_quote_resource_audit=_resource_scalars(s['original_quote_resource_audit']),
                canonical_price_resource_audit=_resource_scalars(s['canonical_price_resource_audit']))
                for policy, s in policies.items()} for role, policies in case['summaries'].items()}
        else:
            item.update(error_type=case['error_type'], message=case['message'])
        public_cases.append(item)
    public = {k: v for k, v in full.items() if k not in ('protocol', 'cases')}
    public.update(protocol_sha256=sha(output / 'protocol.json'), cases=public_cases,
        probability_scope='conditional historical-template excitation frequencies; no topology recovery probability or field guarantee',
        reporting_scope='scalars only; no P/Q/V, impedance matrices, modulation curves, or random factor arrays',
        rule='historical3percent/newday2percent with historical:newday weight .6:.4; parameters frozen')
    write(output / 'summary_public_candidate.json', public)
    write(output / 'ended.json', dict(status=full['status'], created_utc=now(),
        private_summary=ref(output / 'summary_private.json'),
        public_summary_candidate=ref(output / 'summary_public_candidate.json'),
        completed_conditions=len(complete), planned_conditions=5))
    write(output / 'progress.json', dict(status=full['status'], completed=len(rows), planned=5, rows=rows))
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()
