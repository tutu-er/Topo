"""Read-only geometric supplement for frozen budget-reference observations.

freeze reads the protocol and this source only. run requires every planning/test
phase, the oracle choice and aggregate to be frozen. It never calls a physical
producer, projected sensitivity fit, LP, WZZT search or price selection. Failed
joint fits stay failed; retained frontends only add geometric RNJ/NJ controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def freeze(root):
    root = Path(root).resolve()
    protocol = root/'protocol.json'
    lock = root/'audit_failure_fronts_lock.json'
    value = dict(script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
                 protocol_sha256=sha(protocol), frozen_utc=now(),
                 scope='post-freeze geometric-only supplement; original joint failures unchanged',
                 no_new_QP_LP_AC_or_identifier_calls=True)
    if lock.exists():
        old = read(lock)
        if old['script_sha256'] != value['script_sha256'] or old['protocol_sha256'] != value['protocol_sha256']:
            raise RuntimeError('existing supplement lock differs; never overwrite a frozen lock')
        return old
    write(lock, value)
    return value


def error(family, truth):
    missing, extra = truth-family, family-truth
    return dict(full_errors=len(missing)+len(extra), full_recovery=not missing and not extra,
                missing=[list(s) for s in sorted(missing)], extra=[list(s) for s in sorted(extra)],
                predicted_blocks=len(family), truth_blocks=len(truth))


def support_family(supports, labels=None):
    if labels is None:
        return {tuple(sorted(int(v) for v in s)) for s in supports}
    return {tuple(sorted(int(labels[i]) for i in s)) for s in supports}


def risk_interval(values, planned, n):
    if len(values) > planned:
        raise RuntimeError('duplicate observations exceed declared denominator')
    missing = planned-len(values)
    total = sum(v['full_errors'] for v in values)
    recovered = sum(bool(v['full_recovery']) for v in values)
    return dict(planned=planned, evaluated=len(values), unresolved=missing,
                full_error_sum_known=total, mean_error_lower=total/planned,
                mean_error_upper=(total+missing*4*n)/planned,
                full_recovery_known=recovered, recovery_fraction_lower=recovered/planned,
                recovery_fraction_upper=(recovered+missing)/planned,
                unknown_loss_cap=4*n, denominator_is_fixed=True)


def run(root):
    root = Path(root).resolve()
    lock_path = root/'audit_failure_fronts_lock.json'
    lock = read(lock_path)
    if sha(__file__) != lock['script_sha256'] or sha(root/'protocol.json') != lock['protocol_sha256']:
        raise RuntimeError('script/protocol must match the pre-execution supplement lock')
    cfg = read(root/'protocol.json')
    repo = Path(cfg['repository']).resolve()
    expected_paths = [root/'oracle_selected.json', root/'oracle_selected.sha256.json', root/'aggregate.json']
    for case in cfg['cases']:
        for phase in ('planning', 'test'):
            expected_paths += [root/case/phase/'fits_frozen.json', root/case/phase/'phase_ended.json']
    absent = [str(p) for p in expected_paths if not p.is_file()]
    if absent:
        raise RuntimeError('all four phases and final aggregate must finish before any result/truth read: '+str(absent))
    output = root/'supplement_failure_fronts_v1'
    if (output/'report.json').exists() or (output/'audit_ended.json').exists():
        raise FileExistsError('completed supplement retained; no implicit rerun')
    oracle_path = root/'oracle_selected.json'
    oracle = read(oracle_path)
    if sha(oracle_path) != read(root/'oracle_selected.sha256.json')['sha256']:
        raise RuntimeError('oracle hash receipt mismatch')
    if oracle['status'] != 'frozen' or oracle['protocol_sha256'] != lock['protocol_sha256']:
        raise RuntimeError('oracle is not frozen under the same protocol')
    aggregate = read(root/'aggregate.json')
    if aggregate['status'] != 'aggregate_complete' or aggregate['oracle_selection_sha256'] != sha(oracle_path):
        raise RuntimeError('main aggregate is not terminal under the frozen reference')
    for path, expected in cfg['source_sha256'].items():
        item = Path(path)
        if not item.is_absolute():
            item = repo/item
        if sha(item) != expected:
            raise RuntimeError('bound source drift: '+str(item))
    # No phase/result/truth inputs are opened before every gate above passes.
    inputs = {str(p):sha(p) for p in [root/'protocol.json', lock_path, *expected_paths]}
    phases = {}
    for case in cfg['cases']:
        for phase in ('planning', 'test'):
            folder = root/case/phase
            item, frozen = read(folder/'phase_ended.json'), read(folder/'fits_frozen.json')
            if item['protocol_sha256'] != lock['protocol_sha256'] or frozen['protocol_sha256'] != lock['protocol_sha256']:
                raise RuntimeError('phase protocol mismatch')
            if phase == 'planning' and sha(folder/'phase_ended.json') != oracle['planning_phase_sha256'][case]:
                raise RuntimeError('planning changed after reference selection')
            if phase == 'test' and item['oracle_selection_sha256'] != sha(oracle_path):
                raise RuntimeError('test did not use the frozen reference')
            for rows in (item['rows'], frozen['rows']):
                keys = [r['record_id'] for r in rows]
                if len(keys) != len(set(keys)):
                    raise RuntimeError('duplicate recorded observation')
            frozen_by = {r['record_id']:r for r in frozen['rows']}
            if set(frozen_by) != {r['record_id'] for r in item['rows']}:
                raise RuntimeError('phase/freeze observation inventories differ')
            for row in item['rows']:
                for field in ('fit_result_path', 'fit_result_sha256', 'physical_receipt_path', 'physical_receipt_sha256'):
                    if row.get(field) != frozen_by[row['record_id']].get(field):
                        raise RuntimeError('phase/freeze observation binding differs: '+field)
            phases[case,phase] = item
    sys.path.insert(0, str(repo))
    import numpy as np
    from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
    output.mkdir(exist_ok=True)
    running = output/'audit.lock'
    with running.open('x', encoding='utf-8') as stream:
        json.dump(dict(script_sha256=lock['script_sha256'], started_utc=now()), stream)
    rows, groups, topology_signatures = [], [], {}
    function_calls = dict(rooted_neighbor_joining=0, infer_classical_nj=0, QP=0, LP=0, AC=0,
                          fit_observed=0, WZZT_search=0, price_selection=0)

    def load(path):
        path = Path(path)
        inputs[str(path)] = sha(path)
        return read(path)

    def geometry(front, cal, labels, root_label):
        r, x = np.asarray(front['R'], float), np.asarray(front['X'], float)
        if r.shape != (len(labels), len(labels)) or x.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(x).all():
            raise ValueError('invalid retained raw frontend')
        if not float(cal['mu_r']) > 0 or not float(cal['mu_x']) > 0:
            raise ValueError('invalid retained frontend calibration')
        kernel = .75*r/float(cal['mu_r'])+.25*x/float(cal['mu_x'])
        rnj = rooted_neighbor_joining(kernel, np.diag(kernel), labels, root_label,
                    group_tolerance=.16*max(float(np.median(np.diag(kernel))),1e-12))
        function_calls['rooted_neighbor_joining'] += 1
        nj = infer_classical_nj(kernel, kernel, labels, root_label, collapse_factor=.16)
        function_calls['infer_classical_nj'] += 1
        singleton = {(int(v),) for v in labels}
        return {name:singleton|support_family(rooted_clades(tree.edges, root_label, labels))
                for name,tree in (('geometric_RNJ_RX75_0.16',rnj),('geometric_NJ_RX75_0.16',nj))}

    try:
        for (case,phase), ended in phases.items():
            case_cfg = cfg['cases'][case]
            plan = load(case_cfg['plan_lock']['path'])
            if inputs[case_cfg['plan_lock']['path']] != case_cfg['plan_lock']['sha256']:
                raise RuntimeError('plan lock hash mismatch')
            labels = list(plan['labels']);n = len(labels)
            paths = cfg['planning_paths'] if phase == 'planning' else cfg['test_paths']
            originals = {(r['path'],r['policy']):r for r in ended['rows']}
            for path in paths:
                for policy in cfg['policies']:
                    original = originals.get((path,policy))
                    record = dict(case=case, phase=phase, path=path, policy=policy,
                                  original_status=original.get('status') if original else 'absent',
                                  status='unresolved', models=None,
                                  original_joint_state_unchanged=True, replaces_main_record=False)
                    rows.append(record)
                    if original is None:
                        record['reason']='missing original observation record';continue
                    record['primary_joint_failure_retained'] = original['status'] != 'evaluated'
                    record['original_actual_payment']=original.get('actual_payment')
                    location = root/case/phase/'records'/original['record_id']
                    try:
                        receipt_path = Path(original.get('physical_receipt_path',location/'physical'/'events_receipt.json'))
                        if not receipt_path.exists():
                            record['reason']='physical receipt absent';continue
                        receipt = load(receipt_path)
                        if original.get('physical_receipt_sha256') and sha(receipt_path) != original['physical_receipt_sha256']:
                            raise RuntimeError('physical receipt changed after phase freeze')
                        truth_path = receipt_path.parent/'truth_evaluation.json'
                        if not truth_path.exists():
                            record['reason']='evaluator truth unavailable';continue
                        truth_data = load(truth_path)
                        if sha(truth_path) != receipt['private_files']['truth_evaluation.json']['sha256']:
                            raise RuntimeError('evaluator truth hash mismatch')
                        truth = support_family(truth_data['true_supports'])
                        signature = hashlib.sha256(json.dumps(sorted(truth),separators=(',',':')).encode()).hexdigest()
                        if case in topology_signatures and topology_signatures[case] != signature:
                            raise RuntimeError('physical topology changed within case')
                        topology_signatures[case]=signature
                        fit_path = original.get('fit_result_path')
                        if fit_path:
                            fitted = load(fit_path)
                            if sha(fit_path) != original['fit_result_sha256']:
                                raise RuntimeError('completed fit changed after freeze')
                        else:
                            request_path=location/'fit'/'fit_request.json'
                            if not request_path.exists():
                                record['reason']='no identifier attempt reference';continue
                            request=load(request_path)
                            cache=(root/case/'fit_cache'/request['cache_key']).resolve()
                            if cache.parent != (root/case/'fit_cache').resolve():
                                raise RuntimeError('cache reference outside case cache')
                            meta=load(cache/'cache_record.json')
                            if meta['identity']['algorithm_source_sha256'] != cfg['algorithm_source_sha256']:
                                raise RuntimeError('failed fit source identity mismatch')
                            fit_path=cache/'result.json'
                            if fit_path.exists():
                                fitted=load(fit_path)
                                if meta.get('result_sha256') and sha(fit_path) != meta['result_sha256']:
                                    raise RuntimeError('failed result hash mismatch')
                            else:
                                fit_path=cache/'checkpoint.json'
                                if not fit_path.exists():
                                    record['reason']='failed result/checkpoint both absent';continue
                                fitted=load(fit_path)
                        record.update(retained_fit_path=str(fit_path), retained_fit_sha256=sha(fit_path),
                                      original_fit_status=fitted.get('status'), original_fit_stage=fitted.get('stage'),
                                      original_exception=fitted.get('exception'),
                                      original_lp_time_limit_seconds=fitted.get('settings',{}).get('lp_time_limit_seconds'),
                                      nonoptimal_fixed_family_LPs=[dict(call_index=i,**v) for i,v in enumerate(fitted.get('LP_solves',[])) if v.get('status') != 2 or not v.get('accepted')],
                                      fixed_family_LP_calls_retained=len(fitted.get('LP_solves',[])))
                        if fitted.get('source_sha256') != cfg['algorithm_source_sha256']:
                            raise RuntimeError('retained fit source binding mismatch')
                        if list(fitted.get('terminal_labels',[])) != labels:
                            raise RuntimeError('retained frontend terminal order mismatch')
                        if fitted.get('root_label') != case_cfg.get('root_label',1):
                            raise RuntimeError('retained frontend root mismatch')
                        if 'raw' not in fitted.get('fronts',{}) or 'raw' not in fitted.get('calibrations',{}):
                            record['reason']='no retained raw frontend; no new QP allowed';continue
                        family=geometry(fitted['fronts']['raw'],fitted['calibrations']['raw'],labels,case_cfg.get('root_label',1))
                        record['models']={name:dict(error(f,truth),support_labels=[list(s) for s in sorted(f)]) for name,f in family.items()}
                        bank=[r for r in fitted.get('bank',[]) if r.get('config')=='raw_RX75_0.16']
                        if fitted.get('status')=='complete':
                            if len(bank)!=1:
                                raise RuntimeError('successful fit lacks unique raw_RX75_.16 bank source')
                            native=support_family(bank[0]['native_family'],labels)
                            record['successful_native_RNJ_replay_matches']=native==family['geometric_RNJ_RX75_0.16']
                            if not record['successful_native_RNJ_replay_matches']:
                                raise RuntimeError('recomputed RNJ disagrees with successful saved native family')
                        feasible=receipt.get('all_operation_checks_pass') is True and receipt.get('quote_qualification_pass') is True
                        record['status']='supplement_evaluated' if feasible else 'supplement_operation_failed'
                    except Exception as exc:
                        record.update(status='supplement_failed', exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            if phase=='test':
                for policy in cfg['policies']:
                    selected=[r for r in rows if r['case']==case and r['phase']==phase and r['policy']==policy]
                    by_method={}
                    for method in ('geometric_RNJ_RX75_0.16','geometric_NJ_RX75_0.16'):
                        known=[r['models'][method] for r in selected if r['status']=='supplement_evaluated']
                        by_method[method]=risk_interval(known,len(paths),n)
                    for method in ('multistart_original_rule','rnj_bank','baseline25'):
                        known=[r['models'][method] for r in ended['rows'] if r['policy']==policy and r['status']=='evaluated']
                        by_method[method]=risk_interval(known,len(paths),n)
                    groups.append(dict(case=case, policy=policy, models=by_method,
                        frozen_reference_selected=oracle['cases'][case]['selected_policy']==policy,
                        comparison_scope='same observations; geometric controls supplemented; original joint failures remain unresolved'))
        report=dict(status='supplement_complete', completed_utc=now(),script_sha256=lock['script_sha256'],
            protocol_sha256=lock['protocol_sha256'],oracle_selection_sha256=sha(oracle_path),
            primary_aggregate_sha256=sha(root/'aggregate.json'), rows=rows, test_groups=groups,
            calls=function_calls, original_joint_records_modified=0,
            successful_frontend_replays=sum(r.get('successful_native_RNJ_replay_matches') is True for r in rows),
            failed_joint_frontend_supplements=sum(r['status']=='supplement_evaluated' and r.get('primary_joint_failure_retained') for r in rows),
            no_oracle_or_price_reselection=True, no_original_failure_reclassification=True,
            topology_truth_used_only_for_post_freeze_scoring=True,
            input_sha256=inputs)
        # Verify every read artifact remained unchanged during this pure replay.
        for path,expected in inputs.items():
            if sha(path)!=expected:raise RuntimeError('input changed during supplement: '+path)
        write(output/'report.json',report)
        write(output/'audit_ended.json',dict(status='complete',report_sha256=sha(output/'report.json'),calls=function_calls))
        return dict(status='complete',report=str(output/'report.json'),rows=len(rows),calls=function_calls)
    except Exception as exc:
        write(output/'audit_ended.json',dict(status='failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),calls=function_calls))
        raise
    finally:
        running.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.root) if args.command=='freeze' else run(args.root),ensure_ascii=True))
