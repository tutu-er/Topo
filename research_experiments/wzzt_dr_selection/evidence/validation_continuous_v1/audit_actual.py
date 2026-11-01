"""Independent read-only audit of the new bounded continuous-price actual pilot.

All fit-freeze/terminal gates precede reads of any observation pickle, physical
truth, true supports or scored results.  No AC, identifier or solver is called.
Only scalar/discrete-support summaries are emitted; numerical R/X and all
trajectories stay in the existing private artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np


REPO = Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
HELPER = REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/effect_test/audit.py'
_spec = importlib.util.spec_from_file_location('independent_old_continuous_audit_helpers', HELPER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def stamp(value): return datetime.fromisoformat(value.replace('Z','+00:00'))


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def family(point):
    """The full requested family, not just its positive/pruned fitted support."""
    values=point.get('full_fit',{}).get('supports', point.get('native_family', point.get('supports',point.get('support_indices'))))
    return frozenset(tuple(sorted(map(int,s))) for s in values)


def digest_arrays(values):
    digest=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(value,dtype=float)
        digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def relative_noise(scene,event):
    pairs=[(scene['P_terminal'].to_numpy(),np.asarray(event['P_pu'])),
           (scene['Q_terminal'].to_numpy(),np.asarray(event['Q_pu'])),
           (scene['V_terminal'].to_numpy(),np.asarray(event['V_pu'])),
           (scene['V_root_observed'].to_numpy(),np.asarray(event['root_pu']))]
    return [(np.divide(a,b,out=np.zeros_like(a),where=np.abs(b)>1e-12),np.abs(b)>1e-12) for a,b in pairs]


def audit(root):
    root=Path(root).resolve();phase=root/'test'
    # Do not open scored results/truth/PQV until the controller fit-freeze exists.
    needed=[phase/'fits_frozen.json',phase/'scored.json']
    if any(not p.is_file() for p in needed):
        return dict(status='waiting_for_fits_frozen_and_scored',missing=[p.name for p in needed if not p.exists()],
                    truth_or_observations_opened=False,new_solver_calls=0,new_AC=0),None
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    master=read(root/'price_freeze.json')
    c=read(old.reference(master['protocol']))
    frozen=read(phase/'fits_frozen.json')
    check('fit_freeze_status',frozen['status']=='frozen' and frozen['truth_has_not_yet_been_opened_by_controller'])
    check('source_price_frozen',master['status']=='frozen' and master['prices_frozen_before_actual'])
    for worker in range(c['workers']):
        check(f'worker_{worker}_terminal',(phase/f'worker_{worker}_ended.json').is_file())
    # Reading scored is allowed only after the explicit freeze above.
    ledger=read(phase/'observations.json');scored=read(phase/'scored.json')
    check('freeze_observation_binding',frozen['observation_ledger_sha256']==sha(phase/'observations.json'))
    check('scored_freeze_binding',scored['fits_frozen_sha256']==sha(phase/'fits_frozen.json'))
    check('frozen_protocol_binding',ledger['protocol_sha256']==sha(root/'protocol.json'))
    check('main_case_preserved',c['case']=='flynn16' and c['case_key']=='main' and c['case_index']==0)
    check('same_history_budget_rows',c['hard_four_event_budget']==c['incentive_budget']==1.6 and c['original_history_rows']==[288,288] and c['future_rows']==[480,480])
    check('same_noise_bases',c['voltage_sigma']==.0005 and c['pq_sigma']==.005 and c['base_kw']==200. and c['root_label']==1)
    check('new_prelocked_paths',c['test_paths']==[1200,1201] and c['response_seed']==2026110133 and c['replicate_base']==470000)
    check('legacy_excitation_explicitly_obsolete',c['legacy_linear_excitation_obsolete'] is True)
    for name in ['planning_protocol','planning_summary','source_snapshot']:
        old.reference(master[name])
        check('master_'+name+'_hash',True)
    for filename,wanted in c['source_sha256'].items():
        check('source_hash_'+Path(filename).name,sha(filename)==wanted)
    plans={policy:read(old.reference(item)) for policy,item in c['plans'].items()}
    for name in ['history_packet','problem','source_metadata','historical_baseline_result']:
        old.reference(c[name]);check('input_hash_'+name,True)
    check('four_frozen_policies',c['policies']==['zero','rotation','continuous_validation','historical_frozen'])
    expected={(p,d) for p in c['policies'] for d in c['test_paths']}
    check('complete_record_grid',len(ledger['rows'])==len(scored['rows'])==ledger['planned']==scored['planned']==8 and {(r['policy'],r['path']) for r in ledger['rows']}==expected)
    for policy,plan in plans.items():
        check('obsolete_linear_'+policy,plan.get('legacy_linear_excitation_obsolete') is True)
        values=[]
        for e in range(4):
            x=np.asarray(plan['expected_amplitudes_by_event'][e],float)
            price=np.asarray(plan['prices_by_event'][e],float)
            check(f'{policy}_{e}_elasticity',np.allclose(x,.1*price,atol=1e-12,rtol=0.))
            if e in (0,2):values.extend(x.tolist())
        check(policy+'_vector_consistency',np.array_equal(values,plan['global_expected_amplitudes']))
        check(policy+'_hard_budget',plan['hard_budget_bound']<=1.6+1e-8)
    # All metadata gates passed.  Historical/data/truth packets may now be read.
    history=old.read_pickle(old.reference(c['history_packet']))['history']
    check('all_original_training_rows',sum(len(s['P_terminal']) for s in history['training'])==288)
    check('all_original_validation_rows',sum(len(s['P_terminal']) for s in history['validation'])==288)
    baseline=read(old.reference(c['historical_baseline_result']))
    comp=c['competition']
    requested_coarse=family(baseline['selection_candidates'][comp['selected_index']])
    target=tuple(c['target_labels'])
    check('declared_real_dispute',target==(201,202,203))
    problem=read(old.reference(c['problem']));jobs={j['key']:j for j in frozen['jobs']}
    rows=[];crn={};noise_crn={};natural_by_path={};unique=set()
    counts=dict(physical_complete=0,joint_complete=0,unique_joint_complete=0,failed_or_unresolved=0,
                physical_events=0,AC_timepoints=0,actual_LP_calls=0,wall_limited_complete=0)
    for source_row in scored['rows']:
        policy,path=source_row['policy'],source_row['path'];rid=source_row['record_id']
        row=dict(record_id=rid,policy=policy,path=path,status=source_row['status'],
                 actual_payment=source_row.get('actual_payment'),models=None,
                 current_coarse_exit_margin=None,current_coarse_exits_band=None)
        physical=Path(source_row['physical_output']);receipt=read(physical/'events_receipt.json')
        request_path=physical.parent/'request.json';request=read(request_path)
        check(rid+'_request_binding',sha(request_path)==source_row['request_sha256'])
        check(rid+'_frozen_before_future',stamp(c['created_utc'])<=stamp(master['created_utc'])<=stamp(request['created_utc'])<=stamp(receipt['started_utc']))
        check(rid+'_request_protocol_binding',request['protocol']['sha256']==sha(root/'protocol.json'))
        check(rid+'_plan_binding',request['plan_packet']==c['plans'][policy])
        for e in range(4):
            expected_factors=np.random.default_rng(np.random.SeedSequence([c['response_seed'],c['case_index'],path,e])).uniform(-1,1,2)
            check(rid+f'_event{e}_seed_factors',np.array_equal(request['factors_by_event'][e],expected_factors))
        if receipt['status']!='complete':
            row.update(physical_status=receipt['status'],known_actual_payment=receipt.get('known_actual_payment'),
                       unknown_payment_events=receipt.get('unknown_payment_events'))
            counts['failed_or_unresolved']+=1;rows.append(row);continue
        for file in ['packet.pkl','physical_truth.pkl','truth_evaluation.json']:
            check(rid+'_hash_'+file,sha(physical/file)==receipt['private_files'][file]['sha256'])
        packet=old.read_pickle(physical/'packet.pkl');private=old.read_pickle(physical/'physical_truth.pkl')
        # The old helper's first version expected this field on the receipt.
        # Use the immutable request as its source without changing any artifact.
        receipt_for_math=dict(receipt,incentive_budget=request['incentive_budget'])
        physical_math=old.audit_physics(plans[policy],request,receipt_for_math,private,packet,c['labels'],problem)
        row.update(actual_payment=physical_math['actual_payment'],expected_payment=physical_math['expected_payment'],
                   hard_payment=physical_math['hard_payment'],physical_status='complete')
        check(rid+'_independent_payment',abs(row['actual_payment']-source_row['actual_payment'])<=1e-9)
        counts['physical_complete']+=1;counts['physical_events']+=4;counts['AC_timepoints']+=384
        for e,event in enumerate(private['events']):
            cell=(path,e)
            data=tuple(np.asarray(event[k]) for k in ['natural_P_pu','natural_Q_pu','root_pu','factors'])
            if cell in crn:
                check(rid+f'_event{e}_CRN',all(np.array_equal(a,b) for a,b in zip(data,crn[cell],strict=True)))
            else:crn[cell]=data
            scene=packet['training'][e] if e<2 else packet['validation'][e-2]
            gains=relative_noise(scene,event)
            if cell in noise_crn:
                for channel,((a,mask),(b,oldmask)) in enumerate(zip(gains,noise_crn[cell],strict=True)):
                    both=mask&oldmask
                    check(rid+f'_event{e}_channel{channel}_noise_CRN',np.max(np.abs(a[both]-b[both]),initial=0.)<=1e-12)
            else:noise_crn[cell]=gains
        nat=[event[k] for event in private['events'] for k in ['natural_P_pu','natural_Q_pu','root_pu']]
        digest=digest_arrays(nat)
        if path in natural_by_path:check(rid+'_whole_path_CRN',digest==natural_by_path[path])
        else:natural_by_path[path]=digest
        if source_row['status']!='complete':
            counts['failed_or_unresolved']+=1;rows.append(row);continue
        fit=read(old.reference(source_row['result']));job=jobs[source_row['fit_key']]
        check(rid+'_terminal_fit_binding',job['status']=='complete' and job['result']['sha256']==source_row['result']['sha256'])
        check(rid+'_complete_rows',fit['training_rows']==fit['validation_rows']==480)
        settings=fit['settings']
        check(rid+'_unchanged_rules',settings['final_validation_fraction']==.03 and settings['resolution_factor']==.16 and settings['relative_gain_threshold']==.001 and settings['coefficient_bound']==2.)
        check(rid+'_no_final_refit',settings['final_refit'] is False)
        for split in ['training','validation']:
            full=history[split]+packet[split]
            check(rid+'_'+split+'_all_rows',sum(len(s['P_terminal']) for s in full)==480)
            hashed,frames=old.own_input_hash(full)
            check(rid+'_'+split+'_input_hash',hashed==fit['input_sha256'][split])
            if split=='training':train_frames=frames
            else:valid_frames=frames
        truth=read(physical/'truth_evaluation.json')
        check(rid+'_zero_weak_truth_retained',truth['zero_impedance_merged'] and truth['arbitrarily_weak_positive_lines_retained'])
        own_models,diag=old.audit_selection(fit,truth['true_supports'],valid_frames)
        for name in old.MODELS:
            for metric in ['full_errors','full_recovery']:
                check(rid+'_'+name+'_'+metric,own_models[name][metric]==source_row['models'][name][metric])
        candidates=fit['selection_candidates']
        values=[];matches=[]
        for index,point in enumerate(candidates):
            v=old.own_loss(point,valid_frames);j=old.own_loss(point,train_frames)
            check(rid+f'_candidate{index}_validation_loss',abs(v-point['validation_mae'])<=1e-11)
            check(rid+f'_candidate{index}_training_loss',abs(j-point['objective'])<=1e-11)
            values.append(v)
            if family(point)==requested_coarse:matches.append(index)
        minimum=min(values)
        if matches:
            lowest=min(values[i] for i in matches);highest=max(values[i] for i in matches)
            margin=lowest-1.03*minimum-1e-12
            row.update(current_coarse_exit_margin=margin,current_coarse_exits_band=margin>0,
                       current_coarse_best_validation=lowest,current_coarse_max_validation=highest,
                       current_coarse_eligible_versions=sum(values[i]<=1.03*minimum+1e-12 for i in matches))
        model=fit['models']['multistart_original_rule'];labels=fit['terminal_labels']
        raw=model['selected_uncontracted'];true_supports={tuple(sorted(s)) for s in truth['true_supports']}
        check(rid+'_target_is_true',target in true_supports)
        row.update(models=own_models,diagnostics=diag,
            actual_candidate_count=len(candidates),actual_global_validation_minimum=minimum,
            historical_current_coarse_requested_blocks=len(requested_coarse),
            current_coarse_matching_candidates=len(matches),
            current_coarse_comparison_available=bool(matches),
            target_in_selected_raw=target in old.supports(raw,labels),
            target_in_final=target in old.supports(model,labels),
            full_errors=own_models['multistart_original_rule']['full_errors'],
            full_recovery=own_models['multistart_original_rule']['full_recovery'],
            raw_full_errors=own_models['multistart_original_rule']['raw_full_errors'],
            selected_origin=model.get('selected_origin'))
        counts['joint_complete']+=1;counts['wall_limited_complete']+=int(diag['wall_limited'])
        if source_row['fit_key'] not in unique:
            unique.add(source_row['fit_key']);counts['unique_joint_complete']+=1;counts['actual_LP_calls']+=diag['lp_calls']
        rows.append(row)
    check('no_reused_background_among_completed_paths',len(set(natural_by_path.values()))==len(natural_by_path))
    tables=[];paired=[];cap=4*len(c['labels'])-2
    for policy in c['policies']:
        rr=[r for r in rows if r['policy']==policy];ok=[r for r in rr if r['models'] is not None]
        errors=sum(r['full_errors'] for r in ok);recover=sum(r['full_recovery'] for r in ok);unresolved=len(rr)-len(ok)
        tables.append(dict(policy=policy,denominator=len(rr),completed=len(ok),unresolved=unresolved,
            final_errors=[next(r for r in rr if r['path']==path).get('full_errors') for path in c['test_paths']],
            full_error_sum_known=errors,mean_full_errors=errors/len(rr) if not unresolved else None,
            mean_error_bounds=[errors/len(rr),(errors+cap*unresolved)/len(rr)],
            recoveries=recover,recovery_fraction_bounds=[recover/len(rr),(recover+unresolved)/len(rr)],
            actual_payment_known_sum=sum(r['actual_payment'] for r in rr if r['actual_payment'] is not None),
            mean_actual_payment=sum(r['actual_payment'] for r in rr)/len(rr) if all(r['actual_payment'] is not None for r in rr) else None,
            current_coarse_comparison_available=sum(r.get('current_coarse_comparison_available',False) for r in rr),
            current_coarse_exits=sum(r.get('current_coarse_exits_band') is True for r in rr),
            raw_target_kept=sum(r.get('target_in_selected_raw') is True for r in rr),
            final_target_kept=sum(r.get('target_in_final') is True for r in rr)))
    for path in c['test_paths']:
        zero=next(r for r in rows if r['policy']=='zero' and r['path']==path)
        for row in [r for r in rows if r['path']==path and r['policy']!='zero']:
            paired.append(dict(path=path,policy=row['policy'],zero_status=zero['status'],paid_status=row['status'],
                full_error_improvement_zero_minus_paid=(zero['full_errors']-row['full_errors']) if zero['models'] is not None and row['models'] is not None else None,
                actual_payment=row['actual_payment']))
    check('denominator_preserved',len(rows)==8)
    report=dict(status='complete',planned_records=8,policies=c['policies'],independent_natural_paths=2,
        completed_distinct_natural_backgrounds=len(natural_by_path),
        path_denominator=c['test_paths'],rows=rows,tables=tables,paired=paired,totals=counts,
        scalar_check_count=len(checks),passed_checks=sum(r['passed'] for r in checks),checks=checks,
        audit_source_sha256=sha(__file__),reused_independent_helper_sha256=sha(HELPER),
        actual_protocol_sha256=sha(root/'protocol.json'),price_freeze_sha256=sha(root/'price_freeze.json'),
        fits_frozen_sha256=sha(phase/'fits_frozen.json'),new_solver_calls=0,new_AC=0,
        quote_feasibility_status=c['quote_feasibility_status'],
        criterion_scope='current historical requested complete coarse family vs actual global validation minimum; unavailable if that family was not generated; not all no-C families',
        probability_scope='90pct excitation requirement only; this two-path diagnostic does not certify90pct or topology recovery',
        physical_legacy_linear_g_obsolete=True,
        boundaries=['Controller ordering is hash/receipt-audited, not an operating-system access-log claim.',
                    'Actual error evaluation retains all strictly positive weak lines; no truth is used for planning.',
                    'Historical-family aliases used by planning do not create new physical trees.',
                    'Original physical checker does not enforce branch ampacity.'])
    private=dict(actual_protocol=ref(root/'protocol.json'),price_freeze=ref(root/'price_freeze.json'),
                 observation_ledger=ref(phase/'observations.json'),fits_frozen=ref(phase/'fits_frozen.json'),
                 scored=ref(phase/'scored.json'),planning_protocol=c['planning_protocol'],planning_summary=c['planning_summary'])
    return report,private


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('audit output exists; retain it')
    try:
        report,private=audit(args.root)
    except Exception as error:
        report=dict(status='audit_failed',error_type=type(error).__name__,message=str(error),
                    traceback=traceback.format_exc(),audit_source_sha256=sha(__file__),new_solver_calls=0,new_AC=0)
        private=None
    if report['status'].startswith('waiting'):
        print(json.dumps(report));return 3
    if private is not None:
        private_path=args.output.with_name(args.output.stem+'_private_manifest.json')
        write_new(private_path,private);report['private_manifest_sha256']=sha(private_path)
    write_new(args.output,report)
    print(json.dumps({key:report.get(key) for key in ['status','scalar_check_count','passed_checks','totals','tables','message']},ensure_ascii=False))
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
