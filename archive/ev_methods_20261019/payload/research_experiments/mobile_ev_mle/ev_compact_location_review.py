"""Thin independent branch/SSE replay adapter for compact location evidence.

No new fit, selector or experiment generator is used. The stage-16 reviewer
supplies separate branch-current AC arithmetic and complete residual replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import numpy as np
from . import ev_original_location_review as base
from .information_fit_recovery import read_json,write_json

METHODS=('refined_v2g_mobile','capped_v2g_mobile')


def replay_views(saved,parent):
    """Adapt labels/config, without altering frozen parent or ablation bytes."""
    p=dict(parent,config=deepcopy(parent['config']['physical']))
    device=(parent.get('operations_config') or {}).get('device') or {}
    p['config']['device']['power_kw']=device.get('power_kw',parent['config']['command_power_kw'])
    s=dict(saved,counts={k:saved['counts'][k] for k in ('candidate_fit_calls','fit_AC_calls','forward_AC_calls')})
    s['conditions']=[]
    for condition in saved['conditions']:
        c=dict(condition)
        if condition.get('resource_scope') is not None:
            scope=dict(condition['resource_scope'])
            base.require(scope.get('equal_service_comparison') is False,'electrical scope mislabelled equal-service')
            # Legacy wording used by the reused reviewer means no blanket
            # executable/equal-service claim. Compact keeps separate checks.
            scope['remap_has_no_operation_feasibility_or_equal_service_claim']=True
            c['resource_scope']=scope
        s['conditions'].append(c)
    return s,p


def signed_mobile_config(config,method):
    base.require(method in METHODS,'unexpected compact primary method')
    return base.service_config(config,'v2g_mobile')


def replay_history(saved,parent,expected_hash):
    view,p=replay_views(saved,parent)
    audit,candidates=base.replay_history(view,p,expected_hash,methods=METHODS,
                                      service_config_builder=signed_mobile_config)
    base.require(saved['statistical_threshold_applied'] is False,'old policy calibration applied to new remap process')
    base.require(saved['counts']['actual_returned_fit_calls']==audit['counts']['candidate_fit_calls'],'extra actual fit count')
    responses=[];optimality=[];new_fit_states=0
    for condition in saved['conditions']:
        base.require(condition.get('statistical_threshold_applied') is False
                     and condition.get('calibrated_rejection_decision') is None,'unjustified remap rejection decision')
        physical={}
        for key in ('p','q','v','master_p','master_q'):
            difference=base.arr(condition['original']['physical'][key])-base.arr(condition['remap']['physical'][key])
            physical[key]=dict(original_minus_remap=difference.tolist(),max_abs=float(np.max(abs(difference))),
                               RMS=float(np.sqrt(np.mean(difference**2))))
        responses.append(dict(seed=saved['seed'],method=condition['method'],
            true_mean_difference=physical,EV_total_waveform_error_kw=condition['waveform_max_abs_difference_kw'],
            master_P_max_abs_difference_kw=physical['master_p']['max_abs']*p['config']['base_kw'],
            master_Q_max_abs_difference_kvar=physical['master_q']['max_abs']*p['config']['base_kw'],
            voltage_max_abs_difference_pu=physical['v']['max_abs'],
            root_meter_recomputed_not_fixed=True))
        for label in ('original','remap'):
            variant=condition[label]
            for kind in ('noisy','noiseless'):
                detail=variant[kind];losses={f['name']:float(f['loss']) for f in detail['fits']}
                pairs=detail['pairwise_half_SSE_differences']
                base.require(len(pairs)==12 and len({(r['a'],r['b']) for r in pairs})==12,'twelve signed differences')
                for pair in pairs:base.close(pair['value'],.5*(losses[pair['b']]-losses[pair['a']]),'signed pair cost')
                fit=next(f for f in detail['fits'] if f['name']==next(r['true_candidate'] for r in candidates if r['method']==condition['method']))
                for key in ('r','x'):
                    expected=float(np.linalg.norm(base.arr(fit[key])-base.arr(parent['truth'][key]))/np.linalg.norm(base.arr(parent['truth'][key])))
                    base.close(detail['true_candidate_'+key.upper()+'_relative'],expected,'true-candidate parameter error')
                new=label=='remap' and not detail['cache_hit']
                new_fit_states+=len(detail['fits']) if new else 0
                for fit in detail['fits']:
                    optimality.append(dict(seed=saved['seed'],method=condition['method'],variant=label,kind=kind,
                        candidate=fit['name'],success=bool(fit['success']),status=fit.get('status'),
                        nfev=fit.get('nfev'),optimality=fit.get('optimality'),new_fit_state=new))
        base.close(condition['delta_half_M'],condition['delta_M']/2,'paired half-M')
        scope=condition['resource_scope'];base.require(scope['original_operation_feasible'],'original execution invalid')
        base.close(scope['terminal_minus_drive_identity_error_kwh'],0.,'terminal/drive identity')
        base.require(scope['original_curve_matches_operation'],'inherited operation mismatches final action')
    base.require(new_fit_states==saved['counts']['actual_returned_fit_calls'],'new fit-state coverage')
    audit.update(new_fit_states_replayed=new_fit_states,physical_responses=responses,solver_state_metadata=optimality)
    return audit,candidates


def _write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def review_directory(directory,output=None):
    directory=Path(directory);freeze=read_json(directory/'freeze.json');parentdir=Path(freeze['parent_directory'])
    sources=base.source_audit(directory,freeze)
    base.require(sources['current_source_matches'],'ablation numerical source changed')
    base.require(base.file_hash(parentdir/'freeze.json')==freeze['parent_freeze_sha256'],'parent freeze changed')
    identities={int(r['raw_name'].split('_')[1].split('.')[0]):r for r in freeze['parent_trial_hashes']}
    histories=[];errors=[];conditions=[];candidates=[];input_hashes={}
    for seed in freeze['seeds']:
        path=directory/f'history_{seed}.json.gz'
        if not path.exists():
            conditions.extend(dict(seed=seed,method=m,saved_status='missing_history',audited=False) for m in METHODS)
            continue
        saved=read_json(path);identity=identities[seed];parentpath=parentdir/identity['name']
        base.require(base.file_hash(parentpath)==identity['sha256'],'parent trial changed')
        base.require(saved['source_sha256']==freeze['source_sha256'] and saved['parameter_sha256']==freeze['parameter_sha256'],'location freeze differs')
        input_hashes[path.name]=base.file_hash(path)
        base.require([r['method'] for r in saved['conditions']]==list(METHODS),'method denominator')
        record_indices=[]
        for c in saved['conditions']:
            record_indices.append(len(conditions));conditions.append(dict(seed=seed,method=c['method'],saved_status=c['status'],audited=False))
        try:
            h,rows=replay_history(saved,read_json(parentpath),identity['sha256'])
            histories.append(h);candidates.extend(rows)
            for index in record_indices:conditions[index]['audited']=True
        except (ValueError,KeyError,RuntimeError,IndexError,TypeError) as exc:
            errors.append(dict(seed=seed,error=type(exc).__name__+':'+str(exc),
                retained_conditions=[r['method'] for r in saved['conditions']],
                scope='failed/incomplete conditions remain in planned denominators; unverified replay is explicit'))
    counts={k:sum(h['counts'][k] for h in histories) for k in ('candidate_fit_calls','fit_AC_calls','forward_AC_calls')}
    all_complete=len(histories)==len(freeze['seeds']) and not errors
    if all_complete:
        summary=read_json(directory/'summary.json')
        for k,value in counts.items():base.require(summary['counts'][k]==value,'summary call counts')
    responses=[r for h in histories for r in h['physical_responses']]
    metadata=[r for h in histories for r in h['solver_state_metadata']]
    finite=[float(r['optimality']) for r in metadata if r['optimality'] is not None]
    new_finite=[float(r['optimality']) for r in metadata if r['new_fit_state'] and r['optimality'] is not None]
    parent=read_json(parentdir/'freeze.json');physical=parent['config']['physical']
    result=dict(success=all_complete,phase=freeze['phase'],planned_histories=freeze['planned_histories'],
        audited_histories=len(histories),planned_conditions=freeze['planned_conditions'],
        available_condition_records=sum(c['saved_status']!='missing_history' for c in conditions),
        audited_conditions=sum(c['audited'] for c in conditions),condition_records=conditions,review_errors=errors,
        parent_policy_id=freeze['parent_policy_id'],seeds=freeze['seeds'],history_sha256=input_hashes,
        source_audit=sources,counts=counts,total_new_counted_AC_calls=counts['fit_AC_calls']+counts['forward_AC_calls'],
        saved_fit_states_replayed=sum(h['saved_fit_states_replayed'] for h in histories),
        new_fit_states_replayed=sum(h['new_fit_states_replayed'] for h in histories),
        physical_states_replayed=sum(h['physical_states_replayed'] for h in histories),
        new_forward_states_replayed=counts['forward_AC_calls'],
        max_abs_white_error=max((h['max_abs_white_error'] for h in histories),default=None),
        exact_reused_conditions=sum(not c['new_remap'] for h in histories for c in h['conditions']),
        replay_coverage='counts cover successfully audited histories; incomplete/error histories remain explicit and make success false',
        physical_responses=responses,solver_state_metadata=metadata,
        solver_metadata_summary=dict(all_success=all(r['success'] for r in metadata),
            saved_states=len(metadata),max_reported_optimality=max(finite,default=None),
            new_saved_states=sum(r['new_fit_state'] for r in metadata),max_new_reported_optimality=max(new_finite,default=None)),
        initialization_contract=dict(initial='same candidate H-fit R/X and historical states; future states from their own meters',
            multistart=physical['fit']['multistart'],max_nfev=physical['fit']['max_nfev'],tol=physical['fit']['tol'],
            linear_solver=physical['fit']['linear_solver'],same_candidate_seed_rule=True,
            noisy_seed='parent policy seed + 802 + candidate index',noiseless_seed='parent policy seed + 1202 + candidate index',
            second_start='same seeded R/X perturbation and box clipping; corresponding future states may differ',
            global_MLE_claim=False,basis='frozen parent evaluator and reused fit helper; optimizer is not rerun'),
        statistical_threshold_applied=False,formal_rejection_evaluation_histories=0,
        equal_service_comparison=False,new_fit_or_selector_calls_by_reviewer=0,
        independent_unit='one H; two methods are paired, not two independent histories',
        reviewer=dict(file=Path(__file__).name,sha256=base.file_hash(__file__),
            reused_reviewer=Path(base.__file__).name,reused_sha256=base.file_hash(base.__file__),science_source=False),
        histories=histories)
    _write_csv(directory/'physical_response_differences.csv',responses)
    _write_csv(directory/'fit_solver_metadata.csv',metadata)
    if output:write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=review_directory(args.directory,args.output)
    print(json.dumps({k:result[k] for k in ('success','audited_histories','audited_conditions','counts','saved_fit_states_replayed',
        'new_fit_states_replayed','physical_states_replayed','new_forward_states_replayed','max_abs_white_error','review_errors')}))


if __name__=='__main__':main()
