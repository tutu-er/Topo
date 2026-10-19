"""Predeclared H-only Taylor stress selection and a thin nonlinear audit driver.

This is a conditional diagnostic, outside the frozen/calibrated policy. It
does not inspect truth, evaluator seed, realized future observations, or final
joint-fit likelihood. An identical prior H/action nonlinear audit can be reused
only after input, source, and optimization configuration checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter


METHODS=('constant_fixed','refit_milp','fixed_mle_milp')
DOMAINS=('same_local','full_physical')
RULE='max_finite_selection_actual_rollback_joint_taylor_norm'


def _dump(path,value):
    # Real AC diagnostics contain nested ndarrays and NumPy scalars.
    from research_experiments.mobile_ev_mle.adaptive_result_audit import _dump as research_dump
    research_dump(path,value)


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()
def _load(path):return json.loads(path.read_text(encoding='utf-8'))


def semantic_pair(name):
    """Ignore candidate ordering, retain centre indices/scenario/source mode."""
    pattern=r'(.+)_centre_(\d+)__(.+)_centre_(\d+)__(.+)__(fixed|local_refit)'
    match=re.fullmatch(pattern,name)
    if match is None:raise ValueError('unsupported saved pair name')
    a,ia,b,ib,scenario,mode=match.groups()
    return (tuple(sorted(((a,int(ia)),(b,int(ib))))),scenario,mode)


def selection_view(trial):
    """Only the selector's H/predicted-AC rollback fields enter selection."""
    rows=[];failed=[]
    for method,details in sorted(trial['selection']['methods'].items()):
        for index,item in enumerate(details.get('actual_rollback',[])):
            error=item.get('joint_taylor_error_norm')
            if not isinstance(error,(int,float)) or not math.isfinite(error) or error<0:
                failed.append(dict(method=method,rollback_index=index,pair=item.get('pair_name'),
                    reason='missing_or_invalid_Taylor_norm',physically_feasible=item.get('physically_feasible'),
                    saved_error=item.get('error')))
                continue
            pair=item.get('pair_name')
            if not isinstance(pair,str):
                failed.append(dict(method=method,rollback_index=index,reason='missing_pair_name'))
                continue
            semantic_pair(pair)
            rows.append(dict(method=method,rollback_index=index,pair=pair,
                joint_taylor_error_norm=float(error),physically_feasible=item.get('physically_feasible')))
    rows.sort(key=lambda item:(-item['joint_taylor_error_norm'],item['method'],item['pair']))
    return dict(rollback_rows=rows,failure_ledger=failed,largest_computable=None if not rows else rows[0],
        required_rollback_count=len(rows)+len(failed),computable_rollback_count=len(rows),
        all_required_Taylor_norms_available=not failed)


def _nlp():
    from research_experiments.mobile_ev_mle import nonlinear_pair_audit
    return nonlinear_pair_audit


def _inputs(file,nlp):
    history,actions=nlp.load_history_actions(file)
    return history,actions,dict(history_hash=nlp.history_input_hash(history),
        actions_hash=_digest({name:dict(u_kw=actions[name]['u_kw'].tolist(),
            q_kw=actions[name]['q_kw'].tolist(),sites=actions[name]['sites'].tolist()) for name in sorted(actions)}),
        method_control_hashes={name:nlp._hash(actions[name]['u_kw']) for name in METHODS})


def prepare_manifest(input_folder,out,*,count=2,expected_histories=6):
    """Freeze the available-error rule and all six denominators before solving."""
    input_folder,out=Path(input_folder),Path(out)
    if out.exists():raise ValueError('predeclared selection manifest already exists; never overwrite')
    files=sorted(input_folder.glob('trial_[0-9]*.json'))
    if len(files)!=expected_histories:raise ValueError('predeclared history denominator mismatch')
    if not isinstance(count,int) or not 1<=count<=len(files):raise ValueError('invalid selection count')
    nlp=_nlp();rank=[];invalid=[]
    for file in files:
        trial=_load(file);view=selection_view(trial)
        try:
            _,_,identity=_inputs(file,nlp)
            row=dict(input_trial_filename=file.name,trial_json_sha256=_sha(file),
                history_action_identity=identity,selection_whitelist_sha256=_digest(view),**view)
            if view['largest_computable'] is None:invalid.append(row)
            else:rank.append(row)
        except (ValueError,KeyError,RuntimeError,OSError) as error:
            invalid.append(dict(input_trial_filename=file.name,trial_json_sha256=_sha(file),**view,
                input_failure_type=type(error).__name__,input_failure=str(error)))
    rank.sort(key=lambda row:(-row['largest_computable']['joint_taylor_error_norm'],row['input_trial_filename']))
    if len(rank)<count:raise ValueError('too few computable histories; failure ledger cannot justify selection')
    result=dict(version=1,selection_rule=RULE,input_folder=str(input_folder.resolve()),
        required_history_count=expected_histories,selected_history_count=count,
        selected_file_names=[row['input_trial_filename'] for row in rank[:count]],
        selected_histories=rank[:count],all_history_ranking_by_available_norm=rank,
        required_history_input_failures=invalid,
        unavailable_norms_are_not_zero=True,
        incomplete_history_error_ordering=any(not row['all_required_Taylor_norms_available'] for row in rank),
        statistical_sample=False,statistical_coverage_claim=False,conditioned_on_large_prediction_Taylor_error=True,
        selector_uses_truth=False,selector_uses_future_observations=False,selector_uses_final_likelihood=False,
        hashes_of_whole_files_are_integrity_only=True,
        primary_frozen_policy_modified=False,
        mathematical_scope='joint H and predicted future residual Taylor norm at saved affine witnesses',
        worst_pair_scope='semantic topology/centre labels; audit rebuilds H-only centres, not exact old expansion point',
        computation=dict(starts=2,max_nfev=300,extra_pairs=2,methods=METHODS,domains=DOMAINS),
        reuse_allowed=True,manifest_precedes_new_nonlinear_solves=True)
    out.parent.mkdir(parents=True,exist_ok=True);_dump(out,result)
    return result


def validate_reuse(folder,file,history,actions,identity,nlp,config):
    """Reuse only the same H/actions, byte-identical current sources and starts."""
    folder=Path(folder);saved_cfg=_load(folder/'config.json')
    if saved_cfg['source_sha256']!=nlp.source_hashes():
        raise ValueError('reuse rejected: nonlinear source hashes differ')
    for key in ('starts','max_nfev','extra_pairs'):
        if saved_cfg[key]!=config[key]:raise ValueError('reuse rejected: optimization configuration differs')
    if tuple(saved_cfg['methods'])!=METHODS or tuple(saved_cfg['domains'])!=DOMAINS:
        raise ValueError('reuse rejected: method/domain scope differs')
    archived=folder/f'audit_{file.stem}.json';result=_load(archived)
    if result['history_hash']!=identity['history_hash'] or result['input_trial_filename']!=file.name:
        raise ValueError('reuse rejected: historical input identity differs')
    rows=result['records']
    if len(rows)!=48 or len(result['diagnostic_pair_names'])!=8:
        raise ValueError('reuse rejected: required pair denominator differs')
    for method in METHODS:
        group=[row for row in rows if row['method']==method]
        if len(group)!=16:raise ValueError('reuse rejected: method denominator differs')
        for row in group:
            if row['control_hash']!=identity['method_control_hashes'][method]:
                raise ValueError('reuse rejected: control hash differs')
            if len(row.get('restarts',[]))!=config['starts']:
                raise ValueError('reuse rejected: restart denominator differs')
    return result,dict(status='reused_not_new_run',source_folder=str(folder.resolve()),
        original_audit_json_sha256=_sha(archived),original_config_sha256=_sha(folder/'config.json'),
        identical_history_actions=True,identical_source_hashes=True,identical_optimization_configuration=True)


def check_worst_pair_scope(result,selected):
    required=semantic_pair(selected['largest_computable']['pair'])
    actual={semantic_pair(name) for name in result.get('diagnostic_pair_names',[])}
    if required not in actual:
        raise ValueError('required worst-Taylor semantic pair missing; preserve failure, do not call scope complete')
    return dict(required_saved_pair=selected['largest_computable']['pair'],
        semantic_pair_present=True,centre_zero_base_pair=all(index==0 for _,index in required[0]),
        old_expansion_point_reused=False,audit_centres_rebuilt_from_H=True)


def run_stress(manifest_file,out,*,reuse_folder=None):
    """Run the predeclared shortlist without changing the original NLP module."""
    began=perf_counter();manifest_file,out=Path(manifest_file),Path(out)
    manifest=_load(manifest_file);source=Path(manifest['input_folder']);nlp=_nlp()
    if manifest['selection_rule']!=RULE or manifest['computation']['starts']!=2:
        raise ValueError('unsupported predeclared stress rule')
    if out.exists():raise ValueError('use a fresh Taylor stress output directory')
    out.mkdir(parents=True);cfg=manifest['computation']
    config=dict(**cfg,source_sha256=nlp.source_hashes(),selection_manifest_sha256=_sha(manifest_file),
        selection_manifest=manifest,driver_sha256=_sha(Path(__file__)),reuse_folder=None if reuse_folder is None else str(reuse_folder),
        statistical_sample=False,primary_frozen_policy_modified=False)
    _dump(out/'config.json',config);results=[];new_calls=0
    for selected in manifest['selected_histories']:
        start=perf_counter();file=source/selected['input_trial_filename']
        try:
            if _sha(file)!=selected['trial_json_sha256']:raise ValueError('selected input JSON changed after predeclaration')
            if _digest(selection_view(_load(file)))!=selected['selection_whitelist_sha256']:
                raise ValueError('selected H/prediction Taylor fields changed')
            history,actions,identity=_inputs(file,nlp)
            if identity!=selected['history_action_identity']:raise ValueError('selected H/action arrays changed')
            reused=reuse_folder is not None and (Path(reuse_folder)/f'audit_{file.stem}.json').exists()
            if reused:
                result,provenance=validate_reuse(reuse_folder,file,history,actions,identity,nlp,cfg)
            else:
                result=nlp.audit_history(history,actions,starts=cfg['starts'],
                    max_nfev=cfg['max_nfev'],extra_pairs=cfg['extra_pairs'])
                provenance=dict(status='new_nonlinear_run',source_folder=None)
                new_calls+=sum(row.get('ac_predict_calls',0) for row in result['records'])
            result['stress_worst_pair_scope']=check_worst_pair_scope(result,selected)
            result['stress_execution_provenance']=provenance
            result['stress_execution_seconds_current_run']=perf_counter()-start
            result['reused_not_new_run']=reused
        except Exception as error:
            result=dict(status='required_stress_history_failure',records=[],history_hash=None,
                error_type=type(error).__name__,error=str(error),seconds=perf_counter()-start,reused_not_new_run=False)
        result['input_trial_filename']=file.name
        _dump(out/f'audit_{file.stem}.json',result);results.append(result)
        _dump(out/'progress.json',dict(completed=len(results),selected_file_count=manifest['selected_history_count'],
            last_file=file.name,diagnostic_not_statistical_sample=True))
        print(json.dumps(dict(file=file.name,status=result['status'],reused_not_new_run=result['reused_not_new_run'],
            execution_seconds=result.get('stress_execution_seconds_current_run',result['seconds']))),flush=True)
    summary=dict(history_count=len(results),selected_file_count=manifest['selected_history_count'],
        methods=METHODS,domains=DOMAINS,config=config,
        histories=[dict(input_trial_filename=row['input_trial_filename'],history_hash=row['history_hash'],
            status=row['status'],seconds=row['seconds'],reused_not_new_run=row['reused_not_new_run']) for row in results],
        new_run_history_count=sum(not row['reused_not_new_run'] and 'stress_execution_provenance' in row for row in results),
        reused_history_count=sum(row['reused_not_new_run'] for row in results),
        required_history_failure_count=sum(row['status']=='required_stress_history_failure' for row in results),
        selected_audit_inner_AC_calls_including_reuse=sum(x.get('ac_predict_calls',0) for row in results for x in row['records']),
        current_run_new_inner_AC_calls=new_calls,current_run_wall_seconds=perf_counter()-began,
        recorded_seconds_include_old_reused_audit_time=True,statistical_sample=False,
        primary_frozen_policy_modified=False,failure_denominator_preserved=True)
    _dump(out/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--input',type=Path,required=True)
    prepare.add_argument('--manifest',type=Path,required=True)
    run=sub.add_parser('run');run.add_argument('--manifest',type=Path,required=True)
    run.add_argument('--out',type=Path,required=True);run.add_argument('--reuse',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare_manifest(args.input,args.manifest)
    else:run_stress(args.manifest,args.out,reuse_folder=args.reuse)


if __name__=='__main__':main()
