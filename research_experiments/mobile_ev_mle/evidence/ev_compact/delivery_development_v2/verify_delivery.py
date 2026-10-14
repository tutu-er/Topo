"""Saved raw/slim byte and scientific-field comparison; no AC or optimization."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from research_experiments.mobile_ev_mle import ev_compact_analysis as a
from research_experiments.mobile_ev_mle import ev_compact_delivery as d

ROOT=Path(__file__).resolve().parent
ANALYSIS_FIELDS=('status','true_candidate_joint_SSE','minimum_joint_SSE','actual_M','actual_half_M',
    'correct_min_loss','noiseless_M','noiseless_half_M','R_relative','X_relative',
    'predicted_mean','predicted_worst','predicted_cap20_mean','selected_origin','raw_best_origin',
    'initial_best_score','selected_task_score','inheritance_check','proxy_status','proxy_lower','proxy_upper',
    'proxy_gap','proxy_completed','proxy_unfinished','main_MILP_calls','new_MILP_proposals',
    'selected_is_new_MILP_proposal','grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes','moves',
    'terminal_energy_kwh','command_power_limit_kw','forecast_tracking_within_hardware')
MASTER_FIELDS=('u','operation_x','objective_upper','objective_lower','surrogate_incumbent','solver_best_bound',
    'coarse_box_upper','bound_consistent','rejected_solver_upper','status','message','solver_success','solver_mip_gap',
    'solver_nodes','backend','backend_details','scale','numeric_guard','n_variables','n_rows','n_binary',
    'n_operation_binary','n_shared_square_binary','n_residual_binary','n_product_binary','seconds')
EVALUATION_FIELDS=('u','origin','score_lower','score_upper','mean_score_interval','worst_score_interval',
    'pair_score_lower','pair_score_upper','capped_pair_score_lower','capped_pair_score_upper',
    'capped_mean_score_interval','capped_worst_score_interval','operation_validation','all_qp_converged')


def verify():
    counts=Counter();errors=[];files=[]
    def check(category,label,left,right):
        counts[category]+=1
        if left!=right:
            digest=lambda x:hashlib.sha256(x).hexdigest() if isinstance(x,bytes) else a.digest(x)
            errors.append(dict(category=category,field=label,left_sha256=digest(left),right_sha256=digest(right)))
    for label in ('development_ensemble_v2','development_center_v2'):
        target=ROOT/label;manifest=d.read_json(target/'delivery_manifest.json');source=Path(manifest['source_directory'])
        check('schema',label,manifest['artifact_schema'],d.SCHEMA)
        check('phase',label,manifest['phase'],'development')
        check('complete_batch',label,manifest['missing_seeds'],[])
        for entry in manifest['source_files']:
            check('frozen_file_bytes',label+'/'+entry['name'],(source/entry['name']).read_bytes(),(target/entry['name']).read_bytes())
        for entry in manifest['trial_files']:
            original=Path(entry['raw_path']);slim_path=target/entry['slim_file']
            raw_hash=d.sha256_file(original);check('raw_sha256',original.name,raw_hash,entry['raw_sha256'])
            check('slim_sha256',slim_path.name,d.sha256_file(slim_path),entry['slim_sha256'])
            raw=d.read_json(original);slim=d.read_json(slim_path)
            check('slim_schema',slim_path.name,slim['artifact_schema'],d.SCHEMA)
            check('raw_provenance',slim_path.name,slim['delivery']['source_raw_sha256'],raw_hash)
            for key in ('seed','phase','source_sha256','config','operations_config','truth','measurement_models',
                        'history','historical_fits','historical_failures','proposal','rows'):
                check('complete_measurements_fits_and_truth',original.name+':'+key,raw.get(key),slim.get(key))
            for section in ('predicted_AC','all_prediction_records'):
                before=(raw['selection'].get(section) or ({} if section=='predicted_AC' else []))
                after=(slim['selection'].get(section) or ({} if section=='predicted_AC' else []))
                check('prediction_denominator',original.name+':'+section,len(before),len(after))
                iterable=before if isinstance(before,dict) else range(len(before))
                for key in iterable:
                    p,q=before[key],after[key]
                    for field in ('pairs','success','failures','fit_calls','fit_ac_calls','forecast_calls','ac_calls',
                                  'required_fit_states','returned_fit_states','power_kw'):
                        check('all_H_only_scores_and_failures',str(key)+':'+field,(p or {}).get(field),(q or {}).get(field))
            for method,record in (raw['selection'].get('methods') or {}).items():
                out=slim['selection']['methods'][method]
                for field in ('selected_origin','initial_best_origin','initial_best_score','selected_score','initial_candidates',
                              'operation','guard','inheritance_protected','failures','raw_proxy_gate'):
                    check('selection_and_operation',method+':'+field,record.get(field),out.get(field))
                for round_name in ('before_recenter','raw'):
                    r=record.get(round_name);s=out.get(round_name)
                    check('exchange_presence',method+':'+round_name,r is None,s is None)
                    if r is None:continue
                    for field in ('required_pairs','weights','offsets','score_caps','status','completed','unfinished',
                                  'objective_lower','objective_upper','gap','target_gap','best_evaluation_index','qp_calls','failures'):
                        check('exchange_global_bounds',method+':'+round_name+':'+field,r.get(field),s.get(field))
                    check('all_eval_denominator',method,len(r.get('evaluations') or []),len(s.get('evaluations') or []))
                    for index,(ev,fv) in enumerate(zip(r.get('evaluations') or [],s.get('evaluations') or [])):
                        for field in EVALUATION_FIELDS:check('all_eval_pair_intervals',method+':'+str(index)+':'+field,ev.get(field),fv.get(field))
                        check('all_eval_QP_denominator',method,len(ev.get('qp_results') or []),len(fv.get('qp_results') or []))
                        if index==r.get('best_evaluation_index'):
                            check('complete_best_QP',method+':'+round_name,ev.get('qp_results'),fv.get('qp_results'))
                            check('complete_best_operation',method+':'+round_name,ev.get('operation_x'),fv.get('operation_x'))
                        else:
                            for k,(qp,sq) in enumerate(zip(ev.get('qp_results') or [],fv.get('qp_results') or [])):
                                for field in d.QP_TRACE_FIELDS:check('all_other_QP_status_and_bounds',method+':'+str(k)+':'+field,qp.get(field),sq.get(field))
                    check('all_solver_iteration_denominator',method,len(r.get('iterations') or []),len(s.get('iterations') or []))
                    for index,(it,jt) in enumerate(zip(r.get('iterations') or [],s.get('iterations') or [])):
                        for field in MASTER_FIELDS:check('all_solver_trace_bounds',method+':'+str(index)+':'+field,
                            (it.get('master') or {}).get(field),(jt.get('master') or {}).get(field))
            before=a.analyze_trial(raw);after=a.analyze_trial(slim)
            check('analysis_method_denominator',original.name,len(before['method_rows']),len(after['method_rows']))
            for x,y in zip(before['method_rows'],after['method_rows']):
                check('analysis_method_identity',original.name,x['method'],y['method'])
                for field in ANALYSIS_FIELDS:check('recomputed_analysis',x['method']+':'+field,x.get(field),y.get(field))
            check('raw_unchanged_after_verification',original.name,d.sha256_file(original),raw_hash)
            files.append(dict(label=label,seed=raw['seed'],raw_sha256=raw_hash,slim_sha256=d.sha256_file(slim_path),
                methods=len(before['method_rows']),actual_noisy_fit_records=sum(len(row.get('fits') or []) for row in raw['rows']),
                actual_noiseless_fit_records=sum(len((row.get('noiseless_AC_refit') or {}).get('fits') or []) for row in raw['rows'])))
    result=dict(success=not errors,checks_by_category=dict(counts),total_checks=sum(counts.values()),errors=errors,files=files,
        new_AC_fits=0,new_optimizations=0,scope='raw/slim field identity and saved-evidence arithmetic comparison; no new scientific fit',
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.core.write_json(ROOT/'raw_slim_verification.json',result)
    if errors:raise ValueError('raw/slim verification failed: '+repr(errors[:5]))
    print(json.dumps(dict(success=True,files=len(files),total_checks=sum(counts.values()),method_rows=sum(f['methods'] for f in files))))
    return result


if __name__=='__main__':verify()
