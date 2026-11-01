"""Independent post-freeze weighted-selection audit, without fit/AC/solver.

Reconstruct every voltage prediction by block sums and every selection from
the declared scalar rule. All independent decisions are written before this
audit opens receipt-bound truth labels. No production scorer is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,pickle,traceback,zipfile
from pathlib import Path

import numpy as np

EPS=1e-12
WEIGHTS=(1,2,4)
FRACTIONS=(.03,.02,.01)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def stamp():return datetime.now(timezone.utc).isoformat()
def write_new(p,value):
    p=Path(p)
    with p.open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def near(a,b,tol=2e-12):return abs(float(a)-float(b))<=tol
def family(blocks):return {tuple(sorted(b)) for b in blocks}
def errors(blocks,true):
    blocks=family(blocks)
    return dict(full_errors=len(blocks^true),missing=[list(b) for b in sorted(true-blocks)],extra=[list(b) for b in sorted(blocks-true)])


class Checks:
    def __init__(self):self.rows=[];self.categories=Counter();self.truth_opened=0;self.observations_opened=0
    def check(self,category,name,condition):
        self.rows.append(dict(category=category,name=name,passed=bool(condition)));self.categories[category]+=1
        if not condition:raise AssertionError(category+': '+name)
    def reference(self,item):
        p=Path(item['path']);self.check('hash',p.name,sha(p)==item['sha256']);return p


def arrays(scene,labels):
    for key in ('P_terminal','Q_terminal','V_terminal'):
        if key in scene and hasattr(scene[key],'columns') and list(scene[key].columns)!=labels:
            raise ValueError('terminal order differs: '+key)
    p=np.asarray(scene['P_terminal'],float);q=np.asarray(scene['Q_terminal'],float)
    if 'V_terminal' in scene and 'V_root_observed' in scene:
        voltage=np.asarray(scene['V_terminal'],float);root=np.asarray(scene['V_root_observed'],float).reshape(-1)
        y=root[:,None]**2-voltage**2
        if 'drop_target' in scene and not np.allclose(y,np.asarray(scene['drop_target']),rtol=0,atol=1e-12):
            raise ValueError('observed target mismatch')
    else:y=np.asarray(scene['drop_target'],float)
    if p.ndim!=2 or q.shape!=p.shape or y.shape!=p.shape or p.shape[1]!=len(labels):raise ValueError('observation shape differs')
    if not all(np.isfinite(a).all() for a in (p,q,y)):raise ValueError('nonfinite observations')
    return p,q,y


def block_prediction(point,p,q):
    result=np.zeros_like(p)
    for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        signal=float(r)*p[:,support].sum(axis=1)+float(x)*q[:,support].sum(axis=1)
        result[:,support]+=signal[:,None]
    return result


def contract_labels(point,calibration,labels):
    keep=[i for i,b in enumerate(point['supports']) if len(b)==1 or
        calibration['a']*point['r'][i]+calibration['b']*point['x'][i]+1e-10>=.16*calibration['median_root_depth']]
    return [point['supports'][i] for i in keep],[[labels[j] for j in point['supports'][i]] for i in keep]


def decide(order,counts,means,scope,threshold):
    reference=min(scope,key=lambda k:(means[k],order.index(k)))
    bound=threshold(reference)
    eligible=[k for k in scope if means[k]<=bound]
    selected=min(eligible,key=lambda k:(counts[k],means[k],order.index(k)))
    return dict(reference=reference,threshold=bound,eligible=eligible,selected=selected,
        means=means,margins={k:means[k]-bound for k in order})


def independent_rules(pool,h_abs,d_abs,new_cells):
    order=pool['selection_order'];h_cells=pool['historical_validation_cells']
    models={m['candidate_id']:m for m in pool['models']};counts={k:len(models[k]['point']['supports']) for k in order}
    h_means={k:h_abs[k]/h_cells for k in order};hbest=min(h_means.values())
    h_eligible=[k for k in order if h_means[k]<=1.03*hbest+EPS]
    combined={k:(h_abs[k]+d_abs[k])/(h_cells+new_cells) for k in order}
    original=decide(order,counts,combined,order,lambda j:1.03*combined[j]+EPS)
    day={k:d_abs[k]/new_cells for k in order}
    day_only=decide(order,counts,day,order,lambda j:1.03*day[j]+EPS)
    rules={}
    for weight in WEIGHTS:
        mass=h_cells+weight*new_cells
        means={k:(h_abs[k]+weight*d_abs[k])/mass for k in order}
        for fraction in FRACTIONS:
            rid=f'w{weight}_tau{round(100*fraction)}pct'
            result=decide(order,counts,means,h_eligible,
                lambda j: 1.03*means[j]+EPS if fraction==.03 else
                means[j]+(.03*h_abs[j]+fraction*weight*d_abs[j])/mass+EPS)
            ref=result['reference'];result.update(weight=weight,fraction=fraction,
                historical_tolerance=.03*h_abs[ref]/mass,new_tolerance=fraction*weight*d_abs[ref]/mass)
            rules[rid]=result
    rules['original_ungated_w1_tau3pct']=original;rules['day_only_3pct']=day_only
    return h_eligible,rules


def run(root,output=None):
    root=Path(root).resolve();output=Path(output) if output else root/'independent_weighted_selection_audit.json'
    if output.exists():raise FileExistsError('retain original independent audit')
    decision_file=output.with_name(output.stem+'_decisions_frozen.json')
    checks=Checks();result=dict(status='started',started_utc=stamp(),new_fit_calls=0,new_solver_calls=0,new_AC=0,
        source_sha256=sha(__file__),scope='independent block predictions, all fixed selection rules and paired rule/DR effects')
    try:
        for name in ('actual/observations_frozen.json','actual/predictions_frozen.json','actual/scored.json','summary.json','ended.json'):
            checks.check('boundary','complete_'+name,(root/name).is_file())
        end=read(root/'ended.json');checks.reference(end['summary'])
        # Result files contain evaluated true-support errors. Hash/existence
        # checks are sufficient until every independent decision is fixed.
        prediction_path=root/'actual/predictions_frozen.json';predictions=read(prediction_path)
        observation_path=checks.reference(predictions['observations_frozen']);observations=read(observation_path)
        preparation=read(root/'preparation_ended.json');freeze_path=checks.reference(preparation['price_freeze']);freeze=read(freeze_path)
        checks.reference(observations['price_freeze'])
        protocol_path=checks.reference(freeze['protocol']);protocol=read(protocol_path)
        snapshot_path=checks.reference(freeze['source_snapshot'])
        sources=protocol['source_sha256']
        with zipfile.ZipFile(snapshot_path) as archive:
            members=archive.namelist();checks.check('source','snapshot_count',len(members)==len(sources))
            for i,file in enumerate(sorted(sources)):
                checks.check('source','runtime_'+Path(file).name,sha(file)==sources[file])
                name=f'{i:03d}_'+Path(file).name
                checks.check('source','snapshot_'+name,hashlib.sha256(archive.read(name)).hexdigest()==sources[file])
        checks.check('protocol','five_conditions',len(protocol['cases'])==5)
        checks.check('protocol','three_networks',len({c['case'] for c in protocol['cases']})==3)
        checks.check('protocol','rules_frozen',protocol['new_day_weights']==[1,2,4] and protocol['new_day_tolerances']==[.03,.02,.01])
        checks.check('protocol','history_gate_declared',protocol['history_gate'] is True)
        checks.check('protocol','primary_rule',protocol['primary_rule']=='w2_tau2pct')

        pools={};historical={};metadata={};case_by_id={c['name']:c for c in protocol['cases']}
        for case in protocol['cases']:
            condition=case['name'];declared=read(checks.reference(case['case_protocol']))
            pool=read(checks.reference(case['pool']));runtime=read(checks.reference(case['runtime_protocol']))
            fingerprint={k:v for k,v in pool.items() if k!='pool_sha256'}
            checks.check('parameters',condition+'_pool_fingerprint',hashlib.sha256(json.dumps(fingerprint,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==pool['pool_sha256'])
            checks.check('parameters',condition+'_runtime_pool',runtime['frozen_pool']==case['pool'])
            with checks.reference(pool['history_packet']).open('rb') as stream:history=pickle.load(stream)['history']
            labels=pool['labels'];n=len(labels);h_abs={};old=read(checks.reference(pool['historical_fit']))
            counts={role:sum(len(s['P_terminal']) for s in history[role]) for role in ('training','validation')}
            checks.check('history',condition+'_full288_288',counts==dict(training=288,validation=288))
            for model in pool['models']:
                identity=model['candidate_id'];point=model['point'];tag=condition+'_'+identity
                checks.check('parameters',tag+'_saved_point',point==old['selection_candidates'][model['historical_index']])
                rm=np.zeros((n,n));xm=np.zeros((n,n))
                for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
                    rm[np.ix_(block,block)]+=r;xm[np.ix_(block,block)]+=x
                checks.check('parameters',tag+'_R',np.array_equal(rm,np.asarray(model['R'])))
                checks.check('parameters',tag+'_X',np.array_equal(xm,np.asarray(model['X'])))
                output_i,output_l=contract_labels(point,pool['calibration'],labels)
                checks.check('parameters',tag+'_frozen_output_indices',output_i==model['frozen_output_supports'])
                checks.check('parameters',tag+'_frozen_output_labels',output_l==model['frozen_output_support_labels'])
                for role in ('training','validation'):
                    total=0
                    for scene in history[role]:
                        pp,qq,yy=arrays(scene,labels);total+=float(np.abs(block_prediction(point,pp,qq)-yy).sum())
                    checks.check('history',tag+'_'+role+'_loss',near(total,model['historical_'+role+'_abs_sum'],2e-10))
                    if role=='validation':h_abs[identity]=total
            pools[condition]=pool;historical[condition]=h_abs;metadata[condition]=declared

        expected={(c['name'],path,policy) for c in protocol['cases'] for path in c['paths'] for policy in protocol['policies']}
        observed={(r['condition'],r['path'],r['policy']) for r in observations['rows']}
        checks.check('denominator','all60_requests',len(expected)==60 and observed==expected and len(observations['rows'])==60)
        projected_by_id={r['record_id']:r for r in predictions['rows']}
        checks.check('denominator','all60_predicted',len(projected_by_id)==60 and len(predictions['rows'])==60)
        independent={}
        for row in observations['rows']:
            identity=row['record_id'];condition=row['condition'];pool=pools[condition];case=case_by_id[condition]
            if row['status']!='observed':
                checks.check('status',identity+'_retained',projected_by_id[identity]['status']==row['status'])
                continue
            request=read(checks.reference(row['request']));receipt=read(checks.reference(row['receipt']))
            checks.check('request',identity+'_pool_ref',request['frozen_pool']==case['pool'])
            checks.check('request',identity+'_plan_ref',request['plan']==freeze['cases'][condition]['plans'][row['policy']])
            checks.check('request',identity+'_protocol_ref',request['protocol']==case['runtime_protocol'])
            checks.check('request',identity+'_replicate',request['replicate']==protocol['replicate_base']+10*row['path'])
            checks.check('request',identity+'_receipt_request',receipt['request_sha256_expected']==row['request']['sha256'])
            checks.check('fee',identity+'_receipt_payment',near(row['actual_payment'],receipt['actual_payment'],1e-10))
            packet_path=checks.reference(row['packet'])
            checks.check('hash',identity+'_receipt_packet',sha(packet_path)==receipt['private_files']['packet.pkl']['sha256'])
            with packet_path.open('rb') as stream:packet=pickle.load(stream)
            checks.observations_opened+=1
            checks.check('observations',identity+'_one_day',len(packet['validation'])==1)
            pp,qq,yy=arrays(packet['validation'][0],pool['labels'])
            checks.check('observations',identity+'_96rows',len(pp)==96)
            scene_predictions={};d_abs={}
            for model in pool['models']:
                pred=block_prediction(model['point'],pp,qq);scene_predictions[model['candidate_id']]=pred
                d_abs[model['candidate_id']]=float(np.abs(pred-yy).sum())
            h_eligible,rules=independent_rules(pool,historical[condition],d_abs,pp.size)
            first,second=pool['selection_order'];distance=float(np.abs(scene_predictions[first]-scene_predictions[second]).mean())
            saved=read(checks.reference(projected_by_id[identity]['scores']))
            checks.check('selection',identity+'_history_gate',saved['historical_eligible_candidate_ids']==h_eligible)
            checks.check('selection',identity+'_no_parameter_update',saved['parameters_changed'] is False and saved['calibration_changed'] is False and saved['new_fit_calls']==saved['new_search_calls']==0)
            checks.check('distance',identity+'_paired_distance',near(distance,saved['new_prediction_pair_distance_l1']))
            checks.check('denominator',identity+'_actual_rows',saved['actual_validation_rows']==384)
            for r in saved['loss_rows']:
                k=r['candidate_id']
                checks.check('loss',identity+'_'+k+'_historical',near(r['historical_validation_mae'],historical[condition][k]/pool['historical_validation_cells']))
                checks.check('loss',identity+'_'+k+'_newday',near(r['new_validation_mae'],d_abs[k]/pp.size))
            saved_rules={r['rule_id']:r for r in saved['rules']}
            checks.check('denominator',identity+'_allnine_rules',set(saved_rules)=={k for k in rules if k.startswith('w')})
            for rid,rule in rules.items():
                if rid=='original_ungated_w1_tau3pct':current=saved['original_ungated_selection']
                elif rid=='day_only_3pct':current=saved['original_ungated_selection']['day_only_selection']
                else:current=saved_rules[rid]
                checks.check('selection',identity+'_'+rid+'_chosen',current['selected_candidate_id']==rule['selected'])
                checks.check('selection',identity+'_'+rid+'_eligible',current['eligible_candidate_ids']==rule['eligible'])
                model=next(m for m in pool['models'] if m['candidate_id']==rule['selected'])
                checks.check('selection',identity+'_'+rid+'_output',current['selected_output_support_labels']==model['frozen_output_support_labels'])
                if rid in saved_rules:
                    checks.check('selection',identity+'_'+rid+'_reference',current['reference_candidate_id']==rule['reference'])
                    for k in pool['selection_order']:
                        checks.check('margin',identity+'_'+rid+'_'+k,near(current['exit_margins'][k],rule['margins'][k]))
                        checks.check('loss',identity+'_'+rid+'_'+k,near(current['weighted_validation_mae'][k],rule['means'][k]))
                    checks.check('selection',identity+'_'+rid+'_historic_tolerance',near(current['historical_tolerance_mean'],rule['historical_tolerance']))
                    checks.check('selection',identity+'_'+rid+'_newday_tolerance',near(current['new_day_tolerance_mean'],rule['new_tolerance']))
                    checks.check('selection',identity+'_'+rid+'_raw_count',current['selected_raw_support_count']==len(model['point']['supports']))
                    checks.check('denominator',identity+'_'+rid+'_effective_rows',current['effective_validation_rows']==288+rule['weight']*96)
            checks.check('selection',identity+'_primary',saved['primary_rule']['rule_id']=='w2_tau2pct')
            independent[identity]=dict(condition=condition,path=row['path'],policy=row['policy'],rules=rules,
                distance=distance,historical_eligible_ids=h_eligible,new_abs=d_abs,historical_abs=historical[condition])
        checks.check('boundary','zero_truth_before_decisions',checks.truth_opened==0)
        write_new(decision_file,dict(status='frozen',created_utc=stamp(),independent_decisions=independent,
            no_truth_read=True,observed_packets=checks.observations_opened,source_sha256=sha(__file__)))

        summary=read(root/'summary.json');scored=read(root/'actual/scored.json')
        checks.reference(summary['price_freeze'])
        checks.check('hash','scored_prediction_chain',checks.reference(scored['predictions_frozen'])==prediction_path)
        scored_by_id={r['record_id']:r for r in scored['rows']}
        checks.check('denominator','all60_scored',len(scored_by_id)==60 and len(scored['rows'])==60)
        evaluated={};fees={};attempted=0;failed_model_payments=[]
        for row in observations['rows']:
            identity=row['record_id'];saved=scored_by_id[identity]
            if row.get('receipt'):
                receipt=read(checks.reference(row['receipt']));attempted+=96*bool(receipt.get('AC_attempted'))
            if identity not in independent:
                checks.check('status',identity+'_final_status_retained',saved['status']==row['status'])
                failed_model_payments.append(dict(record_id=identity,condition=row['condition'],policy=row['policy'],
                    status=row['status'],model_calculated_payment=row.get('actual_payment'),
                    AC_attempted=bool(receipt.get('AC_attempted')) if row.get('receipt') else None,
                    physical_delivery_verified=False,
                    interpretation='worker computed this payment before whole-load component qualification; it is not a verified delivered-payment observation'))
                continue
            truth_path=Path(row['physical_output'])/'truth_evaluation.json'
            checks.check('truth_hash',identity,sha(truth_path)==receipt['private_files']['truth_evaluation.json']['sha256'])
            true=family(read(truth_path)['true_supports']);checks.truth_opened+=1
            pool=pools[row['condition']];models={m['candidate_id']:m for m in pool['models']}
            candidate_errors={k:errors(m['frozen_output_support_labels'],true) for k,m in models.items()}
            checks.check('truth',identity+'_candidate_errors',candidate_errors==saved['candidate_errors'])
            correct=[k for k in pool['selection_order'] if candidate_errors[k]['full_errors']==0]
            checks.check('truth',identity+'_correct_candidates',correct==saved['correct_candidate_ids'])
            expected_evaluations={};actual_evaluations={e['rule_id']:e for e in saved['evaluations']}
            checks.check('denominator',identity+'_11_evaluations',set(actual_evaluations)==set(independent[identity]['rules']))
            for rid,rule in independent[identity]['rules'].items():
                error=errors(models[rule['selected']]['frozen_output_support_labels'],true)
                expected_evaluations[rid]=dict(error,selected=rule['selected'],eligible=rule['eligible'])
                current=actual_evaluations[rid]
                checks.check('truth',identity+'_'+rid+'_errors',all(error[k]==current[k] for k in error))
                checks.check('selection',identity+'_'+rid+'_final_id',current['selected_candidate']==rule['selected'])
                if rid.startswith('w'):
                    wrong=[k for k in models if k not in correct]
                    excluded=bool(correct and wrong and all(k not in rule['eligible'] for k in wrong))
                    checks.check('truth',identity+'_'+rid+'_excluded',current['wrong_models_excluded']==excluded)
            main=expected_evaluations['w2_tau2pct'];original=expected_evaluations['original_ungated_w1_tau3pct']
            checks.check('score',identity+'_main_errors',saved['full_errors']==main['full_errors'] and saved['missing']==main['missing'] and saved['extra']==main['extra'])
            checks.check('score',identity+'_rule_gain',saved['rule_error_reduction']==original['full_errors']-main['full_errors'])
            checks.check('score',identity+'_no_parameter_change',saved['parameters_changed'] is False)
            checks.check('fee',identity+'_scored_payment',near(saved['actual_payment'],row['actual_payment'],1e-10))
            fees[identity]=row.get('actual_payment')
            evaluated[identity]=dict(independent[identity],evaluations=expected_evaluations,actual_payment=row.get('actual_payment'))

        table=[];all_rule_table=[]
        for condition in case_by_id:
            case=case_by_id[condition]
            for policy in protocol['policies']:
                records=[r for r in observations['rows'] if r['condition']==condition and r['policy']==policy]
                valid=[evaluated[r['record_id']] for r in records if r['record_id'] in evaluated]
                paired=[]
                for entry in valid:
                    identity=next(r['record_id'] for r in records if r['path']==entry['path'])
                    zero=next((x for x in evaluated.values() if x['condition']==condition and x['path']==entry['path'] and x['policy']=='zero'),None)
                    if zero is None:continue
                    gain=zero['evaluations']['w2_tau2pct']['full_errors']-entry['evaluations']['w2_tau2pct']['full_errors']
                    checks.check('score',identity+'_DR_gain',scored_by_id[identity]['paired_DR_error_reduction']==gain)
                    distance_gain=entry['distance']-zero['distance']
                    checks.check('distance',identity+'_DR_distance',near(scored_by_id[identity]['distance_gain'],distance_gain))
                    paired.append(gain)
                main_errors=[v['evaluations']['w2_tau2pct']['full_errors'] for v in valid]
                original_errors=[v['evaluations']['original_ungated_w1_tau3pct']['full_errors'] for v in valid]
                known_fees=[r['actual_payment'] for r in records if r.get('actual_payment') is not None]
                retained_errors=[evaluated[r['record_id']]['evaluations']['w2_tau2pct']['full_errors']
                    if r['record_id'] in evaluated else None for r in records]
                table.append(dict(condition=condition,policy=policy,denominator=4,complete=len(valid),unresolved=4-len(valid),
                    full_errors=retained_errors,recoveries=sum(x==0 for x in main_errors),paired_complete=len(paired),
                    mean_payment=sum(known_fees)/4 if len(known_fees)==4 else None,
                    known_payment_total=sum(known_fees),unknown_payment_records=4-len(known_fees),
                    valid_delivery_payment_total=sum(v['actual_payment'] for v in valid if v['actual_payment'] is not None),
                    valid_delivery_payment_records=sum(v['actual_payment'] is not None for v in valid),
                    raw_payment_aggregate_scope='includes worker prequalification calculations for failed rows; not all verified delivered fees',
                    DR_improved=sum(x>0 for x in paired),DR_worsened=sum(x<0 for x in paired),
                    rule_improved=sum(a>b for a,b in zip(original_errors,main_errors)),
                    rule_worsened=sum(a<b for a,b in zip(original_errors,main_errors)),
                    original_correct_now_wrong=sum(a==0 and b>0 for a,b in zip(original_errors,main_errors))))
                for rid in ('w1_tau3pct','w1_tau2pct','w1_tau1pct','w2_tau3pct','w2_tau2pct','w2_tau1pct','w4_tau3pct','w4_tau2pct','w4_tau1pct','original_ungated_w1_tau3pct','day_only_3pct'):
                    these=[v['evaluations'][rid] for v in valid]
                    gains=[]
                    for entry in valid:
                        zero=next((x for x in evaluated.values() if x['condition']==condition and x['path']==entry['path'] and x['policy']=='zero'),None)
                        if zero is not None:gains.append(zero['evaluations'][rid]['full_errors']-entry['evaluations'][rid]['full_errors'])
                    all_rule_table.append(dict(condition=condition,policy=policy,rule_id=rid,denominator=4,complete=len(valid),
                        total_errors=sum(e['full_errors'] for e in these),recoveries=sum(e['full_errors']==0 for e in these),
                        extra_blocks=sum(len(e['extra']) for e in these),missing_blocks=sum(len(e['missing']) for e in these),
                        DR_error_reduction=sum(gains),DR_improved=sum(g>0 for g in gains),DR_worsened=sum(g<0 for g in gains)))
        saved_table={(t['condition'],t['policy']):t for t in summary['table']}
        for item in table:
            saved=saved_table[item['condition'],item['policy']]
            for key in ('denominator','complete','unresolved','full_errors','recoveries','DR_improved','DR_worsened','rule_improved','rule_worsened','unknown_payment_records'):
                checks.check('aggregate',item['condition']+'_'+item['policy']+'_'+key,saved[key]==item[key])
            checks.check('aggregate',item['condition']+'_'+item['policy']+'_knownfee',near(saved['known_payment_total'],item['known_payment_total'],1e-9))
            checks.check('aggregate',item['condition']+'_'+item['policy']+'_meanfee',
                saved['mean_payment'] is None and item['mean_payment'] is None or
                saved['mean_payment'] is not None and item['mean_payment'] is not None and near(saved['mean_payment'],item['mean_payment'],1e-9))
        checks.check('denominator','summary_complete',summary['complete']==len(evaluated))
        checks.check('denominator','20_case_days',summary['predeclared_case_days']==20)
        checks.check('denominator','attempted_AC_points',summary['AC_attempted_points']==attempted)
        checks.check('denominator','valid_AC_points',summary['valid_AC_points']==96*len(evaluated))
        result.update(status='complete',audit_passed=True,completed_utc=stamp(),table=table,all_rule_table=all_rule_table,
            requested_records=60,complete_records=len(evaluated),unresolved_records=60-len(evaluated),
            predeclared_case_days=20,conditions=5,physical_networks=3,attempted_AC_points=attempted,
            networks_with_valid_observations=len({case_by_id[v['condition']]['case'] for v in evaluated.values()}),
            valid_AC_points=96*len(evaluated),independent_decisions_sha256=sha(decision_file),
            failed_model_calculated_payments=failed_model_payments,
            price_freeze_sha256=sha(freeze_path),protocol_sha256=sha(protocol_path),
            observed_packets_opened=checks.observations_opened,truth_labels_opened=checks.truth_opened)
    except Exception as error:
        result.update(status='audit_failed',audit_passed=False,completed_utc=stamp(),error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    result.update(check_count=len(checks.rows),checks_by_category=dict(checks.categories),
        failures=[r for r in checks.rows if not r['passed']],checks=checks.rows)
    write_new(output,result)
    print(json.dumps({k:result.get(k) for k in ('status','audit_passed','check_count','complete_records','unresolved_records','message')},ensure_ascii=False))
    if not result.get('audit_passed'):raise RuntimeError(result.get('message','audit failed'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.root,args.output)
