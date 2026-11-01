"""Scalar-only figure data from completed frozen-model DR records.

No P/Q/V/AC arrays, full R/X, optimizer, fitter or simulator is opened. The
optional triangle threshold is post-observation diagnostic and never a
deployable universal topology-rejection threshold.
"""
from pathlib import Path
import argparse,hashlib,json
from statistics import mean

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(ref):
    p=Path(ref['path'])
    if sha(p)!=ref['sha256']:raise ValueError('scalar source hash mismatch: '+str(p))
    return p
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def extract(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    s=read(root/'summary.json');p=read(root/'protocol.json');f=read(root/'price_freeze.json')
    records=[];rules=[]
    for row in s['rows']:
        x={k:row.get(k) for k in ['condition','case','policy','path','record_id','status','full_errors','actual_payment']}
        if row['status']!='complete':
            x['excluded_from_valid_means']=True;records.append(x);continue
        zero=next(r for r in s['rows'] if r['condition']==row['condition'] and r['path']==row['path'] and r['policy']=='zero')
        assert zero['status']=='complete'
        score=read(checked(row['scores']))
        d0=zero['new_distance'];d=row['new_distance'];gamma=f['cases'][row['condition']]['gamma_native']
        loss={q['candidate_id']:q for q in score['loss_rows']}
        true=row['correct_candidate_ids'];wrong=[k for k in loss if k not in true]
        x.update(distance_baseline_micro=d0*1e6,distance_controlled_micro=d*1e6,
            distance_gain_micro=(d-d0)*1e6,relative_distance_gain_pct=100*(d/d0-1) if d0 else None,
            frozen_gamma_micro=gamma*1e6,positive_gain=d-d0>1e-12,
            frozen_gamma_met=d-d0>=gamma-1e-12 if gamma>1e-12 else None,
            true_candidate_ids_evaluator_only=true,score_scalar_ref=row['scores'])
        records.append(x)
        for q in score['rules']:
            w=q['new_day_weight'];tau=q['new_day_fraction'];nh=score['history_validation_rows'];nd=score['new_validation_rows']
            r=dict(condition=row['condition'],case=row['case'],policy=row['policy'],path=row['path'],
                record_id=row['record_id'],rule_id=q['rule_id'],new_day_weight=w,new_day_fraction=tau,
                exit_margins_micro={k:v*1e6 for k,v in q['exit_margins'].items()},
                weighted_reference=q['reference_candidate_id'],selected=q['selected_candidate_id'],
                selected_correct=q['selected_candidate_id'] in true,
                wrong_candidate_ids_evaluator_only=wrong,
                actual_distance_micro=d*1e6,distance_baseline_micro=d0*1e6,
                distance_gain_micro=(d-d0)*1e6)
            if len(true)==1 and len(wrong)==1:
                ref=true[0];other=wrong[0]
                h_gap=nh*(loss[other]['historical_validation_mae']-1.03*loss[ref]['historical_validation_mae'])
                error=loss[ref]['new_validation_mae']
                required=((2+tau)*error-h_gap/(w*nd)+1e-12*(nh+w*nd)/(w*nd))
                r.update(observed_reference_residual_micro=error*1e6,
                    triangle_sufficient_distance_micro=required*1e6,
                    triangle_bound_satisfied=d>required,
                    triangle_scope='evaluator-only sufficient bound using observed residual and correct-structure label; not necessary or predictive',
                    wrong_exit_margin_micro=q['exit_margins'][other]*1e6,
                    raw_support_counts={k:v['raw_support_count'] for k,v in loss.items()})
            rules.append(r)
    cases=[]
    for c in p['cases']:
        groups={pol:[x for x in records if x['condition']==c['name'] and x['policy']==pol] for pol in p['policies']}
        case=dict(condition=c['name'],case=c['case'],planned_days=4,
            valid_days=sum(x['status']=='complete' for x in groups['zero']),
            unresolved_days=sum(x['status']!='complete' for x in groups['zero']),policies=[])
        for pol,g in groups.items():
            valid=[x for x in g if x['status']=='complete']
            case['policies'].append(dict(policy=pol,valid=len(valid),denominator=4,
                distance_baseline_micro=mean(x['distance_baseline_micro'] for x in valid),
                distance_controlled_micro=mean(x['distance_controlled_micro'] for x in valid),
                distance_gain_micro=mean(x['distance_gain_micro'] for x in valid),
                relative_distance_gain_pct=mean(x['relative_distance_gain_pct'] for x in valid),
                positive_gain_count=sum(x['positive_gain'] for x in valid),
                frozen_gamma_micro=f['cases'][c['name']]['gamma_native']*1e6,
                frozen_gamma_met_count=sum(x['frozen_gamma_met'] is True for x in valid),
                valid_mean_payment=mean(x['actual_payment'] for x in valid)))
        cases.append(case)
    plans=[]
    for c in p['cases']:
        for pol,item in f['cases'][c['name']]['plans'].items():
            a=read(checked(item))
            plans.append(dict(condition=c['name'],policy=pol,expected_payment=a['expected_payment'],
                hard_payment_bound=a['hard_payment_bound'],
                amplitude_scale_budget_max=(1.6/a['hard_payment_bound'])**.5 if a['hard_payment_bound']>0 else None,
                selected_modules=[m['module_id'] for m,v in zip(a['modules'],a['amplitudes']) if v>0],
                mean_active_amplitudes_kw=[v for v in a['amplitudes'] if v>0]))
    output=dict(status='complete',requested_records=60,complete_records=sum(x['status']=='complete' for x in records),
        planned_condition_days=20,valid_condition_days=sum(c['valid_days'] for c in cases),
        records=records,cases=cases,rule_records=rules,plans=plans,
        source_hashes={name:sha(root/name) for name in ['summary.json','protocol.json','price_freeze.json']},
        computation=dict(new_solver=0,new_fit=0,new_AC=0,raw_PQV_or_RX_arrays_opened=0),
        threshold_warning='gamma is the frozen prediction-distance-increment target. It is not a topology rejection threshold. Actual rejection uses exit_margin>0.',
        triangle_warning='Triangle bound uses observed residual and evaluator label. It is sufficient and may be very conservative; it is not needed for primary plots.')
    write(out/'figure_scalars.json',output)
    print(json.dumps(dict(cases=cases,complete=output['complete_records'],valid_days=output['valid_condition_days']),ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    x=a.parse_args();extract(x.root,x.output)
