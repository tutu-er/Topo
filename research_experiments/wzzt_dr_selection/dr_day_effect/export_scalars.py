"""Scalar-only release of a completed independent actual-day DR experiment."""
from pathlib import Path
import argparse,hashlib,json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def export(root,output,physical,selection,mechanism):
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');f=read(root/'freeze.json');s=read(root/'summary_private.json')
    if read(root/'ended.json')['status']!='complete':raise ValueError('completed and frozen experiment required')
    pa=read(physical);sa=read(selection)
    if pa['checks']!=pa['passed'] or sa['checks']!=sa['passed']:raise ValueError('independent audit must pass')
    plans={k:read(v['path']) for k,v in p['plans'].items()}
    config=dict(date='2026-11-01',condition=p['condition'],qualification=p['qualification'],
        days=12,planned_records=60,slots=96,interval_hours=.25,policies=p['policies'],primary_rule=p['primary_rule'],
        historical_rows=p['full_history_rows'],actual_validation_rows=p['actual_validation_rows'],effective_weight_rows=p['effective_weight_rows'],
        case=p['case'],root_label=p['declared']['root_label'],terminals=len(p['declared']['labels']),base_kw=p['declared']['base_kw'],
        voltage_sigma=p['declared']['voltage_sigma'],pq_sigma=p['declared']['pq_sigma'],
        optimizer_payment_cap=None,execution_payment_mode=p['payment_mode'],
        comparison=p['comparison'],low_selection=p['selected_low_rule'],high_role=p['high_role'],
        trials=p['trials'],prior_identity_audit=p['prior_request_identity_audit'],
        plans={k:dict(sha256=p['plans'][k]['sha256'],expected_payment=v['expected_payment'],payment_bound=v['hard_payment_bound'],prices=v['prices']) for k,v in plans.items()},
        full_matrices_and_daily_profiles='private',environment=p['environment'],scope=p['scope'],protocol_sha256=sha(root/'protocol.json'))
    write(out/'configuration.json',config)
    tables=[]
    for x in s['tables']:
        x=dict(x);x['mean_delivered_payment']=x['delivered_payment']/x['complete'] if x['complete'] else None;tables.append(x)
    write(out/'summary.json',dict(status='complete',tables=tables,pairs=s['pairs'],primary_test=s['primary_one_sided_90pct_test'],
        independent_simulated_days=12,planned_records=60,completed_records=s['completed_records'],
        AC_strategy_days=60,AC_timepoints=5760,new_parameter_fits=0,new_price_solves=0,new_W_search=0,
        qualification=p['qualification'],scope=s['scope'],operational_limits='AC convergence, algebraic balance and voltage checked; branch ampacity not checked',
        probability_scope='fixed paper15 synthetic profile-mixture/measurement/DR law; no field or all-topology probability',
        audit_source_scope='independent audits generate no new data; experiment itself contains60 fresh AC strategy-days'))
    rows=[]
    for x in s['records']:
        q={k:x.get(k) for k in ['day','policy','record_id','status','full_errors','missing','extra','selected_candidate','coarse_excluded','attained','delivered_fee','paired_error_reduction']}
        for k in ['M','baseline_M','delta_M','target_M','distance','distance_gain']:q[k+'_micro']=x[k]*1e6 if k in x else None
        rows.append(q)
    write(out/'records.json',dict(status='complete',denominator_per_policy=12,rows=rows))
    rule_tables=[]
    for rid in [x['rule_id'] for x in s['records'][0]['rule_outcomes']]:
        groups=[]
        for policy in p['policies']:
            raw=[r for r in s['records'] if r['policy']==policy and r['status']=='complete']
            results=[next(q for q in r['rule_outcomes'] if q['rule_id']==rid) for r in raw]
            groups.append(dict(policy=policy,planned=12,complete=len(results),unknown=12-len(results),
                known_full_errors=sum(q['full_errors'] for q in results),recoveries=sum(q['full_errors']==0 for q in results)))
        rule_tables.append(dict(rule_id=rid,primary=rid==p['primary_rule'],tables=groups))
    write(out/'rule_sensitivity.json',dict(scope='all9 prespecified rules; same frozen quotes and same60 observations; do not replace primary after results',rows=rule_tables))
    gaps=[]
    for day in range(12):
        z=next(x for x in rows if x['day']==day and x['policy']=='zero')
        if z['status']!='complete' or z['full_errors']==0:continue
        gap=dict(day=day,M0_micro=z['M_micro'],needed_increment_micro=z['target_M_micro']-z['M_micro'])
        for policy in p['policies'][1:]:
            a=next(x for x in rows if x['day']==day and x['policy']==policy)
            gap[policy]=dict(provided_increment_micro=a['delta_M_micro'],remaining_exit_gap_micro=max(-a['M_micro'],0.) if a['M_micro'] is not None else None,full_errors=a['full_errors'])
        gaps.append(gap)
    write(out/'gaps.json',dict(scope='post-observation mechanism diagnostic; six noDR-wrong days; all other days retained in main denominator',rows=gaps))
    mech=read(mechanism);mech['diagnostic_new_AC']=mech.pop('new_AC',0);write(out/'mechanism.json',mech)
    paths=[root/'protocol.json',root/'freeze.json',root/'source_snapshot.zip',root/'preflight.json',root/'predictions_frozen.json',
        root/'observations_frozen.json',root/'scored.json',root/'summary_private.json',root/'ended.json',Path(physical),Path(selection),Path(mechanism)]
    audit=dict(status='complete',physical_checks=pa['checks'],selection_checks=sa['checks'],
        physical_maximum_errors=pa['maximum_errors'],private_receipt_sha256={str(v):sha(v) for v in paths},
        protocol_sha256=sha(root/'protocol.json'),export_source_sha256=sha(__file__),
        public_scope='no P/Q/V/AC arrays, RX matrices, household profiles or96-slot control curves')
    write(out/'audit_summary.json',audit)
    print(dict(records=len(rows),complete=s['completed_records'],physical_checks=pa['checks'],selection_checks=sa['checks']))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--physical',type=Path,required=True);a.add_argument('--selection',type=Path,required=True);a.add_argument('--mechanism',type=Path,required=True)
    q=a.parse_args();export(q.root,q.output,q.physical,q.selection,q.mechanism)
