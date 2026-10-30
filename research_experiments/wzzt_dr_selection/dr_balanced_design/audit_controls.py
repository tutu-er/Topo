"""Independent read-only contract and common-witness audit.

No experiment/study/sampler is imported, and no optimizer is called. Audit
outputs contain hashes and counts, not observations or per-time trajectories.
The root may still be writing: each invocation records an immutable snapshot.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from collections import Counter, defaultdict
from itertools import combinations, permutations
import argparse
import json
import pickle
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / 'wzzt_b_design_v1'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def hash_file(p): return sha256(Path(p).read_bytes()).hexdigest()
def canon(v): return json.dumps(v, sort_keys=True, separators=(',', ':'))
def c_hash(v): return sha256(canon(v).encode()).hexdigest()

def matrices(supports, r, x):
    R = np.zeros((3,3)); X = np.zeros((3,3))
    for s, rv, xv in zip(supports, r, x):
        z = np.zeros(3); z[list(s)] = 1.
        R += float(rv) * np.outer(z,z); X += float(xv) * np.outer(z,z)
    return R,X

def family(point): return tuple(sorted(tuple(s) for s in point['supports']))

def audit(root):
    root=Path(root).resolve(); cfg=read(root/'protocol.json'); actions=read(root/'actions_frozen.json')
    generated=read(root/'actual_generation_started.json'); witness=read(root/'planning/common_witness.json')
    checks=[]
    def check(label, ok, **detail): checks.append(dict(label=label, ok=bool(ok), **detail))
    def close(a,b): return np.allclose(a,b,atol=1e-12,rtol=0)
    check('protocol_hash', hash_file(root/'protocol.json')==read(root/'protocol.sha256.json')['sha256'])
    check('frozen_actions_bound', hash_file(root/'actions_frozen.json')==generated['actions_sha256'])
    check('before_future', actions.get('future_generated') is False and actions.get('truth_read_by_planner') is False)
    check('protocol_before_generation', datetime.fromisoformat(cfg['created_utc']) <= datetime.fromisoformat(generated['created_utc']))
    check('new_two_flows', len(cfg['flows'])==2 and len({f['behavior'] for f in cfg['flows']})==2 and
          not ({f['behavior'] for f in cfg['flows']} & {914031,914032}))
    check('full_original_rows', cfg['historical_rows']==[288,288] and cfg['actual_rows']==[480,480])
    check('sources_hash', all(hash_file(p)==h for p,h in cfg['source_sha256'].items()))
    for key in ('history','history_state','historical_W','math_gate'):
        check('input_hash_'+key, hash_file(cfg[key]['path'])==cfg[key]['sha256'])

    menus=cfg['menus']; byid={m['id']:m for m in menus}
    check('catalog_22_unique', len(menus)==22 and len(byid)==22)
    check('catalog_kinds', Counter(m['kind'] for m in menus)==dict(zero=1,single1=3,single2=3,rotation=6,pair1=3,pair_rotation=6))
    ordered_pair_schedules=set(permutations(tuple(combinations(range(3),2)),2))
    actual_schedules={tuple(tuple(p) for p in m['pair_rows']) for m in menus if m['kind']=='pair_rotation'}
    check('all_six_different_pair_schedules', actual_schedules==ordered_pair_schedules)
    probs=np.array([0.,.5,.75]); expected_cost={}
    for m in menus:
        rows=np.array(m['price_rows']); pairs=np.array(m['pair_rows'])
        check('menu_price_shape_'+m['id'], rows.shape==(2,3) and np.isin(rows,[0,1,2]).all())
        check('menu_pair_support_'+m['id'], all(len(set(p))==2 and set(np.flatnonzero(rows[e]))<=set(p) for e,p in enumerate(pairs)))
        check('menu_worst_budget_'+m['id'], close(m['worst_payment'],.4*rows.sum()) and .4*rows.sum()<=1.6+1e-12)
        expected_cost[m['id']]=float(.4*np.sum(rows*probs[rows]))

    check('all22_zero_uniform_scores',len(actions['scores'])==22 and all(s['uniform_separation']==0 and s['minimum_pair_separation_probability']==0 for s in actions['scores']))
    for score in actions['scores']:
        check('score_cost_'+score['menu_id'],close(score['expected_payment'],expected_cost[score['menu_id']]))
    check('guided_zero_from_cost_tie', all(a['policies']['guided']=='zero' for a in actions['actions']))
    check('full_families_bound_in_actions', actions['full_historical_families']==witness['families'])

    historical=read(cfg['historical_W']['path'])
    saved_families={family(p) for p in historical['points']}
    witness_families={tuple(sorted(tuple(s) for s in f)) for f in witness['families']}
    check('all_history_families_retained',saved_families==witness_families and len(saved_families)==4)
    history=pickle.loads(Path(cfg['history']['path']).read_bytes())
    check('history_shape_training',sum(len(s['P_terminal']) for s in history['training'])==288)
    check('history_shape_validation',sum(len(s['P_terminal']) for s in history['validation'])==288)
    def merged_loss(R,X,split):
        values=[np.abs(np.asarray(s['drop_target'])-np.asarray(s['P_terminal'])@R.T-np.asarray(s['Q_terminal'])@X.T).ravel() for s in history[split]]
        return float(np.concatenate(values).mean())
    shared=None; witness_diagnostics=[]
    for j,w in enumerate(witness['witnesses']):
        R,X=matrices(w['family'],w['r'],w['x'])
        rv=np.array(w['r']); xv=np.array(w['x'])
        check(f'witness_{j}_bounds', (rv>=0).all() and (xv>=0).all() and (rv<=2).all() and (xv<=2).all())
        train=merged_loss(R,X,'training'); val=merged_loss(R,X,'validation')
        check(f'witness_{j}_loss_replay',close(train,w['train_loss']) and close(val,w['validation_loss']))
        check(f'witness_{j}_inside_face',train<=witness['train_cap']+1e-12 and val<=witness['validation_cap']+1e-12)
        if shared is None: shared=(R,X)
        check(f'witness_{j}_identical_RX',np.array_equal(R,shared[0]) and np.array_equal(X,shared[1]))
        check(f'witness_{j}_additional_weights_zero',all(rv[t]==0 and xv[t]==0 for t,s in enumerate(w['family']) if len(s)>1))
        witness_diagnostics.append(dict(family_index=j, train_loss=train, validation_loss=val,
                                        training_slack=witness['train_cap']-train,validation_slack=witness['validation_cap']-val))
    check('common_witness_proves_all_prices_zero',len(witness['witnesses'])==4 and witness['all_prices_uniform_separation']==0.)

    flowmap={f['behavior']:f for f in cfg['flows']}
    expected_files={f"{f['id']}_{case['id']}_{m['id']}.json":(f,case,m) for f in cfg['flows'] for case in cfg['cases'] for m in menus}
    paths=sorted((root/'execution_receipts').glob('*.json'))
    check('receipts_within_declared_jobs',all(p.name in expected_files for p in paths))
    reuse=defaultdict(list); records=[]; unreadable=[]
    wave=np.r_[np.full(4,-.2),np.full(4,.2)]
    event_totals=Counter(); unique_payment={}
    for path in paths:
        try: d=read(path)
        except (json.JSONDecodeError,PermissionError) as error:
            unreadable.append(dict(name=path.name,error=type(error).__name__));continue
        f,case,m=expected_files[path.name]; ctl=d['controls']; prefix=path.stem
        price=np.array(ctl['prices'],int); accepted=np.array(ctl['accepted'],bool); orientation=np.array(ctl['orientation'],int)
        pairs=np.array(ctl['pairs'],int); dp=np.array(ctl['delta_p_kw'],float); dq=np.array(ctl['delta_q_kvar'],float)
        expected_price=np.array(m['price_rows'])[np.arange(4)%2]; expected_pairs=np.array(m['pair_rows'])[np.arange(4)%2]
        check(prefix+'_price_and_pair',np.array_equal(price,expected_price) and np.array_equal(pairs,expected_pairs))
        check(prefix+'_control_shapes', accepted.shape==(4,3) and orientation.shape==(4,) and dp.shape==(4,8,3) and dq.shape==(4,8,3))
        expected_z=[];expected_s=[]
        for e in range(4):
            # Independent counter-stream reconstruction, no original sampler.
            u=np.random.default_rng(np.random.SeedSequence([f['behavior'],76001,e])).random(3)
            expected_z.append(u<probs[expected_price[e]])
            bit=np.random.default_rng(np.random.SeedSequence([f['behavior'],76002,e])).integers(0,2)
            expected_s.append(1 if bit else -1)
        check(prefix+'_participation_rng',np.array_equal(accepted,np.array(expected_z)))
        check(prefix+'_direction_rng',np.array_equal(orientation,np.array(expected_s)))
        expected_dp=np.zeros((4,8,3))
        for e,(i,j) in enumerate(expected_pairs):
            expected_dp[e,:,i]=expected_s[e]*wave*expected_z[e][i]
            expected_dp[e,:,j]=-expected_s[e]*wave*expected_z[e][j]
        check(prefix+'_full_task_waveform',close(dp,expected_dp))
        check(prefix+'_reactive_waveform',close(dq,.35*expected_dp))
        power=.2+dp; energy=np.cumsum(power,axis=1)*.25
        check(prefix+'_task_power_bounds',(power>=-1e-12).all() and (power<=.4+1e-12).all())
        check(prefix+'_task_energy_bounds',(energy>=-1e-12).all() and (energy<=.4+1e-12).all() and close(energy[:,-1],.4))
        check(prefix+'_net_increment_zero',close(dp.sum(axis=1)*.25,0.))
        payments=.2*np.sum(price*accepted,axis=1)
        check(prefix+'_payment_replay',close(payments,ctl['committed_payment']) and close(payments,ctl['delivery_payment_if_executed']) and close(payments.sum(),d['actual_payment']))
        check(prefix+'_budget',payments.sum()<=1.6+1e-12 and close(d['worst_payment'],.4*np.array(m['price_rows']).sum()))
        check(prefix+'_seed_binding',d['behavior_seed']==f['behavior'] and d['measurement_seed']==f['measurement'] and d['natural_seed']==884001 and ctl['seed']==f['behavior'])
        check(prefix+'_action_binding',d['actions_sha256']==generated['actions_sha256'] and d['menu']==m)
        check(prefix+'_postfreeze_timestamp',datetime.fromisoformat(d['created_utc'])>=datetime.fromisoformat(generated['created_utc']))
        check(prefix+'_event_roles',d['full_event_roles']==['training','training','validation','validation'])
        check(prefix+'_flags',ctl['no_forced_participation'] and ctl['partial_response_cost_retained'] and ctl['acceptance_before_code'] and not d['empirical_response_calibration'] and not d['AC_validation'])
        for e,event in enumerate(d['events']):
            k=f'{prefix}_e{e}'
            delivered=float(np.maximum(-dp[e],0).sum()*.25)
            check(k+'_receipt',event['event']==e and close(event['payment'],payments[e]) and close(event['delivered_reduction_kWh'],delivered) and np.array_equal(event['accepted'],accepted[e].astype(int)) and event['orientation']==orientation[e])
            active=int(np.count_nonzero(accepted[e])); event_totals['all_events']+=1
            if active==0:event_totals['none']+=1
            elif active==1:event_totals['single']+=1
            else:event_totals['pair']+=1
        identity=dict(prices=price.tolist(),accepted=accepted.astype(int).tolist(),orientation=orientation.tolist(),pairs=pairs.tolist(),dp=dp.tolist(),dq=dq.tolist(),payments=payments.tolist())
        key=(f['id'],m['id']); reuse[key].append((case['id'],c_hash(identity),float(payments.sum())))
        unique_payment[key]=float(payments.sum())
        records.append(dict(receipt=path.name,sha256=hash_file(path),flow=f['id'],case=case['id'],menu_id=m['id'],control_sha256=c_hash(identity),actual_payment=float(payments.sum())))
    reuse_rows=[]
    for (flow,mid), group in sorted(reuse.items()):
        check(flow+'_'+mid+'_same_control_across_true_models',len({h for _,h,_ in group})==1)
        check(flow+'_'+mid+'_no_duplicate_case_receipts',len({c for c,_,_ in group})==len(group))
        reuse_rows.append(dict(flow=flow,menu_id=mid,case_receipts=len(group),same_controls=len({h for _,h,_ in group})==1,
                               complete_seven_case_reuse=len(group)==7,actual_payment=group[0][2]))
    evaluation_audit=None
    if (root/'evaluations.json').exists():
        evaluations=read(root/'evaluations.json'); jobs=actions['ordered_jobs']
        check('evaluation_planned_308_recorded',len(evaluations)==cfg['expected_maps']==len(jobs))
        evaluation_keys=[(v['flow'],v['case'],v['menu_id']) for v in evaluations]
        job_keys=[(v['flow'],v['case'],v['menu_id']) for v in jobs]
        check('evaluation_no_drop_duplicate_reorder',evaluation_keys==job_keys and len(set(evaluation_keys))==len(job_keys))
        receipts_index={v['receipt']:v for v in records}
        for e in evaluations:
            key=f"{e['flow']}_{e['case']}_{e['menu_id']}"
            filename=key+'.json'; has_receipt=filename in receipts_index
            check(key+'_execution_receipt_presence',(e['execution_status']=='executed')==has_receipt)
            if e['execution_status']=='executed':
                check(key+'_evaluation_payment_binding',close(e['actual_payment'],receipts_index[filename]['actual_payment']))
                check(key+'_evaluation_receipt_binding',Path(e['execution_receipt']).name==filename)
            else:
                check(key+'_unexecuted_zero_fee',e['execution_status']=='not_executed' and e['actual_payment']==0.)
                check(key+'_unexecuted_marked_unresolved',e['status']=='unresolved' and not e.get('evaluated'))
        execution_counts=dict(Counter(v['execution_status'] for v in evaluations))
        method_counts=dict(Counter(v['status'] for v in evaluations))
        payments_by_flow={f['id']:sum(v['actual_payment'] for v in reuse_rows if v['flow']==f['id']) for f in cfg['flows']}
        reuse_range=dict(Counter(v['case_receipts'] for v in reuse_rows))
        evaluation_audit=dict(recorded=len(evaluations),execution_counts=execution_counts,record_counts=method_counts,
                              unique_control_payments_by_flow=payments_by_flow,
                              control_reuse_histogram_cases_per_flow_menu=reuse_range,
                              payment_interpretation='one payment for each of 44 counterfactual flow/menu deployments; never seven times the same control payment; not one deployable policy bill',
                              no_simulation_for_unexecuted=True)
    failures=[c for c in checks if not c['ok']]
    run_ended=read(root/'run_ended.json') if (root/'run_ended.json').exists() else None
    result=dict(status='passed_snapshot' if not failures else 'failed',created_utc=datetime.now(timezone.utc).isoformat(),
                source_root=str(root),audit_source_sha256=hash_file(__file__),
                protocol_sha256=hash_file(root/'protocol.json'),actions_sha256=hash_file(root/'actions_frozen.json'),
                no_optimizer_no_sampler_import=True,run_ended=run_ended,checks=len(checks),failures=failures,
                snapshot=dict(declared_maps=cfg['expected_maps'],receipt_paths_seen=len(paths),audited_receipts=len(records),
                              still_unseen=cfg['expected_maps']-len(paths),concurrent_unreadable=unreadable,
                              independent_behavior_streams=2,declared_control_trajectories=44,seen_control_trajectories=len(reuse),
                              full_seven_case_reuses=sum(r['complete_seven_case_reuse'] for r in reuse_rows),
                              observed_event_totals=dict(event_totals),events_are_reused_across_truths=True),
                witness=dict(families=len(witness['witnesses']),same_RX=True,per_family=witness_diagnostics,
                             interpretation='all complete historical faces share exact RX, so uniform pairwise minimum is zero for any input'),
                controls=reuse_rows,records=records,
                evaluations=evaluation_audit,
                prior_pure_tests='13 tests already completed separately; not rerun or added to audit-check count',
                interpretation='checks are deterministic replays and are not independent samples or statistical coverage')
    return result,checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(DEFAULT)); a=p.parse_args()
    result,checks=audit(a.root)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for filename,obj in [(f'audit_{stamp}.json',result),(f'checks_{stamp}.json',checks),('latest.json',result)]:
        (HERE/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','controls','witness','run_ended')},ensure_ascii=False))
    sys.exit(1 if result['failures'] else 0)
