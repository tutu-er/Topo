"""Independent read-only B run audit; imports no project model/selector/solver.

Run only after the bounded controller has ended.  Rebuild original observations,
L1 losses, acceptance, validation, contraction, stochastic controls and counts.
No new LP, AC, fitting, or actual future is generated.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter
from datetime import datetime
import argparse, json, pickle, random, sys
import numpy as np
import pandas as pd


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return sha256(Path(p).read_bytes()).hexdigest()
def canonical(f): return tuple(sorted({tuple(sorted(s)) for s in f},key=lambda s:(len(s),s)))
def eff(p): return {tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x']) if max(r,x)>1e-9}
def mat(p,n=3):
    R=np.zeros((n,n));X=np.zeros((n,n))
    for s,r,x in zip(p['supports'],p['r'],p['x']):
        R[np.ix_(s,s)]+=r;X[np.ix_(s,s)]+=x
    return R,X


class Audit:
    def __init__(self): self.count=0;self.failures=[];self.largest={}
    def check(self,ok,label,detail=None):
        self.count+=1
        if not bool(ok): self.failures.append(dict(label=label,detail=detail))
    def close(self,a,b,label,atol=1e-12):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if a.shape!=b.shape: self.check(False,label,dict(shapes=[a.shape,b.shape]));return
        d=float(np.max(abs(a-b),initial=0.));self.largest[label.split(':')[0]]=max(d,self.largest.get(label.split(':')[0],0.))
        self.check(d<=atol,label,dict(max_abs=d,tolerance=atol) if d>atol else None)


def observed(scenes,audit,label):
    chunks=[[],[],[]];indexes=[];labels=None
    for sc in scenes:
        p=sc['P_terminal'];q=sc['Q_terminal'];v=sc['V_terminal'];vr=sc['V_root_observed']
        if labels is None: labels=list(p.columns)
        audit.check(list(p.columns)==labels and p.index.equals(q.index) and p.index.equals(v.index) and p.index.equals(vr.index),label+':alignment')
        audit.check(p.columns.equals(q.columns) and p.columns.equals(v.columns),label+':columns')
        y=np.asarray(vr)[:,None]**2-np.asarray(v)**2
        audit.close(y,np.asarray(sc['drop_target']),label+':observed_drop',0.)
        for dst,value in zip(chunks,(np.asarray(p),np.asarray(q),y)): dst.append(value)
        indexes.extend(p.index)
    audit.check(len(indexes)==len(set(indexes)) and indexes==sorted(indexes),label+':chronology')
    return tuple(np.concatenate(x) for x in chunks)


def losses(p,train,valid):
    R,X=mat(p)
    return tuple(float(np.mean(abs(d[0]@R.T+d[1]@X.T-d[2]))) for d in (train,valid))


def chosen(points,cal):
    best=min(p['validation_mae'] for p in points)
    eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
    idx=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
    p=points[idx]
    keep=[i for i,s in enumerate(p['supports']) if len(s)==1 or cal['a']*p['r'][i]+cal['b']*p['x'][i]+1e-10>=.16*cal['median_root_depth']]
    final=dict(supports=[p['supports'][i] for i in keep],r=[p['r'][i] for i in keep],x=[p['x'][i] for i in keep])
    return idx,p,final


def audit_method(m,train,valid,cal,A,label):
    if m.get('status')!='complete': return dict(status=m.get('status'))
    points=m['points']
    for i,p in enumerate(points):
        tr,va=losses(p,train,valid)
        A.close([tr,va],[p['objective'],p['validation_mae']],label+':point_L1')
        A.check(min(p['r']+p['x'])>=-1e-12 and max(p['r']+p['x'])<=2+1e-12,label+':point_bounds')
    idx,raw,final=chosen(points,cal);s=m['selected']
    A.check(idx==s['selection_index'],label+':selected_index')
    A.check(canonical(raw['supports'])==canonical(s['selected_uncontracted']['supports']),label+':selected_raw_support')
    A.close(raw['r']+raw['x'],s['selected_uncontracted']['r']+s['selected_uncontracted']['x'],label+':selected_raw_weights')
    savedfinal=dict(supports=s.get('support_indices',s['supports']),r=s.get('r_values',s['r']),x=s.get('x_values',s['x']))
    A.check(canonical(final['supports'])==canonical(savedfinal['supports']),label+':contract_support')
    A.close(final['r']+final['x'],savedfinal['r']+savedfinal['x'],label+':contract_weights')
    tr,va=losses(final,train,valid)
    if 'train_mae' in s: A.close([tr,va],[s['train_mae'],s['validation_mae']],label+':final_L1')
    else: A.close(s['validation_mae'],raw['validation_mae'],label+':diagnostic_precontract_val')
    accepted=cross=0;stop=Counter()
    for name,path in m.get('paths',{}).items():
        cursor=0
        for d in path['decisions']:
            curr=path['path'][cursor]
            A.close(d['current'],curr['objective'],label+':decision_current')
            best=None
            for c in d['candidates']:
                if best is None or c['objective']<best['objective']-1e-12: best=c
                cand=set(c['candidate'])
                removed=[ss for ss in curr['supports'] if set(ss)&cand and not(set(ss)<=cand or cand<=set(ss))]
                A.check(canonical(removed)==canonical(c['removed']),label+':crossing_deletion')
            gain=0. if best is None else curr['objective']-best['objective']
            A.close(gain,d['gain'],label+':decision_gain')
            take=best is not None and gain>1e-10+.001*curr['objective']
            if take:
                accepted+=1;cross+=bool(best['removed']);cursor+=1
                A.check(cursor<len(path['path']),label+':accepted_path_available')
                if cursor<len(path['path']): A.close(path['path'][cursor]['objective'],best['objective'],label+':accepted_objective')
        A.check(cursor==len(path['path'])-1,label+':accepted_path_count')
        stop[path['reason']]+=1
    return dict(raw=raw,final=final,accepted=accepted,accepted_cross=cross,path_count=len(m.get('paths',{})),stops=dict(stop),all_scans=m.get('all_scans_complete',True))


def truth(case,cfg):
    z=np.zeros(3);z[case['pair']]=1.
    d=cfg['L']-3*cfg['b']-2*case['c']*z
    K=np.diag(d)+cfg['b']*np.ones((3,3))+case['c']*np.outer(z,z)
    target={(i,) for i,x in enumerate(d) if x>0}
    if cfg['b']>0:target.add((0,1,2))
    if case['c']>0:target.add(tuple(case['pair']))
    return K/(1+cfg['kappa']),K/(1+cfg['kappa']),target


def control_and_observation(receipt,packet,cfg,case,flow,A,label):
    draw=receipt['controls'];prices=np.asarray(draw['prices'],int);Z=np.asarray(draw['accepted'],bool)
    dirs=np.asarray(draw['orientation'],int);pairs=np.asarray(draw['pairs'],int)
    dp=np.asarray(draw['delta_p_kw'],float);dq=np.asarray(draw['delta_q_kvar'],float)
    wave=np.r_[np.full(4,-.2),np.full(4,.2)]
    expected=np.zeros((4,8,3));p=np.array([0.,.5,.75])
    for e in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([flow['behavior'],76001,e]));zz=rng.random(3)<p[prices[e]]
        code=np.random.default_rng(np.random.SeedSequence([flow['behavior'],76002,e]));ss=1 if code.integers(0,2) else -1
        A.check(np.array_equal(zz,Z[e]) and ss==dirs[e],label+':random_stream')
        i,j=pairs[e];expected[e,:,i]=ss*wave*zz[i];expected[e,:,j]=-ss*wave*zz[j]
    A.close(dp,expected,label+':delivered_P',0.);A.close(dq,.35*dp,label+':delivered_Q',0.)
    device=.2+dp;energy=np.cumsum(device,axis=1)*.25
    A.check(bool(((device>=0)&(device<=.4)).all()),label+':task_power')
    A.close(energy[:,-1,:],np.full((4,3),.4),label+':terminal_energy')
    A.close(np.sum(dp,axis=1)*.25,np.zeros((4,3)),label+':net_energy')
    pay=.2*np.sum(prices*Z,axis=1);worst=.2*prices.sum()
    A.close(pay,draw['delivery_payment_if_executed'],label+':event_payments')
    A.close(pay.sum(),receipt['actual_payment'],label+':actual_payment')
    A.close(worst,receipt['worst_payment'],label+':worst_payment')
    A.check(pay.sum()<=worst+1e-12 and worst<=1.6+1e-12,label+':budget')
    R,X,_=truth(case,cfg)
    scenes=packet['training']+packet['validation'];reg=cfg['regime']
    for e,sc in enumerate(scenes):
        day=6+e;rng=np.random.default_rng(np.random.SeedSequence([cfg['seeds']['natural'],day]));t=np.arange(96)/96
        scalar=1+.15*np.sin(2*np.pi*(t-.12))+.03*rng.uniform(-1,1,96)
        p0=np.repeat((scalar/cfg['base_kw'])[:,None],3,axis=1);q0=.35*p0;root=1+.003*np.sin(2*np.pi*(t+.07*day))
        du=np.zeros_like(p0);dv=np.zeros_like(q0);du[76:84]=dp[e]/200.;dv[76:84]=dq[e]/200.
        pf=p0+du;qf=q0+dv;drop=cfg['L']*p0[:,[0]]*np.ones((1,3))+du@R.T+dv@X.T
        v=np.sqrt(root[:,None]**2-drop)
        rng=np.random.default_rng(np.random.SeedSequence([flow['measurement'],day]))
        ep,eq,ev,er=(rng.standard_normal(shape) for shape in (pf.shape,qf.shape,v.shape,root.shape))
        pm=pf*(1+reg['pq_relative']*ep);qm=qf*(1+reg['pq_relative']*eq)
        vm=v*(1+reg['voltage_relative']*ev);rm=root*(1+reg['voltage_relative']*er)
        for key,val in (('P_terminal',pm),('Q_terminal',qm),('V_terminal',vm),('V_root_observed',rm)):
            A.close(np.asarray(sc[key]),val,label+':observed_replay_'+key,1e-14)


def rowkey(row):return tuple(row[k] for k in ('flow','case','menu_id'))
def rowcomplete(row,methods):
    return all(row.get('evaluated',{}).get(m,{}).get('status')=='complete' and row['evaluated'][m].get('all_scans_complete',True) for m in methods)
def canonical_object_hash(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def LP_receipts(record):
    receipts=list(record.get('shared_preprocessing_LP_solves',[]))
    for method in record.get('methods',{}).values():receipts.extend(method.get('LP_solves',[]))
    receipts.extend(record.get('historical_family_diagnostic',{}).get('LP_solves',[]))
    return receipts


def main(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not (root/'run_ended.json').is_file():raise RuntimeError('wait for completion to end; no fitting is initiated')
    A=Audit();cfg=read(root/'protocol.json');source=Path(cfg['source_root']);original=read(source/'protocol.json')
    oldrows=read(source/'evaluations.json');newrows=read(root/'completion_rows.json');merged=read(root/'merged_evaluations.json')
    summary=read(root/'summary.json');end=read(root/'run_ended.json');actions=read(source/'actions_frozen.json')
    methods=original['methods'];idxold={rowkey(r):r for r in oldrows};idxmerged={rowkey(r):r for r in merged}
    original_complete={k:r for k,r in idxold.items() if rowcomplete(r,methods)}
    pending=set(idxold)-set(original_complete);updated={rowkey(r) for r in newrows}
    A.check(sha(root/'protocol.json')==read(root/'protocol.sha256.json')['sha256'],'completion_protocol_sha')
    for kind in ('source_sha256','input_sha256'):
        for p,h in cfg[kind].items():A.check(sha(p)==h,kind+':unchanged')
    A.check(len(oldrows)==308 and len(original_complete)==217 and len(pending)==91,'original_denominators')
    A.check(len(newrows)==91 and len(updated)==91 and updated==pending,'only_original91_updated')
    A.check(len(merged)==308 and len(idxmerged)==308 and set(idxmerged)==set(idxold),'merged_job_scope')
    A.check([rowkey(r) for r in oldrows]==[rowkey(r) for r in merged],'original_job_order')
    for k,r in original_complete.items():
        A.check(idxmerged[k]==r,'completed_original_object_equal')
        A.check(canonical_object_hash(idxmerged[k])==canonical_object_hash(r),'completed_original_serialized_object_equal')
    A.check(set(map(tuple,cfg['pending_keys']))==pending,'protocol_pending_scope')
    A.check(cfg['original_protocol_sha256']==sha(source/'protocol.json'),'original_protocol_unchanged')
    A.check(cfg['original_evaluations_sha256']==sha(source/'evaluations.json'),'old_evaluations_bytes_unchanged')
    A.check(cfg['original_actions_sha256']==sha(source/'actions_frozen.json'),'original_actions_unchanged')
    A.check(all(r['actions_sha256']==cfg['original_actions_sha256'] for r in merged),'all_merged_actions_binding')
    A.check(len(actions['scores'])==22 and all(s['uniform_separation']==0 for s in actions['scores']),'no_price_reoptimization')
    A.check(all(a['policies']['guided']=='zero' for a in actions['actions']),'original_guide_zero')
    A.check(datetime.fromisoformat(cfg['created_utc'])<=datetime.fromisoformat(read(root/'run_started.json')['created_utc']),'protocol_before_completion')
    # Count old and new solver receipts separately. The resumed record must not
    # recount the already-completed shared preprocessor.
    oldrecordpaths=sorted(p for p in (source/'records').glob('s_*.json') if not p.name.endswith('_current_state.json'))
    newrecordpaths=sorted(p for p in (root/'records').glob('s_*.json') if not p.name.endswith('_current_state.json'))
    oldLP=sum(len(LP_receipts(read(p))) for p in oldrecordpaths)
    newLP=sum(len(LP_receipts(read(p))) for p in newrecordpaths)
    A.check(oldLP==3000 and oldLP==cfg['original_LP_calls'],'old_LP_receipts')
    A.check(newLP==summary['completion_LP_calls']==end['new_LP_calls'],'new_LP_receipts')
    A.check(oldLP+newLP==summary['cumulative_LP_calls']==end['cumulative_LP_calls'],'cumulative_LP_receipts')
    A.check(newLP<=cfg['computation']['max_LP_calls']==1800,'new_LP_cap')
    for p in newrecordpaths:
        for rec in LP_receipts(read(p)):A.check(rec['status']==2,'new_LP_status')
    history=pickle.loads(Path(original['history']['path']).read_bytes())
    bycase={c['id']:c for c in original['cases']};byflow={f['id']:f for f in original['flows']};bymenu={m['id']:m for m in original['menus']}
    cache={};mechanisms=[];partial_reused=0;reused_complete=0;executed_new=0
    for row in newrows:
        old=idxold[rowkey(row)]
        A.check(row.get('execution_status')=='executed','completion_execution')
        receipt=read(row['execution_receipt']);h=receipt['packet_sha256']
        if row.get('completion_reused_execution'):
            partial_reused+=1
            A.check(old.get('execution_status')=='executed' and row['execution_receipt']==old['execution_receipt'],'delivered_receipt_reused')
        else:
            executed_new+=1;A.check(old.get('execution_status')=='not_executed','only_unexecuted_generated')
            A.check(receipt['completion_protocol_sha256']==sha(root/'protocol.json'),'new_receipt_completion_binding')
            A.check(datetime.fromisoformat(receipt['created_utc'])>=datetime.fromisoformat(cfg['created_utc']),'new_delivery_after_protocol')
        A.check(receipt['actions_sha256']==cfg['original_actions_sha256'],'new_receipt_old_price_binding')
        A.check(receipt['behavior_seed']==byflow[row['flow']]['behavior'] and receipt['measurement_seed']==byflow[row['flow']]['measurement'],'unchanged_response_measurement_seeds')
        A.check(receipt['natural_seed']==original['seeds']['natural']==884001,'unchanged_natural_seed')
        A.check(receipt['menu']==bymenu[row['menu_id']],'unchanged_menu')
        A.check(receipt['noise_regime']==original['regime'],'unchanged_noise')
        packetpath=root/'private_observations/packets'/(h+'.pkl')
        if not packetpath.exists():packetpath=source/'private_observations/packets'/(h+'.pkl')
        A.check(sha(packetpath)==h,'packet_hash');packet=pickle.loads(packetpath.read_bytes())
        control_and_observation(receipt,packet,original,bycase[row['case']],byflow[row['flow']],A,'completion_delivery_replay')
        A.close(row['actual_payment'],receipt['actual_payment'],'row_actual_payment')
        recordpath=Path(row['completion_record_path']);data=read(recordpath)
        if row.get('completion_reused_complete_fit'):
            reused_complete+=1;A.check(recordpath.parent==source/'records','completed_old_fit_read_only_cache')
            A.check(all(m.get('status')=='complete' for m in data['methods'].values()),'cached_fit_complete')
        if row.get('completion_reused_execution'):
            A.check(data['completion_reused_preprocessing'] is True,'preprocessing_reuse_flag')
            A.check(data['current_state_path']==read(source/'records'/(old['fit_id']+'.json'))['current_state_path'],'preprocessing_same_source')
            A.check(not data['shared_preprocessing_LP_solves'],'old_preprocessing_not_recounted')
            A.check(data['original_preprocessing_receipt_count']==3,'reused_preprocessing_receipts')
            A.check(data['completion_current_state_sha256']==sha(data['current_state_path']),'reused_state_exact_hash')
        fid=row['fit_id']
        if fid not in cache:
            train=observed(history['training']+packet['training'],A,'merged_train');valid=observed(history['validation']+packet['validation'],A,'merged_val')
            A.check(train[0].shape==(480,3) and valid[0].shape==(480,3),'all_original_rows_retained')
            checked={m:audit_method(data['methods'][m],train,valid,data['applied_calibration'],A,'completion_method') for m in methods}
            diag=data.get('historical_family_diagnostic',{})
            if diag.get('status')=='complete':audit_method(diag,train,valid,data['applied_calibration'],A,'completion_history_families')
            cache[fid]=(data,checked)
        data,checked=cache[fid];_,_,target=truth(bycase[row['case']],original)
        for m in methods:
            ev=row['evaluated'][m];v=checked[m]
            A.check(ev['status']=='complete','completion_method_status')
            rawerr=len(eff(v['raw'])^target);finalerr=len(eff(v['final'])^target)
            A.check(ev['raw_errors']==rawerr and ev['final_errors']==finalerr,'completion_errors')
            A.check(set(map(tuple,ev['target']))==target,'completion_target')
            A.check(ev['path_count']==v['path_count'],'corrected_path_metadata')
            points=data['methods'][m]['points'];minval=min(p['validation_mae'] for p in points)
            correct=[p for p in points if eff(p)==target];eligible=[p for p in correct if p['validation_mae']<=1.03*minval+1e-12]
            A.check(ev['correct_saved_count']==len(correct) and ev['correct_eligible_count']==len(eligible),'completion_saved_recovery')
            removed=eff(v['raw'])-eff(v['final'])
            mechanisms.append(dict(flow=row['flow'],case=row['case'],menu_id=row['menu_id'],method=m,fit_id=fid,
                raw_errors=rawerr,final_errors=finalerr,correct_saved=bool(correct),correct_eligible=bool(eligible),
                accepted=v['accepted'],accepted_cross=v['accepted_cross'],path_count=v['path_count'],all_scans=v['all_scans'],
                true_deleted=[list(s) for s in sorted(removed&target)],false_deleted=[list(s) for s in sorted(removed-target)],
                final_missing=[list(s) for s in sorted(target-eff(v['final']))],final_extra=[list(s) for s in sorted(eff(v['final'])-target)]))
    complete_total=sum(rowcomplete(r,methods) for r in merged)
    A.check(complete_total==summary['merged_complete_maps']==308,'all308_complete')
    A.check(partial_reused==summary['reused_delivered_execution_maps']==1,'partial_delivery_count')
    A.check(executed_new==summary['new_execution_maps']==90,'new_execution_count')
    A.check(reused_complete==summary['reused_completed_observation_fits']==15,'completed_cache_reuse_count')
    distinct_pending_fits=len({r['fit_id'] for r in newrows})
    A.check(len(cache)==distinct_pending_fits,'all_distinct_completion_fits_seen')
    aggregate=[]
    for method in methods:
        group=[r['evaluated'][method] for r in merged]
        aggregate.append(dict(method=method,complete_maps=len(group),raw_errors=sum(r['raw_errors'] for r in group),
            final_errors=sum(r['final_errors'] for r in group),raw_full=sum(r['raw_errors']==0 for r in group),full=sum(r['final_errors']==0 for r in group)))
        reported=[r for r in summary['menu_results'] if r['method']==method]
        A.check(sum(r['complete'] for r in reported)==308,'summary_complete_maps')
        A.check(sum(r['final_errors'] for r in reported)==aggregate[-1]['final_errors'],'summary_final_errors')
        A.check(sum(r['raw_errors'] for r in reported)==aggregate[-1]['raw_errors'],'summary_raw_errors')
        A.check(sum(r['full'] for r in reported)==aggregate[-1]['full'],'summary_full_recovery')
    result=dict(status='passed' if not A.failures else 'failed',checks=A.count,failures=A.failures,largest_differences=A.largest,
        original_complete_maps=217,completion_maps=len(newrows),new_execution_maps=executed_new,reused_delivered_maps=partial_reused,
        reused_completed_fits=reused_complete,distinct_completion_fit_ids=distinct_pending_fits,merged_complete_maps=complete_total,new_record_files=len(newrecordpaths),
        original_LP_receipts=oldLP,new_LP_receipts=newLP,cumulative_LP_receipts=oldLP+newLP,aggregate_methods=aggregate,
        original_root_byte_hashes_unchanged=True,completed_original_objects_unchanged=True,
        source_sha256=sha(__file__),new_LP_calls=0,new_AC_calls=0,new_actual_future_generated=False,
        scope='independent read-only evidence reconstruction; no project model, controller or selector imported')
    (out/'audit_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'mechanisms.json').write_text(json.dumps(mechanisms,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failures','largest_differences')},ensure_ascii=False))
    if A.failures:print(json.dumps(A.failures[:10],ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=main(a.root,a.out);sys.exit(0 if r['status']=='passed' else 1)
