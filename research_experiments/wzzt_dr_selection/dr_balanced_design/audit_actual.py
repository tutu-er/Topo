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


def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not (root/'run_ended.json').exists(): raise RuntimeError('wait for bounded controller end; audit never starts or repeats its run')
    A=Audit();cfg=read(root/'protocol.json');actions=read(root/'actions_frozen.json');wit=read(root/'planning/common_witness.json');rows=read(root/'evaluations.json')
    A.check(sha(root/'protocol.json')==read(root/'protocol.sha256.json')['sha256'],'protocol_hash')
    for p,h in cfg['source_sha256'].items():A.check(sha(p)==h,'source_hash:'+str(p))
    for key in ('history','history_state','historical_W','math_gate'):
        A.check(sha(cfg[key]['path'])==cfg[key]['sha256'],'input_hash:'+key)
    start=read(root/'actual_generation_started.json');A.check(start['actions_sha256']==sha(root/'actions_frozen.json'),'actions_time_boundary_hash')
    A.check(datetime.fromisoformat(cfg['created_utc'])<=datetime.fromisoformat(start['created_utc']),'protocol_before_actual')
    A.check(actions['future_generated'] is False and actions['truth_read_by_planner'] is False,'actions_information_declaration')
    jobset={(j['flow'],j['case'],j['menu_id']) for j in actions['ordered_jobs']}
    A.check(len(jobset)==308 and len(actions['ordered_jobs'])==308,'job_denominator')
    A.check(len(rows)==308 and len({(r['flow'],r['case'],r['menu_id']) for r in rows})==308,'record_denominator')
    history=pickle.loads(Path(cfg['history']['path']).read_bytes());ht=observed(history['training'],A,'history_train');hv=observed(history['validation'],A,'history_val')
    hist=read(cfg['historical_W']['path']);A.check(len(hist['points'])==19,'history_saved_denominator')
    allfam={canonical(p['family']) for p in hist['LP_solves']};A.check(allfam=={canonical(f) for f in wit['families']},'full_declared_history_families')
    ref=None
    for w in wit['witnesses']:
        tr,va=losses(dict(supports=w['family'],r=w['r'],x=w['x']),ht,hv)
        A.close([tr,va],[w['train_loss'],w['validation_loss']],'witness_L1')
        A.check(tr<=wit['train_cap']+1e-12 and va<=wit['validation_cap']+1e-12,'witness_face')
        R,X=mat(dict(supports=w['family'],r=w['r'],x=w['x']))
        if ref is None:ref=R,X
        A.check(np.array_equal(R,ref[0]) and np.array_equal(X,ref[1]),'witness_RX_exact')
    A.check(len(actions['scores'])==22 and all(s['uniform_separation']==0 and s['minimum_pair_separation_probability']==0 for s in actions['scores']),'uniform_zero_score')
    A.check(all(a['policies']['guided']=='zero' for a in actions['actions']),'guided_zero')
    bycase={c['id']:c for c in cfg['cases']};byflow={f['id']:f for f in cfg['flows']}
    records={};summ=Counter();mechanisms=[];seenreceipts=0;unique_packet_sha={};source_hashes={}
    for row in rows:
        if row.get('execution_status')!='executed':continue
        receipt=read(row['execution_receipt']);seenreceipts+=1
        A.check(receipt['actions_sha256']==start['actions_sha256'],'receipt_frozen_actions')
        h=receipt['packet_sha256'];path=root/'private_observations'/'packets'/(h+'.pkl')
        A.check(sha(path)==h,'packet_hash');packet=pickle.loads(path.read_bytes())
        unique_packet_sha[h]=str(path)
        A.close(row['actual_payment'],receipt['actual_payment'],'row_payment')
        control_and_observation(receipt,packet,cfg,bycase[row['case']],byflow[row['flow']],A,'delivery')
        fid=row['fit_id']
        if fid not in records:
            data=read(root/'records'/(fid+'.json'));train=observed(history['training']+packet['training'],A,'merged_train');valid=observed(history['validation']+packet['validation'],A,'merged_val')
            A.check(train[0].shape==(480,3) and valid[0].shape==(480,3),'full_rows')
            A.check(data['truth_input_allowed'] is False,'identifier_truth_boundary')
            checked={};cal=data['applied_calibration']
            for method,m in data['methods'].items():
                checked[method]=audit_method(m,train,valid,cal,A,'method')
                for s in m.get('LP_solves',[]):summ['LP_receipts']+=1;A.check(s['status']==2,'LP_status')
            for s in data.get('shared_preprocessing_LP_solves',[]):summ['LP_receipts']+=1;A.check(s['status']==2,'preprocess_LP_status')
            diag=data.get('historical_family_diagnostic',{})
            if diag.get('status')=='complete':
                checked['historical_family_diagnostic']=audit_method(diag,train,valid,cal,A,'history_diagnostic')
                for s in diag.get('LP_solves',[]):summ['LP_receipts']+=1;A.check(s['status']==2,'diagnostic_LP_status')
            records[fid]=(data,checked)
        data,checked=records[fid];_,_,target=truth(bycase[row['case']],cfg)
        for method in cfg['methods']:
            ev=row.get('evaluated',{}).get(method,{})
            if ev.get('status')!='complete':continue
            v=checked[method];rawerr=len(eff(v['raw'])^target);finalerr=len(eff(v['final'])^target)
            A.check(ev['raw_errors']==rawerr and ev['final_errors']==finalerr,'row_errors')
            A.check(set(map(tuple,ev['target']))==target,'row_truth_target')
            if ev.get('path_count')!=v['path_count']:
                summ['known_wrong_path_count_fields']+=1
            points=data['methods'][method]['points'];minval=min(p['validation_mae'] for p in points)
            correct=[p for p in points if eff(p)==target];elig=[p for p in correct if p['validation_mae']<=1.03*minval+1e-12]
            A.check(ev['correct_saved_count']==len(correct) and ev['correct_eligible_count']==len(elig),'saved_recovery_diagnostic')
            removed=eff(v['raw'])-eff(v['final'])
            mechanisms.append(dict(flow=row['flow'],case=row['case'],menu_id=row['menu_id'],method=method,
                raw_errors=rawerr,final_errors=finalerr,correct_saved=bool(correct),correct_eligible=bool(elig),
                accepted=v['accepted'],accepted_cross=v['accepted_cross'],path_count=v['path_count'],all_scans=v['all_scans'],
                true_deleted=[list(s) for s in sorted(removed&target)],false_deleted=[list(s) for s in sorted(removed-target)],
                final_missing=[list(s) for s in sorted(target-eff(v['final']))],final_extra=[list(s) for s in sorted(eff(v['final'])-target)]))
    summary=read(root/'summary.json')
    A.check(summary['unique_fits']==len(records),'unique_fit_count')
    A.check(summary['LP_calls']==summ['LP_receipts'],'LP_call_count',dict(summary=summary['LP_calls'],receipts=summ['LP_receipts']))
    A.check(summary['LP_calls']<=3000,'LP_budget')
    result=dict(status='passed' if not A.failures else 'failed',checks=A.count,failures=A.failures,
        largest_differences=A.largest,recorded_maps=len(rows),executed_maps=seenreceipts,unique_observed_packets=len(unique_packet_sha),
        unique_records=len(records),counts=dict(summ),source_sha256=sha(__file__),new_LP_calls=0,new_AC_calls=0,
        scope='independent private audit; no original study, selector, generator, model or solver imported')
    (out/'audit_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'mechanisms.json').write_text(json.dumps(mechanisms,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('largest_differences','failures')},ensure_ascii=False));
    if A.failures:print(json.dumps(A.failures[:10],ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    result=main(args.root,args.out);sys.exit(0 if result['status']=='passed' else 1)
