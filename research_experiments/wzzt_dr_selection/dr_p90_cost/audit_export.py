"""Independent scalar/physical/selection replay, never reprices or reruns AC."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
stage=Path(__file__).parent;sys.path.insert(0,str(stage/'code'))
import run as rr
r=rr.r;root=stage/'run_v1';p=r.read(root/'protocol.json')
checks=0;failed=[]
def check(flag,label):
    global checks
    checks+=1
    if not flag:failed.append(label)
for path,h in p['source_sha256'].items():check(r.sha(path)==h,'source '+Path(path).name)
for q in p['locked_inputs']:check(r.sha(q['path'])==q['sha256'],'input '+Path(q['path']).name)
check(r.read(root/'run_lock.json')['status']=='complete','batch completed')
freeze=r.read(root/'price_freeze.json');r.checked(freeze['protocol'])
summary=r.read(root/'summary.json');observed=r.read(root/'observations_frozen.json')['rows']
scores={v['record_id']:v for v in r.read(root/'scored_private.json')['rows']}
cfg=r.read(r.checked(p['library']));alloc={v['case']:v for v in r.read(r.checked(p['allocations']))['cases']}
hist=r.read(rr.BASE/'protocol.json');oldpaths={(v['profile_index'],v['replicate']) for v in hist['paths']}
check(not(oldpaths&{(v['profile_index'],v['replicate']) for v in p['paths']}),'no reused parent backgrounds')
unique=set();bycase={};models={};plans={};truth={};devices={}
for c in p['cases']:
    name=c['name'];pool=r.read(root/name/'pool.json');models[name]=pool
    check(pool['validation_rows']==288 and pool['train_rows']==288,name+' full history')
    history=r.load(r.checked(pool['history_packet']));history=history.get('history',history)
    check(sum(np.asarray(v['P_terminal']).shape[0] for v in history['training'])==288,name+' actual training packet')
    check(sum(np.asarray(v['P_terminal']).shape[0] for v in history['validation'])==288,name+' actual validation packet')
    for fn in ['pool.json','problem.pkl','case_protocol.json']:
        check(r.sha(root/name/fn)==r.sha(rr.BASE/name/fn),name+' frozen '+fn)
    plans[name]=r.load(r.checked(freeze['cases'][name]['plans']))
    truth[name]={tuple(sorted(v)) for v in r.read(root/name/'truth_private.json')['missed_blocks']}
    devices[name]=r.Devices(alloc[c['case']],cfg,r.profiles)
    bycase[name]=dict(raw_final_different=0,unique_observations=set())
    dev=devices[name]
    for key,item in plans[name].items():
        u=item['mean_u'];cap=np.where(u>=0,dev.delta_upper_kw,-dev.delta_lower_kw)
        coeff=np.divide(dev.dt*dev.cost_full_activation,cap,out=np.zeros_like(cap),where=cap>0)
        check(abs(float(np.sum(coeff*u*u))-item['cost'])<1e-8,name+' fee '+key)
        check(dev.audit(u)['passed'],name+' nominal device '+key)
        if key=='nominal_milp' and item['alpha'] is not None:
            old=r.load(rr.BASE/name/'plans.pkl')['candidate']['mean_u']
            check(np.max(np.abs(u-item['alpha']*old))<1e-12,name+' MILP frozen ray')

for row in observed:
    name=row['condition'];identity=row['record_id'];check(row['status']=='complete',identity+' complete')
    if row['status']!='complete':continue
    pack=r.load(r.checked(row['packet']));scene=pack['scene'];priv=pack['private'];pool=models[name];dev=devices[name]
    P=np.asarray(scene['P_terminal']);Q=np.asarray(scene['Q_terminal']);V=np.asarray(scene['V_terminal']);V0=np.asarray(scene['V_root_observed']).reshape(-1)
    Y=V0[:,None]**2-V*V;loss={}
    check(P.shape==(96,len(pool['labels'])),identity+' new full day')
    histloss={m['candidate_id']:m['historical_validation_mae'] for m in pool['models']}
    for m in pool['models']:
        key=m['candidate_id'];loss[key]=float(np.abs(Y-P@np.asarray(m['R']).T-Q@np.asarray(m['X']).T).mean())
    order=pool['selection_order'];index={k:i for i,k in enumerate(order)}
    eligible_h=[k for k in order if histloss[k]<=1.03*min(histloss.values())+1e-12]
    J={k:(288*histloss[k]+192*loss[k])/480 for k in order}
    best=min(eligible_h,key=lambda k:(J[k],index[k]));threshold=1.03*J[best]+1e-12
    eligible=[k for k in eligible_h if J[k]<=threshold]
    model={m['candidate_id']:m for m in pool['models']}
    selected=min(eligible,key=lambda k:(len(model[k]['point']['supports']),J[k],index[k]))
    target=scores[identity]
    check(selected==target['selected'],identity+' selected')
    D=1e6*(loss['coarse']-1.03*loss['refined'])
    check(abs(D-target['D_micro'])<1e-8,identity+' D')
    check(('coarse' not in eligible)==target['coarse_excluded'],identity+' exclusion')
    supports={tuple(sorted(v)) for v in model[selected]['frozen_output_support_labels']}
    raw={tuple(sorted(pool['labels'][i] for i in b)) for b in model[selected]['point']['supports']}
    check(len(supports^truth[name])==target['errors'],identity+' final error')
    check(len(raw^truth[name])==target['raw_errors'],identity+' raw error')
    bycase[name]['raw_final_different']+=target['errors']!=target['raw_errors']
    u=priv['nominal'];delivered=priv['delivered'];check(np.array_equal(u,plans[name][row['policy']]['mean_u']),identity+' frozen delivery request')
    check(np.max(delivered-dev.delta_upper_kw)<1e-8 and np.max(dev.delta_lower_kw-delivered)<1e-8,identity+' power bounds')
    check(dev.audit(delivered)['passed'],identity+' energy and SOC')
    dp,dq=dev.aggregate(delivered);base=pool['base_kw']
    check(np.max(np.abs((np.asarray(priv['physical_p'])-np.asarray(priv['natural_p']))*base-dp))<1e-9,identity+' P increment')
    check(np.max(np.abs((np.asarray(priv['physical_q'])-np.asarray(priv['natural_q']))*base-dq))<1e-9,identity+' Q increment')
    check(np.isclose(pack['summary']['resource_audit']['absolute_DR_kwh'],.25*np.abs(delivered).sum()),identity+' adjustment')
    check(pack['summary']['converged']==96 and priv['checks']['electrical_equations_pass'],identity+' AC')
    digest=hashlib.sha256(name.encode()+b''.join(np.ascontiguousarray(v).tobytes() for v in [P,Q,V,V0])).hexdigest()
    unique.add(digest);bycase[name]['unique_observations'].add(digest)

qual=r.read(root/'qualification_summary.json')
for row in qual['rows']:
    delta=np.sqrt(np.log(20)/(2*row['samples']));delta15=np.sqrt(np.log(300)/(2*row['samples']))
    check(abs(row['fixed_policy_95_lower']-max(0,row['probability']-delta))<1e-12,'individual bound '+row['condition']+row['policy'])
    check(abs(row['simultaneous15_95_lower']-max(0,row['probability']-delta15))<1e-12,'joint bound '+row['condition']+row['policy'])
check(len(observed)==360 and summary['complete']==360,'complete360')
audit=dict(checks=checks,failures=failed,passed=not failed,records=360,
    independent_backgrounds=36,paired_response_laws=2,unique_observations=len(unique),
    cases={k:dict(raw_final_different=v['raw_final_different'],unique_observations=len(v['unique_observations'])) for k,v in bycase.items()},
    scope='source,original validation,frozen parameters,fees,device/AC and full-error audit;not statistical model truth')
r.write(root/'audit.json',audit)
if failed:raise AssertionError(failed[:20])
public=stage/'public';public.mkdir(exist_ok=True)
for name in ['planning_summary.json','qualification_summary.json','summary.json','audit.json']:
    value=r.read(root/name)
    if name=='qualification_summary.json':value.pop('price_freeze',None)
    r.write(public/name,value)
cfgout={k:v for k,v in p.items() if k not in ['source_sha256','locked_inputs','cases','previous','parent_complete','older_base','library','allocations']}
cfgout.update(cases=[c['name'] for c in p['cases']],
    source_sha256=[dict(file=Path(k).parent.name+'/'+Path(k).name,sha256=v) for k,v in p['source_sha256'].items()],
    input_sha256=[dict(file=Path(q['path']).parent.name+'/'+Path(q['path']).name,sha256=q['sha256']) for q in p['locked_inputs']])
r.write(public/'protocol.json',cfgout)
print(json.dumps(audit,ensure_ascii=False))
