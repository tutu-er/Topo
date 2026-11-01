"""Independent arithmetic and old/new domain audit, without any optimizer.

Completed records only. Reuses the earlier independent arithmetic functions,
not production fitting, optimization, AC or future-data routines.
"""
from pathlib import Path
import hashlib,json,pickle,math,importlib.util,zipfile
from datetime import datetime,timezone
import numpy as np

STAGE=Path(__file__).resolve().parents[1];RUN=STAGE/'model_pilot_uncapped_v2';OLD=STAGE/'model_pilot_v1'
spec=importlib.util.spec_from_file_location('previous_independent_audit',STAGE/'audit_v1/independent_audit.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
read,sha,check,close,rowstats,comparestats=previous.read,previous.sha,previous.check,previous.close,previous.rowstats,previous.comparestats

def compare_objects(a,b,path=''):
    issues=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for key in set(a)|set(b):
            if key not in a or key not in b:issues.append(path+'/'+str(key)+' key presence')
            else:issues.extend(compare_objects(a[key],b[key],path+'/'+str(key)))
    elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        if len(a)!=len(b):issues.append(path+' length')
        else:
            for i,(x,y) in enumerate(zip(a,b)):issues.extend(compare_objects(x,y,path+'/'+str(i)))
    elif isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        if not np.array_equal(np.asarray(a),np.asarray(b)):issues.append(path+' array changed')
    elif a!=b:issues.append(path+' scalar changed')
    return issues

def resources(p,u):
    c=np.max(np.abs(u),axis=1)/np.asarray(p['elasticities']);d=np.asarray(p['factors']);E=.25*np.maximum(-u,0).sum(axis=1)
    cost=float(c@E);hard=float(((1+np.abs(d).sum(axis=1))*c)@E)
    violation=max(float(np.max(np.abs(.25*u.sum(axis=1)),initial=0.)),float(np.max(c-np.asarray(p['price_caps']),initial=0.)))
    corners=[]
    for z in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        actual=(1+d@np.asarray(z))[:,None]*u;prefix=.25*np.cumsum(actual,axis=1)
        v=0.
        for lo,hi,a in [(p['power_lower'],p['power_upper'],actual),(p['energy_lower'],p['energy_upper'],prefix)]:
            v=max(v,float(np.max(np.asarray(lo)-a,initial=0.)),float(np.max(a-np.asarray(hi),initial=0.)))
        v=max(v,float(np.max(np.abs(actual)[~np.asarray(p['available'],bool)],initial=0.)));violation=max(violation,v)
        corners.append(dict(violation=v,payment=float(c@(.25*np.maximum(-actual,0).sum(axis=1)))))
    return dict(prices=c,expected_payment=cost,hard_payment_bound=hard,maximum_violation=max(0.,violation),feasible=violation<=1e-8,corners=corners)

def replay(name,p,ss,r):
    a=np.asarray(r['mean_u']);ours=rowstats(p,ss,a);comparestats(name,ours,r['training']);ra=resources(p,a)
    close(name+' canonical price',ra['prices'],r['prices'])
    for key in ('expected_payment','hard_payment_bound','maximum_violation'):close(name+' '+key,ra[key],r['resource_audit'][key])
    check(name+' uncapped field',r['resource_audit']['payment_cap'] is None and r['resource_audit']['payment_cap_enforced'] is False)
    check(name+' physical feasible',ra['feasible']==r['resource_audit']['feasible'])
    for i,saved in enumerate(r['resource_audit']['corners']):
        close(name+' corner payment '+str(i),ra['corners'][i]['payment'],saved['payment'])
        close(name+' corner resource '+str(i),ra['corners'][i]['violation'],saved['violation'])
    return ours,ra

def audit_evaluation(protocol):
    if not (RUN/'ended.json').exists():return dict(status='not_complete_at_snapshot')
    ended=read(RUN/'ended.json');evaluation=read(RUN/'model_evaluation.json')
    check('evaluation receipt digest',sha(RUN/'model_evaluation.json')==ended['model_evaluation']['sha256'])
    check('evaluation frozen quote digest',sha(RUN/'price_freeze.json')==ended['price_freeze']['sha256']==evaluation['price_freeze']['sha256'])
    check('no new physical evidence',evaluation['new_AC']==evaluation['new_fit_calls']==evaluation['new_observations']==0)
    outputs=[]
    for case,saved in zip(protocol['cases'],evaluation['cases']):
        check('evaluation condition order '+case['name'],saved['condition']==case['name'])
        with (RUN/'cases'/case['name']/'problem.pkl').open('rb') as stream:blob=pickle.load(stream)
        p=blob['problem'];models={x['candidate_id']:x for x in read(case['pool']['path'])['models']}
        with Path(case['history']['path']).open('rb') as stream:history=pickle.load(stream)
        history=history.get('history',history);f=models[p['reference_id']];b=models[p['competitor_id']]
        curves={}
        for label,ref in case['old_plans'].items():
            plan=read(ref['path']);resource_dtos=plan['resources'];positions={r['resource_id']:i for i,r in enumerate(resource_dtos)};u=np.zeros((2,96))
            for amplitude,module in zip(plan['amplitudes'],plan['modules']):
                for rid,h in module['profiles'].items():u[positions[rid]]+=float(amplitude)*np.asarray(h)
            curves[label]=u
        curves['daily_optimized']=np.asarray(read(RUN/'cases'/case['name']/'result.json')['mean_u'])
        summary=dict(condition=case['name'],roles={})
        for role,seed_key in [('training','training_evaluation_seed'),('validation','validation_evaluation_seed')]:
            templates=[]
            for scene in history[role]:
                P,Q,y=previous.array_scene(scene)
                templates.append(dict(base_bad_micro=(y-P@np.asarray(b['R']).T-Q@np.asarray(b['X']).T)*1e6,base_reference_micro=(y-P@np.asarray(f['R']).T-Q@np.asarray(f['X']).T)*1e6))
            rng=np.random.default_rng(case[seed_key]);ss=[]
            for _ in range(protocol['model_holdout_samples']):
                day=int(rng.integers(3));z=rng.uniform(-1,1,2);template=templates[day]
                ss.append(dict(day=day,z=z,**template))
            summary['roles'][role]={}
            for label,u in curves.items():
                ours=rowstats(p,ss,u);saved_curve=saved['summaries'][role][label]
                comparestats('evaluation '+case['name']+' '+role+' '+label,ours,saved_curve['reference_fine'])
                alt=[]
                for s,normal in zip(ss,ours['rows']):
                    effect=(np.asarray(p['contrast'])@((1+np.asarray(p['factors'])@s['z'])[:,None]*u)).T
                    m=p['history_component_micro']+p['day_weight']*(np.abs(s['base_bad_micro']).mean()-(1+p['new_tolerance'])*np.abs(s['base_reference_micro']-effect).mean())-p['epsilon_micro']
                    alt.append(dict(margin_micro=float(m),increment_micro=float(m-normal['baseline_margin_micro']),target_micro=normal['target_micro'],passed=bool(m>=normal['target_micro']-1e-7)))
                a=saved_curve['alternative_coarse']
                check('alternative counted '+case['name']+role+label,sum(x['passed'] for x in alt)==a['passed'])
                close('alternative mean '+case['name']+role+label,float(np.mean([x['increment_micro'] for x in alt])),a['mean_increment_micro'])
                for key in ('margin_micro','increment_micro','target_micro'):
                    close('alternative rows '+key+' '+case['name']+role+label,[x[key] for x in alt],[x[key] for x in a['rows']])
                summary['roles'][role][label]=dict(reference_fine_passed=ours['passed'],alternative_coarse_passed=a['passed'],denominator=len(ss),expected_payment=float(resources(p,u)['expected_payment']))
        outputs.append(summary)
    return dict(status='complete',rows=outputs,scope='independent arithmetic on the saved 256-factor training/validation residual-block checks; three residual templates per role; no new physical days')

def main():
    protocol=read(RUN/'protocol.json');old_protocol=read(OLD/'protocol.json');rows=[]
    check('new protocol expected digest',sha(RUN/'protocol.json')=='979708eddc066484f07f5480926c91bdab00ff88a90875f1b03f045531943d6b')
    check('uncapped protocol',protocol['budget'] is None and protocol['payment_cap_enforced'] is False)
    for source,digest in protocol['source_sha256'].items():check('frozen source '+Path(source).name,sha(source)==digest)
    check('same five conditions',[c['name'] for c in protocol['cases']]==[c['name'] for c in old_protocol['cases']])
    for key in ('planning_samples','required_samples','model_holdout_samples','slots','slot_hours','history_rows','planning_seconds','cost_seconds','reference_candidate','forecast','target'):
        check('same declaration '+key,protocol[key]==old_protocol[key])
    for case,old_case in zip(protocol['cases'],old_protocol['cases']):
        check('same frozen cases/seed '+case['name'],case==old_case)
        folder=RUN/'cases'/case['name'];rf=folder/'result.json'
        if not rf.exists():continue
        result=read(rf);frozen=read(folder/'problem_frozen.json');pp=Path(frozen['problem']['path'])
        check('problem hash '+case['name'],sha(pp)==frozen['problem']['sha256'])
        with pp.open('rb') as stream:blob=pickle.load(stream)
        with (OLD/'cases'/case['name']/'problem.pkl').open('rb') as stream:old_blob=pickle.load(stream)
        p=blob['problem'];ss=blob['scenarios'];oldp=old_blob['problem'];without_budget=dict(p);without_budget['budget']=oldp['budget']
        diffs=compare_objects(oldp,without_budget)
        check('only problem budget changed '+case['name'],not diffs,diffs)
        check('same complete 24 locked scenarios '+case['name'],not compare_objects(old_blob['scenarios'],ss))
        check('budget removed from problem '+case['name'],oldp['budget']==1.6 and p['budget'] is None)
        raw=read(folder/'raw_solver.json')
        row=dict(condition=case['name'],status=result['status'],new_AC=0,new_fits=0,result_sha256=sha(rf),domain_changes=diffs)
        if result.get('mean_u') is not None:
            ours,ra=replay(case['name']+' result',p,ss,result);before,rb=replay(case['name']+' raw',p,ss,raw)
            u=np.asarray(result['mean_u']);v=np.asarray(raw['mean_u']);change=float(np.max(np.abs(u-v)))
            check('numerical restoration bound '+case['name'],change<=1e-7)
            close('record numerical restoration '+case['name'],change,result['numerical_polish']['maximum_change_kw'],1e-16)
            close('exact restored energy '+case['name'],u.sum(axis=1),[0.,0.],1e-12)
            for key in ('primary','secondary'):check('optimization receipts retained '+case['name']+key,result[key]==raw[key])
            eta_max=1+np.abs(np.asarray(p['factors'])).sum(axis=1)
            lower=np.maximum(np.asarray(p['power_lower'])/eta_max[:,None],-np.asarray(p['elasticities'])[:,None]*np.asarray(p['price_caps'])[:,None]);lower=np.where(p['available'],lower,0.)
            upper=np.minimum(np.asarray(p['power_upper'])/eta_max[:,None],np.asarray(p['elasticities'])[:,None]*np.asarray(p['price_caps'])[:,None]);upper=np.where(p['available'],upper,0.)
            fee_ub=np.asarray(p['price_caps'])*(.25*np.maximum(-lower,0).sum(axis=1))
            check('finite physical fee bounds '+case['name'],np.all(np.isfinite(fee_ub)) and np.all(fee_ub>1.6))
            compilation=dict(sign_stable=0,exact_abs=0,constant=0)
            for s in ss:
                coef=np.asarray(p['contrast'])*(1+np.asarray(p['factors'])@s['z'])[None,:]
                for t in range(96):
                    low=np.asarray(s['base_bad_micro'])[t]+np.minimum(coef*lower[:,t],coef*upper[:,t]).sum(axis=1)-1e-8
                    high=np.asarray(s['base_bad_micro'])[t]+np.maximum(coef*lower[:,t],coef*upper[:,t]).sum(axis=1)+1e-8
                    const=~np.any(coef,axis=1);stable=(low>=0)|(high<=0)
                    compilation['constant']+=int(const.sum());compilation['sign_stable']+=int((~const&stable).sum());compilation['exact_abs']+=int((~const&~stable).sum())
            check('same exact ABS compiler '+case['name'],compilation==result['compilation'])
            plan=read(result['plan']['path']);reconstructed=[]
            for amp,module in zip(plan['amplitudes'],plan['modules']):reconstructed.append(float(amp)*np.asarray(next(iter(module['profiles'].values()))))
            close('generic plan same curve '+case['name'],u,reconstructed,1e-14)
            check('generic plan cap removed '+case['name'],plan['budget'] is None and plan['cap_enforced'] is False)
            primary=result.get('primary') or {};secondary=result.get('secondary') or {}
            if primary.get('global_bound') is not None:check('primary bound direction '+case['name'],primary['global_bound']<=primary['objective']+1e-7)
            if secondary.get('global_bound') is not None:check('cost bound direction '+case['name'],secondary['global_bound']<=ra['expected_payment']+1e-7)
            if result['status']=='sample_target_infeasible':check('new-domain infeasibility bound '+case['name'],primary['global_bound']>1e-6)
            row.update(training_passed=ours['passed'],training_denominator=24,slack_micro=ours['feasibility_slack_micro'],expected_payment=ra['expected_payment'],support_payment_bound=ra['hard_payment_bound'],resource_feasible=ra['feasible'],maximum_polish_kw=change,primary_global_lower_bound_micro=primary.get('global_bound'),secondary_global_lower_bound=secondary.get('global_bound'),cost_optimality='not proven' if secondary.get('status')==9 else 'see original solver receipt')
            old_result=read(OLD/'cases'/case['name']/'result.json')
            if case['name']=='paper15_b2_v0005':old_result=read(STAGE/'paper_adapter_retry_v1/result.json')
            if old_result.get('mean_u') is not None:
                ou=np.asarray(old_result['mean_u']);os=rowstats(p,ss,ou);orr=resources(p,ou)
                row['old_curve_in_new_domain']=dict(passed=os['passed'],denominator=24,resource_feasible=orr['feasible'],expected_payment=orr['expected_payment'],known_feasible_cost_upper_bound=orr['expected_payment'] if os['passed']>=22 and orr['feasible'] else None,not_new_optimization=True)
        rows.append(row)
    evaluation=audit_evaluation(protocol);checks=previous.CHECKS
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(RUN/'protocol.json'),audit_sha256=sha(__file__),cases_complete=len(rows),cases_planned=5,checks=len(checks),passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,model_evaluation=evaluation,scope='completed frozen outputs only; no new optimizer, fit, AC or actual futures; the sole physical problem-domain difference is removal of the monetary cap; warm starts and time-limited search may produce different incumbents, not monotone observed solver performance',global_bound_scope='new numerical lower bounds apply only to this no-payment-cap conditional forecast law and declaration; old budgeted infeasibility is not reused')
    (Path(__file__).parent/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
