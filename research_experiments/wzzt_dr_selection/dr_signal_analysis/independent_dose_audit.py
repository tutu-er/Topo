"""Independent scalar replay of frozen-model dose/time counterfactuals.

No physical solver, optimizer, model scorer or true-topology reader is used.
Only historical P/Q, frozen fitted weights, archived natural P/Q/factors and
published scalar results are inspected. No archived voltage field is read.
"""
from pathlib import Path
from datetime import datetime,timezone
from itertools import product
import argparse,hashlib,json,math,pickle
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vector(x):return np.broadcast_to(np.asarray(x,float),(96,))
def matrix(point,n,key):
    out=np.zeros((n,n))
    for block,value in zip(point['supports'],point[key]):out[np.ix_(block,block)]+=value
    return out
def shape_from_gradient(g,m):
    order=np.argsort(g,kind='stable');h=np.zeros(96);h[order[:m]]=-1;h[order[-m:]]=1
    return h

def audit(root,output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError('retain existing independent audit')
    p=read(root/'protocol.json');s=read(root/'summary.json');pub=read(root/'public_scalars.json')
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    def close(name,a,b,tol=2e-12):check(name,np.allclose(a,b,rtol=0,atol=tol))
    def checked(r):
        q=Path(r['path']);check('input_hash_'+q.name,sha(q)==r['sha256']);return q
    def unpickle(r):
        with checked(r).open('rb') as f:return pickle.load(f)
    for q,h in p['source_sha256'].items():check('source_'+Path(q).name,sha(q)==h)
    for item in p['inputs'].values():checked(item)
    master=read(p['inputs']['protocol.json']['path'])
    old_audit=read(p['inputs']['independent_physical_audit_v2.json']['path'])
    prior={r['record_id']:r for r in old_audit['rows']}
    check('scope_no_computation',all(s[k]==0 for k in ['new_AC','new_fit_calls','new_solver_calls','new_observations']))
    check('scope_19_of20',s['planned_case_days']==20 and s['valid_archived_case_days']==19 and s['unresolved_archived_case_days']==1)
    check('all240_rows_228_valid',len(s['rows'])==240 and sum(r['status']=='counterfactual_evaluated' for r in s['rows'])==228)
    check('public_rows_exact',s['rows']==pub['rows'])
    check('public_tables_exact',s['tables']==pub['tables'])
    physical={(a['condition'],a['path']):a['file'] for a in p['physical_input_refs']}
    check('19_distinct_physical_inputs',len(physical)==19)
    corners=[np.asarray(z,float) for z in product((-1.,1.),repeat=2)]
    rowmap={(r['condition'],r['kind'],r['value'],r['path']):r for r in s['rows']}
    tablemap={(r['condition'],r['kind'],r['value']):r for r in s['tables']}
    output_curves=[];norm_not_dronly=0;constraint_binding_counts={};maximum_beta_snap=0.
    for case in p['cases']:
        name=case['condition'];pool=read(checked(case['pool']));history=unpickle(case['history'])['history']
        plan=read(checked(case['optimized_plan']));problem=read(checked(case['problem']))
        labels=pool['labels'];n=len(labels);kw=pool['base_kw']
        mats=[]
        for model in pool['models']:
            rr,xx=matrix(model['point'],n,'r'),matrix(model['point'],n,'x')
            close(name+'_R'+model['candidate_id'],rr,model['R'],tol=1e-15)
            close(name+'_X'+model['candidate_id'],xx,model['X'],tol=1e-15);mats.append((rr,xx))
        dr,dx=mats[1][0]-mats[0][0],mats[1][1]-mats[0][1]
        days=history['training']+history['validation']
        check(name+'_all6history',len(days)==6 and all(len(x['P_terminal'])==96 for x in days))
        hp=np.stack([x['P_terminal'].to_numpy(float) for x in days]);hq=np.stack([x['Q_terminal'].to_numpy(float) for x in days])
        sign=np.sign(hp.mean(axis=0)@dr.T+hq.mean(axis=0)@dx.T)
        resources=plan['resources'];active=[];gradient=[];oldh=[]
        for res in resources:
            rid=res['resource_id'];matches=[j for j,a in enumerate(plan['amplitudes']) if a>1e-12 and rid in plan['modules'][j]['profiles']]
            check(name+'_oneactive_'+rid,len(matches)==1);j=matches[0];active.append(j)
            node=labels.index(res['node_label']);ratio=vector(res['reactive_ratio'])
            g=(sign@dr[:,node]+ratio*(sign@dx[:,node]))/(96*n*kw);gradient.append(g)
            h=shape_from_gradient(g,8);oldh.append(h)
            close(name+'_historical8_'+rid,h,plan['modules'][j]['profiles'][rid],tol=0)
            check(name+'_resource_scope_'+rid,res['baseline_kw']==1 and res['power_upper_kw']==2 and res['prefix_delta_lower_kwh']==-2 and res['prefix_delta_upper_kwh']==2 and res['available'] is True)
        price=np.asarray(plan['prices'])[active];lam=np.asarray(plan['elasticities'])[active]
        a0=np.asarray(plan['amplitudes'])[active];factors=np.asarray(plan['response_factors'])[active];q=np.abs(factors).sum(axis=1)
        caps=np.asarray(problem['price_upper'])[active];actual_caps=np.asarray(problem['actual_amplitude_upper'])[active]
        close(name+'_price_amplitude',a0,price*lam);close(name+'_actual_caps',actual_caps,[1,1],tol=0)
        fixtures={}
        for path in next(c for c in master['cases'] if c['name']==name)['paths']:
            if (name,path) in physical:
                x=unpickle(physical[(name,path)])
                # Deliberately access no V, AC or static topology fields.
                fixtures[path]=(np.asarray(x['natural_P_pu']),np.asarray(x['natural_Q_pu']),np.asarray(x['factors']))
                pp,qq,zz=fixtures[path]
                check(name+str(path)+'_factors_supported',zz.shape==(2,) and np.max(np.abs(zz))<=1.)
                for res in resources:
                    node=labels.index(res['node_label'])
                    check(name+str(path)+res['resource_id']+'_original_component_present',
                        np.min(pp[:,node]*kw-vector(res['baseline_kw']))>=-1e-9 and
                        np.min(qq[:,node]*kw-vector(res['reactive_ratio'])*vector(res['baseline_kw']))>=-1e-9)
        for kind,values in [('duration',[1.,2.,4.,6.,8.,12.]),('amplitude',p['amplitude_alpha'])]:
            for value in values:
                t=tablemap[(name,kind,value)];m=int(round(value*4)) if kind=='duration' else 8
                shapes=[shape_from_gradient(g,m) for g in gradient] if kind=='duration' else oldh
                tau=m*.25
                for j,h in enumerate(shapes):
                    check(name+str((kind,value,j))+'_shape',np.count_nonzero(h<0)==m and np.count_nonzero(h>0)==m and h.sum()==0)
                # Independently intersect the price ray with every supported
                # corner of raw resource powers and cumulative energy bounds.
                bounds=[math.sqrt(1.6/(tau*np.sum(a0*a0/lam*(1+q))))]
                for j,res in enumerate(resources):
                    bounds.extend([caps[j]/price[j],actual_caps[j]/((1+q[j])*a0[j])])
                    for z in corners:
                        coeff=shapes[j]*a0[j]*(1+factors[j]@z)
                        for v,lo,hi,b in zip(coeff,vector(res.get('power_lower_kw',0)),vector(res['power_upper_kw']),vector(res['baseline_kw'])):
                            if v>1e-15:bounds.append(float((hi-b)/v))
                            elif v < -1e-15:bounds.append(float((lo-b)/v))
                        prefix=.25*np.cumsum(coeff)
                        for v in prefix:
                            if v>1e-15:bounds.append(float(res['prefix_delta_upper_kwh']/v))
                            elif v < -1e-15:bounds.append(float(res['prefix_delta_lower_kwh']/v))
                beta=float(min(bounds)) if kind=='duration' else float(value)
                if kind=='duration' and abs(beta-1)<1e-7:
                    maximum_beta_snap=max(maximum_beta_snap,abs(beta-1));beta=1.
                close(name+str((kind,value))+'_ray_scale',beta,t['beta'],tol=1e-7)
                a=beta*a0;c=a/lam
                close(name+str((kind,value))+'_amps',a,t['mean_amplitudes_kw']);close(name+str((kind,value))+'_prices',c,t['prices'])
                fee=tau*np.sum(a*a/lam);hard=tau*np.sum(a*a/lam*(1+q))
                close(name+str((kind,value))+'_fee',fee,t['expected_payment']);close(name+str((kind,value))+'_hard',hard,t['hard_payment_bound'])
                close(name+str((kind,value))+'_budgetratio',hard/1.6,t['budget_ratio'])
                resource_ok=True
                for j,res in enumerate(resources):
                    for z in corners:
                        actual=a[j]*(1+factors[j]@z);power=vector(res['baseline_kw'])+shapes[j]*actual;prefix=.25*np.cumsum(shapes[j]*actual)
                        good=(np.min(power)>=-1e-7 and np.max(power-vector(res['power_upper_kw']))<=1e-7 and
                              np.max(np.abs(prefix))<=2+1e-7 and actual<=actual_caps[j]+1e-7 and c[j]<=caps[j]+1e-7)
                        resource_ok=resource_ok and good
                check(name+str((kind,value))+'_corner_resource',resource_ok)
                check(name+str((kind,value))+'_original_budget_domain',t['within_original_domain']==bool(hard<=1.6+1e-7 and resource_ok))
                if kind=='duration':check(name+str(value)+'_duration_feasible',hard<=1.6+1e-7)
                else:close(name+str(value)+'_quadratic_scaling',hard,value**2*plan['hard_payment_bound'])
                actual_metrics=[]
                for path in next(c for c in master['cases'] if c['name']==name)['paths']:
                    r=rowmap[(name,kind,value,path)]
                    if path not in fixtures:
                        check(name+str((kind,value,path))+'_failure_preserved',r['status']=='unresolved_original_resource_failure');continue
                    p0,q0,z=fixtures[path];dp=np.zeros((96,n));dq=np.zeros((96,n));actual=a*(1+factors@z)
                    for j,res in enumerate(resources):
                        node=labels.index(res['node_label']);dp[:,node]+=shapes[j]*actual[j];dq[:,node]+=vector(res['reactive_ratio'])*shapes[j]*actual[j]
                    base=p0@dr.T+q0@dx.T;effect=(dp@dr.T+dq@dx.T)/kw
                    d10=np.abs(base).mean();d1=np.abs(base+effect).mean();d20=np.sqrt(np.mean(base**2));d2=np.sqrt(np.mean((base+effect)**2))
                    derived=dict(D1_baseline=d10,D1=d1,D1_gain=d1-d10,D2_baseline=d20,D2=d2,D2_gain=d2-d20,DR_prediction_rms=np.sqrt(np.mean(effect**2)),
                        relative_D1_gain_pct=100*(d1-d10)/d10,actual_modeled_payment=tau*np.sum(c*actual),total_shifted_kwh=tau*actual.sum())
                    for key,v in derived.items():close(name+str((kind,value,path))+'_'+key,v,r[key])
                    close(name+str((kind,value,path))+'_actual_amplitudes',actual,r['actual_amplitudes_kw'])
                    check(name+str((kind,value,path))+'_actual_payment_bound',derived['actual_modeled_payment']<=hard+1e-7)
                    target=[labels.index(res['node_label']) for res in resources]
                    check(name+str((kind,value,path))+'_target_input_nonnegative',np.min((p0+dp/kw)[:,target])>=-1e-9 and np.min((q0+dq/kw)[:,target])>=-1e-9)
                    if abs(np.abs(effect).mean()-(d1-d10))>1e-10 and abs(np.sqrt(np.mean(effect**2))-(d2-d20))>1e-10:norm_not_dronly+=1
                    if (kind=='duration' and m==8) or (kind=='amplitude' and value==1.):
                        old=prior[r['original_record']]
                        close(name+str((kind,value,path))+'_old_D1',d1,old['frozen_distance_physical'])
                        close(name+str((kind,value,path))+'_old_gain',d1-d10,old['paired_physical_distance_gain'])
                        close(name+str((kind,value,path))+'_old_fee',derived['actual_modeled_payment'],old['actual_payment'],tol=1e-9)
                    actual_metrics.append(derived)
                check(name+str((kind,value))+'_denominator',t['denominator']==4 and t['complete']==len(actual_metrics) and t['unresolved']==4-len(actual_metrics))
                for key in ['D1_baseline','D1','D1_gain','D2_baseline','D2','D2_gain','DR_prediction_rms']:
                    values_for_key=[v[key] for v in actual_metrics]
                    for label,value_agg in [('mean',np.mean(values_for_key)),('minimum',np.min(values_for_key)),('maximum',np.max(values_for_key))]:
                        close(name+str((kind,value))+'_'+key+label,value_agg,t[key][label])
    check('norm_difference_not_DR_only',norm_not_dronly>0)
    for agg in pub['aggregate']:
        group=[r for r in s['rows'] if r['kind']==agg['kind'] and r['value']==agg['value'] and r['status']=='counterfactual_evaluated']
        check(str((agg['kind'],agg['value']))+'_aggregate_19of20',agg['complete']==19 and agg['denominator']==20 and agg['unresolved']==1 and len(group)==19)
        for key in ['D1_gain','D2_gain','DR_prediction_rms']:
            close(str((agg['kind'],agg['value']))+'_aggregate_'+key,np.mean([r[key] for r in group]),agg[key+'_mean'])
    check('duration12_not_monotone_claim',next(a for a in pub['aggregate'] if a['kind']=='duration' and a['value']==12)['D1_gain_mean']<next(a for a in pub['aggregate'] if a['kind']=='duration' and a['value']==8)['D1_gain_mean'])
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),check_count=len(checks),failures=[],checks=checks,
        no_new_AC=True,no_new_solver=True,no_new_fit=True,no_observed_voltage_or_static_truth_fields_read=True,
        maximum_beta_snap=maximum_beta_snap,beta_snap_scope='declared |beta-1|<1e-7 baseline-replay convention; feasibility independently checked',
        scope='independent arithmetic of frozen-model retrospective input curves, not new delivered DR or model rejection evidence',
        denominator=dict(planned_case_days=20,archived_valid_case_days=19,unresolved_case_days=1,variants=12,planned_arithmetic_rows=240,valid_arithmetic_rows=228),
        limitations=['Relative old node prices remain fixed; time curves are ray-scaled feasible schedules, not per-duration optimal prices.',
            'Duration means reduction-hours per resource, with equal rebound-hours; positions are historical gradient-selected slots, not a contiguous time extension.',
            'Alpha sqrt2 needs about3.2 budget and is outside the original1.6 domain, though conditional resource checks pass.',
            'L1/RMS prediction-distance improvements do not by themselves establish a likelihood, topology confidence, or final selection improvement.',
            'The archived failed resource day stays unresolved for all variants; no new physical resource/voltage/thermal validation was run.'],
        initial_audit_mismatches_retained=['independent_dose_audit_attempt1.json','independent_dose_audit_attempt2.json'],
        source_sha256=sha(__file__),input_refs={n:sha(root/n) for n in ['protocol.json','summary.json','audit.json','public_scalars.json']})
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',checks=len(checks),failures=0,arithmetic_rows=240,new_AC=0,new_solver=0,new_fit=0)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();audit(x.root,x.output)
