"""Independent direct frozen-matrix replay and scalar paired result projection.

No optimization, fitting, AC, graph or actual/test observation is read.
The working forecast is reconstructed in original P/Q/Y units rather than
using the compiled day-model contrast or margin functions.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('frozen input drift: '+str(p))
    return p
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def curves(plans,resources):
    result={};prices={};ids={r['resource_id']:i for i,r in enumerate(resources)}
    for name,plan in plans.items():
        u=np.zeros((len(resources),96));c=np.zeros(len(resources))
        for module,a,p in zip(plan['modules'],plan['amplitudes'],plan['prices']):
            for rid,h in module['profiles'].items():
                u[ids[rid]]+=float(a)*np.asarray(h)
                if float(a)!=0.:c[ids[rid]]=float(p)
        result[name]=u;prices[name]=c
    return result,prices


def paired(rows,zero,days):
    passed=np.array([r['passed'] for r in rows]);base=np.array([r['passed'] for r in zero])
    delta=np.array([r['margin_micro']-z['margin_micro'] for r,z in zip(rows,zero)])
    def subset(mask):
        return dict(denominator=int(mask.sum()),gained_pass_count=int((mask&passed&~base).sum()),
            lost_pass_count=int((mask&~passed&base).sum()),both_pass_count=int((mask&passed&base).sum()),
            both_fail_count=int((mask&~passed&~base).sum()),
            mean_margin_increment_vs_zero_micro=float(delta[mask].mean()))
    value=subset(np.ones(len(rows),bool))
    value['per_historical_day']={str(i):subset(np.array(days)==i) for i in range(3)}
    return value


def run(root):
    root=Path(root).resolve();p=read(root/'protocol.json');ended=read(root/'ended.json')
    checked(ended['private_summary']);checked(ended['public_summary_candidate'])
    full=read(root/'summary_private.json');public=read(root/'summary_public_candidate.json')
    complete={x['condition']:x for x in full['cases'] if x['status']=='complete'}
    maximum=0.;counts=dict(reference_margin=0,alternative_margin=0,baseline_margin=0,target=0,
        reference_pass=0,alternative_pass=0,expected_payment=0,hard_payment=0,history_days=0,history_rows=0)
    per_case=[]
    for c in p['cases']:
        if c['condition'] not in complete:continue
        stored=complete[c['condition']];pool=read(checked(c['pool']))
        with checked(c['history']).open('rb') as f:packet=pickle.load(f)
        history=packet.get('history',packet);labels=pool['labels'];n=len(labels)
        models={m['candidate_id']:m for m in pool['models']};fine=models['refined'];bad=next(v for k,v in models.items() if k!='refined')
        rf,xf,rb,xb=[np.asarray(x,float) for x in (fine['R'],fine['X'],bad['R'],bad['X'])]
        vals={}
        for name,model in [('fine',fine),('bad',bad)]:
            errors=[]
            for scene in history['validation']:
                pp,qq=_power(scene,labels);yy=_target(scene,pp.shape)
                errors.append(np.abs(yy-pp@np.asarray(model['R']).T-qq@np.asarray(model['X']).T))
            vals[name]=float(np.stack(errors).mean())
        hist=.6*(vals['bad']-1.03*vals['fine'])*1e6
        plans={('old_optimized' if k=='optimized' else k):read(checked(v)) for k,v in c['old_plans'].items()}
        resources=plans['old_optimized']['resources'];us,cs=curves(plans,resources)
        rr=read(checked(c['result']));us['daily_model_candidate']=np.asarray(rr['mean_u']);cs['daily_model_candidate']=np.asarray(rr['prices'])
        case_max=0.
        for role in ('training','validation'):
            scenes=history[role]
            if len(scenes)!=3:raise AssertionError('all three historical days required')
            counts['history_days']+=3
            templates=[]
            for scene in scenes:
                pp,qq=_power(scene,labels);yy=_target(scene,pp.shape)
                if pp.shape!=(96,n):raise AssertionError('complete96-slot historical day required')
                counts['history_rows']+=96;templates.append((pp,qq,yy))
            draws=stored['response_draws'][role];days=np.array([s['day'] for s in draws]);z=np.array([s['z'] for s in draws])
            basep=np.stack([templates[i][0] for i in days]);baseq=np.stack([templates[i][1] for i in days]);basey=np.stack([templates[i][2] for i in days])
            zero_bad=np.abs(basey-basep@rb.T-baseq@xb.T).mean(axis=(1,2))
            zero_fine=np.abs(basey-basep@rf.T-baseq@xf.T).mean(axis=(1,2))
            mzero=hist+.4*(zero_bad-1.02*zero_fine)*1e6-1e-6
            target=.2*np.maximum(-mzero,0.)+.0001
            factor=1+z@np.array([[.15,.05],[.15,-.05]]).T
            for policy in p['policies']:
                u=us[policy];dp=np.zeros_like(basep);dq=np.zeros_like(baseq)
                for j,resource in enumerate(resources):
                    column=labels.index(resource['node_label']);actual=factor[:,j,None]*u[j][None,:]
                    ratio=np.broadcast_to(np.asarray(resource.get('reactive_ratio',.35),float),(96,))
                    dp[:,:,column]+=actual/pool['base_kw'];dq[:,:,column]+=actual*ratio[None,:]/pool['base_kw']
                pp=basep+dp;qq=baseq+dq
                y_fine=basey+dp@rf.T+dq@xf.T;y_bad=basey+dp@rb.T+dq@xb.T
                m_fine=hist+.4*(np.abs(y_fine-pp@rb.T-qq@xb.T).mean(axis=(1,2))
                    -1.02*np.abs(y_fine-pp@rf.T-qq@xf.T).mean(axis=(1,2)))*1e6-1e-6
                m_bad=hist+.4*(np.abs(y_bad-pp@rb.T-qq@xb.T).mean(axis=(1,2))
                    -1.02*np.abs(y_bad-pp@rf.T-qq@xf.T).mean(axis=(1,2)))*1e6-1e-6
                record=stored['summaries'][role][policy]
                for name,values,countname in [('reference_fine',m_fine,'reference_margin'),('alternative_coarse',m_bad,'alternative_margin')]:
                    rows=record[name]['rows'];expected=np.array([x['margin_micro'] for x in rows])
                    error=float(np.max(np.abs(values-expected)));maximum=max(maximum,error);case_max=max(case_max,error)
                    if error>1e-7:raise AssertionError('direct matrix margin mismatch')
                    counts[countname]+=len(draws)
                    if not np.array_equal(values>=target-1e-7,np.array([x['passed'] for x in rows])):raise AssertionError('direct pass mismatch')
                    counts['reference_pass' if name=='reference_fine' else 'alternative_pass']+=len(draws)
                rows=record['reference_fine']['rows']
                for key,value,countname in [('baseline_margin_micro',mzero,'baseline_margin'),('target_micro',target,'target')]:
                    error=float(np.max(np.abs(value-np.array([x[key] for x in rows]))));maximum=max(maximum,error)
                    if error>1e-7:raise AssertionError('direct baseline or target mismatch')
                    counts[countname]+=len(draws)
                cost=float(cs[policy]@(.25*np.maximum(-u,0.).sum(axis=1)))
                hard=1.2*cost
                for key,value,countname in [('expected_payment',cost,'expected_payment'),('hard_payment_bound',hard,'hard_payment')]:
                    if abs(value-record['original_quote_resource_audit'][key])>1e-10:raise AssertionError('direct fee mismatch')
                    counts[countname]+=1
        per_case.append(dict(condition=c['condition'],maximum_direct_margin_error_micro=case_max))
    # Add paired scalar diagnostics to a new output; keep original frozen summaries intact.
    for case in public['cases']:
        if case['status']!='complete':continue
        private=complete[case['condition']]
        for role,policies in case['summaries'].items():
            days=[s['day'] for s in private['response_draws'][role]]
            for policy,summary in policies.items():
                for name in ('reference_fine','alternative_coarse'):
                    summary[name]['paired_vs_zero']=paired(private['summaries'][role][policy][name]['rows'],
                        private['summaries'][role]['zero'][name]['rows'],days)
    public.update(draw_reuse_scope='each policy-role cell has256 model draws; all four policies share that role draw set; only three old residual days per condition/role',
        independent_new_measurement_days=0,old_residual_days_per_condition_per_role=3,
        direct_matrix_audit=dict(status='passed',maximum_error_micro=maximum,comparison_counts=counts),
        source_public_summary_sha256=sha(root/'summary_public_candidate.json'))
    write(root/'summary_public_with_paired_candidate.json',public)
    receipt=dict(status='passed',method='direct original-unit P/Q/Y matrix forecast, exact historical/new-day margin, paired pass and payment replay',
        maximum_error_micro=maximum,comparison_counts=counts,cases=per_case,
        source_script=ref(__file__),protocol=ref(root/'protocol.json'),
        source_private_summary=ref(root/'summary_private.json'),source_public_summary=ref(root/'summary_public_candidate.json'),
        paired_public_summary=ref(root/'summary_public_with_paired_candidate.json'),
        new_AC=0,new_fit_calls=0,new_optimization_calls=0,new_measurement_days=0)
    write(root/'audit_direct_matrix_v1.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
