"""Read-only paired capacity audit; retain all executed policies unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from ... import io
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence
from ..flexibility_sweep_20261113.fitting import calibrated_means
from .data import load_entry
from .verify import evidence_for


def audit(standard,expanded,out):
    if out.exists():raise ValueError('Preserve earlier audit')
    if not all((r/'complete.json').exists() for r in (standard,expanded)):raise ValueError('Wait for both complete batches')
    out.mkdir(parents=True)
    p,q=[io.read(r/'protocol.json') for r in (standard,expanded)]
    entries=io.read(standard/'inputs.json')
    io.write(out/'protocol.json',dict(source=io.ref(__file__),standard=io.ref(standard/'protocol.json'),expanded=io.ref(expanded/'protocol.json'),
        work='Saved-file comparison and direct constraint evaluation only; no optimization, new AC or altered policy'))
    errors=[];counts=Counter();details=[];pairs=[];maxerr=defaultdict(float)
    def check(condition,text):
        if not condition:errors.append(text)
    def same(a,b,name,tol=1e-10):
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));maxerr[name]=max(err,maxerr[name]);check(err<=tol,name)
    for field in ('days','replicate_base','study_seed','baseline_scale','modes'):
        check(p[field]==q[field],'Paired protocol '+field)
    check(p['resource_scale']==1 and q['resource_scale']==2,'Expected contract scales1/2')
    check(p['baseline_scale']==2,'Expected identical installed scale2')
    check(entries==io.read(expanded/'inputs.json'),'Identical frozen histories')
    for entry in entries:
        score,train,val,truth=load_entry(entry)
        cases=[CustomerCase(io.read(r/'resource_configs'/(entry['case']+'.json'))) for r in (standard,expanded)]
        a,b=cases;H=np.array([v['H'] for v in score['parameters']]);target=score['map_index'];other=[i for i in range(len(H)) if i!=target]
        means=calibrated_means(score,val)
        same(b.lower,2*a.lower,'Power lower scaling');same(b.upper,2*a.upper,'Power upper scaling')
        same(b.fee_plus,a.fee_plus,'Positive prices');same(b.fee_minus,a.fee_minus,'Negative prices')
        for day in range(p['days']):
            da,db=[r/'days'/entry['id']/f'day{day:03d}' for r in (standard,expanded)]
            with np.load(da/'background.npz') as x,np.load(db/'background.npz') as y:
                check(x.files==y.files,'Background schema')
                for key in x.files:same(x[key],y[key],'Paired background '+key,0.)
            with np.load(da/'planner.npz') as z:pa={k:np.array(z[k]) for k in z.files}
            with np.load(db/'planner.npz') as z:pb={k:np.array(z[k]) for k in z.files}
            same(pa['forecast'],pb['forecast'],'Identical forecast',0.)
            plans_a,plans_b=[io.read(d/'plans.json') for d in (da,db)]
            generation=io.read(da/'generation.json');base=generation['base_kw'];counts['paired_worlds']+=1
            for mode in p['modes']:
                plan,larger=plans_a[mode],plans_b[mode]
                if not plan.get('feasible'):continue
                u=pa[mode];resource_ok=b.audit(u)['passed'];check(resource_ok,'Standard action feasible under expanded resources')
                same(a.fee(u),b.fee(u),'Same fee for identical action',1e-8)
                if len(H)==1:constraint_ok=True
                elif mode=='old_gamma2':
                    ev=CandidateEvidence(b,H,target,other,pa['forecast'],score['sigma2'],base,means)
                    constraint_ok=bool(np.min(ev.direct(u)-ev.required(2.))>=-1e-7)
                else:
                    ev=evidence_for(mode,b,score,val,pa['forecast'],base,plan,entry['voltage_sigma'])
                    constraint_ok=ev.direct(u)['feasible']
                check(constraint_ok,'Standard action satisfies expanded identification constraints')
                counts['embedded_feasible_standard_policies']+=1
                fee=a.fee(u);larger_fee=None if not larger.get('feasible') else b.fee(pb[mode])
                worse=larger_fee is not None and larger_fee>fee+1e-7
                noquote=not larger.get('feasible')
                counts['expanded_quote_cost_above_known_feasible']+=int(worse)
                counts['expanded_no_quote_with_known_feasible']+=int(noquote)
                conflict=noquote and larger.get('infeasibility_proved',False)
                check(not conflict,'Expanded infeasibility proof contradicts verified standard feasible action')
                details.append(dict(id=entry['id'],day=day,mode=mode,role=entry['role'],standard_cost=fee,expanded_cost=larger_fee,
                    expanded_costlier=worse,expanded_no_quote=noquote,expanded_infeasibility_conflict=conflict))
            ra,rb=[io.read(d/'results.json') for d in (da,db)]
            for row in ra:
                second=next(x for x in rb if (x['strategy'],x['physics'])==(row['strategy'],row['physics']))
                pairs.append(dict(strategy=row['strategy'],physics=row['physics'],role=entry['role'],
                    first_correct=bool(row.get('status')=='complete' and row.get('native_map_correct')),
                    second_correct=bool(second.get('status')=='complete' and second.get('native_map_correct')),
                    p_true_gain=None if row.get('p_true') is None or second.get('p_true') is None else second['p_true']-row['p_true'],
                    first_fee=row.get('settled_fee_yuan'),second_fee=second.get('settled_fee_yuan')))
    comparison={}
    for strategy,physics in sorted({(v['strategy'],v['physics']) for v in pairs}):
        own=[v for v in pairs if (v['strategy'],v['physics'])==(strategy,physics)]
        gains=[v['p_true_gain'] for v in own if v['p_true_gain'] is not None]
        comparison[strategy+'/'+physics]=dict(pairs=len(own),standard_correct=sum(v['first_correct'] for v in own),expanded_correct=sum(v['second_correct'] for v in own),
            expanded_new_recoveries=sum(v['second_correct'] and not v['first_correct'] for v in own),expanded_lost_recoveries=sum(v['first_correct'] and not v['second_correct'] for v in own),
            p_true_pairs=len(gains),mean_p_true_change=None if not gains else float(np.mean(gains)),
            mean_standard_fee=float(np.mean([v['first_fee'] for v in own])),mean_expanded_fee=float(np.mean([v['second_fee'] for v in own])))
    result=dict(passed=not errors,errors=errors,counts=dict(counts),maximum_errors=dict(maxerr),comparison=comparison,
        capacity_conclusion='Same installed baseline, forecast and frozen model; all verified standard feasible actions remain available at the same fee under expanded contract. Any more expensive returned quote reflects finite-time solve quality, not a necessary cost increase from extra capacity.',
        actions_changed=False)
    io.write(out/'summary.json',result);io.write(out/'feasible_embedding_private.json',details)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--standard',type=Path,required=True);p.add_argument('--expanded',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit(a.standard.resolve(),a.expanded.resolve(),a.out.resolve());print('passed',r['passed'],'counts',r['counts'],'errors',r['errors'])
