"""A2d: full-parameter block possibility and two-model price-width audit.

Historical complete-family fits are reused. No truth, future Y, positive weight
floor, or coefficient cap enters presence/width design. Bounds are numerical.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .compatible_functionals import maximize_history_functional, maximize_nonnegative_lp
from .core import all_blocks, block_features
from .experiment import ROOT, History, Evaluator, MENUS, current_environment, digest, jblock, run_lock, save
from .exact_patterns import PATTERNS, probability, pattern_cost
from .face_price_audit import matrix_for_pattern, observed_path, clean_sets

class AuditFailure(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message);self.details=details


TOL=1e-7
INPUT=ROOT/'outputs/a2c_completion_final_v2'
PREVIOUS=ROOT/'outputs/a2b_face_final_v2_clean'


def family(fit):
    return tuple(tuple(c) for c in fit['blocks'])


def inputs(record):
    return tuple(np.asarray(record['history'][k],dtype=float) for k in ('P','Q','Y'))


def maximal_compatible(record):
    """Exact set compression: a subset model extends with zero coefficients."""
    if not record['full_reference_complete']:
        raise ValueError('Cannot discard unsolved historical families')
    epsilon=record['error_budget']
    eligible=[f for f in record['family_fits'] if f['success'] and f['loss']<=epsilon+TOL]
    kept=[f for f in eligible if not any(set(family(f))<set(family(g)) for g in eligible)]
    assert all(any(set(family(f))<=set(family(g)) for g in kept) for f in eligible)
    return eligible,kept


def aggregate(rows):
    if not rows:
        return {'status':'optimal','value':0.,'upper_bound':0.,'bounded':True}
    unbounded=[r for r in rows if r['status']=='unbounded_verified']
    if unbounded:
        return {'status':'unbounded_verified','value':max(r.get('value') or 0. for r in rows),
                'upper_bound':None,'bounded':False}
    if any(r['status']!='optimal' for r in rows):
        return {'status':'unresolved','value':max((r.get('value') or 0. for r in rows),default=0.),
                'upper_bound':None,'bounded':None}
    return {'status':'optimal','value':max(r['value'] for r in rows),
            'upper_bound':max(r['upper_bound'] for r in rows),'bounded':True}


def presence(record):
    P,Q,Y=inputs(record)
    eligible,kept=maximal_compatible(record)
    base={'compatible_family_count':len(eligible),'maximal_family_count':len(kept),
          'maximal_families':[family(f) for f in kept], 'history_hash':digest(record['history']),
          'reused_greedy':record['greedy'],'rows':[], 'lp_calls':0}
    t0=time.perf_counter()
    for C in all_blocks(P.shape[0],include_singletons=False):
        omission=next(c for c in record['candidates'] if tuple(c['C'])==C)
        candidates=[f for f in kept if C in family(f)]
        channels={};details=[]
        for channel,ab in [('R',(1.,0.)),('X',(0.,1.)),('sum',(1.,1.))]:
            solved=[]
            for f in candidates:
                blocks=family(f); k=blocks.index(C); v=np.zeros(2*len(blocks));v[2*k:2*k+2]=ab
                result=maximize_history_functional(P,Q,Y,blocks,v,epsilon=record['error_budget'],
                                                   feasible_beta=np.asarray(f['beta']),tol=TOL)
                solved.append(result);base['lp_calls']+=result['lp_calls']
                details.append({'channel':channel,'family':blocks,'result':result})
            channels[channel]=(aggregate(solved) if eligible else
                {'status':'model_budget_incompatible','value':None,'upper_bound':None,'bounded':None})
        r=channels['R']
        if not eligible: classification='model_budget_incompatible'
        elif omission['reference_requires_block']:classification='shared_block_necessary'
        elif r['status']=='unbounded_verified':classification='P_effect_unbounded'
        elif r['status']!='optimal':classification='numerically_unresolved'
        elif r['value']>TOL:classification='P_effect_possible_unresolved'
        elif r['upper_bound']<=TOL:classification='no_P_effect_within_tolerance'
        else:classification='numerical_boundary'
        base['rows'].append({'C':C,'classification':classification,'channels':channels,
            'omission_reference_upper':omission['reference_upper'],
            'omission_reference_lower':omission['reference_lower'],
            'shared_block_necessary':omission['reference_requires_block'],
            'P_price_candidate':classification in ('P_effect_possible_unresolved','P_effect_unbounded'),
            'details':details})
    base['seconds']=time.perf_counter()-t0
    return clean_sets(base)


def embed_beta(fit,blocks):
    out=np.zeros(2*len(blocks));old=family(fit)
    for j,C in enumerate(old):
        k=blocks.index(C);out[2*k:2*k+2]=fit['beta'][2*j:2*j+2]
    return out


def common_witness(eligible,left,right):
    common=set(left)&set(right)
    for fit in eligible:
        if set(family(fit))<=common:
            return np.r_[embed_beta(fit,left),embed_beta(fit,right)]
    return None


def pair_width(record,left,right,C,Pf=None,Qf=None,feasible_x=None):
    P,Q,Y=inputs(record)
    A=block_features(P,Q,left);B=block_features(P,Q,right)
    E=np.block([[A,np.zeros_like(B)],[np.zeros_like(A),B]])
    e=np.r_[Y.ravel(),Y.ravel()]
    if Pf is not None:
        future=np.column_stack([block_features(Pf,Qf,left),-block_features(Pf,Qf,right)])
        E=np.vstack([E,future]);e=np.r_[e,np.zeros(Pf.size)]
    v=np.zeros(A.shape[1]+B.shape[1])
    if C in left:v[2*left.index(C)]=1.
    if C in right:v[A.shape[1]+2*right.index(C)]=-1.
    if record['error_budget']!=0:raise ValueError('This locked width audit uses zero error budgets')
    result=maximize_nonnegative_lp(v,A_eq=E,b_eq=e,feasible_x=feasible_x,tol=TOL)
    if result.get('x') is not None:
        x=np.asarray(result['x']);k=A.shape[1]
        residuals=[float(np.abs(A@x[:k]-Y.ravel()).sum()),float(np.abs(B@x[k:]-Y.ravel()).sum())]
        future_residual=0. if Pf is None else float(np.abs(future@x).sum())
        result['pair_L1_residuals']={'left_history':residuals[0],'right_history':residuals[1],'future_difference':future_residual}
        if max(*residuals,future_residual)>TOL:
            result.update(status='pair_budget_unverified',success=False)
    return result


def width_design(record,screen):
    """Only history and forecast input patterns; no actual observations or labels."""
    if any(r['classification'] in ('numerically_unresolved','numerical_boundary') for r in screen['rows']):
        raise RuntimeError('Unresolved presence may not be dropped from design')
    eligible,kept=maximal_compatible(record)
    targets=[tuple(r['C']) for r in screen['rows'] if r['P_price_candidate']]
    if not eligible:
        return {'status':'model_budget_incompatible','targets':[],'new_lp_calls':0,'menus':[],
                'choices':{},'no_new_data':None}
    if not targets:
        return {'status':'no_P_targets','targets':[],'new_lp_calls':0,'menus':[],
                'choices':{},'no_new_data':True}
    records=[];patterns=[];lp_calls=0;t0=time.perf_counter()
    for pattern in [None,*PATTERNS]:
        Pf,Qf=(None,None) if pattern is None else matrix_for_pattern(pattern)
        widths=[]
        for C in targets:
            solutions=[]
            # Ordered pairs cover both signs. A left family lacking C has
            # r_C-r'_C<=0 and cannot beat a same-model zero difference.
            for lf in kept:
                left=family(lf)
                if C not in left:continue
                for rf in kept:
                    right=family(rf)
                    result=pair_width(record,left,right,C,Pf,Qf,common_witness(eligible,left,right))
                    lp_calls+=result['lp_calls'];solutions.append(result)
                    records.append({'pattern':pattern,'C':C,'left':left,'right':right,'result':result})
            a=aggregate(solutions)
            if a['status']!='optimal':
                raise AuditFailure('A failed/unbounded width cannot become a finite price score',
                                   {'pattern':pattern,'C':C,'partial_pair_records':records,'aggregate':a})
            widths.append({'C':C,**a})
        score=max(w['upper_bound'] for w in widths)
        P,Q,_=inputs(record)
        Fh=block_features(P,Q,all_blocks(P.shape[0]))
        Ff=np.zeros((0,Fh.shape[1])) if pattern is None else block_features(Pf,Qf,all_blocks(P.shape[0]))
        sign,logdet=np.linalg.slogdet(np.eye(Fh.shape[1])+Fh.T@Fh+Ff.T@Ff)
        assert sign>0
        patterns.append({'pattern':pattern,'widths':widths,'max_width':score,'d_opt':float(logdet)})
    menu_rows=[]
    for menu in MENUS:
        pr=[probability(p,menu,'independent') for p in PATTERNS]
        menu_rows.append({'menu':menu['name'],'remaining_width':sum(w*r['max_width'] for w,r in zip(pr,patterns[1:])),
                         'd_opt':sum(w*r['d_opt'] for w,r in zip(pr,patterns[1:])),
                         'cost':sum(w*pattern_cost(p,menu) for w,p in zip(pr,PATTERNS))})
    choices={}
    for metric,sgn in [('remaining_width',-1),('d_opt',1)]:
        best=max(sgn*r[metric] for r in menu_rows)
        ties=[r for r in menu_rows if sgn*r[metric]>=best-TOL]
        selected=min(ties,key=lambda r:(round(r['cost'],12),next(i for i,m in enumerate(MENUS) if m['name']==r['menu'])))
        choices[metric]={'menu':selected['menu'],'ties':[r['menu'] for r in ties]}
    return clean_sets({'status':'complete','targets':targets,'no_new_data':False,'patterns':patterns,
                       'menus':menu_rows,'choices':choices,'pair_records':records,'new_lp_calls':lp_calls,
                       'seconds':time.perf_counter()-t0,'scope':'R-channel width only; not joint support coverage'})


def evaluations(designs,records):
    """Called after persistent design lock; previous evaluation records reused."""
    reused={};additional={}
    for name,d in designs.items():
        old=json.loads((PREVIOUS/('evaluation_'+name+'.json')).read_text())['result']
        reused[name]={'no_new_data':old['no_new_data'],'patterns':[p for p in old['patterns'] if p['weight']==1.],
                      'menus':old['menus']['1.0']}
    raw=records['weak_line'];P,Q,Y=inputs(raw)
    B=tuple(tuple(c) for c in raw['greedy']['selected'])
    h=History(P,Q,Y,B);s=Y[:,0]
    for C in ((0,1),(0,2),(1,2)):
        R=.4*np.ones((3,3))+.03*jblock(3,C)
        R+=np.diag(s-R.sum(axis=1))
        assert np.all(np.diag(R)>=0) and np.allclose(R@P,Y)
        truth=frozenset(((0,),(1,),(2,),(0,1,2),C))
        ev=Evaluator(R,truth,'independent')
        if C==(0,1):
            outcomes=reused['weak_line']['patterns'];no_new=reused['weak_line']['no_new_data'];new=0
        else:
            no_new=observed_path(h,ev);outcomes=[];new=8
            for pattern in PATTERNS:
                pf,qf=matrix_for_pattern(pattern)
                outcomes.append({'pattern':pattern,**observed_path(h,ev,pf,qf)})
        menus=[]
        for menu in MENUS:
            pr=[probability(p,menu,'independent') for p in PATTERNS]
            menus.append({'menu':menu['name'],
                          'full_recovery':sum(w*o['full_recovery'] for w,o in zip(pr,outcomes)),
                          'weak_block_recovery':sum(w*(C in [tuple(b) for b in o.get('final_effective_blocks',[])]) for w,o in zip(pr,outcomes)),
                          'failed_probability':sum(w*(o['outcome']=='failed') for w,o in zip(pr,outcomes)),
                          'cost':sum(w*pattern_cost(p,menu) for w,p in zip(pr,PATTERNS))})
        additional[str(C)]={'R':R,'true_pair':C,'history_equal':True,'new_pattern_evaluations':new,
                            'no_new_data':no_new,'patterns':outcomes,'menus':menus}
    return clean_sets({'reused':reused,'same_history_models':additional,'new_pattern_evaluations':16,
                       'reused_unique_pattern_evaluations':16,
                       'same_history_model_patterns':24,'same_history_reused_patterns':8})


def summarize(screens,designs,evaluation):
    counts=Counter(r['classification'] for s in screens.values() for r in s['rows'])
    channel_counts={k:dict(Counter(r['channels'][k]['status'] for s in screens.values() for r in s['rows'])) for k in ('R','X','sum')}
    return {'presence_datasets':len(screens),'presence_block_checks':sum(len(s['rows']) for s in screens.values()),
            'classifications':dict(counts),'channel_statuses':channel_counts,
            'presence_lp_calls':sum(s['lp_calls'] for s in screens.values()),
            'presence_seconds':sum(s['seconds'] for s in screens.values()),
            'width_lp_calls':sum(d['new_lp_calls'] for d in designs.values()),
            'width_seconds':sum(d.get('seconds',0.) for d in designs.values()),
            'designs':{name:{k:d[k] for k in ('targets','no_new_data','menus','choices','new_lp_calls')} for name,d in designs.items()},
            'actual':{name:{'menus':e['menus'],'no_new_data':e['no_new_data']} for name,e in evaluation['reused'].items()},
            'same_history_models':{name:{'menus':e['menus'],'no_new_data':e['no_new_data'],'R':e['R']} for name,e in evaluation['same_history_models'].items()},
            'new_actual_patterns':evaluation['new_pattern_evaluations'],'reused_actual_patterns':16,
            'scope':'deterministic small-model audit, no statistical confidence or calibrated behavior'}


def checkpoint(path,fn):
    failure=path.with_name('failure_'+path.name)
    if failure.exists():raise RuntimeError('Prior failure retained; inspect before a new run')
    if path.exists():return json.loads(path.read_text())
    try:result=clean_sets(fn())
    except Exception as exc:
        save(failure,{'error':repr(exc),'phase':path.stem,'details':clean_sets(getattr(exc,'details',None))});raise
    save(path,result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/a2d_presence_width_v1')
    a=ap.parse_args()
    with run_lock(a.output):
        protocol=json.loads((ROOT/'protocol_a2d.json').read_text())
        old=json.loads((INPUT/'manifest.json').read_text())
        names=['presence_width_audit.py','compatible_functionals.py','protocol_a2d.json','core.py','experiment.py','face_price_audit.py','exact_patterns.py','optimal_face.py']
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
        inputfiles=[INPUT/(n+'.json') for n in old['protocol']['cases']]
        registry_path=ROOT/'evidence/a2c_completion_final_v2/raw_run_files.sha256.json'
        registry=json.loads(registry_path.read_text())
        for p in inputfiles:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=registry[p.name]:
                raise RuntimeError('Historical evidence changed: '+p.name)
        inputfiles += [INPUT/'manifest.json',registry_path]+[PREVIOUS/('evaluation_'+n+'.json') for n in protocol['width']['histories']]
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputfiles}
        manifest={'protocol':protocol,'sources':sources,'inputs':hashes,'environment':current_environment()}
        mp=a.output/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Changed source/config/input; use a new output')
        save(mp,manifest)
        records={n:json.loads((INPUT/(n+'.json')).read_text()) for n in old['protocol']['cases']}
        screens={}
        for name,record in records.items():
            screens[name]=checkpoint(a.output/('presence_'+name+'.json'),lambda record=record:presence(record))
            print('Presence:',name,flush=True)
        designs={}
        for name in protocol['width']['histories']:
            designs[name]=checkpoint(a.output/('design_'+name+'.json'),lambda name=name:width_design(records[name],screens[name]))
            print('Design:',name,flush=True)
        fingerprint=digest(designs);lock=a.output/'design_lock.json'
        if lock.exists() and json.loads(lock.read_text())['fingerprint']!=fingerprint:raise RuntimeError('Design changed')
        save(lock,{'fingerprint':fingerprint})
        evaluated=checkpoint(a.output/'evaluation.json',lambda:{'design_fingerprint':fingerprint,**evaluations(designs,records)})
        if evaluated['design_fingerprint']!=fingerprint:raise RuntimeError('Evaluation design mismatch')
        report=summarize(screens,designs,evaluated)
        assert report['presence_datasets']==17 and report['presence_block_checks']==96
        save(a.output/'summary.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('designs','actual','same_history_models')},indent=2))

if __name__=='__main__':main()
