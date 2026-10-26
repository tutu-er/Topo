"""A2c: complete-family omission references versus matrix-relaxation bounds.

Every complete support is fitted once. Tentative greedy blocks never restrict
the completion class. This stage designs no prices and makes no coverage claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import completion_bounds as bounds
from .core import all_blocks, enumerate_laminar_families, fit_l1, select_next_block
from .experiment import ROOT, History, current_environment, digest, jblock, run_lock, save
from .face_price_audit import audit_cases, clean_sets

TOL = 1e-7
EPS = 0.


def extra_history(n, R, P, X=None, Q=None):
    Q = np.zeros_like(P) if Q is None else Q
    X = np.zeros_like(R) if X is None else X
    return History(P, Q, R@P+X@Q, tuple((i,) for i in range(n)))


def specifications():
    previous, metadata = audit_cases()
    specs = [(name, h, (1.,0.), ev.true_blocks) for name,h,_f,ev in previous]
    n=3
    R=np.array([[3.,2.,1.],[2.,3.,1.],[1.,1.,2.]])
    truth3=frozenset(((0,),(1,),(2,),(0,1),(0,1,2)))
    specs.append(('identity_sanity',extra_history(3,R,np.eye(3)),(1.,0.),truth3))
    n=4
    I=np.eye(n); J=jblock(n,tuple(range(n)))
    common=np.column_stack([np.ones(n),2*np.ones(n)])
    def prefix(i):
        b=np.eye(n)[:,i]
        return .2+np.column_stack([b,1-b])
    balanced=I+.4*J+.3*jblock(n,(0,1))+.15*jblock(n,(2,3))
    nested=I+.4*J+.3*jblock(n,(0,1,2))+.15*jblock(n,(0,1))
    zero=I+.4*J+.2*jblock(n,(2,3))
    weak=zero+1e-6*jblock(n,(0,1))
    single=tuple((i,) for i in range(n))
    examples=[
        ('balanced_partial_4',balanced,np.column_stack([common,prefix(0)]),((0,1),(2,3),(0,1,2,3))),
        ('nested_partial_4',nested,np.column_stack([common,prefix(3)]),((0,1),(0,1,2),(0,1,2,3))),
        ('zero_complete_4',zero,np.column_stack([np.ones(n),I]),((2,3),(0,1,2,3))),
        ('weak_complete_4',weak,np.column_stack([np.ones(n),I]),((0,1),(2,3),(0,1,2,3)))]
    for name,r,p,positive in examples:
        specs.append((name,extra_history(n,r,p),(1.,0.),frozenset(single+positive)))
    K=np.eye(3)+.4*jblock(3,(0,1,2))+.2*jblock(3,(0,1))
    I=np.eye(3)
    pq=extra_history(3,.5*K,I,.5*K,I)
    onlyx=extra_history(3,np.zeros((3,3)),np.zeros((3,3)),K,I)
    for name,h,coefficient in [('pq_R_only',pq,(1.,0.)),('pq_sum',pq,(1.,1.)),
                              ('x_R_only',onlyx,(1.,0.)),('x_sum',onlyx,(1.,1.))]:
        specs.append((name,h,coefficient,truth3))
    bad=R.copy();bad[0,1]+=.5
    specs.append(('incompatible_observations',History(I,np.zeros_like(I),bad,((0,),(1,),(2,))),
                  (1.,0.),None))
    return specs,metadata


def greedy_to_stop(history):
    B=history.selected
    trace=[]
    calls=0
    for _ in range(len(all_blocks(history.P.shape[0]))+1):
        step=select_next_block(history.P,history.Q,history.Y,B,tie_tol=TOL)
        calls+=1+len(step.fits)
        if step.stopped is None:
            return {'status':'failed','trace':trace,'selected':B,'lp_calls':calls}
        if step.stopped:
            return {'status':'stopped','trace':trace,'selected':B,'lp_calls':calls,
                    'effective':step.base_fit.effective_blocks(),'loss':step.base_fit.loss}
        c=min(step.best_candidates)
        trace.append({'chosen':c,'best_candidates':step.best_candidates,'improvement':step.improvement})
        B=B+(c,)
    raise RuntimeError('Greedy sequence exceeded dictionary support count')


class BranchCache:
    """A cache belongs to exactly one dataset, never to a truth or support class."""
    def __init__(self, P, Q, Y):
        self.data=(P,Q,Y)
        self.original=bounds.matrix_relaxation_fit
        self.cache={}
        self.calls=0
        self.hits=0
        self.seconds=0.

    def __call__(self,P,Q,Y,order=None,confirmed=(),coefficient=(1.,0.),**kwargs):
        if not all(np.array_equal(x,y) for x,y in zip(self.data,(P,Q,Y))):
            raise ValueError('A branch cache cannot be reused for another dataset')
        normalized=[]
        for i,j,k in bounds._orders(order,P.shape[0]):
            if k==-1:
                i,j=sorted((i,j))
            normalized.append((i,j,k))
        normalized=tuple(sorted(set(normalized)))
        key=(normalized,tuple(tuple(c) for c in confirmed),tuple(coefficient),
             kwargs.get('certificate_tol',TOL),repr(kwargs.get('options')))
        if key in self.cache:
            self.hits+=1
            return deepcopy(self.cache[key])
        t0=time.perf_counter()
        result=self.original(P,Q,Y,order=normalized,confirmed=confirmed,
                             coefficient=coefficient,**kwargs)
        self.seconds+=time.perf_counter()-t0
        self.calls+=1
        self.cache[key]=deepcopy(result)
        return result


def analyze_history(history,coefficient=(1.,0.),epsilon=EPS):
    """No evaluator, true blocks or future observations are accepted."""
    n=history.P.shape[0]
    t0=time.perf_counter()
    greedy=greedy_to_stop(history)
    greedy_seconds=time.perf_counter()-t0
    families=enumerate_laminar_families(n)
    t0=time.perf_counter()
    fitted=[fit_l1(history.P,history.Q,history.Y,f) for f in families]
    exact_seconds=time.perf_counter()-t0
    exact_complete=all(f.success for f in fitted)
    valid=[f for f in fitted if f.success]
    best=min((f.loss for f in valid),default=float('inf'))
    compatible=best<=epsilon+TOL
    base=bounds.matrix_relaxation_fit(history.P,history.Q,history.Y,coefficient=coefficient)
    dictionary=list(all_blocks(n,include_singletons=False))
    tentative=set(tuple(c) for c in greedy.get('selected',()))
    dictionary.sort(key=lambda c:(c not in tentative,c))
    cache=BranchCache(history.P,history.Q,history.Y)
    records=[]
    started=time.perf_counter()
    with patch.object(bounds,'matrix_relaxation_fit',side_effect=cache):
        for C in dictionary:
            omission=[f for f in fitted if C not in f.blocks]
            all_null=all(f.success for f in omission)
            successful=[f for f in omission if f.success]
            upper=min((f.loss for f in successful),default=None)
            lower=min((f.dual_bound for f in successful),default=None) if all_null else None
            witness=min(successful,key=lambda f:f.loss) if successful else None
            relaxed=bounds.block_omission_lower_bound(history.P,history.Q,history.Y,C,coefficient=coefficient)
            lb=relaxed['lower_bound']
            if not exact_complete or not relaxed['success']:
                classification='incomplete'
            elif lb is not None and upper is not None and lb>upper+TOL:
                classification='invalid_bound'
            elif not compatible:
                classification='model_budget_incompatible'
            elif lb is not None and lb>epsilon+TOL:
                classification='necessary_by_matrix'
            elif upper is not None and upper<=epsilon+TOL:
                classification='legal_alternative_found'
            elif lower is not None and lower>epsilon+TOL:
                classification='necessary_but_relaxation_unresolved'
            else:
                classification='numerical_boundary'
            records.append({'C':C,'greedy_tentative':C in tentative,
                'reference_complete':all_null,'reference_lower':lower,'reference_upper':upper,
                'reference_family_count':len(omission),
                'witness':None if witness is None else {'blocks':witness.blocks,'beta':witness.beta,
                    'effective':witness.effective_blocks(),'loss':witness.loss},
                'matrix_lower':lb,'matrix_status':relaxed['status'],'classification':classification,
                'reference_requires_block':bool(compatible and all_null and lower is not None and lower>epsilon+TOL),
                'lower_exceeds_reference':bool(all_null and lb is not None and upper is not None and lb>upper+TOL),
                'lower_below_matrix_base':bool(base['success'] and lb is not None and lb<base['dual_bound']-TOL),
                'bound':relaxed})
    return clean_sets({'n':n,'coefficient':coefficient,'error_budget':epsilon,'numerical_tolerance':TOL,
        'history':{'P':history.P,'Q':history.Q,'Y':history.Y,'initial_B':history.selected},
        'greedy':greedy,'full_reference_complete':exact_complete,'compatible_model_exists':compatible,
        'best_full_loss':best,'compatibility_status':('compatible_witness' if compatible else ('incompatible_all_families_verified' if exact_complete else 'not_found_incomplete')),'matrix_base':base,'candidates':records,
        'family_fits':[{'blocks':f.blocks,'status':f.status,'success':f.success,'loss':f.loss,
                       'dual_bound':f.dual_bound,'gap':f.gap,'beta':f.beta} for f in fitted],
        'lp_calls':{'greedy':greedy['lp_calls'],'complete_family_reference':len(fitted),
                    'matrix_base':1,'matrix_branches':cache.calls,'branch_cache_hits':cache.hits},
        'seconds':{'greedy':greedy_seconds,'complete_family_reference':exact_seconds,
                   'matrix_branch_solves':cache.seconds,'matrix_screen_with_cache':time.perf_counter()-started},
        'parameter_optimization':'all continuous weights within each support, not single representative omission',
        'confirmed_used':[],'strict_arithmetic_certificate':False,'statistical_coverage':False})


def annotate(result,truth):
    """Offline labels only, after the history analysis has completed."""
    if truth is None:
        return {'truth_available':False,'false_matrix_necessities':None}
    true=set(truth)
    positive=[c for c in true if len(c)>1]
    false=sum(r['classification']=='necessary_by_matrix' and tuple(r['C']) not in true
              for r in result['candidates'])
    return {'truth_available':True,'true_positive_blocks':sorted(true),
            'true_nontrivial_count':len(positive),'false_matrix_necessities':false,
            'greedy_full_recovery':set(tuple(c) for c in result['greedy'].get('effective',()))==true,
            'greedy_wrong_effective':[c for c in result['greedy'].get('effective',()) if tuple(c) not in true]}


def wrong_confirmation_diagnostic():
    R=np.array([[3.,2.,1.],[2.,3.,1.],[1.,1.,2.]])
    P=np.eye(3);Q=np.zeros_like(P)
    base=bounds.matrix_relaxation_fit(P,Q,R)
    wrong=bounds.matrix_relaxation_fit(P,Q,R,confirmed=((0,2),))
    return {'purpose':'deliberately invalid confirmation, excluded from normal inference',
            'base':base,'wrong_confirmed':wrong,'expected_base_loss':0,'expected_wrong_loss':2}


def summarize(results,labels):
    cases={}
    for name,r in results.items():
        counts=dict(Counter(c['classification'] for c in r['candidates']))
        cases[name]={'n':r['n'],'greedy':r['greedy'],'best_full_loss':r['best_full_loss'],
                     'compatible_model_exists':r['compatible_model_exists'],'classifications':counts,
                     'lower_exceeds_reference':sum(c['lower_exceeds_reference'] for c in r['candidates']),
                     'lower_below_matrix_base':sum(c['lower_below_matrix_base'] for c in r['candidates']),
                     'lp_calls':r['lp_calls'],'seconds':r['seconds'],'truth_evaluation':labels[name],
                     'blocks':[ {k:c[k] for k in ('C','greedy_tentative','reference_lower','reference_upper',
                          'matrix_lower','classification','reference_requires_block')} for c in r['candidates']]}
    return {'datasets':len(results),'block_checks':sum(len(r['candidates']) for r in results.values()),
            'cases':cases,'prices_designed':0,'inference':'numerical fixed-model necessity/alternative diagnostics only',
            'confirmed_feedback_applied':False}


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('completion_audit.py','completion_bounds.py','protocol_a2c.json','core.py',
         'experiment.py','face_price_audit.py','optimal_face.py','exact_patterns.py')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'a2c_completion_final_v2')
    args=parser.parse_args()
    with run_lock(args.output):
        protocol=json.loads((ROOT/'protocol_a2c.json').read_text(encoding='utf-8'))
        manifest={'protocol':protocol,'sources':source_hashes(),'environment':current_environment()}
        mf=args.output/'manifest.json'
        if mf.exists() and json.loads(mf.read_text(encoding='utf-8'))!=manifest:
            raise RuntimeError('Changed source/config/environment; use a new output directory')
        save(mf,manifest)
        specs,metadata=specifications()
        assert [s[0] for s in specs]==protocol['cases']
        save(args.output/'prefix_metadata.json',metadata)
        results={}
        for name,h,coefficient,_ in specs:
            path=args.output/(name+'.json')
            failed=args.output/('failure_'+name+'.json')
            if failed.exists():
                raise RuntimeError('Prior case failure is retained; inspect and use a new run')
            if path.exists():
                results[name]=json.loads(path.read_text(encoding='utf-8'))
            else:
                try:
                    results[name]=analyze_history(h,coefficient,protocol['observation_error_budget'])
                except Exception as exc:
                    save(failed,{'case':name,'phase':'history_analysis','error':repr(exc)})
                    raise
                save(path,results[name])
            print('Completed history:',name,flush=True)
        # Truth labels are consulted only after all complete history analyses are stored.
        labels={name:annotate(results[name],truth) for name,_h,_c,truth in specs}
        out=summarize(results,labels)
        assert out['block_checks']==protocol['planned_nontrivial_block_checks']
        assert sum(r['lp_calls']['complete_family_reference'] for r in results.values())==protocol['exact_support_fit_count']
        save(args.output/'truth_evaluation.json',labels)
        save(args.output/'summary.json',out)
        save(args.output/'wrong_confirmation.json',wrong_confirmation_diagnostic())
        print('Complete:',out['datasets'],'datasets;',out['block_checks'],'blocks')
        for name,c in out['cases'].items():
            print(name,c['classifications'],'invalid lower',c['lower_exceeds_reference'])


if __name__=='__main__':
    main()
