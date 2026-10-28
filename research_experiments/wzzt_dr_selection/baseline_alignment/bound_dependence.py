"""Independent fixed-family L1 LPs with nonnegative, unbounded-above weights.

This is a bounded post hoc audit of saved N0 families. Truth-named alternatives
are read from the earlier diagnostic protocol and never generate deployable
candidates. No free-support MILP, RNJ, new data or noise is executed here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def design(scenarios, supports):
    """Explicit [P z z.T, Q z z.T] columns, independent of core model builders."""
    labels = list(scenarios[0]['P_terminal'].columns)
    positions = {label:i for i,label in enumerate(labels)}
    indicators = []
    for support in supports:
        z = np.zeros(len(labels))
        z[[positions[label] for label in support]] = 1.
        if not z.any():
            raise ValueError('empty support')
        indicators.append(z)
    parts, targets = [], []
    for scenario in scenarios:
        if list(scenario['P_terminal'].columns) != labels or list(scenario['Q_terminal'].columns) != labels:
            raise ValueError('inconsistent labels')
        p = scenario['P_terminal'].to_numpy(dtype=float)
        q = scenario['Q_terminal'].to_numpy(dtype=float)
        y = scenario['drop_target'].to_numpy(dtype=float)
        if p.shape != q.shape or y.shape != p.shape:
            raise ValueError('inconsistent observed array dimensions')
        columns = [(np.outer(v @ z, z)).reshape(-1) for v in (p, q) for z in indicators]
        parts.append(np.column_stack(columns))
        targets.append(y.reshape(-1))
    a, y = np.vstack(parts), np.concatenate(targets)
    if not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite observations')
    return a, y


def solve_nonnegative_l1(a, y, *, time_limit=1800., upper_bound=None):
    """Solve min mean |y-Aw| with w>=0; upper=None is the audit target."""
    a = np.asarray(a,dtype=float); y = np.asarray(y,dtype=float)
    if a.ndim != 2 or y.shape != (a.shape[0],) or len(y) == 0:
        raise ValueError('invalid design dimensions')
    m, k = a.shape
    objective = np.r_[np.zeros(k), np.full(m,1./m)]
    constraints = vstack([hstack([csc_matrix(a),-eye(m,format='csc')]),
                          hstack([-csc_matrix(a),-eye(m,format='csc')])],format='csc')
    rhs = np.r_[y,-y]
    bounds = [(0., upper_bound)]*k + [(0.,None)]*m
    started = perf_counter()
    fit = linprog(objective,A_ub=constraints,b_ub=rhs,bounds=bounds,method='highs',
                  options={'time_limit':time_limit,'presolve':True})
    result = {'status_code':int(fit.status),'success':bool(fit.success),
              'message':fit.message,'elapsed_seconds':perf_counter()-started,
              'iterations':getattr(fit,'nit',None),'variable_count':k+m,
              'constraint_count':2*m,'weight_count':k,'upper_bound':upper_bound}
    if fit.x is None:
        return {**result,'status':'failed','objective':None,'dual_objective':None,'weights':None}
    weights = fit.x[:k]
    residual = y-a@weights
    absolute_mae = float(np.mean(np.abs(residual)))
    dual = np.asarray(fit.ineqlin.marginals)
    lower = np.asarray(fit.lower.marginals)
    upper = np.asarray(fit.upper.marginals)
    dual_objective = float(rhs@dual)
    if upper_bound is not None:
        dual_objective += float(upper_bound*np.sum(upper[:k]))
    stationarity = objective - constraints.T@dual - lower - upper
    violation = max(0.,float(np.max(constraints@fit.x-rhs)),-float(np.min(fit.x)))
    if upper_bound is not None:
        violation=max(violation,float(np.max(weights-upper_bound)))
    gap = float(fit.fun-dual_objective)
    checks = {'independent_absolute_mae':absolute_mae,
        'objective_difference':abs(absolute_mae-float(fit.fun)),
        'maximum_primal_violation':violation,'primal_dual_gap':gap,
        'maximum_stationarity_residual':float(np.max(np.abs(stationarity))),
        'maximum_inequality_dual_positive_violation':max(0.,float(np.max(dual))),
        'maximum_lower_dual_negative_violation':max(0.,-float(np.min(lower))),
        'maximum_complementarity_residual':float(np.max(np.abs(dual*(rhs-constraints@fit.x))))}
    verified = fit.status == 0 and checks['objective_difference'] <= 1e-9 and violation <= 1e-7 and abs(gap) <= 1e-9 and checks['maximum_stationarity_residual'] <= 1e-7 and checks['maximum_inequality_dual_positive_violation'] <= 1e-9 and checks['maximum_lower_dual_negative_violation'] <= 1e-9
    result.update({'status':'complete' if verified else 'numerically_unresolved',
        'objective':float(fit.fun),'dual_objective':dual_objective,'weights':weights.tolist(),
        'checks':checks,'numerical_primal_dual_verified':bool(verified),
        'max_weight':float(weights.max()),'weights_above_2_count':int(np.sum(weights>2.+1e-8)),
        'strict_symbolic_certificate':False})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--n0-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    repo=args.repo.resolve(); output=args.output.resolve()
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'rnj_wzzt_core'))
    evid=repo/'research_experiments/wzzt_dr_selection/evidence'
    files={'aligned':evid/'baseline_alignment_v1/alignment_results.json',
           'previous_protocol':evid/'baseline_selection_v2/wrong_block_diagnosis_protocol.json',
           'previous_results':evid/'baseline_selection_v2/wrong_block_diagnosis.json',
           'observations':args.n0_run/'prepared_case.pkl'}
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    aligned=read(files['aligned'])['records'][0]
    prior_protocol=read(files['previous_protocol'])['conditions'][0]
    prior_records={r['id']:r for r in read(files['previous_results'])['records']}
    with files['observations'].open('rb') as stream:
        case,_=pickle.load(stream)
    training,validation=case.training,case.validation
    tasks=[]
    for i,name in ((0,'initial_leaf_plus_rnj'),(1,'original_first_winner')):
        p=aligned['path'][i]
        tasks.append({'id':name,'supports':p['supports'],
            'saved_bounded2':{k:p[k] for k in ('r_values','x_values','train_mae','validation_mae')}})
    for task in prior_protocol['lp_plan']:
        if task['kind']=='first_cross_true_alternative':
            r=prior_records[task['id']]
            tasks.append({'id':task['id'],'supports':task['supports'],
                'post_hoc_true_alternative':task['alternative_true_block'],
                'saved_bounded2':{k:r[k] for k in ('r_values','x_values','train_mae','validation_mae')}})
    tasks.append({'id':'initial_rnj_only','supports':aligned['rnj']['selected'],'saved_bounded2':None})
    if len(tasks)!=7:
        raise ValueError('exact seven-family diagnostic expected')
    output.parent.mkdir(parents=True,exist_ok=True)
    protocol_path=output.with_name('bounds_protocol.json')
    if output.exists() or protocol_path.exists():
        raise FileExistsError('Existing evidence; no duplicate LPs')
    lock=output.with_suffix('.lock')
    with lock.open('x',encoding='utf-8') as stream:stream.write(now())
    protocol={'created_utc':now(),'question':'Does artificial weight cap2 change the first paper15 candidate comparison?',
        'planned_lp_count':7,'new_milp_calls':0,'new_data_or_noise':0,
        'truth_use':'Existing truth-named four alternatives are post hoc diagnostic only, not algorithm candidate generation',
        'solver':'scipy.optimize.linprog/highs','objective':'raw mean absolute residual, no intercept',
        'weight_bounds':[0,None],'per_lp_limit_seconds':1800.,
        'validation':'same saved validation observations, evaluated only',
        'tasks':tasks,'source_sha256':{str(Path(__file__).resolve()):sha(__file__)},
        'input_sha256':{str(v.resolve()):sha(v) for v in files.values()},
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()}}
    write=lambda path,obj:path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write(protocol_path,protocol)
    result={'status':'running','protocol_sha256':sha(protocol_path),'planned_lp_count':7,
            'attempted_lp_count':0,'completed_lp_count':0,'failed_or_unresolved_lp_count':0,
            'new_milp_calls':0,'records':[]}
    write(output,result)
    try:
        for task in tasks:
            a,y=design(training,task['supports']); av,yv=design(validation,task['supports'])
            rec={'id':task['id'],'supports':task['supports'],'observation_count':len(y)}
            saved=task['saved_bounded2']
            if saved is not None:
                old=np.r_[saved['r_values'],saved['x_values']]
                old_train=float(np.mean(np.abs(y-a@old)))
                old_valid=float(np.mean(np.abs(yv-av@old)))
                if not np.isclose(old_train,saved['train_mae'],atol=1e-10,rtol=1e-7) or not np.isclose(old_valid,saved['validation_mae'],atol=1e-10,rtol=1e-7):
                    raise ValueError('independent feature matrix does not reproduce bounded2 fit')
                rec['bounded2_recomputed']={'train_mae':old_train,'validation_mae':old_valid,
                    'max_weight':float(old.max()),'at_bound_count':int(np.sum(old>=1.99))}
            result['attempted_lp_count']+=1
            fit=solve_nonnegative_l1(a,y)
            rec['unbounded_above']=fit
            if fit['weights'] is not None:
                w=np.asarray(fit['weights']);k=len(task['supports'])
                rec.update({'r_values':w[:k].tolist(),'x_values':w[k:].tolist(),
                    'train_mae':float(np.mean(np.abs(y-a@w))),
                    'validation_mae':float(np.mean(np.abs(yv-av@w)))})
                if saved is not None:
                    rec['train_gain_from_removing_cap']=saved['train_mae']-rec['train_mae']
            result['records'].append(rec)
            result['completed_lp_count']+=fit['status']=='complete'
            result['failed_or_unresolved_lp_count']+=fit['status']!='complete'
            write(output,result)
        result.update({'status':'complete' if result['completed_lp_count']==7 else 'complete_with_unresolved',
            'finished_utc':now(),'inputs_unchanged':all(sha(p)==h for p,h in protocol['input_sha256'].items())})
        write(output,result)
        print(json.dumps({'status':result['status'],'completed':result['completed_lp_count'],
            'records':[{'id':r['id'],'mae':r.get('train_mae'),'max_weight':r['unbounded_above'].get('max_weight'),
              'gain':r.get('train_gain_from_removing_cap'),'gap':r['unbounded_above'].get('checks',{}).get('primal_dual_gap')} for r in result['records']]},indent=2))
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
