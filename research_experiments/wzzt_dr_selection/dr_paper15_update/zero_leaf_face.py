"""Bounded numerical optimal-face checks for automatically detected zero leaves.

No new fit, line, tolerance or target is chosen from topology truth. Every
other coefficient remains free in the selected native family. This reports
parameter ranges, never a line-truth probability or necessity certificate.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, canonical
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash


def _plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):_plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [_plain(x) for x in v]
    return v


def audit_zero_leaves(training,original_fit,*,model_names=('multistart_original_rule',),
                      bound=2.,loss_tolerances=(0.,1e-9),max_LP_records=24,
                      zero_tolerance=1e-9,output_path=None):
    obs,days,_=normalize(training,'training');labels=obs[0].columns.tolist()
    if labels!=original_fit['terminal_labels']:raise ValueError('labels mismatch')
    identity=input_hash(obs,days)
    if original_fit.get('input_sha256',{}).get('training') not in (None,identity):raise ValueError('training hash mismatch')
    if any(t<0 or not np.isfinite(t) for t in loss_tolerances):raise ValueError('loss tolerances must be finite nonnegative')
    tasks={}
    for name in model_names:
        model=original_fit['models'][name];source=model.get('selected_uncontracted',model)
        family=canonical(source.get('native_family',source['supports']))
        for block,r,x in zip(source['supports'],source['r'],source['x']):
            if len(block)==1 and max(r,x)<=zero_tolerance:
                key=(family,tuple(block));tasks.setdefault(key,[]).append(name)
    families=sorted({f for f,_ in tasks},key=lambda f:(len(f),f))
    planned=len(families)+2*len(tasks)*len(loss_tolerances)
    result=dict(status='prepared',training_sha256=identity,original_fit_fingerprint=original_fit.get('fingerprint'),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_names=model_names,settings=dict(bound='unbounded' if np.isinf(bound) else bound,
        loss_tolerances=loss_tolerances,zero_tolerance=zero_tolerance,max_LP_records=max_LP_records,
        fixed_early_weights=False,positive_weight_floor=False),planned_LP_records=planned,
        detected_zero_leaf_tasks=len(tasks),base_fits=[],face_solves=[],
        scope='Numerical parameter ranges on exact/tolerance-expanded L1 optimum; no topology probability or physical confidence guarantee.')
    def finish():
        out=_plain(result)
        if output_path is not None:
            p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        return out
    if planned>max_LP_records:result.update(status='not_run_LP_cap');return finish()
    if not tasks:result.update(status='complete',LP_records=0);return finish()
    pool=canonical(s for f in families for s in f)
    lp=PoolLP(obs,pool,'raw',bound=bound);gp=lp.gp;lp.model.Params.DualReductions=0;lp.model.Params.InfUnbdInfo=1
    try:
        for family in families:
            base=deepcopy(lp.fit(family));optimum=float(lp.model.ObjVal)
            result['base_fits'].append(dict(family=family,point=base,solver_optimum=optimum))
            for (f,leaf),methods in tasks.items():
                if f!=family:continue
                i=lp.lookup[leaf];target=lp.weights[[i,i+len(lp.pool)]].sum()
                for tolerance in loss_tolerances:
                    face=lp.model.addConstr(lp.t.sum()/lp.y.size<=optimum+tolerance)
                    try:
                        for direction,sense in [('minimum',gp.GRB.MINIMIZE),('maximum',gp.GRB.MAXIMIZE)]:
                            lp.model.setObjective(target,sense);start=perf_counter();lp.model.optimize()
                            receipt=dict(family=family,leaf=leaf,leaf_label=labels[leaf[0]],methods=methods,
                                direction=direction,loss_tolerance=tolerance,base_optimum=optimum,
                                status=int(lp.model.Status),seconds=perf_counter()-start,solution_count=int(lp.model.SolCount))
                            if lp.model.Status==gp.GRB.OPTIMAL:
                                weights=np.asarray(lp.weights.X);actual=float(np.mean(np.abs(lp.features@weights-lp.y.ravel())))
                                receipt.update(value=float(weights[i]+weights[i+len(lp.pool)]),physical_mean_abs_error=actual,
                                    max_constraint_violation=float(lp.model.ConstrVio),max_bound_violation=float(lp.model.BoundVio),
                                    face_objective_violation=max(0.,actual-optimum-tolerance),all_coefficients=weights.tolist())
                            elif lp.model.Status==gp.GRB.UNBOUNDED:receipt.update(value=None,classification='unbounded')
                            else:receipt.update(value=None,classification='numerical_or_solver_unresolved')
                            result['face_solves'].append(receipt)
                    finally:
                        lp.model.remove(face);lp.model.setObjective(lp.t.sum()/lp.y.size,gp.GRB.MINIMIZE);lp.model.update()
        result['LP_records']=len(lp.solves)+len(result['face_solves']);result['base_LP_receipts']=lp.solves
        result['status']='complete' if all(r['status'] in (gp.GRB.OPTIMAL,gp.GRB.UNBOUNDED) for r in result['face_solves']) else 'completed_with_unresolved'
    finally:lp.close()
    return finish()
