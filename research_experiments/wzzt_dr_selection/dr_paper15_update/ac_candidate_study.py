"""Same-family AC-L1 correction and identical held-out selection for all groups."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json
import traceback
from .ac_family_refit import fit_family_ac,predict_ac
from .candidates_update import write,plain
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract


def select_ac(points,calibration,train,valid,physical):
    best=min(p['validation_mae'] for p in points)
    eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
    index=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
    before=deepcopy(points[index]);point=contract(before,calibration)
    tr=predict_ac(point,train,**physical);va=predict_ac(point,valid,**physical)
    labels=list(train[0].columns);rr,xx=matrices(len(labels),point['supports'],point['r'],point['x'])
    return dict(point,objective=tr['mae'],train_mae=tr['mae'],validation_mae=va['mae'],
        selection_validation_mae=before['validation_mae'],selection_index=index,
        selected_uncontracted=before,score_model='AC_squared_voltage_drop_L1',
        selected_origin='ac_iterative_family',support_labels=[[labels[i] for i in s] for s in point['supports']],
        r_matrix=rr.tolist(),x_matrix=xx.tolist(),post_contraction_convergence=dict(training=tr['convergence'],validation=va['convergence']))


def run_ac_candidates(training,validation,original_fit,linear_result,*,base_kv,base_mva,
                      outer_iterations=3,max_families=512,output_path=None):
    if linear_result['status'] not in ('complete','completed_with_failures'):
        raise ValueError('complete linear family stage required')
    if len(linear_result['candidates'])>max_families:
        raise ValueError('AC candidate cap exceeded before fitting')
    train,_,_=normalize(training,'training');valid,_,_=normalize(validation,'validation')
    physical=dict(base_kv=base_kv,base_mva=base_mva,root_label=1)
    started=perf_counter();records=[];failures=[]
    result=dict(status='running',settings=dict(outer_iterations=outer_iterations,max_families=max_families,
        coefficient_domain='nonnegative unbounded',selection='same 3% support-count rule; original contraction; AC rescored after contraction',
        fixed_observed_PQ=True,validation_used_in_refit=False,physical=physical),candidates=records,failures=failures,models={})
    for c in linear_result['candidates']:
        try:
            fitted=fit_family_ac(train,c['family'],initial_point=c['point'],coefficient_bound=float('inf'),
                                 outer_iterations=outer_iterations,**physical)
            item=dict(index=c['index'],family=c['family'],provenance=c['provenance'],fit=fitted)
            records.append(item)
            if not fitted['success']:
                failures.append(dict(index=c['index'],status=fitted['status'],stage='AC_training',error=fitted.get('error')))
            else:
                va=predict_ac(fitted['point'],valid,**physical)
                item['point']=dict(fitted['point'],validation_mae=va['mae'],candidate_index=c['index'],
                    provenance=c['provenance'],validation_convergence=va['convergence'],origin='ac_iterative_family')
        except Exception as exc:
            failures.append(dict(index=c['index'],stage='AC_training_or_validation',error=repr(exc),traceback=traceback.format_exc()))
        result.update(elapsed_seconds=perf_counter()-started,fitted_candidates=len(records))
        write(output_path,result)
        if (c['index']+1)%10==0:
            print(json.dumps(dict(stage='AC_family_refit',candidate=c['index']+1,total=len(linear_result['candidates']),
                                 seconds=result['elapsed_seconds'],failures=len(failures))),flush=True)
    groups={name:[] for name in ('rnj_refit','wzzt_refit','rnj_contractions','wzzt_contractions')}
    for c in records:
        if 'point' not in c:continue
        prov=c['provenance'];membership=dict(rnj_refit=any(p['source']=='rnj_bank' and p['operation']=='parent' for p in prov),
            wzzt_refit=any(p['operation']=='parent' for p in prov),rnj_contractions=any(p['source']=='rnj_bank' for p in prov),
            wzzt_contractions=True)
        for name,yes in membership.items():
            if yes:groups[name].append(c['point'])
    for name,points in groups.items():
        if not points:continue
        try:
            result['models'][name]=select_ac(points,original_fit['calibrations']['raw'],train,valid,physical)
        except Exception as exc:
            failures.append(dict(stage='post_contraction_selection',group=name,error=repr(exc),traceback=traceback.format_exc()))
    result.update(status='complete' if not failures else 'completed_with_failures',elapsed_seconds=perf_counter()-started,
                  candidate_counts={k:len(v) for k,v in groups.items()},global_optimality_claim=False)
    write(output_path,result);return plain(result)
