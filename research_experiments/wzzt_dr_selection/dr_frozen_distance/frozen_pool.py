"""Two historical predictors, frozen before any new price/response experiment.

There is no topology search, parameter fitting or validation refitting here.
The existing historical competition rule chooses the two saved points.  The
original3% selection and historical.16 output mapping are then reused exactly.
The pair is a conditional hypothesis set, not a guarantee it contains truth.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np

VALIDATION_FRACTION = .03
VALIDATION_EPSILON = 1.e-12
CONTRACTION_FRACTION = .16
CONTRACTION_EPSILON = 1.e-10


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=_sha(path))


def _checked(reference):
    path=Path(reference['path'])
    if _sha(path)!=reference['sha256']:
        raise ValueError('input hash mismatch: '+str(path))
    return path


def _finite(value,name,shape=None):
    if hasattr(value,'to_numpy'):value=value.to_numpy(dtype=float)
    array=np.asarray(value,dtype=float)
    if shape is not None and array.shape!=shape:
        raise ValueError(f'{name} has shape{array.shape}, expected{shape}')
    if not np.isfinite(array).all():raise ValueError(name+' contains nonfinite values')
    return array


def _power(scene,labels=None):
    p0,q0=scene['P_terminal'],scene['Q_terminal']
    if labels is not None:
        for key,frame in [('P_terminal',p0),('Q_terminal',q0),('V_terminal',scene.get('V_terminal'))]:
            if frame is not None and hasattr(frame,'columns') and list(frame.columns)!=list(labels):
                raise ValueError(key+' terminal order differs from frozen predictor')
        if 'labels' in scene and list(scene['labels'])!=list(labels):
            raise ValueError('explicit terminal order differs from frozen predictor')
    p=_finite(p0,'P_terminal');q=_finite(q0,'Q_terminal')
    if p.ndim!=2 or q.shape!=p.shape or p.shape[0]==0:
        raise ValueError('P/Q must be finite nonempty time-by-terminal arrays')
    if labels is not None and p.shape[1]!=len(labels):raise ValueError('terminal count differs from frozen predictor')
    return p,q


def _target(scene,shape):
    if 'V_terminal' in scene and 'V_root_observed' in scene:
        voltage=_finite(scene['V_terminal'],'V_terminal',shape)
        root=_finite(scene['V_root_observed'],'V_root_observed').reshape(-1)
        if root.shape!=(shape[0],):raise ValueError('root voltage row count differs')
        y=root[:,None]**2-voltage**2
        if 'drop_target' in scene:
            supplied=_finite(scene['drop_target'],'drop_target',shape)
            if not np.allclose(supplied,y,rtol=0,atol=1.e-12):
                raise ValueError('drop_target differs from observed root/terminal squared voltages')
        return y
    if 'drop_target' in scene:
        return _finite(scene['drop_target'],'drop_target',shape)
    raise ValueError('observed V_terminal/V_root_observed or explicit drop_target required')


def _matrices(point,n):
    supports=point['supports'];r=point['r'];x=point['x']
    if not len(supports)==len(r)==len(x):raise ValueError('saved support/weight lengths differ')
    matrices=[]
    for key,weights in [('r',r),('x',x)]:
        values=_finite(weights,key,(len(supports),))
        if np.any(values<0) or np.any(values>2):
            raise ValueError('saved weight outside original[0,2] domain')
        matrix=np.zeros((n,n))
        for block,value in zip(supports,values,strict=True):
            if not block or len(set(block))!=len(block) or any(int(v)!=v or v<0 or v>=n for v in block):
                raise ValueError('invalid frozen support')
            matrix[np.ix_(block,block)]+=value
        matrices.append(matrix)
    return matrices


def predict(model,scene):
    """Return T-by-n predicted squared-voltage drop in frozen terminal order.

    DataFrame columns are checked; plain arrays are interpreted in the frozen
    label order. This operation never uses measured Y to change parameters.
    """
    labels=model.get('labels')
    p,q=_power(scene,labels)
    n=p.shape[1]
    r=_finite(model['R'],'R',(n,n));x=_finite(model['X'],'X',(n,n))
    return p@r.T+q@x.T


def _absolute_sum(model,scene):
    prediction=predict(model,scene)
    return float(np.abs(prediction-_target(scene,prediction.shape)).sum()),prediction.size


def _output(point,calibration):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract
    return deepcopy(contract(point,calibration)['supports'])


def _canonical_family(point):
    family=point.get('full_fit',{}).get('supports',point.get('native_family',point['supports']))
    return {tuple(sorted(s)) for s in family}


def _fingerprint(pool):
    value={k:v for k,v in pool.items() if k!='pool_sha256'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze_pool(case_protocol_path,output_path,competition_selector=None):
    """Freeze existing selected historical point and rule-chosen competitor.

    Only the case protocol, historical observation pickle and saved historical
    fit are read. Source graph/truth/metadata and any new observation are not.
    Output is exclusive: an existing freeze is never silently overwritten.
    """
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.prepare import _choose_competition
    case_path=Path(case_protocol_path).resolve();case=_json(case_path)
    history_path=_checked(case['history_packet']);fit_path=_checked(case['historical_baseline_result'])
    with history_path.open('rb') as stream:packet=pickle.load(stream)
    history=packet['history'] if 'history' in packet else packet
    baseline=_json(fit_path)
    labels=list(case['labels'])
    if list(history['labels'])!=labels:raise ValueError('historical terminal order differs from case protocol')
    if float(history['base_kw'])!=float(case['base_kw']):raise ValueError('historical power base differs')
    if history.get('root_label',case['root_label'])!=case['root_label']:raise ValueError('historical root differs')
    train=history['training'];valid=history['validation']
    train_rows=sum(_power(s,labels)[0].shape[0] for s in train)
    valid_rows=sum(_power(s,labels)[0].shape[0] for s in valid)
    if train_rows!=288 or valid_rows!=288:raise ValueError('full original288/288 historical rows required')
    coarse,refined,rule=(competition_selector or _choose_competition)(baseline)
    calibration=deepcopy(baseline['calibrations']['raw'])
    for name in ['a','b','median_root_depth']:
        if not np.isfinite(float(calibration[name])):raise ValueError('invalid historical calibration')
    models=[];validation_cells=valid_rows*len(labels)
    for identity,point,index in [('coarse',coarse,rule['selected_index']),('refined',refined,rule['alternative_index'])]:
        if point.get('preprocess','raw')!='raw':raise ValueError('only common raw validation scores may be compared')
        r,x=_matrices(point,len(labels))
        model=dict(candidate_id=identity,historical_index=int(index),point=deepcopy(point),
            labels=labels.copy(),R=r.tolist(),X=x.tolist())
        training_abs=sum(_absolute_sum(model,scene)[0] for scene in train)
        validation_abs=sum(_absolute_sum(model,scene)[0] for scene in valid)
        training_mae=training_abs/(train_rows*len(labels));validation_mae=validation_abs/validation_cells
        if abs(training_mae-float(point['objective']))>1.e-8:
            raise ValueError(identity+' saved training loss fails replay')
        if abs(validation_mae-float(point['validation_mae']))>1.e-10:
            raise ValueError(identity+' saved validation loss fails replay')
        output=_output(point,calibration)
        model.update(historical_training_abs_sum=training_abs,historical_training_mae=training_mae,
            historical_validation_abs_sum=validation_abs,historical_validation_mae=validation_mae,
            raw_support_count=len(point['supports']),frozen_output_supports=output,
            frozen_output_support_labels=[[labels[i] for i in block] for block in output],
            frozen_output_support_count=len(output))
        models.append(model)
    added=_canonical_family(refined)-_canonical_family(coarse)
    removed=_canonical_family(coarse)-_canonical_family(refined)
    # Descriptive historical resource qualification only, not future capacity
    # certification. These statistics do not alter the fixed predictors.
    qualification=[]
    for node in (201,213):
        if node not in labels:continue
        position=labels.index(node)
        p=np.concatenate([_power(s,labels)[0][:,position] for s in train+valid])*float(case['base_kw'])
        q=np.concatenate([_power(s,labels)[1][:,position] for s in train+valid])*float(case['base_kw'])
        qualification.append(dict(node_label=node,historical_P_min_kw=float(p.min()),
            historical_Q_min_kvar=float(q.min()),nominal_1kw_component_supported=bool(p.min()>=1),
            scope='noisy historical observations only; no future availability or behavior calibration'))
    pool=dict(schema_version=1,status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),
        labels=labels,base_kw=float(case['base_kw']),root_label=int(case['root_label']),
        case_protocol=_reference(case_path),history_packet=deepcopy(case['history_packet']),
        historical_fit=deepcopy(case['historical_baseline_result']),calibration=calibration,
        models=models,selection_order=['coarse','refined'],competition_rule=rule,
        train_rows=train_rows,validation_rows=valid_rows,historical_validation_cells=validation_cells,
        dispute_added_supportlabels=[[labels[i] for i in block] for block in sorted(added)],
        dispute_removed_supportlabels=[[labels[i] for i in block] for block in sorted(removed)],
        validation_fraction=VALIDATION_FRACTION,validation_epsilon=VALIDATION_EPSILON,
        contraction_fraction=CONTRACTION_FRACTION,contraction_epsilon=CONTRACTION_EPSILON,
        selector_api='dr_pool_v2.selection.select',contract_api='dr_pool_v2.selection.contract',
        resource_qualification=qualification,new_fit_calls=0,new_search_calls=0,
        source_history_used='all288training plus288validation; parameters only from saved training fits',
        coverage_scope='two rule-chosen historical explanations; no true-structure coverage assertion',
        new_future_or_truth_read=False,parameter_rule='all historical R/X and output calibration frozen')
    pool['pool_sha256']=_fingerprint(pool)
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(pool,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return pool


def _select(pool,losses):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import select
    order=pool['selection_order'];model_by_id={m['candidate_id']:m for m in pool['models']}
    if set(order)!=set(model_by_id) or len(order)!=len(model_by_id):raise ValueError('invalid frozen candidate order')
    # Only the scalar validation score is new. Original parameters, full point
    # metadata and historical calibration are passed unchanged to the existing
    # selector, including its original3% and contraction behavior.
    points=[dict(deepcopy(model_by_id[identity]['point']),candidate_id=identity,
                 validation_mae=float(losses[identity])) for identity in order]
    selected=select(points,pool['calibration']);chosen=selected['candidate_id']
    best=min(losses.values())
    eligible=[identity for identity in order if losses[identity]<=1.03*best+VALIDATION_EPSILON]
    model=model_by_id[chosen]
    if selected['supports']!=model['frozen_output_supports']:
        raise ValueError('existing contraction differs from frozen historical output map')
    return dict(selected_candidate_id=chosen,eligible_candidate_ids=eligible,minimum_validation_mae=float(best),
        selected_output_supports=deepcopy(selected['supports']),
        selected_output_support_labels=deepcopy(model['frozen_output_support_labels']),
        selected_raw_support_count=len(selected['selected_uncontracted']['supports']),
        selected_output_support_count=len(selected['supports']),
        original_selector_index=int(selected['selection_index']))


def score_frozen(pool,new_scene):
    """Score frozen predictors on full historical validation plus new scene(s).

    Main selection keeps all historical validation cells. Day-only selection is
    separately named diagnostic output using the same3%/support-count rule.
    No parameter/output-map update or candidate regeneration is possible here.
    """
    if pool.get('status')!='frozen':raise ValueError('frozen historical pool required')
    if 'pool_sha256' in pool and _fingerprint(pool)!=pool['pool_sha256']:
        raise ValueError('frozen pool content changed')
    scenes=[new_scene] if isinstance(new_scene,Mapping) else list(new_scene)
    if not scenes:raise ValueError('at least one new observed scene required')
    models=pool['models'];rows=[];cumulative={};day={};all_predictions={};new_cells=None
    for model in models:
        total=0.;cells=0;predictions=[]
        for scene in scenes:
            prediction=predict(model,scene);predictions.append(prediction)
            total+=float(np.abs(prediction-_target(scene,prediction.shape)).sum());cells+=prediction.size
        if new_cells is None:new_cells=cells
        if cells!=new_cells:raise ValueError('predictors used unequal observation cells')
        identity=model['candidate_id'];day[identity]=total/cells
        cumulative[identity]=(model['historical_validation_abs_sum']+total)/(pool['historical_validation_cells']+cells)
        rows.append(dict(candidate_id=identity,historical_validation_abs_sum=model['historical_validation_abs_sum'],
            new_validation_abs_sum=total,new_validation_cells=cells,new_validation_mae=day[identity],
            cumulative_validation_mae=cumulative[identity]))
        all_predictions[identity]=np.concatenate(predictions,axis=0)
    chosen=_select(pool,cumulative);day_only=_select(pool,day)
    first,second=pool['selection_order']
    return dict(status='complete',rows=rows,**chosen,day_only_selection=day_only,
        cumulative_validation_cells=pool['historical_validation_cells']+new_cells,
        cumulative_validation_rows=(pool['historical_validation_cells']+new_cells)//len(pool['labels']),
        historical_validation_rows=pool['validation_rows'],new_validation_rows=new_cells//len(pool['labels']),
        cumulative_pair_exclusion_margin=cumulative[first]-1.03*cumulative[second]-VALIDATION_EPSILON,
        cumulative_current_coarse_exit_margin=cumulative[first]-1.03*min(cumulative.values())-VALIDATION_EPSILON,
        day_only_pair_exclusion_margin=day[first]-1.03*day[second]-VALIDATION_EPSILON,
        day_only_current_coarse_exit_margin=day[first]-1.03*min(day.values())-VALIDATION_EPSILON,
        new_prediction_pair_distance_l1=float(np.abs(all_predictions[first]-all_predictions[second]).mean()),
        parameters_changed=False,calibration_changed=False,new_fit_calls=0,new_search_calls=0,
        probability_scope='deterministic frozen-predictor scores; no recovery or excitation probability asserted',
        output_map_scope='unchanged historical.16 contraction; chosen model may differ')
