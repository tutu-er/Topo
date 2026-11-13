"""Normalize existing frozen histories; no identifier or parameter refitting."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from ... import io
from ...observations import load
from ...history import freeze_history
from ..posterior_mse_20261112.scoring import active_topology,topology_id
from ..unified_posterior_dr_20261112.history_scan import load_request

ROOT=Path(__file__).resolve().parents[3]

def checked(reference):
    if io.ref(reference['path'])['sha256']!=reference['sha256']:
        raise ValueError('Frozen input drift: '+reference['path'])
    return Path(reference['path'])

def load_entry(entry):
    """Return (normalized_score, observed_train, observed_validation, offline_truth).

    Truth is a separate return value used only by the evaluator. Never pass it
    into procurement. Entry role/p_true fields are not inserted into score.
    """
    source=entry['pool'];request=entry['request']
    score=deepcopy(io.read(checked(entry['score'])))
    if source=='large96':
        native,train,validation=load_request(request)
        original_truth=io.read(Path(request['batch'])/'evaluation'/(request['job']+'.json'))
        if list(original_truth['labels'])!=list(validation.labels):
            raise ValueError('Truth/observation terminal order differs')
        clades=sorted(tuple(sorted(validation.labels[i] for i in c)) for c in original_truth['proper_clades'])
        truth=dict(truth_id=topology_id(clades),truth_clades=clades,labels=list(validation.labels),
            source=io.ref(Path(request['batch'])/'evaluation'/(request['job']+'.json')))
    elif source=='customer':
        train=load(checked(request['observed_refs']['train']))
        validation=load(checked(request['observed_refs']['validation']))
        native=io.read(Path(request['directory'])/'native_result.json')
        truth=io.read(ROOT/'outputs/customer_flexibility_sweep_20261113_v1/offline_truth.json')
    else:
        raise ValueError('This new-test adapter supports large96 and customer histories, not older diagnostic controls: '+source)
    frozen=freeze_history(native,validation)
    raw_index=score.get('raw_map_index',score['map_index'])
    score['raw_map_index']=raw_index
    score['map_index']=next(i for i,m in enumerate(score['models']) if m['topology_id']==score['map_topology_id'])
    score['reference_mse']=native['reference_mse']
    score['validation_tie']=native['config']['validation_tie']
    score['native_bank']=[]
    for i,row in enumerate(native['frozen_candidates']):
        active=active_topology(row['point'],validation.labels,native['config']['zero_tol'])
        score['native_bank'].append(dict(H=frozen.H[i],history_sse=float(frozen.sse[i]),
            internal=int(frozen.internal[i]),topology_id=topology_id(active),active_clades=active))
    for model,param in zip(score['models'],score['parameters'],strict=True):
        i=model['index']
        if param['index']!=i or not np.allclose(param['H'],frozen.H[i],rtol=1e-12,atol=1e-12):
            raise ValueError('Representative matrix differs from original frozen point')
        if not np.isclose(param['history_sse'],frozen.sse[i],rtol=1e-11,atol=1e-13):
            raise ValueError('Historical representative score differs')
    if list(score['labels'])!=list(validation.labels) or len(train.p)!=score['training_points'] or len(validation.p)!=score['validation_points']:
        raise ValueError('Historical observation interface differs')
    score.update(id=entry['id'],case=entry['case'],method=entry['method'])
    score['adapter_receipt']=dict(source=io.ref(__file__),frozen_score=entry['score'],
        parameter_refits=0,new_observations=0,full_native_points=len(score['native_bank']),
        scoring_scope='Original validation only; training observations retained for planning diagnostics.')
    return score,train,validation,truth
