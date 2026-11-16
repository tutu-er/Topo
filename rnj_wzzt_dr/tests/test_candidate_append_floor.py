"""Append-only output-floor integration; small fixed-support QP, never MIQP."""
from copy import deepcopy
from dataclasses import asdict
import json
from unittest.mock import patch

import numpy as np
import pytest

from ..observations import Observation, fingerprint
from ..identification import algorithm, output_floor
from ..identification.optimization import QPFitter, errors, response_matrices
from ..history import freeze_history
from ..experiments.distribution_panel_20261115 import data


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def fixture(*,floor=.08,legacy=False):
    p=np.eye(4)[:,:2];q=np.eye(4)[:,2:]
    raw=dict(supports=[[0],[1],[0,1]],r=[.2,.3,.01],x=[.1,.2,.01])
    R,X=response_matrices(raw,2)
    observed=Observation(p,q,p@R.T+q@X.T,(101,102))
    fitter=QPFitter(observed,reference_mse=.1,bound=2.,qp_seconds=10.)
    base=fitter.fit([[0],[1]])
    config=algorithm.SearchConfig(internal_floor=floor)
    points=[base]
    calibration=dict(a=1.,b=1.,median_root_depth=1.,m=1.,terminal_count=2,
                     labels=[101,102],training_sha256=fingerprint(observed))
    contraction=None
    if floor:
        points,contraction=output_floor.contract_bank(points,fitter,floor,calibration)
    selected,mae,bank=algorithm._output(points,observed,observed,.1,config)
    original=dict(method='101',selected=selected,selected_mse_view=deepcopy(selected),selected_mae_view=mae,
        frozen_candidates=bank,path=deepcopy(points),events=[],reference_mse=.1,config=asdict(config),
        output_contraction=contraction,QP=deepcopy(fitter.records),MIQP=[],
        fresh_seconds=2.5,total_seconds=3.5,common_seconds=1.,
        handoff=dict(schema='rnj_wzzt_dr_v1',training_sha256=fingerprint(observed),validation_sha256=fingerprint(observed)))
    if legacy:original['config'].pop('internal_floor')
    raw.update(mse=errors(observed,raw)['mse'],objective=errors(observed,raw)['mse']/.1,
               reference_mse=.1,fit_loss='normalized_MSE')
    return observed,original,raw


def test_legacy_config_missing_floor_is_explicitly_zero_and_does_not_refit():
    obs,original,raw=fixture(floor=0.,legacy=True)
    before=encoded(original)
    with (patch.object(output_floor,'contract_bank',side_effect=AssertionError('legacy must not contract')),
          patch.object(QPFitter,'fit',side_effect=AssertionError('legacy must not fit'))):
        result=data.append_candidate(original,raw,obs,obs)
    assert encoded(original)==before
    assert result['frozen_candidates'][:-1]==original['frozen_candidates']
    assert result['frozen_candidates'][-1]['point']==raw
    assert result['candidate_append_records'][-1]['internal_floor']==0.
    assert result['candidate_append_records'][-1]['legacy_missing_floor_means_zero']
    assert result['output_contraction'] is None
    freeze_history(result,obs)


def test_positive_floor_contracts_only_new_point_and_updates_full_receipt():
    obs,original,raw=fixture()
    old_bytes=encoded(original);prefix=encoded(original['frozen_candidates'])
    old_receipts=encoded(original['output_contraction']['candidates'])
    result=data.append_candidate(original,raw,obs,obs)
    assert encoded(original)==old_bytes
    assert encoded(result['frozen_candidates'][:-1])==prefix
    assert encoded(result['output_contraction']['candidates'][:-1])==old_receipts
    added=result['frozen_candidates'][-1]
    assert added['index']==1 and added['point']['supports']==[[0],[1]]
    assert result['output_contraction']['candidate_count']==2
    assert result['output_contraction']['completed_candidates']==2
    assert result['output_contraction']['candidates'][-1]['index']==1
    assert result['output_contraction']['candidates'][-1]['final_supports']==[[0],[1]]
    assert result['output_contraction']['calibration']==original['output_contraction']['calibration']
    for key in ('total_seconds','fresh_seconds','common_seconds','QP','MIQP','path','events'):
        assert result[key]==original[key]
    append=result['candidate_append_records'][-1]
    assert append['status']=='complete' and len(append['QP'])==1
    assert append['seconds']>=append['floor_seconds']>=0.
    assert append['prefix_parameters_unchanged'] and append['prefix_history_SSE_unchanged']
    frozen=freeze_history(result,obs)
    np.testing.assert_array_equal(frozen.H[0],freeze_history(original,obs).H[0])
    tampered=deepcopy(result)
    tampered['output_contraction']['candidates'][-1]['final_parameter_sha256']='0'*64
    with pytest.raises(ValueError,match='changed after output contraction'):
        freeze_history(tampered,obs)


def test_repeated_appends_remap_indices_without_rewriting_prior_receipts():
    obs,original,raw=fixture()
    first=data.append_candidate(original,raw,obs,obs)
    previous=encoded(first['output_contraction']['candidates'])
    second=data.append_candidate(first,raw,obs,obs)
    assert encoded(second['output_contraction']['candidates'][:-1])==previous
    assert [r['index'] for r in second['output_contraction']['candidates']]==[0,1,2]
    assert second['output_contraction']['candidate_count']==3
    assert len(second['candidate_append_records'])==2
    assert len(second['output_contraction']['append_lineage'])==2
    freeze_history(second,obs)


def test_none_point_returns_original_content_without_new_records_or_work():
    obs,original,_=fixture()
    with patch.object(data,'freeze_history',side_effect=AssertionError('no append work')):
        result=data.append_candidate(original,None,obs,obs)
    assert encoded(result)==encoded(original)


def test_failed_new_contraction_raises_structured_details_and_keeps_original():
    obs,original,raw=fixture();before=encoded(original)
    failure=TimeoutError('fixture soft fixed-QP timeout');failure.receipt={'completed_steps':0}
    with patch.object(output_floor,'contract_bank',side_effect=failure):
        with pytest.raises(RuntimeError,match='contract_extra_point') as caught:
            data.append_candidate(original,raw,obs,obs)
    details=caught.value.append_receipt
    assert details['status']=='failed' and details['original_bank_preserved']
    assert details['contraction_failure']=={'completed_steps':0}
    assert details['error_type']=='TimeoutError'
    assert encoded(original)==before


def test_additional_search_old_config_is_zero_without_running_miqp():
    obs,original,raw=fixture(floor=0.,legacy=True)
    original['events']=[dict(stage='E',pool=[[0],[1],[0,1]],keep=[[0],[1]],point=raw)]
    called=[]
    def fake_search(train,pool,keep,warm,config,reference,seconds):
        called.append(config.internal_floor)
        return dict(closed_point=None)
    with patch.object(data.pool_diagnostic,'run_search',side_effect=fake_search):
        result=data.additional_search(original,obs,seconds=1.)
    assert called==[0.] and result['status']=='no_returned_point'
