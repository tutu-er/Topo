"""Independent audit regressions; no fitting, AC or identification calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from ...observations import Observation
from .audit_report import (Audit, check_pruning, family, loss, matrix, qp_certificate,
                           response_metrics, strict_select, summarize, support_metrics)


def p(supports,r,x=None):
    return dict(supports=supports,r=r,x=[0.]*len(r) if x is None else x)


def obs(n=2):
    a=np.eye(2*n)
    return Observation(a[:,:n],a[:,n:],np.zeros((2*n,n)),tuple(range(101,101+n)))


def test_matrix_and_physical_scope_do_not_confuse_internal_with_full():
    point=p([[0],[1],[2],[0,1],[0,1,2]],[.1,.2,.3,.4,.5],[.01,.02,.03,.04,.05])
    expected=np.array([[1.,.9,.5],[.9,1.1,.5],[.5,.5,.8]])
    np.testing.assert_allclose(matrix(point,3)[:,:3],expected)
    truth=[[0],[1],[2],[0,1]]
    stats=support_metrics(point,truth,3,1e-9)
    assert stats['internal_exact'] and not stats['full_exact']
    assert stats['full_extra']==1 and stats['common_extra']==1
    point['r'][0]=point['x'][0]=0.
    stats=support_metrics(point,truth,3,1e-9)
    assert stats['singleton_missing']==1 and stats['full_missing']==1


def test_strict_selection_uses_stored_internal_count_and_original_order():
    o=obs(3)
    c=[dict(source_candidate_id=8,point=p([[0,1]],[0.])),
       dict(source_candidate_id=3,point=p([[0],[1],[2]],[0.,0.,0.]))]
    result=strict_select(c,o,2.,1e-12)
    assert result['selected']==3 and result['eligible']==[8,3]
    c[1]['point']=p([[1,2]],[0.])
    assert strict_select(c,o,2.,1e-12)['selected']==8


def test_cumulative_scoring_weights_observation_cells_not_days():
    val=Observation(np.ones((1,1)),np.zeros((1,1)),np.ones((1,1)),(1,))
    new=Observation(np.ones((3,1)),np.zeros((3,1)),np.full((3,1),3.),(1,))
    c=[dict(source_candidate_id=7,point=p([[0]],[1.])),dict(source_candidate_id=4,point=p([[0]],[3.]))]
    result=strict_select(c,val,1.,1e-12,new)
    assert result['selected']==4 and result['scoring_cells']==4
    assert [r['scored_mse'] for r in result['losses']]==[3.,1.]


def test_supporting_plane_certificate_detects_nonoptimal_coefficient_without_solving():
    o=obs(1)
    exact=qp_certificate(p([[0]],[0.]),o,1.,2.)
    wrong=qp_certificate(p([[0]],[1.]),o,1.,2.)
    assert exact==dict(objective=0.,lower_bound=0.,certificate_gap=0.)
    assert wrong['objective']==.5 and wrong['lower_bound']==0. and wrong['certificate_gap']==.5
    empty=qp_certificate(p([],[]),o,1.,2.)
    assert empty==exact


def test_zero_threshold_compares_entire_original_point():
    o=obs(1);original=p([[0]],[0.]);original['source']='original'
    rec=dict(steps=[],deletions=0,refits=0)
    result=check_pruning(original,deepcopy(original),rec,dict(threshold=0.),None,None,{},o)
    assert result['deletions']==0
    modified=deepcopy(original);modified['source']='changed'
    with pytest.raises(ValueError,match='Zero threshold changed'):
        check_pruning(original,modified,rec,dict(threshold=0.),None,None,{},o)


def test_trace_rejects_basis_readdition_and_wrong_weakest_deletion():
    o=obs();original=p([[0],[1]],[.01,.02]);middle=p([[1]],[.02]);end=p([],[])
    setting=dict(threshold=.1,metric='ohm',scope='all')
    rec=dict(numerical_absolute_tolerance=1e-11,observed_labels=[101,102],observations_removed=False,
        fixed_supports_added=False,steps=[],floor_satisfied=True,minimum_final_strength=None)
    for before,after,removed,strength in [(original,middle,[0],.01),(middle,end,[1],.02)]:
        rec['steps'].append(dict(removed_support=removed,removed_strength=strength,
            support_count_before=len(before['supports']),support_count_after=len(after['supports']),
            training_loss_before=loss(before,o),training_loss_after=loss(after,o)))
    cache={family(middle):middle,():end}
    good=check_pruning(original,end,rec,setting,None,2.,cache,o)
    assert good['removed_singletons']==2
    bad=deepcopy(rec);bad['steps'][0]['removed_support']=[1]
    with pytest.raises(ValueError,match='weakest'):
        check_pruning(original,end,bad,setting,None,2.,cache,o)
    with pytest.raises(ValueError,match='Final point differs'):
        check_pruning(original,original,rec,setting,None,2.,cache,o)


def test_failed_rows_remain_in_group_denominators_and_no_new_day_is_invented():
    successful=dict(input_id='ok',source_group='legacy',case='a',source_condition='noise',condition_value=.1,
        metric='rx75',scope='all',threshold=.1,phase='historical',status='complete',
        bank_has_internal_truth=True,bank_has_full_truth=False,train_mse=1.,validation_mse=2.,new_day_mse=None,
        scored_mse=2.,processing_seconds=3.,response=None,**support_metrics(p([[0],[1]],[.1,.1]),[[0],[1],[0,1]],2,1e-9))
    failed={k:v for k,v in successful.items() if k in ('source_group','case','source_condition','condition_value','metric','scope','threshold','phase')}
    failed.update(input_id='bad',status='failed')
    group=next(r for r in summarize([successful,failed]) if r['case']=='ALL')
    assert group['planned']==2 and group['completed']==1 and group['failed']==1
    assert group['new_day_mse_mean'] is None
    assert group['internal_exact']==1 and group['full_exact']==0


def test_device_columns_are_explicit_and_not_full_matrix_error_aliases():
    o=obs();truth=np.c_[np.eye(2),np.eye(2)]
    candidate=p([[0],[1]],[1.,1.],[1.,2.])
    m=response_metrics(candidate,truth,o,o,{'P':[101],'Q':[101]})
    assert m['X_relative_error']==pytest.approx(1/np.sqrt(2))
    assert m['resource_columns']['Q']['relative_error']==0.
    assert m['resource_columns']['Q']['labels']==[101]
    json.dumps(m,allow_nan=False)


def test_hash_mismatch_is_fail_closed(tmp_path):
    worker=Audit(tmp_path);file=tmp_path/'data.json';file.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):
        worker.checked(dict(path=str(file),sha256='0'*64))
