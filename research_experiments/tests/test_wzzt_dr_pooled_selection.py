"""Common-rule model selection without an automatic RNJ preference."""
import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.baseline_alignment import pooled_selection as pooled


def point(mean, se, size):
    return {'support_indices':[[i] for i in range(size)],
            'validation_mae':mean, 'validation_se':se}


def test_one_common_threshold_uses_best_candidate_se_not_each_candidates_se():
    path=[point(1.5,1.0,2),point(1.1,.3,3)]
    rnj=point(1.0,.05,4)
    selected=pooled.select_with_complete_rnj(path,rnj)
    assert selected['threshold_pu2']==pytest.approx(1.05)
    assert selected['eligible_candidate_indices']==[2]
    assert selected['selected_origin']=='complete_rnj'
    assert selected['original_wzzt_selected_path_index']==1


def test_simpler_wzzt_point_can_win_even_when_rnj_has_lower_mean():
    selected=pooled.select_with_complete_rnj([point(1.04,.01,2)],point(1.,.05,3))
    assert selected['selected_origin']=='wzzt_path'
    assert selected['eligible_candidate_indices']==[0,1]


def test_tie_keeps_original_path_order_and_counts_stored_zero_supports():
    original=point(1.,.1,3)
    original['r_values']=[.2,0,0]
    rnj=point(1.,.1,3)
    rnj['r_values']=[.1,.1,.1]
    selected=pooled.select_with_complete_rnj([original,point(1.,.1,3)],rnj)
    assert selected['selected_candidate_index']==0
    assert selected['candidates'][0]['stored_support_count']==3


def test_wzzt_path_order_and_outcome_remain_available_when_rnj_wins():
    path=[point(3.,.1,2),point(2.,.1,3)]
    before=repr(path)
    selected=pooled.select_with_complete_rnj(path,point(1.,.1,3))
    assert repr(path)==before
    assert selected['original_wzzt_selected_path_index']==1
    assert selected['selected_origin']=='complete_rnj'


def test_missing_validation_keeps_original_last_point_without_claiming_comparison():
    path=[point(None,None,2),point(None,None,3)]
    selected=pooled.select_with_complete_rnj(path,point(1.,.1,4))
    assert selected['status']=='unavailable'
    assert selected['selected_path_index']==1
    assert not selected['rnj_candidate_included']


def test_failed_or_invalid_candidate_cannot_replace_wzzt():
    selected=pooled.select_with_complete_rnj([point(1.,.1,2)],point(float('nan'),.1,3))
    assert selected['selected_origin']=='wzzt_path'
    assert selected['status']=='rnj_candidate_unavailable'
    assert selected['rnj_failure']['type']=='ValueError'
    failed=pooled.select_with_complete_rnj([point(1.,.1,2)],rnj_failure={'message':'LP time limit'})
    assert failed['rnj_failure']['message']=='LP time limit'


def test_unproved_rnj_fit_is_excluded():
    rnj=point(.5,.01,3)
    rnj['solver']=dict(status=1,success=False,message='time limit',objective=.5,
        dual_bound=None,mip_gap=None,node_count=None,runtime_seconds=1.,
        variable_count=3,binary_variable_count=0,constraint_count=10)
    selected=pooled.select_with_complete_rnj([point(1.,.1,2)],rnj)
    assert selected['selected_origin']=='wzzt_path'
    assert not selected['rnj_candidate_included']


def test_validation_failure_in_original_path_is_not_silently_filtered():
    with pytest.raises(ValueError,match='finite'):
        pooled.select_with_complete_rnj([point(float('nan'),.1,2)],point(1.,.1,2))
    with pytest.raises(ValueError,match='nonempty'):
        pooled.select_with_complete_rnj([],point(1.,.1,2))


def test_live_rnj_fit_uses_training_only_and_does_not_refit_validation():
    rng=np.random.default_rng(240)
    labels=[101,102,103]
    r=.1*np.eye(3); r[:2,:2]+=.03
    x=.05*np.eye(3); x[:2,:2]+=.01
    def scenario(name,offset):
        p=rng.uniform(.01,.06,(24,3));q=rng.uniform(.005,.025,(24,3))
        y=p@r.T+q@x.T+offset
        return {'name':name,'P_terminal':pd.DataFrame(p,columns=labels),
                'Q_terminal':pd.DataFrame(q,columns=labels),
                'drop_target':pd.DataFrame(y,columns=labels)}
    training=[scenario('train',0.)];validation=[scenario('validation',.01)]
    fitted=pooled.fit_complete_rnj_candidate(training,validation,[(101,102)],
        r_upper_bound=2.,x_upper_bound=2.,time_limit=10.)
    assert fitted.train_mae<1e-9
    assert fitted.validation_mae==pytest.approx(.01,abs=1e-8)
    assert len(fitted.support_indices)==4
    assert fitted.solver.binary_variable_count==0
    assert set(fitted.support_labels)=={(101,),(102,),(103,),(101,102)}
