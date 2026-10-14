"""Small checks of post-hoc scope, incumbent retention and box adaptation."""
import copy
from types import SimpleNamespace
import pytest
from . import ev_compact_rejection_optimizer_diagnostic as d


def state(loss,success=True):
    return dict(loss=loss,success=success,status=1,optimality=.001,
                independent_replay=dict(success=True,full_SSE=loss,loglik=-loss/2))


def test_original_remains_incumbent_when_new_is_worse():
    original=state(3.);new=state(4.)
    run=dict(starts=[],returned_best_fit=new,returned_best_fit_replay=new['independent_replay'])
    best=d.retain_best(original,run)
    assert best['label']=='original' and best['SSE']==3.


def test_feasible_unconverged_state_is_upper_bound_with_status_preserved():
    original=state(3.);new=state(2.,False)
    run=dict(starts=[dict(index=0,returned=new)],returned_best_fit=None)
    best=d.retain_best(original,run)
    assert best['SSE']==2. and best['success'] is False
    assert best['global_MLE_certified'] is False


def test_authorized_targets_only_and_no_modified_decision():
    records=[]
    for seed,method in d.EXPECTED:
        records.append(dict(history_id=str(seed),true_name='star',methods={method:dict(false_rejection=True,
            gaps={'star':1.},thresholds={'star':.5}), 'other':dict(false_rejection=False)}))
    risk=dict(individual_policies=dict(evaluation=dict(records=records)))
    assert [(r['seed'],r['method']) for r in d.rejection_targets(risk)]==list(d.EXPECTED)
    extra=copy.deepcopy(risk);extra['individual_policies']['evaluation']['records'][0]['methods']['other']['false_rejection']=True
    extra['individual_policies']['evaluation']['records'][0]['methods']['other'].update(gaps={'star':1.},thresholds={'star':.5})
    with pytest.raises(ValueError,match='four authorized'):d.rejection_targets(extra)


def test_saved_asset_box_and_interface_are_used():
    t=dict(domain='asset',source_interface=1,config=dict(physical=dict(asset_bounds=dict(r_bounds=[.001,.02],x_bounds=[.002,.03]))))
    assert d.saved_fit_options(t)['r_bounds']==[.001,.02]
    t['source_interface']=2
    with pytest.raises(ValueError):d.saved_fit_options(t)
