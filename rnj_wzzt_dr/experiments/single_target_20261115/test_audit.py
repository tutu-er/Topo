"""Independent single-target scope regression without optimization or AC."""
import numpy as np
import pytest
from rnj_wzzt_dr.experiments.single_target_20261115 import audit


def toy():
    return dict(score={'native_index':2,'cells':1,'reference_mse':1.,'validation_tie':0.},
        H=np.array([[[.5,0.]],[[.75,0.]],[[1.,0.]],[[1.5,0.]]]),
        history=np.array([1.,2.,0.,3.]),representatives=[0,2,3],means=np.array([[[.5,0.]],[[1.,0.]],[[1.5,0.]]]),
        targets=[0,1,2],weights=np.array([.3,.4,.3]),families=['wrong','wrong','target','other'],
        covariance=np.array([[.001]]),sigma2=1.,n=1,forecast_relative_sd=.05,internal=np.zeros(4,int))


def test_strict_history_winner_not_first_or_highest_assumed_target():
    old=toy();new=audit.historical_target(old)
    assert new['targets']==[1] and old['targets']==[0,1,2]


def test_same_family_representative_does_not_replace_missing_exact_native():
    old=toy();old['score']['native_index']=1
    old['history'][1]=-1.
    with pytest.raises(ValueError,match='identical representative'):audit.historical_target(old)


@pytest.mark.parametrize('mode',['normal_fixed','mgf_fixed','mgf_forecast'])
def test_all_foreign_points_retained_and_guards_only_point_forward(mode):
    model=audit.historical_target(toy());forecast=np.array([[1.,0.]])
    result=audit.exact_gates(model,forecast,forecast*1.2,mode)
    assert [r['competitor_native'] for r in result['rows']]==[0,1,3]
    assert all(r['target_native']==2 and r['alpha']==pytest.approx(.1/3) for r in result['rows'])
    assert [(g['target'],g['competitor']) for g in result['guards']]==[(1,0),(1,2)]


def test_policy_list_has_no_hidden_oracle_budget_or_rotation():
    assert audit.policy_names({'modes':['normal_fixed','mgf_fixed','mgf_forecast']})==[
        'single_normal_fixed','single_mgf_fixed','single_mgf_forecast']


def test_target_must_match_original_historical_score_choice():
    model=toy();model['score']['native_index']=0
    with pytest.raises(ValueError,match='historical MSE'):audit.historical_target(model)


def test_current_driver_seals_before_original_ac_adapter():
    assert audit.planning_boundary(audit.Path(audit.__file__).with_name('run.py'))['passed']


def test_executed_snapshot_is_verified_when_current_file_changes(tmp_path):
    original=tmp_path/'source.py';snapshot=tmp_path/'saved_source.py'
    original.write_text('old',encoding='utf8');reference=audit.base.ref(original)
    snapshot.write_bytes(original.read_bytes());original.write_text('new',encoding='utf8')
    checker=audit.Audit();checker.source_snapshots[(str(original.resolve()),reference['sha256'])]=snapshot
    assert checker.checked(reference)==snapshot
    snapshot.write_text('tampered',encoding='utf8')
    fresh=audit.Audit();fresh.source_snapshots=checker.source_snapshots
    with pytest.raises(AssertionError,match='hash differs'):fresh.checked(reference)
