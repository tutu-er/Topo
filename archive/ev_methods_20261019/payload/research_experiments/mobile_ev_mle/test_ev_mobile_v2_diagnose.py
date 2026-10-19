"""The read-only audit retains signs, factors and excludes formal outcomes."""
import gzip
import json
import pytest

from .ev_mobile_v2_diagnose import _compensation,_physical_matches,extract,screen_reference_floors


def test_compensation_identity_uses_half_SSE_and_same_pair():
    def fit(fixed,joint):return dict(loss=joint,feasible_witness=dict(loss=fixed))
    profile=dict(success=True,references=[
        dict(a='a',fits={'a':fit(10,8),'b':fit(20,12)}),
        dict(a='b',fits={'a':fit(12,10),'b':fit(5,4)})])
    result=_compensation(profile)
    assert result['fixed_training_theta_mean_nearest_gap']==4.25
    assert result['shared_line_mean_nearest_gap']==2.5
    assert result['mean_reference_loss_reduction']==1.5
    assert result['mean_competitor_loss_reduction']==5
    assert result['maximum_pair_identity_error']==0


def test_physical_match_is_explicit_reporting_tolerance():
    methods=dict(a=dict(power_kw=[[1,0]]),b=dict(power_kw=[[1.00001,0]]))
    assert _physical_matches([[1+1e-9,0]],methods)==['a']
    assert _physical_matches(None,methods)==[]


@pytest.mark.parametrize('phase',['calibration','evaluation'])
def test_development_audit_rejects_formal_records(tmp_path,phase):
    path=tmp_path/'case_000.json.gz'
    path.write_bytes(gzip.compress(json.dumps(dict(phase=phase)).encode()))
    with pytest.raises(ValueError,match='only declared pilot/development'):
        extract(path)


def test_source_only_initial_profiles_need_no_joint_score_field(tmp_path):
    path=tmp_path/'case_000.json.gz'
    row=dict(phase='development',physical_bank='toy',case_id='toy',index=0,seed=0,
        config=dict(training_points=8),true_name='a',methods=[dict(name='cap_fixed',power_kw=[[0.]],
            inference=dict(joint_total=dict(M=1.,complete=True)),operation=dict(success=True),planning_seconds=0.)],
        selection=dict(methods=dict(cap_fixed=dict(success=True,raw=dict(selected_score=1.,cap=2.,
            initial_candidates=[dict(origin='parent',profile=dict(success=True,power_kw=[[0.]]))])))))
    path.write_bytes(gzip.compress(json.dumps(row).encode()))
    result=extract(path)
    assert result['methods']['cap_fixed']['M']==1
    assert result['methods']['cap_fixed']['best_B_initial_matches']==[]


def test_capped_floor_screen_uses_predictions_and_fixed_cap_not_outcome():
    def record(actual):
        return dict(phase='development',seed=1,case_id='toy',bank='toy',true_name='a',
            methods=dict(jointgrad_fixed=dict(B_prediction=dict(reference_gaps={'a':1.,'b':20.})),
                jointgrad_mobile=dict(B_prediction=dict(reference_gaps={'a':1.,'b':6.}),active_remote=True),
                cap_fixed=dict(cap=5.411)),mobile_fixed=dict(jointgrad=dict(actual_M_gain=actual,prediction_order_reversal=False)))
    rows=[record(3.)]
    assert screen_reference_floors(rows,capped=False)['blocked']['records']==1
    assert screen_reference_floors(rows,capped=True)['passed']['records']==1
    assert screen_reference_floors([record(-3.)],capped=True)['passed']['records']==1
