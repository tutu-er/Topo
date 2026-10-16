"""Mock-only electrical relocation audit; no AC solver or real formal data."""
from copy import deepcopy
import numpy as np
import pytest
from . import ev_mobile_v2_spatial_audit as audit


def observation(power):
    u=np.asarray(power,float);T,n=u.shape
    return dict(p=(u+.2).tolist(),q=np.full_like(u,.05).tolist(),v=(1-.01*u).tolist(),
        root_v=np.ones(T).tolist(),master_p=u.sum(axis=1).tolist(),master_q=np.zeros(T).tolist(),
        source_p=None,source_q=None)


def fixture(moved=True):
    u=np.array([[0,1.,0],[0,-.5,0]]) if moved else np.array([[1.,0,0],[-.5,0,0]])
    rows=[]
    for family in audit.FAMILIES:
        for side in ('fixed','mobile'):
            power=u if side=='mobile' else np.array([[.2,0,0],[.3,0,0]])
            rows.append(dict(name=family+'_'+side,power_kw=power.tolist(),action_key='mobile' if side=='mobile' else 'fixed',
                operation=dict(success=True,audit=dict(moves=int(moved and side=='mobile'),
                    driver_cost=30. if moved and side=='mobile' else 0.,travel_energy_kwh=.2 if moved and side=='mobile' else 0.)),
                inference=dict(joint_total=dict(complete=True,M=10. if side=='mobile' else 3.))))
    return dict(phase='evaluation',physical_bank='n3',index=0,seed=1,case_id='mock',true_name='a',
        methods=rows,future_records=dict(mobile=dict(observations=observation(u))))


class MockBackend:
    def __init__(self,*,mismatch=False,fit_complete=True):
        self.generated=0;self.observed=0;self.restored=0;self.fitted=0
        self.mismatch=mismatch;self.fit_complete=fit_complete
    def regenerate(self,record):self.generated+=1;return dict(mock_case=True)
    def observe(self,case,power):
        self.observed+=1;value=observation(power)
        if self.mismatch:value['v'][0][0]+=.1
        return value
    def restore(self,record,case):self.restored+=1;return dict(restored=True)
    def fit(self,context,future):
        self.fitted+=1
        return dict(complete=self.fit_complete,losses={'a':1.,'b':8.},
            comparisons=dict(a=dict(saved_fit_state='mock')),failures={} if self.fit_complete else {'b':'mock optimization failure'},
            counters=dict(fit_calls=4,solver_calls=8,ac_calls=20),seconds=.1)


def test_relocation_keeps_the_same_realized_net_curve_and_signs():
    u=np.array([[0,2,-.5],[0,0,-1.]])
    zero=audit.relocate_net_power(u)
    np.testing.assert_array_equal(zero[:,0],u.sum(axis=1))
    assert np.count_nonzero(zero[:,1:])==0


def test_all_original_channels_are_checked_before_counterfactual_fits():
    backend=MockBackend(mismatch=True);result=audit.audit_record(fixture(),backend)
    assert not result['success'] and len(result['methods'])==4
    assert all(not row['success'] for row in result['methods'])
    assert backend.fitted==backend.restored==0
    assert all(row['moved'] for row in result['methods'])


def test_regeneration_failure_preserves_known_motion_cost_and_all_slots():
    class FailedGenerator(MockBackend):
        def regenerate(self,record):raise ValueError('source version changed')
    backend=FailedGenerator();result=audit.audit_record(fixture(),backend)
    assert not result['success'] and len(result['methods'])==4
    assert all(row['moved'] and row['worker_minutes']==30 and row['driving_energy_kwh']==.2 for row in result['methods'])
    assert backend.observed==backend.restored==backend.fitted==0


def test_duplicate_power_still_checks_each_saved_observation_record():
    record=fixture();original=record['future_records']['mobile']['observations']
    changed=deepcopy(original);changed['v'][0][0]+=.1
    record['future_records']['bad_copy']=dict(observations=changed)
    next(m for m in record['methods'] if m['name']=='jointgrad_mobile')['action_key']='bad_copy'
    backend=MockBackend();result=audit.audit_record(record,backend)
    assert len(result['methods'])==4 and sum(row['success'] for row in result['methods'])==3
    assert not next(r for r in result['methods'] if r['family']=='jointgrad')['success']
    assert backend.observed==2  # one shared replay, one counterfactual


def test_four_methods_with_same_net_curve_share_one_counterfactual_fit_batch():
    backend=MockBackend();result=audit.audit_record(fixture(),backend)
    assert result['success'] and result['counterfactual_unique_fit_batches']==1
    assert backend.fitted==backend.restored==1
    assert backend.observed==2  # one original replay and one same-net counterfactual
    assert sum(row['counterfactual_fit_reused'] for row in result['methods'])==3
    for row in result['methods']:
        assert row['M_space']==3. and row['M_curve_extra']==4. and row['total_M_gain']==7.
        assert row['identity_error']==0 and row['worker_minutes']==30 and row['driving_energy_kwh']==.2
        assert row['counterfactual_terminal_energy_increment_kwh']==.2
    assert not result['counterfactual_is_executable_policy'] and not result['statistical_decision_applied']


def test_unmoved_methods_have_exact_zero_space_and_no_new_refit():
    backend=MockBackend();result=audit.audit_record(fixture(False),backend)
    assert result['success'] and backend.fitted==backend.restored==0
    assert all(row['M_space']==0. and row['M_site0_same_net']==row['M_mobile'] for row in result['methods'])
    assert result['counterfactual_unique_fit_batches']==0


def test_failed_counterfactual_retains_all_slots_fitstates_and_actual_motion_cost():
    backend=MockBackend(fit_complete=False);result=audit.audit_record(fixture(),backend)
    assert not result['success'] and len(result['methods'])==4 and backend.fitted==1
    assert all(not row['success'] and row['moved'] for row in result['methods'])
    saved=next(iter(result['counterfactuals'].values()))
    assert saved['failures']=={'b':'mock optimization failure'} and saved['comparisons']


def test_original_inference_failure_does_not_remove_its_slot():
    record=fixture();record['methods'][0]['inference']['joint_total']['complete']=False
    result=audit.audit_record(record,MockBackend())
    assert len(result['methods'])==4 and sum(row['success'] for row in result['methods'])==3


def test_missing_fixed_method_only_fails_its_paired_slot():
    record=fixture();record['methods']=[m for m in record['methods'] if m['name']!='square_fixed']
    result=audit.audit_record(record,MockBackend())
    assert len(result['methods'])==4 and sum(row['success'] for row in result['methods'])==3


def test_nonfinite_original_margin_is_retained_as_failure():
    record=fixture();record['methods'][0]['inference']['joint_total']['M']=float('nan')
    result=audit.audit_record(record,MockBackend())
    assert len(result['methods'])==4 and sum(row['success'] for row in result['methods'])==3
    assert 'finite' in next(r for r in result['methods'] if r['family']=='square')['failure']


def test_nonselected_index_and_nonevaluation_are_rejected_without_generator_call():
    for key,value in [('index',1),('phase','calibration')]:
        record=fixture();record[key]=value;backend=MockBackend()
        with pytest.raises(ValueError,match='prespecified'):audit.audit_record(record,backend)
        assert backend.generated==0


def test_observation_verifier_checks_source_channels_when_present():
    a=observation([[1.,0.]]);b=deepcopy(a);a['source_p']=[.2];b['source_p']=[.3]
    result=audit.verify_observations(a,b)
    assert not result['success'] and not result['channels']['source_p']['success']


def test_dry_run_preserves_eight_histories_and_32_missing_method_slots(tmp_path):
    backend=MockBackend();result=audit.run([tmp_path/'absent3'],[tmp_path/'absent4'],tmp_path/'output',backend=backend)
    assert result['history_denominator']==8 and result['method_slot_denominator']==32
    assert result['complete_histories']==result['complete_slots']==0 and result['failed_slots']==32
    assert backend.generated==backend.fitted==0
    with pytest.raises(ValueError,match='retain prior'):audit.run([],[],tmp_path/'output',backend=backend)


def test_history_cache_restore_uses_saved_states_without_any_AC_call():
    from . import likelihood as lk
    c=lk.RadialCandidate.from_supports('a',2,[]);m=c.m
    H,V=2,1
    def obs(T):return dict(p=np.full((T,2),.1).tolist(),q=np.full((T,2),.03).tolist(),
        v=np.full((T,2),.99).tolist(),root_v=np.ones(T).tolist(),master_p=np.full(T,.2).tolist(),
        master_q=np.full(T,.05).tolist(),source_p=None,source_q=None)
    meter=dict(p_sigma=.01,q_sigma=.01,v_sigma=.001,master_p_sigma=.01,master_q_sigma=.01)
    line=dict(r=np.full(m,.01).tolist(),x=np.full(m,.002).tolist())
    trained=dict(candidate='a',**line,source_p=[.01]*H,source_q=[.005]*H)
    validation=dict(**line,source_p=[.01]*V,source_q=[.005]*V)
    state=dict(**line,source_p=[.01]*(H+V),source_q=[.005]*(H+V))
    record=dict(config=dict(training={},source={},ratio_bounds=[.1,.5]),training=obs(H),validation=obs(V),
        training_meter=meter,validation_meter=meter,future_meter=meter,seed=1,
        training_result=dict(fits=[trained]),history_prefits=dict(a=dict(fixed=dict(validation_fit=validation),
            joint=dict(state=state),seconds=.1)))
    backend=object.__new__(audit.ACBackend);ctx=backend.restore(record,dict(bank=[c]))
    saved=ctx['histories']['a']
    np.testing.assert_array_equal(saved.fixed_state['source_p'],[.01]*3)
    assert saved.training_fit is trained and saved.joint_pre is record['history_prefits']['a']['joint']
    assert ctx['options']['multistart']==2 and ctx['seed']==1
