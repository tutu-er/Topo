"""Independent oracle algebra tests, with no optimization or AC simulation."""
import numpy as np
import pytest

from . import audit


def test_physical_first_order_response_has_two_and_impedance_base():
    metadata=dict(root_bus=1,base_kv=.4,base_mva=.2)
    branches=[dict(from_bus=1,to_bus=2,r_ohm=.04,x_ohm=.02,status='True'),
              dict(from_bus=2,to_bus=101,r_ohm=.01,x_ohm=.03,status='True'),
              dict(from_bus=2,to_bus=102,r_ohm=.02,x_ohm=.01,status='True')]
    H,truth=audit.physical_matrix(metadata,branches,[101,102])
    np.testing.assert_allclose(H[:,:2],[[.125,.1],[.1,.15]])
    np.testing.assert_allclose(H[:,2:],[[.125,.05],[.05,.075]])
    assert truth=={(0,),(1,),(0,1)}


def test_error_protection_and_distance_increase_do_not_force_true_margin_improvement():
    Hc=np.array([[2.,2.]])
    Hj=np.array([[1.,2.-np.sqrt(2)]])
    Hstar=np.array([[3.,1.5]])
    H=np.array([Hc,Hj])
    zero=audit.direction_metrics(Hstar,H,np.array([[1.,0.]]),1.,0,[1])
    new=audit.direction_metrics(Hstar,H,np.array([[0.,1.]]),1.,0,[1])
    assert audit.direct_constraints(new,zero,'protected',.5)['passed']
    assert new['e']<zero['e'] and new['J'][0]>zero['J'][0]
    assert new['rows'][0]['linear_logodds']<zero['rows'][0]['linear_logodds']


def test_protected_common_gain_includes_error_fraction():
    baseline=dict(e=10.,J=[2.,.5],competitors=[1,2])
    current=dict(e=8.,J=[4.,2.],competitors=[1,2])
    assert audit.achievable_z(current,baseline,'distance_only')['z']==pytest.approx(1.)
    assert audit.achievable_z(current,baseline,'protected')['z']==pytest.approx(.2)
    assert not audit.direct_constraints(current,baseline,'protected',.3)['passed']


def test_zero_initial_error_is_not_reported_as_strictly_reduced():
    baseline=dict(e=0.,J=[1.],competitors=[1])
    current=dict(e=0.,J=[3.],competitors=[1])
    result=audit.achievable_z(current,baseline,'protected')
    assert result['z']==2 and result['zero_baseline_error']
    assert not result['strictly_decreased_error']


def test_all_foreign_parameter_points_are_scored_including_nonrepresentative():
    Hstar=np.array([[1.,0.]])
    H=np.array([[[.9,0.]],[[0.,0.]],[[1.,0.]]])
    x=np.array([[1.,0.]])
    rows=audit.direction_metrics(Hstar,H,x,1.,0,[1,2],np.zeros(3))
    assert len(rows['rows'])==2
    assert rows['rows'][1]['linear_logodds']<0


def test_termwise_box_bound_uses_full_linear_coefficient_and_covers_grid():
    Q=np.array([[[1.,-.3],[-.3,2.]]]);a=np.array([[3.,-4.]])
    lo=np.array([[-2.,.5]]);hi=np.array([[1.,2.]])
    upper=audit.interval_quadratic_upper(Q,a,lo,hi)['upper']
    for u0 in np.linspace(-2,1,11):
        for u1 in np.linspace(.5,2,11):
            u=np.array([u0,u1])
            assert u@Q[0]@u+a[0]@u<=upper
    # In one dimension the separate positive terms attain their maxima together.
    exact=audit.interval_quadratic_upper(np.array([[[2.]]]),np.array([[3.]]),np.array([[0.]]),np.array([[2.]]))
    assert exact['upper']==pytest.approx(14.,abs=1e-7)


def test_maximization_bound_reversal_withdraws_upper_not_lower(tmp_path):
    trace=tmp_path/'trace.jsonl';trace.write_text(
        '{"event":"mip","bound":0.2,"native_runtime_seconds":1}\n'
        '{"event":"native_end","attributes":{"ObjBound":0.8}}\n',encoding='utf8')
    result=audit.maximization_history(None,trace,.5)
    assert result['conflict'] and len(result['callback_conflicts'])==1
    assert not result['terminal_conflict']


def test_forecast_pv_headroom_and_hour_window_are_not_nameplate_availability():
    config=dict(resources=[dict(kind='pv',node_label=101,contract_kw=2.)],action_units=['kW'])
    # Positive net consumption supplies no net-export curtailment proxy.
    lo,hi=audit.resource_box(config,[101],np.ones((96,2)),200.)
    assert not lo.any() and not hi.any()
    lo,hi=audit.resource_box(config,[101],-np.ones((96,2)),200.)
    assert np.count_nonzero(hi)==4 and hi.sum()==8.


def test_current_runner_places_eta_after_two_action_seal():
    result=audit.planning_boundary(audit.Path(audit.__file__).with_name('run.py'))
    assert result['passed'] and result['seal_line']<result['eta_read_line']
