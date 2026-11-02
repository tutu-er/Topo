import numpy as np
import pytest
from ac_bridge import execute_policy
from test_ac_bridge import _fixture

def run(u,z,a,rule='device_clip'):
    rt,bg,calls=_fixture()
    result=execute_policy(rt,bg,u,z,resource_labels=[101,102,103],capacity_kw=a,
                          multiplier_mean=1.,cv=.1,delivery_rule=rule)
    return result,calls

def test_saturation_both_bounds_and_zero_capacity_keeps_price_and_fee():
    a=np.array([.5,1.,0.]);u=np.zeros((3,96));u[0]=.5;u[1]=-1.
    z=np.full_like(u,10.)
    raw,_=run(u,z,a,'unbounded');sat,calls=run(u,z,a)
    np.testing.assert_array_equal(sat['private']['delivered_kw'],np.clip(raw['private']['delivered_kw'],-a[:,None],a[:,None]))
    np.testing.assert_array_equal(sat['private']['prices'],raw['private']['prices'])
    np.testing.assert_array_equal(sat['private']['mean_u'],u)
    assert sat['summary']['planned_cost']==raw['summary']['planned_cost']
    assert sat['summary']['resource_violation_cells']==0
    assert sat['summary']['saturated_cells']==192
    assert sat['summary']['removed_abs_modulation_kwh']==36.
    assert not sat['summary']['no_clipping'] and sat['summary']['no_redraw']
    assert len(calls['ac'])==len(calls['meter'])==1
    pp,qq,_=calls['ac'][0]
    assert pp.iloc[0,0]==pytest.approx(.01+.5/200)
    assert qq.iloc[0,1]==pytest.approx(.0035-.35/200)

def test_unsaturated_delivery_is_exactly_unchanged():
    a=np.array([.5,1.,0.]);u=np.zeros((3,96));u[0]=.1;u[1]=-.2
    raw,_=run(u,np.zeros_like(u),a,'unbounded');sat,_=run(u,np.zeros_like(u),a)
    for key in ['physical_P_pu','physical_Q_pu','delivered_kw','prices']:
        np.testing.assert_array_equal(raw['private'][key],sat['private'][key])
    assert sat['summary']['saturated_cells']==0

def test_small_positive_capacity_is_not_dropped():
    a=np.array([1e-12,.5,0.]);u=np.zeros((3,96));u[0]=1e-12
    result,_=run(u,np.ones_like(u)*10,a)
    assert np.all(result['private']['delivered_kw'][0]==1e-12)
    assert np.all(result['private']['prices'][0]==2)

def test_saturation_projection_has_no_resource_or_energy_increase():
    rng=np.random.default_rng(55);a=np.array([.5,.75,.75]);u=rng.uniform(-1,1,(3,96))*a[:,None]
    result,_=run(u,rng.normal(size=u.shape)*20,a)
    x=result['private']['delivered_kw'];raw=result['private']['proposed_delivered_kw']
    assert np.all(np.abs(x)<=a[:,None])
    assert np.max(np.sum(np.abs(x),axis=0))<=2.
    assert np.all(np.abs(x)<=np.abs(raw))

def test_unknown_rule_rejected():
    with pytest.raises(ValueError,match='delivery rule'):run(np.zeros((3,96)),np.zeros((3,96)),np.zeros(3),'redraw')
