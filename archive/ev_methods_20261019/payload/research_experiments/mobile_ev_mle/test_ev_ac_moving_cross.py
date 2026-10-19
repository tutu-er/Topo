"""Coordinate equivalence and independent AC derivative checks."""
from types import SimpleNamespace
import numpy as np
import pytest

from .ev_ac_moving_cross import (affine_box_range, transform_domain, original_bounds,
    physical_parameter, parameter_increment, build_from_saved, residual, matrix_at,
    mixed_control_derivatives, future_geometry, fixed_action_qp, relative_score_bounds,
    channel_slices, control_map, self_saved_model)
from .ev_ac_moving_cross import add_control_curvature
from .ev_profile_ac_proxy import build_pair_models
from .candidates import RadialCandidate
from .simulation import generate, observe, model
from .symmetric_ac_local import mean_and_jacobian, whiten_observations


def test_domain_shift_exact_and_outer_contains_every_action():
    center=np.array([.1,.4]); lo=np.array([-.09,-.5]); hi=np.array([.9,.6])
    U=np.array([[0.,0.],[.05,-.1]]); u0=np.array([3.,2.])
    d=transform_domain(center,lo,hi,U,u0,[0.,0.],[6.,6.]); m=dict(domain=d)
    for u in ([0.,0.],[6.,0.],[0.,6.],[2.,4.]):
        a,b=original_bounds(m,u)
        assert np.all(d['outer_lower']<=a) and np.all(d['outer_upper']>=b)
        assert physical_parameter(m,u,a)==pytest.approx(center+lo)
        assert physical_parameter(m,u,b)==pytest.approx(center+hi)
        assert parameter_increment(m,u,center)==pytest.approx(U@(u0-np.asarray(u)))
    assert np.all(d['safe_lower']>=d['outer_lower'])
    assert np.all(d['safe_upper']<=d['outer_upper'])


def test_exact_domain_is_not_common_outer_box():
    d=transform_domain(np.array([.15]),np.array([-.15]),np.array([.85]),
        np.array([[.05]]),np.array([3.]),np.array([0.]),np.array([6.]))
    m=dict(domain=d)
    z=np.array([d['outer_lower'][0]])
    assert physical_parameter(m,[0.],z)[0]<d['physical_lower'][0]
    assert physical_parameter(m,[6.],z)[0]==pytest.approx(d['physical_lower'][0])


def test_affine_box_range_with_negative_controls():
    lo,hi=affine_box_range(np.array([[1.,-2.],[-3.,4.]]),[-1.,0.],[2.,3.])
    assert lo==pytest.approx([-7.,-6.]); assert hi==pytest.approx([2.,15.])


@pytest.fixture(scope='module')
def ac_model():
    path=generate(92133,history_count=4,future_count=4,truth_family=((0,1),))
    history,_=observe(path,slice(0,4)); meter=model(3)
    candidates=[RadialCandidate.from_supports('clade01',3,((0,1),)),
                RadialCandidate.from_supports('clade02',3,((0,2),))]
    fits={}; obs=whiten_observations(history,meter)
    for c in candidates:
        r=np.array([.005,.003,.028,.031,.034]); x=np.array([.0035,.002,.011,.013,.015])
        g=mean_and_jacobian(c,r,x,path.p[:4],path.q[:4],path.source_p[:4],path.source_q[:4],history.root_v,meter)
        fits[c.name]=SimpleNamespace(r=r,x=x,p=path.p[:4],q=path.q[:4],
            source_p=path.source_p[:4],source_q=path.source_q[:4],loss=float(np.sum((g['mean']-obs)**2)),success=True)
    old=build_pair_models(history,candidates,fits,meter,meter)['models'][0]
    c=next(c for c in candidates if c.name==old['b'])
    moved=build_from_saved(old,c,meter)
    return old,moved,c,meter,candidates


def test_old_linear_model_and_moving_linear_coordinate_are_equal(ac_model):
    old,m,c,meter,_=ac_model
    first=build_from_saved(old,c,meter,include_mixed=False)
    rng=np.random.default_rng(313); u=np.array([0.,6.,2.,4.])
    physical=np.asarray(old['full']['parameter_center'])+rng.normal(size=len(old['full']['lower']))*1e-5
    z=parameter_increment(first,u,physical)
    old_delta=physical-np.asarray(old['full']['parameter_center'])
    old_r=np.asarray(old['full']['c'])+np.asarray(old['full']['B'])@u+np.asarray(old['full']['C'])@old_delta
    assert residual(first,u,z)==pytest.approx(old_r,abs=2e-12)
    assert first['B']==pytest.approx(first['old_B']+first['C_at_reference']@first['domain']['shift'])
    assert np.linalg.norm(first['B']-first['old_B'])>1.


def test_reference_intercept_and_shared_line_columns(ac_model):
    old,m,c,meter,_=ac_model
    assert residual(m,m['reference_kw'],np.zeros(m['A0'].shape[1]))==pytest.approx(old['full']['reference_residual'],abs=1e-12)
    assert matrix_at(m,m['reference_kw'])==pytest.approx(old['full']['C'],abs=1e-12)
    assert np.all(m['At'][:,:m['history_rows']]==0)
    assert np.all(m['At'][:,:,m['line_columns']:m['line_columns']+m['history_state_columns']]==0)


def test_batched_mixed_derivative_matches_individual_time_fd(ac_model):
    old,m,c,meter,_=ac_model; f=old['forecast_b']; theta=np.asarray(old['full']['parameter_center'])[:2*c.m]
    channels=tuple(old['channels']); t=2; eps=.0007
    plus={k:np.asarray(f[k]).copy() for k in ('p','q','source_p','source_q')}
    minus={k:np.asarray(f[k]).copy() for k in plus}
    plus['p'][t,old['site']]+=eps/old['base_kw']; minus['p'][t,old['site']]-=eps/old['base_kw']
    gp=future_geometry(c,theta,plus,f['root_v'],meter,channels)
    gm=future_geometry(c,theta,minus,f['root_v'],meter,channels)
    assert m['mixed_derivatives']['line'][t]==pytest.approx((gp['line_jacobian']-gm['line_jacobian'])/(2*eps),rel=2e-6,abs=2e-6)
    assert m['mixed_derivatives']['state'][t]==pytest.approx((gp['state_jacobian']-gm['state_jacobian'])/(2*eps),rel=2e-5,abs=2e-6)


def test_mixed_line_control_partial_commutes_and_direct_meters_zero(ac_model):
    old,m,c,meter,_=ac_model; f=old['forecast_b']; theta=np.asarray(old['full']['parameter_center'])[:2*c.m]
    U=control_map(c.n,len(f['p']),old['site'],old['base_kw']); eps=1e-6; col=1
    plus=theta.copy(); minus=theta.copy(); plus[col]+=eps; minus[col]-=eps
    gp=future_geometry(c,plus,f,f['root_v'],meter,tuple(old['channels']))
    gm=future_geometry(c,minus,f,f['root_v'],meter,tuple(old['channels']))
    reverse=(gp['state_jacobian']@U-gm['state_jacobian']@U)/(2*eps)
    forward=m['mixed_derivatives']['line'][:,:,col].T
    assert forward==pytest.approx(reverse,rel=2e-5,abs=2e-5)
    for k,s in channel_slices(old['channels'],len(f['p']),c.n).items():
        if k in ('p','q','source_p','source_q'):
            assert np.linalg.norm(m['mixed_derivatives']['line'][:,s])<1e-10
            assert np.linalg.norm(m['mixed_derivatives']['state'][:,s])<1e-10


def test_synthetic_Rp_mixed_term_is_exact():
    # Residual: source u minus competitor (R0+dR)*(u+e), with R0=2.
    m=dict(c=np.array([0.]),B=np.array([[-1.]]),A0=np.array([[0.,-2.]]),
        At=np.array([[[-1.,0.]]]))
    for u,R,e in ((1.,.3,0.),(4.,-.2,0.),(0.,.3,.4)):
        exact=u-(2+R)*(u+e)
        assert residual(m,[u],[R,e])[0]-R*e==pytest.approx(exact)


def test_fixed_action_original_qp_and_outer_relation(ac_model):
    old,m,c,meter,_=ac_model; u=np.array([0.,6.,6.,0.])
    exact=fixed_action_qp(m,u,domain='original'); outer=fixed_action_qp(m,u,domain='outer')
    assert exact['success'] and outer['success']
    assert exact['physical_bound_violation']<1e-9
    assert outer['primal_upper']<=exact['primal_upper']+1e-7
    assert exact['tangent_lower_numeric']<=exact['primal_upper']


def test_reference_profile_requires_cost_difference_bounds(ac_model):
    old,m,c,meter,candidates=ac_model
    self_old=self_saved_model(old); own=next(x for x in candidates if x.name==old['a'])
    self_model=build_from_saved(self_old,own,meter)
    assert np.linalg.norm(self_model['B'])<1e-10
    u=np.array([0.,6.,0.,6.]); a=fixed_action_qp(m,u); b=fixed_action_qp(self_model,u)
    interval=relative_score_bounds(a,b)
    value=.5*(a['primal_upper']-b['primal_upper'])
    assert interval['lower']<=value<=interval['upper']


def test_control_curvature_matches_independent_mean_second_difference(ac_model):
    old,m,c,meter,candidates=ac_model
    own=next(x for x in candidates if x.name==old['a'])
    curved=add_control_curvature(m,old,own,meter)
    time=1; step=.05; hrows=m['history_rows']
    def second(candidate,forecast,theta):
        plus={k:np.asarray(forecast[k]).copy() for k in ('p','q','source_p','source_q')}
        minus={k:np.asarray(forecast[k]).copy() for k in plus}
        plus['p'][time,old['site']]+=step/old['base_kw']; minus['p'][time,old['site']]-=step/old['base_kw']
        gs=[future_geometry(candidate,theta,s,forecast['root_v'],meter,tuple(old['channels']))['mean']
            for s in (plus,forecast,minus)]
        return (gs[0]-2*gs[1]+gs[2])/step**2
    thb=np.asarray(old['full']['parameter_center'])[:2*c.m]
    tha=np.asarray(old['self_model']['full']['parameter_center'])[:2*own.m]
    ha=second(own,old['forecast_a'],tha); hb=second(c,old['forecast_b'],thb)
    assert curved['control_curvature_reference'][hrows:,time]==pytest.approx(ha,rel=1e-4,abs=5e-6)
    assert curved['control_curvature_competitor'][hrows:,time]==pytest.approx(hb,rel=1e-4,abs=5e-6)
    assert curved['control_curvature_difference'][hrows:,time]==pytest.approx(ha-hb,rel=1e-4,abs=5e-6)
    assert curved['exchange_MILP_compatible_without_modification'] is False
    assert residual(curved,m['reference_kw'],np.zeros(m['A0'].shape[1]))==pytest.approx(old['full']['reference_residual'],abs=1e-12)


def test_bad_domain_and_difference_steps_rejected():
    with pytest.raises(ValueError): affine_box_range(np.ones((2,2)),[1.,0.],[0.,1.])
    with pytest.raises(ValueError): transform_domain([0.],[1.],[0.],[[1.]],[0.],[0.],[1.])
