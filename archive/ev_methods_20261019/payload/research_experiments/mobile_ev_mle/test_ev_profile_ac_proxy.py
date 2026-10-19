"""Independent algebra and actual-AC checks of local pair profiles."""
from types import SimpleNamespace
import numpy as np
import pytest
from .ev_profile_ac_proxy import (profile_affine, evaluate_model, bounded_audit,
    internal_line_bounds, build_pair_models)
from .candidates import RadialCandidate
from .simulation import generate, observe, model
from .symmetric_ac_local import mean_and_jacobian, whiten_observations


def test_projection_matches_direct_least_squares_and_keeps_constant():
    rng=np.random.default_rng(12)
    C=rng.normal(size=(12,4))*np.array([1e-7,1.,1e4,3.])[None,:]
    B=rng.normal(size=(12,3)); c=rng.normal(size=12)
    p=profile_affine(c,B,C)
    for u in (np.zeros(3),np.ones(3),rng.normal(size=3)):
        rhs=c+B@u; scaled=C*p['scale']
        z=np.linalg.lstsq(scaled,-rhs,rcond=1e-10)[0]
        expected=np.linalg.norm(rhs+scaled@z)**2
        assert np.linalg.norm(p['d']+p['T']@u)**2==pytest.approx(expected,rel=1e-10,abs=1e-11)
    assert len(p['d'])<=4
    assert p['qr_gram_error']<1e-10


def test_rank_deficiency_preserves_unexplained_constant():
    p=profile_affine(np.array([1.,3.,4.]),np.zeros((3,2)),np.array([[1.,2.],[0.,0.],[0.,0.]]))
    assert p['rank']==1
    assert np.dot(p['d'],p['d'])==pytest.approx(25.)
    assert np.linalg.norm(p['T'])<1e-12


def test_no_compensation_columns_is_exact_fixed_model():
    c=np.array([1.,2.,4.]); B=np.array([[2.],[1.],[0.]])
    p=profile_affine(c,B,np.empty((3,0)))
    for u in (-2.,0.,3.):
        assert np.linalg.norm(p['d']+p['T'][:,0]*u)**2==pytest.approx(np.linalg.norm(c+B[:,0]*u)**2)


def test_bounded_primal_upper_and_tangent_lower():
    c=np.array([1.,2.]); B=np.array([[1.],[0.]]); C=np.array([[1.],[1.]])
    p=profile_affine(c,B,C)
    full=dict(c=c,B=B,C=C,parameter_center=np.array([2.]),
        lower=np.array([-.2]),upper=np.array([.3]),scale=p['scale'],
        free_z0=p['z0'],free_Zu=p['Zu'])
    a=bounded_audit(dict(d=p['d'],T=p['T'],offset=0.,full=full),np.array([0.]))
    assert a['success'] and a['feasible']
    assert a['free_value']==pytest.approx(.5)
    assert a['primal_upper']==pytest.approx(.8**2+1.8**2)
    assert a['tangent_lower_numeric']<=a['primal_upper']+1e-12
    assert a['primal_upper']-a['tangent_lower_numeric']<1e-9
    assert a['free_bound_violation']>1. and not a['free_feasible']


def test_internal_floor_preserves_root_and_leaves():
    star=RadialCandidate.from_supports('star',3,())
    cherry=RadialCandidate.from_supports('cherry',3,((0,1),))
    sb=internal_line_bounds(star,.004); cb=internal_line_bounds(cherry,.004)
    assert np.all(sb['r_bounds'][0]==1e-6)
    assert np.flatnonzero(cb['internal']).tolist()==[1]
    assert cb['r_bounds'][0][1]==.004
    assert cb['x_bounds'][0][1]==pytest.approx(.004*2/3)
    assert np.all(cb['r_bounds'][0][~cb['internal']]==1e-6)


@pytest.fixture(scope='module')
def ac_inputs():
    path=generate(9123,history_count=4,future_count=4,truth_family=((0,1),))
    history,_=observe(path,slice(0,4)); meter=model(3)
    candidates=[RadialCandidate.from_supports('clade01',3,((0,1),)),
                RadialCandidate.from_supports('clade02',3,((0,2),))]
    fits={}; observed=whiten_observations(history,meter)
    for candidate in candidates:
        r=np.array([.005,.003,.028,.031,.034]); x=np.array([.0035,.002,.011,.013,.015])
        physical=mean_and_jacobian(candidate,r,x,path.p[:4],path.q[:4],
            path.source_p[:4],path.source_q[:4],history.root_v,meter)
        loss=float(np.sum((physical['mean']-observed)**2))
        fits[candidate.name]=SimpleNamespace(r=r,x=x,p=path.p[:4],q=path.q[:4],
            source_p=path.source_p[:4],source_q=path.source_q[:4],loss=loss,success=True)
    return history,candidates,fits,meter


def test_complete_AC_self_constant_and_positive_competitor(ac_inputs):
    h,cs,fs,m=ac_inputs
    built=build_pair_models(h,cs,fs,m,m,floor_r=.001)
    assert built['success'],built['diagnostics']
    assert built['diagnostics']['ac_calls']==4
    assert len(built['models'])==len(built['fixed_models'])==2
    for row in built['models']:
        assert row['full']['line_columns']==10
        assert row['full']['internal_edge_indices']==[1]
        assert row['full']['lower'][1]==pytest.approx(-.002)
        assert row['self_profile']['constant_verified']
        assert row['self_profile']['control_norm']<1e-8
        for u in (np.zeros(4),np.full(4,3.),np.array([0.,6.,6.,0.])):
            assert evaluate_model(row['self_model'],u)['q']==pytest.approx(row['offset'],rel=1e-8,abs=1e-8)
            local=bounded_audit(row,u)
            assert local['feasible']
            assert local['free_value']<=local['primal_upper']+1e-7
            assert evaluate_model(row,u)['q']==pytest.approx(local['free_value'],rel=1e-8,abs=1e-7)


def test_forecast_EV_once_and_common_PQ(ac_inputs):
    h,cs,fs,m=ac_inputs; built=build_pair_models(h,cs,fs,m,m,site=1)
    assert built['success']
    row=built['models'][0]; f=row['forecasts']
    expected=f['common_background']['p'].copy(); expected[:,1]+=.15
    assert f['a']['p']==pytest.approx(expected)
    assert f['b']['p']==pytest.approx(expected)
    assert f['a']['q']==pytest.approx(f['b']['q'])
    assert f['a']['source_p']==pytest.approx(np.full(4,fs[row['a']].source_p.mean()))
    assert built['diagnostics']['source_node']==1


def test_reference_intercept_and_Jacobian_signs(ac_inputs):
    h,cs,fs,m=ac_inputs; built=build_pair_models(h,cs,fs,m,m)
    for row in built['models']:
        full=row['full']; hrows=len(whiten_observations(h,m)); nline=full['line_columns']
        assert full['c']+full['B']@row['reference_kw']==pytest.approx(full['reference_residual'],abs=1e-12)
        assert full['C'][:hrows,:nline]==pytest.approx(row['geometry_b']['history']['line_jacobian'])
        assert full['C'][hrows:,:nline]==pytest.approx(-row['geometry_b']['future']['line_jacobian'])
        assert np.all(full['B'][:hrows]==0.)


def test_failed_required_fit_is_not_silently_deleted(ac_inputs):
    h,cs,fs,m=ac_inputs
    failed={name:SimpleNamespace(**vars(fit)) for name,fit in fs.items()}
    failed[cs[1].name].success=False
    result=build_pair_models(h,cs,failed,m,m)
    assert not result['success'] and result['models']==[] and result['fixed_models']==[]
    assert cs[1].name in result['diagnostics']['failure']


def test_floor_excluding_center_is_explicit_failure(ac_inputs):
    h,cs,fs,m=ac_inputs; result=build_pair_models(h,cs,fs,m,m,floor_r=.004)
    assert not result['success'] and 'violates' in result['diagnostics']['failure']


def test_invalid_controls_and_nonfinite_projection():
    with pytest.raises(ValueError): profile_affine(np.array([np.nan]),np.ones((1,1)),np.ones((1,1)))
    with pytest.raises(ValueError): internal_line_bounds(RadialCandidate.from_supports('s',3),.1)
    with pytest.raises(ValueError): evaluate_model(dict(d=np.zeros(2),T=np.zeros((2,1)),offset=0.),np.ones(2))
