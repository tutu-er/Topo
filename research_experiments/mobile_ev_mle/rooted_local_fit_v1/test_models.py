from copy import deepcopy
import numpy as np
import pytest
from scipy.optimize import minimize,LinearConstraint,Bounds
from .model import prepare_case,line_map,project,public_domain,evaluate,linear_support
from .response import prepare,fit_qp,fit_profile,score_fixed,solve_quadratic
from .ac_fit import fit
from ..local_response_score.scoring import _profile_h


def coordinates(candidate,payload,r,x):
    transform,active=line_map(candidate,payload);d=payload['domain'];s=np.asarray(r)[active]/d['r_cap']
    b=(np.asarray(x)[active]/d['r_cap']-d['ratio_min']*s)/(d['ratio_max']-d['ratio_min'])
    return np.r_[s-b,b]


@pytest.mark.parametrize('context',['FULL','CUT','OLD-RATIO','FOCUS-V'])
def test_observed_boundary_and_future_isolation(context):
    p,h,t=prepare_case('paper15',35991,12,context)
    assert len(p['observed']['p'])==12 and len(h['observed']['p'])==96
    assert not set(p)&{'r','x','truth','physical_network','true_root','v_true'}
    assert (p['root_meter'] is None)==(context=='CUT')
    assert p['domain']['impedance_lower_bound']==0
    assert p['candidates'][0].n==(3 if context=='CUT' else 15)
    if context!='CUT':assert p['root_meter'][0].shape==(12,)


def test_ratio_domain_contains_new_service_and_old_is_nested():
    p,_,t=prepare_case('paper15',35991,4);c=p['candidates'][0]
    theta=coordinates(c,p,t['r'],t['x']);np.testing.assert_allclose(project(theta),theta,atol=1e-14)
    old=deepcopy(p);old['domain']=public_domain(True)
    wrong=coordinates(c,old,t['r'],t['x']);assert np.max(abs(project(wrong)-wrong))>1e-3
    rng=np.random.default_rng(99)
    for _ in range(20):
        oldtheta=project(rng.normal(.4,.6,len(theta)));r,x=np.split(line_map(c,old)[0]@oldtheta,2)
        newtheta=coordinates(c,p,r,x);np.testing.assert_allclose(newtheta,project(newtheta),atol=1e-14)


def test_simplex_projection_and_linear_support_are_exact():
    rng=np.random.default_rng(90)
    for _ in range(8):
        z=rng.normal(size=6);p=project(z)
        result=minimize(lambda a:(.5*np.sum((a-z)**2),a-z),np.ones(6)/4,jac=True,
            method='SLSQP',bounds=Bounds(0,1),constraints=LinearConstraint(np.c_[np.eye(3),np.eye(3)],-np.inf,1),options={'ftol':1e-13})
        np.testing.assert_allclose(p,result.x,atol=1e-6)
        assert linear_support(z)<=z@p+1e-12


@pytest.mark.parametrize('context',['FULL','CUT','FOCUS-V'])
def test_ac_shared_and_local_derivatives(context):
    p,_,t=prepare_case('paper11',35991,3,context);c=p['candidates'][0]
    theta=coordinates(c,p,t['r'],t['x']);local=np.zeros((3,2*c.n+1));local[:,-1]=(t['root_v'][:3]-1)/.01
    rng=np.random.default_rng(3);dt=rng.normal(size=len(theta));dl=rng.normal(size=local.shape)
    # Physical lines remain nonnegative for this derivative test.
    dt*=.005;dl*=.05;eps=1e-5
    base=evaluate(c,p,theta,local)
    plus=evaluate(c,p,theta+eps*dt,local+eps*dl,derivatives=False)['residual']
    minus=evaluate(c,p,theta-eps*dt,local-eps*dl,derivatives=False)['residual']
    analytic=np.einsum('tri,i->tr',base['shared'],dt)+np.einsum('tri,ti->tr',base['state'],dl)
    np.testing.assert_allclose((plus-minus)/(2*eps),analytic,rtol=3e-5,atol=2e-6)


def test_root_aware_profile_matches_explicit_raw_squared_voltage_fit():
    p,_,_=prepare_case('paper11',35991,3);c=p['candidates'][0];d=prepare(c,p)
    theta=np.full(d['basis'].shape[2],.01);h=np.einsum('aje,e->aj',d['basis'],theta)
    closed=_profile_h(d,h);b,bs=p['root_meter'];sig=p['sigma'];total=0
    for i in range(3):
        sy=np.r_[4*bs**2+2*bs**4,4*sig['v']**2+2*sig['v']**4]
        # Unknowns are latent P/Q plus common squared root; raw voltage rows
        # retain independent noise before eliminating the root measurement.
        design=np.block([[np.eye(2*c.n),np.zeros((2*c.n,1))],
                         [np.zeros((1,2*c.n)),np.ones((1,1))],[-h,np.ones((c.n,1))]])
        obs=np.r_[d['x'][i],b[i]**2-bs**2,p['observed']['v'][i]**2-sig['v']**2]
        scale=np.r_[d['sigma_x'][i],np.sqrt(sy)]
        fitted=np.linalg.lstsq(design/scale[:,None],obs/scale,rcond=None)[0]
        total+=np.sum(((design@fitted-obs)/scale)**2)
    assert abs(total-closed['loss'])<1e-7
    assert np.all(d['cov_y'][:,0,1]>0)


@pytest.mark.parametrize('context',['FULL','CUT'])
def test_profile_gradient_and_qp_certificate(context):
    p,_,_=prepare_case('paper11',35991,6,context);c=p['candidates'][0];d=prepare(c,p)
    z=np.full(d['basis'].shape[2],.01);v=_profile_h(d,np.einsum('aje,e->aj',d['basis'],z))
    g=np.einsum('aj,aje->e',v['gradient_H'],d['basis']);direction=np.random.default_rng(1).normal(size=len(z));eps=1e-7
    f=lambda x:_profile_h(d,np.einsum('aje,e->aj',d['basis'],x),derivatives=False)['loss']
    assert abs((f(z+eps*direction)-f(z-eps*direction))/(2*eps)-g@direction)<1e-3
    q=fit_qp(c,p);assert q['qualified'];assert q['lower_bound']<=q['loss']+1e-8
    assert q['absolute_gap']<=max(1e-5,1e-7*q['loss'])
    np.testing.assert_allclose(project(q['theta']),q['theta'],atol=1e-13)


def test_noiseless_ac_fixed_lines_zero_and_descent_with_noisy_data():
    p,h,t=prepare_case('paper11',35991,12);c=p['candidates'][0]
    theta=coordinates(c,p,t['r'],t['x']);clean=deepcopy(p)
    clean['observed']={k:t[k][:12].copy() for k in ('p','q','v')};clean['root_meter']=(t['root_v'][:12].copy(),.0005)
    fit0=fit(c,clean,theta,fixed_lines=(t['r'],t['x']));assert fit0['loss']<1e-12
    q=fit_qp(c,p);a=fit(c,p,q['theta'],multistart=1,max_iter=80)
    assert a['physical'];assert a['qualified']
    assert all(np.diff(a['trace'])<0)
    held=fit(c,h,a['theta'],fixed_lines=(a['r'],a['x']))
    assert not held['lines_updated'];np.testing.assert_array_equal(held['r'],a['r'])


def test_zero_pair_embeds_base_without_inventing_a_minimum_impedance():
    p,_,t=prepare_case('paper15',35991,4);base,pair=p['candidates'][:2]
    r=np.zeros(pair.m);x=np.zeros(pair.m)
    for j,s in enumerate(pair.supports):
        if s in base.supports:
            i=base.supports.index(s);r[j],x[j]=t['r'][i],t['x'][i]
    from .. import likelihood as lk
    v=lk.ac_predict(pair,r,x,t['p'][:4],t['q'][:4],root_v=t['root_v'][:4])['v']
    np.testing.assert_allclose(v,t['v'][:4],atol=1e-13)
