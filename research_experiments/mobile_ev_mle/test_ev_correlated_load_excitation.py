"""Known-P linear examples and shared-square MILP audit; no AC fits."""
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from .ev_profile_chord_milp import FrozenQuadratic,solve_profile_chord_milp


def two_leaf(u,*,hist=6,a=.12,r0=.003,r1=.03,r2=.035,sigma=.0003,free_common=False):
    v=np.r_[np.zeros(hist),u];p1=a+v;p2=np.full(len(v),a)
    y1=(r0+r1)*p1+r0*p2;y2=r0*p1+(r0+r2)*p2
    if free_common:
        X=np.zeros((2*len(v),3));X[:len(v),0]=p1+p2;X[len(v):,0]=p1+p2
        X[:len(v),1]=p1;X[len(v):,2]=p2
        y=np.r_[y1,y2];s=np.linalg.lstsq(X,y,rcond=None)[0];residual=y-X@s
    else:
        s=np.array([p1@y1/(p1@p1),p2@y2/(p2@p2)])
        residual=np.r_[y1-s[0]*p1,y2-s[1]*p2]
    A=p1@p1;B=p2@p2;C=p1@p2
    formula=(r0/sigma)**2*(A+B-C*C*(1/A+1/B))
    return float(residual@residual/sigma**2),formula,s,residual


def three_leaf(u,*,hist=6,a=.12,r0=.005,rc=.003,leaves=(.03,.032,.035),sigma=.0003):
    v=np.r_[np.zeros(hist),u];L=len(v);p=np.column_stack((a+v,np.full(L,a),np.full(L,a)))
    X=np.zeros((3*L,4));sums=p.sum(axis=1)
    for i in range(3):
        X[i*L:(i+1)*L,0]=sums;X[i*L:(i+1)*L,i+1]=p[:,i]
    y=X@np.r_[r0,leaves]
    y[:L]+=rc*(p[:,0]+p[:,1]);y[L:2*L]+=rc*(p[:,0]+p[:,1])
    s=np.linalg.lstsq(X,y,rcond=None)[0];residual=y-X@s
    E=float(v.sum());Q=float(v@v);mean=E/L
    formula=(rc/sigma)**2*.5*(Q-E*E/L)
    slopes=np.array([r0+rc/2,leaves[0]+rc/2,leaves[1]+rc/2*(1+mean/a),
                     leaves[2]-rc/2*(3+mean/a)])
    return float(residual@residual/sigma**2),formula,s,slopes,residual


@pytest.mark.parametrize('u',[[.15]*4,[.3,.3,0.,0.],[.2,.1,.25,.05]])
def test_two_leaf_shared_history_future_projection_formula(u):
    q,formula,s,residual=two_leaf(u)
    assert q==pytest.approx(formula,rel=1e-12)
    assert np.all(s>=1e-6) and np.all(s<=.1)
    assert residual.shape==(20,)


def test_two_leaf_common_trunk_freedom_is_an_exact_failure():
    for u in ([.15]*4,[.3,.3,0.,0.],[.2,.1,.25,.05]):
        q,formula,s,residual=two_leaf(u,free_common=True)
        assert q<1e-20
        assert s==pytest.approx([.003,.03,.035],abs=1e-14)
        assert formula>0  # restricted-star formula is not applicable here


@pytest.mark.parametrize('u',[[.15]*4,[.3,.3,0.,0.],[.2,.1,.25,.05]])
def test_three_leaf_free_common_trunk_and_all_slopes_jointly_refit(u):
    q,formula,s,slopes,residual=three_leaf(u)
    assert q==pytest.approx(formula,rel=1e-12)
    assert s==pytest.approx(slopes,abs=1e-14)
    assert np.all(slopes>=1e-6) and np.all(slopes<=.1)
    assert residual.shape==(30,)


def test_external_leaf_voltage_residual_opposes_inside_leaf_and_zero_ev_is_unresolved():
    q,formula,s,slopes,res=three_leaf([.3,.3,0.,0.])
    assert np.max(abs(res[:10]))<1e-15
    assert res[10:20]==pytest.approx(-res[20:30],abs=1e-15)
    assert three_leaf([0.]*4)[0]<1e-20
    assert three_leaf([0.]*4)[1]==0.


def test_same_energy_high_low_is_selected_by_continuous_square_milp():
    q=FrozenQuadratic('sum_power_square',np.zeros(4),np.eye(4))
    con=LinearConstraint([[1.]*4],.6,.6)
    solved=solve_profile_chord_milp([q],np.zeros(4),np.full(4,.3),con,segments=4,
        time_limit=2.,feasible_incumbents=[np.full(4,.15)])
    u=solved['u']
    assert sorted(u)==pytest.approx([0.,0.,.3,.3],abs=1e-7)
    assert solved['objective_lower']==pytest.approx(.18,abs=1e-8)
    assert solved['objective_upper']>=.18-1e-8
    constant=three_leaf([.15]*4)[0];optimized=three_leaf(u)[0]
    assert constant==pytest.approx(2.7)
    assert optimized==pytest.approx(7.2)
    assert optimized>constant
    # The same actuator injection gives the same electrical response for ESS.
    assert three_leaf(u)[0]==three_leaf(np.array(u))[0]


def test_missing_projected_slope_domain_invalidates_unconstrained_formula_application():
    q,formula,s,slopes,res=three_leaf([.3,.3,0.,0.],leaves=(.03,.032,.001))
    assert slopes[-1]<0.
    assert not (np.all(slopes>=1e-6) and np.all(slopes<=.1))


def test_saved_evidence_keeps_failure_and_all_projected_slopes():
    path=Path(__file__).resolve().parent/'evidence/ev_cross_all/diagnostics/correlated_excitation_toy.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['AC_fit_calls']==0 and data['independent_AC_histories']==0
    assert data['two_leaf_free_common_trunk_failure_preserved']
    assert len(data['curves']['optimized']['three_leaf']['slopes'])==4
    assert data['curves']['optimized']['three_leaf']['q']>data['curves']['constant']['three_leaf']['q']
