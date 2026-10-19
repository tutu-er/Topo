"""Arithmetic/target checks only; no AC or optimizer calls."""
from itertools import product

import numpy as np
import pytest

from .ev_mobile_v3_score_attribution import FACTORS, factor_parameters, make_target, margin, shapley_change
from .likelihood import MeasurementModel


def test_shapley_signed_main_effects_and_all_interactions():
    values={s:4+2*s[0]-3*s[1]+5*s[2]+6*s[0]*s[1]+9*s[0]*s[1]*s[2]
            for s in product((0,1),repeat=3)}
    result=shapley_change(values)
    assert result['contributions']==pytest.approx(dict(lines=8.,background=3.,source=8.))
    assert result['total_change']==pytest.approx(19.)
    assert abs(result['identity_error'])<1e-12


def test_shapley_rejects_missing_or_failed_factor_states():
    states={s:0. for s in product((0,1),repeat=3)}
    states.pop((1,1,1))
    with pytest.raises(ValueError,match='Eight complete'):shapley_change(states)
    states[(1,1,1)]=float('nan')
    with pytest.raises(ValueError,match='Eight complete'):shapley_change(states)


def test_margin_requires_every_candidate_and_negative_gaps_are_retained():
    assert margin({'a':4.,'b':2.,'c':9.,'d':8.},'a',['a','b','c','d'])==-2.
    assert margin({'a':4.,'b':2.},'a',['a','b','c','d']) is None
    noiseless_delta=-3.;observed_delta=2.
    assert observed_delta-noiseless_delta==5.


def test_factor_substitution_keeps_groups_and_common_root():
    predicted={k:'p_'+k for k in ('r','x','p','q','sp','sq','root')}
    true={k:'t_'+k for k in predicted}
    params=factor_parameters(predicted,true,(1,0,1))
    assert params==dict(r='t_r',x='t_x',p='p_p',q='p_q',sp='t_sp',sq='t_sq',root='p_root')


def test_target_has_ev_once_fixed_bias_and_no_redrawn_noise():
    fm=MeasurementModel(p_sigma=.1,q_sigma=.1,v_sigma=.1,master_p_sigma=.1,master_q_sigma=.1,
                        bias_p=.01,bias_q=.02,bias_v=.03)
    params=dict(r=np.array([.01]),x=np.array([.005]),p=np.array([[.1,.2]]),q=np.array([[.04,.05]]),
                sp=np.array([.03]),sq=np.array([.01]),root=np.ones(1))
    seen={}
    def fake(candidate,r,x,p,q,sp,sq,root):
        seen['p']=p.copy()
        return dict(v=np.array([[.98,.97]]),master_p=p.sum(axis=1)+sp,master_q=q.sum(axis=1)+sq,
                    ac_equation_residual=0.,power_balance_residual=0.,iterations=1)
    observation,diagnostic=make_target(None,params,[[2.,-1.]],fm,base_kw=20.,ac_predict=fake)
    np.testing.assert_allclose(seen['p'],[[.2,.15]])
    np.testing.assert_allclose(observation.p,[[.21,.16]])
    np.testing.assert_allclose(observation.q,[[.06,.07]])
    np.testing.assert_allclose(observation.v,[[1.01,1.]])
    np.testing.assert_allclose(observation.master_p,[.38])
    assert observation.source_p is None and observation.source_q is None
    assert diagnostic['future_white_noise'] is False and diagnostic['EV_in_total_PQ_once']
