"""Independent physical-unit and contraction checks for the new LV family."""
import numpy as np
import pytest
from . import likelihood as lk
from .candidates import RadialCandidate
from .ev_original_study import make_case,FAMILIES,ASSET_BOUNDS,BASE_KW,ZBASE,CONFIG

@pytest.mark.parametrize('family',FAMILIES)
def test_declared_cable_family_covers_all_true_edges(family):
    case=make_case(988990,family)
    assert np.all(case['r']>=ASSET_BOUNDS['r_bounds'][0])
    assert np.all(case['r']<=ASSET_BOUNDS['r_bounds'][1])
    assert np.all(case['x']>=ASSET_BOUNDS['x_bounds'][0])
    assert np.all(case['x']<=ASSET_BOUNDS['x_bounds'][1])
    for asset,r,x in zip(case['assets'],case['r'],case['x']):
        assert .025<=asset['length_km']<=.15
        ratio=.642/.083 if asset['cable_mm2']==50 else .208/.08
        assert r/x==pytest.approx(ratio)
    assert ZBASE==pytest.approx(.4**2/(BASE_KW/1000))

def test_short_edge_is_explicit_domain_miss():
    case=make_case(988991,((0,1),),short_edge=True)
    internal=[i for i,s in enumerate(case['candidate'].supports) if len(s)==2]
    assert len(internal)==1
    assert case['r'][internal[0]]<ASSET_BOUNDS['r_bounds'][0]
    assert case['x'][internal[0]]<ASSET_BOUNDS['x_bounds'][0]

def test_generator_is_reproducible_and_history_noise_separate_from_future():
    a=make_case(988992,());b=make_case(988992,());c=make_case(988993,())
    np.testing.assert_array_equal(a['history'].v,b['history'].v)
    assert not np.array_equal(a['history'].v,c['history'].v)
    assert len(set(a['seeds'].values()))==5
    assert a['history'].t==CONFIG['history']
    assert len(a['white']['v'])==CONFIG['history']+CONFIG['horizon']

def test_exact_zero_internal_edge_contracts_to_star_in_complete_AC():
    star=RadialCandidate.from_supports('star',3,())
    cherry=RadialCandidate.from_supports('cherry',3,((0,1),))
    def lines(c,epsilon):
        r=np.array([.001 if len(s)==3 else .007+.001*s[0] if len(s)==1 else epsilon for s in c.supports])
        x=np.array([.0004 if len(s)==3 else .001 if len(s)==1 else epsilon/3 for s in c.supports])
        return r,x
    p=np.array([[.12,.13,.14],[.42,.13,.14],[.12,-.17,.14]])
    q=np.full_like(p,.04);sp=np.array([.1,.11,.09]);sq=.03*np.ones(3)
    ref=lk.ac_predict(star,*lines(star,0),p,q,sp,sq)
    same=lk.ac_predict(cherry,*lines(cherry,0),p,q,sp,sq)
    for name in ('v','master_p','master_q'):
        np.testing.assert_allclose(ref[name],same[name],atol=3e-14,rtol=0)
    errors=[]
    for epsilon in (1e-4,1e-5,1e-6):
        pred=lk.ac_predict(cherry,*lines(cherry,epsilon),p,q,sp,sq)
        errors.append(np.linalg.norm(pred['v']-ref['v']))
    assert errors[1]/errors[0]==pytest.approx(.1,rel=.002)
    assert errors[2]/errors[1]==pytest.approx(.1,rel=.002)

def test_electrical_identity_does_not_depend_on_EV_or_ESS_label():
    case=make_case(988994,((1,2),));h=CONFIG['history'];power=np.zeros((6,3))
    power[:,1]=[6,-3,6,-3,6,0]
    args=(case['candidate'],case['r'],case['x'],case['p'][h:]+power/BASE_KW,
          case['q'][h:],case['source_p'][h:],case['source_q'][h:])
    ev=lk.ac_predict(*args);ess=lk.ac_predict(*args)
    np.testing.assert_array_equal(ev['v'],ess['v'])
    np.testing.assert_array_equal(ev['master_p'],ess['master_p'])
