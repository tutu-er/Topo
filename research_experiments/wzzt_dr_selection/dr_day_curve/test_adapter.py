from copy import deepcopy
import numpy as np
import pytest

from .adapter import build_plan


def resources():
    return [dict(resource_id=k,node_label=n,baseline_kw=1.,power_upper_kw=2.,
        prefix_delta_lower_kwh=-2.,prefix_delta_upper_kwh=2.,available=True,
        reactive_ratio=.35,declaration_source='explicit synthetic original N5 resource')
        for k,n in [('focus',101),('partner',103)]]


def curves(scale=.2):
    u=np.zeros((2,96));u[0,24:28]=-scale;u[0,28:32]=scale
    u[1,60:64]=-scale;u[1,64:68]=scale
    return u


def make(u=None,c=None,**kwargs):
    u=curves() if u is None else u
    c=np.max(np.abs(u),axis=1)/.5 if c is None else c
    return build_plan(resources(),u,c,node_labels=[101,102,103],**kwargs)


def test_custom_96_curve_preserves_original_resource_and_metadata_and_units():
    r=resources();before=deepcopy(r);meta=dict(case='frozen_case',labels=[101,102,103],provenance=dict(no_true_tree=True))
    x=build_plan(r,curves(),[.4,.4],node_labels=[101,102,103],metadata=meta)
    assert r==before and x['plan']['resources']==r and x['plan']['metadata']==meta
    assert len(x['plan']['modules'])==2
    assert x['contract_data']['exclusive_groups']==[]
    h=np.asarray(x['contract_data']['profiles']);v=h@np.asarray(x['plan']['amplitudes'])
    p=v[:288].reshape(96,3);q=v[288:].reshape(96,3)
    assert np.allclose(p[:,0],curves()[0]) and np.allclose(p[:,2],curves()[1])
    assert np.allclose(p[:,1],0) and np.allclose(q,.35*p)
    assert x['audit']['input_prices_changed'] is False


def test_arbitrary_nonbinary_daily_curves_and_exact_cost_identity():
    t=np.arange(96);u=np.vstack([.1*np.sin(2*np.pi*t/96),.08*np.cos(4*np.pi*t/96)])
    c=np.max(np.abs(u),axis=1)/.5
    x=make(u,c)
    assert np.allclose(x['audit']['mean_daily_increment_kwh'],0,atol=1e-12)
    assert x['audit']['expected_payment']==pytest.approx(float(c@(.25*np.maximum(-u,0).sum(axis=1))))
    assert x['audit']['hard_payment_bound']<=1.6
    assert len(x['audit']['support_corners'])==4
    assert all(np.max(np.abs(z['daily_increment_kwh']))<1e-12 for z in x['audit']['support_corners'])


def test_zero_price_uses_nonzero_dummy_with_zero_physical_response():
    x=make(np.zeros((2,96)),[0,0])
    assert x['audit']['zero_increment_plan'] and not x['audit']['downstream_component_check_needed']
    assert len(x['audit']['inactive_dummy_resources'])==2
    assert x['plan']['expected_payment']==x['plan']['hard_payment_bound']==0
    for m in x['plan']['modules']:
        h=np.asarray(next(iter(m['profiles'].values())))
        assert np.any(h!=0) and h.sum()==0
    assert np.max(np.abs(np.asarray(x['contract_data']['profiles'])@np.asarray(x['plan']['amplitudes'])))==0


def test_zero_price_does_not_silently_clip_even_weak_nonzero_curve():
    with pytest.raises(ValueError,match='zero price requires exactly zero'):
        make(curves(1e-12),[0,0])


def test_weak_nonzero_curve_is_retained_without_artificial_minimum():
    x=make(curves(1e-9),[2e-9,2e-9])
    assert x['plan']['prices']==[2e-9,2e-9]
    assert not x['audit']['inactive_dummy_resources']


def test_free_shape_cannot_create_unpriced_amplitude():
    with pytest.raises(ValueError,match='unpriced power'):make(curves(.4),[.2,.2])


def test_noncanonical_price_is_reported_not_silently_changed():
    x=make(curves(.1),[.4,.4])
    assert x['plan']['prices']==[.4,.4]
    assert x['audit']['canonical_daily_prices']==pytest.approx([.2,.2])
    assert x['audit']['price_scale_slack']==pytest.approx([.2,.2])


def test_nonzero_price_zero_curve_requires_canonicalization():
    with pytest.raises(ValueError,match='canonicalize price to0'):make(np.zeros((2,96)),[.4,.4])


def test_energy_nonzero_is_rejected_by_existing_module():
    u=curves();u[0,0]=.01
    with pytest.raises(ValueError,match='zero daily incremental energy'):make(u)


def test_original_prefix_energy_bounds_and_actual_amplitude_cap_are_checked():
    u=np.zeros((2,96));u[:,:48]=-.6;u[:,48:]=.6
    with pytest.raises(ValueError,match='resource/module constraints'):make(u)
    short=np.zeros((2,96));short[:,24]=-.9;short[:,25]=.9
    with pytest.raises(ValueError,match='support corner'):make(short,[1.8,1.8])


def test_budget_violation_is_rejected():
    u=np.zeros((2,96));u[:,::2]=-.4;u[:,1::2]=.4
    with pytest.raises(ValueError,match='hard budget'):make(u)


def test_uncapped_payment_retains_resource_checks_and_original_cost():
    u=np.zeros((2,96));u[:,::2]=-.4;u[:,1::2]=.4
    x=make(u,budget=None)
    assert x['plan']['budget'] is None and x['plan']['cap_enforced'] is False
    assert x['audit']['payment_cap'] is None and x['audit']['cap_enforced'] is False
    assert x['audit']['existing_actual_worker_compatible'] is False
    assert x['audit']['expected_payment']==pytest.approx(7.68)
    assert x['audit']['hard_payment_bound']==pytest.approx(9.216)
    assert x['audit']['hard_payment_bound']>1.6
    assert len(x['audit']['support_corners'])==4
    assert all(z['maximum_violation']<=1e-10 for z in x['audit']['support_corners'])
    assert x['audit']['exact_box_worst_payment']<=x['audit']['hard_payment_bound']+1e-10
    with pytest.raises(ValueError,match='hard budget'):make(u,budget=1.6)


def test_uncapped_payment_does_not_remove_device_or_energy_constraints():
    short=np.zeros((2,96));short[:,24]=-.9;short[:,25]=.9
    with pytest.raises(ValueError,match='support corner'):make(short,[1.8,1.8],budget=None)
    u=curves();u[0,0]=.01
    with pytest.raises(ValueError,match='zero daily incremental energy'):make(u,budget=None)


def test_time_independent_multipliers_break_energy_but_whole_day_law_preserves_it():
    # This is a counterexample, not a new implemented response law.
    u=np.array([-1.,1.])
    assert np.sum(1.2*u)==0
    assert .25*np.sum(np.array([.8,1.2])*u)==pytest.approx(.1)
