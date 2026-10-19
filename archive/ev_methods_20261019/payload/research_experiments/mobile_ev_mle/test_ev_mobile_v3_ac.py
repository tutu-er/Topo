import numpy as np
import pytest
from . import ev_mobile_v3_cases as cases
from . import ev_mobile_cases_v2 as oldcases
from . import ev_mobile_v3_ac as ac
from . import ev_fixed_train_design as fixed
from . import ev_rx_study as rx
from .ev_gradient_flow_study import public_curves


@pytest.fixture(scope='module')
def setup():
    case=cases.make_case('n3',2026101601,0,H=8,T=12)
    fits,rec=rx.train_bank(case['bank'],case['train'],case['tm'],case['config'],123)
    assert rec['all_complete']
    fc=cases.forecast_for(case['train'],case['tm'],fits,case['config'])
    u=public_curves(12,3)['constant']
    ctx=fixed.build_fixed_contexts(case['train'],case['bank'],fits,case['tm'],case['fm'],
        -fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],u,
        base_kw=20,forecast=fc,**case['config']['source'])
    return case,ctx,u


def test_training_residual_and_nuisance_layout(setup):
    case,ctx,u=setup;data=ac.geometry_inputs(ctx,u)
    assert len(data['pairs'])==12 and len(data['own'])==4
    H,T,n=case['train'].t,*u.shape
    for row in data['pairs']+data['own']:
        b=next(c for c in case['bank'] if c.name==row['b'])
        assert row['J'].shape==((H+T)*(n+2),2*b.m+2*H+2*T)
        assert np.count_nonzero(row['B'][:H*(n+2)])==0
    for v in data['diagnostics'].values():
        assert abs(v['history_SSE']-v['training_record_SSE'])<1e-8
    assert data['line_adjustment_shared_across_time']


def test_control_derivative_is_full_ac_difference(setup):
    _,ctx,u=setup;data=ac.geometry_inputs(ctx,u)
    direction=np.random.default_rng(1234).normal(size=u.shape);eps=1e-4
    plus=ac.geometry_inputs(ctx,u+eps*direction)
    minus=ac.geometry_inputs(ctx,u-eps*direction)
    for row,p,m in zip(data['pairs'],plus['pairs'],minus['pairs']):
        fd=(p['residual']-m['residual'])/(2*eps)
        np.testing.assert_allclose(fd,row['B']@direction.ravel(),rtol=2e-5,atol=2e-7)


def test_window_and_training_length_do_not_redraw_trajectory():
    a=cases.make_case('n3',2026101603,1,H=8,T=8)
    b=cases.make_case('n3',2026101603,1,H=96,T=12)
    for k in ('p','q','source_p','source_q','root_v'):
        np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['train'].v,b['train'].v[-8:])
    assert np.sum(a['config']['public_voltage_repeat_counts'])==30.
    assert np.sum(b['config']['public_voltage_repeat_counts'])==30.


def test_existing_four_terminal_generator_is_unchanged():
    a=cases.make_case('n4',2026101604,3,H=96,T=12)
    b=oldcases.make_case('twinblock12',2026101604,3,training_points=96)
    for k in ('p','q','source_p','source_q'):
        np.testing.assert_array_equal(a[k],b[k])
    np.testing.assert_array_equal(a['train'].v,b['train'].v)
