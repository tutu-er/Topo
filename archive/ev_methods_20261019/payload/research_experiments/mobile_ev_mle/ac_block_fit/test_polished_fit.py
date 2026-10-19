"""Budget/acceptance-rule tests and independent full-AC replay checks.

The synthetic fixture isolates the acceptance rule. Its supplied local blocks
are deliberately controlled, and are not used as evidence about AC accuracy.
"""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem
from ..reference_benchmark.test_fitting import fixture
from .blocks import problem_blocks
from .nonlinear import fit_rx_blocks
from .polished_fit import polish_stationarity, fit_rx_blocks_polished
from . import polished_fit as polished_module


class AcceptanceFixture:
    def __init__(self, change=0., physical_failure=False, derivative_after=.01,
                 start=0., lower=-100., upper=100.):
        self.t=1
        self.lower=np.array([lower,0.,0.]);self.upper=np.array([upper,0.,0.])
        self.scale=np.ones(3)
        self.mapping=SimpleNamespace(contains=lambda r,x: True)
        self.base=SimpleNamespace(cache={},ac_calls=0,voltage_bounds=(.8,1.2))
        self.change=change;self.physical_failure=physical_failure
        self.derivative_after=derivative_after;self.start=start

    def unpack(self,x):
        return np.array([x[0]]),np.zeros(1),x[1:].reshape(1,2)

    def evaluate(self,x):
        self.base.ac_calls+=1
        moved=not np.array_equal(x,np.array([self.start,0.,0.]))
        residual=np.array([np.sqrt(200.+2.*self.change*moved),0.,0.])
        self.base.cache=dict(vector=x.copy(),residual=residual.copy(),physical=dict(
            voltage=np.array([[.7 if moved and self.physical_failure else 1.]]),
            ac_equation_residual=0.,power_balance_residual=0.))
        return residual

    def blocks(self,problem,x):
        moved=not np.array_equal(x,np.array([self.start,0.,0.]))
        line=np.array([[[self.derivative_after if moved else 1.],[0.],[0.]]])
        return line,np.zeros((1,3,2))


def proposed_step(step=-.5,success=True):
    def call(*args,**kwargs):
        return dict(step=np.array([step,0.,0.]),success=success,kkt=0.,
                    error_bound=0.,iterations=1)
    return call


def test_no_budget_means_no_trial_and_separate_entry_audit():
    p=AcceptanceFixture()
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=0,
        block_function=p.blocks,step_function=proposed_step())
    assert out['trial_nfev']==0 and out['entry_audit_ac_calls']==1
    assert out['actual_trial_AC_calls']==0 and out['status']=='no_remaining_budget'
    assert not out['success']


def test_tiny_numerical_increase_is_kept_and_explicit():
    band=256*np.finfo(float).eps*100.
    p=AcceptanceFixture(change=band/2)
    out=polish_stationarity(p,np.zeros(3),target=1.,remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step())
    assert out['success'] and out['trial_nfev']==1
    assert 0<out['actual_F_change']<=out['loss_tolerance']
    assert out['accepted_positive_F_changes']==[out['actual_F_change']]
    assert not out['loss_band_accumulates']


@pytest.mark.parametrize('change',[1e-10,.1])
def test_any_loss_increase_above_entry_band_is_rejected(change):
    p=AcceptanceFixture(change=change)
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step())
    assert out['accepted_steps']==0 and np.array_equal(out['x'],np.zeros(3))
    assert out['status']=='loss_above_initial_numerical_band'
    assert out['final_F']==out['initial_F']
    assert p.base.cache['vector'][0]==0.


def test_physical_failure_is_retained_as_failure_and_restores_cache():
    p=AcceptanceFixture(physical_failure=True)
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step())
    assert out['accepted_steps']==0 and out['status']=='trial_physical_failure'
    assert out['trace'][0]['physical']['voltage_feasible'] is False
    assert out['physical']['passed'] and p.base.cache['vector'][0]==0.


def test_unchanged_common_gradient_is_rejected():
    p=AcceptanceFixture(derivative_after=1.)
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step())
    assert out['accepted_steps']==0
    assert out['status']=='common_projected_gradient_not_strictly_lower'


def test_strict_inner_failure_has_no_full_ac_trial():
    p=AcceptanceFixture()
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step(success=False))
    assert out['status']=='inner_failure' and out['trial_nfev']==0


def test_step_is_bounded_in_original_coordinates():
    p=AcceptanceFixture(lower=-.1)
    out=polish_stationarity(p,np.zeros(3),target=.01,remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step(step=-.1))
    assert out['success'] and out['x'][0]==-.1
    assert out['physical']['optimizer_and_source_box_feasible']


def test_infeasible_inner_step_is_rejected_rather_than_clipped():
    p=AcceptanceFixture(lower=-.1)
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step(step=-.5))
    assert out['status']=='inner_failure' and out['trial_nfev']==0
    assert out['trace'][0]['reason']=='bounded_QR_returned_infeasible_step'


def test_nonpositive_model_descent_rejected_without_ac_trial():
    p=AcceptanceFixture()
    out=polish_stationarity(p,np.zeros(3),remaining_nfev=1,
        block_function=p.blocks,step_function=proposed_step(step=.5))
    assert out['status']=='non_descent_step' and out['trial_nfev']==0


def test_full_ac_wrapper_replays_loss_state_and_cost():
    c,obs,model,opts,_=fixture()
    opts.update(max_nfev=150,tol=1e-9,scale_mode='jac')
    result=fit_rx_blocks_polished(c,obs,model,**opts)
    assert result.diagnostics['normal_success']
    record=result.diagnostics['polish']
    assert record['attempted']
    assert record['independent_replay']['physical_audit']['passed']
    assert record['optimization_nfev_total']<=opts['max_nfev']
    assert result.nfev==record['optimization_nfev_total']
    assert record['independent_loss_matches']
    physical=lk.ac_predict(c,result.r,result.x,obs.p,obs.q,result.source_p,result.source_q)
    assert np.max(abs(result.predicted['physical_v']-physical['v']))<1e-13
    assert abs(result.loglik+.5*(result.loss+result.diagnostics['output_normalizer']))<1e-10
    assert result.success==bool(record['independent_replay']['common_pg']<=1e-5)
    assert result.diagnostics['stationarity_pass']==result.diagnostics['common_stationarity_pass']
    assert result.diagnostics['stationarity_tolerance']==1e-5
    assert result.diagnostics['normal_stationarity_pass']==result.diagnostics['normal_fit']['stationarity_pass']
    assert result.diagnostics['normal_stationarity_tolerance']==result.diagnostics['normal_fit']['stationarity_tolerance']


def test_wrapper_does_not_rescue_original_budget_failure():
    c,obs,model,opts,_=fixture();opts['max_nfev']=1
    result=fit_rx_blocks_polished(c,obs,model,**opts)
    assert not result.success and result.status==0
    assert not result.diagnostics['normal_success']
    assert result.diagnostics['polish']['trial_nfev']==0
    assert result.diagnostics['polish']['status']=='normal_budget_failure'
    assert np.isfinite(result.loss)


def test_fixed_lines_retain_original_source_only_path():
    c,obs,model,opts,state=fixture()
    opts.update(fixed_r=state['r'],fixed_x=state['x'])
    result=fit_rx_blocks_polished(c,obs,model,**opts)
    assert result.success
    assert result.diagnostics['polish']['status']=='fixed_lines_original_method_retained'


def test_zero_virtual_edge_full_ac_uses_original_domain():
    c,obs,model,opts,_=fixture(True)
    opts.update(max_nfev=200,tol=1e-9,scale_mode='jac')
    result=fit_rx_blocks_polished(c,obs,model,**opts)
    assert np.isfinite(result.loss)
    if result.diagnostics['polish']['attempted']:
        physical=result.diagnostics['polish']['independent_replay']['physical_audit']
        assert physical['passed'] and physical['RX_domain_feasible']


def test_actual_entry_ac_exception_accounts_cost_and_failure(monkeypatch):
    c,obs,model,opts,_=fixture();counter={}

    def fail_ac(*args,**kwargs):
        raise RuntimeError('injected full-AC entry failure')

    def normal_then_fail(*args,**kwargs):
        fitted=fit_rx_blocks(*args,**kwargs)
        assert fitted.success
        monkeypatch.setattr(lk,'ac_predict',fail_ac)
        return fitted

    result=fit_rx_blocks_polished(c,obs,model,normal_fitter=normal_then_fail,
        call_counter=counter,**opts)
    d=result.diagnostics
    assert not result.success and result.status==-8
    assert 'injected full-AC entry failure' in result.message
    assert d['failure_kind']=='computation'
    assert d['polish_and_replay_actual_AC_calls']==1
    assert d['ac_evaluations']==counter['ac_calls']==d['normal_AC_calls']+1
    assert result.nfev==d['total_nfev']==d['normal_fit']['nfev']
    assert d['seconds']>=d['normal_seconds'] and d['seconds']>0
    assert not d['stationarity_pass'] and not d['common_stationarity_verified']


def test_actual_replay_ac_exception_keeps_all_trial_costs(monkeypatch):
    c,obs,model,opts,_=fixture();counter={}
    original_replay=polished_module._independent_replay

    def fail_ac(*args,**kwargs):
        raise RuntimeError('injected independent full-AC replay failure')

    def replay_failure(problem,vector):
        with monkeypatch.context() as context:
            context.setattr(lk,'ac_predict',fail_ac)
            return original_replay(problem,vector)

    monkeypatch.setattr(polished_module,'_independent_replay',replay_failure)
    result=fit_rx_blocks_polished(c,obs,model,polish_target=1e-20,call_counter=counter,**opts)
    d=result.diagnostics;trial_count=d['polish']['trial_nfev']
    assert trial_count>=1
    assert not result.success and result.status==-8
    assert 'injected independent full-AC replay failure' in result.message
    assert d['failure_kind']=='computation'
    assert result.nfev==d['total_nfev']==d['normal_fit']['nfev']+trial_count
    assert d['ac_evaluations']==counter['ac_calls']
    assert d['polish_and_replay_actual_AC_calls']==trial_count+2
    assert d['ac_evaluations']==d['normal_AC_calls']+trial_count+2
    assert d['polish']['optimization_nfev_total']==result.nfev
    assert not d['stationarity_pass'] and not d['common_stationarity_verified']
    assert d['seconds']>=d['normal_seconds']


def test_reconstruction_failure_accounts_normal_cost_without_trial():
    c,obs,model,opts,_=fixture();counter={}

    def inconsistent_normal(*args,**kwargs):
        fitted=fit_rx_blocks(*args,**kwargs)
        assert fitted.success
        return replace(fitted,r=fitted.r+1.)

    result=fit_rx_blocks_polished(c,obs,model,normal_fitter=inconsistent_normal,
        call_counter=counter,**opts)
    d=result.diagnostics
    assert not result.success and result.status==-8
    assert 'reconstruction differs' in result.message
    assert d['failure_kind']=='computation'
    assert d['polish_and_replay_actual_AC_calls']==0
    assert result.nfev==d['total_nfev']==d['normal_fit']['nfev']
    assert d['ac_evaluations']==counter['ac_calls']==d['normal_AC_calls']
    assert not d['stationarity_pass'] and not d['common_stationarity_verified']


def test_replay_failure_without_external_counter_still_records_ac_cost(monkeypatch):
    c,obs,model,opts,_=fixture()
    original_replay=polished_module._independent_replay

    def replay_failure(problem,vector):
        def fail_ac(*args,**kwargs):
            raise RuntimeError('injected replay failure without shared counter')
        with monkeypatch.context() as context:
            context.setattr(lk,'ac_predict',fail_ac)
            return original_replay(problem,vector)

    monkeypatch.setattr(polished_module,'_independent_replay',replay_failure)
    result=fit_rx_blocks_polished(c,obs,model,**opts)
    d=result.diagnostics
    assert result.status==-8 and not result.success
    assert not d['external_counter_supplied']
    assert d['ac_evaluations']==d['normal_AC_calls']+d['polish_and_replay_actual_AC_calls']
    assert d['polish_and_replay_actual_AC_calls']>=2
    assert result.nfev==d['total_nfev']

