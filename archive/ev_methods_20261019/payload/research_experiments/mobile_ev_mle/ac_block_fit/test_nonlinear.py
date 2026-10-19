import numpy as np
import pytest
from .. import likelihood as lk
from ..reference_benchmark.test_fitting import fixture
from ..reference_benchmark.fitting import fit_rx_inputs
from .nonlinear import fit_rx_blocks


@pytest.mark.parametrize('scale_mode',['reference','jac'])
def test_complete_ac_matches_reference_noisy_objective(scale_mode):
    c,obs,model,opts,_=fixture()
    rng=np.random.default_rng(995)
    noisy=lk.Observations(obs.p,obs.q,obs.v+rng.normal(0,.0001,obs.v.shape),
        master_p=obs.master_p+rng.normal(0,.0001,obs.t),
        master_q=obs.master_q+rng.normal(0,.0001,obs.t))
    opts.update(max_nfev=150,tol=1e-9)
    old=fit_rx_inputs(c,noisy,model,**opts)
    new=fit_rx_blocks(c,noisy,model,scale_mode=scale_mode,**opts)
    assert old.success and new.success, (old.message,new.message)
    assert abs(new.loss-old.loss)<1e-4
    assert new.diagnostics['full_global_Jacobian_formed'] is False
    assert new.diagnostics['ac_equation_residual']<1e-10
    assert new.diagnostics['power_balance_residual']<1e-10
    assert abs(new.loglik+.5*(new.loss+new.diagnostics['output_normalizer']))<1e-10
    assert all(row['actual_decrease']>0 for row in new.diagnostics['trace'] if row['accepted'])


def test_zero_virtual_edge_is_covered_without_truth_constraint():
    c,obs,model,opts,_=fixture(True)
    opts.update(max_nfev=200,tol=1e-9)
    fit=fit_rx_blocks(c,obs,model,**opts)
    assert fit.success,fit.message
    assert fit.loss<.001
    assert fit.diagnostics['physical_voltage_feasible']


def test_budget_failure_retains_finite_feasible_result():
    c,obs,model,opts,_=fixture()
    opts.update(max_nfev=1)
    fit=fit_rx_blocks(c,obs,model,**opts)
    assert not fit.success and fit.status==0
    assert np.isfinite(fit.loss)
    assert fit.diagnostics['failure_kind']=='computation'
    assert fit.diagnostics['physical_voltage_feasible']


def test_invalid_scaling_is_explicit_failure():
    c,obs,model,opts,_=fixture()
    with pytest.raises(ValueError,match='scale_mode'):
        fit_rx_blocks(c,obs,model,scale_mode='unknown',**opts)


def test_gap_acceptance_records_the_same_convex_model_bound():
    c,obs,model,opts,_=fixture()
    opts.update(max_nfev=30)
    strict=fit_rx_blocks(c,obs,model,inner_tol=1e-25,**opts)
    gap=fit_rx_blocks(c,obs,model,inner_tol=1e-25,
        inner_acceptance='certified_gap',**opts)
    assert not strict.success
    accepted=[r for r in gap.diagnostics['trace'] if r.get('inner_gap_pass')]
    assert accepted
    assert all(r['inner_gap_upper_bound']<=r['inner_gap_threshold'] for r in accepted)
    assert gap.loss<strict.loss


def test_unknown_backend_is_not_silently_ignored():
    c,obs,model,opts,_=fixture()
    with pytest.raises(ValueError,match='backend'):
        fit_rx_blocks(c,obs,model,backend='unknown',**opts)
