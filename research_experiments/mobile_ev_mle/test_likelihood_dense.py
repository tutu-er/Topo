"""A second numerical linear step must preserve the complete AC objective."""
import numpy as np
import pytest
from research_experiments.mobile_ev_mle.likelihood import fit_candidate
from research_experiments.mobile_ev_mle.simulation import generate,observe,model

def test_dense_and_sparse_steps_agree_on_same_interior_noisy_problem():
    path=generate(901,'strong');obs,_=observe(path,slice(0,12));meter=model(3)
    initial=dict(r=path.r,x=path.x,p=path.p[:12],q=path.q[:12],source_p=path.source_p[:12],source_q=path.source_q[:12])
    fits=[fit_candidate(path.candidate,obs,meter,initial=initial,multistart=1,max_nfev=300,linear_solver=s) for s in ('lsmr','exact')]
    assert all(f.success for f in fits)
    assert fits[0].loss==pytest.approx(fits[1].loss,rel=1e-5,abs=1e-4)
    assert fits[0].loglik==pytest.approx(fits[1].loglik,abs=1e-3)
    assert all(f.diagnostics['ac_equation_residual']<1e-10 for f in fits)
    assert all(not f.diagnostics['global_optimality_certified'] for f in fits)

def test_unsupported_numerical_solver_is_rejected():
    path=generate(902,'weak');obs,_=observe(path,slice(0,12))
    with pytest.raises(ValueError,match='linear_solver'):
        fit_candidate(path.candidate,obs,model(3),linear_solver='unknown')
