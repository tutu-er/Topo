import numpy as np
import pytest

from .. import likelihood as lk
from .common_gradient_diagnosis import stable_changes, _ac_precision, _profile_spectrum, decimal_evaluate


def test_stable_prediction_retains_sub_ulp_rss_decrease():
    residual=np.array([1.,1.])
    delta=np.array([1e-9,-1e-9-1e-16])
    result=stable_changes(residual,residual+delta,delta)
    expected=-sum(float(x) for x in delta)-.5*sum(float(x*x) for x in delta)
    assert result['compensated_predicted']==pytest.approx(expected,rel=1e-14,abs=1e-30)
    assert result['compensated_predicted']>0


def test_stable_actual_matches_algebra_for_resolvable_changes():
    r=np.array([3.,-2.,.5]);rn=np.array([2.9,-1.9,.4])
    result=stable_changes(r,rn)
    assert result['raw_actual']==pytest.approx(result['compensated_actual'],abs=1e-14)


def test_temporary_AC_precision_hook_restored_on_exception():
    original=lk.ac_predict
    with pytest.raises(RuntimeError):
        with _ac_precision(1e-15):
            assert lk.ac_predict is not original
            raise RuntimeError('check restoration')
    assert lk.ac_predict is original


def test_profile_projection_removes_source_compensation():
    source=np.array([[[1.,0.],[0.,1.],[0.,0.]]])
    line=np.array([[[1.,0.],[0.,0.],[0.,1.]]])
    result=_profile_spectrum(line,source,np.ones(2))
    assert result['singular_values'][0]==pytest.approx(1.)
    assert result['singular_values'][1]==pytest.approx(0.)


def test_decimal_forward_replays_same_physical_AC_inputs():
    from ..reference_benchmark.test_fitting import fixture
    from ..ev_rx_fit import _RxInputProblem
    c,data,meter,options,state=fixture()
    domain={k:v for k,v in options.items() if k not in ('max_nfev','tol','multistart')}
    problem=_RxInputProblem(c,data,meter,initial=state,**domain)
    x=problem.start
    problem.evaluate(x)
    expected=problem.base.cache['physical']
    actual=decimal_evaluate(problem,x,digits=50)
    np.testing.assert_allclose(np.asarray(actual['voltage'],float),expected['v'],atol=3e-15,rtol=0.)
    np.testing.assert_allclose(np.asarray(actual['master'],float),
        np.c_[expected['master_p'],expected['master_q']],atol=3e-15,rtol=0.)


def test_decimal_timeout_is_explicit():
    from ..reference_benchmark.test_fitting import fixture
    from ..ev_rx_fit import _RxInputProblem
    c,data,meter,options,state=fixture()
    domain={k:v for k,v in options.items() if k not in ('max_nfev','tol','multistart')}
    problem=_RxInputProblem(c,data,meter,initial=state,**domain)
    with pytest.raises(TimeoutError):
        decimal_evaluate(problem,problem.start,timeout=-1.)
