"""A fast generative smoke test; full calibration uses the declared 2048 run."""
import inspect
import numpy as np
from . import ev_bayes_response_linear_calibration as calibration
from .test_ev_bayes_response_ac import run_generated_linear_calibration as archived_kernel


def test_standalone_kernel_preserves_archived_2048_audit_source():
    assert inspect.getsource(calibration.run_generated_linear_calibration)==inspect.getsource(archived_kernel)


def test_independent_prior_generated_histories_and_mass_ratio_replay():
    result=calibration.run_generated_linear_calibration(16,2026111702)
    assert result['status']=='complete' and result['independent_histories']==16
    assert len(result['rows'])==16 and sum(r['count'] for r in result['probability_bins'])==16
    assert result['max_sequential_mean_covariance_error']<1e-12
    assert result['max_sequential_log_evidence_error']<1e-12
    assert result['max_exact_predictive_48_vs96_quadrature_CDF_error']<2e-4
    assert {r['true_model'] for r in result['rows']}=={0,1}
    assert all(0<=r['theta']<=1 for r in result['rows'])
    assert all(0<=r['posterior0']<=1 for r in result['rows'])
    # Small smoke-test samples establish code paths, not statistical coverage.
    assert result['model']['hyperparameters_known']
    assert not result['model']['AC_or_empirical_hyper_calibration']
