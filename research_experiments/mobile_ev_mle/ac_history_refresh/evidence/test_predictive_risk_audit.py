import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('predictive_risk_audit',Path(__file__).with_name('predictive_risk_audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_identical_predictive_models_have_analytical_self_risk():
    # With identical densities, the optimal classifier always uses prior .8.
    bank=dict(means=np.zeros((1,2,1)),covariances=np.ones((1,2,1,1)),log_prior=np.log([.8,.2]))
    r=audit.integrate(bank,1.,81,draws=32,replicates=3)
    assert abs(r['model_risk']-.2)<1e-14
    assert abs(r['Gaussian_Bhattacharyya_upper_bound']-.2)<1e-14
    assert r['independent_replicate_SE']<1e-14
    assert r['zero_MC_SE_is_not_a_population_or_numerical_error_bound']


def test_same_seed_is_common_integration_not_new_physical_noise():
    bank=dict(means=np.array([[[-1.],[1.]]]),covariances=np.full((1,2,1,1),2.),log_prior=np.log([.5,.5]))
    first=audit.integrate(bank,1.,82,draws=32,replicates=2)
    scaled=audit.integrate(bank,4.,82,draws=32,replicates=2)
    assert first['noise_sha256']==scaled['noise_sha256']
    assert scaled['model_risk']>first['model_risk']
