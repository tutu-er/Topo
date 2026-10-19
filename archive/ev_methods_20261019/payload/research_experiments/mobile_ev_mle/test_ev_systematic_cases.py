from types import SimpleNamespace
import numpy as np
import pytest
from . import ev_systematic_cases as c
from . import ev_fixed_train_design as fixed


@pytest.mark.parametrize('case_id',c.CASES)
def test_declared_case_shapes_and_common_sampling_budget(case_id):
    case=c.make_case(case_id,996001,1)
    assert case['train'].t==case['config']['training_points']
    assert case['validation'].t==4
    assert sum(case['config']['public_voltage_repeat_counts'])==30
    future=c.observe_future(case,np.zeros((case['config']['horizon'],case['train'].n)))
    assert future.t==case['config']['horizon']
    assert future.n==case['train'].n
    assert np.isfinite(future.v).all()


def test_training_lengths_hold_same_endpoint_assets_and_future():
    paths=[c.make_case(k,996002,2) for k in ('ordinary96','train32','train8')]
    for x in paths[1:]:
        h=x['train'].t
        np.testing.assert_array_equal(x['train'].p,paths[0]['train'].p[-h:])
        np.testing.assert_array_equal(x['validation'].p,paths[0]['validation'].p)
        np.testing.assert_array_equal(x['p'][100:],paths[0]['p'][100:])
        np.testing.assert_array_equal(x['truth']['r'],paths[0]['truth']['r'])


def test_horizon_extension_keeps_original_past_and_noise_prefix():
    ordinary=c.make_case('ordinary96',996002,2)
    longer=c.make_case('long_horizon96',996002,2)
    for key in ('p','q','v','master_p','master_q'):
        np.testing.assert_array_equal(getattr(ordinary['train'],key),getattr(longer['train'],key))
        np.testing.assert_array_equal(ordinary['white'][key],longer['white'][key])


def test_stress_cases_have_explicit_domain_and_coverage_failure():
    weak=c.make_case('weak_asset96',996003,1)
    assert np.min(weak['truth']['r'])<weak['config']['training']['r_bounds'][0]
    assert c.make_case('weak_numeric96',996003,1)['config']['training']['r_bounds'][0]==1e-6
    miss=c.make_case('candidate_miss96',996004,0)
    assert not any(k.supports==miss['truth']['candidate'].supports for k in miss['bank'])
    assert c.candidate_bank('candidate_miss96')[0].name=='star'


def test_physical_ev_base_and_source_transforms():
    physical=c.make_case('case33local12',996005,0)
    assert physical['base_kw']==1000.
    assert np.mean(physical['p'][100:])>.08
    zero=c.make_case('no_source96',996005,0)
    assert np.max(abs(zero['source_p']))==0
    ordinary=c.make_case('ordinary96',996005,0)
    strong=c.make_case('strong_source96',996005,0)
    np.testing.assert_allclose(strong['source_p']-.1,4*(ordinary['source_p']-.1))


def test_physical_mean_forecast_does_not_clip_at_household_12kw():
    case=c.make_case('case33local12',996006,0)
    fits={k.name:SimpleNamespace(source_p=np.zeros(12),source_q=np.zeros(12)) for k in case['bank']}
    fore=c.forecast_for(case['train'],case['tm'],fits,case['config'])
    assert np.mean(fore['p'])*1000>90
    assert fore['metadata']['uses_truth'] is False
    assert fore['metadata']['uses_validation'] is False


def test_operation_domains_allow_baseline_under_access_and_longer_window():
    for case_id,T in [('access96',6),('long_horizon96',8)]:
        u=np.zeros((T,3));u[:,0]=3./(T*.25)
        for mobile in (False,True):
            d=c.service_domain(case_id,T,3,mobile=mobile)
            assert fixed._realize(d,u)['success']
