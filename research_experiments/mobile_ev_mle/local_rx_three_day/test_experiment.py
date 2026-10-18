import numpy as np

from .experiment import observed_case, specifications


def test_all_main_histories_are_at_least_three_ordinary_days_and_paired():
    specs=specifications()
    assert len(specs)==48
    assert {s['N'] for s in specs}=={288,672}
    assert len({(s['network'],s['seed']) for s in specs})==8
    assert not ({s['seed'] for s in specs}&{32991,31001,31002,31003})


def test_noisy_boundary_is_observation_not_truth_and_u_does_not_receive_it():
    s=dict(network='paper15',seed=33001,N=288,boundary='U')
    u,_=observed_case(s)
    r,truth=observed_case(dict(s,boundary='R'))
    k,_=observed_case(dict(s,boundary='K'))
    assert u['root_known'] is None and u['root_meter'] is None
    assert r['root_known'] is None
    assert not np.array_equal(r['root_meter'][0],truth['truth']['root_v'][:288])
    np.testing.assert_array_equal(k['root_known'],truth['truth']['root_v'][:288])
    for channel in ('p','q','v'):
        np.testing.assert_array_equal(u['observed'][channel],r['observed'][channel])
    assert 'truth' not in u and 'r' not in u and 'x' not in u
