import numpy as np
from research_experiments.mobile_ev_mle import ev_cross_all_study as old
from research_experiments.mobile_ev_mle import ev_cross_all_study_v2 as new
from research_experiments.mobile_ev_mle.ev_bilinear_score_exchange_v2 import exchange_bilinear_scores_v2

def test_v2_changes_numerical_policy_without_changing_data_or_physical_domain():
    assert new.CONFIG['minimum_ac_gain']==new.CONFIG['exchange']['absolute_gap']==.01
    assert new.exchange_bilinear_scores is exchange_bilinear_scores_v2
    assert new.CONFIG['numeric_positive_floor']==old.CONFIG['numeric_positive_floor']
    assert new.CASES==old.CASES and new.METHODS==old.METHODS
    assert new.CONFIG['seeds']==old.CONFIG['seeds']
    for correlated in (False,True):
        a=old.make_development_path(987105,((0,2),),correlated)
        b=new.make_development_path(987105,((0,2),),correlated)
        for name in ('r','x','p','q','source_p','source_q'):
            np.testing.assert_array_equal(getattr(a[0],name),getattr(b[0],name))
        for name in a[3]:np.testing.assert_array_equal(a[3][name],b[3][name])

def test_new_version_freeze_includes_repaired_solver_and_policy_sources():
    hashes=new.source_hashes()
    assert 'ev_cross_all_study_v2.py' in hashes
    assert 'ev_bilinear_score_exchange_v2.py' in hashes
    assert 'ev_cross_v2_protocol.md' in hashes
    assert new.CONFIG['version']!=old.CONFIG['version']
