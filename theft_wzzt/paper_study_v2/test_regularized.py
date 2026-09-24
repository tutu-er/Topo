import numpy as np
from paper_study.test_profiles import fixture
from paper_study.lp_profile import lp_profile
from paper_study_v2.regularized import regularized_profile
from paper_study_v2.profile import fit_profile

def test_zero_regularization_recovers_unrestricted_lp():
    tree,data,weights,bounds=fixture()
    a=lp_profile(tree,data,bounds);b=regularized_profile(tree,data,weights,bounds,strength=0.)
    assert abs(a['h0_loss']-b['h0_loss'])<1e-6
    assert abs(a['h1_loss']-b['h1_loss'])<1e-6
    assert np.allclose([r['loss'] for r in a['profile']],[r['loss'] for r in b['profile']],atol=1e-6)

def test_gain_decomposition_includes_prior():
    tree,data,weights,bounds=fixture()
    result=regularized_profile(tree,data,weights,bounds,strength=10.)
    assert result['gain']>=0
    assert abs(result['gain']-result['gain_voltage']-result['gain_balance']-result['gain_penalty'])<1e-7
    assert result['best_location']==1

def test_one_scale_is_subset_of_two_scales():
    tree,data,weights,_=fixture()
    a=fit_profile(tree,data,weights,two_scales=False);b=fit_profile(tree,data,weights,two_scales=True)
    assert b['h0_loss']<=a['h0_loss']+1e-6
    assert all(y['loss']<=x['loss']+1e-6 for x,y in zip(a['profile'],b['profile']))
