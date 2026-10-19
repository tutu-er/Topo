from fractions import Fraction as F
import math
import numpy as np

from research_experiments.mobile_ev_mle.nominal_weak_information import exact_certificate,outward_information_limits


def test_exact_domain_margins_and_nonrounded_gaussian_bound():
    c,checks,margins=exact_certificate()
    assert all(checks.values()) and all(x>0 for x in margins.values())
    assert c['C']==F('.02714416')
    assert F('.10715701')<c['kl_exact']<F('.10715702')


def test_all_independent_scale_leaf_differences_and_shared_terms():
    c,_,_=exact_certificate()
    ranges=[((F('.028')+F('.003')*i)*F('.8'),(F('.028')+F('.003')*i)*F('1.2')) for i in range(3)]
    xr=[((F('.011')+F('.002')*i)*F('.8'),(F('.011')+F('.002')*i)*F('1.2')) for i in range(3)]
    for i,j in ((0,1),(0,2),(1,2)):
        for ri in ranges[i]:
            for rj in ranges[j]:
                for xi in xr[i]:
                    for xj in xr[j]:
                        assert (ri-rj)**2+(xi-xj)**2<=c['dzleaf']**2
    assert F('.008')**2*2<F('.012')**2


def test_translated_uniform_overlap_all_three_pairs():
    c,_,_=exact_certificate()
    expected={'01':F(4261491,133203125),'02':F(4555403,150390625),'12':F(12101834,431640625)}
    assert c['prior_tv']==expected
    assert all(F(0)<x<F('.033') for x in expected.values())


def test_outward_limits_have_correct_directions():
    limits=outward_information_limits()
    assert limits['tv_outward']>F(str(math.sqrt(.055)))+F('.033')
    assert limits['equal_binary_error_lower']==F('.3662395')
    assert limits['full_population_forced_error_lower']==F('.18311975')
    assert limits['binary_joint_MI_upper']>limits['tv_outward']*F(str(math.log(2)))
    assert float(limits['binary_joint_MI_upper'])<.185432


def test_binary_entropy_chord_and_full_population_error_reduction():
    for t in np.linspace(.0001,.9999,1001):
        entropy=-t*math.log(t)-(1-t)*math.log1p(-t)
        assert entropy+1e-14>=2*math.log(2)*min(t,1-t)
    # Independent finite density example: full weighted overlap is bounded
    # below by min prior mass times unweighted overlap, as used in the proof.
    p=np.array([.1,.3,.6]);q=np.array([.4,.4,.2])
    assert np.minimum(.25*p,.375*q).sum()>=.25*np.minimum(p,q).sum()
