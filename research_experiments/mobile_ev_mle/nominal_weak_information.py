"""Exact rational AC/parameter-prior bounds for the declared nominal population.

The theory, causal quantifiers and scope are in nominal_weak_information_math.md.
This is not a fit, estimated topology probability or nonlinear MLE certificate.
"""
from fractions import Fraction as F


def exact_certificate():
    from research_experiments.mobile_ev_mle.weak_branch_information import exact_certificate as foundation
    base,base_checks,base_margins=foundation()
    prior_tv = {}
    for i,j in ((0,1),(0,2),(1,2)):
        overlap = F(1)
        for k in (i,j):
            rn,xn=F('.028')+F('.003')*k,F('.011')+F('.002')*k
            overlap *= 1-F('.000072')/(F('.4')*rn)
            overlap *= 1-F('.000048')/(F('.4')*xn)
        prior_tv[f'{i}{j}']=1-overlap
    values={**base,'prior_tv':prior_tv,'prior_tv_outward':F('.033'),
            'foundation_scope':'all nominal independent R/X scales [.8,1.2], complex modulus caps',
            'new_result':'translate uniform parameter prior then TV data processing'}
    checks=dict(foundation=all(x>0 for x in base_margins.values()),
                parameter_tv=max(prior_tv.values())<F('.033'))
    margins=dict(gaussian_kl=base['kl_upper']-base['kl_exact'],
                 parameter_tv=F('.033')-max(prior_tv.values()))
    if not all(checks.values()):
        raise AssertionError('exact declared-domain information bounds failed')
    return values,checks,margins


def serialise_rationals(value):
    if isinstance(value,F):
        return dict(numerator=value.numerator,denominator=value.denominator,
                    decimal=float(value),displayed_decimal_is_not_exact=True)
    if isinstance(value,dict):
        return {key:serialise_rationals(v) for key,v in value.items()}
    return value


def outward_information_limits():
    """Conservative rational sqrt/log upper bounds, not floating certificates."""
    values,checks,margins = exact_certificate()
    # .234521^2>.055 and log(2)<.693148. These rational outward constants
    # give the advertised weaker decimals without floating-rounding claims.
    tv = F('.234521')+values['prior_tv_outward']
    assert F('.234521')**2 > F('.055')
    from math import factorial
    assert sum(F('.693148')**n/F(factorial(n)) for n in range(10))>2
    return dict(tv_outward=tv,equal_binary_error_lower=(1-tv)/2,
        full_population_forced_error_lower=(1-tv)/4,
        binary_joint_MI_upper=tv*F('.693148'),
        binary_future_conditional_MI_scope='population average only, not each observed H',
        classification_requires_one_label=True,retained_set_false_rejection_bound=False,
        nominal_star_conditional_size_not_proved_by_C=True)
