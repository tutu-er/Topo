"""Small independent arithmetic/failure fixtures; no production imports."""
from copy import deepcopy
from . import ev_systematic_statistics_audit as independent


def example():
    return dict(true_name='a', candidate_miss=False, candidates=[dict(name='a'), dict(name='b')],
        config=dict(training_points=8, validation_points=4, horizon=1), methods=[dict(name='none',
            operation=dict(success=True), power_kw=[[0., 0., 0.]], inference=dict(
                fixed_total=dict(complete=True, losses={'a': 20., 'b': 10.})))])


def test_exact_ranks_and_small_group_infinity():
    assert independent.cutoff(list(range(108)), '.1')['order_rank'] == 99
    assert independent.cutoff(list(range(108)), '.05')['order_rank'] == 104
    assert [independent.cutoff(list(range(n)), '.1')['order_rank'] for n in (16, 18, 74)] == [16, 18, 68]
    assert independent.cutoff([0.], '.1')['infinite']


def test_scores_use_conditional_output_count_and_failure_keeps_all():
    row = example(); value = independent.values(row, 'none', 'fixed_total')
    assert value['R_true'] == 10. and value['Nobs'] == 65 and value['A_true'] == 20. / 65
    row['methods'][0]['operation']['success'] = False
    value = independent.values(row, 'none', 'fixed_total')
    assert value['R_true'] == value['A_true'] == 0.
    assert independent.decide(value, 'R', dict(infinite=False, threshold=0.)) == dict(
        category='failed_keep_all', retained=['a', 'b'], false_reject=False)


def test_missing_candidate_and_empty_dual_set_remain_distinct():
    row = example(); row['true_name'] = 'c'; row['candidate_miss'] = True
    value = independent.values(row, 'none', 'fixed_total')
    assert independent.decide(value, 'R', dict(infinite=False, threshold=0.))['category'] == 'candidate_miss'
    value = independent.values(example(), 'none', 'fixed_total')
    gates = dict(R=dict(infinite=False, threshold=20.), A=dict(infinite=False, threshold=.01))
    result = independent.decide(value, 'dual', gates)
    assert result['retained'] == [] and result['false_reject'] and result['all_models_fail_adequacy']


def test_mismatch_is_retained_as_an_audit_error():
    check = independent.Audit(); check.equal(99, 104, 'tampered rank')
    assert check.checks == 1 and check.errors[0]['check'] == 'tampered rank'
