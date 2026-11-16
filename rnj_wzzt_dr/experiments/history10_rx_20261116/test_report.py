"""Small independent audit checks; do not launch study fits or AC."""
import json
import numpy as np
import pytest

from .audit_report import Inputs, fixed_qp_check, reconstruct_law, response, loss, summarize_main, summarize_offline, topology, ref, check_offline_coverage


def test_response_prediction_and_full_vs_internal_support_are_distinct():
    point = dict(supports=[[0], [1], [0, 1]], r=[.1, .2, .3], x=[.01, .02, 0.])
    h = response(point, 2)
    np.testing.assert_allclose(h[:, :2], [[.4, .3], [.3, .5]])
    top = topology(point, [[0], [1]], 2, 1e-9)
    assert top['internal_exact'] and not top['full_exact']
    assert top['full_extra'] == 1


def test_offline_qp_bound_recomputed_on_original_coefficient_box():
    p = np.array([[1.], [2.]])
    obs = dict(P=p, Q=np.zeros_like(p), Y=3*p)
    point = dict(supports=[[0]], r=[2.], x=[0.], reference_mse=1., mse=2.5)
    record = dict(coefficient_bound=2., objective=2.5, lower_bound=2.5, certificate_gap=0.)
    checked = fixed_qp_check(point, obs, record)
    assert checked['lower_bound'] == pytest.approx(2.5)
    bad = dict(record, lower_bound=3.)
    with pytest.raises(AssertionError): fixed_qp_check(point, obs, bad)


def test_laws_distinguish_input_noise_from_voltage_noise():
    clean = dict(P=np.array([[.1], [.2]]), Q=np.array([[.03], [.04]]),
                 V=np.array([[1.], [1.01]]), root=np.array([1.02, 1.02]))
    noise = {key: np.ones_like(value) for key, value in clean.items()}
    star = np.array([[.1, .2]])
    linear = reconstruct_law(clean, noise, 'cleanlinear', star)
    v_only = reconstruct_law(clean, noise, 'Vonly', star)
    all_ = reconstruct_law(clean, noise, 'all', star)
    np.testing.assert_array_equal(linear['P'], clean['P'])
    np.testing.assert_array_equal(v_only['P'], clean['P'])
    np.testing.assert_allclose(all_['P'], clean['P']*1.005)
    np.testing.assert_array_equal(all_['Y'], v_only['Y'])
    assert loss(star, linear)['mse'] < 1e-30


def test_source_mutation_is_rejected_including_after_cached_read(tmp_path):
    path = tmp_path/'sealed.json'; path.write_text('{"a":1}', encoding='utf-8')
    source = ref(path); reader = Inputs()
    assert reader.json(source) == {'a': 1}
    path.write_text('{"a":2}', encoding='utf-8')
    with pytest.raises(AssertionError): reader.final_check()
    with pytest.raises(AssertionError): Inputs().json(source)


def test_failures_and_missing_tasks_keep_planned_denominator():
    tasks = [dict(id=f'a{days}{h}', case='a', train_days=days, history_repeat=h)
             for days in (2, 8) for h in range(4)]
    rows = [dict(**tasks[0], status='failed', seconds=2., error='kept')]
    result = summarize_main(tasks, rows)
    assert result['planned'] == 8
    assert result['statuses'] == {'failed': 1, 'missing': 7}
    assert all(r['planned'] == 4 and r['checked_complete'] == 0 for r in result['groups'])
    assert all(r['status'] == 'unavailable' for r in result['paired'])
    json.dumps(result, allow_nan=False)


def test_deterministic_clean_repeats_are_not_independent_evidence():
    rows = []
    for h in range(4):
        rows.append(dict(case='a', support_role='oracle', support_aliases=['oracle'], law='cleanAC', train_days=8,
            history_repeat=h, status='complete', metrics={'R': {'relative_error': .1}, 'X': {'relative_error': .2}},
            clean_AC_new_MSE=1e-6, design={'X_given_R': {'residual_energy_fraction': .01}},
            independent_QP={'certificate_gap': 0.}))
    group = summarize_offline(rows)[0]
    assert group['completed'] == 4
    assert group['independent_noise_repetitions'] == 0
    assert group['deterministic_repeats']


def test_offline_denominator_and_aliases_are_independent_of_results():
    families = {'a': {((0,), (1,)): ['oracle', 'old_correct']}}
    rows = [dict(case='a', history_repeat=h, train_days=2, law='all',
                 support_role='oracle', support_aliases=['oracle', 'old_correct'], status='complete',
                 point={'supports': [[0], [1]]}) for h in range(4)]
    assert check_offline_coverage(rows, families, [2], ['all']) == 4
    with pytest.raises(AssertionError): check_offline_coverage(rows[:-1], families, [2], ['all'])
    with pytest.raises(AssertionError): check_offline_coverage(rows+[rows[0]], families, [2], ['all'])
    wrong = [dict(row) for row in rows]; wrong[0]['support_aliases'] = ['oracle']
    with pytest.raises(AssertionError): check_offline_coverage(wrong, families, [2], ['all'])
    wrong = [dict(row) for row in rows]; wrong[0]['point'] = {'supports': [[0, 1]]}
    with pytest.raises(AssertionError): check_offline_coverage(wrong, families, [2], ['all'])
