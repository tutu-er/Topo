"""Audit-record regressions; no production numerical behavior is changed."""
from pathlib import Path
import sys
from time import perf_counter

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import run_advantage_stress as study
from rooted_ablation_support import BoundedPath


@pytest.mark.parametrize('initial,raw,admissible', [
    (((0,), (1,), (2,), (3,)), 1., 1.),
    (((0, 1),), 1., 1.),
    (((0, 2),), 1., 0.),
    (((0, 1, 2),), 1., .5),
])
def test_raw_candidate_coverage_is_distinct_from_initial_laminar_compatibility(initial, raw, admissible):
    labels = [10, 20, 30, 40]
    members = {label: frozenset([label]) for label in labels}
    truth = {frozenset([10, 20]), frozenset([30, 40])}
    result = study.candidate_coverage(((0, 1), (2, 3)), initial, labels, members, truth)
    assert result['candidate_raw_truth_recall'] == raw
    assert result['candidate_admissible_truth_recall'] == admissible
    assert result['candidate_truth_recall'] == admissible


def test_coverage_expands_contracted_singletons_before_counting():
    labels = [900, 901, 902]
    members = {900: frozenset([10, 20]), 901: frozenset([30]), 902: frozenset([40])}
    truth = {frozenset([10, 20]), frozenset([30, 40])}
    result = study.candidate_coverage(((1, 2),), ((0,), (1,), (2,)), labels, members, truth)
    assert set(result.values()) == {1.}


@pytest.mark.parametrize('pool,truth,expected', [
    (None, {frozenset([10, 20])}, None),
    ((), {frozenset([10, 20])}, 0.),
    ((), set(), 1.),
])
def test_coverage_handles_unrestricted_empty_and_star_domains(pool, truth, expected):
    result = study.candidate_coverage(pool, ((0,), (1,)), [10, 20],
                                      {10: frozenset([10]), 20: frozenset([20])}, truth)
    assert all(value == expected for value in result.values())


@pytest.mark.parametrize('enumerate_pool,required,forbidden', [
    (True, 'fixed-support LP', 'native MILP'),
    (False, 'native MILP', 'all certified fixed-support LP'),
])
def test_certificate_scope_identifies_the_actual_solver_adapter(enumerate_pool, required, forbidden):
    adapter = BoundedPath(enumerate_pool=enumerate_pool)
    adapter.started = perf_counter()
    scope = adapter.records()['certificate_scope']
    assert required in scope
    assert forbidden not in scope
    assert 'forward path remains greedy' in scope


def test_no_available_refit_records_explicit_hybrid_failure(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('injected fixed-model failure')

    monkeypatch.setattr(study, 'fixed_fit', failed)
    monkeypatch.setattr(study.BoundedPath, 'fit', failed)
    monkeypatch.setattr(study, '_select_boundary_blocks', lambda *args, **kwargs: (set(), {}, [], 0., 1.))
    job = dict(tier='synthetic', shape='balanced', n=2, repeat=0, stress='reference', seed=90210,
               cluster_id='failure_case', job_id='failure_case', output=str(tmp_path), samples=8,
               bootstrap=1, search_seconds=.1, solve_seconds=.1)
    rows = study.measurement_job(job)
    assert {row['method'] for row in rows} == set(study.MEASUREMENT_METHODS)
    assert len(rows) == len(study.MEASUREMENT_METHODS)
    hybrid = next(row for row in rows if row['method'] == 'validation_selected_hybrid')
    assert hybrid['status'] == 'failed'
    assert hybrid['clade_f1'] is None
    assert hybrid['error'] == 'no available fitted candidate model'
    assert 'selected_method' not in hybrid
    assert 'test_mae' not in hybrid and 'validation_mae' not in hybrid
