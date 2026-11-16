"""Small fixed-support baseline and report checks; no MIQP, AC or private data."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from ...history import freeze_history
from ...observations import Observation
from ...identification import algorithm
from ...identification.optimization import criterion
from ..rnj_pool_only_20261116.compare import audit_native
from .run import direct_baseline
from .report import origins, summarize


def synthetic_observations():
    """An arbitrary reproducible response fixture, with no physical truth input."""
    rng = np.random.default_rng(20261116)
    n = 4
    grouping = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    R = .2*np.eye(n)+.16*grouping.T@grouping+.02*np.ones((n, n))
    X = .1*np.eye(n)+.07*grouping.T@grouping+.01*np.ones((n, n))
    def draw(count):
        p, q = rng.uniform(.02, .2, (count, n)), rng.uniform(-.05, .08, (count, n))
        y = p@R.T+q@X.T+rng.normal(0., .001, (count, n))
        return Observation(p, q, y, (101, 102, 103, 104))
    return draw(32), draw(16)


@pytest.fixture(scope='module')
def baselines():
    train, validation = synthetic_observations()
    config = algorithm.SearchConfig(total_seconds=30., per_mip_seconds=5., qp_seconds=10.,
                                    output_reserve_seconds=3., internal_floor=.08)
    with patch.object(algorithm, 'StructuralSolver', side_effect=AssertionError('No structural solver in direct baselines')):
        common = algorithm.prepare_common(train, config)
        native = {method:direct_baseline(algorithm, train, validation, common, config, method)
                  for method in ('RNJ1', 'RNJ15')}
    return train, validation, common, config, native


def test_RNJ_baselines_pass_same_native_freeze_and_output_floor(baselines):
    _, validation, _, config, native = baselines
    for method, result in native.items():
        frozen = freeze_history(result, validation)
        assert frozen.current == result['selected']['index']
        assert frozen.H.shape[0] == len(result['frozen_candidates'])
        assert result['output_contraction']['threshold'] == config.internal_floor
        assert result['output_contraction']['candidate_count'] == len(result['frozen_candidates'])
        assert result['selection_metric'] == 'validation_mse'
        assert result['events'] == [] and result['MIQP'] == []
        assert result['truth_used'] is False and result['test_used'] is False
        assert result['method'] == method


def test_RNJ15_contains_same_processed_reference_and_cannot_worsen_validation(baselines):
    _, _, common, config, native = baselines
    one, many = native['RNJ1'], native['RNJ15']
    families = list(dict.fromkeys(algorithm.trees.canonical(family)
                                 for family in common['guidance']['pool_families']))
    reference = algorithm.trees.canonical(common['guidance']['reference_family'])
    index = 1+families.index(reference)  # common basis comes first in both banks
    for key in ('supports', 'r', 'x'):
        assert many['frozen_candidates'][index]['point'][key] == one['frozen_candidates'][1]['point'][key]
    numerical_tie = common['reference_mse']*config.validation_tie
    assert many['selected']['validation_mse'] <= one['selected']['validation_mse']+numerical_tie+1e-15
    assert many['raw_RNJ_family_count'] == 15
    assert one['raw_RNJ_family_count'] == 1


def test_both_methods_keep_basis_star_and_one_origin_per_candidate(baselines):
    train, _, _, _, native = baselines
    basis = algorithm.trees.basis(train.n)
    star = algorithm.trees.canonical((i,) for i in range(train.n))
    for result in native.values():
        raw = result['uncontracted_output_path']
        assert algorithm.trees.canonical(raw[0]['supports']) == basis
        assert algorithm.trees.canonical(raw[-1]['supports']) == star
        assert result['candidate_origins'][0] == 'basis'
        assert result['candidate_origins'][-1] == 'singleton_protection'
        assert len(origins(result)) == len(raw) == len(result['frozen_candidates'])
        assert len(raw) == result['unique_RNJ_family_count']+2
        assert algorithm.trees.canonical(result['frozen_candidates'][-1]['point']['supports']) == star


def complete_row(method, *, events=(), seconds=3., full=False):
    selected = dict(internal_exact=True, full_exact=full)
    return dict(method=method, status='complete', total_cap_exceeded=False,
        raw=dict(topology=selected), audit=dict(seconds=seconds, events=list(events), phases={
            'validation':dict(topology=selected), 'cumulative':dict(topology=selected)}))


def test_summary_retains_failed_and_missing_planned_denominators():
    rows = [complete_row('RNJ1'), dict(method='RNJ1', status='failed'),
            dict(method='RNJ1', status='missing'), dict(method='RNJ15', status='audit_failed')]
    summary = summarize(rows)
    assert summary['RNJ1']['planned'] == 3
    assert summary['RNJ1']['completed'] == 1
    assert summary['RNJ1']['failed'] == 2
    assert summary['RNJ1']['internal'] == 1 and summary['RNJ1']['full'] == 0
    assert summary['RNJ1']['mean_method_seconds'] == 3.
    assert summary['RNJ15']['planned'] == 1 and summary['RNJ15']['failed'] == 1
    assert summary['RNJ15']['mean_method_seconds'] is None
    assert summary['100']['planned'] == 0 and summary['100']['mean_method_seconds'] is None


def test_unchanged_native_initial_is_not_counted_as_search_improvement(baselines):
    train, validation, _, config, values = baselines
    native = deepcopy(values['RNJ1'])
    current, seed = native['path'][:2]
    before = criterion(current, train.n, config.alpha)
    after = criterion(seed, train.n, config.alpha)
    assert before > after  # positive preparation gain is deliberately present
    native['events'] = [dict(stage='E', accepted=True, before_score=before, current_score=before,
        after_score=after, point=seed, initial_point=seed, initial_score=after,
        initial_source='single_RNJ_reference', current_to_initial_gain=before-after,
        raw_point=deepcopy(seed), native_returned_score=after, closed_point=seed, closed_score=after,
        native_improved_initial=False, solver=dict(solve_started=True, status=9, objective=after,
            fixed_family_qp_objective=after, bound=0., gap=1., seconds=1.))]
    def observed(data):
        return dict(P=data.p, Q=data.q, Y=data.y, labels=list(data.labels))
    audited = audit_native(native, observed(train), observed(validation), observed(validation))
    event = audited['events'][0]
    assert event['current_to_initial_gain'] > 0.
    assert event['native_gain'] == pytest.approx(0., abs=1e-12)
    assert not event['native_improved_initial']
    summary = summarize([complete_row('101', events=audited['events'])])['101']
    assert summary['MIQP_started'] == 1 and summary['MIQP_timeouts'] == 1
    assert summary['native_improved_initial'] == 0
    positive_control = dict(solve_started=True, timed_out=False, native_improved_initial=True)
    mixed = summarize([complete_row('101', events=[*audited['events'], positive_control])])['101']
    assert mixed['native_improved_initial'] == 1
