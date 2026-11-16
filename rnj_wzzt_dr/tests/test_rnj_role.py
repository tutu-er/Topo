"""RNJ connection seeds and one optional pre-fitted E reference; no tree contest."""
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace
import inspect
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ..observations import Observation
from ..identification import algorithm, trees
from ..identification.optimization import QPFitter, criterion, errors, response_matrices


def fixture(n=4):
    supports = [list(C) for C in trees.canonical([*trees.basis(n), (0, 1)])]
    target = dict(supports=supports,
                  r=[.4 if C == [0, 1] else .1 for C in supports],
                  x=[.2 if C == [0, 1] else .05 for C in supports])
    design = np.eye(2*n)
    R, X = response_matrices(target, n)
    data = Observation(design[:, :n], design[:, n:], design[:, :n]@R.T+design[:, n:]@X.T,
                       tuple(range(101, 101+n)))
    fitter = QPFitter(data, reference_mse=.1, qp_seconds=10.)
    current = fitter.fit(trees.basis(n))
    config = algorithm.SearchConfig(total_seconds=30., per_mip_seconds=5., qp_seconds=10.,
                                    output_reserve_seconds=3., internal_floor=0.)
    return data, fitter, current, target, config


class RNJRoleTests(unittest.TestCase):
    def test_generated_fifteen_families_are_pool_provenance_not_warm_trees(self):
        data, _, _, _, _ = fixture()
        with patch.object(trees, 'fit_sensitivity', return_value=(np.eye(4), np.eye(4), {})) as fit:
            guidance = trees.rnj_guidance(data)
        fit.assert_called_once_with(data)
        self.assertEqual(len(guidance['pool_families']), 15)
        self.assertEqual(len(guidance['family_origins']), 15)
        self.assertNotIn('warm_families', guidance)
        self.assertNotIn('initial', guidance)
        self.assertEqual(guidance['pool'], trees.canonical(C for f in guidance['pool_families'] for C in f))
        reference_index = guidance['family_origins'].index(dict(weight=.75, tolerance=.16))
        self.assertEqual(guidance['reference_family'], guidance['pool_families'][reference_index])

    def test_W_retains_beneficial_pool_pretrial_without_native_solution(self):
        _, fitter, current, _, config = fixture()
        pool = trees.canonical([*trees.basis(4), (0, 1), (0, 2), (2, 3)])
        solver = SimpleNamespace(extend=Mock(return_value=(None, dict(status=9))))
        result, event = algorithm._forward(current, fitter, solver, config, pool,
            perf_counter()+60., reference_family=((2, 3),))
        args = solver.extend.call_args.args
        self.assertIs(args[0], current); self.assertIs(args[2], event['pretrial_best'])
        self.assertEqual(args[1], (0, 1))
        self.assertIs(result, event['pretrial_best'])
        self.assertEqual(event['initial_connection_source'], 'RNJ_pool_single_support_QP')
        self.assertEqual(event['initial_new_active'], 1)
        self.assertGreater(event['initial_new_r'], 0.)
        self.assertGreater(event['pretrial_QP_count'], 0)
        self.assertEqual(event['pretrial_candidates_attempted'], 3)
        self.assertTrue(event['accepted']); self.assertFalse(event['has_native_solution'])
        self.assertFalse(event['native_improved_initial'])
        self.assertEqual(event['returned_point_source'], 'pool_pretrial_QP')
        self.assertGreater(event['current_to_initial_gain'], 0.)
        self.assertIsNone(event['closed_point']); self.assertIsNone(event['native_returned_score'])
        self.assertTrue(event['unrestricted_support_domain'])

    def test_W_structural_fallback_is_available_without_RNJ(self):
        _, fitter, current, _, config = fixture()
        solver = SimpleNamespace(extend=Mock(return_value=(None, dict(status=9))))
        with patch.object(fitter, 'fit', side_effect=AssertionError('No pool trial')):
            _, event = algorithm._forward(current, fitter, solver, config, trees.basis(4), perf_counter()+60.)
        candidate = tuple(event['initial_connection'])
        self.assertNotIn(candidate, trees.basis(4))
        self.assertEqual(event['initial_connection_source'], 'structural_connection')
        self.assertTrue(all(trees.compatible(candidate, C) for C in current['supports']))

    def test_W_only_closes_a_returned_native_point_and_audits_scores(self):
        data, fitter, current, target, config = fixture()
        solver = SimpleNamespace(extend=Mock(return_value=(deepcopy(target), dict(status=2))))
        before = len(fitter.records)
        result, event = algorithm._forward(current, fitter, solver, config,
                                           trees.canonical(target['supports']), perf_counter()+60.)
        self.assertEqual(len(fitter.records)-before, 1)
        self.assertTrue(event['has_native_solution']); self.assertTrue(event['accepted'])
        self.assertFalse(event['native_improved_initial'])
        self.assertAlmostEqual(event['current_score'], criterion(current, data.n, config.alpha))
        self.assertAlmostEqual(event['native_returned_score'],
            errors(data, target)['mse']/fitter.reference_mse+config.alpha)
        self.assertAlmostEqual(event['closed_score'], criterion(event['closed_point'], data.n, config.alpha))
        self.assertAlmostEqual(criterion(result, data.n, config.alpha), event['closed_score'])
        self.assertIs(event['initial_point'], event['pretrial_best'])
        self.assertEqual(event['pretrial_QP_count'], 1)

    def test_T_and_E_keep_current_when_native_has_no_solution(self):
        _, fitter, current, target, config = fixture()
        self.assertNotIn('warm_families', inspect.signature(algorithm._reconstruct).parameters)
        for stage in ('T', 'E'):
            solver = SimpleNamespace(select=Mock(return_value=(None, dict(status=9))))
            with self.subTest(stage=stage), patch.object(fitter, 'fit', side_effect=AssertionError('No tree trials')):
                result, event = algorithm._reconstruct(current, fitter, solver, config,
                    pool=target['supports'], keep=trees.basis(4), deadline=perf_counter()+60., stage=stage)
            self.assertIs(solver.select.call_args.args[2], current)
            self.assertIs(result, current); self.assertIs(event['point'], current)
            self.assertEqual(event['initial_source'], 'current')
            self.assertFalse(event['has_native_solution']); self.assertFalse(event['native_improved_initial'])
            self.assertFalse(event['accepted']); self.assertEqual(event['pretrial_QP_count'], 0)
            self.assertEqual(event['pool'], trees.canonical(target['supports']))

    def test_expired_W_E_budget_never_uses_a_reference_tree_as_result(self):
        _, fitter, current, target, config = fixture()
        solver = SimpleNamespace(select=Mock(side_effect=AssertionError('Expired')),
                                 extend=Mock(side_effect=AssertionError('Expired')))
        with patch.object(fitter, 'fit', side_effect=AssertionError('Expired')):
            w, we = algorithm._forward(current, fitter, solver, config,
                                      trees.canonical(target['supports']), -1.)
            e, ee = algorithm._reconstruct(current, fitter, solver, config,
                pool=target['supports'], keep=trees.basis(4), deadline=-1., stage='E', reference_point=target)
        self.assertIs(w, current); self.assertIs(e, current)
        for event in (we, ee):
            self.assertFalse(event['solver']['solve_started'])
            self.assertFalse(event['has_native_solution'])
            self.assertFalse(event['accepted'])

    def test_fit_never_reads_legacy_warm_families_or_outputs_reference_without_native(self):
        data, _, _, target, config = fixture()
        class Poison:
            def __iter__(self): raise AssertionError('Legacy warm families must not be read')
        guidance = dict(R=np.eye(4), X=np.eye(4), pool=trees.basis(4),
                        reference_family=trees.canonical(target['supports']), warm_families=Poison())
        common = dict(guidance=guidance, reference_mse=.1, reference_point=target,
            seconds=0., n=4, labels=data.labels, training_sha256=algorithm._training_fingerprint(data))
        calls = []
        class NoNative:
            def __init__(self, *args): self.records = []
            def extend(self, current, candidate, warm, **kwargs):
                calls.append(('W', current, warm, candidate)); return None, dict(status=9)
            def select(self, pool, keep, warm, **kwargs):
                calls.append(('E', None, warm, pool)); return None, dict(status=9)
            def close(self): pass
        with patch.object(algorithm, 'StructuralSolver', NoNative):
            result = algorithm.fit(data, data, method='101', config=config, common=common)
        self.assertEqual([r[0] for r in calls], ['W', 'E'])
        self.assertIs(calls[0][1], calls[0][2])
        self.assertEqual(calls[1][2]['supports'], target['supports'])
        self.assertEqual(result['events'][-1]['initial_source'], 'single_RNJ_reference')
        self.assertEqual(len(result['path']), 1)
        self.assertEqual(len(result['frozen_candidates']), 2)  # original point and star remain
        self.assertEqual(len(result['QP']), 2)  # no RNJ-family QP trials
        self.assertFalse(result['complete_RNJ_output_fallback'])
        for row in result['frozen_candidates']:
            self.assertNotIn([0, 1], row['point']['supports'])

    def test_single_reference_is_rescored_not_refitted_or_mutated(self):
        data, fitter, current, target, config = fixture()
        reference = dict(target, objective=-100., mse=999., reference_mse=999.)
        before = deepcopy(reference)
        solver = SimpleNamespace(select=Mock(return_value=(None, dict(status=9))))
        with patch.object(fitter, 'fit', side_effect=AssertionError('Must reuse common point')):
            result, event = algorithm._reconstruct(current, fitter, solver, config,
                pool=target['supports'], keep=trees.basis(4), deadline=perf_counter()+60.,
                stage='E', reference_point=reference)
        seed = solver.select.call_args.args[2]
        self.assertEqual(reference, before)
        self.assertAlmostEqual(seed['mse'], errors(data, reference)['mse'])
        self.assertAlmostEqual(seed['objective'], seed['mse']/fitter.reference_mse)
        self.assertEqual(seed['reference_mse'], fitter.reference_mse)
        self.assertIs(event['initial_point'], seed)
        self.assertEqual(event['initial_source'], 'single_RNJ_reference')
        self.assertEqual(event['reference_refit_count'], 0)
        self.assertAlmostEqual(event['current_to_initial_gain'], event['current_score']-event['initial_score'])
        self.assertGreater(event['current_to_initial_gain'], 0.)
        self.assertIs(result, current); self.assertFalse(event['accepted'])
        self.assertIsNone(event['closed_point'])

    def test_T_never_uses_the_E_reference(self):
        _, fitter, current, target, config = fixture()
        solver = SimpleNamespace(select=Mock(return_value=(None, dict(status=9))))
        with patch.object(fitter, 'fit', side_effect=AssertionError('No QP trial')):
            _, event = algorithm._reconstruct(current, fitter, solver, config,
                pool=target['supports'], keep=trees.basis(4), deadline=perf_counter()+60.,
                stage='T', reference_point=target)
        self.assertIs(solver.select.call_args.args[2], current)
        self.assertFalse(event['reference_considered'])
        self.assertEqual(event['initial_source'], 'current')

    def test_reference_ineligible_or_not_better_does_not_change_pool_or_initial(self):
        _, fitter, current, target, config = fixture()
        missing = deepcopy(target)
        missing['supports'].pop(0); missing['r'].pop(0); missing['x'].pop(0)
        outside = deepcopy(target); outside['supports'][4] = [2, 3]
        crossing = deepcopy(target)
        crossing['supports'].append([1, 2]); crossing['r'].append(.1); crossing['x'].append(.1)
        unbounded = deepcopy(target); unbounded['x'][0] = config.response_bound+.01
        malformed = deepcopy(target); malformed['supports'][0] = [True]
        for name, candidate in (('missing', missing), ('outside', outside), ('crossing', crossing),
                                ('bounds', unbounded), ('coordinate', malformed), ('not_better', current)):
            solver = SimpleNamespace(select=Mock(return_value=(None, dict(status=9))))
            with self.subTest(name=name), patch.object(fitter, 'fit', side_effect=AssertionError('No QP trial')):
                _, event = algorithm._reconstruct(current, fitter, solver, config,
                    pool=target['supports'], keep=trees.basis(4), deadline=perf_counter()+60.,
                    stage='E', reference_point=candidate)
            self.assertIs(solver.select.call_args.args[2], current)
            self.assertEqual(event['pool'], trees.canonical(target['supports']))
            self.assertFalse(event['reference_selected'])
            self.assertEqual(event['initial_source'], 'current')
            self.assertEqual(event['reference_eligible'], name == 'not_better')

    def test_native_reference_incumbent_gain_is_separate_from_search_gain(self):
        _, fitter, current, target, config = fixture()
        solver = SimpleNamespace(select=Mock(return_value=(deepcopy(target), dict(status=9))))
        before = len(fitter.records)
        result, event = algorithm._reconstruct(current, fitter, solver, config,
            pool=target['supports'], keep=trees.basis(4), deadline=perf_counter()+60.,
            stage='E', reference_point=target)
        self.assertEqual(len(fitter.records)-before, 1)  # only returned-family closure
        self.assertTrue(event['has_native_solution']); self.assertTrue(event['accepted'])
        self.assertFalse(event['native_improved_initial'])
        self.assertAlmostEqual(event['native_returned_score'], event['initial_score'])
        self.assertGreater(event['current_to_initial_gain'], 0.)
        self.assertIs(result, event['closed_point'])

    def test_R_off_and_missing_reference_source_keep_current_initial(self):
        data, _, _, target, config = fixture()
        guidance = dict(R=np.eye(4), X=np.eye(4), pool=trees.basis(4),
                        reference_family=trees.canonical(target['supports']))
        common = dict(guidance=guidance, reference_mse=.1, reference_point=target,
            seconds=0., n=4, labels=data.labels, training_sha256=algorithm._training_fingerprint(data))
        calls = []
        class NoNative:
            def __init__(self, *args): self.records = []
            def extend(self, current, candidate, warm, **kwargs): return None, dict(status=9)
            def select(self, pool, keep, warm, **kwargs):
                calls.append(warm); return None, dict(status=9)
            def close(self): pass
        for method, source in (('001', common), ('101', {k: v for k, v in common.items() if k != 'reference_point'})):
            with self.subTest(method=method), patch.object(algorithm, 'StructuralSolver', NoNative):
                result = algorithm.fit(data, data, method=method, config=config, common=source)
            event = result['events'][-1]
            self.assertEqual(event['initial_source'], 'current')
            self.assertFalse(event['reference_considered'])
            self.assertIs(calls[-1], result['path'][0])

    def test_W_pretrial_timeout_retains_earlier_verified_point_and_original_budget(self):
        _, fitter, current, _, config = fixture()
        now = [0.]
        original_fit = fitter.fit
        attempts = []
        def bounded_fit(family):
            attempts.append((family, fitter.qp_seconds))
            if (2, 3) in family:
                now[0] = 1.1
                fitter.records.append(dict(family=family, soft_cap_exceeded=True, seconds=.9))
                raise TimeoutError('fixture second QP soft timeout')
            point = original_fit(family)
            now[0] = .2
            return point
        solver = SimpleNamespace(extend=Mock(side_effect=AssertionError('Budget exhausted')))
        pool = trees.canonical([*trees.basis(4), (0, 1), (2, 3)])
        with patch.object(algorithm, 'perf_counter', side_effect=lambda: now[0]), patch.object(fitter, 'fit', side_effect=bounded_fit):
            result, event = algorithm._forward(current, fitter, solver, config, pool, 1.)
        self.assertEqual([limit for _, limit in attempts], [1., .8])
        self.assertEqual(fitter.qp_seconds, 10.)
        self.assertTrue(event['accepted']); self.assertFalse(event['has_native_solution'])
        self.assertIn([0, 1], result['supports'])
        self.assertEqual(event['pretrial_stop_reason'], 'qp_soft_timeout')
        self.assertEqual(event['pretrial_candidates'][-1]['status'], 'timeout')
        self.assertEqual(event['pretrial_QP_count'], 2)
        self.assertTrue(event['pretrial_deadline_exceeded'])
        self.assertEqual(event['returned_point_source'], 'pool_pretrial_QP')

    def test_W_expired_pretrial_budget_does_not_fit_or_search(self):
        _, fitter, current, target, config = fixture()
        solver = SimpleNamespace(extend=Mock(side_effect=AssertionError('Budget exhausted')))
        with patch.object(fitter, 'fit', side_effect=AssertionError('No budget')):
            result, event = algorithm._forward(current, fitter, solver, config,
                trees.canonical(target['supports']), -1.)
        self.assertIs(result, current)
        self.assertEqual(event['pretrial_QP_count'], 0)
        self.assertEqual(event['pretrial_candidates_attempted'], 0)
        self.assertEqual(event['pretrial_stop_reason'], 'deadline')


if __name__ == '__main__':
    unittest.main()
