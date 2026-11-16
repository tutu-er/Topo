"""Selection and failure accounting on tiny arrays; no identification or AC."""
from types import SimpleNamespace
import unittest

import numpy as np

from ...observations import Observation
from . import run


def point(supports, r=None, x=None):
    return dict(supports=supports, r=[0.] * len(supports) if r is None else r,
                x=[0.] * len(supports) if x is None else x)


class RunTests(unittest.TestCase):
    def test_grid_denominators_and_zero_cells(self):
        current = run.grid_for({'source_group': 'history10'})
        broad = run.grid_for({'source_group': 'large96_main'})
        self.assertEqual((len(current), len(broad)), (40, 20))
        self.assertEqual(len({r['id'] for r in current}), 40)
        self.assertEqual(sum(r['threshold'] == 0 for r in current), 4)
        self.assertTrue(all(r['metric'] == 'rx75' for r in broad))

    def test_original_order_ids_and_internal_tie_rule(self):
        obs = Observation(np.ones((2, 3)), np.zeros((2, 3)), np.ones((2, 3)), (1, 2, 3))
        candidates = [dict(source_candidate_id=8, point=point([[0, 1]])),
                      dict(source_candidate_id=3, point=point([[0], [1], [2]]))]
        scored = run.score_bank(candidates, obs, 2., 1e-12)
        self.assertEqual(scored['selection']['selected'], 3)
        self.assertEqual(scored['selection']['eligible'], [8, 3])
        self.assertEqual(scored['selection']['K'], .5)
        self.assertEqual(scored['selection']['epsilon_physical'], 2e-12)
        candidates[1]['point'] = point([[0, 1]])
        self.assertEqual(run.score_bank(candidates, obs, 2., 1e-12)['selection']['selected'], 8)

    def test_cumulative_scoring_uses_cells_not_equal_day_means(self):
        val = Observation(np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), (1,))
        new = Observation(np.ones((3, 1)), np.zeros((3, 1)), np.full((3, 1), 3.), (1,))
        candidates = [dict(source_candidate_id=0, point=point([[0]], [1.])),
                      dict(source_candidate_id=1, point=point([[0]], [3.]))]
        self.assertEqual(run.score_bank(candidates, val, 1., 1e-12)['selection']['selected'], 0)
        scored = run.score_bank(candidates, val, 1., 1e-12, new_day=new)
        self.assertEqual(scored['selection']['selected'], 1)
        self.assertEqual(scored['selection']['scoring_cells'], 4)
        self.assertEqual(scored['selection']['mse'], {0: 3., 1: 1.})

    def test_failed_candidate_does_not_disappear(self):
        candidates = [dict(source_candidate_id=i, point=point([[i]])) for i in range(3)]
        fitter = SimpleNamespace(records=[])
        called = []
        def prune(p, f, threshold, **kwargs):
            called.append(p['supports'][0][0])
            if p['supports'] == [[1]]:
                raise ValueError('fixture candidate failure')
            return p, {'floor_satisfied': True}
        plan = run.prune_bank(candidates, fitter, dict(threshold=.02, metric='rx75', scope='all'),
                              {}, .8, prune_function=prune)
        self.assertEqual(called, [0, 1, 2])
        self.assertEqual(plan['status'], 'failed')
        self.assertEqual(plan['failed_candidate_ids'], [1])
        self.assertEqual([r['source_candidate_id'] for r in plan['candidates']], [0, 1, 2])
        self.assertEqual(run.evaluate_bank(plan, None, 1., 1e-12, 1e-9, {})['status'],
                         'unavailable_failed_candidate_bank')

    def test_zero_pruning_preserves_source_point_and_ids(self):
        candidates = [dict(source_candidate_id=11, point=point([[0], [0, 1]], [.1, 0.]))]
        fitter = SimpleNamespace(records=[])
        def unchanged(p, *args, **kwargs):
            return p, {'threshold_zero': True}
        plan = run.prune_bank(candidates, fitter, dict(threshold=0., metric='ohm', scope='all'),
                              {}, .8, prune_function=unchanged)
        self.assertIs(plan['candidates'][0]['point'], candidates[0]['point'])
        self.assertEqual(plan['candidates'][0]['source_candidate_id'], 11)

    def test_calibration_failure_preserves_zero_and_physical_grid(self):
        candidates = [dict(source_candidate_id=0, point=point([[0]], [.1]))]
        fitter = SimpleNamespace(records=[])
        calls = []
        def unchanged(p, *args, **kwargs):
            calls.append(kwargs['metric'])
            return p, {'floor_satisfied': True}
        for metric, threshold, expected in [('rx75', .01, 'failed'),
                                             ('rx75', 0., 'complete'),
                                             ('ohm', .01, 'complete')]:
            plan = run.grid_bank(candidates, fitter,
                dict(metric=metric, threshold=threshold, scope='all'), None,
                {'error': 'fixture calibration failure'}, .8, prune_function=unchanged)
            self.assertEqual(plan['status'], expected)
        self.assertEqual(calls, ['rx75', 'ohm'])

    def test_full_support_and_internal_recovery_are_separate(self):
        p = point([[0], [1], [2], [0, 1], [0, 1, 2]], [.1] * 5)
        truth = [[0], [1], [2], [0, 1]]
        t = run.topology(p, truth, 3, 1e-9)
        self.assertTrue(t['internal_exact'])
        self.assertFalse(t['full_exact'])
        self.assertEqual(t['full_extra'], 1)
        p['r'][-1] = 1e-9
        self.assertTrue(run.topology(p, truth, 3, 1e-9)['full_exact'])

    def test_history_only_sources_have_no_fabricated_new_day(self):
        obs = Observation(np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), (1,))
        candidates = [dict(source_candidate_id=2, point=point([[0]], [1.]))]
        plan = dict(status='complete', candidates=candidates,
                    historical_scoring=run.score_bank(candidates, obs, 1., 1e-12))
        result = run.evaluate_bank(plan, obs, 1., 1e-12, 1e-9,
                                   dict(truth_supports=[[0]], new_day=None))
        self.assertFalse(result['independent_new_day_available'])
        self.assertIsNone(result['cumulative'])
        self.assertTrue(result['history_selected_topology']['full_exact'])


if __name__ == '__main__':
    unittest.main()
