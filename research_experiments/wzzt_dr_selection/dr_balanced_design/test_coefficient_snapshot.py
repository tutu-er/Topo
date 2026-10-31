"""Audit metadata regression with a fixed numerical solver receipt.

No optimizer, network, or physical data is used. The fixture includes a positive
weak block deliberately removed by the unchanged support-pruning rule.
"""
from types import SimpleNamespace
import unittest
import numpy as np
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, search
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract


class CoefficientSnapshotTests(unittest.TestCase):
    def fixture(self):
        lp = PoolLP.__new__(PoolLP)
        lp.pool = ((0,), (1,), (0, 1)); lp.lookup = {s:i for i,s in enumerate(lp.pool)}
        lp.bound = 2.; lp.preprocess = 'raw'; lp.cache = {}; lp.solves = []
        lp.gp = SimpleNamespace(GRB=SimpleNamespace(OPTIMAL=2))
        # Full requested support also contains a weak R-only parent; X is zero.
        values = np.array([.1, .2, 5e-10, 0., 0., 0.])
        lp.features = np.array([[1., 0., 1., 0., 0., 0.], [0., 1., 1., 0., 0., 0.]])
        lp.y = lp.features @ values
        lp.weights = SimpleNamespace(X=values)
        lp.model = SimpleNamespace(Status=2, ObjVal=0., ConstrVio=0., BoundVio=0., SolCount=1,
                                   optimize=lambda:None)
        return lp

    def test_pruned_positive_parent_is_saved_without_changing_old_output(self):
        lp = self.fixture()
        point = lp.fit(lp.pool)
        self.assertEqual(point['supports'], [[0], [1]])
        self.assertEqual(point['r'], [.1, .2])
        self.assertEqual(point['x'], [0., 0.])
        self.assertEqual(point['objective'], 0.)
        self.assertEqual(point['full_fit']['supports'], [[0], [1], [0, 1]])
        self.assertEqual(point['full_fit']['r'][-1], 5e-10)
        self.assertEqual(point['full_fit']['x'][-1], 0.)
        self.assertEqual(point['full_fit']['objective'], 0.)
        full = np.array(point['full_fit']['r']+point['full_fit']['x'])
        np.testing.assert_array_equal(lp.features @ full, lp.y)
        # Reconstructing only retained coefficients loses the weak response.
        retained = np.array([.1, .2, 0., 0., 0., 0.])
        self.assertGreater(float(abs(lp.features @ retained-lp.y).max()), 0.)

    def test_complete_receipt_survives_cache_and_output_contraction(self):
        lp = self.fixture(); point = lp.fit(lp.pool)
        self.assertIs(lp.fit(lp.pool), point)
        self.assertEqual(len(lp.solves), 1)
        cut = contract(point, dict(a=1., b=1., median_root_depth=1.))
        self.assertEqual(cut['supports'], [[0], [1]])
        self.assertEqual(cut['full_fit']['r'][-1], 5e-10)

    def test_exact_zero_singleton_remains_an_output_slot_not_a_positive_block(self):
        lp = self.fixture(); lp.weights.X[0] = 0.; lp.y = lp.features @ lp.weights.X
        point = lp.fit(lp.pool)
        self.assertIn([0], point['supports'])
        self.assertEqual(point['full_fit']['r'][0], 0.)
        self.assertEqual(point['full_fit']['x'][0], 0.)

    def test_preprune_objective_is_not_relabelled_as_retained_model_loss(self):
        lp = self.fixture(); point = lp.fit(lp.pool)
        after_pruning = lp.features @ np.array([*point['r'], 0., *point['x'], 0.])
        self.assertGreater(float(np.mean(abs(after_pruning-lp.y))), point['objective'])
        self.assertEqual(point['objective'], point['full_fit']['objective'])

    def test_mocked_search_keeps_same_acceptance_and_numerical_pruning(self):
        lp = self.fixture()
        path = search(lp, lp.pool, seconds=1., repair=True, max_steps=2)
        self.assertEqual(path['reason'], 'pool_local_optimum')
        self.assertEqual(len(path['path']), 1)
        self.assertEqual(path['path'][0]['supports'], [[0], [1]])
        self.assertEqual(path['decisions'][0]['gain'], 0.)
        self.assertEqual(path['path'][0]['full_fit']['r'][-1], 5e-10)


if __name__ == '__main__':
    unittest.main()
