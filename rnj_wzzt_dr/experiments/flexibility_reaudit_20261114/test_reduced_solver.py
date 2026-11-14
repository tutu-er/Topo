"""Algebraic equivalence tests only; no optimizer or AC invocation."""
import unittest
from itertools import product
from unittest.mock import patch
import numpy as np

from .reduced_solver import BUFFER, pair_reduction, plan
from ..absolute_identification_20261114.solver import SymmetricRecovery
from ..reactive_flex_20261114.domain import PQIncrementCase
from ..reactive_flex_20261114.test_domain import fixture


class ReductionTests(unittest.TestCase):
    def check(self, c, z=1.8339146358159146):
        tie = 1e-7; h = c+tie+BUFFER
        reduced = pair_reduction(h, z, tie)
        J = np.r_[0., np.geomspace(1e-12, 100., 4000)]
        original = h+J-z*np.sqrt(2*J)-tie-BUFFER
        if reduced['kind'] == 'all_J':
            equivalent = np.ones(len(J), bool)
        elif reduced['kind'] == 'lower_bound':
            equivalent = J >= reduced['lower_J']
        else:
            equivalent = (J <= reduced['upper_small_branch_J']) | (J >= reduced['lower_J'])
        away = np.abs(original) > 1e-10
        np.testing.assert_array_equal(equivalent[away], original[away] >= 0.)
        # Taking s=sqrt(J) gives the smallest allowable RHS in the old model.
        s = np.sqrt(J)
        np.testing.assert_allclose(J, s*s, atol=1e-12)
        np.testing.assert_allclose(h+J-z*np.sqrt(2)*s-tie-BUFFER, original, atol=1e-12)
        return reduced

    def test_negative_offset_single_high_branch(self):
        reduced = self.check(-.002807796732654111)
        self.assertEqual(reduced['kind'], 'lower_bound')
        self.assertAlmostEqual(reduced['lower_J'], 6.732100205320604, places=12)

    def test_large_offset_all_distances_allowed(self):
        self.assertEqual(self.check(762.)['kind'], 'all_J')

    def test_two_branches_kept_for_fallback(self):
        reduced = self.check(.5)
        self.assertEqual(reduced['kind'], 'two_intervals')
        self.assertGreater(reduced['upper_small_branch_J'], 0.)

    def test_zero_boundary_preserves_isolated_zero(self):
        # Exactly construct the cancellation used inside pair_reduction.
        reduced = pair_reduction(0., 2., 0., buffer=0.)
        self.assertEqual(reduced['kind'], 'two_intervals')
        self.assertEqual(reduced['upper_small_branch_J'], 0.)
        self.assertEqual(reduced['lower_J'], 8.)

    def test_double_root_all_distances_allowed(self):
        reduced = pair_reduction(2., 2., 0., buffer=0.)
        self.assertEqual(reduced['kind'], 'all_J')

    def test_resource_box_upper_covers_PQ_samples_and_small_box_corners(self):
        context, labels, forecast, resources = fixture()
        case = PQIncrementCase(context, labels, forecast, 200., q_resources=resources)
        rng = np.random.default_rng(61114); H = rng.normal(size=(2, 3, 6))
        score = dict(parameters=[dict(H=h.tolist(), history_sse=.2*i) for i, h in enumerate(H)],
            sigma2=.7, cells=288, reference_mse=.7, validation_tie=1e-12)
        evidence = SymmetricRecovery(case, score, None, forecast, 200., 'symmetric')
        def check(u):
            # Independent full-matrix J, not the quadratic expansion or upper-bound routine.
            p, q = case.aggregate(case.delivery(u)); X = forecast+np.c_[p, q]/200.
            J = np.array([np.sum((X@(H[i]-H[j]).T)**2)/(2*.7) for i, j in evidence.pairs])
            self.assertTrue(np.all(J <= evidence.upper+1e-10))
            self.assertTrue(np.all(np.sqrt(J) <= np.sqrt(evidence.upper)*(1+1e-9)+1e-10))
        # Test the larger power box, including points outside energy coupling.
        for _ in range(128):
            check(rng.uniform(case.lower, case.upper))
        # Exhaust the four vertices of an actual two-resource Q sub-box at 10:00.
        for choices in product((0, 1), repeat=2):
            u = np.zeros(case.shape)
            for offset, choice in enumerate(choices):
                j = case.P_R+offset
                u[2, j] = (case.lower if choice == 0 else case.upper)[2, j]
            self.assertTrue(case.audit(u)['passed'])
            check(u)

    def test_impossible_single_branch_returns_without_optimizer(self):
        context, labels, forecast, resources = fixture()
        case = PQIncrementCase(context, labels, forecast, 200., q_resources=resources)
        H = np.zeros((2, 3, 6)); H[1, :, 4] = 1.  # Q at terminal102 has no Q contract.
        score = dict(parameters=[dict(H=h.tolist(), history_sse=.2*i) for i, h in enumerate(H)],
            sigma2=.7, cells=288, reference_mse=.7, validation_tie=1e-12,
            models=[dict(topology_id='a'), dict(topology_id='b')])
        with patch('gurobipy.Model', side_effect=AssertionError('Unexpected optimizer call')) as native:
            receipt, action = plan(case, score, None, forecast, 200., mode='symmetric', seconds=1., env=None)
        self.assertEqual(native.call_count, 0)
        self.assertIsNone(action)
        self.assertEqual(receipt['status'], 'infeasible_by_resource_box_bound')
        self.assertIsNone(receipt['native_status'])
        self.assertFalse(receipt['optimizer_invoked'])
        self.assertEqual(receipt['resource_box_checks'][0]['relaxed_power_box_upper_J'], 0.)


if __name__ == '__main__':
    unittest.main()
