"""Direct node sums, exposed linear rows and a tiny native attachment check."""
from copy import deepcopy
import unittest

import numpy as np

from .bandwidth import build_bandwidth
from .domain import BAND_RELATIVE_BUFFER, NodeTimePQCase
from .test_bandwidth import training_fixture


class PhysicalFixture:
    H, N, R, P_R, Q_R = 24, 2, 5, 3, 2
    labels = (101, 102)
    shape, dimension = (24, 5), 120
    repeat, dt = 4, 1.

    def __init__(self):
        self.clock = (8.+np.arange(24)+.5) % 24
        self.resources = [{'node_label': node, 'id': str(i)}
            for i, node in enumerate([101, 101, 102, 101, 102])]
        self.node = np.array([0, 0, 1, 0, 1])
        self.B = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], float)
        self.lower, self.upper = np.full(self.shape, -10.), np.full(self.shape, 10.)
        self.A_eq, self.b_eq = np.zeros((0, self.dimension)), np.zeros(0)
        self.A_ub = np.zeros((1, self.dimension)); self.A_ub[0, 0] = 1.
        self.b_ub = np.array([9.])

    def array(self, u):
        value = np.asarray(u, float)
        if value.shape != self.shape or not np.isfinite(value).all():
            raise ValueError('Invalid toy control')
        return value

    def audit(self, u, tolerance=1e-7):
        value = self.array(u)
        error = max(float(np.max(self.lower-value)), float(np.max(value-self.upper)),
                    float(np.max(self.A_ub@value.ravel()-self.b_ub)), 0.)
        return {'passed': error <= tolerance, 'physical_violation': error}

    def fee(self, u):
        return float(np.abs(self.array(u)).sum())

    def attach(self, model, name='toy'):
        u = model.addMVar(self.dimension, lb=self.lower.ravel(), ub=self.upper.ravel(), name=name)
        absolute = model.addMVar(self.dimension, lb=0.)
        model.addConstr(absolute >= u); model.addConstr(absolute >= -u)
        model.addConstr(self.A_ub@u <= self.b_ub)
        return u.reshape(self.shape), absolute.sum()


def wrapped_fixture():
    p, q, labels, stamps, *_ = training_fixture()
    arrays, metadata = build_bandwidth(p, q, labels, stamps, 1.)
    physical = PhysicalFixture()
    return physical, arrays, metadata, NodeTimePQCase(physical, arrays, metadata)


class DomainTests(unittest.TestCase):
    def test_zero_and_hand_aggregated_same_node_contracts(self):
        physical, arrays, _, wrapper = wrapped_fixture()
        u = np.zeros(physical.shape)
        self.assertTrue(wrapper.audit(u)['passed'])
        u[0] = [1., 2., 4., 1.5, -1.]
        direct = wrapper.band_audit(u)
        np.testing.assert_array_equal(direct['delta_p_kw'][0], [3., 4.])
        np.testing.assert_array_equal(direct['delta_q_kvar'][0], [1.5, -1.])
        np.testing.assert_allclose(direct['deviation_kvar'][0], [0., 0.], atol=1e-15)
        self.assertTrue(direct['passed'])
        u[0, 3] += .001  # Node101 has exactly zero range at hour08.
        self.assertFalse(wrapper.band_audit(u)['passed'])
        self.assertAlmostEqual(wrapper.band_audit(u)['max_violation_kvar'], .001)

    def test_matrix_and_independent_node_sum_agree(self):
        physical, arrays, _, wrapper = wrapped_fixture()
        rng = np.random.default_rng(918)
        u = rng.normal(size=physical.shape)
        independent = np.asarray(wrapper.band_audit(u)['deviation_kvar'])
        encoded = wrapper.band_A_ub[:24*physical.N]@u.ravel()
        np.testing.assert_allclose(encoded.reshape(24, physical.N), independent, rtol=2e-15, atol=2e-15)
        np.testing.assert_array_equal(wrapper.band_b_ub[:24*physical.N].reshape(24, physical.N), arrays['band_kvar'])
        self.assertEqual(len(wrapper.A_ub), len(physical.A_ub)+48*physical.N)
        np.testing.assert_array_equal(wrapper.A_ub[0], physical.A_ub[0])
        np.testing.assert_array_equal(physical.A_ub.shape, (1, 120))

    def test_physical_failure_and_band_tolerance_are_distinct(self):
        physical, _, _, wrapper = wrapped_fixture()
        u = np.zeros(physical.shape)
        u[0] = [10., 0., 0., 5., 0.]  # Band holds; original physical row fails.
        self.assertTrue(wrapper.band_audit(u)['passed'])
        checked = wrapper.audit(u)
        self.assertFalse(checked['physical_passed']); self.assertFalse(checked['passed'])
        u[:] = 0.; u[0, 3] = 1e-9
        self.assertFalse(wrapper.audit(u, tolerance=1.)['passed'])

    def test_buffer_preserves_zero_and_keeps_original_domain_explicit(self):
        physical, arrays, _, wrapper = wrapped_fixture()
        zero = np.zeros(physical.shape)
        self.assertTrue(wrapper.band_audit(zero)['encoded_passed'])
        point = zero.copy()
        point[1, 3] = arrays['band_kvar'][1, 0]
        self.assertTrue(wrapper.band_audit(point)['passed'])
        self.assertFalse(wrapper.band_audit(point)['encoded_passed'])
        point[1, 3] *= 1.-BAND_RELATIVE_BUFFER
        self.assertTrue(wrapper.band_audit(point)['encoded_passed'])
        np.testing.assert_array_less(wrapper.encoded_A_ub@point.ravel(),
            wrapper.encoded_b_ub+1e-13)
        np.testing.assert_allclose(wrapper.encoded_A_eq@point.ravel(), wrapper.encoded_b_eq, atol=1e-14)
        self.assertGreater(len(wrapper.encoded_band_A_eq), 0)

    def test_label_clock_mapping_and_hash_fail_closed(self):
        physical, arrays, metadata, _ = wrapped_fixture()
        physical.clock = np.roll(physical.clock, 1)
        with self.assertRaisesRegex(ValueError, 'clock'):
            NodeTimePQCase(physical, arrays, metadata)
        physical = PhysicalFixture(); physical.B[2, 3] = .5
        with self.assertRaisesRegex(ValueError, 'catalog'):
            NodeTimePQCase(physical, arrays, metadata)
        physical = PhysicalFixture()
        modified = {key: value.copy() for key, value in arrays.items()}
        modified['band_kvar'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'hash'):
            NodeTimePQCase(physical, modified, metadata)
        changed = deepcopy(metadata); changed['decision_hours'] = list(range(24))
        with self.assertRaisesRegex(ValueError, 'clock'):
            NodeTimePQCase(physical, arrays, changed)

    def test_native_linear_attachment_small_problem(self):
        # This is only a fixed-action feasibility toy (no evidence/AC experiment).
        try:
            import gurobipy as gp
        except ImportError:
            self.skipTest('Gurobi not available for the optional attachment toy')
        physical, _, _, wrapper = wrapped_fixture()
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            try:
                env.start()
            except gp.GurobiError as exc:
                self.skipTest(f'Gurobi license unavailable: {exc}')
            with gp.Model(env=env) as model:
                model.Params.TimeLimit = 1.
                u, fee = wrapper.attach(model)
                point = np.zeros(physical.shape)
                point[0] = [1., 2., 4., 1.5, -1.]
                model.addConstr(u == point)
                model.setObjective(fee)
                model.optimize()
                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                self.assertTrue(wrapper.audit(u.X)['passed'])
                self.assertAlmostEqual(model.ObjVal, physical.fee(point), places=8)
            with gp.Model(env=env) as model:
                model.Params.TimeLimit = 1.
                u, fee = wrapper.attach(model)
                point[0, 3] += .01
                model.addConstr(u == point)
                model.setObjective(fee)
                model.optimize()
                self.assertEqual(model.Status, gp.GRB.INFEASIBLE)


if __name__ == '__main__':
    unittest.main()
