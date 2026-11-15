"""Independent algebra and failure checks; no optimizer, fitting or AC calls."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from ...observations import Observation
from . import geometry


def fixture(repeat=4):
    rng = np.random.default_rng(2026111519)
    p = rng.normal(.06, .02, (31, 2))
    q = p*np.array([.2, -.1])+rng.normal(0, .006, p.shape)+np.array([-.01, .02])
    supports = [[0], [1], [0, 1]]
    point = dict(supports=supports, r=[.1, .2, .04], x=[.04, 0., .01])
    z = np.array([[1., 0.], [0., 1.], [1., 1.]])
    response = np.c_[(z.T*np.array(point['r']))@z, (z.T*np.array(point['x']))@z]
    root = np.ones(len(p))*1.01
    voltage = np.ones_like(p)
    y = np.c_[p, q]@response.T
    train = Observation(p, q, y, (10, 20), voltage, root)
    score = dict(labels=[10, 20], sigma2=2e-6,
        parameters=[dict(index=0, point=point, H=response.tolist(), history_sse=.001)],
        native_bank=[dict(H=response.tolist())])
    B = np.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    case = SimpleNamespace(H=3, R=3, N=2, repeat=repeat, labels=(10, 20), B=B)
    return case, score, train


class GeometryTests(unittest.TestCase):
    def test_design_matches_support_operator_and_row_order(self):
        p = np.array([[1., 2.], [-3., 4.]])
        q = np.array([[2., 5.], [1., -2.]])
        supports = [[0], [1], [0, 1]]
        F = geometry.design_blocks(p, q, supports)
        theta = np.array([1., 2., 3., 4., 5., 6.])
        actual = F.reshape(-1, 6)@theta
        expected = []
        for t in range(2):
            for node in range(2):
                value = 0.
                for k, support in enumerate(supports):
                    if node in support:
                        value += theta[k]*sum(p[t, j] for j in support)
                        value += theta[3+k]*sum(q[t, j] for j in support)
                expected.append(value)
        np.testing.assert_array_equal(actual, expected)

    def test_sandwich_matches_explicit_noise_and_keeps_rx_correlation(self):
        case, score, train = fixture()
        blocks = geometry.design_blocks(train.p, train.q, score['parameters'][0]['point']['supports'])
        C, G, info = geometry.parameter_covariance(blocks, 3e-6, 2e-6)
        F = blocks.reshape(-1, 6)
        noise = np.kron(np.eye(len(train.p)), 3e-6*np.ones((2, 2))+2e-6*np.eye(2))
        influence = np.linalg.solve(F.T@F, F.T)
        expected = influence@noise@influence.T
        np.testing.assert_allclose(C, expected, rtol=2e-11, atol=1e-13)
        np.testing.assert_allclose(G, F.T@F)
        self.assertGreater(np.linalg.norm(C[:3, 3:]), 1e-6)
        self.assertEqual(info['rank'], 6)

    def test_common_root_noise_is_not_an_independent_cell_noise(self):
        blocks = np.array([[[1.], [1.]], [[2.], [2.]]])
        common, _, _ = geometry.parameter_covariance(blocks, 1., 0.)
        independent, _, _ = geometry.parameter_covariance(blocks, 0., 1.)
        np.testing.assert_allclose(common, 2*independent)

    def test_rank_deficiency_is_rejected(self):
        p = np.arange(1., 7.).reshape(3, 2)
        blocks = geometry.design_blocks(p, .2*p, [[0], [1]])
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            geometry.parameter_covariance(blocks, 1e-6, 1e-6)

    def test_build_direct_trace_and_nodewise_pq_scaling(self):
        case, score, train = fixture()
        matrices, metadata = geometry.build_budgets(case, score, train, train, 200., targets=[0])
        u = np.array([[1., .7, -.8], [0., 2., .6], [-1., -.5, .3]])
        audit = geometry.direct_audit(u, matrices, metadata)
        injection = u@case.B.T/200.
        slope = np.array([np.cov(train.p[:, j], train.q[:, j], ddof=0)[0, 1]
                          /np.var(train.p[:, j]) for j in range(2)])
        residual = (train.q-train.q.mean(0))-(train.p-train.p.mean(0))*slope
        expected_pq = np.mean((injection[:, 2:]-slope*injection[:, :2])**2/np.mean(residual**2, 0))
        self.assertAlmostEqual(audit['rows'][0]['value'], expected_pq, places=12)
        C = matrices['covariance_native0']
        delta_blocks = geometry.design_blocks(injection[:, :2], injection[:, 2:],
            score['parameters'][0]['point']['supports'])
        expected_response = sum(np.trace(f@C@f.T) for f in delta_blocks)/(3*2*score['sigma2'])
        self.assertAlmostEqual(audit['rows'][1]['value'], expected_response, places=12)
        self.assertEqual(metadata['targets'][0]['at_zero_count'], 1)
        self.assertGreater(C[4, 4], 0.)  # frozen zero-X coefficient is not known exactly
        self.assertTrue(geometry.direct_audit(np.zeros_like(u), matrices, metadata)['passed'])
        for kind in ('pq', 'response'):
            for guard in geometry.guards(matrices, metadata, kind):
                self.assertGreaterEqual(np.linalg.eigvalsh(guard['Q']).min(), -1e-12)

    def test_repeat_and_power_units_cancel_correctly(self):
        case, score, train = fixture()
        a, meta = geometry.build_budgets(case, score, train, train, 200.)
        case.repeat = 1
        b, _ = geometry.build_budgets(case, score, train, train, 200.)
        np.testing.assert_array_equal(a['response_native0'], b['response_native0'])
        c, _ = geometry.build_budgets(case, score, train, train, 400.)
        np.testing.assert_allclose(c['response_native0'], a['response_native0']/4)
        np.testing.assert_allclose(c['pq'], a['pq']/4)

    def test_fail_closed_for_mapping_missing_voltage_and_mutated_matrix(self):
        case, score, train = fixture()
        with self.assertRaisesRegex(ValueError, 'target representatives'):
            geometry.build_budgets(case, score, train, train, 200., targets=[1])
        invalid = Observation(train.p, train.q, train.y, train.labels)
        with self.assertRaisesRegex(ValueError, 'V/root'):
            geometry.build_budgets(case, score, train, invalid, 200.)
        broken = deepcopy(score)
        broken['parameters'][0]['point']['x'][0] += .1
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            geometry.build_budgets(case, broken, train, train, 200.)
        matrices, metadata = geometry.build_budgets(case, score, train, train, 200.)
        matrices['pq'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            geometry.guards(matrices, metadata, 'pq')


if __name__ == '__main__':
    unittest.main()
