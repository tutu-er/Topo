"""Independent branch-flow and small-load AC checks for physical LinDistFlow."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from .linear_truth import build_response, linear_voltages, observe_linear


def network():
    return dict(root_bus=1, base_kv=.4, base_mva=.2, base_kw=200.,
        buses=[dict(bus_id=n) for n in (1, 2, 101, 102)],
        branches=[dict(from_bus=1, to_bus=2, r_ohm=.012, x_ohm=.008, status=True),
            dict(from_bus=2, to_bus=101, r_ohm=.020, x_ohm=.010, status=True),
            dict(from_bus=102, to_bus=2, r_ohm=.030, x_ohm=.015, status=True)])


class LinearTruthChecks(unittest.TestCase):
    def test_hand_shared_path_coefficients(self):
        H = build_response(network(), (101, 102))
        z = .4**2/.2
        R = 2*np.array([[.032, .012], [.012, .042]])/z
        X = 2*np.array([[.018, .008], [.008, .023]])/z
        np.testing.assert_allclose(H, np.c_[R, X], rtol=0., atol=1e-16)

    def test_independent_branch_flow_and_terminal_permutation(self):
        p = np.array([[.10, .20], [-.04, .06]])
        q = np.array([[.03, .08], [-.01, .02]])
        root = np.array([1.01, 1.02])
        v, audit = linear_voltages(network(), (101, 102), p, q, root)
        z = .4**2/.2
        hidden_squared = root**2-2*(.012*p.sum(1)+.008*q.sum(1))/z
        expected = np.c_[hidden_squared-2*(.020*p[:, 0]+.010*q[:, 0])/z,
                         hidden_squared-2*(.030*p[:, 1]+.015*q[:, 1])/z]
        np.testing.assert_allclose(v*v, expected, rtol=0., atol=5e-16)
        reverse, _ = linear_voltages(network(), (102, 101), p[:, ::-1], q[:, ::-1], root)
        np.testing.assert_allclose(reverse[:, ::-1], v)
        self.assertFalse(audit['fitted_candidate_used'])

    def test_original_ac_remainder_is_second_order(self):
        source = Path(__file__).resolve().parents[3]/'outputs/astra_confirmation_20261113_v1/physical/source_runtime/main/data.py'
        if not source.exists():
            self.skipTest('Original frozen AC runtime unavailable')
        spec = importlib.util.spec_from_file_location('_physical_linear_ablation_ac_test', source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        p = np.array([[.10, .20]])
        q = np.array([[.03, .08]])
        root = np.array([1.01])
        errors = []
        for scale in (.5, .25):
            ac, _ = module._ac_voltages(network(), (101, 102), scale*p, scale*q, root)
            linear, _ = linear_voltages(network(), (101, 102), scale*p, scale*q, root)
            errors.append(float(np.linalg.norm(ac**2-linear**2)))
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertLess(errors[0]/errors[1], 4.2)

    def test_original_meter_streams_and_power_preserved(self):
        p, q = np.array([[.10, .20]]), np.array([[.03, .08]])
        b = dict(network=network(), labels=(101, 102), p=p, q=q, root=np.array([1.01]),
            metadata=dict(pq_sigma=.005, voltage_sigma=.0005),
            noise=[np.array([[1., -1.]]), np.array([[.5, -.5]]), np.array([[2., -2.]]), np.array([.25])])
        observed, audit, physical = observe_linear(None, b)
        np.testing.assert_array_equal(physical[0], p)
        np.testing.assert_array_equal(physical[1], q)
        np.testing.assert_allclose(observed.p, p*(1+.005*b['noise'][0]))
        np.testing.assert_allclose(observed.q, q*(1+.005*b['noise'][1]))
        np.testing.assert_allclose(observed.root, b['root']*(1+.0005*b['noise'][3]))
        np.testing.assert_allclose(observed.voltage, physical[2]*(1+.0005*b['noise'][2]))
        self.assertEqual(audit['physical_model'], 'physical_branch_LinDistFlow')


if __name__ == '__main__':
    unittest.main()
