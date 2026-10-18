"""Independent derivative/physics checks for feasible AC *upper* bounds.

A large local score is never a rejection rule.  Small-score points must satisfy
the original physical domain, and their measured-P/Q latent choice is explicit.
"""
import unittest

import numpy as np

from .bounds import Measurements, PhysicalDomain, TreeSpec
from .compatibility import ac_voltage, fit_compatible_point, physical_checks
from ..reference_benchmark.representation import physical_tree_ac


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tree = TreeSpec(root=0, edges=((0, 1), (1, 2), (1, 3)), observed_nodes=(1, 2, 3))
        self.p = np.array([[.05, .08, .06], [.07, .03, .10], [.08, .06, .04], [.04, .09, .07]])
        self.q = np.array([[.02, .03, .01], [.01, .02, .03], [.04, .02, .01], [.02, .02, .04]])
        self.root = np.array([1., 1.005, .995, 1.])
        self.x = np.r_[np.log([.008, .012, .010]), [.6, .8, .5]]
        self.domain = PhysicalDomain(z_min=.001, z_max=.03, v_min=.8, v_max=1.1,
            power_caps={1:.3, 2:.3, 3:.3}, ratio_min=.25, ratio_max=4.)

    def observations(self, voltage):
        return Measurements(p=self.p, q=self.q, v=voltage, root_v=self.root,
                            sigma_p=.001, sigma_q=.001, sigma_v=.0005)

    def test_implicit_jacobian_against_central_differences(self):
        result = ac_voltage(self.tree, self.p, self.q, self.root, self.x, derivatives=True)
        finite = np.empty_like(result["magnitude_jacobian"])
        for j in range(len(self.x)):
            step = np.zeros_like(self.x); step[j] = 2e-6
            plus = ac_voltage(self.tree, self.p, self.q, self.root, self.x+step)
            minus = ac_voltage(self.tree, self.p, self.q, self.root, self.x-step)
            finite[:, :, j] = (plus["magnitude"]-minus["magnitude"])/(4e-6)
        np.testing.assert_allclose(result["magnitude_jacobian"], finite, rtol=3e-5, atol=1e-10)

    def test_independent_native_tree_backward_forward_oracle(self):
        point = ac_voltage(self.tree, self.p, self.q, self.root, self.x)
        branches = [dict(from_bus=a, to_bus=b, r_ohm=float(point["impedance"][e].real),
                        x_ohm=float(point["impedance"][e].imag), status=True)
                    for e, (a, b) in enumerate(self.tree.edges)]
        # These bases have one ohm impedance base; oracle is independent of D/Zbus.
        oracle = physical_tree_ac(branches, self.tree.observed_nodes, self.tree.root,
            1., 1000., self.p*1000, self.q*1000, source_bus=0, root_v=self.root)
        np.testing.assert_allclose(point["voltage"], oracle["measured_voltage"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(point["master_p"], oracle["master_p"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(point["master_q"], oracle["master_q"], atol=1e-12, rtol=0)
        self.assertLess(point["ac_equation_residual"], 1e-11)
        self.assertLess(point["power_balance_residual"], 1e-11)

    def test_compatible_point_and_large_upper_never_reject(self):
        point = ac_voltage(self.tree, self.p, self.q, self.root, self.x)
        theta = np.r_[point["impedance"].real, point["impedance"].imag]
        fitted = fit_compatible_point(self.tree, self.observations(point["magnitude"]),
                                      self.domain, theta, max_nfev=3)
        self.assertTrue(fitted["ac_noise_ball_feasible"])
        self.assertTrue(all(fitted["best_point"]["checks"].values()))
        self.assertFalse(fitted["fitted_or_MLE"])
        self.assertFalse(fitted["reject"])
        bad_voltage = np.full_like(self.p, .90)
        failed = fit_compatible_point(self.tree, self.observations(bad_voltage),
                                      self.domain, theta, max_nfev=1, early_stop=False)
        self.assertTrue(failed["available"])
        self.assertGreater(failed["best_point"]["raw_whitened_AC_sse_upper"], failed["chi_square_threshold"])
        self.assertFalse(failed["ac_noise_ball_feasible"])
        self.assertFalse(failed["reject"])

    def test_domain_and_source_failures_are_retained(self):
        point = ac_voltage(self.tree, self.p, self.q, self.root, self.x)
        obs = self.observations(point["magnitude"])
        low_caps = PhysicalDomain(z_min=.001, z_max=.03, v_min=.8, v_max=1.1,
                                 power_caps=.01, ratio_min=.25, ratio_max=4.)
        self.assertFalse(physical_checks(self.tree, obs, low_caps, point)["nodal_power"])
        source_tree = TreeSpec(root=0, edges=self.tree.edges, observed_nodes=self.tree.observed_nodes,
                               source_node=1)
        result = fit_compatible_point(source_tree, obs, self.domain, np.r_[np.ones(3), np.ones(3)])
        self.assertEqual(result["status"], "unsupported_source")
        self.assertFalse(result["available"])
        self.assertFalse(result["reject"])

    def test_unmetered_native_bus_not_silently_zeroed(self):
        partial = TreeSpec(root=0, edges=self.tree.edges, observed_nodes=(2, 3))
        with self.assertRaisesRegex(ValueError, "all nonroot"):
            ac_voltage(partial, self.p[:, 1:], self.q[:, 1:], self.root, self.x)


if __name__ == "__main__":
    unittest.main()
