"""Meaningful exact-bound and physical-equivalence tests; no policy changes."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from research_experiments.mobile_ev_mle.weak_branch_information import exact_certificate, nominal_domain_checks, rational_constants
from research_experiments.mobile_ev_mle.weak_branch_ac_audit import (DEFAULT_WORKSPACE, paired_networks, project_imports,
                                  reduced_ac, reduced_indices)


class ExactConstantsTests(unittest.TestCase):
    def test_nominal_domain_caps_without_rounding(self):
        checks = nominal_domain_checks()
        for name, value in checks.items():
            if not name.endswith("_cap_sq"):
                self.assertLess(value, checks[name.replace("_sq", "_cap_sq")], name)

    def test_ball_invariance_and_contraction(self):
        c, _, margins = exact_certificate()
        self.assertEqual(c["C"], F(".02714416"))
        self.assertLess(c["C"] / c["vmin"], c["radius"])
        self.assertLess(c["K"], 1)
        self.assertGreater(c["L_den"], 0)
        self.assertTrue(all(value > 0 for value in margins.values()))

    def test_current_contrast_solves_the_derived_inequality(self):
        c = rational_constants()
        for u in (F(0), F(".075"), F(".15"), F(".3")):
            left = c["L0"] + c["L1"] * u
            right = ((c["db"] + u) / c["vmin"] + c["mb"] / c["vmin"]**2
                     * (c["zleaf"] * left + c["dzleaf"] * c["mb"] / c["vmin"]))
            self.assertEqual(left, right)
            self.assertEqual(c["a"] + c["b"] * u, c["zweak"] * left / (1 - c["K"]))

    def test_energy_is_grid_energy_not_soc_energy(self):
        c = rational_constants()
        self.assertEqual(c["sum_u"], F("1.2"))
        self.assertEqual(c["sum_u2"], F(".36"))
        # A full-cap four-slot pulse saturates both energy bounds.
        pulse = [F(0)] * 12 + [c["umax"]] * 4 + [F(0)] * 4
        self.assertEqual(sum(pulse), c["sum_u"])
        self.assertEqual(sum(u**2 for u in pulse), c["sum_u2"])
        self.assertEqual(sum((c["a"] + c["b"] * u)**2 for u in pulse), c["sum_delta_v2"])

    def test_complete_gaussian_weight_and_exact_kl_bound(self):
        c = rational_constants()
        self.assertEqual(c["gaussian_weight"], F(3) / c["sigma_v"]**2
                         + c["master_gain"]**2 / c["sigma_master"]**2)
        self.assertEqual(c["kl_exact"], F(3668918839949454447125, 34238718246575913269706))
        self.assertLess(c["kl_exact"], F(".11"))
        self.assertGreater(c["kl_exact"], F(".107"))


class PhysicalConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Candidate, predict, _ = project_imports(DEFAULT_WORKSPACE)
        cls.ac_predict = staticmethod(predict)

    def test_reduced_impedance_identity_all_pairs(self):
        for pair in ((0, 1), (0, 2), (1, 2)):
            cand = self.Candidate.from_supports("template", 3, (pair,))
            rscale, xscale = np.linspace(.8, 1.2, cand.m), np.linspace(1.2, .8, cand.m)
            ch, st, r, x, rs, xs, zs = paired_networks(pair, rscale, xscale, self.Candidate)
            bch, bst = ch.B[reduced_indices(ch)], st.B[reduced_indices(st)]
            zch, zst = (bch * (r + 1j * x)) @ bch.T, (bst * (rs + 1j * xs)) @ bst.T
            d = np.zeros(4); d[1 + pair[0]], d[1 + pair[1]] = 1, -1
            np.testing.assert_allclose(zst - zch, zs * np.outer(d, d), rtol=0, atol=2e-17)
            self.assertEqual(ch.edges[0], st.edges[0])
            np.testing.assert_array_equal(zst[0], zch[0])

    def test_full_ac_reduction_and_all_measurement_channels(self):
        ch, st, r, x, rs, xs, _ = paired_networks((0, 1), np.ones(5), np.ones(5), self.Candidate)
        p = np.array([[.136 + .3, .128, .136], [.104, .112, .104]])
        q = np.array([[.047, .039, .047], [.033, .041, .033]])
        sp, sq = np.array([.135, .065]), np.array([.05, .02])
        physical = [self.ac_predict(c, rr, xx, p, q, sp, sq)
                    for c, rr, xx in ((ch, r, x), (st, rs, xs))]
        loads = np.c_[sp + 1j * sq, p + 1j * q]
        for candidate, pred in zip((ch, st), physical):
            ids = reduced_indices(candidate)
            reduced, residual, _ = reduced_ac(pred["zbus"][np.ix_(ids, ids)], loads)
            np.testing.assert_allclose(reduced, pred["voltage"][:, ids], rtol=0, atol=5e-13)
            independent_master = np.conj(np.conj(loads / reduced).sum(axis=1))
            np.testing.assert_allclose(independent_master,
                pred["master_p"] + 1j * pred["master_q"], rtol=0, atol=5e-13)
            np.testing.assert_array_equal(pred["p"], p)
            np.testing.assert_array_equal(pred["q"], q)
            self.assertLess(residual, 5e-14)
            self.assertLess(pred["power_balance_residual"], 2e-10)

    def test_competitor_can_leave_nominal_pop_but_remains_physical(self):
        ch, st, r, x, rs, xs, _ = paired_networks((0, 1), 1.2 * np.ones(5),
                                                1.2 * np.ones(5), self.Candidate)
        # The added positive weak impedance exceeds terminal 0's POP edge maximum.
        edge = st.supports.index((0,))
        self.assertGreater(rs[edge], 1.2 * .028)
        self.assertGreater(xs[edge], 1.2 * .011)
        self.assertTrue(np.all((rs >= 1e-6) & (rs <= .1)))
        self.assertTrue(np.all((xs >= 1e-6) & (xs <= .1)))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report = dict(status="passed" if result.wasSuccessful() else "failed",
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped),
                  files_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("weak_branch_information.py", "weak_branch_ac_audit.py",
                                   "test_weak_branch_information.py")})
    Path(__file__).with_name("weak_branch_information_test_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
