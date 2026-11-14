"""Pure-Q units, old-P preservation and evidence/gradient algebra; no AC batch."""
from copy import deepcopy
import unittest

import numpy as np

from ...observations import Observation
from ..absolute_identification_20261114.solver import SymmetricRecovery
from ..customer_flexibility_20261113.placement import CandidateEvidence
from ..expected_logodds_admm_20261113.evidence import build, direct_gains
from .domain import PQIncrementCase


def fixture(q_resources=None):
    labels = (101, 102, 103)
    context = dict(node_mapping=[dict(label=label, feeder_id=i+1, scale=.5) for i, label in enumerate(labels)],
        resource_metadata_at_history_start=[dict(feeder=i+1, date='2024-01-01', PV_systems_kW=20+i,
            EV_chargers_kW=11+i, hot_water_tanks_kW=5+i) for i in range(3)])
    forecast = np.zeros((96, 6)); forecast[:, :3] = -.02
    if q_resources is None:
        q_resources = [dict(id='q101', node_label=101, capacity_kvar=5.,
            fee_positive_yuan_per_kvarh=.02, fee_negative_yuan_per_kvarh=.03),
            dict(id='q103', node_label=103, capacity_kvar=10., available_clock_hours=[10, 14],
            fee_positive_yuan_per_kvarh=.04, fee_negative_yuan_per_kvarh=.05)]
    return context, labels, forecast, q_resources


class DomainTests(unittest.TestCase):
    def make_case(self, rows=None):
        context, labels, forecast, rows = fixture(rows)
        return PQIncrementCase(context, labels, forecast, 200., q_resources=rows)

    def test_P_contract_and_random_prefix_preserved(self):
        case = self.make_case(); old = case.p_case
        np.testing.assert_array_equal(case.lower[:, :old.R], old.lower)
        np.testing.assert_array_equal(case.upper[:, :old.R], old.upper)
        self.assertEqual(case.resources[:old.R], old.resources)
        direction = np.sin(np.arange(old.dimension)).reshape(old.shape)
        u_old, _ = old.linear_optimum(direction)
        u = np.zeros(case.shape); u[:, :old.R] = u_old
        np.testing.assert_allclose(case.A_eq@u.ravel(), old.A_eq@u_old.ravel(), atol=1e-14)
        np.testing.assert_allclose(case.A_ub@u.ravel(), old.A_ub@u_old.ravel(), atol=1e-14)
        self.assertAlmostEqual(case.fee(u), old.fee(u_old))
        old_eta = np.random.default_rng(773).uniform(.9, 1., old.R)
        eta = np.random.default_rng(773).uniform(.9, 1., case.R)
        np.testing.assert_array_equal(eta[:old.R], old_eta)
        actual = np.zeros((96, old.N))
        a, ra = old.execute(u_old, actual, old_eta)
        b, rb = case.execute(u, actual, eta)
        np.testing.assert_array_equal(b[:, :old.R], a)
        self.assertEqual(rb['delivered_abs_kwh'], ra['delivered_abs_kwh'])
        self.assertEqual(rb['PV_clipped_kwh'], ra['PV_clipped_kwh'])

    def test_pure_Q_is_not_P_and_units_fees(self):
        case = self.make_case(); u = np.zeros(case.shape)
        u[0, case.P_R] = -2.; u[2:6, case.P_R+1] = 3.
        self.assertTrue(case.audit(u)['passed'])
        p, q = case.aggregate(case.delivery(u))
        self.assertTrue(np.all(p == 0.))
        np.testing.assert_array_equal(q[:4, 0], np.full(4, -2.))
        np.testing.assert_array_equal(q[8:24, 2], np.full(16, 3.))
        np.testing.assert_allclose(case.shift_pq_pu(u, 200.), np.column_stack((p, q))/200.)
        self.assertEqual(case.amounts(u), dict(active_abs_kwh=0., reactive_abs_kvarh=14.))
        self.assertAlmostEqual(case.fee(u), 2*.03+12*.04)
        delivered, receipt = case.execute(u, np.zeros((96, 3)), np.full(case.R, .95))
        self.assertAlmostEqual(receipt['delivered_abs_kvarh'], 13.3)
        self.assertEqual(receipt['delivered_abs_kwh'], 0.)
        self.assertAlmostEqual(receipt['response_abs_error_kvarh'], .7)
        self.assertTrue(case.audit(delivered)['passed'])
        with self.assertRaises(AttributeError):
            _ = case.qratio

    def test_window_bounds_and_invalid_sites(self):
        case = self.make_case(); q = case.P_R+1
        np.testing.assert_array_equal(np.flatnonzero(case.upper[:, q]), [2, 3, 4, 5])
        u = np.zeros(case.shape); u[0, q] = .1
        self.assertFalse(case.audit(u)['passed'])
        u = np.zeros(case.shape); u[0, case.P_R] = 5.1
        self.assertFalse(case.audit(u)['passed'])
        context, labels, forecast, rows = fixture()
        explicit_day = deepcopy(rows); explicit_day[0]['available_clock_hours'] = list(range(24))
        all_day = PQIncrementCase(context, labels, forecast, 200., q_resources=explicit_day)
        np.testing.assert_array_equal(all_day.lower[:, all_day.P_R], case.lower[:, case.P_R])
        np.testing.assert_array_equal(all_day.upper[:, all_day.P_R], case.upper[:, case.P_R])
        for key, value in [('node_label', 0), ('capacity_kvar', -1.), ('reservation_fee_yuan_per_kvar_day', 1.)]:
            invalid = deepcopy(rows); invalid[0][key] = value
            with self.assertRaises(ValueError):
                PQIncrementCase(context, labels, forecast, 200., q_resources=invalid)

    def test_linear_domain_extrema_and_no_extra_baseline(self):
        case = self.make_case()
        direction = np.zeros(case.shape); direction[:, case.P_R:] = -1.
        u, receipt = case.linear_optimum(direction, least_fee=True)
        self.assertTrue(receipt['resource_audit']['passed'])
        np.testing.assert_allclose(u[:, case.P_R:], case.upper[:, case.P_R:], atol=1e-8)
        with self.assertRaises(RuntimeError):
            case.device_baseline_pq_kw()

    def test_evidence_full_H_and_gradient(self):
        case = self.make_case(); rng = np.random.default_rng(830)
        H = rng.normal(size=(3, 3, 6)); forecast = rng.normal(size=(96, 6)); sigma2 = .7
        u = np.zeros(case.shape); u[:, case.P_R] = rng.uniform(-2., 2., 24)
        u[2:6, case.P_R+1] = 1.
        evidence = case.evidence(H, 0, [1, 2], forecast, sigma2, 200., 2.)
        np.testing.assert_allclose(evidence.gains(case.delivery(u)),
            direct_gains(H, 0, [1, 2], forecast, sigma2, case, 200., case.delivery(u)), atol=1e-12)
        D = H[0]-H[1]; effect = case.response_effect(D, 200.)
        np.testing.assert_allclose(effect[:, :case.P_R], D[:, case.node[:case.P_R]]/200.)
        np.testing.assert_allclose(effect[:, case.P_R:], D[:, case.N+case.node[case.P_R:]]/200.)
        gradient = (forecast@D.T).reshape(24, 4, 3).sum(1)@effect/sigma2
        d = rng.normal(size=case.shape); step = 1e-5
        plus = direct_gains(H, 0, [1], forecast, sigma2, case, 200., case.delivery(step*d))[0]
        minus = direct_gains(H, 0, [1], forecast, sigma2, case, 200., case.delivery(-step*d))[0]
        self.assertAlmostEqual((plus-minus)/(2*step), float(np.sum(gradient*d)), places=9)

    def test_empty_Q_matches_old_evidence(self):
        case = self.make_case([]); old = case.p_case
        rng = np.random.default_rng(303); H = rng.normal(size=(2, 3, 6)); forecast = rng.normal(size=(96, 6))
        actual = case.evidence(H, 0, [1], forecast, .5, 200., 2.)
        expected = old.evidence(H, 0, [1], forecast, .5, 200., 2.)
        np.testing.assert_allclose(actual.base, expected.base)
        np.testing.assert_allclose(actual.effect, expected.effect)
        np.testing.assert_allclose(actual.scales, expected.scales)

    def test_existing_quadratic_and_box_interfaces(self):
        case = self.make_case(); rng = np.random.default_rng(4001)
        H = rng.normal(size=(3, 3, 6)); forecast = rng.normal(size=(96, 6))
        u = np.zeros(case.shape); u[:, case.P_R] = rng.uniform(-2., 2., 24)
        u[2:6, case.P_R+1] = -1.5
        nominal = case.evidence(H, 0, [1, 2], forecast, .7, 200., 2.)
        existing = build(H, 0, [1, 2], forecast, .7, case, 200., 2.)
        np.testing.assert_allclose(nominal.effect, existing.effect)
        evidence = CandidateEvidence(case, H, 0, [1, 2], forecast, .7, 200., means=.9*H)
        np.testing.assert_allclose(evidence.values(u), evidence.direct(u), atol=1e-12)
        self.assertTrue(np.all(evidence.power_box_upper() >= evidence.direct(u)[:2]-1e-10))
        score = dict(parameters=[dict(H=h.tolist(), history_sse=float(i)) for i, h in enumerate(H)],
            sigma2=.7, cells=288, reference_mse=.7, validation_tie=1e-12)
        observed = Observation(forecast[:, :3], forecast[:, 3:], forecast@H[0].T, case.labels)
        symmetric = SymmetricRecovery(case, score, observed, forecast, 200., 'symmetric_guard')
        symmetric.verify_expansion(u)
        self.assertTrue(np.all(symmetric.upper >= symmetric.direct(u)['J']-1e-10))


if __name__ == '__main__':
    unittest.main()
