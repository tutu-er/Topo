"""Small analytic and mocked checks; optional one-second native toy solve."""
import argparse
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import solver


class Toy:
    H, R, P_R, N, repeat, dt = 2, 2, 1, 1, 4, 1.
    shape, dimension = (2, 2), 4

    def __init__(self):
        self.lower = np.full(self.shape, -2.)
        self.upper = np.full(self.shape, 2.)
        self.A_eq = np.zeros((0, 4)); self.b_eq = np.zeros(0)
        self.A_ub = np.zeros((0, 4)); self.b_ub = np.zeros(0)
        self.fee_plus = np.tile([.2, .05], (2, 1)); self.fee_minus = self.fee_plus.copy()

    def array(self, value):
        value = np.asarray(value, float).reshape(self.shape)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite action')
        return value

    def fee(self, u):
        return float(np.sum(self.fee_plus*np.abs(self.array(u))))

    def audit(self, u, tolerance=1e-7):
        u = self.array(u)
        violation = max(0., float((self.lower-u).max()), float((u-self.upper).max()))
        return dict(passed=violation <= tolerance, maximum_violation=violation)

    def attach(self, model, name='toy'):
        u = model.addMVar(4, lb=self.lower.ravel(), ub=self.upper.ravel(), name=name)
        absolute = model.addMVar(4, lb=0., name=name+'_absolute')
        model.addConstr(absolute >= u); model.addConstr(absolute >= -u)
        return u.reshape(self.shape), self.fee_plus.ravel()@absolute

    def response_effect(self, D, base):
        return D/base

    def shift_pq_pu(self, u, base):
        return np.repeat(self.array(u), self.repeat, axis=0)/base


def fixture():
    H = [np.array([[0., 0.]]), np.array([[0., 1.]])]
    score = dict(parameters=[dict(H=h.tolist(), history_sse=0.) for h in H],
        sigma2=1., cells=4, reference_mse=1., validation_tie=1e-12,
        models=[dict(topology_id='a'), dict(topology_id='b')])
    validation = SimpleNamespace(p=np.zeros((4, 1)), q=np.zeros((4, 1)), y=np.zeros((4, 1)))
    return Toy(), score, validation, np.zeros((8, 2)), np.array([[0., 1.5], [0., 0.]])


class Tests(unittest.TestCase):
    def test_cap_does_not_use_physical_tolerance(self):
        case, *_ = fixture(); wrapped = solver.CallBudgetCase(case, .1)
        u = np.array([[0., 2.], [0., 0.]])
        self.assertTrue(wrapped.audit(u)['passed'])
        u[0, 1] = np.nextafter(2., np.inf)
        self.assertTrue(case.audit(u)['passed'])
        self.assertFalse(wrapped.audit(u)['passed'])
        self.assertGreater(wrapped.call_audit(u)['maximum_violation_yuan'], 0.)

    def test_filter_and_known_feasible_fallback(self):
        case, score, validation, forecast, anchor = fixture()
        rejected = np.array([[0., 1.8], [0., 1.8]])
        def fake(domain, *args, **kwargs):
            self.assertNotIn('over_cap', kwargs['seed_points'])
            self.assertIn(solver.ANCHOR_NAME, kwargs['seed_points'])
            self.assertIn('zero_warm_only', kwargs['seed_points'])
            self.assertIsInstance(domain, solver.CallBudgetCase)
            return dict(status='unresolved', lower_bound_yuan=0., global_optimality_verified=False), None
        with patch.object(solver.economic.engine, 'plan', side_effect=fake):
            receipt, u = solver.plan(case, score, validation, forecast, 1., anchor=anchor,
                env=None, seed_points={'over_cap':rejected, 'zero_warm_only':np.zeros(case.shape)})
        np.testing.assert_array_equal(u, anchor)
        self.assertTrue(receipt['fallback_used'])
        self.assertTrue(receipt['call_budget']['passed'])
        self.assertEqual(receipt['seed_count_accepted_without_anchor'], 1)
        self.assertFalse(receipt['lower_bound_valid_for_uncapped_problem'])
        self.assertAlmostEqual(receipt['cost_yuan'], .825)

    def test_engine_cap_violation_is_rejected_independently(self):
        case, score, validation, forecast, anchor = fixture()
        bad = np.array([[0., 1.7+1e-8], [0., 0.]])
        cheaper_valid_shared = np.array([[0., 1.2], [0., .5]])
        raw = dict(status='feasible', lower_bound_yuan=0., global_optimality_verified=True)
        with patch.object(solver.economic.engine, 'plan', return_value=(raw, bad)):
            receipt, u = solver.plan(case, score, validation, forecast, 1., anchor=anchor, env=None,
                seed_points={'cheaper_valid_shared':cheaper_valid_shared})
        np.testing.assert_array_equal(u, anchor)
        self.assertTrue(receipt['seed_filter'][0]['evidence_feasible'])
        self.assertEqual(receipt['fallback_reason'], 'invalid_engine_return')
        self.assertEqual(receipt['fallback_seed_name'], solver.ANCHOR_NAME)
        self.assertFalse(receipt['rejected_engine_point']['call_budget']['passed'])
        self.assertFalse(receipt['global_optimality_verified'])

    def test_anchor_is_not_replaced_if_infeasible(self):
        case, score, validation, forecast, _ = fixture()
        with patch.object(solver.economic.engine, 'plan') as native:
            receipt, u = solver.plan(case, score, validation, forecast, 1., anchor=np.zeros(case.shape), env=None)
        self.assertEqual(native.call_count, 0)
        self.assertEqual(receipt['status'], 'invalid_or_missing_anchor')
        self.assertIsNone(u)
        self.assertFalse(receipt['infeasibility_proved'])

    def test_locked_allowance(self):
        case, score, validation, forecast, anchor = fixture()
        with self.assertRaises(ValueError):
            solver.plan(case, score, validation, forecast, 1., anchor=anchor, env=None, allowance_yuan=.02)


def native_smoke(out):
    import gurobipy as gp
    case, score, validation, forecast, anchor = fixture()
    cap = case.fee(anchor)+.01
    z = NormalDist().inv_cdf(.9)
    required_J = .5*(z+math.sqrt(z*z+2*(1e-7+1e-8)))**2
    required_norm2 = required_J/2  # Four repeats per hour: J=2*(q0²+q1²).
    sum_q = cap/.05
    # At the optimum, both norm and calling budget are active. Smaller peak
    # spreads Q between the two hours; the budget limits how far it can spread.
    peak = .5*(sum_q+math.sqrt(2*required_norm2-sum_q*sum_q))
    expected = cap+.5*peak
    safe_sum = sum_q-2e-8
    safe_peak = .5*(safe_sum+math.sqrt(2*(required_norm2+1e-8)-safe_sum*safe_sum))
    safe = np.array([[0., safe_peak], [0., safe_sum-safe_peak]])
    seeds = dict(analytic_feasible=safe, rejected_uncapped_spread=np.array([[0., 1.], [0., 1.]]))
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        # Fixed over-budget action: the encoded linear constraint itself rejects it.
        with gp.Model('fixed_call_cap_check', env=env) as model:
            model.Params.OutputFlag = 0; model.Params.TimeLimit = 1.
            wrapped = solver.CallBudgetCase(case, cap)
            u, objective = wrapped.attach(model)
            model.addConstr(u == np.array([[0., 1.8], [0., 0.]]))
            model.setObjective(objective)
            model.optimize()
            assert model.Status == gp.GRB.INFEASIBLE
        receipt, u = solver.plan(case, score, validation, forecast, 1., anchor=anchor,
            env=env, seed_points=seeds, seconds=1., record_directory=out/'native')
    assert receipt['feasible'] and receipt['final_independent_audit']['passed']
    assert case.fee(u) <= cap
    assert receipt['lower_bound_yuan'] <= expected+1e-6
    assert abs(receipt['cost_yuan']-expected) < 3e-5
    assert receipt['call_budget']['allowance_tolerance_yuan'] == 0.
    assert receipt['additional_linear_constraints'] == 9 and receipt['added_quadratic_constraints'] == 0
    assert not receipt['call_budget_recording_errors']
    return dict(expected_restricted_optimum_yuan=expected, activation_cap_yuan=cap,
        encoded_over_cap_fixed_point_status='infeasible', cost_yuan=receipt['cost_yuan'],
        lower_bound_yuan=receipt['lower_bound_yuan'], gap=receipt['relative_gap'],
        final_call=receipt['call_budget'], final_source=receipt['source'],
        fallback_used=receipt['fallback_used'], native_status=receipt.get('native_status'),
        total_seconds=receipt['seconds'], action=u.tolist(),
        private_receipt=receipt['call_budget_receipt'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve earlier test receipts')
    args.out.mkdir(parents=True)
    text = io.StringIO()
    result = unittest.TextTestRunner(stream=text, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (args.out/'unit_tests.txt').write_text(text.getvalue(), encoding='utf-8')
    if not result.wasSuccessful():
        raise AssertionError(text.getvalue())
    report = dict(status='passed', unit_tests=result.testsRun,
        solver=solver.economic.source_ref(solver.__file__), test_source=solver.economic.source_ref(__file__),
        actual_networks_or_days_used=False, full_experiment_started=False,
        native_test=None if not args.native_smoke else native_smoke(args.out))
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
