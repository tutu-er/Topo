"""Boundary cases newly exposed by a multi-network, multi-family panel."""
from copy import deepcopy
import unittest

import numpy as np

from ..distribution_gate_20261114.test_evidence import toy
from ..distribution_gate_20261114.evidence import DistributionEvidence
from ..distribution_gate_20261114.solver import plan
from ..reactive_flex_20261114.domain import PQIncrementCase
from .test_resources import fixture
from .resources import build_resources
from .run import rotation_actions, MODES
from ..flexibility_stability_20261114.solver import cost_breakdown


class PanelBoundaryTests(unittest.TestCase):
    def test_single_family_zero_is_vacuously_feasible_in_all_three_modes(self):
        case, score, val, forecast, _ = toy()
        score['parameters'] = score['parameters'][:1]
        score['models'] = score['models'][:1]
        score['native_bank'] = [score['native_bank'][0], score['native_bank'][3]]
        case.audit = lambda u: dict(passed=bool(np.all(u >= case.lower) and np.all(u <= case.upper)))
        for mode in MODES:
            with self.subTest(mode=mode):
                model = DistributionEvidence(case, score, val, forecast, 200., mode=mode)
                direct = model.verify_expansion(np.zeros(case.shape))
                self.assertTrue(direct['feasible'])
                self.assertEqual(len(model.rows), 0)
                self.assertEqual(direct['worst_recovery_lower'], 1.)
                receipt, point = plan(case,score,val,forecast,200.,mode=mode,seconds=1.,env=None)
                self.assertEqual(receipt['status'], 'zero_sufficient')
                self.assertFalse(receipt['optimizer_invoked'])
                np.testing.assert_array_equal(point, np.zeros(case.shape))

    def test_three_rotation_bills_match_without_changing_feasible_domain(self):
        entry, score, context, train = fixture()
        resources = build_resources(entry,score,context,train,100.)
        forecast = np.tile(np.r_[np.full(4,.05),np.full(4,.01)], (96,1))
        case = PQIncrementCase(context,train['labels'],forecast,100.,q_resources=resources['q_resources'])
        # Construct feasible zero-energy active movements via the actual linear resource LP.
        direction = np.where((np.arange(case.H)[:,None]//2+np.arange(case.R)[None,:])%2==0,1.,-1.)
        full, _ = case.linear_optimum(direction, least_fee=True)
        controls = {m:s*full for m,s in zip(MODES,(0.,.2,.9))}
        before = deepcopy(controls)
        receipts = rotation_actions(case,controls)
        for m in MODES:
            np.testing.assert_array_equal(controls[m], before[m])
            self.assertTrue(case.audit(controls[m+'_rotation'])['passed'])
            self.assertTrue(receipts[m]['same_planned_fee'])
            self.assertAlmostEqual(cost_breakdown(case,controls[m])['total_yuan'],
                                   cost_breakdown(case,controls[m+'_rotation'])['total_yuan'],places=8)


if __name__ == '__main__':
    unittest.main()
