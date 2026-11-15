from copy import deepcopy
import unittest
import numpy as np

from .feedback import FeedbackController,seal_monitor,evaluate_monitor
from .state import ScoreState,FrozenBank
from .test_evidence import fixture


class FeedbackTests(unittest.TestCase):
    def test_negative_score_explained_by_competitor_is_not_outside_all(self):
        bank,_,_=fixture();snap=ScoreState(bank).snapshot();monitor=seal_monitor(bank,snap)
        x=np.array([1.,.2]);y=bank.means[1]@x
        result=evaluate_monitor(bank,monitor,x,y)
        self.assertLess(result['actual_new_logodds'],0)
        self.assertFalse(result['outside_union']);self.assertIn(1,result['explained_by_native'])
        controller=FeedbackController();result=controller.observe(bank,monitor,x,y,previous_target=0,current_target=1)
        self.assertEqual(result['state'],'caution');self.assertEqual(result['extra_action_fraction'],.25)
        self.assertFalse(result['causal_procurement_gain_observed'])

    def test_union_is_not_convex_hull_and_two_hours_are_eight_steps(self):
        bank,_,_=fixture();base=seal_monitor(bank,ScoreState(bank).snapshot());controller=FeedbackController()
        middle_x=np.array([1.,.2])
        middle_y=(bank.means[0]@middle_x+bank.means[1]@middle_x)/2
        middle=evaluate_monitor(bank,base,middle_x,middle_y)
        self.assertTrue(middle['outside_union'])
        self.assertGreater(middle['actual_new_logodds'],min(middle['mean_under_each_native']))
        self.assertLess(middle['actual_new_logodds'],max(middle['mean_under_each_native']))
        x=np.array([1.,.2]);y=np.array([100.])
        for k in range(8):
            monitor=dict(base,step=k)
            result=controller.observe(bank,monitor,x,y,previous_target=0,current_target=0)
            self.assertEqual(result['state'],'pause' if k==7 else 'caution')
        self.assertEqual(result['elapsed_confirmed_outside_hours'],2.)
        self.assertEqual(result['extra_action_fraction'],0.)
        for k in range(8,16):
            result=controller.observe(bank,dict(base,step=k),x,bank.means[0]@x,previous_target=0,current_target=0)
            self.assertEqual(result['state'],'caution' if k==15 else 'pause')
        result=controller.observe(bank,dict(base,step=16),x,bank.means[0]@x,previous_target=0,current_target=0)
        self.assertEqual(result['state'],'normal')

    def test_missing_monitor_never_fabricates_recovery(self):
        bank,_,_=fixture();base=seal_monitor(bank,ScoreState(bank).snapshot());controller=FeedbackController()
        x=np.array([1.,.2])
        for k in range(8):controller.observe(bank,dict(base,step=k),x,[100.],previous_target=0,current_target=0)
        missing=dict(base,step=8,pair=None,available=False,competitor_native=None)
        result=controller.observe(bank,missing,x,[1.],previous_target=0,current_target=0)
        self.assertEqual(result['state'],'pause');self.assertIsNone(result['passed'])
        self.assertEqual(result['recovery_streak'],0)
        checkpoint=controller.snapshot();other=FeedbackController();other.restore(checkpoint)
        self.assertEqual(other.snapshot(),checkpoint)

    def test_presealed_pair_and_chronology_are_required(self):
        bank,_,_=fixture();monitor=seal_monitor(bank,ScoreState(bank).snapshot());controller=FeedbackController()
        with self.assertRaises(ValueError):controller.observe(bank,dict(monitor,step=1),[1.,.2],[1.],previous_target=0,current_target=0)
        with self.assertRaises(ValueError):controller.observe(bank,monitor,[1.,.2],[1.],previous_target=1,current_target=0)


if __name__=='__main__':unittest.main()
