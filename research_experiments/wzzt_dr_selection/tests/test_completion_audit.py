import unittest
from unittest.mock import patch

import numpy as np

from research_experiments.wzzt_dr_selection import completion_audit as audit
from research_experiments.wzzt_dr_selection.experiment import History,cases


class CompletionAuditTests(unittest.TestCase):
    def test_wrong_tentative_block_does_not_restrict_completions(self):
        _,h,_,_=cases()[0]
        wrong=History(h.P,h.Q,h.Y,h.selected+((0,2),))
        result=audit.analyze_history(wrong)
        rows={tuple(r['C']):r for r in result['candidates']}
        self.assertEqual(len(rows),4)
        self.assertIn((0,1),rows)
        self.assertEqual(rows[(0,2)]['classification'],'legal_alternative_found')
        self.assertAlmostEqual(rows[(0,2)]['reference_upper'],0)
        self.assertEqual(result['lp_calls']['complete_family_reference'],8)

    def test_exact_and_relaxed_omission_losses_are_distinct(self):
        R=np.array([[3.,2.,1.],[2.,3.,1.],[1.,1.,2.]])
        h=audit.extra_history(3,R,np.eye(3))
        result=audit.analyze_history(h)
        rows={tuple(r['C']):r for r in result['candidates']}
        self.assertAlmostEqual(rows[(0,1)]['reference_upper'],2)
        self.assertAlmostEqual(rows[(0,1)]['matrix_lower'],2)
        self.assertAlmostEqual(rows[(0,1,2)]['reference_upper'],4)
        self.assertAlmostEqual(rows[(0,1,2)]['matrix_lower'],2)
        self.assertTrue(all(not r['lower_exceeds_reference'] for r in rows.values()))

    def test_incompatible_history_is_not_vacuous_confirmation(self):
        Y=np.array([[2.,1.5,0.],[1.,2.,0.],[0.,0.,1.]])
        h=History(np.eye(3),np.zeros((3,3)),Y,((0,),(1,),(2,)))
        result=audit.analyze_history(h)
        self.assertFalse(result['compatible_model_exists'])
        self.assertTrue(all(r['classification']=='model_budget_incompatible' for r in result['candidates']))

    def test_cache_reuses_equivalent_root_order_and_rejects_new_data(self):
        P=np.eye(3);Q=np.zeros_like(P);Y=np.eye(3)+np.ones((3,3))
        cache=audit.BranchCache(P,Q,Y)
        a=cache(P,Q,Y,order=(0,1,-1))
        b=cache(P,Q,Y,order=(1,0,-1))
        self.assertEqual(cache.calls,1)
        self.assertEqual(cache.hits,1)
        self.assertEqual(a['loss'],b['loss'])
        with self.assertRaises(ValueError):
            cache(P,Q,Y+1,order=(0,1,-1))

    def test_analysis_has_no_truth_or_case_catalogue_dependency(self):
        _,h,_,_=cases()[0]
        with patch.object(audit,'specifications',side_effect=AssertionError('truth catalogue')), \
             patch.object(audit,'annotate',side_effect=AssertionError('truth labels')):
            r=audit.analyze_history(h)
        self.assertTrue(r['compatible_model_exists'])
        self.assertEqual(r['confirmed_used'],[])
        self.assertFalse(r['statistical_coverage'])



    def test_conflicting_lower_bound_cannot_report_necessity(self):
        _,h,_,_=cases()[0]
        fake={'lower_bound':999.,'success':True,'status':'optimal'}
        with patch.object(audit.bounds,'block_omission_lower_bound',return_value=fake):
            result=audit.analyze_history(h)
        self.assertTrue(all(r['classification']=='invalid_bound' for r in result['candidates']))
        self.assertTrue(all(r['lower_exceeds_reference'] for r in result['candidates']))


if __name__=='__main__':
    unittest.main()
