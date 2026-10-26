import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from research_experiments.wzzt_dr_selection import face_price_audit as audit
from research_experiments.wzzt_dr_selection.core import block_features
from research_experiments.wzzt_dr_selection.experiment import checked_fit, cases
from research_experiments.wzzt_dr_selection.optimal_face import fit_future_on_history_optimal_face


class FacePriceAuditTests(unittest.TestCase):
    def test_dual_combination_brackets_finite_batch_beyond_local_range(self):
        Ph, Qh, Yh = np.ones((1,1)), np.zeros((1,1)), np.zeros((1,1))
        Pf, Qf, Yf = 2*np.ones((1,1)), np.zeros((1,1)), 2*np.ones((1,1))
        blocks = ((0,),)
        h = checked_fit(Ph,Qh,Yh,blocks)
        f = fit_future_on_history_optimal_face(Ph,Qh,Yh,Pf,Qf,Yf,blocks)
        self.assertTrue(f['success'])
        self.assertAlmostEqual(f['future_loss'],2)
        for w in audit.WEIGHTS:
            interval = audit.dual_interval(h,f,block_features(Ph,Qh,blocks),
                block_features(Pf,Qf,blocks),Yh.ravel(),Yf.ravel(),w)
            actual = checked_fit(np.column_stack([Ph,w*Pf]), np.column_stack([Qh,w*Qf]),
                                 np.column_stack([Yh,w*Yf]),blocks)
            self.assertTrue(interval['numerically_feasible'])
            self.assertLessEqual(interval['lower'], actual.loss+1e-7)
            self.assertGreaterEqual(interval['upper'], actual.loss-1e-7)
            if interval['local_range']:
                self.assertAlmostEqual(interval['lower'],interval['upper'])
        self.assertAlmostEqual(actual.loss,1)

    def test_numerical_winner_must_exclude_competitors_and_no_gain(self):
        intervals = {None:{'lower':10,'upper':10,'numerically_feasible':True},
                     (0,1):{'lower':2,'upper':3,'numerically_feasible':True},
                     (0,2):{'lower':4,'upper':5,'numerically_feasible':True}}
        self.assertEqual(audit.numerical_winner(intervals),((0,1),))
        intervals[(0,2)]['lower']=2.5
        self.assertEqual(audit.numerical_winner(intervals),())
        intervals[(0,2)]['lower']=4
        intervals[None]['lower']=2
        self.assertEqual(audit.numerical_winner(intervals),())

    def test_prefix_keeps_a_wrong_observed_proposal(self):
        fake = SimpleNamespace(stopped=False,best_candidates=((0,2),))
        with patch.object(audit,'select',return_value=fake):
            specs,metadata = audit.audit_cases()
        self.assertEqual(len(specs),7)
        for name,h,_,ev in specs[4:]:
            self.assertIn((0,2),h.selected)
            self.assertNotIn((0,2),ev.true_blocks)
            self.assertEqual(metadata[name]['selected'],h.selected)

    def test_design_cannot_access_evaluator_or_case_catalogue(self):
        _,h,f,_ = cases()[0]
        with patch.object(audit,'cases',side_effect=AssertionError('truth catalogue')), \
             patch.object(audit,'observed_path',side_effect=AssertionError('evaluator')):
            d = audit.design_case(h,f)
        self.assertEqual(len(d['menus']),4)
        self.assertFalse(d['parameter_coverage'])
        self.assertEqual(d['failures'],[])
        for row in d['rows']:
            self.assertTrue(row['numerical_winner_matches_exact'])
            self.assertTrue(all(v['brackets_exact'] for v in row['intervals'].values()))

    def test_common_input_stops_without_fabricating_full_recovery(self):
        _,h,_,ev = cases()[2]
        result = audit.observed_path(h,ev)
        self.assertEqual(result['outcome'],'stopped')
        self.assertFalse(result['full_recovery'])
        self.assertFalse(result['next_correct'])
        self.assertEqual(result['target_total'],2)



    def test_all_menu_pairs_include_strict_reversal(self):
        r=audit.rank_pairs([1,0,2,3,4,5],[0,1,2,3,4,5])
        self.assertEqual(r['pairs'],15)
        self.assertEqual(len(r['strict_reversals']),1)

    def test_design_lock_conflict_preserves_existing_bytes(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'design_lock.json'
            audit.save(p,{'fingerprint':audit.digest({'old':1})})
            before=p.read_bytes()
            with self.assertRaises(RuntimeError):
                audit.validate_design_lock(p,{'new':2})
            self.assertEqual(p.read_bytes(),before)

    def test_design_failure_is_retained_and_not_silently_retried(self):
        from pathlib import Path
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as d:
            error=audit.DesignAuditFailure('deliberate fixture',{'failed_pattern':[1,0,0]})
            with patch('sys.argv',['audit','--output',d]), \
                 patch.object(audit,'design_case',side_effect=error):
                with self.assertRaises(audit.DesignAuditFailure):
                    audit.main()
            p=Path(d)/'failure_design_omitted_lines.json'
            value=json.loads(p.read_text())
            self.assertEqual(value['details']['failed_pattern'],[1,0,0])
            before=p.read_bytes()
            with patch('sys.argv',['audit','--output',d]), \
                 patch.object(audit,'design_case',side_effect=AssertionError('must not retry')):
                with self.assertRaisesRegex(RuntimeError,'retained design failure'):
                    audit.main()
            self.assertEqual(p.read_bytes(),before)

    def test_correct_root_proposal_can_leave_weak_block_unrecovered(self):
        _,h,_,ev=cases()[1]
        P,Q=audit.matrix_for_pattern((0,0,1))
        result=audit.observed_path(h,ev,P,Q)
        self.assertTrue(result['next_correct'])
        self.assertFalse(result['full_recovery'])
        self.assertNotIn((0,1),result['final_effective_blocks'])
        P,Q=audit.matrix_for_pattern((1,0,0))
        result=audit.observed_path(h,ev,P,Q)
        self.assertTrue(result['full_recovery'])

if __name__ == '__main__':
    unittest.main()
