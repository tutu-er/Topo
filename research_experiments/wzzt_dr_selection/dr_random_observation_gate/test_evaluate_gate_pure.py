import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location('independent_eval',Path(__file__).with_name('evaluate_gate.py'))
evaluate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(evaluate)


class EvaluatorPureTests(unittest.TestCase):
    def test_labels_indices_and_zero_weak_accounting(self):
        p=dict(supports=[[0],[1],[0,1]],support_labels=[[101],[102],[101,102]],r=[0.,.2,5e-10],x=[0.,0.,0.])
        target=[(1,),(0,1)]
        numeric=evaluate.score(p,[101,102],target)
        strict=evaluate.score(p,[101,102],target,0.)
        self.assertEqual(numeric['missed'],[[0,1]])
        self.assertEqual(numeric['errors'],1)
        self.assertEqual(strict['errors'],0)
        self.assertNotIn([0],numeric['predicted'])

    def test_raw_and_final_matrices_and_losses_are_separate(self):
        raw=dict(supports=[[0],[1],[0,1]],r=[.1,.2,.3],x=[0.,0.,0.])
        final=dict(supports=[[0],[1]],r=[.1,.2],x=[0.,0.],selected_uncontracted=raw)
        P=np.array([[1.,2.],[2.,1.]]);Q=np.zeros_like(P)
        R=np.array([[.4,.3],[.3,.5]]);Y=P@R.T
        self.assertAlmostEqual(evaluate.loss(raw,[101,102],(P,Q,Y)),0.)
        self.assertGreater(evaluate.loss(final,[101,102],(P,Q,Y)),0.)
        self.assertIs(evaluate.raw_point(final),raw)

    def test_unresolved_does_not_become_zero_error(self):
        row=dict(method='direct',snapshot_id='s0',status='no_direct_family',final=None,raw=None)
        summary=evaluate.summarize([row],['method'])[0]
        self.assertEqual(summary['planned'],1)
        self.assertEqual(summary['known'],0)
        self.assertEqual(summary['unknown'],1)
        self.assertIsNone(summary['final_errors'])


if __name__=='__main__':unittest.main()
