"""Checks that rejection scores and failure denominators follow the declared rule."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ... import io
from ...observations import Observation
from .experiment import rejection_scores, summarize


class ExperimentChecks(unittest.TestCase):
    def test_reference_score_matches_augmented_softmax(self):
        score = dict(sigma2=1., cells=2,
            parameters=[dict(H=[[0., 0.]], history_sse=0.), dict(H=[[2., 0.]], history_sse=1.)],
            models=[dict(topology_id='target'), dict(topology_id='other')])
        observed = Observation(np.ones((2, 1)), np.zeros((2, 1)), np.ones((2, 1)), (101,))
        results, loss = rejection_scores(score, observed, dict(truth_id='target'), 0, .5)
        np.testing.assert_allclose(loss, [2., 3.])
        raw = np.exp(-np.r_[loss, 4.]/2)
        raw /= raw.sum()
        self.assertAlmostEqual(results['2.0']['p_target'], raw[0])
        self.assertAlmostEqual(results['2.0']['rejection_weight'], raw[-1])
        self.assertTrue(results['2.0']['correct'])
        tighter, _ = rejection_scores(score, observed, dict(truth_id='target'), 0, .1)
        self.assertTrue(tighter['2.0']['rejected'])
        self.assertFalse(tighter['2.0']['correct'])

    def test_missing_truth_never_counted_as_recovery(self):
        score = dict(sigma2=1., cells=2, parameters=[dict(H=[[0., 0.]], history_sse=0.)],
            models=[dict(topology_id='target')])
        observed = Observation(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)), (101,))
        result, _ = rejection_scores(score, observed, dict(truth_id='absent'), 0, 1.)
        self.assertIsNone(result['2.0']['p_true'])
        self.assertFalse(result['2.0']['correct'])
        self.assertTrue(result['2.0']['wrong'])

    def test_failed_days_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            io.write(out/'protocol.json', dict(total_new_days=2,
                entries=[dict(id='history', role='weak', new_days=2)], strategies=['no_dfs']))
            row = dict(id='history', day=0, strategy='no_dfs', status='complete', native_map_correct=True,
                map_correct=True, p_true=.8, p_target=.8, settled_fee_yuan=0., delivered_abs_kwh=0.,
                zero_fallback=False, rejection_rules={'2.0': dict(correct=True, wrong=False, rejected=False)})
            io.write(out/'confirmation/history/day000/results.json', [row])
            io.write(out/'confirmation/history/day000/complete.json', dict(status='complete'))
            io.write(out/'confirmation/history/day001/worker_failure.json', dict(status='worker_failed'))
            summary = summarize(out)
            group = summary['groups']['history']['strategies']['no_dfs']
            self.assertEqual(group['planned_days'], 2)
            self.assertEqual(group['completed_days'], 1)
            self.assertEqual(group['native_correct'], 1)
            self.assertEqual(summary['failed_jobs'], 1)


if __name__ == '__main__':
    unittest.main()
