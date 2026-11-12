import unittest
from .analyze import summary,case_summaries


def observation(strategy,day=0,**changes):
    row=dict(id='case',case='unit_case',day=day,strategy=strategy,status='complete',p_true=.6,p_target=.6,
        map_correct=True,correct_unique=False,wrong_unique=False,true_pass90=False,
        retained_count=2,missing=0,extra=0,cost=1.,request_rms=1.,request_max=1.,
        absolute_DR_kwh=1.,voltage_violation_cells=0,negative_P_fraction=0.,planning_failed=False)
    row.update(changes);return row


class Tests(unittest.TestCase):
    def test_failed_AC_fee_and_recovery_denominator_are_retained(self):
        rows=[observation('no_dr',0),observation('no_dr',1),
              observation('optimized',0,status='AC_or_scoring_failed',cost=9.,request_max=3.),
              observation('optimized',1,p_true=.9)]
        row=next(r for r in summary(rows,'all','unit') if r['strategy']=='optimized')
        self.assertEqual(row['mean_fee'],5.)
        self.assertEqual(row['max_request'],3.)
        self.assertEqual(row['full_recovery_rate_all'],.5)
        self.assertEqual(row['failed_paths'],1)
        self.assertEqual(row['posterior_paths_with_truth'],1)
        self.assertAlmostEqual(row['mean_true_posterior'],.9)

    def test_missing_true_candidate_has_no_fabricated_posterior(self):
        row=summary([observation('no_dr',p_true=None,map_correct=False)],'missing','unit')[0]
        self.assertIsNone(row['mean_true_posterior'])
        self.assertEqual(row['posterior_paths_with_truth'],0)
        self.assertEqual(row['full_recoveries'],0)
        self.assertEqual(row['planned_paths'],1)
        example=case_summaries([observation('no_dr',p_true=None,map_correct=False),
            observation('optimized_g4',p_true=None,map_correct=False,cost=9.)],'unit')[0]
        self.assertIsNone(example['baseline_mean'])
        self.assertEqual(example['mean_fee'],9.)


if __name__=='__main__':unittest.main()
