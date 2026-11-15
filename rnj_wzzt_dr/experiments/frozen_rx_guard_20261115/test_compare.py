import copy
import unittest

from .compare import baseline_identity, paired_response, unique


def row(repeat,correct,*,status='passed',origin='new_guard_execution',strategy='normal_fixed__response'):
    return dict(id='case',forecast=0,repeat=repeat,strategy=strategy,data_origin=origin,source='same/path',
        audit_status=status,original_status='complete',evaluation=dict(native_correct=correct,native_selected_index=int(correct)),
        planned_cost={'total_yuan':1},settled_cost={'total_yuan':1})


class ComparisonTests(unittest.TestCase):
    def test_paired_loss_and_gain_preserve_failure(self):
        a={'path_records':[row(0,True),row(1,False),row(2,False,status='recorded_failure')]}
        b={'path_records':[row(0,False),row(1,True),row(2,True)]}
        p=paired_response(a,b,'case','normal_fixed')
        self.assertEqual((p['n'],p['new_recoveries'],p['lost_recoveries'],p['tight_recorded_failures']),(3,2,1,1))

    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):paired_response({'path_records':[row(0,True)]},{'path_records':[]},'case','normal_fixed')

    def test_baseline_counted_once_and_source_checked(self):
        a={'path_records':[row(0,True,origin='reused_baseline',strategy='no_dfs')]};b=copy.deepcopy(a)
        self.assertEqual(len(baseline_identity(a,b)),1)
        b['path_records'][0]['source']='different/path'
        with self.assertRaises(ValueError):baseline_identity(a,b)

    def test_baseline_content_checked(self):
        a={'path_records':[row(0,True,origin='reused_baseline',strategy='no_dfs')]};b=copy.deepcopy(a)
        b['path_records'][0]['evaluation']['native_correct']=False
        with self.assertRaises(ValueError):baseline_identity(a,b)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):unique([row(0,True),row(0,True)])


if __name__=='__main__':unittest.main()
