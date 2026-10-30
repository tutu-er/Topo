"""No solver, runtime helper, simulator or private observations are imported."""
import copy
import unittest
from completion import METHODS,complete,complete_fit,key,pending,merge_rows

def row(n,done=True):
    return dict(flow='r0',case=str(n),menu_id='zero',evaluated={m:dict(status='complete' if done else 'not_started_budget',all_scans_complete=True) for m in METHODS})

class CompletionTests(unittest.TestCase):
    def test_pending_includes_missing_method_and_partial_scan(self):
        a,b,c=row(1),row(2),row(3);b['evaluated'].pop(METHODS[2]);c['evaluated'][METHODS[1]]['all_scans_complete']=False
        self.assertEqual([key(r) for r in pending([a,b,c])],[key(b),key(c)])
    def test_pending_preserves_original_order_and_does_not_mutate(self):
        source=[row(3,False),row(2),row(1,False)];before=copy.deepcopy(source);p=pending(source);p[0]['case']='changed'
        self.assertEqual(source,before);self.assertEqual([r['case'] for r in pending(source)],['3','1'])
    def test_duplicate_source_or_update_jobs_rejected(self):
        a=row(1,False)
        with self.assertRaises(ValueError):pending([a,a])
        with self.assertRaises(ValueError):merge_rows([a],[row(1),row(1)])
    def test_completed_rows_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):merge_rows([row(1)],[row(1)])
    def test_unplanned_scope_cannot_be_added(self):
        with self.assertRaises(ValueError):merge_rows([row(1,False)],[row(2)])
    def test_merge_preserves_done_exactly_and_replaces_only_pending(self):
        a,b=row(1),row(2,False);a['frozen_payload']={'x':[1,2,3]};source=[a,b];done=row(2)
        merged=merge_rows(source,[done]);self.assertEqual(merged[0],a);self.assertEqual(merged[1],done);self.assertEqual(source[1],b)
    def test_fit_cache_requires_every_complete_method_and_scan(self):
        fit={'methods':{m:dict(status='complete',all_scans_complete=True) for m in METHODS}}
        self.assertTrue(complete_fit(fit));fit['methods'][METHODS[1]]['all_scans_complete']=False;self.assertFalse(complete_fit(fit))

if __name__=='__main__':unittest.main()
