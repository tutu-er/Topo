"""Pure device/lock checks only: no physical profile, AC, or LP invocation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import coded_physical as coded


class CodedPhysicalPureTests(unittest.TestCase):
    def test_three_templates_fixed_task_and_no_expanded_power(self):
        self.assertEqual(coded.pure_checks()['status'],'passed')

    def test_negative_net_Q_does_not_become_available_EV(self):
        p=np.full((96,2),.02);q=np.full((96,2),.008);q[76:84,0]=-.0001
        _,_,r=coded.apply_coded_response(p,q,[1,1],[True,False],[1,-1],'opposed',[0,1])
        self.assertEqual(r['status'],'contract_not_available')
        self.assertEqual(r['accepted_participants'],[1,0])
        self.assertIsNone(r['actual_payment'])
        self.assertEqual(r['committed_payment'],.2)

    def test_nonaccepted_bad_quote_is_not_execution_failure(self):
        p=np.full((96,2),.02);q=np.full((96,2),.008);q[76:84,0]=-.0001
        actual,_,r=coded.apply_coded_response(p,q,[1,1],[False,True],[1,-1],'opposed',[0,1])
        self.assertEqual(r['status'],'complete')
        self.assertFalse(r['quote_qualification']['quote_qualification_pass'])
        self.assertTrue(np.array_equal(actual[:,0],p[:,0]))

    def test_explicit_lock_device_required_and_unchanged(self):
        lock=dict(status='frozen',labels=[101,102],active_labels=[101,102],
                  plans={'a':[[1,1],[1,1]]},directions={'a':[[1,-1],[1,-1]]},
                  templates={'a':'opposed'})
        fake=SimpleNamespace(validate_plan=lambda value,n:np.asarray(value,int))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path=Path(temporary)/'lock.json'
            path.write_text(json.dumps(lock),encoding='utf-8')
            with self.assertRaises(RuntimeError):coded._lock(path,coded.sha(path),'a',fake)
            lock['device_contract']=dict(baseline_kw=.2,baseline_kvar=.07,upper_kw=.4,
                start_hour=19,end_hour=21,initial_energy_kWh=0,final_energy_kWh=.4,efficiency=1.,non_V2G=True)
            path.write_text(json.dumps(lock),encoding='utf-8')
            self.assertEqual(coded._lock(path,coded.sha(path),'a',fake)[2],'opposed')
            lock['device_contract']['upper_kw']=.8
            path.write_text(json.dumps(lock),encoding='utf-8')
            with self.assertRaises(RuntimeError):coded._lock(path,coded.sha(path),'a',fake)


if __name__=='__main__':unittest.main()
