from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario

spec=importlib.util.spec_from_file_location('zero_leaf_face_tested',Path(__file__).with_name('zero_leaf_face.py'))
face=importlib.util.module_from_spec(spec);spec.loader.exec_module(face)


class FaceTests(unittest.TestCase):
    def fixture(self):
        labels=[101,102];idx=pd.date_range('2021-01-01',periods=2,freq='h')
        p=pd.DataFrame(np.eye(2),index=idx,columns=labels);q=p*0
        v=pd.DataFrame(np.sqrt(1-np.diag([.01,0.])),index=idx,columns=labels)
        tr=[make_observed_scenario(p,q,v,pd.Series(1.,index=idx),'unit')]
        point=dict(supports=[[0],[1]],r=[.01,0.],x=[0.,0.])
        fit=dict(terminal_labels=labels,models=dict(multistart_original_rule=dict(point,selected_uncontracted=point)))
        return tr,fit

    def test_zero_return_is_not_unique_when_reactive_input_is_unobserved(self):
        tr,fit=self.fixture();out=face.audit_zero_leaves(tr,fit,loss_tolerances=(0.,))
        self.assertEqual(out['status'],'complete');self.assertEqual(out['detected_zero_leaf_tasks'],1)
        minimum,maximum=out['face_solves']
        self.assertAlmostEqual(minimum['value'],0.,places=9)
        self.assertAlmostEqual(maximum['value'],2.,places=9)
        self.assertLess(maximum['physical_mean_abs_error'],1e-10)

    def test_unbounded_face_is_retained_instead_of_being_reported_zero(self):
        tr,fit=self.fixture();out=face.audit_zero_leaves(tr,fit,bound=float('inf'),loss_tolerances=(0.,))
        self.assertEqual(out['status'],'complete')
        maximum=out['face_solves'][1]
        self.assertEqual(maximum['classification'],'unbounded');self.assertIsNone(maximum['value'])


if __name__=='__main__':unittest.main(verbosity=2)
