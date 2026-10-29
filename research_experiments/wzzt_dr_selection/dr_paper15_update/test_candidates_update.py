"""Small mechanism tests; no paper15 observation or truth is read."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario

spec=importlib.util.spec_from_file_location('candidate_update_tested',Path(__file__).with_name('candidates_update.py'))
update=importlib.util.module_from_spec(spec);spec.loader.exec_module(update)


def fixture():
    labels=[101,102];family=[(0,),(1,),(0,1)]
    def scene(date,drop):
        index=pd.date_range(date,periods=2,freq='h')
        p=pd.DataFrame(np.eye(2),columns=labels,index=index);q=p*0
        v=pd.DataFrame(np.sqrt(1-np.asarray(drop)),columns=labels,index=index)
        return make_observed_scenario(p,q,v,pd.Series(1.,index=index),date)
    # One fixed star is observed in both splits. Training has off-diagonal
    # measurement errors, creating an exactly fitted spurious common block.
    training=[scene('2021-01-01',[[.02,.005],[.005,.02]])]
    validation=[scene('2021-01-02',[[.02,0],[0,.02]])]
    point=dict(supports=family,r=[.015,.015,.005],x=[0,0,0])
    original=dict(terminal_labels=labels,bank=[dict(point,native_family=family,config='unit_parent')],
        searches=dict(primary=dict(path=[point])),calibrations=dict(raw=dict(a=1.,b=0.,median_root_depth=.0001)),models={})
    return training,validation,original


class CandidateTests(unittest.TestCase):
    def test_nested_deletion_needs_validation_and_joint_refit(self):
        tr,va,fit=fixture();out=update.fit_with_contractions(tr,va,fit)
        self.assertEqual(out['status'],'complete');self.assertEqual(out['planned_unique_families'],2)
        parent=next(c['point'] for c in out['candidates'] if len(c['family'])==3)
        deleted=next(c['point'] for c in out['candidates'] if len(c['family'])==2)
        self.assertLess(parent['objective'],1e-12)
        self.assertGreater(deleted['objective'],parent['objective']+1e-4)
        self.assertLess(deleted['validation_mae'],1e-12)
        self.assertGreater(parent['validation_mae'],1e-4)
        np.testing.assert_allclose(deleted['r'],[.02,.02],atol=1e-10)
        self.assertEqual(out['models']['rnj_refit']['supports'],[[0],[1],[0,1]])
        self.assertEqual(out['models']['rnj_contractions']['supports'],[[0],[1]])

    def test_apply_same_neighbor_expansion_to_rnj_and_wzzt(self):
        tr,va,fit=fixture();out=update.fit_with_contractions(tr,va,fit)
        for c in out['candidate_plans']:
            self.assertTrue({(0,),(1,)}<=set(map(tuple,c['family'])))
            self.assertEqual({p['source'] for p in c['provenance']},{'rnj_bank','wzzt_path'})
        a=out['models']['rnj_contractions'];b=out['models']['wzzt_contractions']
        self.assertEqual(a['supports'],b['supports']);self.assertEqual(a['group_candidate_count'],2)
        self.assertEqual(b['group_candidate_count'],2)
        self.assertAlmostEqual(a['validation_mae'],b['validation_mae'],places=12)

    def test_preflight_cap_performs_no_fit_and_does_not_drop_parents(self):
        tr,va,fit=fixture();called=[]
        def forbidden(*args,**kwargs):called.append(True);raise AssertionError('LP factory called after failed preflight')
        out=update.fit_with_contractions(tr,va,fit,max_families=1,lp_factory=forbidden)
        self.assertEqual(out['status'],'not_run_family_cap');self.assertEqual(out['planned_unique_families'],2)
        self.assertEqual(called,[]);self.assertEqual(out['models'],{})

    def test_no_leaf_deletion_or_crossing_input_is_allowed(self):
        _,_,fit=fixture();bad=deepcopy(fit);bad['bank'][0]['native_family']=[(0,),(0,1)]
        with self.assertRaises(ValueError):update.enumerate_contractions(bad)


if __name__=='__main__':unittest.main(verbosity=2)
