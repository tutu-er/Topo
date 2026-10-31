"""LP method is part of cache identity; fake fits require no optimizer."""
from pathlib import Path
import functools
import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario
from research_experiments.wzzt_dr_selection.dr_pool_v2 import matrix_runner as candidate


def scene(date):
    ix=pd.date_range(date,periods=3,freq='h');columns=[201,202]
    p=pd.DataFrame([[.01,.02],[.02,.01],[.02,.02]],index=ix,columns=columns)
    s=make_observed_scenario(p,p*.3,p*0+.99,pd.Series(1.,index=ix),date)
    return s


class TestSharedFitterLPMethod(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def observed_fixture(self,tmp_path):
        self.folder=tmp_path
        self.train=[scene('2026-01-01')];self.valid=[scene('2026-01-02')]
        self.calls=[]

    def fake(self,training,validation,output_dir,*,root_label=1,search_seconds=25.,lp_method=1):
        self.calls.append(lp_method)
        return dict(status='complete',lp_method=lp_method,
            settings=dict(lp_method=lp_method,actual_lp_method=lp_method),LP_solves=[dict(lp_method=lp_method)])

    def fitter(self,fn=None):
        return candidate.SharedFitter(self.folder/'cache',sources={'fixture':'fixed'},fit_function=self.fake if fn is None else fn)

    def call(self,fitter,output,**kwargs):
        return fitter(self.train,self.valid,self.folder/output,**kwargs)

    def test_keys_and_independent_method_reuse(self):
        fitter=self.fitter()
        one=self.call(fitter,'one')
        two=self.call(fitter,'two',lp_method=2)
        one_again=self.call(fitter,'one_again',lp_method=1)
        two_again=self.call(fitter,'two_again',lp_method=2)
        self.assertNotEqual(one['matrix_cache']['cache_key'],two['matrix_cache']['cache_key'])
        self.assertEqual(one['matrix_cache']['cache_key'],one_again['matrix_cache']['cache_key'])
        self.assertEqual(two['matrix_cache']['cache_key'],two_again['matrix_cache']['cache_key'])
        self.assertEqual(self.calls,[1,2]);self.assertEqual(fitter.new_count,2);self.assertEqual(fitter.reused_count,2)
        for value,method in ((one,1),(two,2)):
            meta=json.loads(Path(value['matrix_cache']['cache_record']).read_text())
            self.assertEqual(meta['identity']['lp_method'],method)
            self.assertEqual(meta['lp_method'],method)
            self.assertEqual(meta['recorded_lp_method'],method)

    def test_old_default_fake_signature_still_works(self):
        calls=[]
        def old_fake(training,validation,output_dir,*,root_label=1,search_seconds=25.):
            calls.append('called');return {'status':'complete'}
        fitter=self.fitter(old_fake)
        one=self.call(fitter,'one');two=self.call(fitter,'two',lp_method=1)
        self.assertEqual(calls,['called']);self.assertTrue(two['matrix_cache']['reused'])
        self.assertEqual(one['matrix_cache']['lp_method_recording'],'legacy_default_unrecorded')
        self.assertIsNone(one['matrix_cache']['recorded_lp_method'])

    def test_partial_method_two_cannot_poison_default_one(self):
        fitter=self.fitter(functools.partial(self.fake,lp_method=2))
        with self.assertRaisesRegex(RuntimeError,'differs from requested'):
            self.call(fitter,'bad_default')
        good=self.call(fitter,'explicit_two',lp_method=2)
        self.assertEqual(good['matrix_cache']['lp_method'],2)
        self.assertEqual(self.calls,[2,2])

    def test_mixed_actual_and_declared_methods_rejected(self):
        def wrong(*args,**kwargs):
            return {'status':'complete','lp_method':2,'settings':{'lp_method':2,'actual_lp_method':1}}
        with self.assertRaisesRegex(RuntimeError,'differs from requested'):
            self.call(self.fitter(wrong),'mixed',lp_method=2)

    def test_nondefault_missing_method_record_rejected(self):
        def absent(*args,**kwargs):return {'status':'complete'}
        with self.assertRaisesRegex(RuntimeError,'explicit method record'):
            self.call(self.fitter(absent),'missing',lp_method=2)

    def test_cached_recording_checked_even_if_hash_receipt_matches(self):
        fitter=self.fitter();value=self.call(fitter,'original',lp_method=2)
        result=Path(value['matrix_cache']['cache_result']);meta_path=Path(value['matrix_cache']['cache_record'])
        content=json.loads(result.read_text());content['settings']['actual_lp_method']=0
        result.write_text(json.dumps(content),encoding='utf-8')
        meta=json.loads(meta_path.read_text());meta['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta),encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'differs from requested'):
            self.call(fitter,'reuse',lp_method=2)
        self.assertEqual(self.calls,[2])

    def test_new_saved_result_also_checked(self):
        def inconsistent(training,validation,output_dir,*,root_label=1,search_seconds=25.):
            Path(output_dir,'result.json').write_text(json.dumps({'status':'complete','lp_method':2}),encoding='utf-8')
            return {'status':'complete','lp_method':1}
        with self.assertRaisesRegex(RuntimeError,'differs from requested'):
            self.call(self.fitter(inconsistent),'saved_wrong')

    def test_identity_validation_and_numpy_integer(self):
        for method in (0,1,2,np.int64(2)):
            _,identity=candidate.observation_identity(self.train,self.valid,sources={'fixture':'fixed'},lp_method=method)
            self.assertEqual(identity['lp_method'],int(method))
        for method in (-1,3,True,np.bool_(True),1.,None,'2'):
            with self.subTest(method=repr(method)):
                with self.assertRaises(ValueError):
                    candidate.observation_identity(None,None,sources={},lp_method=method)
                with self.assertRaises(ValueError):
                    self.call(self.fitter(),'invalid',lp_method=method)
