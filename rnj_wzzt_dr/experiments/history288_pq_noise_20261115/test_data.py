"""Independent split/noise, source identity and no-budget-reset checks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import data


def clean_fixture(n=3):
    t=np.arange(384)[:,None]
    p=.03+.0001*t+np.arange(n)[None]*.01
    q=-.004+.00002*t+np.arange(n)[None]*.003
    root=1.02+.001*np.sin(np.arange(384)/7.)
    v=root[:,None]-.002-np.arange(n)[None]*.001
    return dict(P=p,Q=q,V=v,root=root,labels=np.arange(301,301+n),
                timestamps=np.asarray([str(i) for i in range(384)]))


def standard(n=3,index=18,repeat=0):
    history=data.split_noise(n,index,repeat,split=0)
    evaluation=data.split_noise(n,index,repeat,split=1,path=0)
    return {key:np.concatenate([history[key],evaluation[key]]) for key in history}


class DataTests(unittest.TestCase):
    def test_all_history_rows_and_disjoint_new_day(self):
        idx=data.sample_indices()
        self.assertEqual([len(v) for v in idx.values()],[192,96,96])
        np.testing.assert_array_equal(np.concatenate(list(idx.values())),np.arange(384))
        self.assertFalse(set(idx['train'])&set(idx['validation']))
        self.assertFalse(set(idx['validation'])&set(idx['new_day']))

    def test_PQ_levels_do_not_change_voltage_and_zero_is_exact(self):
        clean=clean_fixture();before=deepcopy(clean);noise=standard()
        reference=data.observed_splits(clean,noise,0.)
        for sd in data.PQ_SDS:
            current=data.observed_splits(clean,noise,sd)
            for name,idx in data.sample_indices().items():
                obs=current[name]
                np.testing.assert_array_equal(obs.p,clean['P'][idx]*(1+sd*noise['P'][idx]))
                np.testing.assert_array_equal(obs.q,clean['Q'][idx]*(1+sd*noise['Q'][idx]))
                np.testing.assert_array_equal(obs.voltage,reference[name].voltage)
                np.testing.assert_array_equal(obs.root,reference[name].root)
                np.testing.assert_array_equal(obs.y,reference[name].y)
                np.testing.assert_array_equal(obs.y,obs.root[:,None]**2-obs.voltage**2)
        np.testing.assert_array_equal(reference['train'].p,clean['P'][:192])
        np.testing.assert_array_equal(reference['validation'].q,clean['Q'][192:288])
        for key in clean:np.testing.assert_array_equal(clean[key],before[key])

    def test_noise_reproducibility_and_independent_coordinates(self):
        a=data.split_noise(3,18,0,split=0)
        again=data.split_noise(3,18,0,split=0)
        evaluation=data.split_noise(3,18,0,split=1,path=0)
        for name in a:
            np.testing.assert_array_equal(a[name],again[name])
            self.assertFalse(np.array_equal(a[name][:96],evaluation[name]))
            self.assertFalse(np.array_equal(a[name],data.split_noise(3,18,1,split=0)[name]))
            self.assertFalse(np.array_equal(a[name],data.split_noise(3,19,0,split=0)[name]))
        self.assertFalse(np.array_equal(a['P'],a['Q']))
        self.assertFalse(np.array_equal(evaluation['P'],data.split_noise(3,18,0,split=1,path=1)['P']))
        expected=np.random.default_rng(np.random.SeedSequence([data.NOISE_SEED,2718,18,0,1,0,2])).normal(size=(96,3))
        np.testing.assert_array_equal(evaluation['P'],expected)

    def test_bad_inputs_do_not_create_pseudo_results(self):
        clean=clean_fixture();noise=standard()
        with self.assertRaises(ValueError):data.observed_splits(clean,noise,.003)
        bad=deepcopy(clean);bad['V'][0,0]=np.nan
        with self.assertRaises(ValueError):data.observed_splits(bad,noise,.005)
        bad=deepcopy(noise);bad['root']=bad['root'][:-1]
        with self.assertRaises(ValueError):data.observed_splits(clean,bad,.005)
        for kwargs in (dict(split=2),dict(split=0,path=1),dict(split=1,path=16)):
            with self.assertRaises(ValueError):data.split_noise(3,18,0,**kwargs)

    def test_fixed_source_selection_requires_same_duplicate_catalogue(self):
        records=[]
        for i,case in enumerate(data.CASES):
            records.append(dict(id=case+'__old',case=case,start_date='2024-01-04',index=i,base_kw=200.,
                physical_cache={'original':{'path':'physical','sha256':'p'}},resource_context={'path':'context','sha256':'c'},
                forecast_ref={'path':'f','sha256':'f'},equipment_spec={'path':'s','sha256':'s'}))
        records.append(deepcopy(records[0]));records[-1]['id']='zcopy';records[-1]['equipment_spec']['path']='copy'
        chosen=data.source_entries(records)
        self.assertEqual(len(chosen),2)
        self.assertFalse(any('q_resources' in item for item in chosen))
        records[-1]['equipment_spec']['sha256']='changed'
        with self.assertRaises(ValueError):data.source_entries(records)

    def test_fitter_receives_192_96_only_and_failure_does_not_reset_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);clean=clean_fixture();noise=standard()
            np.savez_compressed(out/'physical.npz',**clean);np.savez_compressed(out/'noise.npz',**noise)
            ref=data.io.ref(out/'noise.npz')
            task=dict(id='test',case='pengwah18',history_repeat=0,pq_relative_sd=.005,
                physical_cache={'original':data.io.ref(out/'physical.npz')},standard_noise=ref,
                history_noise=ref,evaluation_noise=ref)
            data.io.write(out/'protocol.json',dict(method_config=data.asdict(data.validate._config(120.)),additional_E_pool_search_seconds=60.))
            def stop(train,val,**kwargs):
                self.assertEqual((train.rows,val.rows),(192,96))
                self.assertEqual(kwargs['method'],'101')
                self.assertEqual(kwargs['config'].total_seconds,120.)
                self.assertEqual(kwargs['config'].qp_seconds,120.)
                raise RuntimeError('Deliberate fit failure, no real fitting')
            with patch.object(data,'fit_observed',side_effect=stop) as fit, patch.object(data.physical,'offline_truth') as truth:
                first=data.fit_cell(task,out);second=data.fit_cell(task,out)
                self.assertEqual(first,second);self.assertEqual(first['status'],'failed')
                self.assertEqual(first['stage'],'fit101');self.assertEqual(fit.call_count,1)
                truth.assert_not_called()
            self.assertTrue((out/'histories/test/new_day.npz').exists())

    def test_partial_and_changed_completed_files_are_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'cell';folder.mkdir()
            with self.assertRaisesRegex(RuntimeError,'no automatic budget reset'):
                data.cached_completion(folder,{'id':'x'})
            result=folder/'result.json';data.io.write(result,{'status':'failed'})
            data.io.write(folder/'complete.json',dict(task={'id':'x'},status='failed',result=data.io.ref(result)))
            self.assertEqual(data.cached_completion(folder,{'id':'x'})['status'],'failed')
            with self.assertRaises(ValueError):data.cached_completion(folder,{'id':'y'})
            data.io.write(result,{'status':'changed'})
            with self.assertRaises(RuntimeError):data.cached_completion(folder,{'id':'x'})


if __name__=='__main__':unittest.main()
