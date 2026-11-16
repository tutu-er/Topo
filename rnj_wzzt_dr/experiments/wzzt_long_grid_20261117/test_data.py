"""Independent small data checks: no network, AC or optimization invocation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rnj_wzzt_dr.experiments.wzzt_long_grid_20261117 import data


def fixture(n=3):
    t=np.arange(1056)[:,None]
    p=.03+.00001*t+np.arange(n)[None]*.002
    q=-.006+.000002*t-np.arange(n)[None]*.001
    root=1.02+.0001*np.sin(np.arange(1056))
    return dict(P=p,Q=q,V=root[:,None]-.002-np.arange(n)[None]*.0002,
                root=root,labels=np.arange(301,301+n))


class DataTests(unittest.TestCase):
    def test_complete_fixed_grid_and_long_subset(self):
        rows=data.coordinates()
        self.assertEqual(len(rows),160);self.assertEqual(len({r['id'] for r in rows}),160)
        self.assertEqual(sum(r['long_selected'] for r in rows),16)
        for case in data.CASES:
            self.assertEqual(sum(r['case']==case for r in rows),40)
        for r in rows:
            if r['long_selected']:
                self.assertIn(r['training_rows'],(96,768));self.assertIn(r['pq_relative_sd'],(.005,.02))
                self.assertEqual(r['history_repeat'],0)
        extra=data.root_coordinates()
        self.assertEqual(len(extra),16)
        self.assertTrue(set(r['id'] for r in rows).isdisjoint(r['id'] for r in extra))
        self.assertEqual({r['root_relative_sd'] for r in extra},{0.,.001})
        self.assertTrue(all(r['paired_primary_id'] in {p['id'] for p in rows} for r in extra))
        uniform=data.sampling_coordinates()
        self.assertEqual(len(uniform),16)
        self.assertEqual(len({r['id'] for r in rows+extra+uniform}),192)
        self.assertTrue(all(r['sample_mode']=='uniform8d' and r['pq_sd']==.005
                            and r['repeat']==0 and r['N']!=768 for r in uniform))

    def test_uniform_sampling_fixes_span_without_duplicate_or_future_rows(self):
        for n in (48,96,192,384,768):
            ix=data.split_indices(n,'uniform8d')
            expected=[int(round(767*k/(n-1))) for k in range(n)]
            self.assertEqual(ix['train'],expected)
            self.assertEqual(len(set(ix['train'])),n)
            self.assertEqual((ix['train'][0],ix['train'][-1]),(0,767))
            self.assertEqual(ix['validation'],list(range(768,960)))
            self.assertEqual(ix['new_day'],list(range(960,1056)))
        self.assertEqual(data.split_indices(768,'uniform8d'),data.split_indices(768))
        with self.assertRaises(ValueError):data.split_indices(96,'unknown')

    def test_common_cutoff_disjoint_later_data_and_nested_noise(self):
        clean=fixture();z=data.standard_noise(3,'pengwah18',0);obs=data.observe(clean,z,.005)
        for n in (48,96,192,384,768):
            ix=data.split_indices(n)
            self.assertEqual(ix['train'],list(range(768-n,768)))
            self.assertEqual(ix['validation'],list(range(768,960)))
            self.assertEqual(ix['new_day'],list(range(960,1056)))
            self.assertFalse(set(ix['train']) & set(ix['validation']))
            np.testing.assert_array_equal(obs.p[ix['train']],obs.p[:768][-n:])
        with self.assertRaises(ValueError):data.split_indices(64)

    def test_independent_future_stream_exact_entropy(self):
        z=data.standard_noise(3,'soumalas11',1)
        expected=np.random.default_rng(np.random.SeedSequence([2026111701,1717,2,1,2,3])).normal(size=(96,3))
        np.testing.assert_array_equal(z['Q'][960:],expected)
        other=data.standard_noise(3,'soumalas11',0)
        self.assertFalse(np.array_equal(z['Q'],other['Q']))
        self.assertFalse(np.array_equal(z['Q'][960:],z['Q'][768:864]))
        self.assertFalse(np.array_equal(z['Q'],z['P']))

    def test_noise_amplitude_pairing_and_squared_root_common_error(self):
        clean=fixture();before=deepcopy(clean);z=data.standard_noise(3,'pengwah18',0)
        zero=data.observe(clean,z,0.);one=data.observe(clean,z,.005);two=data.observe(clean,z,.01)
        np.testing.assert_array_equal(zero.p,clean['P']);np.testing.assert_array_equal(zero.q,clean['Q'])
        np.testing.assert_allclose(two.p-clean['P'],2*(one.p-clean['P']),atol=1e-17)
        np.testing.assert_array_equal(zero.y,one.y);np.testing.assert_array_equal(one.y,two.y)
        v=clean['V']*(1+.0005*z['V']);r=clean['root']*(1+.0005*z['root'])
        np.testing.assert_array_equal(one.y,r[:,None]**2-v**2)
        for key in clean:np.testing.assert_array_equal(clean[key],before[key])

    def test_root_only_panel_changes_one_shared_channel(self):
        clean=fixture();z=data.standard_noise(3,'pengwah18',0)
        a=data.observe(clean,z,.005,0.);b=data.observe(clean,z,.005,.001)
        np.testing.assert_array_equal(a.p,b.p);np.testing.assert_array_equal(a.q,b.q)
        np.testing.assert_array_equal(a.voltage,b.voltage)
        change=b.root**2-a.root**2
        np.testing.assert_allclose(b.y-a.y,np.repeat(change[:,None],3,axis=1),rtol=0,atol=3e-16)
        np.testing.assert_array_equal(a.root,clean['root'])

    def test_frozen_scaling_signs_and_missing_rejection(self):
        n=11;times=data.previous.wanted_times().strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy()
        source=dict(timestamps_utc=times,feeder_ids=np.arange(1,19),p_kw=np.full((1056,18),-7.),
                    q_kvar=np.full((1056,18),3.),valid_minutes=np.full((1056,18),15))
        context=dict(case='soumalas11',start_date='2024-01-04',source_archive_sha256=data.ARCHIVE_SHA,
            node_mapping=[dict(label=101+i,feeder_id=i+1,scale=.1*(i+1)) for i in range(n)])
        profile=data.extract_profile('soumalas11',source,context)
        np.testing.assert_allclose(profile['p_kw'][0],-7*np.arange(1,12)*.1)
        np.testing.assert_allclose(profile['q_kvar'][0],3*np.arange(1,12)*.1)
        source['p_kw'][-96:]*=100
        changed=data.extract_profile('soumalas11',source,context)
        np.testing.assert_array_equal(changed['p_kw'][:-96],profile['p_kw'][:-96])
        self.assertEqual(changed['metadata']['scales'],profile['metadata']['scales'])
        source['valid_minutes'][0,0]=11
        with self.assertRaises(ValueError):data.extract_profile('soumalas11',source,context)

    def test_only_observations_saved_to_fit_npz(self):
        clean=fixture();obs=data.observe(clean,data.standard_noise(3,'pengwah18',0),.005)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'train.npz';detail=data._save_observation(path,obs,range(576,768))
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'P','Q','Y','labels'})
                np.testing.assert_array_equal(z['P'],obs.p[576:768])
            self.assertEqual(detail['rows'],192)

    def test_source_hash_and_prior_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);file=path/'source';file.write_text('a');ref=data.io.ref(file);file.write_text('b')
            with self.assertRaises(ValueError):data.checked(ref)
            with self.assertRaises(FileExistsError):data.prepare(path)

    def test_validation_input_errors_are_explicit(self):
        clean=fixture();z=data.standard_noise(3,'pengwah18',0)
        with self.assertRaises(ValueError):data.observe(clean,z,.1)
        clean['P'][0,0]=np.nan
        with self.assertRaises(ValueError):data.observe(clean,z,.005)
        with self.assertRaises(ValueError):data.standard_noise(3,'pengwah18',2)


if __name__=='__main__':unittest.main()
