"""Small algebra/selection checks; no optimizer, fitting, or AC is executed."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import data


def clean_fixture(n=3):
    t=np.arange(384, dtype=float)[:,None]
    p=.03+.0001*t+np.arange(n)[None]*.01
    q=-.004+.00002*t+np.arange(n)[None]*.003
    root=1.02+.001*np.sin(np.arange(384)/7.)
    v=root[:,None]-.002-np.arange(n)[None]*.001
    return dict(P=p,Q=q,V=v,root=root,labels=np.arange(301,301+n),
                timestamps=np.asarray([str(i) for i in range(384)]))


class DataChecks(unittest.TestCase):
    def test_locked_sampling_keeps_disjoint_windows(self):
        for stride, nt, nv in ((1,192,96),(2,96,48),(4,48,24)):
            idx=data.sample_indices(stride)
            self.assertEqual((len(idx['train']),len(idx['validation']),len(idx['new_day'])),(nt,nv,96))
            self.assertTrue(all(i<192 for i in idx['train']))
            self.assertTrue(all(192<=i<288 for i in idx['validation']))
            self.assertEqual(idx['new_day'].tolist(),list(range(288,384)))
            self.assertFalse(set(idx['train'])&set(idx['validation']))
        with self.assertRaises(ValueError):data.sample_indices(3)

    def test_meter_formula_and_no_physical_mutation(self):
        clean=clean_fixture();before=deepcopy(clean)
        noise={k:np.full_like(clean[k],.7) for k in ('P','Q','V','root')}
        obs=data.observed_splits(clean,noise,2,2.)['validation'];i=np.arange(192,288,2)
        np.testing.assert_array_equal(obs.p,clean['P'][i]*(1+.01*.7))
        np.testing.assert_array_equal(obs.q,clean['Q'][i]*(1+.01*.7))
        np.testing.assert_array_equal(obs.voltage,clean['V'][i]*(1+.001*.7))
        np.testing.assert_array_equal(obs.root,clean['root'][i]*(1+.001*.7))
        np.testing.assert_array_equal(obs.y,obs.root[:,None]**2-obs.voltage**2)
        for k in before:np.testing.assert_array_equal(clean[k],before[k])

    def test_noise_nesting_and_multiplier_identity(self):
        clean=clean_fixture();noise=data.standard_noise(3,18)
        for multiplier in data.METER_MULTIPLIERS:
            full=data.observed_splits(clean,noise,1,multiplier)
            thin=data.observed_splits(clean,noise,4,multiplier)
            for name in ('train','validation'):
                for field in ('p','q','y','voltage','root'):
                    np.testing.assert_array_equal(getattr(thin[name],field),getattr(full[name],field)[::4])
            np.testing.assert_array_equal(thin['new_day'].y,full['new_day'].y)
        half=data.observed_splits(clean,noise,1,.5)['train']
        double=data.observed_splits(clean,noise,1,2.)['train']
        np.testing.assert_allclose(double.p-clean['P'][:192],4*(half.p-clean['P'][:192]),atol=6e-17,rtol=1e-12)

    def test_independent_components_and_original_index(self):
        a=data.standard_noise(3,18);b=data.standard_noise(3,18);c=data.standard_noise(3,19)
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])
            self.assertFalse(np.array_equal(a[name],c[name]))
        self.assertFalse(np.array_equal(a['P'],a['Q']))
        self.assertFalse(np.array_equal(a['P'],a['V']))

    def test_input_selection_uses_availability_only(self):
        entries=[dict(id=c+'__'+d,case=c,start_date=d,irrelevant_quality='unused')
                 for c,d in data.REQUESTED.items()]
        entries += [dict(id='paper15__2024-02-22',case='paper15',start_date='2024-02-22'),
                    dict(id='paper15__2024-02-12',case='paper15',start_date='2024-02-12')]
        chosen,ledger=data.choose_inputs(entries,lambda e:e['id']!='paper15__2024-01-04')
        self.assertEqual(next(x for x in chosen if x['case']=='paper15')['id'],'paper15__2024-02-12')
        self.assertEqual(len(chosen),4)
        self.assertEqual(sum(x.get('fallback_used',False) for x in ledger),1)
        with self.assertRaises(ValueError):
            data.choose_inputs(entries,lambda e:e['case']!='pengwah18')

    def test_hash_change_is_not_an_availability_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.json';path.write_text('original')
            reference=data.io.ref(path);data.checked(reference)
            path.write_text('changed')
            with self.assertRaises(RuntimeError):data.checked(reference)
        def corrupt(_):raise RuntimeError('Changed sealed input')
        entries=[dict(id=c+'__'+d,case=c,start_date=d) for c,d in data.REQUESTED.items()]
        with self.assertRaises(RuntimeError):data.choose_inputs(entries,corrupt)

    def test_bad_arrays_are_rejected(self):
        clean=clean_fixture();noise=data.standard_noise(3,18)
        bad=deepcopy(clean);bad['V'][0,0]=np.nan
        with self.assertRaises(ValueError):data.observed_splits(bad,noise,1,1.)
        bad=deepcopy(noise);bad['root']=bad['root'][:-1]
        with self.assertRaises(ValueError):data.observed_splits(clean,bad,1,1.)
        with self.assertRaises(ValueError):data.observed_splits(clean,noise,1,3.)


if __name__=='__main__':unittest.main()
