"""Small array checks; these tests never generate AC, identify or optimize."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from . import data


def clean_fixture(n=3):
    t = np.arange(data.POINTS)[:, None]
    p = .02+.00003*t+np.arange(n)[None]*.002
    q = -.003+.000005*t+np.arange(n)[None]*.001
    root = 1.02+.001*np.sin(np.arange(data.POINTS)/11.)
    v = root[:, None]-.001-np.arange(n)[None]*.0003
    return dict(P=p, Q=q, V=v, root=root, labels=np.arange(301,301+n))


class DataTests(unittest.TestCase):
    def test_nested_history_and_shared_later_windows(self):
        ix = data.indices(); clean = clean_fixture(); noise = data.standard_noise(3, 'pengwah18', 0)
        obs = data.observations(clean, noise)
        two, four, eight = [data.slice_observations(obs, ix, d) for d in (2, 4, 8)]
        self.assertEqual([two['train'].points, four['train'].points, eight['train'].points], [192,384,768])
        for part in (two, four, eight):
            self.assertEqual((part['validation'].points, part['new_day'].points), (192,96))
            np.testing.assert_array_equal(part['validation'].y, obs.y[768:960])
            np.testing.assert_array_equal(part['new_day'].y, obs.y[960:])
        np.testing.assert_array_equal(two['train'].y, eight['train'].y[-192:])
        np.testing.assert_array_equal(four['train'].q, eight['train'].q[-384:])
        self.assertFalse(set(ix['train_indices']['8']) & set(ix['validation_indices']))
        self.assertFalse(set(ix['validation_indices']) & set(ix['new_day_indices']))

    def test_noise_coordinates_and_pairing(self):
        a = data.standard_noise(3, 'soumalas11', 0)
        b = data.standard_noise(3, 'soumalas11', 0)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            self.assertFalse(np.array_equal(a[key], data.standard_noise(3, 'soumalas11', 1)[key]))
            self.assertFalse(np.array_equal(a[key], data.standard_noise(3, 'pengwah18', 0)[key]))
        expected = np.random.default_rng(np.random.SeedSequence([2026111601,1610,0,0,2])).normal(size=(1056,3))
        np.testing.assert_array_equal(a['P'], expected)
        self.assertFalse(np.array_equal(a['P'], a['Q']))

    def test_laws_keep_clean_and_input_noise_separate(self):
        clean = clean_fixture(); previous = deepcopy(clean)
        noise = data.standard_noise(3, 'pengwah18', 0)
        H = np.c_[np.eye(3)*.1, np.eye(3)*.05]
        linear = data.observations(clean, noise, 'cleanlinear', H)
        np.testing.assert_array_equal(linear.y, np.c_[clean['P'],clean['Q']]@H.T)
        ac = data.observations(clean, noise, 'cleanAC')
        vonly = data.observations(clean, noise, 'Vonly')
        all_noise = data.observations(clean, noise, 'all')
        np.testing.assert_array_equal(ac.y, clean['root'][:,None]**2-clean['V']**2)
        np.testing.assert_array_equal(vonly.p, clean['P'])
        np.testing.assert_array_equal(vonly.q, clean['Q'])
        np.testing.assert_array_equal(all_noise.y, vonly.y)
        np.testing.assert_array_equal(all_noise.p, clean['P']*(1+.005*noise['P']))
        np.testing.assert_array_equal(all_noise.root, clean['root']*(1+.0005*noise['root']))
        for key in clean:
            np.testing.assert_array_equal(clean[key], previous[key])

    def test_quality_uses_finiteness_count_and_keeps_all31(self):
        times = pd.date_range('2024-01-04T07:00:00Z', periods=41*96, freq='15min')
        flags = np.ones((len(times),18), bool); counts = np.full(flags.shape,15)
        # A missing ninth-day bin invalidates Jan04 without reading magnitudes.
        flags[8*96,8] = False
        rows = data.quality_rows(times, flags, counts, np.arange(1,19))
        self.assertEqual(len(rows),31)
        self.assertEqual(rows[0]['cases']['soumalas11']['incomplete_node_time_bins'],1)
        self.assertEqual(rows[0]['cases']['soumalas11']['by_day'][8],1)
        self.assertFalse(rows[0]['both_complete']); self.assertTrue(rows[9]['both_complete'])
        counts[10*96,0] = 11
        self.assertEqual(data.quality_rows(times, flags, counts, np.arange(1,19))[0]['cases']['pengwah18']['incomplete_node_time_bins'],2)
        with self.assertRaises(ValueError):
            data.quality_rows(times, flags.astype(float), counts, np.arange(1,19))

    def test_scales_are_frozen_and_signed_PQ_preserved(self):
        n = 11; shape=(data.POINTS,18); times=data.wanted_times().strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy()
        archive=dict(timestamps_utc=times, feeder_ids=np.arange(1,19), p_kw=np.full(shape,-7.),
                     q_kvar=np.full(shape,3.), valid_minutes=np.full(shape,15))
        context=dict(node_mapping=[dict(label=101+j,feeder_id=j+1,scale=.1*(j+1)) for j in range(n)])
        p=data.extract_profile('soumalas11', archive, context)
        np.testing.assert_allclose(p['p_kw'][0], -7*np.arange(1,12)*.1)
        np.testing.assert_allclose(p['q_kvar'][0], 3*np.arange(1,12)*.1)
        self.assertFalse(p['metadata']['scales_refitted'])
        archive['p_kw'][-96:]*=20
        changed=data.extract_profile('soumalas11', archive, context)
        np.testing.assert_array_equal(changed['p_kw'][:-96],p['p_kw'][:-96])
        self.assertEqual(changed['metadata']['scales'],p['metadata']['scales'])
        archive['valid_minutes'][0,0]=11
        with self.assertRaises(ValueError): data.extract_profile('soumalas11',archive,context)

    def test_clock_and_dst(self):
        time=data.wanted_times(); self.assertEqual(len(time),1056)
        self.assertEqual(str(time[0]),'2024-01-21 07:00:00+00:00')
        self.assertEqual(str(time[-1]),'2024-02-01 06:45:00+00:00')
        with self.assertRaises(ValueError): data.wanted_times('2024-03-25')

    def test_bad_arrays_and_laws_rejected(self):
        clean=clean_fixture(); noise=data.standard_noise(3,'pengwah18',0)
        with self.assertRaises(ValueError): data.observations(clean,noise,'unknown')
        with self.assertRaises(ValueError): data.observations(clean,noise,'cleanlinear')
        bad=deepcopy(clean); bad['P'][0,0]=np.nan
        with self.assertRaises(ValueError): data.observations(bad,noise)
        with self.assertRaises(ValueError): data.standard_noise(3,'pengwah18',4)

    def test_existing_directory_never_restarts(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp); protocol=path/'PROTOCOL.md';protocol.write_text('locked')
            with self.assertRaises(FileExistsError):
                data.prepare_worlds(path,path/'unused.json',path/'unused.npz',data.io.ref(protocol))


if __name__ == '__main__':
    unittest.main()
