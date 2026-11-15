"""Pairing, physical-ID seed mapping and pre-seal information boundary."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ... import io
from . import run


class TinyCase:
    H, P_R, Q_R, R = 2, 2, 2, 4
    shape = (2, 4)
    resources = [
        dict(id='pv_3', physical_device_id='pv_3', old_resource_id='pv_3', kind='pv', node_label=3),
        dict(id='battery_3', physical_device_id='battery_3', kind='battery', node_label=3),
        dict(id='pv_3_q', physical_device_id='pv_3', kind='reactive', node_label=3),
        dict(id='battery_3_q', physical_device_id='battery_3', kind='reactive', node_label=3),
    ]

    def audit(self, value):
        value = np.asarray(value)
        return dict(passed=bool(value.shape == self.shape and np.isfinite(value).all()
                                and np.max(abs(value)) <= 1.), maximum=float(np.max(abs(value))))


OLD = dict(resources=[dict(id='pv_3', node_label=3, kind='pv'),
                      dict(id='old_svg_3', node_label=3, kind='reactive')],
           action_units=['kW', 'kvar'])


class PairingTests(unittest.TestCase):
    def test_physical_PQ_share_eta_and_battery_does_not_borrow_svg(self):
        eta, receipt = run.response_multipliers(TinyCase(), OLD, [.91, .999],
                                              seed=14, original_index=5, repeat=3)
        self.assertEqual(eta[0], .91)
        self.assertEqual(eta[2], .91)
        self.assertEqual(eta[1], eta[3])
        self.assertGreaterEqual(eta[1], .9)
        self.assertLessEqual(eta[1], 1.)
        self.assertNotEqual(eta[1], .999)
        self.assertEqual(receipt[0]['source'], 'reused_old_physical_P_device')
        self.assertEqual(receipt[1]['source'], 'new_keyed_physical_device')
        eta2, _ = run.response_multipliers(TinyCase(), OLD, [.91, .92],
                                         seed=14, original_index=5, repeat=3)
        np.testing.assert_array_equal(eta, eta2)
        # Old SVG realization has no influence on either new physical device.

    def test_new_stream_is_order_independent_and_repeat_specific(self):
        first, key = run.keyed_multiplier(17, 2, 4, 'battery_9')
        run.keyed_multiplier(17, 2, 4, 'other_device')
        again, same_key = run.keyed_multiplier(17, 2, 4, 'battery_9')
        other, _ = run.keyed_multiplier(17, 2, 5, 'battery_9')
        self.assertEqual(first, again)
        self.assertEqual(key, same_key)
        self.assertNotEqual(first, other)

    def test_mapped_seed_preserves_only_matching_P_and_records_rejection(self):
        original = {'a':np.array([[.5, 100.], [.2, -100.]]),
                    'bad':np.array([[2., 0.], [0., 0.]]),
                    'duplicate':np.array([[.5, 99.], [.2, 9.]])}
        points, receipt = run.mapped_seeds(TinyCase(), OLD, original)
        self.assertEqual(set(points), {'zero', 'mapped_a'})
        np.testing.assert_array_equal(points['mapped_a'][:, 0], [.5, .2])
        np.testing.assert_array_equal(points['mapped_a'][:, 1:], np.zeros((2,3)))
        rows = {r['name']:r for r in receipt['records']}
        self.assertFalse(rows['bad']['physical_and_band_audit']['passed'])
        self.assertEqual(rows['duplicate']['duplicate_of'], 'mapped_a')
        self.assertFalse(receipt['prior_final_policy_or_oracle_seeds'])

    def test_id_collision_and_battery_relabel_are_rejected(self):
        class WrongNode(TinyCase):
            resources = [dict(r) for r in TinyCase.resources]
        WrongNode.resources[0]['node_label'] = 4
        with self.assertRaisesRegex(ValueError, 'different physical'):
            run.old_p_mapping(WrongNode(), OLD)
        class BatteryRelabel(TinyCase):
            resources = [dict(r) for r in TinyCase.resources]
        BatteryRelabel.resources[1]['old_resource_id'] = 'old_svg_3'
        with self.assertRaisesRegex(ValueError, 'cannot inherit'):
            run.old_p_mapping(BatteryRelabel(), OLD)

    def test_no_future_context_before_readable_seal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.paired.old, 'context') as future:
            with self.assertRaises(FileNotFoundError):
                run.execute({}, {}, Path(tmp), {}, None, None, None, {}, {}, {})
            future.assert_not_called()

    def test_device_states_npz_is_numeric_and_json_keeps_labels(self):
        state = {'battery':dict(soc=[.5,.6,.5], note='declared state', p_kw=np.zeros(96))}
        with tempfile.TemporaryDirectory() as tmp:
            refs = run.save_states(Path(tmp), state, state)
            saved = io.read(refs['json']['path'])
            self.assertEqual(saved['requested']['battery']['note'], 'declared state')
            with np.load(refs['arrays']['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(z['delivered__battery__soc'], [.5,.6,.5])
                self.assertNotIn('requested__battery__note', z.files)


if __name__ == '__main__':
    unittest.main()
