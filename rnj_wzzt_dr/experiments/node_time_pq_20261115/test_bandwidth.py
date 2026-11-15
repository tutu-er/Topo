"""Independent hand-calculated fit, clock and input-boundary checks."""
import copy
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from .bandwidth import build_bandwidth


def training_fixture(start=datetime(2024, 1, 4, 7, tzinfo=timezone.utc)):
    stamps = [(start+timedelta(hours=i)).isoformat().replace('+00:00', 'Z') for i in range(48)]
    # Equal P profiles on both days; opposing residuals cancel in centered OLS.
    phase = np.tile(np.arange(24, dtype=float), 2)
    p = np.column_stack((phase+1., 2.*phase+3.))
    half_range = np.column_stack((np.arange(24) % 3, (np.arange(24)+1) % 4)).astype(float)
    signed = np.concatenate((half_range, -half_range))
    kappa, intercept = np.array([.5, -.25]), np.array([-3., 2.])
    q = intercept+p*kappa+signed
    return p, q, [101, 102], stamps, kappa, intercept, 2.*half_range


class BandwidthTests(unittest.TestCase):
    def test_exact_hand_fit_hour_order_and_zero_width(self):
        p, q, labels, stamps, kappa, intercept, expected = training_fixture()
        p0, q0 = p.copy(), q.copy()
        arrays, receipt = build_bandwidth(p/200., q/200., labels, stamps, 200., source={'train': 'test-only'})
        np.testing.assert_allclose(arrays['kappa'], kappa, atol=1e-14, rtol=0)
        np.testing.assert_allclose(arrays['intercept_kvar'], intercept, atol=1e-13, rtol=0)
        np.testing.assert_allclose(arrays['band_kvar'], expected, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(arrays['decision_hours'], list(range(8, 24))+list(range(8)))
        np.testing.assert_array_equal(arrays['counts'], np.full((24, 2), 2))
        self.assertEqual(receipt['source'], {'train': 'test-only'})
        self.assertIsNone(receipt['bandwidth_floor'])
        self.assertTrue(np.any(arrays['band_kvar'] == 0.))
        self.assertFalse(arrays['band_kvar'].flags.writeable)
        np.testing.assert_array_equal(p, p0); np.testing.assert_array_equal(q, q0)

    def test_common_unit_conversion_and_deterministic_content(self):
        p, q, labels, stamps, *_ = training_fixture()
        first, r1 = build_bandwidth(p, q, labels, stamps, 1.)
        second, r2 = build_bandwidth(p/8., q/8., labels, stamps, 8.)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        # Unit-source hashes differ, while the resulting quantities agree.
        self.assertNotEqual(r1['input_content_sha256']['P_pu'], r2['input_content_sha256']['P_pu'])
        _, r3 = build_bandwidth(p.copy(), q.copy(), list(labels), list(stamps), 1.)
        self.assertEqual(r1['fit_receipt_sha256'], r3['fit_receipt_sha256'])
        self.assertLess(np.min(q), 0.)  # Negative net Q is retained, not rectified.

    def test_clock_uses_local_hour_in_summer(self):
        p, q, labels, stamps, *_ = training_fixture(datetime(2024, 7, 4, 6, tzinfo=timezone.utc))
        arrays, receipt = build_bandwidth(p, q, labels, stamps, 1.)
        self.assertEqual(receipt['training_local_hour'][0], 8)
        self.assertEqual(arrays['decision_hours'][0], 8)
        self.assertTrue(receipt['training_start_local'].endswith('+02:00'))

    def test_fail_closed_inputs_and_time(self):
        p, q, labels, stamps, *_ = training_fixture()
        changes = []
        const = p.copy(); const[:, 0] = 1.
        nan = q.copy(); nan[0, 0] = np.nan
        duplicate = stamps.copy(); duplicate[1] = duplicate[0]
        naive = stamps.copy(); naive[0] = naive[0].removesuffix('Z')
        changes.extend([(const, q, labels, stamps), (p, nan, labels, stamps),
            (p, q, [101, 101], stamps), (p, q, labels, duplicate),
            (p, q, labels, naive), (p[:-1], q[:-1], labels, stamps[:-1]),
            (p, q, labels, stamps[:-1])])
        for arguments in changes:
            with self.subTest(arguments=str(arguments[2:])[:70]), self.assertRaises(ValueError):
                build_bandwidth(*arguments, 1.)
        with self.assertRaisesRegex(ValueError, 'DST'):
            p2, q2, lab2, t2, *_ = training_fixture(datetime(2024, 3, 30, 7, tzinfo=timezone.utc))
            build_bandwidth(p2, q2, lab2, t2, 1.)

    def test_source_metadata_is_copied(self):
        p, q, labels, stamps, *_ = training_fixture()
        source = {'train': {'sha256': 'abc'}}
        _, receipt = build_bandwidth(p, q, labels, stamps, 1., source=source)
        saved = copy.deepcopy(receipt)
        source['train']['sha256'] = 'changed'
        self.assertEqual(receipt, saved)


if __name__ == '__main__':
    unittest.main()
