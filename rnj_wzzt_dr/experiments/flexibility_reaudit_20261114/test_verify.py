"""Cross-placement identity checks that do not require an experiment run."""
import unittest

from .verify import Verify


def device(identifier,node):
    return dict(id=identifier,node_label=node)


class DevicePairingTests(unittest.TestCase):
    def test_different_lengths_match_shared_identities(self):
        v=Verify();key=(('case',0),0)
        v.set_Q_identities([device('svg_303',303),device('svg_304',304)])
        v.paired(key,'eta_Q',[.91,.97])
        # Reordering also cannot silently change which physical device pairs.
        v.set_Q_identities([device('svg_305',305),device('svg_304',304),device('svg_303',303)])
        v.paired(key,'eta_Q',[.99,.97,.91])
        self.assertEqual(v.counts['cross_placement_Q_device_multipliers_verified'],2)
        self.assertEqual(v.counts['distinct_Q_device_path_multipliers'],3)

    def test_shared_identity_different_multiplier_fails(self):
        v=Verify();v.set_Q_identities([device('svg_303',303)])
        v.paired('path','eta_Q',[.91])
        with self.assertRaises(AssertionError):
            v.paired('path','eta_Q',[.92])

    def test_unrelated_ID_or_node_not_paired(self):
        v=Verify();v.set_Q_identities([device('contract_A',303)])
        v.paired('path','eta_Q',[.91])
        v.set_Q_identities([device('contract_B',303),device('contract_A',304)])
        v.paired('path','eta_Q',[.93,.95])
        self.assertEqual(v.counts['cross_placement_Q_device_multipliers_verified'],0)
        self.assertEqual(v.counts['distinct_Q_device_path_multipliers'],3)


if __name__=='__main__':
    unittest.main()
