"""Independent provenance/scale tests for the synthetic equipment catalogue."""
from copy import deepcopy
import unittest

import numpy as np

from .resources import build_spec


def context():
    return dict(node_mapping=[dict(label=101+i, feeder_id=i+1, scale=.2) for i in range(4)],
        resource_metadata_at_history_start=[dict(feeder=i+1, date='2023-01-01',
            PV_systems_kW=10.*(i+1), EV_chargers_kW=5.*(i+1),
            hot_water_tanks_kW=2.*(i+1), batteries_kW=8.*(i+1)) for i in range(4)])


class ResourcesTest(unittest.TestCase):
    def test_selection_and_scaling_once(self):
        spec = build_spec(context(), [101, 102, 103, 104], '2024-01-04')
        self.assertEqual(len(spec['devices']), 9)
        self.assertEqual([d['node_label'] for d in spec['devices'] if d['kind']=='pv'], [104, 103, 102])
        for device in spec['devices']:
            self.assertAlmostEqual(device['power_kw'], .1*.2*device['registered_kw'])
            if device['kind']=='battery':
                self.assertAlmostEqual(device['energy_kwh'], 2.*device['power_kw'])
                self.assertAlmostEqual(device['reactive_cap_kvar'], .4*device['inverter_kva'])

    def test_no_prediction_or_candidate_input_and_determinism(self):
        first = build_spec(context(), [101, 102, 103, 104], '2024-01-04')
        self.assertEqual(first, build_spec(context(), [101, 102, 103, 104], '2024-01-04'))
        pv = next(d for d in first['devices'] if d['kind']=='pv')
        gross = np.asarray(pv['available_generation_kw'])
        self.assertTrue(np.all(gross >= 0))
        np.testing.assert_array_equal(np.asarray(pv['baseline_p_kw']), -gross)
        self.assertGreater(np.count_nonzero(gross), 0)
        self.assertLess(np.count_nonzero(gross), 96)
        # Seasonal rule is predeclared; no opportunistic net-export cap exists.
        summer = build_spec(context(), [101, 102, 103, 104], '2024-07-04')
        self.assertGreater(summer['solar']['daylight_hours'], first['solar']['daylight_hours'])

    def test_registration_and_mapping_fail_closed(self):
        data = context(); data['resource_metadata_at_history_start'][0]['date'] = '2025-01-01'
        with self.assertRaises(ValueError): build_spec(data, [101, 102, 103, 104], '2024-01-04')
        data = context(); data['node_mapping'][0]['scale'] = -1
        with self.assertRaises(ValueError): build_spec(data, [101, 102, 103, 104], '2024-01-04')
        with self.assertRaises(ValueError): build_spec(context(), [104, 103, 102, 101], '2024-01-04')


if __name__ == '__main__': unittest.main()
