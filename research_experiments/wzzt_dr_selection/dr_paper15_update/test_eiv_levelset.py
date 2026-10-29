from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd
from rnj_wzzt.models.network import TerminalizedNetwork
from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
from rnj_wzzt.estimation.ac_measurement_fit import ACMeasurementSigma

spec=importlib.util.spec_from_file_location('eiv_levelset_tested',Path(__file__).with_name('eiv_levelset.py'))
level=importlib.util.module_from_spec(spec);spec.loader.exec_module(level)


def fixture(power=.1,reactive=.04,r=.01,x=.005):
    buses=pd.DataFrame([dict(bus_id=1,bus_type='root',is_observed=True,has_load=False),
        dict(bus_id=101,bus_type='observed_terminal',is_observed=True,has_load=True)])
    branches=pd.DataFrame([dict(from_bus=1,to_bus=101,r_ohm=r,x_ohm=x,is_true_closed=True,branch_type='synthetic')])
    net=TerminalizedNetwork(buses,branches,1,.4,.2,{}, {},{})
    index=pd.date_range('2021-01-01',periods=3,freq='h')
    p=pd.DataFrame(power,index=index,columns=[101]);q=pd.DataFrame(reactive,index=index,columns=[101]);root=pd.Series(1.,index=index)
    ac=solve_ac_power_flow_timeseries(net,p.reindex(columns=[1,101],fill_value=0.),q.reindex(columns=[1,101],fill_value=0.),v_root=root)
    observed=dict(P_terminal=p,Q_terminal=q,V_terminal=ac['V_bus_mag'][[101]]+2e-4,root_voltage=root,root_observation='noisy')
    sigma=ACMeasurementSigma(p=.005*max(abs(power),1e-12),q=.005*max(abs(reactive),1e-12),v=2e-4,root_v=2e-4)
    return net,observed,sigma


class LevelTests(unittest.TestCase):
    def test_positive_load_data_bound_contains_incumbent_and_uses_full_L(self):
        net,obs,sigma=fixture();out=level.derive_eiv_levelset_bounds(net,obs,sigma)
        self.assertTrue(out['finite_certified']);self.assertTrue(out['api_usable'])
        self.assertEqual(out['bound_kind'],'triangle')
        self.assertAlmostEqual(out['incumbent_loss_L'],3.,places=8)
        self.assertAlmostEqual(out['standardized_residual_radius'],np.sqrt(3.),places=8)
        self.assertEqual(out['r_bounds_ohm'][0],0.)
        self.assertGreater(out['global_upper_ohm'],abs(.01+1j*.005))
        self.assertTrue(all(e['incumbent_inside_bound'] for e in out['edges']))

    def test_zero_impedance_is_allowed_without_positive_floor(self):
        net,obs,sigma=fixture(r=0.,x=0.);out=level.derive_eiv_levelset_bounds(net,obs,sigma)
        self.assertTrue(out['finite_certified']);self.assertEqual(out['r_bounds_ohm'][0],0.)
        self.assertEqual(out['edges'][0]['initial_impedance_magnitude_ohm'],0.)

    def test_zero_load_is_unresolved_not_silently_dropped(self):
        net,obs,sigma=fixture(power=0.,reactive=0.);out=level.derive_eiv_levelset_bounds(net,obs,sigma)
        self.assertFalse(out['finite_certified']);self.assertIsNone(out['global_upper_ohm'])
        self.assertEqual(len(out['edges']),1);self.assertIsNone(out['edges'][0]['upper_ohm'])

    def test_export_does_not_receive_a_positive_flow_assumption(self):
        net,obs,sigma=fixture(power=-.1,reactive=-.04);out=level.derive_eiv_levelset_bounds(net,obs,sigma)
        self.assertFalse(out['finite_certified']);self.assertEqual(out['usable_nonnegative_load_times'],0)


if __name__=='__main__':unittest.main(verbosity=2)
