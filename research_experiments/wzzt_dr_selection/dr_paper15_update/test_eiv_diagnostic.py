import unittest
import numpy as np
import pandas as pd
from eiv_diagnostic import core_observed,fixed_sigmas,point_from_net
from research_experiments.wzzt_dr_selection.dr_paper15_update.ac_family_refit import family_network

class TestEIVAdapter(unittest.TestCase):
    def obs(self):
        idx=pd.date_range('2010-01-01',periods=3,freq='h')
        p=pd.DataFrame([[.01,.02],[.02,.03],[.03,.01]],index=idx,columns=[101,102])
        return p,.35*p,1-p*.01,pd.Series([1.,1.001,1.002],index=idx)
    def test_raw_channels_and_root_alignment(self):
        obs=self.obs();out=core_observed(obs)
        self.assertEqual(set(out),{'P_terminal','Q_terminal','V_terminal','root_voltage','root_observation'})
        self.assertIs(out['root_voltage'],obs[3]);self.assertNotIn('drop_target',out)
    def test_absolute_sigmas_preserve_labels(self):
        obs=self.obs();s=fixed_sigmas(obs,.002)
        pd.testing.assert_frame_equal(s.p,obs[0].abs()*.005)
        pd.testing.assert_frame_equal(s.v,obs[2].abs()*.002)
        pd.testing.assert_series_equal(s.root_v,obs[3].abs()*.0002)
    def test_zero_sigma_is_not_silently_floored(self):
        obs=list(self.obs());obs[1]=obs[1].copy();obs[1].iloc[0,0]=0.
        with self.assertRaises(ValueError):fixed_sigmas(obs,.002)
    def test_physical_roundtrip_preserves_zero_and_supports(self):
        point=dict(supports=[[0],[1],[0,1]],r=[0.,.01,.02],x=[.04,0.,.03])
        net=family_network(point,[101,102],base_kv=.4,base_mva=.2)
        restored=point_from_net(net,point)
        self.assertEqual(restored['supports'],point['supports'])
        np.testing.assert_allclose(restored['r'],point['r'],rtol=1e-14,atol=0.)
        np.testing.assert_allclose(restored['x'],point['x'],rtol=1e-14,atol=0.)

if __name__=='__main__':unittest.main()
