"""Data amount, leakage and paired physical counterfactual checks."""
import unittest

import numpy as np
import pandas as pd

from research_experiments.wzzt_systematic.common import checked_point, indices, scenes_from_observed


class SamplingProtocolTests(unittest.TestCase):
    def test_interrupted_epigraph_incumbent_is_a_loss_upper_bound(self):
        # y=.002 and .1*.02=.002 give zero physical residual; an interrupted
        # incumbent can retain t=1e-6 while still satisfying t>=|residual|.
        p=pd.DataFrame({7:[.1]})
        q=pd.DataFrame({7:[0.]})
        v=pd.DataFrame({7:[np.sqrt(.998)]})
        obs=(p,q,v,pd.Series([1.]))
        candidate=checked_point([7],[(0,)],[.02],[0.],obs,1e-6)
        self.assertLess(candidate["search_objective"],1e-14)
        self.assertGreater(candidate["epigraph_slack"],.99e-6)
        with self.assertRaises(RuntimeError):
            checked_point([7],[(0,)],[.04],[0.],obs,0.)

    def test_nested_physical_sampling_preserves_daily_extent(self):
        # More samples must reveal the same physical days rather than change
        # the dynamics, meter RNG draw or train/test trajectory seed.
        for total in (24,48,96,288):
            selection=indices(total)
            self.assertEqual(3*len(selection),total)
            self.assertEqual(selection[0],0)
            self.assertGreaterEqual(selection[-1],84)
        for a,b in ((24,48),(48,96),(96,288)):
            self.assertTrue(set(indices(a))<set(indices(b)))

    def test_root_measurement_shared_and_day_boundaries_retained(self):
        p=pd.DataFrame(np.arange(24*2).reshape(24,2),columns=[7,9])
        q=p*.1
        v=pd.DataFrame(np.ones((24,2)),columns=p.columns)
        root=pd.Series(np.linspace(1.,1.04,24))
        scenes=scenes_from_observed((p,q,v,root))
        self.assertEqual([len(s["P_terminal"]) for s in scenes],[8,8,8])
        for scene in scenes:
            rows=scene["P_terminal"].index
            np.testing.assert_allclose(scene["drop_target"],np.repeat((root.loc[rows].to_numpy()**2-1)[:,None],2,axis=1))


if __name__=="__main__":
    unittest.main()
