import unittest
import numpy as np
from .prices import economic_dispatch,PRICE


class Tests(unittest.TestCase):
    def test_energy_generation_and_time_units_match_hand_calculation(self):
        F=np.c_[np.full((96,1),.5),np.full((96,1),.175)]
        a=economic_dispatch(F,[1.],1.,.3,fixed=np.zeros((24,1)))
        b=economic_dispatch(F,[1.],1.,.3,fixed=np.full((24,1),.1))
        self.assertTrue(a['global_optimality_verified']);self.assertTrue(b['global_optimality_verified'])
        expected=.25*PRICE.sum()*.1-24*350*.1+24*.5*(.4*500/.6)*.1**2
        self.assertAlmostEqual(b['cost_yuan']-a['cost_yuan'],expected,places=5)
        self.assertAlmostEqual(b['components']['bought_MWh']-a['components']['bought_MWh'],2.4,places=7)

    def test_fixed_net_plan_can_be_proved_unimplementable(self):
        F=np.c_[np.full((96,1),.5),np.full((96,1),.175)]
        r=economic_dispatch(F,[1.],1.,.5,fixed=np.full((24,1),.5))
        self.assertFalse(r['feasible']);self.assertTrue(r['infeasibility_proved']);self.assertIsNone(r['cost_yuan'])

    def test_cost_offset_preserves_currency_and_implementation(self):
        F=np.c_[np.full((96,1),.5),np.full((96,1),.175)];point=np.full((24,1),.1)
        a=economic_dispatch(F,[1.],1.,.3,fixed=point)
        b=economic_dispatch(F,[1.],1.,.3,fixed=point,objective_offset=a['cost_yuan'])
        self.assertAlmostEqual(a['cost_yuan'],b['cost_yuan'],places=5)
        self.assertAlmostEqual(b['lower_bound_yuan'],b['cost_yuan'],places=5)


if __name__=='__main__':unittest.main()
