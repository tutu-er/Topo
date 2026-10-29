"""Bounded interface checks on analytic and genuinely AC-generated observations."""
from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
from ac_family_refit import family_network,predict_ac,fit_family_ac


def observations(p,q,labels,root=1.):
    index=pd.date_range('2020-01-01',periods=len(p),freq='h')
    return (pd.DataFrame(p,index=index,columns=labels),pd.DataFrame(q,index=index,columns=labels),
            pd.DataFrame(np.ones(np.shape(p)),index=index,columns=labels),pd.Series(root,index=index))


class ACBridgeTests(unittest.TestCase):
    def test_two_bus_closed_form_and_conversion(self):
        p=np.array([[.03],[.08],[.1]]);q=np.array([[.01],[.03],[.04]])
        obs=observations(p,q,[101]);point=dict(supports=[[0]],r=[.2],x=[.1])
        net=family_network(point,[101],base_kv=.4,base_mva=.1)
        self.assertAlmostEqual(net.branches.iloc[0].r_ohm,.16)
        self.assertAlmostEqual(net.branches.iloc[0].x_ohm,.08)
        result=predict_ac(point,obs,base_kv=.4,base_mva=.1)
        aa=1-2*(.1*p+.05*q)
        exact_squared=(aa+np.sqrt(aa**2-4*(.1**2+.05**2)*(p*p+q*q)))/2
        np.testing.assert_allclose(result['voltage'].to_numpy()**2,exact_squared,atol=2e-12,rtol=0)

    def test_exact_zero_edges(self):
        point=dict(supports=[[0],[1],[0,1]],r=[0.,.1,0.],x=[0.,.04,0.])
        obs=observations(np.array([[.1,.1],[.05,.08]]),np.array([[.02,.04],[.02,.03]]),[101,102])
        result=predict_ac(point,obs,base_kv=.4,base_mva=.1)
        self.assertEqual(float(result['net'].branches.r_ohm.min()),0.)
        np.testing.assert_array_equal(result['voltage'][101].to_numpy(),np.ones(2))

    def test_rejects_label_and_nonlaminar_mismatch(self):
        point=dict(supports=[[0],[1]],r=[.1,.2],x=[.05,.1])
        obs=observations(np.ones((2,2))*.03,np.ones((2,2))*.01,[101,102])
        bad=(obs[0],obs[1].loc[:,[102,101]],obs[2],obs[3])
        with self.assertRaisesRegex(ValueError,'terminal order'):predict_ac(point,bad,base_kv=.4,base_mva=.1)
        badpoint=dict(supports=[[0],[1],[2],[0,1],[1,2]],r=[.1]*5,x=[.05]*5)
        with self.assertRaisesRegex(ValueError,'laminar'):family_network(badpoint,[101,102,103],base_kv=.4,base_mva=.1)

    def test_five_node_true_ac_correction_and_no_validation_argument(self):
        rng=np.random.default_rng(2026103007)
        labels=[101,102,103,104,105]
        family=[(i,) for i in range(5)]+[(0,1),(2,3),(0,1,2,3,4)]
        physical=dict(supports=[list(s) for s in family],r=[.03,.04,.025,.035,.02,.05,.04,.03],
                      x=[.015,.01,.02,.015,.02,.025,.03,.02])
        blank=observations(rng.uniform(.03,.15,(24,5)),rng.uniform(.01,.06,(24,5)),labels)
        generated=predict_ac(physical,blank,base_kv=.4,base_mva=.1)
        obs=(blank[0],blank[1],generated['voltage'],blank[3])
        initial=deepcopy(physical);initial['r']=(np.asarray(initial['r'])*.8).tolist();initial['x']=(np.asarray(initial['x'])*1.15).tolist()
        result=fit_family_ac(obs,family,base_kv=.4,base_mva=.1,initial_point=initial,outer_iterations=3)
        self.assertTrue(result['success']);self.assertFalse(result['settings']['validation_used'])
        self.assertEqual(result['settings']['coefficient_bound'],'unbounded')
        self.assertLess(result['objective'],result['initial_objective']-1e-8)
        self.assertTrue(all(row['trials'][-1]['objective']<row['current_objective'] for row in result['iterations'] if row['accepted']))
        self.assertEqual(set(map(tuple,result['native_point']['supports'])),set(family))
        self.assertTrue(result['final_convergence']['all_converged'])


if __name__=='__main__':unittest.main(verbosity=2)
