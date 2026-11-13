"""Scaling preserves the full linear power/energy/temperature domain."""
import unittest
import numpy as np

from .catalog import resources


class CatalogChecks(unittest.TestCase):
    def setUp(self):
        self.entry=dict(case='paper15',labels=list(range(101,116)))
        self.one=resources(self.entry)
        self.two=resources(self.entry,2.)

    def test_full_domain_scales_with_capacity(self):
        a,b=self.one,self.two
        for name in ('baseline','lower','upper','solar_available','b_ub','b_eq'):
            np.testing.assert_allclose(getattr(b,name),2*getattr(a,name),rtol=1e-13,atol=1e-13)
        for name in ('A_eq','A_ub','qratio','fee_plus','fee_minus','clock'):
            np.testing.assert_allclose(getattr(b,name),getattr(a,name),rtol=0.,atol=0.)
        for k in a.thermal:
            for name in ('capacity_kwh_per_C','heat_demand_kw','lower_heat_kwh','upper_heat_kwh','electric_buffer_kwh'):
                np.testing.assert_allclose(b.thermal[k][name],2*np.asarray(a.thermal[k][name]),rtol=1e-13,atol=1e-13)

    def test_feasible_controls_delivery_fee_and_temperatures(self):
        a,b=self.one,self.two
        directions=[np.where((np.arange(a.H)[:,None]//2+np.arange(a.R)[None,:])%2==0,1.,-1.),
                    np.sin(np.arange(a.dimension).reshape(a.shape)*.37)]
        for direction in directions:
            u,_=a.linear_optimum(direction)
            self.assertTrue(a.audit(u)['passed'])
            self.assertTrue(b.audit(2*u)['passed'])
            self.assertTrue(a.audit((2*u)/2)['passed'])
            np.testing.assert_allclose(b.shift_pq_pu(2*u,200.),2*a.shift_pq_pu(u,200.),rtol=1e-13,atol=1e-13)
            self.assertAlmostEqual(b.fee(2*u),2*a.fee(u),places=11)
            for node,t in a.temperatures(u).items():np.testing.assert_allclose(b.temperatures(2*u)[node],t)
        for p,q in zip(a.device_baseline_pq_kw(),b.device_baseline_pq_kw(),strict=True):
            np.testing.assert_allclose(q,2*p)

    def test_labels_only_mapping_and_invalid_scale(self):
        entry=dict(case='eleven',labels=[9,15,21,28,36,49,65,70,81,97,104])
        a=resources(entry);b=resources(dict(entry,truth='ignored',fitted_H='ignored'),2)
        np.testing.assert_array_equal(a.node,b.node)
        self.assertEqual(a.labels,tuple(entry['labels']))
        for value in (0.,-1.,np.nan,np.inf):
            with self.assertRaises(ValueError):resources(entry,value)


if __name__=='__main__':unittest.main()
