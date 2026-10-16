import unittest
import numpy as np
from . import ev_mobile_v3_proxy_AC_audit as audit


class ProxyAuditTests(unittest.TestCase):
    def test_free_LS_scaled_rank_deficient(self):
        J=np.array([[1.,100.],[2.,200.],[0.,0.]])
        solver=audit.FreeLS(J);z,r,q=solver.solve(np.array([2.,4.,3.]))
        self.assertEqual(solver.rank,1)
        np.testing.assert_allclose(r,[0.,0.,3.],atol=1e-12)
        self.assertAlmostEqual(q,4.5)
        np.testing.assert_allclose(J.T@r,0.,atol=1e-10)

    def test_channel_major_H_and_E(self):
        s=audit.channel_squares(np.arange(1.,13.),1,2,2)
        self.assertEqual(s['blocks']['H']['channels'],dict(v=5.,master_p=9.,master_q=16.))
        self.assertEqual(s['blocks']['E']['channels']['master_p'],181.)
        self.assertEqual(s['blocks']['E']['channels']['master_q'],265.)

    def test_parameter_domain_detects_bounds_and_cone(self):
        cfg=dict(training=dict(r_bounds=[.001,.1],x_bounds=[.001,.1]),ratio_bounds=[.1,.4],
            source=dict(source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5]))
        state=dict(r=np.array([.01]),x=np.array([.02]),source_H=np.array([[.8,0.]]),source_E=np.array([[0.,0.]]))
        d=audit.domain_check(state,cfg)
        self.assertFalse(d['feasible']);self.assertEqual(d['violations']['X_R_cone'],1)
        self.assertEqual(d['violations']['source_H_P'],1)


if __name__=='__main__':unittest.main()
