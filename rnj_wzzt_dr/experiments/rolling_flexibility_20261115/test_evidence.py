"""Independent SSE, Gaussian integral and sequential-score regression tests."""
from copy import deepcopy
import unittest
import numpy as np

from ...observations import Observation
from .state import FrozenBank,ScoreState
from .evidence import RollingEvidence,Ledger,EpochResetRequired


def fixture():
    p=np.array([[1.],[.8],[1.2],[.9]]);q=np.array([[.1],[.2],[.3],[.4]])
    y=p.copy();v=np.sqrt(1.-.01*y);root=np.ones(4)
    # y is an algebra test score; meter medians are independently positive.
    obs=Observation(p,q,y,(1,),v,root)
    H=[[[1.,0.]],[[0.,1.]],[[1.,0.]]]
    native=[]
    for k,h in enumerate(H):
        s=float(np.square(y-np.c_[p,q]@np.asarray(h).T).sum())
        native.append(dict(H=h,history_sse=s,internal=[1,0,2][k],topology_id=['a','b','a'][k]))
    score=dict(native_bank=native,cells=4,sigma2=1.,reference_mse=1.,validation_tie=1e-12,native_index=0)
    return FrozenBank.from_score(score,obs),obs,score


class Case:
    N=1;R=2;repeat=1;base_kw=1.;labels=(1,)
    def __init__(self,T=4):
        self.H=T;self.shape=(T,2);self.lower=-5*np.ones(self.shape);self.upper=5*np.ones(self.shape)
    def array(self,u):
        u=np.asarray(u,float)
        if u.shape!=self.shape or not np.isfinite(u).all(): raise ValueError('shape')
        return u
    def response_effect(self,D,base): return np.asarray(D)/base
    def shift_pq_pu(self,u,base): return self.array(u)/base
    def audit(self,u):
        u=self.array(u);return dict(passed=bool((u>=self.lower).all() and (u<=self.upper).all()))


class EvidenceTests(unittest.TestCase):
    def test_profile_duplicate_invariant_and_native_not_representative_can_win(self):
        bank,obs,score=fixture();state=ScoreState(bank);snap=state.snapshot()
        self.assertEqual(snap['native_target'],0)
        self.assertEqual(len(snap['weights']),2)
        expected=1/(1+np.exp(-(bank.initial_sse[1]-bank.initial_sse[0])/2))
        self.assertAlmostEqual(snap['weights']['a'],expected)
        # Same active topology, different stored zero-weight support counts is legal.
        self.assertEqual(bank.internal[2],2)
        new=Observation(np.array([[1.]]),np.array([[4.]]),np.array([[4.]]),(1,))
        state.update(new,step=0)
        self.assertEqual(state.snapshot()['native_target'],1)
        self.assertEqual(state.snapshot()['cells'],5)
        direct=bank.initial_sse+np.square(4-np.einsum('knc,c->kn',bank.H,[1.,4.])).sum(1)
        np.testing.assert_allclose(state.sse,direct)
        with self.assertRaises(ValueError):state.update(new,step=0)
        with self.assertRaises(ValueError):state.update(obs)
        restored=ScoreState(bank);restored.restore(state.snapshot())
        np.testing.assert_array_equal(restored.sse,state.sse)

    def test_all_native_calibration_matches_scalar_closed_form(self):
        bank,obs,_=fixture();X=np.c_[obs.p,obs.q]
        for i,H in enumerate(bank.H):
            mu=X@H.T
            c=(np.sum(mu*obs.y,axis=0)+len(X)*bank.sigma2)/(np.sum(mu*mu,axis=0)+len(X)*bank.sigma2)
            np.testing.assert_allclose(bank.means[i],c[:,None]*H)
        with self.assertRaises(ValueError):bank.H[0,0,0]=2

    def test_pair_mean_and_variance_against_direct_sse(self):
        bank,_,_=fixture();state=ScoreState(bank);case=Case()
        forecast=np.tile([1.,.2],(case.H,1));u=np.array([[.2,-.1],[.4,.3],[-.1,.5],[.6,-.2]])
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            model=RollingEvidence(case,bank,state.snapshot(),forecast,mode=mode)
            result=model.verify_expansion(u);x=forecast+u;t=model.target;j=model.rows[0]['competitor_native']
            Y=x@bank.means[t].T
            raw=(np.square(Y-x@bank.H[j].T).sum()-np.square(Y-x@bank.H[t].T).sum())/(2*bank.sigma2)
            self.assertAlmostEqual(result['mu'][0],raw+model.rows[0]['h']-model.tie)
            d=x@(bank.H[t]-bank.H[j]).T
            variance=float(sum(v@model.sigma_e@v for v in d))/bank.sigma2**2
            self.assertAlmostEqual(result['variance'][0],variance)
            # Fixed-power MGF is the elementary Gaussian linear-score MGF.
            if mode=='mgf_fixed':
                lam=model.rows[0]['lambda_selected']
                expected=-lam*result['mu'][0]+lam*lam*variance/2
                self.assertAlmostEqual(result['log_pair_failure_bound'][0],expected)

    def test_random_power_integral_independent_joint_gaussian(self):
        bank,_,_=fixture();case=Case(2);f=np.tile([.9,.3],(2,1))
        model=RollingEvidence(case,bank,ScoreState(bank).snapshot(),f,mode='mgf_forecast')
        u=np.array([[.4,-.1],[.1,.2]]);x=f+u
        row=model.rows[0];t,j=row['target_native'],row['competitor_native'];lam=row['lambda_selected']
        D=bank.H[t]-bank.H[j];E=bank.means[t]-bank.H[t]
        B=(D.T@D+E.T@D+D.T@E)/(2*bank.sigma2)
        A=np.block([[B,D.T/(2*bank.sigma2)],[D/(2*bank.sigma2),np.zeros((1,1))]])
        log_mgf=-lam*(row['h']-row['threshold'])
        for k,m in enumerate(x):
            covariance=np.zeros((3,3));covariance[:2,:2]=np.diag(model.power_sd[k]**2);covariance[2:,2:]=model.sigma_e
            C=np.linalg.cholesky(covariance);G=np.eye(3)+2*lam*C.T@A@C
            z=np.r_[m,0.];a=C.T@A@z
            log_mgf+=-.5*np.linalg.slogdet(G)[1]-lam*z@A@z+2*lam*lam*a@np.linalg.solve(G,a)
        self.assertAlmostEqual(model.direct(u)['log_pair_failure_bound'][0],log_mgf,places=10)

    def test_profile_threshold_implies_weight(self):
        for G in (2,3,9):
            beta=.9;B=np.log((G-1)*beta/(1-beta))
            self.assertAlmostEqual(1/(1+(G-1)*np.exp(-B)),beta)

    def test_ledger_freezes_weights_and_rejects_epoch_changes(self):
        bank,_,_=fixture();case=Case();snap=ScoreState(bank).snapshot();f=np.tile([1.,.2],(4,1))
        model=RollingEvidence(case,bank,snap,f,mode='mgf_fixed')
        action=np.tile([3.,0.],(4,1))
        self.assertTrue(model.direct(action)['feasible'])
        ledger=Ledger.from_plan(model,action,epoch_id='first')
        constrained=RollingEvidence(case,bank,snap,f,mode='mgf_fixed',progress=ledger)
        self.assertTrue(constrained.verify_expansion(action)['feasible'])
        self.assertLess(constrained.direct(np.zeros(case.shape))['progress_margins'][0],0)
        bad=deepcopy(snap);bad['physical_epoch']=1
        with self.assertRaises(EpochResetRequired):RollingEvidence(case,bank,bad,f,mode='mgf_fixed',progress=ledger)

    def test_single_family_has_no_fabricated_competitor(self):
        bank,obs,score=fixture();score['native_bank']=[score['native_bank'][0]]
        one=FrozenBank.from_score(score,obs);case=Case(1)
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            model=RollingEvidence(case,one,ScoreState(one).snapshot(),np.array([[1.,.2]]),mode=mode)
            self.assertEqual(model.rows,[]);self.assertTrue(model.verify_expansion(np.zeros(case.shape))['feasible'])


if __name__=='__main__':unittest.main()
