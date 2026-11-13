"""Closed-form two-candidate checks and frozen-case smoke; no AC execution."""
from pathlib import Path
import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..flexibility_sweep_20261113.execution import fixed_rotation
from .solver import SymmetricRecovery, plan, target_set


class ToyCase:
    N=1; H=1; repeat=1; R=1; shape=(1,1)
    node=np.array([0]); qratio=np.array([0.])
    lower=np.array([[-1.]]); upper=np.array([[4.]])
    def shift_pq_pu(self,u,base):
        return np.c_[np.asarray(u).reshape(-1)/base,np.zeros(1)]
    def fee(self,u):
        return float(np.abs(u).sum())
    def audit(self,u):
        return dict(passed=bool(np.shape(u)==self.shape and np.all(u>=self.lower-1e-7) and np.all(u<=self.upper+1e-7)))
    def attach(self,m):
        u=m.addMVar(self.shape,lb=self.lower,ub=self.upper)
        absolute=m.addMVar(self.shape,lb=0.)
        m.addConstr(absolute>=u);m.addConstr(absolute>=-u)
        return u,absolute.sum()


def run(out):
    if out.exists():
        raise ValueError('Preserve existing checks')
    out.mkdir(parents=True)
    io.write(out/'protocol.json',dict(solver=io.ref(Path(__file__).with_name('solver.py')),
        test=io.ref(__file__),new_AC_days=0,old_case='s0_c013 sealed day000 forecast'))
    case=ToyCase()
    score=dict(parameters=[dict(H=[[0.,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=4.)],
        sigma2=1.,cells=4,reference_mse=1.,validation_tie=1e-12,
        models=[dict(topology_id='candidate0'),dict(topology_id='candidate1')],map_index=0)
    valid=SimpleNamespace(p=np.arange(1.,5.)[:,None],q=np.zeros((4,1)),y=np.arange(1.,5.)[:,None])
    forecast=np.array([[1.,0.]])
    model=SymmetricRecovery(case,score,valid,forecast,1.,target_mass=1.)
    h=2.
    required=.5*(model.z+np.sqrt(model.z**2+2*(h+model.tie)))**2
    for value in (0.,.2,1.,2.,3.,4.):
        u=np.array([[value]])
        info=model.verify_expansion(u)
        J=(1+value)**2/2
        expected=bool(J>=required)
        assert info['feasible']==expected
        # Both possible true hypotheses are explicitly protected.
        for i,offset in ((0,h),(1,-h)):
            error=float(norm.cdf((model.tie-offset-J)/np.sqrt(2*J)))
            assert abs(error-info['target_failure_sums'][str(i)])<1e-12
    ids,_=target_set([0.,0.,100.],1.,.49)
    assert ids==[0,1]  # cumulative boundary must retain tied candidates.
    ids,_=target_set([0.,1.,10000.],1.,1.)
    assert len(ids)==3  # full mode retains even numerically tiny weight.
    tied=dict(score,parameters=[dict(H=[[1.,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=0.)])
    assert not SymmetricRecovery(case,tied,valid,forecast,1.,target_mass=1.).direct(np.zeros((1,1)))['feasible']
    report=dict(toy_closed_form='passed',tie_coverage='passed',zero_distance_regression='passed',toy_smoke={},old_smoke={})
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for mode in ('symmetric','symmetric_guard'):
            result,u=plan(case,score,valid,forecast,1.,mode=mode,seconds=2.,env=env,
                seed_points={'common_maximum':np.array([[4.]])},target_mass=1.)
            assert result['feasible'] and case.audit(u)['passed']
            assert model.direct(u)['feasible']
            exact_cost=np.sqrt(2*required)-1
            assert result['lower_bound_yuan']<=exact_cost+1e-5
            assert result['cost_yuan']>=exact_cost-1e-5
            io.write(out/(mode+'_toy.json'),result)
            report['toy_smoke'][mode]=dict(cost=result['cost_yuan'],closed_form_cost=exact_cost,
                lower=result['lower_bound_yuan'],status=result['status'],source=result['source'])
        root=Path(__file__).resolve().parents[3]
        base=root/'outputs/customer_flexibility_sweep_20261113_v1'
        old=io.read(base/'histories/s0_c013/score.json')
        val=load(base/'histories/s0_c013/validation.npz')
        resources=CustomerCase(io.read(base/'resource_configs/higher.json'))
        with np.load(base/'confirmation/s0_c013/day000/planner.npz') as z:
            F=z['forecast'].copy()
        direction,_=fixed_rotation(resources)
        for mode in ('symmetric','symmetric_guard'):
            evidence=SymmetricRecovery(resources,old,val,F,200.,mode)
            for scale in (0.,.3,1.):
                point=scale*direction
                assert resources.audit(point)['passed']
                data=evidence.verify_expansion(point)
                assert np.all(data['J']<=evidence.upper+1e-7)
            result,u=plan(resources,old,val,F,200.,mode=mode,seconds=2.,env=env,
                seed_points={'zero':np.zeros(resources.shape),'rotation':direction})
            if result['feasible']:
                assert resources.audit(u)['passed'] and evidence.verify_expansion(u)['feasible']
            io.write(out/(mode+'_old.json'),result)
            report['old_smoke'][mode]={k:result.get(k) for k in ('feasible','status','source','cost_yuan','relative_gap','seconds','targets','pair_count','guard_count')}
    report['status']='passed'
    io.write(out/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.out.resolve()),indent=2))
