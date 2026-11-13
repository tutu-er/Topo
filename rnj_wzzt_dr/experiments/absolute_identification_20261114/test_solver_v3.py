"""Risk-allocation and exact two-hypothesis regression checks for V3."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..flexibility_sweep_20261113.execution import fixed_rotation
from .test_solver import ToyCase
from .solver_v3 import WeightedRecovery,plan


def run(out):
    if out.exists():raise ValueError('Preserve old checks')
    out.mkdir(parents=True)
    io.write(out/'protocol.json',dict(source=io.ref(Path(__file__).with_name('solver_v3.py')),
        tests=io.ref(__file__),new_AC_days=0))
    toy=ToyCase();x=np.arange(1.,5.)[:,None]/10
    val=SimpleNamespace(p=x,q=np.zeros_like(x),y=x)
    F=np.array([[1.,0.]])
    score=dict(parameters=[dict(H=[[0.,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=4.)],
        sigma2=1.,cells=4,reference_mse=1.,validation_tie=1e-12,
        models=[dict(topology_id='candidate0'),dict(topology_id='candidate1')],map_index=0)
    three=dict(score,parameters=[dict(H=[[0.,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=0.),dict(H=[[20.,0.]],history_sse=400.)],
        models=[dict(topology_id='c0'),dict(topology_id='c1'),dict(topology_id='c2')])
    weighted=WeightedRecovery(toy,three,val,F,1.)
    assert weighted.targets==[0,1]
    for target,record in weighted.allocation.items():
        assert abs(record['total']-.1)<1e-12
        assert len(record['comparisons'])==2
        assert min(v['alpha'] for v in record['comparisons'])>=.0005-1e-15
        assert max(v['alpha'] for v in record['comparisons'])>.09
    original=json.dumps(weighted.allocation,sort_keys=True)
    weighted.verify_expansion(np.array([[3.]]))
    assert json.dumps(weighted.allocation,sort_keys=True)==original
    report=dict(allocation_and_freeze='passed',synthetic={},old={})
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for mode in ('weighted_nominal','weighted_capped'):
            evidence=WeightedRecovery(toy,score,val,F,1.,mode,1.)
            for c in evidence.absolute:assert abs(c['alpha']-.1)<1e-14
            required=.5*(evidence.absolute[0]['z']+np.sqrt(evidence.absolute[0]['z']**2+2*(2+evidence.tie)))**2
            exact=np.sqrt(2*required)-1
            result,u=plan(toy,score,val,F,1.,mode=mode,seconds=2.,env=env,target_mass=1.,seed_points={'maximum':np.array([[4.]])})
            assert result['feasible'] and evidence.verify_expansion(u)['feasible']
            assert result['lower_bound_yuan']<=exact+1e-5 and result['cost_yuan']>=exact-1e-5
            report['synthetic'][mode]=dict(cost=result['cost_yuan'],exact_cost=exact,lower=result['lower_bound_yuan'])
        root=Path(__file__).resolve().parents[3];base=root/'outputs/customer_flexibility_sweep_20261113_v1'
        old=io.read(base/'histories/s0_c013/score.json');validation=load(base/'histories/s0_c013/validation.npz')
        case=CustomerCase(io.read(base/'resource_configs/higher.json'))
        with np.load(base/'confirmation/s0_c013/day000/planner.npz') as z:forecast=z['forecast'].copy()
        direction,_=fixed_rotation(case)
        for mode in ('weighted_nominal','weighted_capped'):
            evidence=WeightedRecovery(case,old,validation,forecast,200.,mode)
            for amount in (0.,.3,1.):
                info=evidence.verify_expansion(amount*direction)
                if info['feasible']:assert info['worst_recovery_lower']>=.9-1e-8
            for record in evidence.allocation.values():assert abs(record['total']-.1)<1e-12
            result,u=plan(case,old,validation,forecast,200.,mode=mode,seconds=2.,env=env,
                seed_points={'zero':np.zeros(case.shape),'rotation':direction})
            if result['feasible']:assert evidence.verify_expansion(u)['feasible'] and case.audit(u)['passed']
            io.write(out/(mode+'.json'),result)
            report['old'][mode]={k:result.get(k) for k in ('feasible','status','source','cost_yuan','relative_gap','seconds')}
    report['status']='passed';io.write(out/'summary.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.out.resolve()),indent=2))
