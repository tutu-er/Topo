"""Independent mean/variance/cap checks for V2, using only sealed old inputs."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import numpy as np

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..flexibility_sweep_20261113.execution import fixed_rotation
from .test_solver import ToyCase
from .solver_v2 import CalibratedSymmetricRecovery,plan


def run(out):
    if out.exists():raise ValueError('Preserve prior checks')
    out.mkdir(parents=True)
    io.write(out/'protocol.json',dict(source=io.ref(Path(__file__).with_name('solver_v2.py')),
        test=io.ref(__file__),new_AC_days=0,old_case='s0_c013 sealed day000'))
    toy=ToyCase()
    score=dict(parameters=[dict(H=[[0.,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=4.)],
        sigma2=1.,cells=4,reference_mse=1.,validation_tie=1e-12,
        models=[dict(topology_id='candidate0'),dict(topology_id='candidate1')],map_index=0)
    x=np.arange(1.,5.)[:,None]/10
    validation=SimpleNamespace(p=x,q=np.zeros_like(x),y=x)
    forecast=np.array([[1.,0.]])
    report=dict(synthetic={},frozen_expansion={},old_smoke={})
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for mode in ('multiplicative_absolute','capped_absolute'):
            evidence=CalibratedSymmetricRecovery(toy,score,validation,forecast,1.,mode,1.)
            assert np.allclose(evidence.variance,1.)
            assert np.allclose(evidence.means,evidence.H)
            required=.5*(evidence.z+np.sqrt(evidence.z**2+2*(2+evidence.tie)))**2
            exact=np.sqrt(2*required)-1
            receipt,u=plan(toy,score,validation,forecast,1.,mode=mode,seconds=2.,env=env,
                target_mass=1.,seed_points={'maximum':np.array([[4.]])})
            assert receipt['feasible'] and evidence.verify_expansion(u)['feasible']
            assert receipt['cost_yuan']>=exact-1e-5 and receipt['lower_bound_yuan']<=exact+1e-5
            report['synthetic'][mode]=dict(cost=receipt['cost_yuan'],exact_cost=exact,lower=receipt['lower_bound_yuan'])
        capscore=dict(score,parameters=[dict(H=[[.5,0.]],history_sse=0.),dict(H=[[1.,0.]],history_sse=0.)])
        capvalidation=SimpleNamespace(p=x,q=np.zeros_like(x),y=2*x)
        cap=CalibratedSymmetricRecovery(toy,capscore,capvalidation,forecast,1.,'capped_absolute',1.)
        info=cap.verify_expansion(np.array([[3.]]))
        assert info['cap_active_count']>0
        assert np.all(info['certificate_mean']<=info['actual_mean']+1e-12)
        assert np.all(info['pair_failure']>=info['actual_mean_pair_failure']-1e-12)
        report['cap_regression']='passed'
        root=Path(__file__).resolve().parents[3]
        base=root/'outputs/customer_flexibility_sweep_20261113_v1'
        old=io.read(base/'histories/s0_c013/score.json')
        val=load(base/'histories/s0_c013/validation.npz')
        resources=CustomerCase(io.read(base/'resource_configs/higher.json'))
        with np.load(base/'confirmation/s0_c013/day000/planner.npz') as z:F=z['forecast'].copy()
        direction,_=fixed_rotation(resources)
        for mode in ('multiplicative_absolute','capped_absolute'):
            evidence=CalibratedSymmetricRecovery(resources,old,val,F,200.,mode)
            max_mu_error=0.;max_var_error=0.
            for fraction in (0.,.3,1.):
                u=fraction*direction
                info=evidence.verify_expansion(u)
                P=F+resources.shift_pq_pu(u,200.)
                for position,c in enumerate(evidence.absolute):
                    i,j=c['left'],c['right']
                    mean=P@evidence.means[i].T
                    ri=mean-P@evidence.H[i].T;rj=mean-P@evidence.H[j].T
                    mu=(evidence.history[j]-evidence.history[i]+np.sum(rj*rj)-np.sum(ri*ri))/(2*evidence.sigma2)
                    max_mu_error=max(max_mu_error,abs(mu-info['actual_mean'][position]))
                    covariance=evidence.variance[i]*np.sum(((rj-ri)/evidence.sigma2)**2)
                    expanded=2*c['ratio']*info['J'][c['pair']]
                    max_var_error=max(max_var_error,abs(covariance-expanded))
            assert max_mu_error<1e-8 and max_var_error<1e-7
            report['frozen_expansion'][mode]=dict(mean_error=float(max_mu_error),variance_error=float(max_var_error),
                selected_targets=evidence.targets,comparisons=len(evidence.absolute),variance_ratios=(evidence.variance/evidence.sigma2).tolist())
            result,u=plan(resources,old,val,F,200.,mode=mode,seconds=2.,env=env,
                seed_points={'zero':np.zeros(resources.shape),'rotation':direction})
            if result['feasible']:
                assert resources.audit(u)['passed'] and evidence.verify_expansion(u)['feasible']
            io.write(out/(mode+'_old.json'),result)
            report['old_smoke'][mode]={k:result.get(k) for k in ('feasible','status','source','cost_yuan','relative_gap','seconds')}
    report['status']='passed'
    io.write(out/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.out.resolve()),indent=2))
