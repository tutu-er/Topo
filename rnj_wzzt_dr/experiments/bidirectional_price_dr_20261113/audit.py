"""Replay the existing independent AC audit, then check capacities and pairing."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..linear_ac_population_20261113.audit import parallel
from .run import SCENARIOS


def run(out,phase,workers=4):
    destination=out/f'{phase}_capacity_audit.json'
    require(not destination.exists(),'Preserve previous audit')
    cfg=io.read(out/'protocol.json');complete=io.read(out/f'{phase}_complete.json')
    require(not any(complete['worker_failures'].values()),'Worker failures require review')
    independent={}
    for scenario in SCENARIOS:
        root=out/scenario;receipt=root/phase/'independent_audit.json'
        independent[scenario]=io.read(receipt) if receipt.exists() else parallel(root,phase,workers)
        require(independent[scenario]['status']=='passed','Physical/posterior audit failed')
    checked=plans=qualified=proofs=unresolved=zero_fallback=0;max_bound=max_pairing=0.;statuses=Counter()
    for identity in cfg[phase]['ids']:
        for day in range(cfg[phase]['days']):
            base=out/'unbounded'/phase/'jobs'/identity/f'day{day:03d}'
            with np.load(base/'planner.npz',allow_pickle=False) as p:forecast=p['forecast'].copy()
            with np.load(base/'physical.npz',allow_pickle=False) as p:
                background={k:p[k].copy() for k in ('baseline_P','baseline_Q','true_root','forecast_noise',
                    'standardized_meter_P','standardized_meter_Q','standardized_meter_V','standardized_meter_root')}
            baseline=next(r for r in io.read(base.parent/'results.json') if r['day']==day and r['strategy']=='no_dr')
            for scenario,alpha in SCENARIOS.items():
                directory=out/scenario/phase/'jobs'/identity/f'day{day:03d}'
                records=io.read(directory/'plans.json')
                other=next(r for r in io.read(directory.parent/'results.json') if r['day']==day and r['strategy']=='no_dr')
                for key in ('selected_index','p_true','p_target','retained_count','map_correct','correct_unique','wrong_unique'):
                    require(other.get(key)==baseline.get(key),'Unpaired baseline decision')
                with np.load(directory/'physical.npz',allow_pickle=False) as p:
                    for key,value in background.items():max_pairing=max(max_pairing,float(np.max(abs(p[key]-value))))
                with np.load(directory/'planner.npz',allow_pickle=False) as p:
                    max_pairing=max(max_pairing,float(np.max(abs(p['forecast']-forecast))))
                    for name in p.files:
                        if name=='forecast':continue
                        x=p[name];checked+=1
                        if alpha is not None:max_bound=max(max_bound,max(0.,float(np.max(abs(x)))-alpha))
                        if not name.startswith('optimized_'):continue
                        plan=records[name.split('_',1)[1]];plans+=1;statuses[plan['status']]+=1
                        if plan['qualified']:
                            qualified+=1;require(abs(float(np.mean(x*x))-plan['cost'])<1e-12,'Fee/action mismatch')
                        else:
                            require(np.all(x==0),'Failed quote must apply explicit zero fallback');zero_fallback+=1
                            if plan.get('infeasibility_proved'):
                                proofs+=1
                                if plan['status']=='cost_lower_bound_proves_infeasible':
                                    require(plan['lower_bound']>alpha**2,'Invalid box cost proof')
                                elif plan['status']=='box_upper_proves_infeasible':
                                    gamma=next(g for g in cfg['phase_gamma'][phase] if f'g{g:.6g}'==name.split('_',1)[1])
                                    require(min(plan['gain_upper'])<gamma,'Invalid gain upper proof')
                                else:require(plan.get('native',{}).get('status')==3,'Invalid native infeasibility proof')
                            else:unresolved+=1
    require(max_bound<=1e-8,'Capacity violation');require(max_pairing<1e-12,'Capacity scenarios use different backgrounds/noise')
    receipt=dict(status='passed',created_utc=utc(),phase=phase,independent_AC_posterior=independent,
        paired_unique_new_days=complete['unique_new_days'],new_independent_days=0,
        actions_checked=checked,plans_checked=plans,qualified=qualified,proven_infeasible=proofs,
        unresolved_or_exception=unresolved,explicit_zero_fallback=zero_fallback,status_counts=dict(statuses),
        maximum_normalized_capacity_violation=max_bound,maximum_pairing_difference=max_pairing,
        source=io.ref(__file__))
    io.write(destination,receipt);print({k:v for k,v in receipt.items() if k not in ('source','independent_AC_posterior')},flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.run.resolve(),a.phase,a.workers)
