"""Independent saved-observation audit plus SSE-form residual guard checks."""
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from ... import io
from ...observations import load
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..linear_ac_population_20261113.audit import parallel
from .study import SCENARIOS


def run(out,phase,workers=4):
    destination=out/f'{phase}_guard_audit.json';require(not destination.exists(),'Preserve existing audit')
    cfg=io.read(out/'protocol.json');completion=io.read(out/f'{phase}_complete.json')
    require(not any(completion['worker_failures'].values()),'Worker failures require explicit review')
    receipts={}
    for scenario in SCENARIOS:
        root=out/scenario;receipt=root/phase/'independent_audit.json'
        receipts[scenario]=io.read(receipt) if receipt.exists() else parallel(root,phase,workers)
        require(receipts[scenario]['status']=='passed','Physical/posterior audit failed')
    entries={e['request']['id']:e for e in io.read(out/'nominal30/inputs.json')}
    max_pair=max_calibration=max_guard_error=max_guard_violation=max_capacity=0.;plans=qualified=proofs=unresolved=0;statuses=Counter()
    for identity in cfg[phase]['ids']:
        e=entries[identity];s=io.read(e['scores']['path']);H=np.array([p['H'] for p in s['parameters']],dtype=np.longdouble)
        history=load(e['request']['observed']['path']);P=np.c_[history.p[64:],history.q[64:]].astype(np.longdouble)
        Y=history.y[64:].astype(np.longdouble);mu=np.einsum('tc,knc->ktn',P,H)
        c=1+np.sum(mu*(Y[None]-mu),axis=1)/(np.sum(mu*mu,axis=1)+len(P)*s['sigma2']);means=c[:,:,None]*H
        for scenario,(alpha,guard) in SCENARIOS.items():
            with np.load(out/scenario/'calibration'/(identity+'.npz'),allow_pickle=False) as packet:
                max_calibration=max(max_calibration,float(np.max(abs(packet['mean']-means))))
        for day in range(cfg[phase]['days']):
            base=out/'nominal30'/phase/'jobs'/identity/f'day{day:03d}'
            with np.load(base/'planner.npz',allow_pickle=False) as p:forecast=p['forecast'].copy()
            with np.load(base/'physical.npz',allow_pickle=False) as p:
                shared={k:p[k].copy() for k in ('baseline_P','baseline_Q','true_root','forecast_noise',
                    'standardized_meter_P','standardized_meter_Q','standardized_meter_V','standardized_meter_root')}
            for scenario,(alpha,guard) in SCENARIOS.items():
                directory=out/scenario/phase/'jobs'/identity/f'day{day:03d}';records=io.read(directory/'plans.json')
                with np.load(directory/'physical.npz',allow_pickle=False) as p:
                    for k,value in shared.items():max_pair=max(max_pair,float(np.max(abs(p[k]-value))))
                with np.load(directory/'planner.npz',allow_pickle=False) as p:
                    max_pair=max(max_pair,float(np.max(abs(p['forecast']-forecast))))
                    for label,plan in records.items():
                        x=p['optimized_'+label];plans+=1;statuses[plan['status']]+=1
                        max_capacity=max(max_capacity,max(0.,float(np.max(abs(x)))-alpha))
                        max_capacity=max(max_capacity,max(0.,float(np.max(abs(p['rotation_'+label])))-alpha))
                        if not plan['qualified']:
                            require(np.all(x==0),'Unqualified quote did not fall back to zero')
                            proofs+=bool(plan.get('infeasibility_proved'));unresolved+=not bool(plan.get('infeasibility_proved'));continue
                        qualified+=1
                        if not guard:continue
                        F=forecast.astype(np.longdouble);dp=np.repeat(x,4,axis=0)*np.array(e['scale'])
                        U=F+np.c_[dp,cfg['qratio']*dp];values=[]
                        for j in e['competitors']:
                            for a,b in ((e['target'],j),(j,e['target'])):
                                def odds(Z):
                                    prediction=Z@means[a].T
                                    return (np.sum((prediction-Z@H[b].T)**2)-np.sum((prediction-Z@H[a].T)**2))/(2*s['sigma2'])
                                values.append(float(odds(U)-odds(F)))
                        if values:
                            max_guard_error=max(max_guard_error,float(np.max(abs(np.array(values)-plan['guard_gains']))))
                            max_guard_violation=max(max_guard_violation,max(0.,-min(values)))
    require(max_calibration<1e-12,'History calibration differs');require(max_pair<1e-12,'Scenarios are not paired')
    require(max_capacity<=1e-8,'Capacity violation');require(max_guard_error<1e-6,'Independent corrected SSE differs')
    require(max_guard_violation<=1e-7,'Historical-residual guard violation')
    result=dict(status='passed',created_utc=utc(),phase=phase,independent_physical=receipts,plans=plans,qualified=qualified,
        proven_infeasible=proofs,unresolved=unresolved,status_counts=dict(statuses),
        maximum_pairing_difference=max_pair,maximum_calibration_error=max_calibration,maximum_capacity_violation=max_capacity,
        maximum_guard_SSE_error=max_guard_error,maximum_guard_violation=max_guard_violation,
        paired_unique_new_days=completion['unique_new_days'],new_independent_days=0,source=io.ref(__file__))
    io.write(destination,result);print({k:v for k,v in result.items() if k not in ('source','independent_physical')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.run.resolve(),a.phase,a.workers)
