"""Locate the old model/AC gap using recorded baseline powers and the fixed quote."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
import argparse
import numpy as np
from ... import io
from ...observations import load
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..expected_logodds_admm_20261113.run import load_target
from ..expected_logodds_admm_20261113.evidence import direct_gains
from .diagnostics import PREVIOUS


def run(out):
    require(not out.exists(),'Keep previous diagnostics');out.mkdir(parents=True)
    rows=io.read(PREVIOUS/'confirmation/actual_results.json')
    entries=io.read(PREVIOUS/'confirmation/protocol.json')['policies'];records=[]
    for entry in entries:
        if entry['id']!='main__input054__101' or entry['target']!=6:continue
        data=load_target(entry,float(np.log(2)));o=data['original'];H=data['H'];j=entry['competitors'][0]
        path=entry['policies']['admm']['nominal']['path']
        require(sha(path)==entry['policies']['admm']['nominal']['sha256'],'Nominal policy changed')
        with np.load(path,allow_pickle=False) as a:u=a['mean_u']
        dp,dq=o.aggregate(u);shift=np.column_stack((dp,dq))/data['base'];D=H[6]-H[j];sigma2=data['score']['sigma2']
        baseline={r['replicate']:r for r in rows if r['id']==entry['id'] and r['target']==6 and r['strategy']=='no_dr' and r['status']=='complete'}
        for row in rows:
            if row['id']!=entry['id'] or row['target']!=6 or row['strategy']!='admm' or row['status']!='complete':continue
            obs0=load(baseline[row['replicate']]['observation']['path']);obs1=load(row['observation']['path'])
            p0=np.column_stack((obs0.p,obs0.q));p1=np.column_stack((obs1.p,obs1.q))
            b=p0@D.T;planned=shift@D.T;actual=(p1-p0)@D.T
            nominal_gain=float(np.sum(2*b*planned+planned**2)/(2*sigma2))
            actual_gain=float(np.sum(2*b*actual+actual**2)/(2*sigma2))
            records.append(dict(replicate=row['replicate'],nominal_on_actual_background_gain=nominal_gain,
                observed_power_distance_gain=actual_gain,execution_and_meter_difference=actual_gain-nominal_gain))
        result=dict(status='passed',paths=len(records),new_samples=0,source=io.ref(__file__),
            planned_on_forecast_gain=float(direct_gains(H,6,[j],data['forecast'],sigma2,o,data['base'],u)[0]),
            nominal_on_actual_background_mean=float(np.mean([r['nominal_on_actual_background_gain'] for r in records])),
            observed_power_distance_mean=float(np.mean([r['observed_power_distance_gain'] for r in records])),
            execution_and_meter_difference_mean=float(np.mean([r['execution_and_meter_difference'] for r in records])),
            interpretation='Diagnostic using already observed baseline powers, not an available planning input or a new success test.')
        io.write(out/'results.json',records);io.write(out/'summary.json',result)
        print({k:v for k,v in result.items() if k not in ('source','interpretation')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
