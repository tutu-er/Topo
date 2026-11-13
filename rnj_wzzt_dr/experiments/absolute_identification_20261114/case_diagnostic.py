"""Offline frozen-model evidence decomposition for a reported fixed history."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ... import io
from .data import load_entry


def diagnose(run,identity,out):
    if out.exists():raise ValueError('Preserve previous diagnosis')
    entry=next(e for e in io.read(run/'inputs.json') if e['id']==identity)
    s,train,val,truth=load_entry(entry);H=np.array([p['H'] for p in s['parameters']]);variance=s['sigma2']
    target=next(i for i,m in enumerate(s['models']) if m['topology_id']==truth['truth_id'])
    rows=[];statuses={m:Counter() for m in ('symmetric_guard','weighted_nominal')}
    for d in sorted((run/'days'/identity).glob('day*')):
        if not (d/'complete.json').exists():raise ValueError('Incomplete day')
        plans=io.read(d/'plans.json')
        for mode in statuses:statuses[mode][plans[mode]['status']]+=1
        with np.load(d/'background.npz') as z:root=np.array(z['root'])
        with np.load(d/'planner.npz') as z:forecast=np.array(z['forecast'])
        for physics in ('AC','linear'):
            with np.load(d/('no_dfs_'+physics+'.npz')) as z:
                power=np.c_[z['true_P'],z['true_Q']];Y=root[:,None]**2-z['true_V']**2
                logp=z['log_probability'];competitor=max((j for j in range(len(H)) if j!=target),key=lambda j:logp[j])
                measured=float(logp[target]-logp[competitor])
            D=H[target]-H[competitor];difference=power@D.T;residual=Y-power@H[target].T
            history=(s['parameters'][competitor]['history_sse']-s['parameters'][target]['history_sse'])/(2*variance)
            J=float(np.sum(difference**2)/(2*variance));bias=float(np.sum(residual*difference)/variance)
            clean=float(history+J+bias);forecastJ=float(np.sum((forecast@D.T)**2)/(2*variance))
            direct=history+(float(np.sum((Y-power@H[competitor].T)**2))-float(np.sum(residual**2)))/(2*variance)
            if abs(direct-clean)>1e-8:raise ArithmeticError('Evidence identity failed')
            rows.append(dict(day=d.name,physics=physics,true_index=target,competitor_index=competitor,
                history_logodds=history,forecast_J=forecastJ,actual_J=J,actual_bias=bias,clean_logodds=clean,
                measured_logodds=measured,meter_contribution=measured-clean,identity_error=direct-clean))
    aggregate={}
    for physics in ('AC','linear'):
        own=[r for r in rows if r['physics']==physics]
        fields=['history_logodds','forecast_J','actual_J','actual_bias','clean_logodds','measured_logodds','meter_contribution']
        aggregate[physics]=dict(days=len(own),competitors=dict(Counter(r['competitor_index'] for r in own)),
            mean={f:float(np.mean([r[f] for r in own])) for f in fields},
            minimum={f:float(min(r[f] for r in own)) for f in fields},maximum={f:float(max(r[f] for r in own)) for f in fields},
            positive_clean=sum(r['clean_logodds']>0 for r in own),positive_measured=sum(r['measured_logodds']>0 for r in own))
    result=dict(source=io.ref(__file__),batch=io.ref(run/'protocol.json'),score=entry['score'],
        input={k:entry[k] for k in ('id','case','role','profile','level','voltage_sigma','p_map','history_partition')},
        true_index=target,historical_target=s['map_index'],candidates=len(H),quote_statuses={k:dict(v) for k,v in statuses.items()},
        interpretation='Offline true representative versus strongest no-service representative competitor on each day. clean_logodds=history_logodds+actual_J+actual_bias; measured adds meter contribution. Physical true P/Q/V are used only for this after-execution explanation, never for planning.',
        aggregate=aggregate,rows=rows)
    io.write(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=diagnose(a.run.resolve(),a.id,a.out.resolve());print(r['input'],r['quote_statuses'],r['aggregate'])
