"""Post-policy decomposition of actual evidence; never called by procurement."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from ... import io
from .data import load_entry
from .driver import resources

def run(out):
    entries={e['id']:e for e in io.read(out/'inputs.json')};rows=[];cache={}
    for f in sorted(out.glob('days/*/*/results.json')):
        d=f.parent;records=io.read(f);entry=entries[d.parent.name]
        if entry['id'] not in cache:cache[entry['id']]=load_entry(entry)
        s,train,val,truth=cache[entry['id']];H=np.asarray([p['H'] for p in s['parameters']]);scale=s['sigma2']
        true=next((i for i,m in enumerate(s['models']) if m['topology_id']==truth['truth_id']),None)
        if true is None or len(H)<2:continue
        case=resources(entry)
        with np.load(d/'background.npz') as bg:root=bg['root'].copy()
        with np.load(d/'planner.npz') as z:forecast=z['forecast'].copy();controls={k:z[k].copy() for k in z.files if k!='forecast'}
        for physics in ('AC','linear'):
            with np.load(d/('no_dfs_'+physics+'.npz')) as z:
                ll=z['log_probability'];competitor=max((j for j in range(len(H)) if j!=true),key=lambda j:ll[j])
            D=H[true]-H[competitor];history=(s['parameters'][competitor]['history_sse']-s['parameters'][true]['history_sse'])/(2*scale)
            values={}
            for strategy,u in controls.items():
                path=d/(strategy+'_'+physics+'.npz')
                if not path.exists():continue
                with np.load(path) as z:
                    P=np.column_stack((z['true_P'],z['true_Q']));Y=root[:,None]**2-z['true_V']**2
                    distance=P@D.T;residual=Y-P@H[true].T
                    J=float(np.sum(distance*distance)/(2*scale));bias=float(np.sum(residual*distance)/scale)
                    ll=z['log_probability'];measured=float(ll[true]-ll[competitor]);clean=history+J+bias
                predicted=forecast+case.shift_pq_pu(u,200.)
                forecastJ=float(np.sum((predicted@D.T)**2)/(2*scale))
                values[strategy]=dict(actual_J=J,actual_bias=bias,clean_logodds=clean,measured_logodds=measured,
                    meter_contribution=measured-clean,predicted_J=forecastJ)
            baseline=values['no_dfs']
            for strategy,v in values.items():
                rows.append(dict(id=entry['id'],day=d.name,case=entry['case'],role=entry['role'],physics=physics,strategy=strategy,
                    true_index=true,competitor_index=competitor,history_logodds=history,**v,
                    **{'delta_'+k:v[k]-baseline[k] for k in v}))
    io.write(out/'evidence_decomposition.json',rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['physics'],r['strategy'],r['role'])].append(r)
    summary={}
    for key,values in sorted(groups.items()):
        fields=[k for k in values[0] if k.startswith('delta_')]
        summary['/'.join(key)]=dict(n=len(values),mean={k:float(np.mean([v[k] for v in values])) for k in fields},
            negative_measured_gain=sum(v['delta_measured_logodds']< -1e-8 for v in values),
            negative_clean_gain=sum(v['delta_clean_logodds']< -1e-8 for v in values),
            identity_max_error=max(abs(v['delta_measured_logodds']-v['delta_actual_J']-v['delta_actual_bias']-v['delta_meter_contribution']) for v in values))
    io.write(out/'evidence_summary.json',dict(groups=summary,scope='Offline true representative versus strongest noDFS representative competitor; same competitor fixed for all paired strategies. Physical200kWbase verified across all sourcecases; no new optimization or sampling.',input_protocol=io.ref(out/'protocol.json'),source=io.ref(__file__)))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=run(a.out.resolve());print('decomposition groups',len(s))
