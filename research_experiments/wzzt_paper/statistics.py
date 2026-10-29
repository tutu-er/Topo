import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from research_experiments.wzzt_systematic.common import write


def interval(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:return dict(mean=float(a.mean()) if len(a) else None,lower=None,upper=None,independent_blocks=len(a),sign_p=None)
    rng=np.random.default_rng(20261029);means=a[rng.integers(0,len(a),size=(20000,len(a)))].mean(axis=1)
    nonzero=a[abs(a)>1e-12]
    return dict(mean=float(a.mean()),lower=float(np.quantile(means,.025)),upper=float(np.quantile(means,.975)),
        independent_blocks=len(a),positive_blocks=int((a>1e-12).sum()),negative_blocks=int((a < -1e-12).sum()),
        sign_p=float(binomtest(int((nonzero>0).sum()),len(nonzero),.5).pvalue) if len(nonzero) else 1.)


def statistics(destination):
    destination=Path(destination);f=pd.read_csv(destination/'per_job.csv');main=f[f.panel=='main'];contrasts=[]
    for reference in ('NJ','multistart','restart_control'):
        paired=main[main.variant.isin(['ridge_union',reference])].pivot(index=['job_id','bundle'],columns='variant',values='clade_f1').reset_index()
        effect=paired.ridge_union-paired[reference];blocks=effect.groupby(paired.bundle).mean()
        contrasts.append(dict(contrast='ridge_union vs '+reference,**interval(blocks),paired_inputs=len(paired),
            wins=int((effect>1e-10).sum()),ties=int((abs(effect)<=1e-10).sum()),losses=int((effect < -1e-10).sum())))
    running=0.
    for i,j in enumerate(sorted(range(len(contrasts)),key=lambda j:1. if contrasts[j]['sign_p'] is None else contrasts[j]['sign_p'])):
        if contrasts[j]['sign_p'] is None:contrasts[j]['holm_sign_p']=None;continue
        running=max(running,min(1.,(len(contrasts)-i)*contrasts[j]['sign_p']));contrasts[j]['holm_sign_p']=running
    write(destination/'primary_statistics.json',contrasts);pd.DataFrame(contrasts).to_csv(destination/'primary_statistics.csv',index=False)
    effects=[]
    for panel,group in f.groupby('panel'):
        for method in sorted(set(group.variant)-{'multistart'}):
            d=group[group.variant==method].copy()
            block='case' if panel=='topology' else 'bundle'
            effects.append(dict(panel=panel,variant=method,block=block,**interval(d.groupby(block).delta_vs_multistart.mean()),
                inputs=len(d),wins=int((d.delta_vs_multistart>1e-10).sum()),losses=int((d.delta_vs_multistart < -1e-10).sum())))
    pd.DataFrame(effects).to_csv(destination/'panel_effect_intervals.csv',index=False)
    cells=[]
    for (sigma,N,method),g in main.groupby(['voltage_sigma','N','variant']):
        cells.append(dict(voltage_sigma=sigma,N=N,variant=method,**interval(g.groupby('bundle').clade_f1.mean())))
    pd.DataFrame(cells).to_csv(destination/'main_cell_intervals.csv',index=False)
    g=f[f.variant=='ridge_union'].copy();g['unidentified']=g.clade_f1<1-1e-12
    g['pool_missing']=g.unidentified&(g.candidate_pool_true_recall<1-1e-12)
    import json
    g['crossing']=g.diagnostic_json.fillna('{}').map(lambda s:bool(json.loads(s).get('crossing_pairs',[])))
    g.groupby(['panel','condition','voltage_sigma','N']).agg(inputs=('job_id','count'),unidentified=('unidentified','sum'),pool_missing=('pool_missing','sum'),crossing=('crossing','sum')).to_csv(destination/'failure_modes.csv')
    f.groupby(['panel','variant','wzzt_search_status'],dropna=False).size().to_csv(destination/'termination_counts.csv',header=['rows'])
    print('statistical summaries saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args();statistics(a.destination)
