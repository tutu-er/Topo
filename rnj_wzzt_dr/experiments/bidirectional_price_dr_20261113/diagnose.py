"""Offline explanation of old AC evidence changes; true parameters never plan."""
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,utc
from ..unified_posterior_dr_20261112.physical import open_runtime

OLD=ROOT/'outputs/linear_ac_population_20261113_v2'
IDS=('main__input039__101','main__input195__101','pressure__input026__101','pressure__input025__101')


def physical_H(module,meta,labels):
    network=module.load_case(meta['case']);order,incoming,desc=module._day96_descendants(network,labels)
    if meta['profile']=='rx_ratio':
        for edge in incoming.values():
            r,x=edge['r_ohm'],edge['x_ohm'];magnitude=np.hypot(r,x);denominator=np.hypot(meta['level']*r,x)
            if denominator:edge['r_ohm'],edge['x_ohm']=magnitude*meta['level']*r/denominator,magnitude*x/denominator
    elif meta['profile']=='weak_edge':
        weak=meta['weak_edge']
        for edge in incoming.values():
            if edge['from_bus']==weak['from_bus'] and edge['to_bus']==weak['to_bus']:
                edge['r_ohm'],edge['x_ohm']=weak['r_ohm'],weak['x_ohm']
    physical_rows=[[e['from_bus'],e['to_bus'],e['r_ohm'],e['x_ohm']] for e in incoming.values()]
    require(module._day96_digest(physical_rows)==meta['physical_branch_sha256'],'Offline physical version differs')
    n=len(labels);R=np.zeros((n,n));X=R.copy();zbase=network['base_kv']**2/network['base_mva']
    for child,edge in incoming.items():
        C=list(desc[child])
        if C:
            R[np.ix_(C,C)]+=2*edge['r_ohm']/zbase;X[np.ix_(C,C)]+=2*edge['x_ohm']/zbase
    return np.column_stack((R,X))


def run(out,phase='confirmation',identities=IDS):
    require(not out.exists(),'Preserve earlier diagnosis');out.mkdir(parents=True)
    entries={e['request']['id']:e for e in io.read(OLD/'inputs.json')}
    truth={e['id']:e for e in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')}
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),ids=list(identities),phase=phase,
        scope='offline decomposition of already completed actual AC; no true matrix enters new planning',new_days=0))
    modules={};aggregate=[];records=[]
    for identity in identities:
        entry=entries[identity];score=io.read(entry['scores']['path']);H=np.array([p['H'] for p in score['parameters']]);sigma2=score['sigma2']
        ti=next(i for i,m in enumerate(score['models']) if m['topology_id']==truth[identity]['truth_id'])
        planned=entry['target'];cohort=entry['request']['cohort']
        if cohort not in modules:modules[cohort]=open_runtime(OLD,cohort)[0]
        counts=Counter()
        for d in sorted((OLD/phase/'jobs'/identity).glob('day*')):
            meta=io.read(d/'generation.json');Hp=physical_H(modules[cohort],meta,score['labels'])
            with np.load(d/'physical.npz',allow_pickle=False) as physical,np.load(d/'observed.npz',allow_pickle=False) as obs,np.load(d/'planner.npz',allow_pickle=False) as packet:
                names=obs['strategy'].tolist();baseline={};current={}
                for strategy in ('no_dr','optimized_g0.693147','optimized_g2','optimized_g4'):
                    if strategy not in names:continue
                    i=names.index(strategy);P=np.column_stack((physical['P'][i],physical['Q'][i]));Y=physical['true_root'][:,None]**2-physical['V'][i]**2
                    ll=obs['log_probability'][i];counts[strategy,int(np.argmax(ll))]+=1
                    for j in range(len(H)):
                        if j==ti:continue
                        D=H[ti]-H[j];difference=P@D.T
                        distance=float(np.sum(difference**2)/(2*sigma2))
                        parameter=float(np.sum((P@(Hp-H[ti]).T)*difference)/sigma2)
                        AC=float(np.sum((Y-P@Hp.T)*difference)/sigma2)
                        history=score['models'][ti]['log_probability']-score['models'][j]['log_probability']
                        observed=float(ll[ti]-ll[j]);meter=observed-history-distance-parameter-AC
                        item=dict(distance=distance,parameter=parameter,AC=AC,meter=meter,log_odds=observed)
                        if strategy=='no_dr':baseline[j]=item
                        else:
                            change={k:item[k]-baseline[j][k] for k in item}
                            require(abs(sum(change[k] for k in ('distance','parameter','AC','meter'))-change['log_odds'])<1e-7,'Evidence decomposition differs')
                            records.append(dict(id=identity,strategy=strategy,competitor=j,truth_index=ti,historical_target=planned,
                                target_correct=ti==planned,**change))
        for strategy,j in sorted(set((r['strategy'],r['competitor']) for r in records if r['id']==identity)):
            rows=[r for r in records if r['id']==identity and r['strategy']==strategy and r['competitor']==j]
            aggregate.append(dict(id=identity,strategy=strategy,competitor=j,truth_index=ti,historical_target=planned,
                target_correct=ti==planned,paths=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ('distance','parameter','AC','meter','log_odds')},
                competitor_MAP_count=counts[strategy,j]))
    io.write(out/'details.json',records);io.write(out/'summary.json',aggregate)
    io.write(out/'complete.json',dict(status='passed',new_days=0,new_AC_calls=0,source=io.ref(__file__)))
    for r in aggregate:
        if r['id']=='main__input039__101' and r['strategy']=='optimized_g4' and r['competitor']==r['historical_target']:print(r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['population','confirmation'],default='confirmation');p.add_argument('--ids',nargs='+')
    a=p.parse_args();run(a.output.resolve(),a.phase,IDS if a.ids is None else a.ids)
