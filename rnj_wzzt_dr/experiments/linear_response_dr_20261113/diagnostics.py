"""Recompute old observations: distance changes and target-model residual effects."""
from pathlib import Path
import argparse
import numpy as np
from ... import io
from ...observations import load
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..expected_logodds_admm_20261113.run import REFERENCE
from ..expected_logodds_admm_20261113.confirm import means_interval

PREVIOUS=ROOT/'outputs/expected_logodds_admm_20261113_v2'


def run(out):
    require(not out.exists(),'Keep existing diagnostics');out.mkdir(parents=True)
    raw=PREVIOUS/'confirmation/actual_results.json'
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),previous=io.ref(raw),
        new_samples=0,definition='Jdist=||D P||^2/(2 sigma2); exact odds=history+Jdist+<e,D P>/sigma2',
        old_J_note='Older code uses J for several different quantities; exponential score d^T W d and normalized residual loss S/2 are distinct from Jdist. No numerical equality across versions assumed.'))
    rows=io.read(raw);records=[];cache={}
    for row in rows:
        if row['status']!='complete':continue
        identity=row['id'];target=row['target']
        if identity not in cache:
            score=io.read(REFERENCE/'scores'/(identity+'.json'))
            cache[identity]=score,np.asarray([p['H'] for p in score['parameters']])
        score,H=cache[identity];sigma2=score['sigma2']
        source=REFERENCE/'development/targets'/(identity+f'__target{target}.json')
        competitors=io.read(source)['gates']['0.9']['allocation']['active']
        require(sha(row['observation']['path'])==row['observation']['sha256'],'Saved observations changed')
        obs=load(row['observation']['path']);power=np.column_stack((obs.p,obs.q));residual=obs.y-power@H[target].T
        for j in competitors:
            d=power@(H[target]-H[j]).T
            distance=float(np.sum(d*d)/(2*sigma2));cross=float(np.sum(residual*d)/sigma2)
            history=score['models'][target]['log_probability']-score['models'][j]['log_probability']
            direct=float((np.sum((obs.y-power@H[j].T)**2)-np.sum(residual**2))/(2*sigma2))
            require(abs(distance+cross-direct)<1e-7,'Exact distance/residual decomposition failed')
            records.append(dict(id=identity,target=target,strategy=row['strategy'],replicate=row['replicate'],competitor=j,
                distance=distance,raw_distance=2*sigma2*distance,cross=cross,new_log_odds=history+direct,
                historical_log_odds=history,p_target=row['p_target']))
    aggregates=[];paired=[]
    for identity,target,strategy,j in sorted(set((r['id'],r['target'],r['strategy'],r['competitor']) for r in records)):
        group=[r for r in records if (r['id'],r['target'],r['strategy'],r['competitor'])==(identity,target,strategy,j)]
        item=dict(id=identity,target=target,strategy=strategy,competitor=j,paths=len(group))
        for k in ('distance','raw_distance','cross','new_log_odds','historical_log_odds','p_target'):item['mean_'+k]=float(np.mean([r[k] for r in group]))
        aggregates.append(item)
        if strategy=='no_dr':continue
        baseline={r['replicate']:r for r in records if (r['id'],r['target'],r['strategy'],r['competitor'])==(identity,target,'no_dr',j)}
        change=dict(id=identity,target=target,strategy=strategy,competitor=j,paths=len(group))
        for k in ('distance','raw_distance','cross','new_log_odds','p_target'):
            change[k]=means_interval(np.array([r[k]-baseline[r['replicate']][k] for r in group]))
        require(abs(change['distance']['mean']+change['cross']['mean']-change['new_log_odds']['mean'])<1e-8,'Paired decomposition failed')
        paired.append(change)
    io.write(out/'aggregates.json',aggregates);io.write(out/'paired.json',paired)
    io.write(out/'complete.json',dict(status='passed',observation_pair_comparisons=len(records),new_samples=0,
        input_sha256=sha(raw),source_sha256=sha(__file__)))
    for x in paired:
        if x['id']=='main__input054__101' and x['target']==6 and x['strategy']=='admm':
            print({k:x[k]['mean'] for k in ('distance','cross','new_log_odds','p_target')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
