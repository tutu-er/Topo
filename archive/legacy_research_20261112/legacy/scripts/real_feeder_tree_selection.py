"""Simplified exact finite-tree enumeration with integer theft selection.

Enumerating a three-valued topology switch is exact over the retained pool.
Each branch still solves a source-location MILP, never continuous theft LP.
Normal-only train/validation chooses the pool before seeing the theft window.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import lsq_linear

try:
    from scripts import real_feeder_phase_wzzt as phase
except ModuleNotFoundError:
    import real_feeder_phase_wzzt as phase
base=phase.base


def subset(data, indices):
    return {key:value[indices] for key,value in data.items() if isinstance(value,np.ndarray)}


def candidate_trees(history, keep=3):
    """RNJ candidates ranked by normal-only held-out phase voltage prediction."""
    n=history['p'].shape[1];split=3*len(history['p'])//4
    training=subset(history,slice(0,split));validation=subset(history,slice(split,None))
    coef=np.linalg.lstsq(np.column_stack([training['p'],training['q']]),training['y'],rcond=None)[0]
    r=(coef[:n]+coef[:n].T)/2;x=(coef[n:]+coef[n:].T)/2
    candidates={}
    for alpha in (1.,.75,0.):
        scores=alpha*r/max(np.linalg.norm(r),1e-12)+(1-alpha)*x/max(np.linalg.norm(x),1e-12)
        depths=np.maximum(np.diag(scores),0);scale=max(np.median(depths),1e-6)
        for fraction in (0.,.02,.08,.16):
            tree=base.rooted_neighbor_joining(scores,depths,list(range(n)),-1000,fraction*scale)
            graph=nx.Graph();graph.add_weighted_edges_from(tree.edges)
            directed=nx.bfs_tree(graph,tree.root)
            pool={frozenset([i]) for i in range(n)}
            for node in directed:
                if node==tree.root:continue
                support=(nx.descendants(directed,node)|{node})&set(range(n))
                if support:pool.add(frozenset(support))
            supports=tuple(sorted(pool,key=lambda s:(len(s),sorted(s))))
            if supports in candidates:continue
            _,design=phase.features(training,supports)
            # Continuous normal-load parameter fit only, not theft inference.
            ridge=.03
            fitting=lsq_linear(np.vstack([design,ridge*np.eye(design.shape[1])]),
                              np.r_[training['yp'].ravel(),np.zeros(design.shape[1])],
                              bounds=(0,1),tol=1e-8,max_iter=200)
            if not fitting.success:continue
            weights=fitting.x.reshape(len(supports),4)
            mae=float(np.mean(abs(phase.phase_predict(validation,supports,weights)-validation['yp'])))
            candidates[supports]=dict(normal_validation_mae=mae,rnj_alpha=alpha,tolerance_fraction=fraction)
    ordered=sorted(candidates,key=lambda key:(candidates[key]['normal_validation_mae'],len(key)))[:keep]
    assert ordered
    return ordered,[candidates[key] for key in ordered]


def objective(fitted,supports):
    if 'weights' not in fitted:return float('inf')
    return fitted['train_mae']+.00015*sum(len(s)>1 for s in supports)+.001*float(np.array(fitted['weights']).sum())


def solve_one(data,supports,amplitude,seconds):
    return phase.fit_phase(data,supports,amplitude,fixed_supports=list(range(len(supports))),seconds=seconds)


def main(output,seeds=2,count=64,seconds=30,source_bus=5):
    output=output.resolve();assert output.is_relative_to(base.ROOT);output.mkdir(parents=True,exist_ok=True)
    feeder=base.read_feeder()
    protocol=dict(dataset=feeder.metadata,seeds=seeds,current_samples=count,history_samples=128,holdout_samples=96,
                  source_bus=source_bus,conditions=['clean','persistent','intermittent'],candidate_trees=3,
                  candidate_selection='RNJ from first 96 normal samples; rank by remaining 32 normal phase-voltage observations',
                  reference='known root voltage at every time; no intercept or centering',
                  parameters='4 positive phase/neutral weights per common z, [0,1] ohm; atom .00015, weight .001 penalties',
                  inference='enumerate complete-tree switch, solve source-onehot MILP and R/X together in each tree',
                  fairness='same history+current observations, amplitude, candidate trees and parameter constraints',
                  loss='history-estimated phase/neutral I2R with normal-only fitted scalar calibration',
                  balanced_three_phase_stationary_source=True,known_q_over_p=.4,source_presence='conditional one source, including null stress test; not calibrated alarm',
                  seconds_per_milp=seconds,hashes={Path(path).name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for path in (__file__,phase.__file__,base.__file__)})
    path=output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol mismatch'
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for seed in range(seeds):
        for balanced in (True,False):
            prefix=f"seed{seed}_{'balanced' if balanced else 'unbalanced'}"
            history=phase.phase_sample(feeder,74000+seed,128,balanced)
            holdout=phase.phase_sample(feeder,78000+seed,96,balanced)
            trees,ranking=candidate_trees(history)
            hpath=output/(prefix+'_history.json')
            if hpath.exists():histfit=json.loads(hpath.read_text())
            else:
                histfit=solve_one(history,trees[0],None,seconds)
                histfit['candidates']=[dict(supports=[sorted(s) for s in tree],**rank) for tree,rank in zip(trees,ranking)]
                hpath.write_text(json.dumps(histfit,indent=2),encoding='utf-8')
            if 'weights' not in histfit:raise RuntimeError('No feasible historical parameter fit')
            raw_loss=phase.phase_loss(history,trees[0],histfit)
            balance=history['p0']-history['p'].sum(axis=1)
            scale=float(np.clip(raw_loss@balance/max(raw_loss@raw_loss,1e-12),.2,5))
            for condition in protocol['conditions']:
                target=output/(prefix+'_'+condition+'.json')
                if target.exists():continue
                current=phase.phase_sample(feeder,76000+seed,count,balanced,condition,source_bus)
                jointdata=phase.combine(history,current)
                loss_hat=scale*phase.phase_loss(current,trees[0],histfit)
                amp=np.maximum(current['p0']-current['p'].sum(axis=1)-loss_hat,0)
                amp_joint=np.r_[np.zeros(len(history['p'])),amp]
                treefits=[]
                for i,tree in enumerate(trees):
                    cp=output/(prefix+'_'+condition+f'_tree{i}.checkpoint.json')
                    if cp.exists():entry=json.loads(cp.read_text())
                    else:
                        entry=dict(tree_index=i,supports=[sorted(s) for s in tree],
                                   topology_only=solve_one(jointdata,tree,None,seconds),
                                   theft=solve_one(jointdata,tree,amp_joint,seconds))
                        cp.write_text(json.dumps(entry,indent=2),encoding='utf-8')
                    treefits.append(entry)
                i0=min(range(len(trees)),key=lambda i:objective(treefits[i]['topology_only'],trees[i]))
                i1=min(range(len(trees)),key=lambda i:objective(treefits[i]['theft'],trees[i]))
                fits={}
                for name,i,key in [('topology_only',i0,'topology_only'),('sequential',i0,'theft'),('joint',i1,'theft')]:
                    fitted=dict(treefits[i][key]);phase.evaluate(feeder,trees[i],fitted,holdout,source_bus)
                    fitted['tree_index']=i;fitted['supports']=[sorted(s) for s in trees[i]]
                    fitted['all_tree_subproblems_optimal']=all(v[key]['status']=='optimal' for v in treefits) if name!='sequential' else treefits[i][key]['status']=='optimal'
                    fits[name]=fitted
                result=dict(seed=seed,balanced=balanced,condition=condition,loss_scale=scale,
                            amplitude_mae_kw=float(np.mean(abs(amp-current['amplitude']))),
                            amplitude_mean_kw=float(amp.mean()),true_amplitude_mean_kw=float(current['amplitude'].mean()),
                            loss_estimate_mae_kw=float(np.mean(abs(loss_hat-current['loss']))),
                            ac_audit=current['audit'],fits=fits,tree_solve_statuses=[dict(tree=i,normal=v['topology_only']['status'],theft=v['theft']['status']) for i,v in enumerate(treefits)])
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(json.dumps(dict(case=prefix+'_'+condition,amp_mae=result['amplitude_mae_kw'],fits={k:dict(status=v['status'],all_optimal=v['all_tree_subproblems_optimal'],correct=v.get('source_region_correct'),source=v.get('source_support'),tree=v['tree_index'],r_error=v.get('r_relative_error'),phase_mae=v.get('heldout_phase_mae_v_approx')) for k,v in fits.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=2);parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--seconds',type=float,default=30);parser.add_argument('--source-bus',type=int,default=5)
    args=parser.parse_args();main(args.output,args.seeds,args.samples,args.seconds,args.source_bus)
