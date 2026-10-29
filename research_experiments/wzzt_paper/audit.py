from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha,matrices
from research_experiments.wzzt_retune.engine import compatible,arrays
from research_experiments.wzzt_refine.core import ridge_alpha
from research_experiments.wzzt_budget.audit import independent_lp
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from .common import observed


def physical_atoms(directory,labels):
    import networkx as nx
    directory=Path(directory);meta=read(directory/'metadata.json');edges=pd.read_csv(directory/'branches.csv');g=nx.Graph()
    for e in edges.to_dict('records'):
        if str(e['is_true_closed']).lower() not in ('true','1'):continue
        g.add_edge(int(e['from_bus']),int(e['to_bus']),r=float(e['r_ohm']),x=float(e['x_ohm']))
    root=int(meta['root_bus']);order=list(nx.bfs_tree(g,root));parent=dict(nx.bfs_predecessors(g,root));positions={v:i for i,v in enumerate(labels)}
    descendants={v:({positions[v]} if v in positions else set()) for v in order};atoms={};zbase=meta['base_kv']**2/meta['base_mva']
    for v in reversed(order[1:]):
        u=parent[v];support=tuple(sorted(descendants[v]));descendants[u]|=descendants[v]
        if support:
            old=atoms.setdefault(support,[0.,0.]);old[0]+=2*g[u][v]['r']/zbase;old[1]+=2*g[u][v]['x']/zbase
    family=sorted(atoms,key=lambda s:(len(s),s))
    return family,[atoms[s][0] for s in family],[atoms[s][1] for s in family]


def audit(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();p=read(output/'protocol.json');checks=[];LP=[];models=[];independent=[];events=[];truths={};atoms={};resolution=[]
    def check(name,passed):checks.append(dict(check=name,passed=bool(passed)))
    check('all frozen producer, input, configuration and physical bytes',all(sha(output/f)==h for f,h in read(output/'manifest.json').items()))
    check('no physical-generation failures',not read(output/'data_errors.json'))
    for d in (output/'data').rglob('diagnostics.json'):
        data=read(d)
        check(str(d.relative_to(output))+' physical days and independent seeds',not data['data_error'] and len(data['records'])==9 and len({r['seed'] for r in data['records']})==9)
        check(str(d.relative_to(output))+' all physical AC points',all(r['converged']==96 and r['voltage_min']>0 for r in data['records']))
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        for b in range(10):
            root=output/'data'/case/f'b{b}'/'reference';reference={}
            for sigma in (0.,.0002,.0005,.001):
                folder=root/f'v{round(sigma*10000):04d}'
                if not folder.exists():continue
                for split in ('train','validation','test'):
                    with np.load(folder/f'{split}_observed.npz') as d:a={k:d[k] for k in ('P','Q','V','root')}
                    with np.load(folder/f'{split}_truth.npz') as d:true={k:d[k] for k in a}
                    key=(split,'truth')
                    if key in reference:check(f'{case}/b{b}/{sigma}/{split} same physical path',all(np.array_equal(reference[key][k],true[k]) for k in a))
                    reference[key]=true
                    for k in ('P','Q'):
                        key=(split,k)
                        if key in reference:check(f'{case}/b{b}/{sigma}/{split} same PQ draws',np.array_equal(reference[key],a[k]))
                        reference[key]=a[k]
                    if sigma:
                        for k in ('V','root'):
                            z=(a[k]-true[k])/(sigma*np.maximum(abs(true[k]),1e-12));key=(split,k)
                            if key in reference:check(f'{case}/b{b}/{sigma}/{split} common normalized {k} draws',np.allclose(reference[key],z,atol=1e-8,rtol=1e-8))
                            reference[key]=z
    for j,job in enumerate(p['jobs']):
        folder=output/'jobs'/job;payload=read(folder/'payload.json')
        if not (folder/'result.json').exists():check(job+' returned',False);continue
        r=read(folder/'result.json');check(job+' planned models present',set(payload['methods'])<=set(r['models']))
        check(job+' no truth used by estimator',r['truth_used_by_estimator'] is False)
        for kind,rows in r['LP'].items():
            LP.append(dict(job_id=job,kind=kind,solves=len(rows),passed=all(x['status']==2 and max(x['max_constraint_violation'],x['max_bound_violation'])<=1e-7 and abs(x['physical']-x['objective'])<=1e-8 for x in rows)))
        base={tuple(s) for s in r['pool']}
        check(job+' augmentation retains original pool',base<={tuple(s) for s in r['augmentation']['pool']})
        check(job+' matched control original domain',base=={tuple(s) for s in r['domains']['restart_control']})
        train=observed(payload['train'],payload['N']);validation=observed(payload['validation'])
        if payload['truth_network'] not in truths:truths[payload['truth_network']]=physical_truth(Path(payload['truth_network']),r['labels'])['clades']
        if payload['truth_network'] not in atoms:
            atoms[payload['truth_network']]=physical_atoms(payload['truth_network'],r['labels'])
            S,Rv,Xv=atoms[payload['truth_network']];R,X=matrices(len(r['labels']),S,Rv,Xv);physical=physical_truth(Path(payload['truth_network']),r['labels'])
            check(payload['truth_network']+' independent physical atom response',np.allclose(R,physical['R'],atol=1e-12,rtol=1e-12) and np.allclose(X,physical['X'],atol=1e-12,rtol=1e-12))
        S,Rv,Xv=atoms[payload['truth_network']];cal=r['calibration']
        strength=np.asarray(Rv)*cal['a']+np.asarray(Xv)*cal['b'];nontrivial=np.array([2<=len(s)<len(r['labels']) for s in S]);kept=strength+1e-10>=.16*cal['median_root_depth']
        resolution.append(dict(job_id=job,panel=payload['panel'],case=payload['case'],condition=payload['condition'],N=payload['N'],voltage_sigma=payload['voltage_sigma'],
            true_clades=int(nontrivial.sum()),physical_coefficient_clades_dropped=int((nontrivial&~kept).sum()),
            physical_coefficient_clade_recall=float(kept[nontrivial].mean()),minimum_edge_to_depth=float((strength[nontrivial]/cal['median_root_depth']).min()),
            truth_used_only_posthoc=True,scope='filter diagnostic at physical coefficients with observed calibration; not an upper bound over refitted models'))
        positions={v:i for i,v in enumerate(r['labels'])}
        true_indices={tuple(sorted(positions[v] for v in s)) for s in truths[payload['truth_network']]}
        for name,trajectory in r['searches'].items():
            path=trajectory.get('path',[]);step=0
            for decision in trajectory.get('decisions',[]):
                if not path or step+1>=len(path):break
                if decision['gain']<=1e-10+.001*decision['current']:break
                current,endpoint=path[step],path[step+1];old={tuple(s) for s in current['supports']}
                new={tuple(s) for s in endpoint['supports'] if 2<=len(s)<len(r['labels'])}-old
                false=new-true_indices
                if false:
                    options=[c['objective'] for c in decision['candidates'] if tuple(c['candidate']) in true_indices]
                    min_true=min(options) if options else None
                    events.append(dict(job_id=job,panel=payload['panel'],search=name,iteration=decision['iteration'],
                        false_positive_additions=json.dumps(sorted(false)),new_false_count=len(false),
                        endpoint_loss=endpoint['objective'],best_tested_true_loss=min_true,true_candidates_tested=len(options),
                        full_sweep=decision['full_sweep'],loss_margin_vs_best_true=None if min_true is None else min_true-endpoint['objective'],
                        strictly_better_than_tested_true=bool(options and endpoint['objective']+1e-8<min_true),
                        truth_used_only_posthoc=True))
                step+=1
        points=list(r['models'].values())+r['base_points']+[p for g in r['candidates'].values() for p in g]
        for s in r['searches'].values():points+=s.get('path',[])
        seen=set();max_error=0.;legal=True
        for point in points:
            key=json.dumps({k:point[k] for k in ('supports','r','x')},sort_keys=True)
            if key in seen:continue
            seen.add(key);family=[tuple(s) for s in point['supports']]
            legal&=len(set(family))==len(family) and all((i,) in family for i in range(len(r['labels']))) and all(compatible(family[:i],s) for i,s in enumerate(family))
            legal&=all(-1e-9<=v<=2+1e-9 for v in point['r']+point['x'])
            for obs,name in ((train,'objective'),(validation,'validation_mae')):
                if name not in point:continue
                P,Q,Y=arrays(obs,'raw');pred=np.zeros_like(Y)
                for s,rv,xv in zip(point['supports'],point['r'],point['x'],strict=True):pred[:,s]+=(rv*P[:,s].sum(axis=1)+xv*Q[:,s].sum(axis=1))[:,None]
                max_error=max(max_error,abs(float(abs(pred-Y).mean())-point[name]))
        models.append(dict(job_id=job,unique_models=len(seen),max_score_error=max_error,passed=bool(legal and max_error<1e-8)))
        for front in r['augmentation']['fronts']:
            P,Q,_=arrays(train,front['preprocess']);alpha,scale=ridge_alpha(P,Q,front['strength'])
            check(job+f" ridge {front['preprocess']} {front['strength']}",abs(alpha-front['alpha'])<1e-12 and abs(scale-front['gram_scale'])<1e-12 and front['diagnostics']['success'])
        sample=payload['bundle']==0 and payload['voltage_sigma']==.001 and payload['N']==48 and (payload['panel'] in ('main','case33') or (payload['panel']=='topology' and payload['case'].startswith('random8_')))
        if sample:
            for method in ('multistart','ridge_union'):
                bank=r['base_points']+r['candidates'].get(method,[])
                point=bank[r['models'][method]['selection_index']]
                independent.append(dict(job_id=job,variant=method,**independent_lp(point,train)))
        if (j+1)%200==0:print('AUDITED',j+1,flush=True)
    frame=pd.read_csv(destination/'per_job.csv')
    check('all planned denominators retained',len(frame)==p['planned_model_rows'] and frame.job_id.nunique()==p['planned'] and not frame.duplicated(['job_id','variant']).any())
    summary=dict(all_passed=all(x['passed'] for group in (checks,LP,models,independent) for x in group),checks=len(checks),
        LP_solves=sum(x['solves'] for x in LP),physical_models=sum(x['unique_models'] for x in models),independent_LP_checks=len(independent),
        returned_inputs=len(models),planned_inputs=p['planned'],failed_checks=[x['check'] for x in checks if not x['passed']])
    write(destination/'audit_summary.json',summary);write(destination/'audit_checks.json',checks);write(destination/'independent_LP.json',independent)
    pd.DataFrame(LP).to_csv(destination/'LP_checks.csv',index=False);pd.DataFrame(models).to_csv(destination/'physical_checks.csv',index=False)
    pd.DataFrame(events).to_csv(destination/'greedy_events.csv',index=False)
    pd.DataFrame(resolution).to_csv(destination/'resolution_diagnostic.csv',index=False)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();audit(a.output,a.destination)
