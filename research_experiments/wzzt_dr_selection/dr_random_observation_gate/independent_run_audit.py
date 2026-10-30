"""Read-only independent audit. No experiment selection/fit helpers imported."""
import argparse,hashlib,json,pickle
from collections import Counter
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(f):return sorted(set(tuple(sorted(s)) for s in f),key=lambda s:(len(s),s))
def check(ok,message):
    if not ok:raise AssertionError(message)
def raw_arrays(scenes):
    p=np.concatenate([s['P_terminal'].to_numpy() for s in scenes]);q=np.concatenate([s['Q_terminal'].to_numpy() for s in scenes])
    y=np.concatenate([s['V_root_observed'].to_numpy()[:,None]**2-s['V_terminal'].to_numpy()**2 for s in scenes])
    return p,q,y
def kernel(point,n):
    out=[]
    for name in ('r','x'):
        A=np.zeros((n,n))
        for s,w in zip(point['supports'],point[name],strict=True):A[np.ix_(s,s)]+=w
        out.append(A)
    return out
def l1(arrays,point):
    p,q,y=arrays;R,X=kernel(point,p.shape[1]);return float(np.abs(p@R.T+q@X.T-y).mean())

def audit(generation,execution,out):
    generation=Path(generation);execution=Path(execution);cfg=read(execution/'protocol.json');summ=read(execution/'summary.json')
    checked_sources=0
    for c in (read(generation/'generation_protocol.json'),cfg):
        for group in ('source_sha256','input_sha256'):
            for p,h in c.get(group,{}).items():check(sha(p)==h,'source/input drift: '+p);checked_sources+=1
    histories={h['id']:pickle.loads(Path(h['path']).read_bytes()) for h in cfg['histories']}
    lp=[];path_reason=Counter();method_status=Counter();discover=Counter();all_path_scans=True;counts=Counter();diff=[];cal=[]
    for p in (execution/'history').glob('*.json'):lp+=read(p).get('LP_solves',[])
    snapshot_diagnostics=[]
    for snap in cfg['snapshots']:
        result=read(execution/'records'/(snap['id']+'.json'));current=read(result['current_state_path']);lp+=current['LP_solves']
        future=pickle.loads(Path(snap['path']).read_bytes());history=histories[snap['history_id']]
        train=raw_arrays(history['training']+future['training']);valid=raw_arrays(history['validation']+future['validation'])
        check(len(train[0])==480 and len(valid[0])==480,'all historical rows must be retained')
        check(result['applied_calibration']==current['calibration'],'shared current calibration')
        calibration=current['calibration'];discover[result['discovery']['status']]+=1
        snapshot_diagnostics.append(dict(id=snap['id'],new_supports=result['new_supports'],
            current_pool=result['current_pool'],candidate_supports=result['discovery']['candidate_supports'],
            discovery_status=result['discovery']['status']))
        for method,m in result['methods'].items():
            method_status[m['status']]+=1;lp+=m.get('LP_solves',[])
            if m['status']!='complete':continue
            counts['selected_outputs']+=1;selected=m['selected'];raw=selected['selected_uncontracted'];points=m['points']
            scores=[l1(valid,p) for p in points];best=min(scores)
            eligible=[i for i,v in enumerate(scores) if v<=1.03*best+1e-12]
            i=min(eligible,key=lambda j:(len(points[j]['supports']),scores[j],j))
            check(i==selected['selection_index'],'independent validation choice '+method)
            check(canon(points[i]['supports'])==canon(raw['supports']),'selected raw identity '+method)
            keep=[j for j,s in enumerate(raw['supports']) if len(s)==1 or calibration['a']*raw['r'][j]+calibration['b']*raw['x'][j]+1e-10>=.16*calibration['median_root_depth']]
            check(selected['supports']==[raw['supports'][j] for j in keep],'independent contract '+method)
            check(selected['r']==[raw['r'][j] for j in keep] and selected['x']==[raw['x'][j] for j in keep],'contract weights '+method)
            for part in ('raw','contract'):
                point=raw if part=='raw' else selected
                for arr,key in ((train,'objective'),(valid,'validation_mae')):
                    error=abs(l1(arr,point)-point[key]);diff.append(error);check(error<=1e-10,'L1 replay '+method)
            for pname,path in m.get('paths',{}).items():
                path_reason[path['reason']]+=1
                accepted=[]
                for d in path['decisions']:
                    counts['decisions']+=1;counts['candidate_evaluations']+=len(d['candidates'])
                    all_path_scans=all_path_scans and d['full_sweep']
                    if d['gain']>1e-10+.001*d['current']:
                        accepted.append(d);counts['accepted_steps']+=1
                        counts['accepted_cross_steps']+=int(any(c['removed'] and abs((d['current']-c['objective'])-d['gain'])<=1e-12 for c in d['candidates']))
                check(len(accepted)==len(path['path'])-1,'actual accept path length')
                check(path['reason']=='pool_local_optimum','unfinished scan')
            cal.append((snap['id'],method,calibration))
    check(len(lp)==summ['real_LP_calls'],'LP denominator')
    check(all(v['status']==2 for v in lp),'LP nonoptimal')
    result=dict(status='passed',source_and_input_hash_checks=checked_sources,
        snapshots=len(cfg['snapshots']),requested_evaluator_records=51,method_statuses=dict(method_status),
        LP_calls=len(lp),LP_statuses=dict(Counter(v['status'] for v in lp)),
        max_constraint_violation=max(v.get('max_constraint_violation',0) for v in lp),
        path_reasons=dict(path_reason),all_candidate_scans_complete=all_path_scans,
        counts=dict(counts),maximum_independent_L1_difference=max(diff,default=0),
        discovery_statuses=dict(discover),snapshots_with_new_supports=sum(bool(s['new_supports']) for s in snapshot_diagnostics),
        snapshot_diagnostics=snapshot_diagnostics,new_solver_calls=0,new_data_generated=0,
        independent_of_experiment_select_measure_and_matrix_helpers=True)
    Path(out).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='snapshot_diagnostics'},ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);p.add_argument('--execution',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.generation,a.execution,a.output)
