"""One pre-AC, constraint-first numerical repair; parent solve is preserved."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,traceback,zipfile
from pathlib import Path
import numpy as np
STAGE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_spatial_correlated_stage')
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src' if (SOURCE/'src').is_dir() else SOURCE))
from runner import read,write,ref,sha,checked,loadp,now,SpatialProbability,cost,feasible
from constrained_refine import refine

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    parent=STAGE/'run_v1';p=read(parent/'protocol.json')
    if (parent/'observations_frozen.json').exists() or (parent/'actual').exists():
        raise RuntimeError('repair must precede all physical observations')
    p.update(id='spatial_correlated_DR_constraint_repair_v2',created_utc=now(),
        parent_protocol=ref(parent/'protocol.json'),parent_prices=ref(parent/'price_freeze.json'),
        refinement=dict(maxiter=12,seconds=60,target=.95,
            reason='finite penalty allowed cost reduction with probability shortfall; not physical infeasibility',
            rule='original128 probability; first reduce shortfall, then hard-linearized cost steps accepted only when original probability feasible',
            bounds='same domains, capacity, seeds, prices and six-policy actual plan; no future read',
            selection='cheapest original128-feasible returned candidate; otherwise largest original128 probability'))
    for file in Path(__file__).parent.glob('*.py'):p['source_sha256'][str(file.resolve())]=sha(file)
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,file in enumerate(p['source_sha256']):z.write(file,f'{i:03d}_{Path(file).name}')
    write(root/'problem_frozen.json',read(parent/'problem_frozen.json'))
    write(parent/'planning_superseded.json',dict(by=str(root),reason=p['refinement']['reason'],actual_observations=0))

def run(root):
    p=read(root/'protocol.json');parent=read(checked(p['parent_prices']))
    for file,digest in p['source_sha256'].items():
        if sha(file)!=digest:raise ValueError('source drift '+file)
    k=loadp(checked(read(root/'problem_frozen.json')['input']))
    if (root/'price_freeze.json').exists():raise RuntimeError('already frozen')
    problem=SpatialProbability(k,p['planning_samples'],p['base_seed'],p['response_seed'])
    write(root/'run_lock.json',dict(status='constraint_refinement',pid=os.getpid()))
    results=[]
    for domain in ['fixed','open']:
        name=domain+'_correlated';old=parent['policies'][name]
        u=np.asarray(old['mean_u']);a=np.asarray(old['capacity_kw']);caps=k['legacy_a'] if domain=='fixed' else k['capacity_caps']
        file=root/('refine_'+domain+'.json')
        if file.exists():r=read(file)
        else:
            try:
                r=refine(problem,caps,u,a,fixed_capacity=k['legacy_a'] if domain=='fixed' else None,**{key:p['refinement'][key] for key in ['target','maxiter','seconds']})
                r['domain']=domain
            except Exception as e:r=dict(domain=domain,status='failed',error=repr(e),traceback=traceback.format_exc(),best=None)
            write(file,r)
        results.append(r)
        print(domain,r['status'],None if r['best'] is None else {key:r['best'][key] for key in ['cost','planning_probability']},flush=True)
    def choose(domain):
        candidates=[]
        for r in results:
            if r['best'] is not None and (domain=='open' or r['domain']==domain):candidates.append(r['best'])
        for name in (['fixed_correlated','open_correlated'] if domain=='open' else ['fixed_correlated']):
            old=parent['policies'][name];candidates.append(dict(mean_u=old['mean_u'],capacity_kw=old['capacity_kw'],cost=old['cost'],planning_probability=problem.evaluate(np.asarray(old['mean_u']))['probability']))
        good=[r for r in candidates if r['planning_probability']>=.95-1e-7]
        return min(good,key=lambda r:r['cost']) if good else max(candidates,key=lambda r:(r['planning_probability'],-r['cost']))
    pol={name:parent['policies'][name].copy() for name in ['zero','legacy_iid_pair']}
    for domain in ['fixed','open']:
        r=choose(domain);pol[domain+'_correlated']=dict(r,role='constraint-first local refinement, '+domain)
    fee=pol['open_correlated']['cost']
    for name,a in [('rotation_uniform',np.full(k['shape'][0],2/k['shape'][0])),('rotation_selected_capacity',np.asarray(pol['open_correlated']['capacity_kw']))]:
        scale=np.sqrt(fee/(48*a.sum())) if fee else 0.
        if scale>1+1e-8:raise RuntimeError('same-fee rotation exceeds resource domain')
        u=scale*a[:,None]*k['rotation_sign'];pol[name]=dict(mean_u=u.tolist(),capacity_kw=a.tolist(),cost=cost(u,a),role='prespecified six-hour rotation; '+name)
    diagnostic=SpatialProbability(k,p['diagnostic_samples'],p['diagnostic_base_seed'],p['diagnostic_response_seed'])
    for name,v in pol.items():
        u=np.asarray(v['mean_u']);a=np.asarray(v['capacity_kw'])
        if not feasible(u,a,k['capacity_caps']):raise ValueError('resource violation')
        pr=problem.evaluate(u);qr=diagnostic.evaluate(u)
        v.update(planning_probability=pr['probability'],diagnostic_probability=qr['probability'],diagnostic_se=qr['conditional_mc_se'],planning_target_met=bool(pr['probability']>=.95-1e-7))
        print('freeze',name,v['cost'],pr['probability'],qr['probability'],flush=True)
    write(root/'planning_summary.json',dict(results=[{key:v for key,v in r.items() if key not in ['trace','best']} for r in results],
        policies={name:{key:v for key,v in r.items() if key!='mean_u'} for name,r in pol.items()},global_optimality_proven=False))
    write(root/'price_freeze.json',dict(status='frozen',created_utc=now(),policies=pol,problem=read(root/'problem_frozen.json')['input'],protocol=ref(root/'protocol.json'),future_read=False))
    write(root/'run_lock.json',dict(status='prices_frozen',pid=None))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run']);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();globals()[args.command](args.output)
