"""Bounded, zero-draw diagnostic of existing frozen prices. No AC or fitting."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import argparse,json,pickle,hashlib,platform,sys,time,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy
from kernel import fixed_quote_metrics,device_screen


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):return pickle.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('input hash changed: '+str(p))
    return p
def now():return datetime.now(timezone.utc).isoformat()


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    base=args.base;previous=read(base/'protocol.json');freeze=read(base/'price_freeze.json')
    checked(freeze['protocol'])
    if read(base/'run_lock.json')['status']!='complete':raise RuntimeError('previous batch not complete')
    inputs=[ref(base/'protocol.json'),ref(base/'price_freeze.json'),ref(base/'run_lock.json')]
    for key in ['library','allocations']:inputs.append(ref(checked(previous[key])))
    for c in previous['cases']:
        folder=base/c['name'];checked(freeze['cases'][c['name']]['plans'])
        for r in read(folder/'inputs.json').values():checked(r)
        for fn in ['problem.pkl','pool.json','plans.pkl']:inputs.append(ref(folder/fn))
    sources=list(Path(__file__).parent.glob('*.py'))+[args.repo/'research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot/devices.py',args.repo/'research_experiments/wzzt_dr_selection/dr_heterogeneous_resources/build_allocations.py']
    protocol=dict(id='socp_loss_screen_v1',created_utc=now(),base=str(base),repo=str(args.repo),
        cases=[c['name'] for c in previous['cases']],policies=['zero','rotation','old_diagnostic','candidate'],
        signs=['historical','plan_mean'],allocations=['uniform','optimal_fixed_quote'],
        expected_policy_records=12,expected_bound_records=48,alpha0=.04,flip_risk=.01,delta_micro=1e-6,
        seed=None,random_draws=0,new_AC=0,new_price_search=0,new_parameter_fits=0,new_W_search=0,
        distinction='Affine pre-controller law only. Original causal delivered distribution is not Gaussian.',
        gate='Proceed to price optimization only if a bound passes and the affine response has a credible physical execution model. Otherwise report this bound as unsuitable on tested quotes; no global infeasibility claim.',
        risk_diagnostic='Optimize only row risk allocation at each fixed quote by analytic convex KKT. Fixed alpha0 and total budget. Does not implement joint variable-risk price SOCP.',
        private_inputs=inputs,source=[ref(f) for f in sources],
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        privacy='Do not read actual observations, truth, or scores. Publish scalar receipts only.',
        limit='One bounded fixed-price batch; deterministic audits allowed, no price or AC expansion on failure.')
    write(out/'protocol.json',protocol)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,r in enumerate(protocol['source']):z.write(r['path'],f'{i:03d}_{Path(r["path"]).name}')
    write(out/'run_lock.json',dict(status='prepared',pid=None))


def context(out):
    p=read(out/'protocol.json')
    for r in p['private_inputs']+p['source']:checked(r)
    repo=Path(p['repo']);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot'))
    from devices import Devices
    from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
    previous=read(Path(p['base'])/'protocol.json');cfg=read(checked(previous['library']))
    alloc={r['case']:r for r in read(checked(previous['allocations']))['cases']}
    return p,previous,cfg,alloc,Devices,profiles


def run(args):
    out=args.output;p,previous,cfg,alloc,Devices,profiles=context(out)
    lock=read(out/'run_lock.json')
    if lock['status']=='complete':print('complete checkpoint retained');return
    if lock.get('pid') and lock['pid']!=os.getpid():raise RuntimeError('existing running lock requires inspection')
    write(out/'run_lock.json',dict(status='running',pid=os.getpid()))
    rows=[];start=time.perf_counter()
    for c in previous['cases']:
        folder=Path(p['base'])/c['name'];packet=load(folder/'problem.pkl');plans=load(folder/'plans.pkl')
        dev=Devices(alloc[c['case']],cfg,profiles)
        for name in p['policies']:
            path=out/'receipts'/f'{c["name"]}_{name}.json'
            if path.exists():rows.append(read(path));continue
            t=time.perf_counter();row=dict(condition=c['name'],policy=name)
            try:
                u=np.asarray(plans[name]['mean_u'],float)
                audit=dev.audit(u)
                if not audit['passed']:raise ArithmeticError('frozen nominal plan failed original constraints')
                fee=dev.fees(u)
                if not np.isclose(fee,plans[name]['cost'],rtol=1e-10,atol=1e-9):raise ArithmeticError('frozen price differs')
                bounds=[fixed_quote_metrics(packet,dev,u,cfg,s,a) for s in p['signs'] for a in p['allocations']]
                physical=device_screen(dev,u,cfg)
                private=physical.pop('private_terminals');write(out/'private_terminals'/path.name,private)
                row.update(status='complete',contract_fee=fee,nominal_absolute_kwh=audit['absolute_DR_kwh'],bounds=bounds,
                    affine_device_diagnostic=physical,original_nominal_feasible=True,seconds=time.perf_counter()-t)
            except Exception as e:row.update(status='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-t)
            write(path,row);rows.append(row)
            write(out/'checkpoint.json',dict(completed=len(rows),expected=p['expected_policy_records'],last=str(path)))
            print(c['name'],name,row['status'],flush=True)
    bounds=[b for r in rows if r['status']=='complete' for b in r['bounds']]
    summary=dict(status='complete' if len(rows)==12 and all(r['status']=='complete' for r in rows) else 'failed_or_incomplete',
        policy_records=len(rows),successful=sum(r['status']=='complete' for r in rows),failed=sum(r['status']!='complete' for r in rows),
        bound_records=len(bounds),passing_bounds=sum(b['certificate_pass'] for b in bounds),
        random_draws=0,new_price_optimization=0,new_AC=0,new_fits=0,seconds=time.perf_counter()-start,rows=rows)
    write(out/'summary.json',summary);write(out/'ended.json',dict(status=summary['status'],summary=ref(out/'summary.json')))
    write(out/'run_lock.json',dict(status=summary['status'],pid=None))


def audit(args):
    out=args.output;p,previous,cfg,alloc,Devices,profiles=context(out);summary=read(out/'summary.json')
    checks=0;failures=[]
    def check(flag,label):
        nonlocal checks
        checks+=1
        if not flag:failures.append(label)
    check(summary['policy_records']==12,'policy denominator');check(summary['bound_records']==48,'bound denominator')
    for row in summary['rows']:
        check(row['status']=='complete',row['condition']+' status')
        if row['status']!='complete':continue
        pool=read(Path(p['base'])/row['condition']/'pool.json')
        hist={m['candidate_id']:m['historical_validation_mae']*1e6 for m in pool['models']}
        h=1.5*(1.03*hist['refined']-hist['coarse'])+2.5e-6
        by={(r['signs'],r['allocation']):r for r in row['bounds']}
        for b in row['bounds']:
            name=row['condition']+'/'+row['policy']+'/'+b['signs']+'/'+b['allocation']
            check(abs(b['threshold_micro']-h)<1e-9,name+' threshold')
            lhs=b['mean_G_micro']-b['main_penalty_micro']-b['flip_penalty_micro']-h-p['delta_micro']
            check(abs(lhs-b['certificate_margin_micro'])<1e-9,name+' full inequality')
            check(b['certificate_pass']==(lhs>=0),name+' pass label')
            check(b['used_flip_risk']<=.01+1e-12,name+' risk sum')
            check(b['row_risk_max_excess']<1e-11,name+' exact Gaussian tail')
            check(abs(b['exact_affine_mean_D_micro']-(b['mean_G_micro']+b['omitted_C_expected_flip_micro']-b['F_expected_flip_micro']))<1e-8,name+' expectation identity')
        for s in p['signs']:
            check(by[s,'optimal_fixed_quote']['flip_penalty_micro']<=by[s,'uniform']['flip_penalty_micro']+1e-8,row['condition']+' optimized row allocation '+s)
        vals=[b['exact_affine_mean_D_micro'] for b in row['bounds']]
        check(max(vals)-min(vals)<1e-8,row['condition']+' mean independent of auxiliary signs')
    result=dict(checks=checks,failures=failures,passed=not failures,scope='deterministic arithmetic replay; unit tests separately check independent covariance and integrals')
    write(out/'audit.json',result);print(json.dumps(result,ensure_ascii=False))
    if failures:raise AssertionError(failures)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','run','audit']);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--repo',type=Path);a.add_argument('--base',type=Path);args=a.parse_args()
    globals()[args.phase](args)
