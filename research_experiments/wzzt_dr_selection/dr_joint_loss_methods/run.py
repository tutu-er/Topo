"""Bounded frozen-price method comparison. Never reads actual outcomes in run."""
from __future__ import annotations
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import argparse,json,pickle,hashlib,sys,time,platform,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy
from kernel import gaussian_packet,hermite_moments,spherical_probability,holder_tail,nominal_milp,abs_mean

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
BASE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_probability_refinement_stage/run_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):return pickle.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan_hash(case,u):
    """Normalize signed zeros; +0 and -0 are the same physical plan."""
    u=np.asarray(u,dtype=np.float64)
    return hashlib.sha256(case.encode()+np.where(u==0.,0.,u).tobytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise RuntimeError('hash changed: '+str(p))
    return p
def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    p=read(BASE/'protocol.json');freeze=read(BASE/'price_freeze.json');checked(freeze['protocol'])
    assert read(BASE/'run_lock.json')['status']=='complete'
    sources=list(Path(__file__).parent.glob('*.py'))
    sources+=[REPO/'research_experiments/wzzt_dr_selection'/v for v in ['dr_heterogeneous_pilot/devices.py','dr_heterogeneous_resources/build_allocations.py','dr_socp_loss_screen/kernel.py']]
    inputs=[ref(BASE/'protocol.json'),ref(BASE/'price_freeze.json'),ref(BASE/'run_lock.json'),ref(checked(p['library'])),ref(checked(p['allocations']))]
    for c in p['cases']:
        for name in ['problem.pkl','pool.json','plans.pkl']:inputs.append(ref(BASE/c['name']/name))
        checked(freeze['cases'][c['name']]['plans'])
    protocol=dict(id='joint_loss_methods_v1',created=datetime.now(timezone.utc).isoformat(),cases=p['cases'],
        base=str(BASE),policies=['zero','rotation','old_diagnostic','candidate'],expected=12,
        hermite_orders=[32,64,128],sphere_power=11,milp_seconds=15,
        sources=[ref(v) for v in sources],inputs=inputs,seed=None,random_draws=0,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        scope='Affine Gaussian before existing execution controller. No changed quote execution; no AC/new fits/search.',
        tails='Paired mgf + Holder keeps arbitrary cross-pair dependence. No independence-product certificate.',
        sphere='Exact radial pieces, deterministic Sobol angular points and convergence comparison; not certified 95%.',
        milp='Only the frozen plan [0,1] scaling ray; exact all-breakpoint reference. Not full price optimization.',
        failures_before_protocol=['unit MILP tiny strict-gap row tolerance detected; loss row rescaled before main batch'],
        tests_before_protocol=dict(passed=12,failed=0),
        gate='Do not deploy raw-Gaussian plans or claim current projected response probability. No automatic new AC.')
    write(out/'protocol.json',protocol)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,r in enumerate(protocol['sources']):z.write(r['path'],f'{i:03d}_{Path(r["path"]).name}')
    write(out/'run_lock.json',dict(status='prepared',pid=None))

def run(out):
    p=read(out/'protocol.json')
    for r in p['sources']+p['inputs']:checked(r)
    lock=read(out/'run_lock.json')
    if lock['status']=='complete':print('complete: nothing rerun');return
    if lock.get('pid') and lock['pid']!=os.getpid():raise RuntimeError('inspect existing process first')
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot'))
    from devices import Devices
    from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
    from research_experiments.wzzt_dr_selection.dr_socp_loss_screen.kernel import residual_variances,variance_of_sum,ar_cov
    old=read(BASE/'protocol.json');cfg=read(checked(old['library']));alloc={c['case']:c for c in read(checked(old['allocations']))['cases']}
    write(out/'run_lock.json',dict(status='running',pid=os.getpid()))
    rows=[];started=time.perf_counter();cache={}
    for c in p['cases']:
        folder=BASE/c['name'];packet=load(folder/'problem.pkl');plans=load(folder/'plans.pkl')
        dev=Devices(alloc[c['case']],cfg,profiles)
        for policy in p['policies']:
            key=c['name']+'_'+policy;receipt=out/'receipts'/(key+'.json')
            if receipt.exists():
                row=read(receipt);rows.append(row)
                if row['status']=='complete':cache[row['plan_hash']]=row
                continue
            u=np.asarray(plans[policy]['mean_u']);ph=plan_hash(c['name'],u)
            row=dict(case=c['name'],policy=policy,plan_hash=ph,contract_fee=plans[policy]['cost'],threshold=packet['threshold'],status='running')
            begun=time.perf_counter()
            try:
                if ph in cache:
                    row.update({k:v for k,v in cache[ph].items() if k not in ['case','policy','seconds']})
                    row['case']=c['name'];row['policy']=policy;row['reused_plan']=True
                else:
                    assert dev.audit(u)['passed']
                    mu,cov,w,mu0,delta,C,F=gaussian_packet(packet,dev,u,cfg)
                    n=len(dev.nodes);T=96
                    marginal={k:residual_variances(packet['effect_'+k],u*dev.relative_sd,packet['L_'+k],dev.node,cfg['response']['main_variance_weights']).ravel() for k in ['C','F']}
                    errors=[np.max(np.abs(np.diag(cov)[C]-marginal['C'])),np.max(np.abs(np.diag(cov)[F]-marginal['F']))]
                    for j in range(3):
                        linear=np.sin(np.arange(len(mu))*(j+1)*.071)
                        f=linear.reshape(T,2*n)
                        bg=f[:,:n].ravel()@packet['L_C']+f[:,n:].ravel()@packet['L_F']
                        coeff=(f[:,:n]@packet['effect_C']+f[:,n:]@packet['effect_F']).T*u*dev.relative_sd
                        oracle=float(bg@bg)+variance_of_sum(coeff,dev.node,cfg['response']['main_variance_weights'],ar_cov(96,cfg['response']['main_ar1_phi']))
                        errors.append(abs(float(linear@(cov@linear))-oracle)/max(1.,abs(oracle)))
                    if max(errors[:2])>1e-6 or max(errors[2:])>1e-10:raise ArithmeticError('covariance independent checks')
                    row['covariance_audit']=errors
                    parts=out/'parts'/key
                    mp=parts/'moments.json'
                    if mp.exists():mom=read(mp)
                    else:
                        mom=hermite_moments(mu,cov,w,tuple(p['hermite_orders']));write(mp,mom)
                    row['moments']=mom
                    print(key,'moments',round(mom['mean'],4),flush=True)
                    last=mom['rows'][-1];sdlo=np.sqrt(last['variance_lower']);sdhi=np.sqrt(last['variance_upper'])
                    row['cantelli_lower']=float(mom['mean']-np.sqrt(19)*sdhi)
                    row['cantelli_pass']=row['cantelli_lower']>packet['threshold']
                    row['gaussian_approx_q05_range']=[float(mom['mean']-1.6448536269514722*sdhi),float(mom['mean']-1.6448536269514722*sdlo)]
                    hp=parts/'holder.json'
                    if hp.exists():tail=read(hp)
                    else:tail=holder_tail(mu,cov,C,F,packet['threshold']);write(hp,tail)
                    row['holder']=tail
                    print(key,'holder',round(tail['lower_success'],5),flush=True)
                    sp=parts/'sphere.json'
                    if sp.exists():sphere=read(sp)
                    else:sphere=spherical_probability(mu,cov,w,packet['threshold'],p['sphere_power']);write(sp,sphere)
                    row['sphere']=sphere
                    print(key,'sphere',round(sphere['probability_estimate'],5),flush=True)
                    ip=parts/'milp.json'
                    if ip.exists():ipoint=read(ip)
                    else:ipoint=nominal_milp(mu0,delta,w,packet['threshold'],p['milp_seconds']);write(ip,ipoint)
                    row['nominal_milp']=ipoint
                    if ipoint['alpha'] is not None:
                        row['nominal_ray_fee']=float(dev.fees(ipoint['alpha']*u))
                        row['nominal_ray_physical_audit']=dev.audit(ipoint['alpha']*u)
                    row['mean_reference_score']=float(w@np.abs(mu))
                    row['status']='complete';row['reused_plan']=False;cache[ph]=row.copy()
            except Exception as e:
                row.update(status='failed',error=repr(e),traceback=traceback.format_exc())
            row['seconds']=time.perf_counter()-begun
            write(receipt,row);rows.append(row);write(out/'checkpoint.json',dict(records=len(rows),last=key))
            print(key,row['status'],round(row['seconds'],2),flush=True)
    summary=dict(status='complete' if len(rows)==12 and all(r['status']=='complete' for r in rows) else 'failed_or_incomplete',
        records=len(rows),complete=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='failed' for r in rows),
        unique_plans=len(set(r['plan_hash'] for r in rows)),seconds=time.perf_counter()-started,
        random_draws=0,new_AC=0,new_parameter_fits=0,new_W_search=0,rows=rows)
    write(out/'summary.json',summary);write(out/'run_lock.json',dict(status=summary['status'],pid=None))
    write(out/'ended.json',dict(summary=ref(out/'summary.json')))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run']);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();globals()[a.phase](a.output)
