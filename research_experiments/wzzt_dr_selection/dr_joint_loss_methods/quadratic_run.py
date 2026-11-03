"""One prespecified six-record quadratic-approximation follow-up."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import sys,time,traceback,json
from pathlib import Path
import numpy as np
from run import BASE,REPO,read,write,ref,checked,load
from kernel import gaussian_packet
from quadratic import evaluate
def main(out):
    old=read(BASE/'protocol.json');prior=out.parent/'run_v1'
    if not out.exists():
        out.mkdir()
        sources=[ref(Path(__file__).parent/x) for x in ['quadratic.py','quadratic_run.py','test_quadratic.py','kernel.py','run.py']]
        inputs=[ref(prior/'summary.json'),ref(prior/'protocol.json')]
        write(out/'protocol.json',dict(id='joint_loss_quadratic_v1',sources=sources,inputs=inputs,
            cases=[c['name'] for c in old['cases']],policies=['zero','candidate'],expected=6,
            purpose='Second-Hermite projection, deterministic CF inversion, and variance-remainder bound. No new quote, AC, fit or draws.',
            cdf_tolerance=1e-7,cdf_cutoff_max=4096,remainder_risks=[.01,.025,.04],tests_passed=6))
    protocol=read(out/'protocol.json')
    for r in protocol['sources']+protocol['inputs']:checked(r)
    if (out/'run_lock.json').exists():
        lk=read(out/'run_lock.json')
        if lk['status']=='complete':return
        if lk.get('pid') and lk['pid']!=os.getpid():raise RuntimeError('inspect existing lock')
    write(out/'run_lock.json',dict(status='running',pid=os.getpid()))
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot'))
    from devices import Devices
    from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
    cfg=read(checked(old['library']));alloc={c['case']:c for c in read(checked(old['allocations']))['cases']}
    rows=[]
    for c in old['cases']:
        dev=Devices(alloc[c['case']],cfg,profiles)
        packet=load(BASE/c['name']/'problem.pkl');plans=load(BASE/c['name']/'plans.pkl')
        for policy in protocol['policies']:
            key=c['name']+'_'+policy;receipt=out/'receipts'/(key+'.json')
            if receipt.exists():rows.append(read(receipt));continue
            begin=time.perf_counter();row=dict(case=c['name'],policy=policy)
            try:
                u=plans[policy]['mean_u']
                if c['name'].startswith('pengwah') and policy=='candidate':
                    q=rows[-1]['result'];row.update(result=q,status='complete',reused_zero_plan=True)
                else:
                    mu,cov,w,*_=gaussian_packet(packet,dev,u,cfg)
                    first=read(prior/'receipts'/(key+'.json'))
                    q=evaluate(mu,cov,w,packet['threshold'],first['moments']['rows'][-1]['variance_upper'])
                    row.update(result=q,status='complete',reused_zero_plan=False)
            except Exception as e:row.update(status='failed',error=repr(e),traceback=traceback.format_exc())
            row['seconds']=time.perf_counter()-begin
            write(receipt,row);rows.append(row)
            write(out/'checkpoint.json',dict(records=len(rows),last=key))
            print(key,row['status'],q.get('approximate_probability',{}).get('p') if row['status']=='complete' else row['error'],flush=True)
    write(out/'summary.json',dict(records=len(rows),complete=sum(r['status']=='complete' for r in rows),rows=rows))
    write(out/'run_lock.json',dict(status='complete',pid=None))
if __name__=='__main__':main(Path(sys.argv[1]))
