"""Paired independent working-distribution audit of already frozen policies.

No optimization. Reuses the exact prelocked qualification streams, avoiding a
comparison between an old 128-point probability and a new 4096-point one.
"""
import argparse,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import refine as r


def evaluate(root):
    root,p,cfg,alloc=r.config(root);f=r.read(root/'price_freeze.json');r.checked(f['protocol']);rows=[]
    for ci,c in enumerate(p['cases']):
        path=root/c['name']/'policy_probability.json'
        if path.exists():rows.extend(r.read(path)['rows']);continue
        packet=r.load(root/c['name']/'problem.pkl');dev=r.Devices(alloc[c['case']],cfg,r.profiles)
        plans=r.load(r.checked(f['cases'][c['name']]['plans']));N=p['qualification_samples']
        values={name:[] for name in plans};started=time.perf_counter()
        for batch in range((N+63)//64):
            count=min(64,N-batch*64)
            prob=r.Probability(packet,dev,count,p['qualification_seed']+ci*1000+batch,p['qualification_response_seed']+ci*1000+batch)
            cache={}
            for name,item in plans.items():
                key=item['mean_u'].tobytes()
                if key in cache:values[name].extend(cache[key]);continue
                U=dev.project(item['mean_u'][None]*prob.multiplier)
                a=prob.a0+np.concatenate([np.einsum('nr,srt->stn',packet['effect_'+k],U).reshape(count,-1) for k in ['C','F']],axis=1)
                vals=[r.conditional_probability(v,prob.b,prob.w,packet['threshold'],coefficient_gradients=False).probability for v in a]
                values[name].extend(vals);cache[key]=vals
        case=[]
        for name,v in values.items():
            v=np.array(v);difference=v-np.asarray(values['old_diagnostic'])
            row=dict(condition=c['name'],policy=name,samples=N,probability=float(v.mean()),standard_error=float(v.std(ddof=1)/np.sqrt(N)),
                paired_probability_difference_vs_old=float(difference.mean()),paired_difference_standard_error=float(difference.std(ddof=1)/np.sqrt(N)),
                cost=plans[name]['cost'],used_for_optimization=False)
            case.append(row)
        r.write(path,dict(rows=case,seconds=time.perf_counter()-started));rows.extend(case)
        print('PROBABILITY_AUDIT',c['name'],[(v['policy'],v['probability']) for v in case],flush=True)
    r.write(root/'policy_probability_summary.json',dict(rows=rows,scope='same frozen independent continuous working-distribution streams'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True,type=Path);args=a.parse_args();evaluate(args.output)
