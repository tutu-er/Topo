"""Score saved AC observations with the physical MSE gate; no new AC or prices.

The existing run.py AC summary contains its OLD standardized-square event.
This entry computes the new physical-MSE event, with tau explicitly supplied.
"""
import argparse
from pathlib import Path
import numpy as np
import miqp_mse as mse


def score(case,actual,out,tau=.05):
    case,cfg,p,a,_=mse.base.load(case)
    actual=Path(actual);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=mse.base.read(actual/'summary.json')
    ledger=mse.base.read(actual/'scored_rows.json');rows=[]
    rt=(a['history_training_drop']-a['history_training_power']@a['H0'].T)*1e-6
    K=float(np.mean(np.abs(rt))/np.mean(rt*rt))
    historical=[(a['history_drop']-a['history_power']@a[k].T)*1e-6 for k in ('H0','H1')]
    for r in ledger:
        row=dict(path=r['path'],policy=r['policy'],status=r['status'],fee=r['fee'])
        if r['status']=='complete':
            z=mse.base.arrays(mse.base.checked(r['observation']))
            residuals=[(z['drop_micro']-z['power']@a[k].T)*1e-6 for k in ('H0','H1')]
            means=[float((np.sum(h*h)+np.sum(n*n))/(h.size+n.size)) for h,n in zip(historical,residuals)]
            gate=mse.pure_mse_gate(*means,K=K,tau=tau)
            combined=r['selection']['combined_losses'];best=min(combined.values())
            eligible=[k for k in p.candidate_ids if combined[k]<=best+1e-6]
            strict=min(eligible,key=lambda k:(p.support_counts[p.candidate_ids.index(k)],combined[k]))
            distance=float(np.mean((z['power']@(a['H1']-a['H0']).T*1e-6)**2))
            row.update(M0=means[0],M1=means[1],relative_gain=(means[0]-means[1])/means[0],
                       gate=gate,errors=r['errors'],strict_same_pair_errors=original['candidate_errors'][strict]['errors'],
                       prediction_distance_mse=distance)
        rows.append(row)
    tables={}
    for name in mse.base.POLICIES:
        group=[r for r in rows if r['policy']==name and r['status']=='complete']
        tables[name]=dict(completed=len(group),gate_successes=sum(r['gate']['accepted'] for r in group),
            mean_relative_gain=float(np.mean([r['relative_gain'] for r in group])) if group else None,
            complete_recoveries=sum(r['errors']==0 for r in group),total_errors=sum(r['errors'] for r in group),
            strict_same_pair_recoveries=sum(r['strict_same_pair_errors']==0 for r in group),
            mean_prediction_distance_mse=float(np.mean([r['prediction_distance_mse'] for r in group])) if group else None)
    summary=dict(status='complete',tau=tau,K=K,rows=tables,original_final_rule_unchanged=True,
                 metric='physical unweighted MSE relative gain; not legacy standardized square',
                 strict_scope='same frozen .16 pair only, not the complete neighbouring MIQP/.24')
    mse.base.write(out/'rows.json',rows);mse.base.write(out/'summary.json',summary)
    return summary


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--actual',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--tau',type=float,default=.05)
    args=parser.parse_args();print(json.dumps(score(args.case,args.actual,args.out,args.tau)))
