"""Independent arithmetic replay; no fitting, AC rerun or price adjustment."""
import argparse,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import refine as r
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target


def audit(root):
    root,p,cfg,alloc=r.config(root);freeze=r.read(root/'price_freeze.json');r.checked(freeze['protocol'])
    ledger=r.read(root/'observations_frozen.json');scored={v['record_id']:v for v in r.read(root/'scored_private.json')['rows']}
    checks=0;fail=[];hashes=set();maxes={'power':0.,'state':0.,'terminal':0.,'ev_soc':0.,'increment':0.}
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not ok:fail.append(label)
    for c in p['cases']:
        d=r.Devices(alloc[c['case']],cfg,r.profiles);plans=r.load(r.checked(freeze['cases'][c['name']]['plans']))
        pool=r.read(root/c['name']/'pool.json');models=pool['models'];order=pool['selection_order']
        true={tuple(sorted(b)) for b in r.read(root/c['name']/'truth_private.json')['missed_blocks']}
        plan={}
        for name,item in plans.items():
            u=item['mean_u'];head=np.where(u>=0,d.delta_upper_kw,-d.delta_lower_kw)
            coeff=np.divide(.25*d.cost_full_activation,head,out=np.zeros_like(head),where=head>0)
            check(abs(float((coeff*u*u).sum())-item['cost'])<1e-7,c['name']+name+'cost')
            plan[name]=item
        check(abs(plan['rotation']['cost']-plan['candidate']['cost'])<1e-7,c['name']+'matched_rotation')
        for old in ledger['rows']:
            if old['condition']!=c['name']:continue
            label=old['record_id'];saved=scored[label]
            if old['status']!='complete':check(saved['status']!='complete',label+'unresolved');continue
            pack=r.load(r.checked(old['packet']));v=pack['private'];u=v['delivered'];e=np.zeros(d.R)
            check(np.allclose(v['nominal'],plan[old['policy']]['mean_u'],atol=0,rtol=0),label+'frozen')
            path=p['paths'][old['day']];ci=p['cases'].index(c)
            z=d.draw(path['response_seed']+ci*100,1,stress=old['law']=='stress')[0]
            check(np.array_equal(z,v['multiplier']),label+'random_stream')
            power=float(max(np.max(-d.delta_lower_kw+0)*0,np.max(d.delta_lower_kw-u),np.max(u-d.delta_upper_kw),0.))
            maxes['power']=max(maxes['power'],power);check(power<1e-7,label+'power')
            state_violation=0.;terminal=0.
            for t in range(96):
                e=np.where(d.reset[:,t],0.,e)
                e+=.25*(d.eta*np.maximum(u[:,t],0)-np.maximum(-u[:,t],0)/d.eta)
                state_violation=max(state_violation,float(np.max(np.maximum(d.energy_low-e,0))),float(np.max(np.maximum(e-d.energy_high,0))))
                end=(d.kind!=1)&((t==95)|((d.kind==2)&(t==31)))
                terminal=max(terminal,float(np.max(np.abs(e[end]),initial=0)))
            maxes['state']=max(maxes['state'],state_violation);maxes['terminal']=max(maxes['terminal'],terminal)
            check(state_violation<1e-7,label+'state');check(terminal<1e-7,label+'terminal')
            for j in np.flatnonzero(d.kind==2):
                a=d.cap[j];capacity=50*a/7;task=20*a/7
                for start,end,soc0 in [(0,32,.4+.9*(6/14)*task/capacity),(72,96,.4)]:
                    soc=soc0+.9*.25*np.cumsum(d.baseline_kw[j,start:end]+u[j,start:end])/capacity
                    violation=float(max(0.,.2-soc.min(),soc.max()-.9));maxes['ev_soc']=max(maxes['ev_soc'],violation)
                    check(violation<1e-8,label+f'ev{j}_{start}')
            dp=np.zeros((96,len(d.nodes)));dq=dp.copy()
            for j,node in enumerate(d.node):dp[:,node]+=u[j];dq[:,node]+=d.qratio[j]*u[j]
            ep=(v['physical_p']-v['natural_p']).to_numpy()*pool['base_kw']-dp
            eq=(v['physical_q']-v['natural_q']).to_numpy()*pool['base_kw']-dq
            error=float(max(np.max(np.abs(ep)),np.max(np.abs(eq))));maxes['increment']=max(maxes['increment'],error)
            check(error<1e-8,label+'one_increment')
            pp,qq=_power(pack['scene'],pool['labels']);yy=_target(pack['scene'],pp.shape)
            hashes.add(hashlib.sha256(pp.tobytes()+qq.tobytes()+yy.tobytes()).hexdigest())
            losses={m['candidate_id']:float(np.abs(yy-pp@np.asarray(m['R']).T-qq@np.asarray(m['X']).T).mean()) for m in models}
            H={m['candidate_id']:m['historical_validation_mae'] for m in models}
            scope=[k for k in order if H[k]<=1.03*min(H.values())+1e-12]
            S={k:.6*H[k]+.4*losses[k] for k in order};ref=min(scope,key=lambda k:(S[k],order.index(k)))
            eligible=[k for k in scope if S[k]<=1.03*S[ref]+1e-12]
            by={m['candidate_id']:m for m in models};chosen=min(eligible,key=lambda k:(len(by[k]['point']['supports']),S[k],order.index(k)))
            check(chosen==saved['selected'],label+'selection')
            D=1e6*(losses['coarse']-1.03*losses['refined']);h=1e6*(1.5*(1.03*H['refined']-H['coarse'])+2.5e-12)
            check(abs(D-saved['D_micro'])<1e-7,label+'D')
            check((D>h)==saved['coarse_excluded'] if len(scope)==2 else True,label+'exclusion_event')
            final={tuple(sorted(b)) for b in by[chosen]['frozen_output_support_labels']}
            raw={tuple(sorted(pool['labels'][i] for i in b)) for b in by[chosen]['point']['supports']}
            check(len(final^true)==saved['errors'],label+'errors');check(len(raw^true)==saved['raw_errors'],label+'raw_errors')
    summary=r.read(root/'summary.json')
    check(len(ledger['rows'])==p['expected_records'],'denominator')
    for tab in summary['tables']:
        rr=[v for v in scored.values() if v['condition']==tab['condition'] and v['law']==tab['law'] and v['policy']==tab['policy'] and v['status']=='complete']
        check(len(rr)==tab['complete'],'complete_count');check(sum(v['errors'] for v in rr)==tab['errors'],'total_errors')
        check(sum(v['errors']==0 for v in rr)==tab['recoveries'],'recovery_count')
    result=dict(checks=checks,failures=fail,passed=not fail,records=len(ledger['rows']),distinct_observations=len(hashes),
        independent_backgrounds=len(p['paths'])*len(p['cases']),max_violation=maxes,
        scope='independent loss/selector/error/cost/state/EV-SOC/CRN/increment replay; no statistical calibration claim')
    r.write(root/'audit.json',result)
    print(result,flush=True)
    if fail:raise AssertionError(fail[:10])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True,type=Path);args=a.parse_args();audit(args.output)
