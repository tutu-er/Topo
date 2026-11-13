"""Independent objective/constraint replay of saved optimization points."""
from pathlib import Path
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc


def original_gains(task,x):
    e=task['entry'];s=io.read(e['scores']['path']);H=np.array([p['H'] for p in s['parameters']],dtype=np.longdouble)
    with np.load(task['planner']['path'],allow_pickle=False) as packet:F=packet['forecast'].astype(np.longdouble)
    dp=np.repeat(x,4,axis=0)*np.array(e['scale']);U=F+np.c_[dp,.35*dp];values=[]
    for j in e['competitors']:
        D=H[e['target']]-H[j]
        values.append(float((np.sum((U@D.T)**2)-np.sum((F@D.T)**2))/(2*s['sigma2'])))
    return values


def ED_check(task,point,prefix,expected,evidence=False):
    x=point[prefix+'x'];g=point[prefix+'dg'];s=point[prefix+'shift'];l=point[prefix+'shed'];b=point[prefix+'buy'];v=point[prefix+'sell']
    e=task['entry'];scale=np.array(e['scale']);mw=scale*task['base_mw'];n=len(scale)
    with np.load(task['planner']['path'],allow_pickle=False) as p:F=p['forecast']
    balance=max(float(np.max(abs(s-l-g-np.repeat(x,4,axis=0)))),float(np.max(abs(b-v-(F[:,:n]*task['base_mw']+np.repeat(x,4,axis=0)*mw).sum(1)))))
    domain=max(0.,float(np.max(abs(x)))-task['alpha'],float(np.max(abs(g)))-.27,float(np.max(abs(s)))-.15,
        float(np.max(abs(np.diff(g,axis=0))))-.15,float(np.max(abs(g[0])))-.15,float(np.max(abs(l))),float(np.max(abs(s.sum(0)))))
    price=np.repeat([300.]*7+[500.]*10+[800.]*5+[500.]*2,4)
    fuel=500*np.array([.7,.9,1.1])[np.arange(n)%3]
    cost=.25*(np.sum(price*b-.9*price*v)+np.sum(mw*fuel*(.33+g)+(.2*500/(.6*mw))*mw*mw*g*g)+100*np.sum(abs(s)*mw)+10000*np.sum(l*mw))
    require(balance<=1e-7 and domain<=1e-7,'Independent ED domain replay failed')
    error=abs(float(cost)-expected);require(error<=1e-6*max(1.,expected),'Independent ED cost replay failed')
    violation=0.
    if evidence:
        gamma=2. if task['label']=='g2' else 4.;violation=max(0.,gamma-min(original_gains(task,x)))
        require(violation<=1e-6,'Independent original ED evidence failed')
    return balance,domain,error,violation


def run(longrun,polish,prices,ed_polish,out):
    require(not out.exists(),'Preserve earlier optimization audit');cfg=io.read(longrun/'protocol.json')
    max_gain=max_fee=max_ED=0.;count=0
    for item in cfg['selection']:
        d=longrun/'jobs'/item['key'];results=io.read(d/'results.json');gamma=2. if item['label']=='g2' else 4.
        with np.load(d/'controls.npz',allow_pickle=False) as p:
            for r in results:
                x=p[f"seconds{r['requested_seconds']:g}"];violation=max(0.,gamma-min(original_gains(item,x)))
                max_gain=max(max_gain,violation);fee=float(np.mean(x*x));max_fee=max(max_fee,abs(fee-r['cost']))
                require(violation<1e-6 and np.max(abs(x))<=item['alpha']+1e-8,'Long solve original constraints failed')
                require(r['lower_bound']<=fee+1e-8*max(fee,1e-20),'Long solve lower bound above feasible fee');count+=1
        d=polish/'jobs'/item['key'];r=io.read(d/'result.json')
        with np.load(d/'point.npz',allow_pickle=False) as p:
            x=p['x'];v=max(0.,gamma-min(original_gains(item,x)));max_gain=max(max_gain,v)
            require(v<1e-6 and np.max(abs(x))<=item['alpha']+1e-8,'Polished original constraints failed')
            require(abs(float(np.mean(x*x))-r['cost'])<1e-12 and not r['lower_bound_from_QP_used'],'Polish objective/bound scope failed');count+=1
    pc=io.read(prices/'protocol.json');lookup={(t['id'],t['scenario'],t['phase'],t['day']):t for t in pc['tasks']}
    joint=io.read(prices/'joint_results.json');joint_lookup={(r['id'],r['scenario']):r for r in joint};EDpoints=0
    for r in joint:
        task=lookup[r['id'],r['scenario'],r['phase'],r['day']]
        with np.load(r['private_point']['path'],allow_pickle=False) as p:
            if r['free_ED']['feasible']:
                values=ED_check(task,p,'free_',r['free_ED']['cost_yuan']);max_ED=max(max_ED,max(values));EDpoints+=1
            if r['feasible']:
                values=ED_check(task,p,'joint_',r['cost_yuan'],True);max_ED=max(max_ED,max(values));EDpoints+=1
                require(r['lower_bound_yuan']<=r['cost_yuan']+1e-6,'Joint cost bound failed')
    for r in io.read(ed_polish/'summary.json'):
        task=lookup[r['id'],r['scenario'],'confirmation',0];old=joint_lookup[r['id'],r['scenario']]
        with np.load(r['private_point']['path'],allow_pickle=False) as p:
            values=ED_check(task,p,'',r['new_cost_yuan'],True);max_ED=max(max_ED,max(values));EDpoints+=1
        require(not r['inner_QP_bound_used'] and r['new_cost_yuan']<=r['old_cost_yuan']+1e-6,'ED inner bound/monotonicity failed')
        require(abs(r['premium_lower_bound_yuan']-max(0.,old['lower_bound_yuan']-old['free_ED']['cost_yuan']))<1e-8,'Original ED bound changed')
    fixed=io.read(prices/'fixed_plan_results.json')
    for r in fixed:
        if r['feasible']:
            cost=sum(v for k,v in r['components'].items() if k.endswith('_yuan'))
            require(abs(cost-r['cost_yuan'])<1e-8,'Fixed ED component sum differs')
    result=dict(status='passed',created_utc=utc(),nominal_points_replayed=count,ED_points_replayed=EDpoints,fixed_checks=len(fixed),
        max_original_gain_violation=max_gain,max_normalized_fee_error=max_fee,max_ED_domain_or_cost_error=max_ED,
        new_independent_days=0,inner_QP_bounds_used=False,source=io.ref(__file__))
    io.write(out,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('longrun','polish','prices','ed-polish','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.longrun.resolve(),a.polish.resolve(),a.prices.resolve(),a.ed_polish.resolve(),a.output.resolve())
