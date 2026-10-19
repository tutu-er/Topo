"""Full raw-meter AC EIV GN with shared lines and time-local states."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .model import evaluate,scales,line_map,project
from .response import solve_quadratic


def bounds(payload):
    obs=payload['observed'];sig=scales(payload);cap=payload['domain']['power_cap'];count=len(obs['p'])
    lower=np.c_[(-cap-obs['p'])/sig['p'],(-cap-obs['q'])/sig['q'],np.full(count,(payload['domain']['root_min']-1)/.01)]
    upper=np.c_[(cap-obs['p'])/sig['p'],(cap-obs['q'])/sig['q'],np.full(count,(payload['domain']['root_max']-1)/.01)]
    return lower,upper


def legal(value,payload):
    d=payload['domain'];v=abs(value['physical']['voltage'])
    return bool(v.min()>=d['v_min'] and v.max()<=d['v_max'] and
        np.all(np.hypot(value['p'],value['q'])<=d['power_cap']+1e-10) and
        value['root_v'].min()>=d['root_min']-1e-10 and value['root_v'].max()<=d['root_max']+1e-10)


def gradients(value,theta,local,lower,upper):
    a,c,res=value['shared'],value['state'],value['residual']
    ga=np.einsum('tri,tr->i',a,res);gc=np.einsum('tri,tr->ti',c,res)
    pg=max(float(np.max(abs(project(theta-ga)-theta))) if len(ga) else 0.,
           float(np.max(abs(np.clip(local-gc,lower,upper)-local))))
    return ga,gc,pg


def step(value,theta,local,lower,upper,damping):
    a,c,res=value['shared'],value['state'],value['residual']
    gc=np.einsum('tri,tr->ti',c,res)
    active=((local<=lower+1e-10)&(gc>0))|((local>=upper-1e-10)&(gc<0))
    c=c*(~active)[:,None,:]
    aa=np.einsum('tri,trj->ij',a,a)+damping*np.eye(a.shape[2]);ga=np.einsum('tri,tr->i',a,res)
    cc=np.einsum('tri,trj->tij',c,c)+damping*np.eye(c.shape[2])
    ca=np.einsum('tri,trj->tij',c,a);gc=np.einsum('tri,tr->ti',c,res)
    solved=np.linalg.solve(cc,np.concatenate((ca,gc[:,:,None]),axis=2))
    za,zr=solved[:,:,:a.shape[2]],solved[:,:,a.shape[2]]
    if a.shape[2]:
        reduced=aa-np.einsum('tli,tlj->ij',ca,za);g=ga-np.einsum('tli,tl->i',ca,zr)
        reduced=(reduced+reduced.T)/2
        point,certificate=solve_quadratic(reduced,g-reduced@theta,
                              .5*theta@reduced@theta-g@theta,theta,certify=True)
        da=point-theta
    else:da=np.empty(0);certificate={}
    dc=-zr-np.einsum('tli,i->tl',za,da)
    return da,dc,certificate


def information(value,transform):
    a,c=value['shared'],value['state'];q,_=np.linalg.qr(c,mode='reduced')
    effective=a-q@(q.transpose(0,2,1)@a)
    _,s,vt=np.linalg.svd(effective.reshape(-1,a.shape[2]),full_matrices=False)
    good=s>max(1e-9,s[0]*1e-7);cov=(vt[good].T/s[good]**2)@vt[good]
    return dict(rank=int(good.sum()),width=len(s),singular_values=s,
        physical_standard_error=np.sqrt(np.maximum(np.diag(transform@cov@transform.T),0)),
        null_physical_loading=np.linalg.norm(transform@vt[~good].T,axis=1),
        weakest_direction=vt[-1],interpretation='local sensitivity reference; boundaries and null directions preclude automatic confidence intervals')


def fit(candidate,payload,initial,*,seed=0,multistart=4,max_iter=200,gtol=1e-5,fixed_lines=None):
    started=perf_counter();n=candidate.n;count=len(payload['observed']['p']);lower,upper=bounds(payload)
    local0=np.zeros((count,2*n+1));transform,_=line_map(candidate,payload)
    if payload['root_meter'] is not None:local0[:,-1]=(np.asarray(payload['root_meter'][0])-1)/.01
    else:
        r,x=np.split(transform@initial,2) if fixed_lines is None else fixed_lines
        obs=payload['observed'];rr=(candidate.M*r)@candidate.M.T;xx=(candidate.M*x)@candidate.M.T
        b2=np.mean(obs['v']**2+2*(obs['p']@rr.T+obs['q']@xx.T),axis=1)
        local0[:,-1]=(np.sqrt(np.maximum(b2,.5))-1)/.01
    local0=np.clip(local0,lower,upper);rng=np.random.default_rng(seed)
    if fixed_lines is None:
        starts=[project(initial)]+[project(initial+rng.normal(0,.08,len(initial))) for _ in range(multistart-1)]
    else:starts=[np.asarray(initial)]
    records=[];attempts=[];calls=0
    for number,start in enumerate(starts):
        theta=start.copy();local=local0.copy();damping=1e-4;trace=[];reason='iteration_limit'
        try:
            for iteration in range(max_iter):
                value=evaluate(candidate,payload,theta,local,fixed_lines=fixed_lines);calls+=1
                loss=float(np.sum(value['residual']**2));_,_,pg=gradients(value,theta,local,lower,upper)
                if not trace:trace.append(loss)
                if pg<=gtol:reason='projected_gradient';break
                moved=False
                for trial_number in range(14):
                    da,dc,cert=step(value,theta,local,lower,upper,damping)
                    newtheta=project(theta+da) if fixed_lines is None else theta
                    newlocal=np.clip(local+dc,lower,upper)
                    try:
                        trial=evaluate(candidate,payload,newtheta,newlocal,derivatives=False,fixed_lines=fixed_lines);calls+=1
                        trial_loss=float(np.sum(trial['residual']**2));ok=legal(trial,payload)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError):trial_loss=np.inf;ok=False
                    if ok and trial_loss<loss:
                        theta,local=newtheta,newlocal;trace.append(trial_loss)
                        damping=max(damping/3,1e-10);moved=True;break
                    damping*=10
                if not moved:reason='no_decreasing_step';break
            value=evaluate(candidate,payload,theta,local,fixed_lines=fixed_lines);calls+=1
            _,_,pg=gradients(value,theta,local,lower,upper);physical=legal(value,payload)
            record=dict(start=number,loss=float(np.sum(value['residual']**2)),projected_gradient=pg,
                qualified=bool(physical and pg<=1e-4),physical=physical,iterations=iteration+1,
                reason=reason,trace=trace,theta=theta)
            attempts.append(record);records.append((record,theta.copy(),local.copy(),value))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(start=number,qualified=False,error=repr(exc),trace=trace))
    if not records:return dict(status='failed',qualified=False,attempts=attempts,seconds=perf_counter()-started,ac_calls=calls)
    record,theta,local,value=min(records,key=lambda z:(not z[0]['physical'],z[0]['loss']))
    info=information(value,transform) if fixed_lines is None else None
    return dict(record,status='qualified' if record['qualified'] else 'unqualified',J=record['loss']/2,
        r=value['r'],x=value['x'],p=value['p'],q=value['q'],root_v=value['root_v'],v=value['physical']['v'],
        residual=value['residual'],local=local,information=info,attempts=attempts,
        seconds=perf_counter()-started,ac_calls=calls,lines_updated=fixed_lines is None,
        ac_equation_residual=value['physical']['ac_equation_residual'],
        power_balance_residual=value['physical']['power_balance_residual'],
        truth_used=False,global_optimality_certified=False)
