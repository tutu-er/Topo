"""Small AC experiment with path-shared, parameter-dependent loss envelopes.

Research only. Fixed candidate bank, known source junction, known X/R=.5.
EV is P-only; background Q=.3 P_background. No production WZZT changes.
The LP gives exclusion lower bounds, never AC feasibility certificates.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
from math import nextafter, inf
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog, least_squares

import design_v7 as v7
import local_shared_source_v6 as v6

BOX = v7.BOX
KAPPA = F(1, 2)
VMIN = F(9, 10)
LOSS_FACTOR = (1 + KAPPA ** 2) / 4  # theta=2r, x=kappa*r
ACTION_NAMES = tuple([f'site{s}_{p}' for s in (1,2,3) for p in ('constant','endpoint')]+['mobile12','mobile13','mobile23'])


def f(v):
    return F(float(v))


def sqrt_up(x):
    if not x:
        return F()
    with localcontext() as ctx:
        ctx.prec = 80
        y = float((Decimal(x.numerator) / Decimal(x.denominator)).sqrt())
    while f(y) ** 2 < x:
        y = nextafter(y, inf)
    return f(y)


def geometry(candidate):
    """Edges: trunk, shared cherry, terminal1, terminal2, terminal3.

    B rows are non-root nodes [junction, cherry-junction, leaf1..3].
    B maps branch drops to bus drops; B.T maps bus currents to branches.
    """
    cherry = v6.CANDIDATES[candidate]
    b = np.zeros((5, 5), dtype=int)
    b[:, 0] = 1
    b[1, 1] = 1
    for j in range(3):
        b[j + 2, j + 2] = 1
        b[j + 2, 1] = int(j in cherry)
    descendants = {e: tuple(g for g in range(5) if g != e and b[g, e]) for e in range(5)}
    return b, b[2:], descendants


def ac_forward(candidate, theta, p, q, source_p, source_q):
    """Complex constant-power AC fixed point; residual is checked explicitly."""
    b, a, descendants = geometry(candidate)
    z = (1 + .5j) * np.asarray(theta) / 2
    zbus = (b * z) @ b.T
    loads = np.r_[complex(source_p, source_q), 0j, np.asarray(p) + 1j*np.asarray(q)]
    voltage = np.ones(5, complex)
    for iteration in range(1000):
        new = 1 - zbus @ np.conj(loads / voltage)
        if np.max(np.abs(new - voltage)) < 2e-14:
            voltage = new
            break
        voltage = new
    else:
        raise RuntimeError("AC fixed point did not converge")
    current = b.T @ np.conj(loads / voltage)
    residual = float(np.max(np.abs(voltage - 1 + zbus @ np.conj(loads / voltage))))
    ell = abs(current) ** 2
    eta = np.array([abs(z[e])**2*ell[e] + 2*sum(
        (z[e].real*z[g].real + z[e].imag*z[g].imag)*ell[g]
        for g in descendants[e]) for e in range(5)])
    y = 1 - abs(voltage[2:])**2
    peff = np.asarray(p) + .5*np.asarray(q)
    linear = a @ (np.asarray(theta)*(a.T @ peff)) + theta[0]*(source_p+.5*source_q)
    identity_error = float(np.max(abs(y - linear - a@eta)))
    if residual > 1e-11 or identity_error > 1e-11:
        raise RuntimeError("AC equation/DistFlow identity mismatch")
    return dict(Y=y, voltage=voltage, ell=ell, eta=eta,
                equation_residual=residual, identity_error=identity_error)


def ac_batch(candidate, theta, p, q, source_p):
    """Vectorized same-network AC predictions for design fitting only."""
    b, _, _ = geometry(candidate)
    zbus = (b*((1+.5j)*np.asarray(theta)/2))@b.T
    source_p = np.broadcast_to(source_p,len(p))
    loads = np.c_[source_p*(1+.3j),np.zeros(len(p)),np.asarray(p)+1j*np.asarray(q)]
    voltage = np.ones_like(loads,dtype=complex)
    for _ in range(1000):
        new = 1-np.conj(loads/voltage)@zbus.T
        if np.max(abs(new-voltage))<2e-14:
            voltage=new
            break
        voltage=new
    else:
        raise RuntimeError('batched AC did not converge')
    residual=float(np.max(abs(voltage-1+np.conj(loads/voltage)@zbus.T)))
    if residual>1e-11:
        raise RuntimeError('batched AC residual exceeds tolerance')
    return 1-abs(voltage[:,2:])**2,voltage,residual


def envelope(observed, candidate, box=BOX):
    """Exact rational two affine upper planes for each branch loss remainder."""
    _, a, descendants = geometry(candidate)
    p, q = np.asarray(observed['P']), np.asarray(observed['Q'])
    sp = np.broadcast_to(observed['source_p_abs_cap'], len(p))
    sq = np.broadcast_to(observed['source_q_abs_cap'], len(p))
    bp = np.broadcast_to(observed.get('P_error', 0.), p.shape)
    bq = np.broadcast_to(observed.get('Q_error', 0.), q.shape)
    bounds = [(f(lo), f(hi)) for lo, hi in box]
    planes, caps = [], []
    for t in range(len(p)):
        s = [sqrt_up((abs(f(p[t,j]))+f(bp[t,j]))**2 +
                     (abs(f(q[t,j]))+f(bq[t,j]))**2) for j in range(3)]
        source = sqrt_up(f(sp[t])**2 + f(sq[t])**2)
        ell = [((sum((s[j] for j in range(3) if a[j,e]), F()) +
                 (source if e == 0 else F()))/VMIN)**2 for e in range(5)]
        caps.append(ell)
        edge_planes = []
        for e in range(5):
            both = []
            for orientation in (0, 1):
                co, const = [F()]*5, F()
                lo, hi = bounds[e]
                co[e] += ell[e]*(lo+hi)
                const -= ell[e]*lo*hi
                for g in descendants[e]:
                    lg, ug = bounds[g]
                    if orientation == 0:
                        co[e] += 2*ell[g]*ug
                        co[g] += 2*ell[g]*lo
                        const -= 2*ell[g]*lo*ug
                    else:
                        co[e] += 2*ell[g]*lg
                        co[g] += 2*ell[g]*hi
                        const -= 2*ell[g]*hi*lg
                both.append((tuple(LOSS_FACTOR*v for v in co), LOSS_FACTOR*const))
            edge_planes.append(both)
        planes.append(edge_planes)
    return planes, caps


def solve_outer(observed, candidate, *, box=BOX, sharing='path', linear_only=False):
    """Certified minimum extra L-infinity slack for an AC outer LP.

    sharing='output' relaxes the same branch envelopes to independent output
    intervals. This is a same-data, same-parameter-domain ablation.
    P/Q input errors enter both the linear response and the current cap.
    """
    began = perf_counter()
    if sharing not in ('path', 'output'):
        raise ValueError(sharing)
    _, a, _ = geometry(candidate)
    planes, caps = envelope(observed, candidate, box)
    p, q, y = (np.asarray(observed[k]) for k in ('P', 'Q', 'Y_drop'))
    count = len(p)
    peff = p+.5*q
    eps = np.broadcast_to(observed['drop_error_bound'], y.shape)
    bp = np.broadcast_to(observed.get('P_error', 0.), p.shape)
    bq = np.broadcast_to(observed.get('Q_error', 0.), q.shape)
    nominal = np.broadcast_to(observed['source_nominal_eff'], count)
    budget = np.broadcast_to(observed['source_error_eff'], count)
    nk = 5 if sharing == 'path' else 3
    nvar = 5 + count + count*nk + 1
    lower = list(map(lambda v:f(v[0]), box)) + [-f(v)*f(box[0][1]) for v in budget]
    upper = list(map(lambda v:f(v[1]), box)) + [f(v)*f(box[0][1]) for v in budget]
    eta_caps = []
    for t in range(count):
        ec = [min(sum((co[k]*f(box[k][1]) for k in range(5)), const)
                  for co,const in planes[t][e]) for e in range(5)]
        eta_caps.extend(ec if nk == 5 else [sum((ec[e] for e in range(5) if a[i,e]), F()) for i in range(3)])
    if linear_only:
        eta_caps = [F()]*len(eta_caps)
    lower += [F()]*len(eta_caps) + [F()]
    upper += eta_caps + [F(1)]
    rows, rhs, exactrows, exactrhs, guards = [], [], [], [], []
    def add(co, value):
        dense = np.zeros(nvar)
        for k,v in co.items():
            dense[k] = float(v)
        guard = sum((abs(f(dense[k])-v)*max(abs(lower[k]),abs(upper[k])) for k,v in co.items()), F())
        rows.append(dense)
        rhs.append(v6._up(value+guard))
        exactrows.append(co)
        exactrhs.append(value)
        guards.append(guard)
    for t in range(count):
        wcol = 5+t
        for sign in (1, -1):
            add({0:-f(budget[t]), wcol:F(sign)}, F())
        for e in range(nk):
            for orientation in (0,1):
                included = [e] if nk == 5 else [g for g in range(5) if a[e,g]]
                co = [sum((planes[t][g][orientation][0][k] for g in included), F()) for k in range(5)]
                const = sum((planes[t][g][orientation][1] for g in included), F())
                add({**{k:-v for k,v in enumerate(co) if v}, 5+count+t*nk+e:F(1)}, const)
        for i in range(3):
            pred = {e:sum((f(p[t,j])+KAPPA*f(q[t,j]) for j in range(3) if a[j,e]), F()) for e in range(5) if a[i,e]}
            pred[0] += f(nominal[t])
            pred[wcol] = F(1)
            for e in range(nk):
                if (nk == 5 and a[i,e]) or (nk == 3 and i == e):
                    pred[5+count+t*nk+e] = F(1)
            eiv = {e:sum((f(bp[t,j])+KAPPA*f(bq[t,j]) for j in range(3) if a[j,e]), F()) for e in range(5) if a[i,e]}
            for sign in (1,-1):
                co = {k:sign*v for k,v in pred.items()}
                for k,v in eiv.items():
                    co[k] = co.get(k,F())-v
                co[nvar-1] = F(-1)
                add(co, sign*f(y[t,i])+f(eps[t,i]))
    matrix = np.array(rows)
    objective = np.r_[np.zeros(nvar-1),1.]
    lo = np.array([v6._down(v) for v in lower])
    hi = np.array([v6._up(v) for v in upper])
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=list(zip(lo,hi)), method='highs',
                     options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    common = dict(candidate=candidate, sharing=sharing, linear_only=linear_only,
                  solver_status=int(result.status), solver_message=result.message,
                  seconds=perf_counter()-began, max_rounding_guard=float(max(guards,default=F())))
    if not result.success:
        return dict(common,status='solver_failure_unresolved',lower=None,upper=None,gap=None)
    # Exact box-corrected weak-duality certificate on the stored outer LP.
    lam = [f(min(0.,v)) for v in result.ineqlin.marginals]
    reduced = [F()]*nvar
    reduced[-1] = F(1)
    bound = sum((f(b)*v for b,v in zip(rhs,lam)), F())
    for row, v in zip(rows,lam):
        if v:
            for k in np.flatnonzero(row):
                reduced[k] -= f(row[k])*v
    bound += sum((min(v*f(l),v*f(u)) for v,l,u in zip(reduced,lo,hi)), F())
    bound = max(F(),bound)
    # Repair an exact feasible point for the ORIGINAL rational outer model.
    point = [min(u,max(l,f(v))) for v,l,u in zip(result.x,lower,upper)]
    for t in range(count):
        wc = 5+t
        limit = f(budget[t])*point[0]
        point[wc] = min(limit,max(-limit,point[wc]))
        for e in range(nk):
            included = [e] if nk == 5 else [g for g in range(5) if a[e,g]]
            limits = [sum((sum((co[k]*point[k] for k in range(5)),c)
                           for g in included for co,c in [planes[t][g][orientation]]),F()) for orientation in (0,1)]
            index = 5+count+t*nk+e
            point[index] = max(F(),min(point[index],*limits))
    point[-1] = F()
    for row,b in zip(exactrows,exactrhs):
        violation = sum((v*point[k] for k,v in row.items()),F())-b
        if row.get(nvar-1) == -1:
            point[-1] += max(F(),violation)
        elif violation > 0:
            raise RuntimeError('non-observation repair failed')
    ub = point[-1]
    if ub > 1 or bound > ub:
        raise RuntimeError('invalid bound interval or active slack cap')
    assert all(sum((v*point[k] for k,v in row.items()),F()) <= b for row,b in zip(exactrows,exactrhs))
    return dict(common,status='excluded' if bound>0 else 'unresolved',lower=v6._down(bound),
                upper=v6._up(ub),gap=v6._up(ub-bound),theta=list(map(float,point[:5])),
                exact_lower=str(bound),exact_upper=str(ub),exact_primal_verified=True,
                exact_box_dual_verified=True,outer_feasibility_is_not_AC_feasibility=True,
                max_current_cap=float(max(max(v) for v in caps)))


def action_plan(site,policy):
    connection=np.full(24,site,int)
    if policy.startswith('mobile'):
        first,second=[int(k)-1 for k in policy[-2:]]
        connection[:11]=first
        connection[11:13]=-1
        connection[13:]=second
        ev=np.zeros(24); ev[1:10]=.3; ev[14:23]=.3
    else:
        ev=np.zeros(24) if policy=='none' else np.full(24,.225) if policy=='constant' else np.tile([0.,.3,.3,.3],6)
    increments=np.zeros((24,3))
    for t,j in enumerate(connection):
        if j>=0:
            increments[t,j]=ev[t]
    return increments,ev,connection


def make_data(candidate, theta, seed, history_kind, site=0, policy='none', epsilon=1e-5):
    rng = np.random.default_rng(seed)
    ph = np.full((12,3),.2)
    if history_kind == 'rich':
        ph += np.tile([[.06,0,0],[0,.06,0],[0,0,.06],[0,0,0]],(3,1))
    elif history_kind != 'flat':
        raise ValueError(history_kind)
    background = np.r_[ph,np.full((24,3),.2)]
    p = background.copy()
    increments,ev,connection=action_plan(site,policy)
    p[12:] += increments
    q = .3*background  # EV has unity power factor, no Q command.
    delta = rng.uniform(-.02,.02,36)
    source = .12+delta
    noise = rng.uniform(-.8*epsilon,.8*epsilon,(36,3))
    answers = [ac_forward(candidate,theta,p[t],q[t],source[t],.3*source[t]) for t in range(36)]
    ytrue = np.array([v['Y'] for v in answers])
    y = ytrue+noise
    obs = dict(P=p,Q=q,Y_drop=y,drop_error_bound=np.full_like(y,epsilon),
               source_nominal_eff=np.full(36,.138),source_error_eff=np.full(36,.115),
               source_p_abs_cap=np.full(36,.22),source_q_abs_cap=np.full(36,.066),
               batch_id=np.r_[np.zeros(12,int),np.ones(24,int)])
    volts = np.array([abs(v['voltage']) for v in answers])
    travel=np.zeros(24)
    if policy.startswith('mobile'):
        travel[11:13]=1.  # declared 2 kWh trip, consumed during two no-injection slots
    soc = .4 + np.r_[0.,np.cumsum(ev*20*.25*.95-travel)]/60.
    truth = dict(topology=candidate,theta=list(theta),seed=seed,history=history_kind,
                 source_p=source.tolist(),Y_true=ytrue.tolist(),actual_noise=noise.tolist(),
                 eta=np.array([v['eta'] for v in answers]).tolist(),
                 min_voltage=float(volts.min()),max_voltage=float(volts.max()),
                 max_ac_residual=max(v['equation_residual'] for v in answers),
                 max_distflow_error=max(v['identity_error'] for v in answers),
                 energy_kwh=float(ev.sum()*20*.25),EV_P_kw=(ev*20).tolist(),
                 EV_bus_index_by_slot=connection.tolist(),travel_kwh=float(travel.sum()),
                 battery_kwh=60.,soc_trajectory=soc.tolist(),travel_by_slot_kwh=travel.tolist(),
                 soc_min=float(soc.min()),soc_max=float(soc.max()),
                 checks=dict(voltage=bool(volts.min()>=.9 and volts.max()<=1.1),
                             source=bool(abs(delta).max()<=.1),noise=bool(abs(noise).max()<=epsilon),
                             soc=bool(soc.min()>=.2 and soc.max()<=.9)))
    return obs,truth


def sliced(obs, rows):
    return {k:np.asarray(v)[rows] for k,v in obs.items()}


def fit_ac_reference(history,candidate,box=BOX):
    """Multistart nonlinear least squares used ONLY for design representatives."""
    count = len(history['P'])
    bounds = np.asarray(box)
    lo = np.r_[bounds[:,0],np.full(count,.02)]
    hi = np.r_[bounds[:,1],np.full(count,.22)]
    outer = solve_outer(history,candidate,box=box)
    def residual(x):
        try:
            prediction,_,_=ac_batch(candidate,x[:5],history['P'],history['Q'],x[5:])
            return ((prediction-history['Y_drop'])/1e-3).ravel()
        except RuntimeError:
            return np.full(count*3,1e3)
    fits = []
    starts = [(lo+hi)/2]
    if outer.get('theta') is not None:
        starts.append(np.r_[outer['theta'],np.full(count,.12)])
    for start in starts:
        result = least_squares(residual,np.clip(start,lo+1e-10,hi-1e-10),bounds=(lo,hi),
                               ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=160)
        fits.append((float(np.dot(result.fun,result.fun)),result))
    _, result = min(fits,key=lambda v:v[0])
    prediction,voltage,ac_residual=ac_batch(candidate,result.x[:5],history['P'],history['Q'],result.x[5:])
    feasible=bool(np.all(abs(prediction-history['Y_drop'])+1e-11<=history['drop_error_bound']) and
                  np.min(abs(voltage))>=.9 and np.max(abs(voltage))<=1.1)
    return dict(candidate=candidate,theta=result.x[:5].tolist(),source_p=result.x[5:].tolist(),
                max_abs_residual=float(abs(result.fun).max()*1e-3),solver_success=bool(result.success),
                historical_numeric_witness=feasible,ac_equation_residual=ac_residual,
                witness_scope='numerical AC residual below1e-11, not interval existence proof',
                interpretation='local AC design representative; never an exclusion lower bound')


def design_ac(history,retained,box=BOX):
    """AC representatives -> predicted outputs -> rival historical joint refit.

    Optimizes verified outer-LP separation at nominal future source. This is a
    scenario design score, not uniform AC experiment success or a probability.
    """
    all_refs = {h:fit_ac_reference(history,h,box) for h in retained}
    refs = {h:r for h,r in all_refs.items() if r['historical_numeric_witness']}
    if not refs:
        return dict(status='compute_before_dispatch',references=all_refs,
                    reason='no numerically verified historical AC reference; no candidate excluded')
    scores, records = {}, []
    for action in ACTION_NAMES:
            site=int(action[4])-1 if action.startswith('site') else 0
            policy=action.split('_')[1] if action.startswith('site') else action
            increments,ev,_=action_plan(site,policy)
            p = np.full((24,3),.2)+increments
            q = np.full((24,3),.06)
            futures = dict(P=p,Q=q,Y_drop=np.zeros_like(p),drop_error_bound=np.full_like(p,1e-5),
                           source_nominal_eff=np.full(24,.138),source_error_eff=np.full(24,.115),
                           source_p_abs_cap=np.full(24,.22),source_q_abs_cap=np.full(24,.066),batch_id=np.ones(24,int))
            vals = []
            for r,ref in refs.items():
                futures['Y_drop'],_,_=ac_batch(r,ref['theta'],p,q,.12)
                combined = {k:np.concatenate([history[k],futures[k]]) for k in history}
                rivals = []
                for h in retained:
                    if h != r:
                        result = solve_outer(combined,h,box=box)
                        records.append(dict(action=action,reference=r,candidate=h,result=result))
                        if result['lower'] is None:
                            return dict(status='numerically_unresolved',references=refs,records=records)
                        rivals.append(result)
                vals.append((min(v['lower'] for v in rivals),min(v['upper'] for v in rivals)))
            scores[action] = dict(
                lower=v6._down(sum((f(v[0]) for v in vals),F())/len(vals)),
                upper=v6._up(sum((f(v[1]) for v in vals),F())/len(vals)))
    endpoint = {k:v for k,v in scores.items() if k.endswith('endpoint')}
    selected = v7.rank_intervals(endpoint,tolerance=1e-9)
    mobile=v7.rank_intervals({k:v for k,v in scores.items() if k.startswith('mobile')},tolerance=1e-9)
    return dict(status='scored',references=all_refs,used_references=list(refs),scores=scores,selection=selected,records=records,
                mobile_selection=mobile,
                interpretation='nominal-future-source uniform reference average, not worst-case guarantee')


def run(output,mode='pilot'):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('use a fresh output folder')
    cases = [(h,kind,profile,seed) for h in (('cherry12',) if mode=='pilot' else tuple(v6.CANDIDATES))
             for kind,profile in [('flat','symmetric'),('rich','symmetric'),('flat','weak')]
             for seed in ((8101,) if mode=='pilot' else (8201,8202))]
    protocol = dict(mode=mode,cases=cases,box=BOX,source_sha256=v7.sha(__file__),
                    dependency_sha256={Path(m.__file__).name:v7.sha(m.__file__) for m in (v7,v6)},
                    python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                    model='balanced constant-power radial AC, no shunts, known X/R=.5 and trunk source',
                    EV='P-only, 20kW base; constant4.5kW / endpoint0or6kW duty75%; both27kWh/6h; min nonzero4.5kW; 60kWh SOC .4 to .8275',
                    nominal_source_P=.12,source_error_P=.1,source_Q_P_ratio=.3,epsilon=1e-5,
                    current_voltage_domain=[.9,1.1],selection='historical AC outer exclusion then nominal AC refit design',
                    no_truth_in_selector=True)
    protocol['actions']=['site1_none',*ACTION_NAMES]
    protocol['mobile']='three routes, two11-slot dwells and2-slot travel gap;9on6kW slots per dwell; same27kWh/6h plus2kWh battery trip'
    v7.dump(output/'protocol.json',protocol)
    episodes=[]
    for h,kind,profile,seed in cases:
        prefix=f'{h}_{kind}_{profile}_{seed}'
        first,first_truth=make_data(h,v7.PROFILES[profile],seed,kind)
        history=sliced(first,slice(0,12))
        history_results={j:solve_outer(history,j) for j in v6.CANDIDATES}
        retained=[j for j,r in history_results.items() if r['status']!='excluded']
        if len(retained)>1:
            design=design_ac(history,retained)
        elif retained:
            witness=fit_ac_reference(history,retained[0])
            design=dict(status='conditional_singleton_skip' if witness['historical_numeric_witness'] else 'compute_before_dispatch',
                        selection=None,references={retained[0]:witness})
        else:
            design=dict(status='inconsistent',selection=None)
        records=[]
        for action in protocol['actions']:
            site=int(action[4])-1 if action.startswith('site') else 0
            policy=action.split('_')[1] if action.startswith('site') else action
            obs,truth=make_data(h,v7.PROFILES[profile],seed,kind,site,policy)
            name=f'{prefix}_{action}'
            np.savez_compressed(output/(name+'.npz'),**obs)
            v7.dump(output/(name+'_truth.json'),truth)
            for j in v6.CANDIDATES:
                for sharing in ('path','output'):
                    result=solve_outer(obs,j,sharing=sharing)
                    records.append(dict(action=action,candidate=j,sharing=sharing,result=result))
                invalid_linear=solve_outer(obs,j,linear_only=True)
                records.append(dict(action=action,candidate=j,sharing='invalid_linear',result=invalid_linear))
        episode=dict(truth=h,profile=profile,history=kind,seed=seed,history_results=history_results,
                     retained=retained,design=design,records=records)
        episodes.append(episode)
        v7.dump(output/(prefix+'_results.json'),episode)
        print(json.dumps(dict(case=prefix,retained=retained,design=design['status'],selected=design.get('selection')),ensure_ascii=False),flush=True)
    inputs={p.name:v7.sha(p) for p in output.glob('*.npz')}
    inputs.update({p.name:v7.sha(p) for p in output.glob('*_truth.json')})
    v7.dump(output/'results.json',dict(protocol=protocol,episodes=episodes,
                                    input_sha256=inputs,source_hash_unchanged=v7.sha(__file__)==protocol['source_sha256']))
    return episodes


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['pilot','formal'],default='pilot')
    args=parser.parse_args()
    run(args.output,args.mode)
