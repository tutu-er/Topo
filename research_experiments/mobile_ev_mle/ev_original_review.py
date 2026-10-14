"""Independent replay of saved original-domain/mobile-EV study evidence.

No selector, data generator, optimization or new AC fit is invoked. Fits and
physical futures are checked by backward branch-current/forward-voltage
arithmetic; original-box QP certificates are checked by convexity arithmetic.
This reviewer is deliberately outside the experimental science freeze.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .information_fit_recovery import read_json, write_json, independent_replay, exact_candidate
from .information_replay import _obs, _meter
from .likelihood import RadialCandidate, concat_observations
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain

HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q')
METHODS = ('none', 'constant', 'quality_fixed', 'quality_mobile', 'random_mobile',
           'charge_fixed', 'v2g_fixed', 'charge_mobile', 'v2g_mobile')
DESIGNS = ('charge_fixed', 'v2g_fixed', 'charge_mobile', 'v2g_mobile')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arr(value):
    return np.asarray(value, float)


def close(actual, expected, message, *, atol=2e-7, rtol=2e-8):
    a, b = arr(actual), arr(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), message)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def control_key(value):
    return np.asarray(value, dtype='<f8').tobytes()


def source_audit(directory, freeze):
    saved = read_json(directory / 'source_snapshot.json.gz')
    hashes = freeze['source_sha256']
    require(set(saved) == set(hashes), 'snapshot/hash-map coverage')
    require(not any('ev_original_review' in k for k in hashes), 'reviewer included in science freeze')
    rows = []; mismatches=[]
    for name, expected in hashes.items():
        payload = base64.b64decode(saved[name]['bytes_base64'], validate=True)
        path = ((HERE.parents[1] / name) if name.startswith('rnj_wzzt_core/') else HERE / name).resolve()
        path.relative_to(HERE.parents[1].resolve())
        require(saved[name]['sha256'] == expected and hashlib.sha256(payload).hexdigest() == expected,
                'snapshot bytes: ' + name)
        current=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if current!=expected:mismatches.append(dict(name=name,frozen_sha256=expected,current_sha256=current))
        rows.append(dict(name=name, sha256=expected,current_sha256=current))
    return dict(success=True, source_count=len(rows), rows=rows,current_source_matches=not mismatches,
                current_source_mismatches=mismatches,
                interpretation='saved source bytes verified; current files may belong to a later version')


def forward_physics(candidate, r, x, p, q, sp, sq, root):
    """Separate AC branch-current sweep; source always at first receiving node."""
    r, x, p, q, sp, sq, root = map(arr, (r, x, p, q, sp, sq, root))
    load = np.zeros((len(p), candidate.m), complex)
    load[:, 0] = sp + 1j * sq
    load[:, candidate.terminal_edges] += p + 1j * q
    voltage = np.broadcast_to(root[:, None], load.shape).astype(complex).copy()
    impedance = r + 1j * x
    for iteration in range(1, 1001):
        current = np.conj(load / voltage)
        for edge in range(candidate.m - 1, -1, -1):
            parent = candidate.parents[edge]
            if parent >= 0:
                current[:, parent] += current[:, edge]
        updated = np.empty_like(voltage)
        for edge, parent in enumerate(candidate.parents):
            updated[:, edge] = (root if parent < 0 else updated[:, parent]) - impedance[edge] * current[:, edge]
        if np.max(abs(updated - voltage)) < 2e-13:
            voltage = updated
            break
        voltage = updated
    else:
        raise ValueError('independent physical AC did not converge')
    current = np.conj(load / voltage)
    for edge in range(candidate.m - 1, -1, -1):
        if candidate.parents[edge] >= 0:
            current[:, candidate.parents[edge]] += current[:, edge]
    master = root * np.conj(current[:, 0])
    equation = max(float(np.max(abs(voltage[:, e] - (root if par < 0 else voltage[:, par])
                                    + impedance[e] * current[:, e])))
                   for e, par in enumerate(candidate.parents))
    balance = float(np.max(abs(master - load.sum(axis=1) - (abs(current)**2 * impedance).sum(axis=1))))
    require(equation < 2e-10 and balance < 2e-10, 'independent physical equations/balance')
    return dict(p=p, q=q, v=abs(voltage[:, candidate.terminal_edges]),
                master_p=master.real, master_q=master.imag, source_p=sp, source_q=sq,
                root_v=root, equation_residual=equation, balance_residual=balance,
                voltage_min=float(abs(voltage).min()), voltage_max=float(abs(voltage).max()),
                iterations=iteration)


def whiten(physical, meter, channels, *, observation=False):
    rows = []
    for k in channels:
        v = arr(physical[k])
        if not observation and k in ('p', 'q', 'v'):
            v = v + np.asarray(getattr(meter, 'bias_' + k))
        rows.append((v / np.broadcast_to(getattr(meter, k + '_sigma'), v.shape)).ravel())
    return np.concatenate(rows)


def unpack_state(vector, n):
    a = arr(vector).reshape(-1, 2 * n + 2)
    return dict(p=a[:, :n], q=a[:, n:2*n], source_p=a[:, 2*n], source_q=a[:, 2*n+1])


def state_vector(state):
    return np.column_stack((state['p'], state['q'], state['source_p'], state['source_q'])).ravel()


def replay_fits(rows, candidates, observations, meter, failures, *, complete, config, domain):
    names = [r['name'] for r in rows]
    require(len(names) == len(set(names)) and set(names) <= set(candidates), 'fit candidate coverage/order')
    require(set(names) | set(failures) == set(candidates), 'failed fits disappear from denominator')
    if complete:
        require(set(names) == set(candidates) and not failures, 'complete fits missing a required model')
    out = []
    for row in rows:
        candidate = candidates[row['name']]
        require(tuple(map(tuple, row['supports'])) == candidate.supports, 'saved edge order')
        if any(row.get(k) is None for k in ('r', 'x', 'p', 'q', 'source_p', 'source_q', 'loss', 'loglik')):
            require(not row['success'], 'successful fit has missing state')
            out.append(dict(name=row['name'], unavailable_failed_state=True))
            continue
        audit = independent_replay(candidate, row, observations, meter)
        require(audit['success'], 'AC/SSE/loglik replay ' + row['name'] + ':' + str(audit['errors']))
        close(row['half_loss'], .5 * audit['full_SSE'], 'half SSE convention')
        close(row['r_ohm'], arr(row['r']) * config['zbase_ohm'], 'R unit conversion')
        close(row['x_ohm'], arr(row['x']) * config['zbase_ohm'], 'X unit conversion')
        if domain == 'asset':
            for k in ('r', 'x'):
                lo, hi = config['asset_bounds'][k + '_bounds']
                require(np.all(arr(row[k]) >= lo - 1e-9) and np.all(arr(row[k]) <= hi + 1e-9),
                        'fit outside declared asset family: ' + k)
        require(not row.get('global_MLE_certified', False), 'local fit labelled global')
        out.append(dict(name=row['name'], declared_success=bool(row['success']), **audit))
    return dict(saved_states=len(rows), replayed_states=sum(not a.get('unavailable_failed_state', False) for a in out),
                failed_states=sum(not r['success'] for r in rows), audits=out)


def service_config(config, method):
    d = config['device']; mobile = 'mobile' in method; discharge = method.startswith('v2g') or method in ('quality_mobile', 'random_mobile')
    T = config['horizon']; access = np.ones((T, 3), bool)
    if not mobile:
        access[:, 1:] = False
    return MobileServiceConfig(n_slots=T, n_sites=3, dt_hours=d['dt_hours'],
        p_charge_max_kw=d['power_kw'], p_discharge_max_kw=d['power_kw'], allow_discharge=discharge,
        battery_capacity_kwh=d['capacity_kwh'], initial_soc=d['initial_energy_kwh']/d['capacity_kwh'],
        terminal_energy_kwh=d['terminal_energy_kwh'], eta_charge=d['eta_charge'], eta_discharge=d['eta_discharge'],
        mobile=mobile, access=access, setup_slots=d['setup_slots'], travel_slots=d['travel_slots'],
        travel_energy_kwh=d['drive_kwh'], max_moves=d['max_moves'], max_throughput_kwh=d['throughput_kwh'],
        travel_cost=d['travel_cost'], setup_cost=d['setup_cost'], driver_budget=d['driver_budget'], cost_units=d['cost_units'],
        min_arrival_throughput_kwh=d.get('min_arrival_throughput_kwh',0.))


def replay_operation_record(saved, cfg):
    """Check saved path/energy without asking a MILP to construct a plan."""
    require(saved['feasible'], 'saved operation explicitly infeasible')
    data = cfg.arrays(); T, J = cfg.n_slots, cfg.n_sites
    u = arr(saved['u_kw']); require(u.shape == (T, J), 'operation power shape')
    connected = np.zeros((T, J), bool); driving = np.zeros(T)
    node = (0, cfg.initial_site, cfg.initial_connected); cost = 0.; moves = 0
    for a in saved['route']:
        start = (a['departure'], a['origin'], a['origin_connected'])
        end = (a['arrival'], a['destination'], a['destination_connected'])
        require(start == node and 0 <= start[0] <= end[0] <= T, 'route continuity/time')
        i, j, t, stop = a['origin'], a['destination'], a['departure'], a['arrival']
        require(0 <= i < J and 0 <= j < J and t < T, 'route node range')
        kind = a['kind']
        if kind == 'connected':
            require(i == j and start[2] and end[2] and stop == t + 1, 'connected arc')
            connected[t, i] = True
        elif kind == 'wait':
            require(i == j and not start[2] and not end[2] and stop == t + 1, 'unconnected wait')
        elif kind == 'setup':
            require(i == j and not start[2] and end[2] and stop == t + data['setup'][i], 'setup arc')
            sl = slice(t, max(t+1, stop))
            require(data['access'][sl, i].all() and data['worker'][sl, i].all(), 'setup access/personnel')
            cost += data['setup_cost'][i]
        elif kind == 'travel':
            require(i != j and not end[2] and data['allowed'][i, j]
                    and stop == t + data['travel'][i, j], 'travel arc')
            driving[t] += data['drive'][i, j]; cost += data['travel_cost'][i, j]; moves += 1
        else:
            raise ValueError('unknown route arc')
        close(a['driving_energy_kwh'], data['drive'][i, j] if kind == 'travel' else 0., 'arc drive energy')
        close(a['driver_cost'], data['travel_cost'][i, j] if kind == 'travel'
              else data['setup_cost'][i] if kind == 'setup' else 0., 'arc driver cost')
        node = end
    require(node[0] == T, 'unfinished path')
    pp, pm = np.maximum(u, 0), np.maximum(-u, 0)
    require(np.all(pp <= connected*data['access']*data['charge']+1e-7)
            and np.all(pm <= connected*data['access']*data['discharge']+1e-7), 'route power/access/charge mode')
    E = np.r_[cfg.initial_soc*cfg.battery_capacity_kwh,
        cfg.initial_soc*cfg.battery_capacity_kwh + np.cumsum(cfg.dt_hours*(cfg.eta_charge*pp.sum(axis=1)
             - pm.sum(axis=1)/cfg.eta_discharge)-driving)]
    require(E.min() >= cfg.soc_min*cfg.battery_capacity_kwh-1e-7
            and E.max() <= cfg.soc_max*cfg.battery_capacity_kwh+1e-7, 'intermediate SOC')
    close(E[-1], cfg.terminal_energy_kwh, 'equal terminal battery service')
    close(saved['energy_kwh'], E, 'saved energy trajectory')
    close(saved['soc'], E/cfg.battery_capacity_kwh, 'saved SOC')
    require(np.array_equal(saved['connected'], connected), 'saved connected occupancy')
    metrics = dict(grid_energy_kwh=cfg.dt_hours*u.sum(), throughput_kwh=cfg.dt_hours*(pp+pm).sum(),
        travel_energy_kwh=driving.sum(), driver_cost=cost, moves=moves,
        conversion_loss_kwh=cfg.dt_hours*((1-cfg.eta_charge)*pp+(1/cfg.eta_discharge-1)*pm).sum())
    for key, value in metrics.items(): close(saved[key], value, 'operation metric: '+key)
    require(moves <= cfg.max_moves and cost <= cfg.driver_budget+1e-7
            and metrics['throughput_kwh'] <= cfg.max_throughput_kwh+1e-7, 'operation budgets')
    require(np.max(np.count_nonzero(abs(u)>1e-7, axis=1)) <= 1, 'single site per slot')
    for a in saved['route']:
        if a['kind']=='travel':
            require(cfg.dt_hours*abs(u[a['arrival']:,a['destination']]).sum()
                    >=cfg.min_arrival_throughput_kwh-1e-7,'arrival destination probing service')
    return dict(feasible=True, **metrics)


def replay_qp(model, u, saved):
    D = model['domain']; u = arr(u); xi = arr(saved['physical_parameter'])
    lo, hi, base, S = (arr(D[k]) for k in ('physical_lower','physical_upper','base_center','shift'))
    require(xi.shape == lo.shape == hi.shape and np.all(xi >= lo-1e-8) and np.all(xi <= hi+1e-8),
            'original QP witness outside physical box')
    z = xi-base-S@u; A = arr(model['A0'])+np.einsum('k,krp->rp', u, arr(model['At']))
    r = arr(model['c'])+arr(model['B'])@u+A@z
    close(saved['z'], z, 'original QP moving map'); close(saved['residual'], r, 'original QP residual')
    primal = float(r@r); gradient = 2*A.T@r
    if saved.get('fallback'):
        require(not saved['success'],'QP exception fallback marked converged')
        close(saved['primal_upper'],primal,'fallback feasible original-box residual')
        close(saved['tangent_lower_numeric'],0.,'fallback nonnegative distance lower')
        return dict(lower=0.,upper=primal,success=False,declared_exception_fallback=True)
    correction = float(np.minimum(gradient*(lo-xi), gradient*(hi-xi)).sum())
    guard = 256*np.finfo(float).eps*(1+abs(primal)+abs(correction)
        +float(abs(gradient)@(abs(lo)+abs(hi)+2*abs(xi))))
    lower = max(0., min(primal, primal+correction-guard))
    close(saved['primal_upper'], primal, 'original QP feasible upper')
    close(saved['tangent_lower_numeric'], lower, 'original QP tangent lower')
    close(saved['numeric_guard'], guard, 'QP numeric guard', atol=2e-10)
    require(lower <= primal+1e-8, 'QP interval order')
    return dict(lower=lower, upper=primal, success=bool(saved['success']))


def model_audit(model, config):
    D = model['domain']; ref = arr(model['reference_kw']); A = arr(model['A0']); At = arr(model['At'])
    close(A+np.einsum('k,krp->rp',ref,At), model['C_at_reference'], 'reference Jacobian closure')
    close(arr(model['c'])+arr(model['B'])@ref, model['reference_residual'], 'reference residual closure')
    require(model['source_interface'] == 1 and not model['global_AC_certificate'], 'source/certificate scope')
    n=3; T=config['horizon']; width=2*n+2; nl=model['line_columns']; nh=model['history_state_columns']
    expected=np.zeros_like(arr(D['shift'])); ids=np.arange(T*n)
    expected[nl+nh+(ids//n)*width+ids%n, ids]=1/config['base_kw']
    close(D['shift'], expected, 'EV controls affect total P exactly once, no Q/source shift', atol=1e-12)
    low, high, base = (arr(D[k]) for k in ('action_lower','action_upper','base_center'))
    slow=np.minimum(expected*low,expected*high).sum(axis=1)
    shigh=np.maximum(expected*low,expected*high).sum(axis=1)
    close(D['outer_lower'],arr(D['physical_lower'])-base-shigh,'outer lower containment')
    close(D['outer_upper'],arr(D['physical_upper'])-base-slow,'outer upper containment')
    return dict(name=model['name'], rows=A.shape[0], physical_columns=A.shape[1])


def history_reference_audit(model, history, hm, hfits, config):
    """Check the frozen reference prediction against H, not simulator truth."""
    fit=hfits[model['a']];T=config['horizon']
    for key in ('r','x'):close(model['reference_'+key],fit[key],'reference parameters come from H fit:'+key,atol=1e-12)
    expected=dict(p=np.tile(np.mean(arr(history.p)-arr(hm.bias_p),axis=0),(T,1)),
                  q=np.tile(np.mean(arr(history.q)-arr(hm.bias_q),axis=0),(T,1)),
                  source_p=np.full(T,np.mean(fit['source_p'])),source_q=np.full(T,np.mean(fit['source_q'])))
    for key,value in expected.items():close(model['reference_background'][key],value,'reference background comes from H:'+key,atol=1e-12)
    close(model['future_root_v'],np.full(T,np.mean(history.root_v)),'reference root comes from H',atol=1e-12)


def asset_unit_audit(trial):
    config=trial['config'];asset=config['asset'];truth=trial['truth']
    zbase=config['base_kv']**2/(config['base_kw']/1000.)
    close(config['zbase_ohm'],zbase,'physical voltage/power impedance base',atol=1e-12)
    require(asset['length_is_synthetic_family'],'synthetic lengths relabelled as measurements')
    c=RadialCandidate.from_supports('asset_audit',3,truth['family'])
    require(len(truth['assets'])==c.m,'asset/edge coverage')
    for k,record in enumerate(truth['assets']):
        require(tuple(record['support'])==c.supports[k],'asset edge parameter order')
        code=(50,120,150).index(record['cable_mm2']);length=record['length_km']
        close(truth['r'][k],asset['R_ohm_km'][code]*length/zbase,'asset ohm/km*km to R pu',atol=1e-12)
        close(truth['x'][k],asset['X_ohm_km'][code]*length/zbase,'asset ohm/km*km to X pu',atol=1e-12)
    length_lo,length_hi=asset['declared_length_km']
    close(config['asset_bounds']['r_bounds'],[min(asset['R_ohm_km'])*length_lo/zbase,
        max(asset['R_ohm_km'])*length_hi/zbase],'declared family R bounds',atol=1e-12)
    close(config['asset_bounds']['x_bounds'],[min(asset['X_ohm_km'])*length_lo/zbase,
        max(asset['X_ohm_km'])*length_hi/zbase],'declared family X bounds',atol=1e-12)
    return dict(edges=c.m,zbase_ohm=zbase,source=asset['source'],lengths_are_declared_simulation=True)


def polynomial(model, witness):
    D=model['domain']; b=arr(witness['physical_intercept']); slope=arr(witness['physical_slope'])
    delta=b-arr(D['base_center']); G=slope-arr(D['shift']); At=arr(model['At'])
    a=arr(model['c'])+arr(model['A0'])@delta
    B=arr(model['B'])+np.einsum('krp,p->rk',At,delta)+arr(model['A0'])@G
    raw=np.einsum('krp,ps->rks',At,G); terms={}
    for t in range(B.shape[1]):
        for s in range(t,B.shape[1]):
            v=raw[:,t,s] if t==s else raw[:,t,s]+raw[:,s,t]
            if np.any(v): terms[t,s]=v
    return a,B,terms


def witness_box_upper(model, witness, zeros, offset, scale, *, control_lower=None, control_upper=None):
    D=model['domain']; low=arr(D['action_lower'] if control_lower is None else control_lower)
    high=arr(D['action_upper'] if control_upper is None else control_upper)
    b,S=arr(witness['physical_intercept']),arr(witness['physical_slope'])
    wl=b+np.minimum(S*low,S*high).sum(axis=1); wh=b+np.maximum(S*low,S*high).sum(axis=1)
    valid_box=np.all(wl>=arr(D['physical_lower'])-1e-12) and np.all(wh<=arr(D['physical_upper'])+1e-12)
    require(valid_box or witness['validation']['method']=='control_box_and_operation_LP_outer_relaxation',
            'affine witness lacks global physical feasibility evidence')
    a,B,terms=polynomial(model,witness); lo=a+np.minimum(B*low,B*high).sum(axis=1)
    hi=a+np.maximum(B*low,B*high).sum(axis=1); magnitude=1+abs(a)+(abs(B)*np.maximum(abs(low),abs(high))).sum(axis=1)
    variable=np.any((B!=0)&(low!=high),axis=1)
    for (t,s),v in terms.items():
        if (t,s) in zeros:l,h=0.,0.
        elif t==s:l,h=(0. if low[t]<=0<=high[t] else min(low[t]**2,high[t]**2)),max(low[t]**2,high[t]**2)
        else:
            corners=[low[t]*low[s],low[t]*high[s],high[t]*low[s],high[t]*high[s]];l,h=min(corners),max(corners)
        lo+=np.minimum(v*l,v*h);hi+=np.maximum(v*l,v*h)
        magnitude+=abs(v)*max(abs(l),abs(h));variable|=(v!=0)&(l!=h)
    pad=np.where(variable,64*np.finfo(float).eps*magnitude,0.);lo-=pad;hi+=pad
    cost=float(np.sum(np.maximum(lo*lo,hi*hi).astype(np.longdouble)))
    guard=256*np.finfo(float).eps*(1+abs(cost)+abs(offset))
    return scale*(cost-offset+guard),bool(valid_box)


def master_backend_audit(master, config):
    """Read solver translation/start metadata; no solver is called here."""
    details=master.get('backend_details')
    if not details:return dict(available=False,version='legacy saved scipy master')
    require(details['objective_sense']=='min' and details['objective_transform']=='none',
            'backend silently changes minimization contract')
    require(details['backend_requested']==master['backend']==config['exchange']['backend'],
            'backend request differs from frozen protocol')
    if master['backend'] in ('gurobi','scipy'):
        require(details['backend_used']==master['backend'] and not details['fallback_used'],
                'explicit backend silently falls back')
    warm=master['warm_start'];count=details.get('start_provided_count') or 0
    require(count==warm['filled_variables'],'warm-start supplied-variable count')
    if warm.get('provided_to_solver'):
        require(details['backend_used']=='gurobi' and details['start_supplied'] and count>0,
                'warm-start claim lacks submitted coordinates')
        require(warm['operation_validation']['feasible'],'infeasible warm-start operation')
    if details.get('start_used'):
        require(details['backend_used']=='gurobi' and count>0,'start submission metadata')
    if details['backend_used']=='gurobi' and details.get('gurobi_status') is not None:
        require(details['threads']==1 and details['seed']==0,'frozen Gurobi thread/seed')
    scale=master['scale'];require(np.isfinite(scale) and scale>0,'positive objective scale')
    lower_min=None;incumbent_min=None
    if master.get('solver_best_bound') is not None:
        lower_min=(master['numeric_guard']-master['solver_best_bound'])/scale
    if master.get('surrogate_incumbent') is not None:
        incumbent_min=-master['surrogate_incumbent']/scale
    if lower_min is not None and incumbent_min is not None and master['bound_consistent']:
        require(lower_min<=incumbent_min+1e-8,'MIN backend lower/upper direction')
    diagnostics=[]
    for c in master.get('compressions',[]):
        correct=.5*(c['product_cost_error_upper']+c['residual_square_error_upper']+2*c['qr_upward_cost_guard'])
        close(c['pwl_score_error_upper'],correct,'two QR errors in witness PWL diagnostic',atol=2e-9)
        diagnostics.append(correct)
    return dict(available=True,backend=details['backend_used'],status=master['status'],
        warm_start_submitted=bool(warm.get('provided_to_solver')),
        warm_start_coordinates=count,backend_seconds=details.get('backend_seconds'),
        raw_MIN_lower=lower_min,raw_MIN_incumbent=incumbent_min,
        interpretation='start_used records submission, not solver acceptance or improvement',
        PWL_diagnostics_checked=len(diagnostics))


def replay_exchange(models, saved, config, cfg, losses):
    if saved is None:return dict(available=False)
    if saved.get('u') is None:
        require(not saved['completed'], 'missing action marked complete')
        return dict(available=True, status=saved['status'], failed_without_action=True, QP_states=0)
    require(saved['required_pairs']==[m['name'] for m in models] and not saved['full_curve_enumeration']
            and not saved['global_nonlinear_AC_optimum_certified'] and saved['physical_domain_preserved'], 'exchange pair/scope')
    w=arr([saved['weights'][m['name']] for m in models]);scale=saved['score_scale']
    require(np.all(w>=0) and abs(w.sum()-1)<1e-12 and scale==.5,'task weights/half-SSE scale')
    close(w,np.full(len(models),1/len(models)),'predeclared equal task weights',atol=1e-12)
    offsets=arr([losses[m['b']] for m in models])
    for m,o in zip(models,offsets):close(saved['offsets'][m['name']],o,'historical loss offset')
    d=build_mobile_service_domain(cfg); entries=[]; QP_states=0
    for e in saved['evaluations']:
        audit=d.check(e['operation_x']);require(audit['feasible'],'exchange full operation x')
        close(arr(e['operation_x'])[d.indices.u].ravel(),e['u'],'u=Ux saved evaluation')
        require(len(e['qp_results'])==len(models),'all-pair QP coverage')
        qs=[replay_qp(m,e['u'],q) for m,q in zip(models,e['qp_results'])];QP_states+=len(qs)
        lo=scale*(arr([q['lower'] for q in qs])-offsets);hi=scale*(arr([q['upper'] for q in qs])-offsets)
        close(e['pair_score_lower'],lo,'pair score lower');close(e['pair_score_upper'],hi,'pair score upper')
        mean=[w@lo,w@hi];worst=[min(lo),min(hi)]
        close(e['mean_score_interval'],mean,'mean interval');close(e['worst_score_interval'],worst,'worst interval')
        selected=mean if saved['aggregation']=='mean' else worst
        close(e['score_lower'],selected[0],'attained score lower');close(e['score_upper'],selected[1],'attained score upper')
        entries.append(selected)
    close(saved['objective_lower'],max(e[0] for e in entries),'best attained lower')
    best=saved['best_evaluation_index'];close(saved['u'],saved['evaluations'][best]['u'],'best curve')
    close(saved['operation_x'],saved['evaluations'][best]['operation_x'],'best complete operation')
    lower,upper=saved['objective_lower'],saved['objective_upper'];valid=upper>=lower
    close(saved['raw_gap'],upper-lower,'signed proxy gap');close(saved['gap'],upper-lower,'proxy gap')
    if saved['completed']:
        require(valid and upper-lower<=saved['target_gap']+1e-8,'completed proxy interval')
    else:
        require(saved['unfinished'],'unfinished proxy missing status')
    zeros={tuple(v) for v in saved['zero_product_pairs']}
    expected={(t*3+j,t*3+k) for t in range(config['horizon']) for j in range(3) for k in range(j+1,3)}
    require(zeros==expected and saved['zero_product_proof'],'zero-product operation structure')
    op=d.as_dict();counts=saved['iterations'][0]['witnesses_before'] if saved['iterations'] else saved['witness_counts']
    initial=[]; LP_unreplayed=0
    for k,m in enumerate(models):
        values=[]
        for i,witness in enumerate(saved['witnesses'][k]):
            value,box=witness_box_upper(m,witness,zeros,offsets[k],scale,
                control_lower=op['control_lower'],control_upper=op['control_upper'])
            LP_unreplayed+=int(not box)
            if i<counts[k]:values.append(value)
        initial.append(min(values))
    initial_upper=float(w@arr(initial)) if saved['aggregation']=='mean' else float(min(initial))
    close(saved['initial_pair_coarse_upper'],initial,'initial pair witness box uppers',atol=2e-5)
    close(saved['initial_fixed_witness_upper'],initial_upper,'initial witness global upper',atol=2e-5)
    previous=saved['initial_fixed_witness_upper']
    master_calls=0;backend_audits=[]
    for it in saved['iterations']:
        master=it['master'];master_calls+=1
        if master.get('scale') is not None:backend_audits.append(master_backend_audit(master,config))
        close(it['previous_upper'],previous,'retained upper sequence')
        if master.get('u') is not None:
            audit=d.check(master['operation_x']);require(audit['feasible'],'master operation incumbent')
            close(arr(master['operation_x'])[d.indices.u].ravel(),master['u'],'master u=Ux')
        if it.get('upper_update_accepted'):
            require(master['bound_consistent'] and it['global_upper']<=previous+1e-8
                    and it['global_upper']>=it['global_lower']-1e-8,'accepted proxy bound update')
            previous=it['global_upper']
        else:require(not saved['completed'],'rejected master bound marked complete')
        if master.get('bound_consistent') and master.get('surrogate_incumbent') is not None:
            require(master['objective_upper']>=master['surrogate_incumbent']-1e-8,'MILP bound/incumbent order')
    close(saved['objective_upper'],previous,'final retained proxy upper')
    require(saved['qp_calls']==QP_states,'QP call/state denominator')
    return dict(available=True,status=saved['status'],completed=saved['completed'],valid_interval=valid,
                lower=lower,upper=upper,QP_states=QP_states,master_calls=master_calls,
                backend_audits=backend_audits,
                affine_LP_feasibility_records_not_independently_replayed=LP_unreplayed,
                scope='declared bilinear proxy; numeric solver bounds, no global AC certificate')


def replay_same_witness(model, u, qp, saved, history, hm, fm, hfits, candidates):
    """Three separate AC sweeps at the saved common absolute explanation."""
    n=3;xi=arr(qp['physical_parameter']);nl=model['line_columns'];nh=model['history_state_columns']
    b=candidates[model['b']];a=candidates[model['a']];theta=xi[:nl]
    wh=unpack_state(xi[nl:nl+nh],n);wn=unpack_state(xi[nl+nh:],n)
    H=forward_physics(b,theta[:b.m],theta[b.m:],**dict(p=wh['p'],q=wh['q'],sp=wh['source_p'],sq=wh['source_q'],root=history.root_v))
    N=forward_physics(b,theta[:b.m],theta[b.m:],wn['p'],wn['q'],wn['source_p'],wn['source_q'],model['future_root_v'])
    ref=unpack_state(state_vector(model['reference_background'])+arr(model['domain']['shift'])[-model['future_state_columns']:]@arr(u),n)
    R=forward_physics(a,model['reference_r'],model['reference_x'],ref['p'],ref['q'],ref['source_p'],ref['source_q'],model['future_root_v'])
    channels=model['channels'];observed={k:getattr(history,k) for k in channels}
    exact=np.r_[whiten(H,hm,channels)-whiten(observed,hm,channels,observation=True),
                whiten(R,fm,channels)-whiten(N,fm,channels)]
    D=model['domain'];z=xi-arr(D['base_center'])-arr(D['shift'])@arr(u)
    proxy=arr(model['c'])+arr(model['B'])@arr(u)+(arr(model['A0'])+np.einsum('k,krp->rp',arr(u),arr(model['At'])))@z
    difference=proxy-exact;hrows=model['history_rows']
    close(saved['complete_AC_residual'],exact,'same-witness full AC residual',atol=2e-6)
    close(saved['proxy_residual'],proxy,'same-witness proxy residual')
    close(saved['proxy_minus_AC'],difference,'same-witness model error',atol=2e-6)
    rms={};offset=0
    for part,times,vector in [('history',history.t,difference[:hrows]),('future',len(ref['p']),difference[hrows:])]:
        offset=0;rms[part]={}
        for k in channels:
            count=times*(n if k in ('p','q','v') else 1);v=vector[offset:offset+count];offset+=count
            rms[part][k]=float(np.linalg.norm(v)/np.sqrt(count))
        require(offset==len(vector),'same-witness channel partition')
        for k,v in rms[part].items():close(saved[part+'_channel_RMS'][k],v,'same-witness channel RMS',atol=2e-6)
    maximum=max(*rms['history'].values(),*rms['future'].values())
    close(saved['max_channel_RMS'],maximum,'same-witness maximum RMS',atol=2e-6)
    for key,value in [('complete_AC_cost',exact@exact),('proxy_cost',proxy@proxy),
                      ('cost_error',proxy@proxy-exact@exact),('error_norm',np.linalg.norm(difference)),
                      ('complete_AC_H_cost',exact[:hrows]@exact[:hrows]),('complete_AC_N_cost',exact[hrows:]@exact[hrows:])]:
        close(saved[key],value,'same-witness cost:'+key,atol=3e-5)
    require(saved['source_interface']==1,'same-witness source physical interface')
    ranges=dict(competitor_history=[H['voltage_min'],H['voltage_max']],
                competitor_future=[N['voltage_min'],N['voltage_max']],reference_future=[R['voltage_min'],R['voltage_max']])
    for key,value in ranges.items():close(saved['voltage_ranges'][key],value,'same-witness voltage range')
    voltage_valid=all(v[0]>=.9 and v[1]<=1.1 for v in ranges.values())
    require(saved['complete_AC_voltage_feasible']==voltage_valid,'same-witness voltage validity flag')
    require(saved['strict_RMS_025_pass']==bool(maximum<=.25),'same-witness RMS validity flag')
    return dict(max_channel_RMS=maximum,physical_box_feasible=bool(saved['physical_box_feasible']),
                voltage_valid=voltage_valid,independent_AC_replays=3)


def predicted_audit(saved, history, hm, fm, jm, candidates, hfits, config, domain):
    require(not saved['uses_truth'] and not saved['uses_future_noise'] and saved['source_interface_shared'], 'predicted score information/interface')
    names=list(candidates);order=[(a,b) for a in names for b in names if a!=b]
    require(list(map(tuple,saved['weight_order']))==order and len(saved['pairs'])==len(order)==12,'all ordered AC tasks')
    power=arr(saved['power_kw']);lossH={n:float(hfits[n]['loss']) for n in names};allfits={};fit_states=0
    background_p=np.tile(np.mean(arr(history.p)-arr(hm.bias_p),axis=0),(len(power),1))
    background_q=np.tile(np.mean(arr(history.q)-arr(hm.bias_q),axis=0),(len(power),1))
    for rec in saved['records']:
        a=rec['a'];forecast=rec['forecast_state']
        close(forecast['p'],background_p+power/config['base_kw'],'H-only P forecast plus one EV',atol=1e-12)
        close(forecast['q'],background_q,'H-only Q forecast',atol=1e-12)
        for k in ('source_p','source_q'):
            close(forecast[k],np.full(len(power),np.mean(hfits[a][k])),'H-only source mean '+k,atol=1e-12)
        close(forecast['root_v'],np.full(len(power),np.mean(history.root_v)),'H-only root forecast',atol=1e-12)
        if rec.get('predicted_observations') is None:
            require(not rec['success'],'successful forecast has no observations');continue
        phy=forward_physics(candidates[a],hfits[a]['r'],hfits[a]['x'],forecast['p'],forecast['q'],forecast['source_p'],forecast['source_q'],forecast['root_v'])
        obs=_obs(rec['predicted_observations'])
        for k in CHANNELS:
            v=getattr(obs,k)
            if v is not None:close(v,phy[k]+(getattr(fm,'bias_'+k) if k in ('p','q','v') else 0),'predicted noiseless AC:'+k,atol=2e-10)
        audit=replay_fits(rec['fits'],candidates,concat_observations(history,obs),jm,rec['failures'],
                          complete=rec['success'],config=config,domain=domain);fit_states+=audit['replayed_states']
        allfits[a]={r['name']:r for r in rec['fits']}
    recomputed={k:[] for k in ('added','joint','self_relative')}
    for (a,b),r in zip(order,saved['pairs']):
        require((r['a'],r['b'])==(a,b),'AC pair order')
        if not r['success']:continue
        own,comp=allfits[a][a],allfits[a][b]
        values=dict(added=.5*(comp['loss']-lossH[b]),joint=.5*(comp['loss']-lossH[a]),self_relative=.5*(comp['loss']-own['loss']))
        for k,v in values.items():close(r[k],v,'AC pair score:'+k);recomputed[k].append(v)
    if saved['success']:
        require(all(len(v)==12 for v in recomputed.values()),'successful AC score missing pair')
        w=arr(saved['weights']);require(np.all(w>=0) and abs(w.sum()-1)<1e-12,'AC task weights')
        close(w,np.full(12,1/12),'predeclared equal AC task weights',atol=1e-12)
        for k,v in recomputed.items():
            close(saved['mean_'+k+'_score'],w@arr(v),'AC mean score:'+k)
            close(saved['worst_'+k+'_score'],min(v),'AC worst score:'+k)
        close(saved['score'],saved['mean_added_score'],'AC selected score')
    return dict(success=bool(saved['success']),replayed_fit_states=fit_states,
                fit_calls=saved['fit_calls'],ac_calls=saved['ac_calls'],power_kw=power)


def future_audit(trial, row, fm):
    config=trial['config'];truth=trial['truth'];h=config['history'];power=arr(row['power_kw'])
    c=RadialCandidate.from_supports('independent_truth',3,truth['family'])
    p=arr(truth['p'])[h:]+power/config['base_kw']
    phy=forward_physics(c,truth['r'],truth['x'],p,arr(truth['q'])[h:],arr(truth['source_p'])[h:],arr(truth['source_q'])[h:],arr(truth['root_v'])[h:])
    for k in ('p','q','v','master_p','master_q'):close(row['physical'][k],phy[k],'true future AC/EV once:'+k,atol=2e-10)
    if row.get('observations') is not None:
        for k,v in row['observations'].items():
            if v is None:continue
            if k=='root_v':expected=phy[k]
            else:expected=phy[k]+np.asarray(getattr(fm,k+'_sigma'))*arr(truth['white'][k])[h:]+(getattr(fm,'bias_'+k) if k in ('p','q','v') else 0)
            close(v,expected,'paired actual observations:'+k,atol=2e-10)
    close(row['grid_net_kwh'],config['dt_hours']*power.sum(),'evaluation grid energy')
    close(row['grid_throughput_kwh'],config['dt_hours']*abs(power).sum(),'evaluation throughput')
    return dict(independent_AC_replays=1,equation_residual=phy['equation_residual'],balance_residual=phy['balance_residual'])


def noiseless_distance_audit(detail, fits, hfits, obs, fm, history_points):
    fit=next((f for f in fits if f['name']==detail['candidate']),None)
    if fit is None:
        require(detail.get('success') is False and detail.get('reason')=='required_future_fit_missing'
            and all(detail[k] is None for k in ('attained_joint_SSE','added_half_SSE','future_channel_SSE','historical_SSE')),
            'missing noiseless fit loses explicit unavailable row')
        return dict(available=False,reason='required_future_fit_missing')
    futureSSE={}
    for k in CHANNELS:
        if getattr(obs,k) is not None:
            futureSSE[k]=float(np.sum(((arr(fit['predicted'][k])[history_points:]-getattr(obs,k))
                                       /np.asarray(getattr(fm,k+'_sigma')))**2))
    require(set(detail['future_channel_SSE'])==set(futureSSE),'noiseless channel partition')
    for k,v in futureSSE.items():close(detail['future_channel_SSE'][k],v,'noiseless future channel SSE')
    close(detail['attained_joint_SSE'],fit['loss'],'noiseless attained joint SSE')
    close(detail['historical_SSE'],fit['loss']-sum(futureSSE.values()),'noiseless historical SSE')
    hf=hfits.get(fit['name']);baseline_ok=hf is not None and hf['success'] and np.isfinite(hf['loss'])
    if baseline_ok:close(detail['added_half_SSE'],.5*(fit['loss']-hf['loss']),'noiseless added score')
    else:require(detail['added_half_SSE'] is None and detail['reason']=='required_historical_fit_failure',
                 'missing historical baseline assigned a noiseless distance')
    if 'success' in detail:require(detail['success']==bool(fit['success'] and baseline_ok),'noiseless distance success flag')
    return dict(available=True,historical_baseline_available=baseline_ok)


def audit_trial(trial):
    config=trial['config'];history=_obs(trial['history']);models=trial['measurement_models']
    hm,fm,jm=(_meter(models[k]) for k in ('hm','fm','jm'))
    require(trial['source_interface']==1 and not trial['global_AC_MLE_certified'] and not trial['statistical_risk_calibrated'],'trial scope/source')
    assets=asset_unit_audit(trial)
    require(len(set(trial['seeds'].values()))==5,'random stream seeds not distinct')
    streams=np.random.SeedSequence(trial['seed']).spawn(5)
    expected_seeds={k:int(s.generate_state(1)[0]) for k,s in zip(('asset','background','source','noise','policy'),streams)}
    require(trial['seeds']==expected_seeds,'independent stream split differs from saved seed')
    require(not trial['proposal']['uses_truth'],'candidate bank uses truth')
    candidates={r['name']:RadialCandidate.from_supports(r['name'],history.n,r['supports']) for r in trial['proposal']['bank']}
    require(len(candidates)==4,'four retained reduced candidates');names=list(candidates)
    hfits={r['name']:r for r in trial['historical_fits']}
    hist=replay_fits(trial['historical_fits'],candidates,history,hm,trial['historical_failures'],
                    complete=not trial['historical_failures'],config=config,domain=trial['domain'])
    # Independent reconstruction of the saved historical noisy measurement.
    truth=trial['truth'];h=config['history'];tc=RadialCandidate.from_supports('truth',3,truth['family'])
    physicalH=forward_physics(tc,truth['r'],truth['x'],arr(truth['p'])[:h],arr(truth['q'])[:h],arr(truth['source_p'])[:h],arr(truth['source_q'])[:h],arr(truth['root_v'])[:h])
    for k in CHANNELS:
        observed=getattr(history,k)
        if observed is not None:close(observed,physicalH[k]+np.asarray(getattr(hm,k+'_sigma'))*arr(truth['white'][k])[:h]
            +(getattr(hm,'bias_'+k) if k in ('p','q','v') else 0),'true historical measurement:'+k,atol=2e-10)
    repeats=arr(config['public_voltage_repeat_counts'])[:,None]
    for k in ('p','q','v'):
        for target in (fm,jm):close(np.broadcast_to(getattr(target,'bias_'+k),(3,)),
            np.broadcast_to(getattr(hm,'bias_'+k),(3,)),'shared fixed calibrated bias:'+k,atol=1e-12)
    close(fm.v_sigma,np.broadcast_to(arr(hm.v_sigma)/np.sqrt(repeats),(config['horizon'],3)),'public voltage quality')
    for k in ('p','q','master_p','master_q'):
        target=arr(getattr(fm,k+'_sigma'))
        close(target,np.broadcast_to(arr(getattr(hm,k+'_sigma'))/np.sqrt(config['other_future_repeat_count']),target.shape),'other public meter repeats:'+k)
    for k in CHANNELS:
        if getattr(history,k) is None:continue
        shape=np.asarray(getattr(history,k)).shape;future_shape=(config['horizon'],)+shape[1:]
        close(getattr(jm,k+'_sigma'),np.concatenate((np.broadcast_to(getattr(hm,k+'_sigma'),shape),
            np.broadcast_to(getattr(fm,k+'_sigma'),future_shape))), 'joint meter concatenation:'+k)
    selection=trial['selection'];require(not selection['uses_truth'] and not selection['uses_future_noise'],'selector information boundary')
    rows={r['method']:r for r in trial['rows']};require(set(rows)==set(METHODS),'all method/failure denominators')
    saved_states=hist['replayed_states']; pred_states=0;unique_pred={};predicted={}
    for name,rec in selection.get('predicted_AC',{}).items():
        if rec is None:continue
        audit=predicted_audit(rec,history,hm,fm,jm,candidates,hfits,config,trial['domain'])
        predicted[name]=audit;pred_states+=audit['replayed_fit_states']
        key=control_key(rec['power_kw'])
        if key in unique_pred:require(identity(rec)==identity(unique_pred[key]),'same predicted action/cache changes records')
        else:unique_pred[key]=rec
    saved_states+=pred_states
    if unique_pred:
        require(selection['unique_predicted_actions']==len(unique_pred),'unique predicted action denominator')
        require(selection['predicted_fit_calls']==sum(r['fit_calls'] for r in unique_pred.values()),'actual predicted fit calls')
        require(selection['predicted_ac_calls']==sum(r['ac_calls'] for r in unique_pred.values()),'actual predicted AC calls')
    context=selection.get('context_models',[]);ordered=[(a,b) for a in names for b in names if a!=b]
    if context:
        require([(m['a'],m['b']) for m in context]==ordered,'all 12 proxy tasks')
        for m in context:model_audit(m,config);history_reference_audit(m,history,hm,hfits,config)
    operations={};exchanges={};checks={};QP_states=0;master_calls=0;check_ac=0;recenter_ac=0
    if selection.get('constant_operation',{}).get('feasible'):
        operations['constant']=replay_operation_record(selection['constant_operation'],service_config(config,'charge_fixed'))
    for name,rec in selection['methods'].items():
        cfg=service_config(config,name)
        if name not in DESIGNS:
            if rec.get('operation',{}).get('feasible'):operations[name]=replay_operation_record(rec['operation'],cfg)
            continue
        if rec.get('operation_initial',{}).get('feasible'):replay_operation_record(rec['operation_initial'],cfg)
        if rec.get('operation') and rec['operation'].get('feasible'):
            operations[name]=replay_operation_record(rec['operation'],cfg)
        local=rec.get('recenter_models') or context
        rounds=[]
        if rec.get('before_recenter') is not None:rounds.append((context,rec['before_recenter']))
        if rec.get('raw') is not None:rounds.append((local,rec['raw']))
        exchanges[name]=[]
        for ms,raw in rounds:
            audit=replay_exchange(ms,raw,config,cfg,{n:r['loss'] for n,r in hfits.items()})
            exchanges[name].append(audit);QP_states+=audit.get('QP_states',0);master_calls+=audit.get('master_calls',0)
        for index,checkrows in enumerate(rec.get('proxy_AC_checks',[])):
            ms,raw=rounds[index];best=raw['evaluations'][raw['best_evaluation_index']]
            require([(r['a'],r['b']) for r in checkrows]==ordered,'same-witness all pair coverage')
            audits=[replay_same_witness(m,raw['u'],q,r['check'],history,hm,fm,hfits,candidates)
                    for m,q,r in zip(ms,best['qp_results'],checkrows)]
            checks.setdefault(name,[]).append(audits);check_ac+=3*len(audits)
        if rec.get('recentered'):
            require(rec.get('before_recenter') is not None and rec.get('recenter_models') is not None,'recenter loses prior model/bounds')
            for old,new in zip(context,local):
                for k in ('physical_lower','physical_upper'):close(old['domain'][k],new['domain'][k],'recenter original domain unchanged')
                if new.get('expanded_at_absolute') is not None:
                    require(new['new_proxy_requires_new_bounds'],'recenter reuses previous proxy bounds')
                    previous=rec['before_recenter'];before=previous['evaluations'][previous['best_evaluation_index']]
                    index=next(i for i,m in enumerate(context) if m['name']==old['name'])
                    close(new['expanded_at_absolute'],before['qp_results'][index]['physical_parameter'],'recenter anchor comes from preceding proxy QP')
                    close(new['expanded_at_control'],previous['u'],'recenter control comes from preceding MILP')
                    history_reference_audit(new,history,hm,hfits,config)
                    recenter_ac+=new['expansion_ac_calls']
        guard=rec.get('guard') or {}
        if 'gain' in guard:
            base=selection['predicted_AC']['constant'];raw=selection['predicted_AC'].get(name+'__raw')
            complete=base['success'] and raw is not None and raw['success']
            require(guard['complete']==complete,'AC guard completeness')
            proxy_pass=all(r['physical_box_feasible'] and r['voltage_valid'] and r['max_channel_RMS']<=config['proxy_channel_RMS_gate']+1e-8
                           for r in checks[name][-1])
            require(guard['proxy_same_witness_gate']==proxy_pass,'proxy AC gate')
            if complete:
                gain=raw['mean_added_score']-base['mean_added_score'];close(guard['gain'],gain,'AC guard gain')
                require(guard['accepted']==bool(proxy_pass and gain>guard['resolution']),'AC gain+proxy acceptance')
            else:require(not guard['accepted'],'incomplete AC guard accepted')
            selected=rec['raw']['u'] if guard['accepted'] else base['power_kw']
            close(rows[name]['power_kw'],np.asarray(selected).reshape(config['horizon'],3),'guard-selected realized curve')
    eval_unique={};mean_unique={};eval_saved=0;mean_saved=0;physical_replays=0;row_audits=[]
    true_names=[n for n,c in candidates.items() if tuple(s for s in c.supports if 1<len(s)<3)==tuple(map(tuple,truth['family']))]
    for name,row in rows.items():
        out=dict(method=name,status=row['status'],candidate_miss=bool(row['candidate_miss']))
        require(row['candidate_miss']==(len(true_names)!=1),'candidate miss label')
        if row.get('physical') is not None:out['physical']=future_audit(trial,row,fm);physical_replays+=1
        if row.get('observations') is not None and 'fits' in row:
            obs=_obs(row['observations']);audit=replay_fits(row['fits'],candidates,concat_observations(history,obs),jm,row['failures'],
                complete=row['status']=='complete',config=config,domain=trial['domain']);eval_saved+=audit['replayed_states'];out['fits']=audit
            key=control_key(row['power_kw'])
            if row['cache_hit']:
                require(key in eval_unique and row['fits']==eval_unique[key]['fits'],'actual future cache mismatch')
            else:require(key not in eval_unique,'duplicate future action refitted');eval_unique[key]=row
            if row['status']=='complete':
                losses={f['name']:f['loss'] for f in row['fits']};require(losses==row['losses'],'evaluation losses')
                ranked=sorted(losses,key=lambda n:(losses[n],n));require(row['candidate_ranking']==ranked and row['winner']==ranked[0],'candidate rank')
                for a,b in ordered:close(row['pair_signed_loss_differences'][a+'__to__'+b],.5*(losses[b]-losses[a]),'actual pair signed LL difference')
                if len(true_names)==1:
                    true=true_names[0];margin=min(v for n,v in losses.items() if n!=true)-losses[true]
                    close(row['margin'],margin,'true signed SSE margin');close(row['loglik_margin'],.5*margin,'true LL margin')
                    fit=next(f for f in row['fits'] if f['name']==true)
                    for k in ('r','x'):close(row['true_impedance_'+k.upper()+'_relative'],np.linalg.norm(arr(fit[k])-arr(truth[k]))/np.linalg.norm(arr(truth[k])),'impedance relative error')
                    out['margin']=margin
        mean=row.get('noiseless_AC_refit')
        if mean:
            obs=_obs(mean['future_mean_observations']);zero=arr(truth['white']['p'])[h:]*0
            phyrow=dict(row);phyrow['observations']=None;future_audit(trial,phyrow,fm)
            for k in CHANNELS:
                v=getattr(obs,k)
                if v is not None:close(v,arr(row['physical'][k])+(getattr(fm,'bias_'+k) if k in ('p','q','v') else 0),'noiseless future mean:'+k,atol=2e-10)
            audit=replay_fits(mean['fits'],candidates,concat_observations(history,obs),jm,mean['failures'],
                complete=mean['all_candidates'],config=config,domain=trial['domain']);mean_saved+=audit['replayed_states']
            key=control_key(row['power_kw'])
            if key in mean_unique:require(mean['fits']==mean_unique[key]['fits'],'noiseless duplicate curve differs')
            else:mean_unique[key]=mean
            for detail in mean['candidate_distances']:
                noiseless_distance_audit(detail,mean['fits'],hfits,obs,fm,h)
            require([r['candidate'] for r in mean['candidate_distances']]==names,'noiseless distance candidate denominator')
        row_audits.append(out)
    saved_states+=eval_saved+mean_saved
    fit_calls=trial['historical_cost']['fit_calls']+sum(r['fit_calls'] for r in unique_pred.values())+sum(r['cost']['fit_calls'] for r in eval_unique.values())+sum(r['cost']['fit_calls'] for r in mean_unique.values())
    ac_calls=trial['historical_cost']['ac_calls']+sum(r['ac_calls'] for r in unique_pred.values())+sum(r['cost']['ac_calls'] for r in eval_unique.values())+sum(r['cost']['ac_calls'] for r in mean_unique.values())
    context_calls=(selection.get('context') or {}).get('ac_calls',0);ac_calls+=context_calls+check_ac+recenter_ac
    return dict(seed=trial['seed'],phase=trial['phase'],success=True,historical=hist,asset_unit_audit=assets,
        saved_fit_states_replayed=saved_states,predicted_saved_states=pred_states,
        unique_predicted_actions=len(unique_pred),unique_predicted_fit_states=sum(r['returned_fit_states'] for r in unique_pred.values()),
        actual_candidate_fit_calls=fit_calls,actual_counted_AC_calls=ac_calls,
        actual_AC_breakdown=dict(history_fits=trial['historical_cost']['ac_calls'],context=context_calls,
            predicted=sum(r['ac_calls'] for r in unique_pred.values()),same_witness=check_ac,recenter=recenter_ac,
            actual_future_fits=sum(r['cost']['ac_calls'] for r in eval_unique.values()),
            noiseless_future_fits=sum(r['cost']['ac_calls'] for r in mean_unique.values())),
        QP_states_replayed=QP_states,nonzero_MILP_master_calls=master_calls,
        physical_future_states_replayed=physical_replays,unique_actual_actions=len(eval_unique),unique_noiseless_actions=len(mean_unique),
        independent_histories=1,formal_calibration_histories=0,formal_rejection_evaluation_histories=0,
        operations=operations,exchanges=exchanges,same_witness=checks,methods=row_audits,
        scope='saved-state arithmetic audit; no optimizer, new fit or statistical calibration')


def review_directory(path, output=None):
    directory=Path(path).resolve();freeze=read_json(directory/'freeze.json');sources=source_audit(directory,freeze)
    planned=freeze['seeds'];trials=[];errors=[];seen=set();statuses=Counter();hashes=[]
    for file in sorted(directory.glob('trial_*.json.gz')):
        trial=read_json(file);seed=trial['seed']
        require(seed in planned and seed not in seen,'unplanned or duplicate independent history');seen.add(seed)
        require(trial['config']==freeze['config'] and trial['source_sha256']==freeze['source_sha256']
                and trial['phase']==freeze['phase'] and trial['domain']==freeze['domain'],'trial/freeze identity')
        try:
            audit=audit_trial(trial);trials.append(audit)
            statuses.update(r['status'] for r in audit['methods'])
        except (ValueError,KeyError,RuntimeError,IndexError,TypeError) as error:
            errors.append(dict(seed=seed,error=type(error).__name__+':'+str(error)))
        hashes.append(dict(name=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    config=freeze['config'];require(not set(config['pilot_seeds'])&set(config['evaluation_seeds']),'pilot/evaluation seeds overlap')
    result=dict(success=not errors,phase=freeze['phase'],domain=freeze['domain'],source_audit=sources,
        planned_independent_histories=len(planned),available_independent_histories=len(seen),audited_histories=len(trials),
        missing_histories=[s for s in planned if s not in seen],review_errors=errors,trial_hashes=hashes,
        method_conditions=len(seen)*len(METHODS),saved_method_status_counts=dict(statuses),
        totals={k:sum(t[k] for t in trials) for k in ('saved_fit_states_replayed','actual_candidate_fit_calls',
            'actual_counted_AC_calls','QP_states_replayed','nonzero_MILP_master_calls','physical_future_states_replayed')},
        formal_calibration_histories=0,formal_rejection_evaluation_histories=0,
        calibration_rule=config.get('statistical_rejection_rule'),trials=trials,
        reviewer=dict(file=Path(__file__).name,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            science_source=False,no_new_fit=True,no_selector=True,no_data_generator=True),
        scope='independent replay of declared simulation; held-out histories are not risk-calibrated rejection trials')
    if output is not None:write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();out=review_directory(args.directory,args.output)
    print(json.dumps({k:out[k] for k in ('success','phase','available_independent_histories','audited_histories','review_errors','totals')}))


if __name__=='__main__':main()
