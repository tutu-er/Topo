"""Observation-only RNJ/WZZT proposals -> AC evidence -> targeted EV -> refit.

Larger synthetic steady-state snapshots, not measured feeder data or averages.
Simulation truth is held outside prepare/choose/update and used only to evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'rnj_wzzt_core'))
import numpy as np
import scipy
from scipy.optimize import lsq_linear
import ac_general_v9 as ac
from research_experiments.mobile_ev_probing import passive_bridge_v9 as bridge

KAPPA = .5
VMIN, VMAX = .9, 1.1
POWER_KW = 20.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v))+'\n', encoding='utf-8')


def parameter_box(candidate):
    return tuple((.002, .014) if len(s) == candidate.n else
                 ((.03, .10) if len(s) == 1 else (1e-6, .04)) for s in candidate.supports)


def physical_case(n, weak=False):
    supports = [tuple(range(j, j+2)) for j in range(0, n, 2)]
    supports += [tuple(range(j, j+4)) for j in range(0, n-3, 4)]
    c = ac.RadialCandidate.from_supports('evaluation_truth', n, supports)
    theta = np.array([.008 if len(s) == n else
                      (.052+.004*(s[0] % 4) if len(s) == 1 else
                       (1e-5 if weak and s == (0, 1) else (.020 if len(s) == 2 else .009)))
                      for s in c.supports])
    return c, theta


def action(n, kind, sites):
    increments = np.zeros((24, n))
    trip = np.zeros(24)
    if kind == 'none':
        connection = np.full(24, -1)
        power = np.zeros(24)
    elif kind in ('constant', 'endpoint'):
        power = np.full(24, .225) if kind == 'constant' else np.tile([0., .3, .3, .3], 6)
        connection = np.full(24, sites[0], int)
        increments[:, sites[0]] = power
    elif kind == 'mobile':
        power = np.r_[np.tile([.3,.3,.3,0.],2),.3,.3,.3,0.,0.,
                      np.tile([.3,.3,.3,0.],2),.3,.3,.3]
        connection = np.r_[np.full(11, sites[0]), [-1,-1], np.full(11, sites[1])].astype(int)
        for t, j in enumerate(connection):
            if j >= 0:
                increments[t,j] = power[t]
        trip[11:13] = 1.
    else:
        raise ValueError(kind)
    soc = .4 + np.cumsum(.95*power*POWER_KW*.25-trip)/60
    assert ((soc >= .1) & (soc <= .9)).all()
    energy = float(power.sum()*POWER_KW*.25)
    assert kind == 'none' or abs(energy-27.) < 1e-10
    return increments, dict(kind=kind, sites=sites, power_kw=power*POWER_KW,
                           connection=connection, energy_kwh=energy,
                           travel_kwh=float(trip.sum()), soc=soc)


def meter_settings(mode):
    if mode == 'gain_dominated':
        beta, jitter = .0008/VMAX, .0002
    elif mode == 'jitter_dominated':
        beta, jitter = 0., .001
    elif mode == 'precision_control':
        beta, jitter = .00008/VMAX, .00002
    else:
        raise ValueError(mode)
    return dict(beta=beta, jitter=jitter, total_bound=beta*VMAX+jitter,
                gain_bounds=(1-beta, 1+beta), root_jitter=jitter,
                terminal_jitter=jitter, vmax=VMAX)


def simulate(n, weak, history_mode, meter_mode, seed, kind='none', sites=(0,)):
    """Seed pairing preserves physical backgrounds and meter draws across actions."""
    rng = np.random.default_rng(seed)
    c, theta = physical_case(n, weak)
    hlen = 24 if n == 6 else 36
    total = hlen+24
    common = .014*np.sin(np.linspace(0,4*np.pi,total))[:,None]
    background = .12+common+rng.uniform(-.065,.065,(total,n))
    if history_mode == 'partial':
        # The first pair has nearly indistinguishable natural trajectories.
        background[:hlen,1] = background[:hlen,0] + rng.uniform(-.002,.002,hlen)
    elif history_mode != 'rich':
        raise ValueError(history_mode)
    # Future varies, but selection receives only an historical mean forecast.
    background[hlen:] = .12+common[hlen:]+rng.uniform(-.012,.012,(24,n))
    q = background*(.30+rng.uniform(-.04,.04,(total,n)))
    increments, device = action(n,kind,sites)
    p = background.copy()
    p[hlen:] += increments
    source = .12+rng.uniform(-.02,.02,total)
    y, voltage, residual = ac.ac_batch(c,theta,p,q,source,.3*source,kappa=KAPPA)
    vm = abs(voltage[:,c.terminal_edges])
    settings = meter_settings(meter_mode)
    gains = 1+rng.uniform(-settings['beta'],settings['beta'],n+1)
    jitter = rng.uniform(-settings['jitter'],settings['jitter'],(total,n+1))
    vhat = gains[None,1:]*vm+jitter[:,1:]
    root_hat = gains[0]+jitter[:,0]
    yhat = root_hat[:,None]**2-vhat**2
    # Spec-inspired bounds, with absolute floors, not claimed field calibration.
    pcap = .02*abs(p)+.0005
    qcap = .025*np.hypot(p,q)+.0005
    p_fraction = .8*rng.uniform(-1,1,(1,n))+.2*rng.uniform(-1,1,(total,n))
    q_fraction = .8*rng.uniform(-1,1,(1,n))+.2*rng.uniform(-1,1,(total,n))
    phat, qhat = p+pcap*p_fraction, q+qcap*q_fraction
    bp = (.02*abs(phat)+.0005)/.98
    bq = (.025*(abs(phat)+bp+abs(qhat))+.0005)/.975
    delta = settings['total_bound']
    eps = 2*abs(root_hat[:,None])*delta+delta**2+2*abs(vhat)*delta+delta**2
    obs = dict(P=phat,Q=qhat,Y_drop=yhat,P_error=bp,Q_error=bq,
               drop_error_bound=eps,source_nominal_eff=np.full(total,.138),
               source_error_eff=np.full(total,.115),source_p_abs_cap=np.full(total,.22),
               source_q_abs_cap=np.full(total,.066),batch_id=np.r_[np.zeros(hlen,int),np.ones(24,int)])
    assert np.all(abs(phat-p) <= bp) and np.all(abs(qhat-q) <= bq)
    assert np.all(abs(yhat-y) <= eps)
    assert abs(voltage).min() >= VMIN and abs(voltage).max() <= VMAX
    truth = dict(candidate=c.metadata(),theta=theta,P=p,Q=q,Y=y,source_p=source,
                 voltage_terminal=vm,voltage_measured=vhat,root_voltage_measured=root_hat,
                 meter_gains=gains,meter_jitter=jitter,meter_settings=settings,
                 min_voltage=float(abs(voltage).min()),max_voltage=float(abs(voltage).max()),
                 equation_residual=residual,history_count=hlen,device=device,seed=seed)
    return obs,truth


def subset(obs, rows):
    return {k:np.asarray(v)[rows] for k,v in obs.items()}


def gains_argument(settings):
    return {k:settings[k] for k in ('gain_bounds','root_jitter','terminal_jitter','vmax')}


def screen(history, proposals, settings, model='shared_gain'):
    records = {}
    for row in proposals['candidates']:
        c = ac.RadialCandidate.from_supports(row['id'],proposals['n_terminal'],row['supports'])
        kwargs = dict(shared_voltage_gain=gains_argument(settings)) if model == 'shared_gain' else {}
        records[row['id']] = ac.solve_outer(history,c,box=parameter_box(c),**kwargs)
    retained = [row['id'] for row in proposals['candidates'] if records[row['id']]['status'] != 'excluded']
    return dict(model=model,records=records,retained=retained,
                status='candidate_bank_inconsistent' if not retained else
                       ('conditional_bank_singleton' if len(retained)==1 else 'unresolved'))


def _proxy_loss(p,q,y,candidate):
    """Cheap nominal design surrogate; no AC exclusion or bound semantics."""
    effective = p+KAPPA*q
    # Spatial centering removes linear common source; time centering removes
    # an additive calibration offset in this design approximation only.
    center = np.eye(candidate.n)-np.ones((candidate.n,candidate.n))/candidate.n
    target = y@center
    target -= target.mean(axis=0)
    columns=[]
    for e in range(candidate.m):
        v = (effective@candidate.M[:,e])[:,None]*candidate.M[:,e][None,:]@center
        v -= v.mean(axis=0)
        columns.append(v.ravel())
    matrix=np.array(columns).T
    box=np.array(parameter_box(candidate))
    fit=lsq_linear(matrix,target.ravel(),bounds=(box[:,0],box[:,1]),method='bvls',max_iter=100)
    return float(np.sum((matrix@fit.x-target.ravel())**2))


def choose_probe(history,proposals,evidence,max_references=2):
    """Budgeted joint-refit T surrogate; final exclusion ALWAYS uses AC LP.

    Uses only history and a mean-load forecast; no evaluation truth/future noise.
    AC reference fits use ordinary meter boxes. They need not be compatible with
    the stronger shared-gain model, so the resulting score is explicitly heuristic.
    """
    began=perf_counter()
    rows=[r for r in proposals['candidates'] if r['id'] in evidence['retained']]
    if len(rows)<=1:
        return dict(status='skip_EV' if rows else 'inconsistent',seconds=perf_counter()-began)
    candidates={r['id']:ac.RadialCandidate.from_supports(r['id'],proposals['n_terminal'],r['supports']) for r in rows}
    references={}
    for row in rows[:max_references]:
        c=candidates[row['id']]
        fit=ac.fit_ac_reference(history,c,box=parameter_box(c),max_nfev=40,
                                outer_result=evidence['records'][row['id']])
        if fit.get('theta') is not None:
            references[row['id']]=fit
    if not references:
        return dict(status='compute_before_dispatch',references=references,seconds=perf_counter()-began)
    n=proposals['n_terminal']
    forecast_p=np.tile(history['P'].mean(axis=0),(24,1))
    forecast_q=np.tile(history['Q'].mean(axis=0),(24,1))
    designs=[('endpoint',(j,)) for j in range(n)]
    designs += [('mobile',pair) for pair in itertools.combinations(range(n),2)]
    old_loss={key:_proxy_loss(history['P'],history['Q'],history['Y_drop'],c) for key,c in candidates.items()}
    scores=[]
    for kind,sites in designs:
        inc,_=action(n,kind,sites)
        values=[]
        for ref,fit in references.items():
            pred,_,_=ac.ac_batch(candidates[ref],fit['theta'],forecast_p+inc,forecast_q,.12,.036)
            joined=[np.concatenate([history[k],new]) for k,new in
                    [('P',forecast_p+inc),('Q',forecast_q),('Y_drop',pred)]]
            losses=[max(0.,_proxy_loss(*joined,c)-old_loss[h]) for h,c in candidates.items() if h!=ref]
            values.append(min(losses))
        scores.append(dict(kind=kind,sites=sites,score=float(np.mean(values))))
    selected={kind:max((s for s in scores if s['kind']==kind),key=lambda s:s['score']) for kind in ('endpoint','mobile')}
    return dict(status='scored',selected=selected,scores=scores,references=references,
                seconds=perf_counter()-began,forecast='historical meter means, not realized future',
                interpretation='nominal AC references plus centered linear joint-refit loss; heuristic action ranking, not validated T optimum or certificate')


def family_metrics(supports,truth_supports,n):
    a={tuple(s) for s in supports if 1<len(s)<n}
    b={tuple(s) for s in truth_supports if 1<len(s)<n}
    return dict(exact=a==b,precision=len(a&b)/len(a) if a else 0.,
                recall=len(a&b)/len(b) if b else 1.,f1=2*len(a&b)/(len(a)+len(b)) if a or b else 1.)


def run_case(config,output,bank_size=12,wzzt_seconds=2.):
    began=perf_counter()
    n,weak,history_mode,meter_mode,seed=config
    name=f'n{n}_{"weak" if weak else "ordinary"}_{history_mode}_{meter_mode}_{seed}'
    output=Path(output); folder=output/name; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        raise FileExistsError(name)
    obs,truth=simulate(*config)
    hlen=truth['history_count']; history=subset(obs,slice(0,hlen)); settings=meter_settings(meter_mode)
    proposals=bridge.build_passive_candidates(history['P'],history['Q'],history['Y_drop'],
        source_p_nominal=.12,source_q_nominal=.036,source_p_error=.1,source_q_error=.03,
        max_candidates=bank_size,wzzt_time_limit=wzzt_seconds,wzzt_max_atoms=n+2,
        coefficient_bound=.14,neighbor_seed_count=3)
    dump(folder/'passive.json',proposals)
    evidence=screen(history,proposals,settings)
    dump(folder/'historical_evidence.json',evidence)
    design=choose_probe(history,proposals,evidence)
    dump(folder/'design.json',design)
    policies={'no_EV':('none',(0,))}
    if design['status']=='scored':
        endpoint=design['selected']['endpoint']; mobile=design['selected']['mobile']
        policies.update(selected_single=('endpoint',tuple(endpoint['sites'])),
                        matched_constant=('constant',tuple(endpoint['sites'])),
                        selected_mobile=('mobile',tuple(mobile['sites'])),
                        fixed_single=('endpoint',(0,)),fixed_mobile=('mobile',(0,1)))
    records={}; truth_records={}; saved={}; summaries={}
    true_candidate,theta=physical_case(n,weak)
    for label,(kind,sites) in policies.items():
        key=kind+'_'+'_'.join(map(str,sites))
        if key not in saved:
            actual,actual_truth=simulate(*config,kind,sites)
            file=folder/(key+'.npz'); np.savez_compressed(file,**actual)
            dump(folder/(key+'_truth.json'),actual_truth)
            saved[key]=file.name
            records[key]={}
            truth_records[key]={}
            for model in ('independent_box','shared_gain'):
                update=screen(actual,proposals,settings,model)
                records[key][model]=update
                kwargs=dict(shared_voltage_gain=gains_argument(settings)) if model=='shared_gain' else {}
                # Post-hoc model/domain audit only: never inserted into selector bank.
                truth_records[key][model]=ac.solve_outer(actual,true_candidate,box=parameter_box(true_candidate),**kwargs)
        summaries[label]=dict(action=key,**{model:dict(
            retained_count=len(records[key][model]['retained']),
            retained_ids=records[key][model]['retained'],
            excluded_count=proposals['retained_count']-len(records[key][model]['retained']))
            for model in records[key]})
    metrics={r['id']:family_metrics(r['supports'],true_candidate.supports,n) for r in proposals['candidates']}
    result=dict(config=config,case=name,proposals=proposals,historical_evidence=evidence,design=design,
                records=records,true_model_evaluation_only=truth_records,policies=summaries,
                candidate_truth_metrics=metrics,truth_in_bank=any(v['exact'] for v in metrics.values()),
                passive_best_metrics=metrics[proposals['candidates'][0]['id']],
                inputs={p.name:sha(p) for p in folder.glob('*.npz')},
                truth_files={p.name:sha(p) for p in folder.glob('*_truth.json')},
                seconds=perf_counter()-began)
    assert all(r['status']!='excluded' for action_results in truth_records.values() for r in action_results.values())
    dump(folder/'result.json',result)
    print(json.dumps(dict(case=name,seconds=result['seconds'],truth_in_bank=result['truth_in_bank'],
                         passive_f1=result['passive_best_metrics']['f1'],historical_retained=len(evidence['retained']),
                         policies={k:v['shared_gain']['retained_count'] for k,v in summaries.items()})),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true'); parser.add_argument('--case-index',type=int)
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    cases=[(6,False,'partial','gain_dominated',9001)] if args.pilot else [
        (n,weak,history,meter,seed)
        for n,weak,history in [(6,False,'rich'),(6,False,'partial'),(6,True,'partial'),(12,False,'partial')]
        for meter in ('gain_dominated','jitter_dominated') for seed in (9101,)]
    protocol=dict(cases=cases,bank_size=12,wzzt_seconds_per_solve=2.,wzzt_max_atoms='n+2',
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(ac.__file__),Path(bridge.__file__)]},
        seed_role='9001 exploratory;9101 frozen mechanism evaluation',
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        scope='6/12terminal synthetic balanced constant-power steady snapshots; finite observation-generated bank; not field statistical accuracy',
        model='known X/R=.5, common-source junction, root physical magnitude1; n+2 WZZT atoms is a proposal compute budget, not certification restriction')
    if args.case_index is None:
        if (output/'protocol.json').exists(): raise FileExistsError('fresh output folder required')
        dump(output/'protocol.json',protocol)
        results=[run_case(c,output) for c in cases]
        dump(output/'results.json',dict(protocol=protocol,results=results))
    else:
        saved=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
        assert saved['source_sha256']==protocol['source_sha256']
        run_case(saved['cases'][args.case_index],output)


if __name__=='__main__': main()
