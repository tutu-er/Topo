"""Frozen small-local experiment; truth is read only after inference is saved."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
import sys
import platform
import numpy as np
import scipy
from scipy.stats import norm

from .. import likelihood as lk
from ..label_free_bounds.bounds import TreeSpec, PhysicalDomain
from ..reference_benchmark.representation import physical_tree_ac
from .cases import make_case, candidate_bank
from .fitting import fit, evaluate, profile_information
from .rejection import certify_observed_candidate, bootstrap_pvalue


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, default=_json, ensure_ascii=False, allow_nan=True,
                      separators=(',', ':')).encode('utf-8')
    if path.suffix == '.gz':
        with path.open('wb') as handle:
            handle.write(gzip.compress(body, mtime=0))
    else:
        path.write_bytes(body)


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def task_key(spec):
    return f"{spec['network']}_{spec['regime']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def specifications():
    return [dict(network=name, regime=regime, seed=seed, N=n, boundary=boundary)
            for name in ('paper15', 'paper11') for regime in ('ordinary', 'correlated')
            for seed in (31001, 31002, 31003) for n in (96, 288) for boundary in ('K', 'U')]


def _sigma(case):
    return {k: np.asarray(case['meter']['sigma_'+k]) for k in ('p', 'q', 'v')}


def _theta(r, x):
    return np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]


def _initial_from(fitted, obs, sigma, root_known, *, fixed_pq=False):
    n, count = np.shape(obs['p'])[1], len(obs['p'])
    if fixed_pq:
        local = np.empty((count, 0))
    else:
        local = np.c_[(np.asarray(fitted['p'])-obs['p'])/sigma['p'],
                       (np.asarray(fitted['q'])-obs['q'])/sigma['q']]
    if root_known is None:
        local = np.c_[local, (np.asarray(fitted['root_v'])-1.)/.01]
    return dict(theta=fitted['theta'], local=local)


def _audit_truth(case, spec, inference):
    """Evaluation-only calculations, after observed inference has been sealed."""
    truth, sigma, n = case['truth'], _sigma(case), spec['N']
    domain, bank = case['domain'], {c.name:c for c in case['candidates']}
    obs = {k:np.asarray(case['observed'][k])[:n] for k in ('p', 'q', 'v')}
    root_known = np.asarray(truth['root_v'])[:n] if spec['boundary'] == 'K' else None
    real = dict(p=np.asarray(truth['p'])[:n], q=np.asarray(truth['q'])[:n], v=np.asarray(truth['v'])[:n])
    theta = _theta(np.asarray(truth['r']), np.asarray(truth['x']))
    local = np.c_[(real['p']-obs['p'])/sigma['p'], (real['q']-obs['q'])/sigma['q']]
    if root_known is None:
        local = np.c_[local, (np.asarray(truth['root_v'])[:n]-1.)/.01]
    derivatives = evaluate(bank['star'], obs, sigma, theta, local, root_known=root_known)
    ideal_info = profile_information(derivatives['shared'], derivatives['state'], theta)
    audits = dict(reference_information_at_truth=ideal_info, truth_inputs_to_inference=False,
                  truth_read_only_after_sealed_inference=True, candidates={})
    # Independent branch implementation at physical truth.
    tree = truth['physical_tree']
    physical = physical_tree_ac(tree['branches'], tree['observation_bus_ids'], tree['root_bus'],
        tree['base_kv'], tree['base_kw'], np.asarray(truth['p'])*tree['base_kw'],
        np.asarray(truth['q'])*tree['base_kw'], tree['source_bus'], 0., 0., truth['root_v'])
    audits['independent_truth_voltage_max_error'] = float(np.max(abs(physical['v']-truth['v'])))
    train = inference['fits'].get('star', {})
    for name, fitted in inference['fits'].items():
        if 'theta' not in fitted:
            audits['candidates'][name] = dict(status='missing_fit'); continue
        r, x, c = np.asarray(fitted['r']), np.asarray(fitted['x']), bank[name]
        entry = dict(qualified=fitted['qualified'])
        if name == 'star':
            rt, xt = np.asarray(truth['r']), np.asarray(truth['x'])
            se = np.asarray(fitted['information']['physical_standard_errors'])
            entry.update(r_relative_l2=float(np.linalg.norm(r-rt)/np.linalg.norm(rt)),
                         x_relative_l2=float(np.linalg.norm(x-xt)/np.linalg.norm(xt)),
                         r_branch_relative=(abs(r-rt)/rt).tolist(),
                         x_branch_relative=(abs(x-xt)/xt).tolist(),
                         r_leaf_relative_l2=float(np.linalg.norm(r[1:]-rt[1:])/np.linalg.norm(rt[1:])),
                         x_leaf_relative_l2=float(np.linalg.norm(x[1:]-xt[1:])/np.linalg.norm(xt[1:])),
                         r_error_ohm=((r-rt)*case['nominal']['zbase_ohm']).tolist(),
                         x_error_ohm=((x-xt)*case['nominal']['zbase_ohm']).tolist(),
                         local_95_interval_contains_truth=(abs(np.r_[r-rt, x-xt])<=1.96*se).tolist())
        # Same true P/Q and cut voltage: diagnosis of the fitted line response.
        pred = lk.ac_predict(c, r, x, truth['p'], truth['q'], root_v=truth['root_v'])['v']
        error = (pred-np.asarray(truth['v']))/sigma['v']
        entry['true_PQ_voltage_RMSE_sigma_train'] = float(np.sqrt(np.mean(error[:n]**2)))
        entry['true_PQ_voltage_RMSE_sigma_holdout'] = float(np.sqrt(np.mean(error[288:384]**2)))
        entry['fixed_two_mean_error_reference_holdout'] = float(norm.cdf(-np.linalg.norm(error[288:384])/2))
        if 'theta' in train:
            trpred = lk.ac_predict(bank['star'], train['r'], train['x'], truth['p'], truth['q'],
                                  root_v=truth['root_v'])['v']
            er = ((trpred-np.asarray(truth['v']))/sigma['v'])[288:384]
            ek = ((pred-trpred)/sigma['v'])[288:384]
            left = float(np.sum(error[288:384]**2))
            components = [float(np.sum(er**2)), float(np.sum(ek**2)), float(2*np.sum(er*ek))]
            entry['decomposition_holdout'] = dict(parameter=components[0], fitted_model_difference=components[1],
                 cross=components[2], total=left, identity_error=abs(left-sum(components)))
        # Distance to the noiseless means, not used by the rejection rule.
        fixed = fit(c, real, sigma, domain, root_known=root_known,
            initial=_initial_from(fitted, real, sigma, root_known, fixed_pq=True),
            multistart=1, max_iter=160, fixed_pq=real)
        entry['clean_fixed_PQ_fit'] = fixed
        if 'theta' in fixed:
            # Starting at this exact feasible fixed-PQ explanation preserves inclusion.
            combined = dict(fixed, p=real['p'], q=real['q'])
            joint = fit(c, real, sigma, domain, root_known=root_known,
                initial=_initial_from(combined, real, sigma, root_known), multistart=1, max_iter=160)
            entry['clean_EIV_fit'] = joint
            entry['clean_local_distance_order_pass'] = bool(joint.get('loss', np.inf) <= fixed['loss']+1e-6)
            entry['PQ_correction_distance_reduction'] = float(fixed['loss']-joint.get('loss', np.nan))
        audits['candidates'][name] = entry
    # Exact structural overlap when a newly inserted pair edge may be zero.
    truth_lines = {tuple(s): complex(r,x) for s,r,x in zip(bank['star'].supports, truth['r'], truth['x'])}
    overlaps={}
    for name,c in bank.items():
        z=np.array([truth_lines.get(tuple(s),0j) for s in c.supports])
        response=lk.ac_predict(c,z.real,z.imag,truth['p'],truth['q'],root_v=truth['root_v'])['v']
        overlaps[name]=float(np.max(abs(response-np.asarray(truth['v']))))
    audits['exact_zero_new_edge_sensitivity']=dict(voltage_max_errors=overlaps,
        inside_primary_positive_domain=False,
        interpretation='exact observational overlap if added hidden edge may be zero; not a primary-domain explanation')
    return audits


def run_one(spec, output):
    begin = perf_counter()
    output = Path(output); key = task_key(spec)
    try:
        case = make_case(spec['network'], spec['regime'], spec['seed'], boundary=spec['boundary'])
        count, sigma, domain = spec['N'], _sigma(case), case['domain']
        obs = {k:np.asarray(case['observed'][k])[:count].copy() for k in ('p','q','v')}
        known = np.asarray(case['observed']['root_known'])[:count] if spec['boundary'] == 'K' else None
        sealed = dict(spec=spec, observed=obs, sigma=sigma, domain=domain, root_known=known,
                      provenance=case['provenance'], fits={}, certificates={}, truth_used=False)
        pdomain = PhysicalDomain(z_min=domain['z_min'], z_max=domain['z_max'],
            v_min=domain['v_min'], v_max=domain['v_max'], power_caps=domain['power_caps'],
            ratio_min=domain['ratio_min'], ratio_max=domain['ratio_max'])
        for j, c in enumerate(case['candidates']):
            sealed['fits'][c.name] = fit(c, obs, sigma, domain, root_known=known,
                                        seed=spec['seed']+j, max_iter=160)
            tree = TreeSpec(0, c.edges, tuple(e+1 for e in c.terminal_edges))
            sealed['certificates'][c.name] = certify_observed_candidate(tree,
                obs['p'], obs['q'], obs['v'], sigma['p'], sigma['q'], sigma['v'],
                pdomain, root_v=known, solver='scipy', solver_time_limit=10.)
        seal_path = output/'inference'/f'{key}.json.gz'
        write(seal_path, sealed)
        # No truth field is included in the inference artifact.
        truth_path = output/'truth'/f'{key}.json.gz'
        write(truth_path, case['truth'])
        audit = _audit_truth(case, spec, sealed)
        record = dict(spec=spec, status='completed', seconds=perf_counter()-begin,
                      inference_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
                      inference_file=str(seal_path.resolve()), truth_file=str(truth_path.resolve()),
                      diagnostics=audit)
        write(output/'results'/f'{key}.json.gz', record)
        return dict(spec=spec, status='completed', seconds=record['seconds'],
                    qualified=sum(f['qualified'] for f in sealed['fits'].values()))
    except Exception as error:
        record = dict(spec=spec, status='failed', seconds=perf_counter()-begin,
                      error=repr(error), traceback=traceback.format_exc())
        write(output/'results'/f'{key}.json.gz', record)
        return record


def run_bootstrap(output, repeats=39, workers=4):
    tasks = []
    for name in ('paper15','paper11'):
        for regime in ('ordinary','correlated'):
            spec = dict(network=name, regime=regime, seed=31001, N=96, boundary='K')
            for candidate in ('star','pair01','pair02','pair12'):
                tasks.append((spec, candidate))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(bootstrap_one, output, spec, c, repeats) for spec,c in tasks]
        for future in as_completed(futures):
            item=future.result();results.append(item);print(json.dumps(item),flush=True)
    write(Path(output)/'bootstrap_summary.json', results)


def bootstrap_one(output, spec, candidate, repeats):
    started=perf_counter(); key=task_key(spec)
    sealed=read(Path(output)/'inference'/f'{key}.json.gz')
    observed, sig = sealed['observed'], sealed['sigma']
    observed={k:np.asarray(v) for k,v in observed.items()}
    sig={k:np.asarray(v) for k,v in sig.items()}
    fitted=sealed['fits'][candidate]
    if not fitted.get('qualified'):
        result=dict(spec=spec,candidate=candidate,status='original_fit_unqualified',reject=False)
    else:
        bank={c.name:c for c in candidate_bank()}
        rng=np.random.default_rng(np.random.SeedSequence([spec['seed'], 170, list(bank).index(candidate),
            11 if spec['network']=='paper11' else 15, 0 if spec['regime']=='ordinary' else 1]))
        records=[]; known=np.asarray(sealed['root_known'])
        for b in range(repeats):
            boot={k:np.asarray(fitted[k])+rng.normal(size=np.shape(fitted[k]))*sig[k] for k in ('p','q','v')}
            # Same observed-only initialization, two starts and iteration budget
            # as the original statistic. No truth or calibration library.
            result_fit=fit(bank[candidate],boot,sig,sealed['domain'],root_known=known,
                multistart=2,max_iter=160,seed=spec['seed']+list(bank).index(candidate))
            records.append(dict(sse=result_fit.get('loss'),success=result_fit['qualified'],
                                stationary=result_fit['qualified'],
                                projected_gradient=result_fit.get('projected_gradient'),
                                seconds=result_fit['seconds']))
        result=dict(spec=spec,candidate=candidate,status='completed',
            observed_sse=fitted['loss'],records=records,
            decision=bootstrap_pvalue(fitted['loss'],records))
        result['reject']=bool(result['decision']['pvalue']<=.05)
    result['seconds']=perf_counter()-started
    write(Path(output)/'bootstrap'/f'{key}_{candidate}.json',result)
    return {k:v for k,v in result.items() if k not in ('records',)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=('pilot','main','bootstrap'),default='pilot')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.mode=='bootstrap':
        run_bootstrap(args.output,workers=args.workers);return
    tasks=specifications()
    if args.mode=='pilot': tasks=[tasks[0],tasks[1]]
    project=Path(__file__).resolve().parents[3]
    source_paths={Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                  if getattr(module,'__file__',None) and str(getattr(module,'__file__','')).endswith('.py')
                  and Path(module.__file__).resolve().is_relative_to(project)}
    source_paths.update((project/'rnj_wzzt_core/rnj_wzzt/data'/name) for name in
                        ('paper_style_case_bank.py','paper_style_terminal_lv.py'))
    manifest={str(path.relative_to(project)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source_paths)}
    write(Path(args.output)/'source_manifest.json',manifest)
    write(Path(args.output)/'configuration.json',dict(tasks=tasks,workers=args.workers,
        model='full_raw_PQV_EIV_AC',seed_unit='12 base trajectories; length and boundary paired',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        source_manifest_sha256=hashlib.sha256((Path(args.output)/'source_manifest.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(Path(__file__).with_name('protocol.md').read_bytes()).hexdigest()))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,task,args.output) for task in tasks]
        for future in as_completed(futures):
            item=future.result();results.append(item);print(json.dumps(item),flush=True)
            write(Path(args.output)/'checkpoint.json',dict(total=len(tasks),finished=len(results),results=results))
    write(Path(args.output)/'completed.json',dict(total=len(tasks),results=results))


if __name__=='__main__': main()
