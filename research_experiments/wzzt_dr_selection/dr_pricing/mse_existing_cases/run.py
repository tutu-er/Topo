"""Small existing-case DR diagnostic; frozen QP candidates, original devices/AC.

Commands prepare, plan, actual keep input preparation, price freeze and physical
evaluation separate. Numeric arrays and observations belong in --out, outside Git.
Case choice is retrospective; price direction uses historical residuals only.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LEGACY = HERE.parent / 'paper15_exponential'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(HERE))
import core
core._old_types()
mse = sys.modules['_mse_existing_original_miqp_mse']
from selection_mse import choose_mse
from vendor.solver import build_device_graph, _values, _attribute
from research_experiments.wzzt_dr_selection.dr_spatial_correlated.ac_bridge import (
    open_runtime, build_background, PRODUCER_MODULES)

io = mse.base
PARENT = Path('D:/0-github_workspace/Topo/outputs/mse_exit_tuning_20261111')
DATA = Path('D:/0-github_workspace/Topo/outputs/day96_t_benchmark_20261110')
RESOURCE = REPO / 'research_experiments/wzzt_dr_selection/dr_heterogeneous_resources'
CASES = [('pengwah18', 1000, 'v0005'), ('soumalas11', 1002, 'v0010')]


def source_files():
    folders = [HERE, LEGACY / 'vendor', LEGACY / 'vendor_ac',
        RESOURCE, RESOURCE.parent / 'dr_heterogeneous_pilot']
    files = [p for folder in folders for p in folder.rglob('*.py')]
    files += [LEGACY / n for n in ('run.py', 'miqp_mse.py', 'selection_mse.py')]
    files += [RESOURCE.parent / n for n in ('dr_spatial_correlated/ac_bridge.py',
        'dr_competition_bridge/physical_bridge.py', 'end_to_end/physical_case.py')]
    source = DATA / 'source/rnj_wzzt_core'
    files += list((source/'rnj_wzzt').rglob('*.py'))
    return sorted(set(p.resolve() for p in files))


def dump(path, obj):
    Path(path).write_bytes(pickle.dumps(obj, protocol=4))


def packet(folder):
    config = io.read(folder / 'case.json')
    catalog = io.read(io.checked(config['catalog']))
    physics=io.read(io.checked(config['source_configuration']))['physics']
    entry=dict(catalog[config['job']],pq_sigma=float(physics['pq_sigma']))
    allocation = next(c for c in io.read(io.checked(config['allocation']))['cases']
                      if c['case'] == config['case'])
    return config, core.load_history_case(io.checked(config['result']),
        entry, allocation, io.read(io.checked(config['library'])))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(p): io.sha(p) for p in source_files()}
    protocol = dict(status='frozen', id='existing_frozen_mse_DR_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='retrospectively selected existing error cases, not an unbiased benchmark',
        cases=CASES, source_sha256=sources, python=sys.version,
        candidate_rule='.24 contracted, training-QP output library; tau .001 historical view',
        pool='retain every unique output model in source order; R/X frozen',
        pair='historical strict-MSE winner versus nearest different support family by validation MSE',
        case_selection_uses_saved_truth=True, price_uses_truth=False,
        selection='strict physical K*MSE, scaled tie tolerance1e-12; full old validation plus new day w1',
        resources='unchanged all-node heterogeneous inventory, original causal power/state projection',
        forecast='mean of all four already-observed complete days; no future data',
        working_law='fixed 5% nodal P/Q co-moving forecast error, AR1 .7, independent nodes; Gaussian remaining voltage error; nominal DR in planning',
        actual_law='original correlated continuous resource response, causal device clipping, original AC and original P/Q/V noise',
        probability_scope='90% sufficient bound conditional on the chosen frozen reference model and working Gaussian law; not topology correctness',
        optimizer=dict(inner_seconds=45, global_seconds=60, mip_gap=.02, threads=1,
                       inner_anchors=['zero', 'best_predetermined_full_direction']),
        fallback='if probability qualification fails, retain best prespecified feasible direction as unqualified diagnostic, not minimum-cost solution',
        policies=['zero', 'optimized', 'rotation'], no_cost_upper_bound=True,
        paths=[dict(day=i, profile=i % 3, replicate=1120000 + 137*i,
                    response_seed=2026111200 + i) for i in range(6)],
        actual_future_opened=False, new_fits=0, new_topology_search=0)
    io.write(out / 'protocol.json', protocol)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(source_files()): z.write(p, f'{i:03d}_{p.name}')
    inventory = []
    for name, bundle, noise in CASES:
        job = f'day96__{name}__b{bundle}__reference__{noise}__N96'
        folder = out / job; folder.mkdir()
        config = dict(case=name, job=job,
            catalog=io.ref(PARENT / 'catalog.json'),
            source_configuration=io.ref(DATA/'configuration.json'),
            result=io.ref(PARENT / f'jobs/{job}/result.json'),
            allocation=io.ref(RESOURCE / 'case_allocations.json'),
            library=io.ref(RESOURCE / 'library.json'),
            metadata=io.ref(DATA / f'networks/{name}/reference/metadata.json'))
        io.write(folder / 'case.json', config)
        _, data = packet(folder)
        inventory.append(dict(case=name, job=job,
            current=data['models'][data['current']]['id'], reference=data['models'][data['reference']]['id'],
            unique_candidates=len(data['models']), train_rows=len(data['train_power']),
            validation_rows=len(data['history_power']), resources=data['p'].device.R,
            original_noise=data['voltage_sigma'], input_references=data['input_references']))
    io.write(out / 'inventory.json', inventory)
    io.write(out / 'run_lock.json', dict(status='prepared', pid=None))
    return inventory


def verify(out):
    protocol = io.read(out / 'protocol.json')
    for path, digest in protocol['source_sha256'].items():
        if io.sha(path) != digest: raise ValueError('Changed frozen source: ' + path)
    return protocol


def tangent(p, gate, anchor, out, seconds):
    """Convex quadratic fee with an affine sufficient score constraint.

    The exact existing mixed-integer causal device graph is retained. The
    affine condition is an inner approximation; its bound is not a global
    bound for the original reverse-quadratic probability constraint.
    """
    import gurobipy as gp
    if (out/'summary.json').exists():
        result=io.read(out/'summary.json')
        pt=p.nominal(p.requests_to_values(io.arrays(out/'candidate.npz')['x'])) if (out/'candidate.npz').exists() else None
        return pt,result
    out.mkdir(parents=True, exist_ok=False)
    pt = p.nominal(p.requests_to_values(anchor))
    v = pt['mean_u'][gate.resources].ravel()
    grad = 2 * gate.factor.T @ (gate.offset + gate.factor @ v)
    rhs = float(gate.h - gate.value(pt['mean_u']) + grad @ v)
    opt = copy.copy(p); opt.device = copy.copy(p.device)
    null = np.setdiff1d(np.arange(p.device.R), gate.resources)
    for name in ('delta_lower_kw', 'delta_upper_kw', 'scale'):
        a = getattr(p.device, name).copy(); a[null] = 0.; setattr(opt.device, name, a)
    anchor=anchor.copy(); anchor[null]=0.
    opt.lower = p.lower.copy(); opt.upper = p.upper.copy()
    mask = np.isin(p.active // p.device.H, null); opt.lower[mask] = 0.; opt.upper[mask] = 0.
    began = time.perf_counter(); best = None
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0); env.start()
        with gp.Model('MSE_price_inner', env=env) as m:
            for key, val in dict(OutputFlag=1, LogToConsole=0, LogFile=str(out/'gurobi.log'),
                TimeLimit=seconds, Threads=1, MIPGap=.02, NonConvex=0, Seed=0,
                MIPFocus=1, FeasibilityTol=1e-8, IntFeasTol=1e-8, DualReductions=0).items(): m.setParam(key, val)
            x, u, cost = build_device_graph(m, opt, anchor)
            linear = gp.LinExpr()
            for a, term in zip(grad, u[gate.resources].ravel()):
                if a: linear += float(a) * term.expr
            # Small numerical interior; the reported gate remains the original one.
            m.addConstr(linear >= rhs+1e-6*max(1.,abs(gate.h)))
            m.setObjective(cost, gp.GRB.MINIMIZE)
            m.update(); m.write(str(out/'model.mps.gz')); m.optimize()
            result = dict(status=int(m.Status), has_solution=bool(m.SolCount),
                inner_lower_bound=_attribute(m, 'ObjBound'),
                inner_gap=_attribute(m, 'MIPGap') if m.SolCount else None,
                seconds=time.perf_counter()-began)
            if m.SolCount:
                raw = _values(x); raw = p.expand(np.clip(raw.ravel()[p.active], p.lower, p.upper))
                best = p.nominal(p.requests_to_values(raw))
                result.update(fee=best['fee'], gate=gate.bound(best['mean_u']),
                              device=p.audit_delivery(best['mean_u']))
    io.write(out/'summary.json', result)
    if best is not None: np.savez_compressed(out/'candidate.npz',x=best['x'])
    return best, result


def plan_case(folder, protocol):
    if (folder/'policy_freeze.json').exists(): return io.read(folder/'plan_summary.json')
    config, data = packet(folder); p = data['p']; gate = core.build_gate(data)
    zero = p.nominal(p.zero)
    seeds = [('zero', zero)]
    grad_u = np.zeros((p.device.R, p.T))
    grad_u[gate.resources] = (2*gate.factor.T@gate.offset).reshape(-1, p.T)
    gx = p.device.hourly_vjp(grad_u, zero['cache']).ravel()[p.active]
    for sign in (1, -1):
        x = np.where(sign*gx >= 0, p.upper, p.lower)
        x[np.isin(p.active//p.device.H,np.setdiff1d(np.arange(p.device.R),gate.resources))]=0.
        seeds.append((f'gradient_full_{sign}', p.nominal(x)))
    trace = []; qualified = []
    if gate.bound(zero['mean_u'])['sufficient']:
        qualified = [('certified_zero', zero)]
    else:
        full = max(seeds[1:], key=lambda kv: gate.value(kv[1]['mean_u']))
        for j, (label, anchor) in enumerate([seeds[0], full]):
            attempt, result = tangent(p, gate, anchor['x'], folder/f'inner_{j}', protocol['optimizer']['inner_seconds'])
            trace.append(dict(anchor=label, result=result))
            if attempt is not None:
                seeds.append((f'inner_{j}', attempt))
                if p.audit_delivery(attempt['mean_u'])['passed'] and gate.bound(attempt['mean_u'])['sufficient']:
                    qualified.append((f'inner_{j}', attempt))
        warm = min(qualified, key=lambda kv: kv[1]['fee'])[1] if qualified else max(seeds, key=lambda kv:gate.value(kv[1]['mean_u']))[1]
        if (folder/'global/summary.json').exists():
            solved=io.read(folder/'global/summary.json')
        else:
            solved = mse.solve_factorized(p, gate, folder/'global', warm['x'],
                dict(seconds=protocol['optimizer']['global_seconds'], mip_gap=.02))
        trace.append(dict(global_solver=solved))
        if solved['qualified']:
            z = io.arrays(folder/'global/quote.npz')
            point=p.nominal(p.requests_to_values(z['x']))
            if gate.bound(point['mean_u'])['sufficient'] and p.audit_delivery(point['mean_u'])['passed']:
                qualified.append(('global_retained',point))
    for label, pt in seeds:
        if gate.bound(pt['mean_u'])['sufficient'] and p.audit_delivery(pt['mean_u'])['passed']:
            qualified.append((label, pt))
    label, best = min(qualified, key=lambda kv:kv[1]['fee']) if qualified else max(seeds, key=lambda kv:gate.value(kv[1]['mean_u']))
    u = best['mean_u']; fee = best['fee']
    try:
        rotate, rotation = io.rotation(p, fee)
    except ValueError:
        r = np.arange(p.device.R)
        x = np.where((np.arange(24)[None,:]//2+(r%4)[:,None])%2==0, 1., -1.)
        x = np.clip(x, p.expand(p.lower), p.expand(p.upper))
        rotate = p.device.nominal(x)
        rotation = dict(cost=p.device.fees(rotate), cost_matched=False,
                        reason='full feasible rotation cheaper than requested fee; original domain retained')
    quotes = dict(zero=zero['mean_u'], optimized=u, rotation=rotate, hourly_requests=best['x'])
    np.savez_compressed(folder/'quotes.npz', **quotes)
    summary = dict(case=config['case'], current=data['models'][data['current']]['id'],
        reference=data['models'][data['reference']]['id'], status='qualified' if qualified else 'unqualified_diagnostic',
        retained_source=label, zero_bound=gate.bound(zero['mean_u']), bound=gate.bound(u),
        fees={k:p.device.fees(quotes[k]) for k in protocol['policies']},
        device={k:p.audit_delivery(quotes[k]) for k in protocol['policies']},
        trace=trace, rotation=rotation, resources=p.device.R,
        history_S=gate.history_S, delta=gate.delta, K=gate.K, scale=gate.scale,
        global_minimum_proven=bool(fee==0 and qualified),
        raw_optimizer_scope='inner feasible upper bounds plus bounded original nonconvex solve; no global optimality claim unless zero')
    io.write(folder/'plan_summary.json', summary)
    io.write(folder/'policy_freeze.json', dict(status='frozen', quote=io.ref(folder/'quotes.npz'),
        case=io.ref(folder/'case.json'), inputs=data['input_references'], source=io.ref(folder.parent/'protocol.json'),
        price_uses_truth=False, actual_future_used=False))
    return summary


def plan(out):
    protocol = verify(out); rows = []
    io.write(out/'run_lock.json', dict(status='planning', pid=os.getpid()))
    for name, bundle, noise in protocol['cases']:
        folder = out/f'day96__{name}__b{bundle}__reference__{noise}__N96'
        row = plan_case(folder, protocol); rows.append(row)
        print('planned', name, row['status'], row['fees'], flush=True)
    io.write(out/'plans.json', rows)
    io.write(out/'all_prices_frozen.json', dict(status='frozen', plans=io.ref(out/'plans.json'), protocol=io.ref(out/'protocol.json')))
    io.write(out/'run_lock.json', dict(status='prices_frozen', pid=None))
    return rows


def _load_execution():
    path=LEGACY/'vendor_ac/execution.py'
    spec=importlib.util.spec_from_file_location('_original_mse_DR_execution', path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def audit_network(runtime, folder, name):
    """Check post-freeze AC network against the unchanged historical network."""
    path=DATA/f'networks/{name}/reference/branches.csv'
    with path.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
    graph=runtime['context'].net.to_networkx_graph()
    expected={tuple(sorted((int(r['from_bus']),int(r['to_bus'])))):
        (float(r['r_ohm']),float(r['x_ohm'])) for r in records
        if r.get('status','True').lower() in ('true','1','1.0')}
    actual={tuple(sorted((int(a),int(b)))):(float(v['r_ohm']),float(v['x_ohm']))
        for a,b,v in graph.edges(data=True)}
    if expected.keys()!=actual.keys(): raise ValueError('AC network edge set differs from historical network')
    error=max((max(abs(x-y) for x,y in zip(expected[e],actual[e])) for e in expected),default=0.)
    if error>1e-12: raise ValueError('AC branch impedance differs from historical network')
    io.write(folder/'physical_network_audit.json',dict(passed=True,edges=len(expected),
        maximum_impedance_error_ohm=error,reference=io.ref(path)))


def actual(out):
    protocol = verify(out); io.checked(io.read(out/'all_prices_frozen.json')['plans'])
    io.write(out/'run_lock.json', dict(status='actual', pid=os.getpid())); rows=[]; truth={}
    for ci, (name, bundle, noise) in enumerate(protocol['cases']):
        folder=out/f'day96__{name}__b{bundle}__reference__{noise}__N96'
        config,data=packet(folder); freeze=io.read(folder/'policy_freeze.json')
        quotes=io.arrays(io.checked(freeze['quote']));dev=data['original_device']
        meta=io.read(io.checked(config['metadata']))
        runtime=open_runtime(DATA/'source/rnj_wzzt_core',dict(config['metadata'],case_name=meta['source']['case_name']),
            repo_root=REPO,protocol_ref=io.ref(out/'protocol.json'),labels=data['labels'],case=name,
            v_sigma=data['voltage_sigma'],pq_sigma=data['pq_sigma'])
        audit_network(runtime,folder,name)
        execution=_load_execution()
        for path in protocol['paths']:
            bg=build_background(runtime,path['profile'],path['replicate']+ci*10000)
            multiplier=dev.draw(path['response_seed']+ci*100,1,stress=False)[0]
            for policy in protocol['policies']:
                result_dir=folder/'actual'/f"day{path['day']}_{policy}"
                result_dir.mkdir(parents=True,exist_ok=True)
                receipt=result_dir/'receipt.json'
                if receipt.exists(): rows.append(io.read(receipt));continue
                row=dict(case=name,job=config['job'],day=path['day'],policy=policy,fee=dev.fees(quotes[policy]))
                try:
                    result=execution.execute(runtime,bg,dev,quotes[policy],multiplier,f"mse_DR_{name}_{path['day']}_{policy}")
                    dump(result_dir/'result.pkl',result)
                    row.update(status=result['summary']['status'],physical=result['summary'])
                    if row['status']=='complete':
                        scene=result['scene'];power=np.column_stack([np.asarray(scene[k]) for k in ('P_terminal','Q_terminal')])
                        drop=np.asarray(scene['drop_target'])
                        new_sse=np.array([float(np.sum((drop-power@h.T)**2)) for h in data['H']])
                        losses=(data['history_sse']+new_sse)/(data['history_drop'].size+drop.size)
                        selection=choose_mse(losses,[m['internal'] for m in data['models']],K=data['K'],candidate_ids=[m['id'] for m in data['models']])
                        a,b=data['current'],data['reference'];margin=float(losses[a]-losses[b]-1e-12/data['K'])
                        np.savez_compressed(result_dir/'observed.npz',power=power,drop=drop,delivered=result['private']['delivered'])
                        row.update(selection=selection,pair_excluded=margin>0,pair_margin=margin,
                            candidate_prediction_distance_mse=float(np.mean((power@(data['H'][b]-data['H'][a]).T)**2)),
                            observation=io.ref(result_dir/'observed.npz'))
                except Exception as exc:
                    row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                io.write(receipt,row);rows.append(row)
                io.write(out/'checkpoint.json',dict(completed=len(rows),planned=len(protocol['cases'])*len(protocol['paths'])*3))
            print('actual',name,path['day'],len(rows),flush=True)
        # Truth is retained only by evaluator, after every quote has been frozen.
        truth[name]=execution.effective_truth_supports(runtime['context'].net)
    io.write(out/'predictions_frozen.json',dict(rows=rows,plans=io.ref(out/'all_prices_frozen.json')))
    for row in rows:
        if row['status']!='complete': continue
        folder=out/row['job'];_,data=packet(folder)
        model=next(m for m in data['models'] if m['id']==row['selection']['selected'])
        pred={tuple(sorted(data['labels'][i] for i in support)) for support in model['point']['supports']}
        target={tuple(sorted(s)) for s in truth[row['case']]}
        missing=target-pred;extra=pred-target;N=len(data['labels'])
        row['errors']=dict(missing=len(missing),extra=len(extra),total=len(missing)+len(extra),
            missing_supports=sorted(missing),extra_supports=sorted(extra),
            proper_total=sum(1 for s in missing|extra if 1<len(s)<N))
    io.write(out/'scored_rows.json',rows)
    summary=[]
    for name,_,_ in protocol['cases']:
        table={}
        for policy in protocol['policies']:
            group=[r for r in rows if r['case']==name and r['policy']==policy]
            ok=[r for r in group if r['status']=='complete']
            table[policy]=dict(planned=len(group),completed=len(ok),failures=sum(r['status']=='failed' for r in group),
                unresolved=sum(r['status'] not in ('complete','failed') for r in group),
                pair_exclusions=sum(r['pair_excluded'] for r in ok),
                complete_recoveries=sum(r['errors']['total']==0 for r in ok),
                total_errors=sum(r['errors']['total'] for r in ok),
                mean_fee=float(np.mean([r['fee'] for r in group])),
                mean_pair_margin=float(np.mean([r['pair_margin'] for r in ok])) if ok else None,
                mean_prediction_distance_mse=float(np.mean([r['candidate_prediction_distance_mse'] for r in ok])) if ok else None,
                selected_models=[r['selection']['selected'] for r in ok],
                absolute_DR_kwh=float(np.mean([r['physical']['resource_audit']['absolute_DR_kwh'] for r in ok])) if ok else None)
        summary.append(dict(case=name,policies=table))
    io.write(out/'summary.json',dict(status='complete',scope='retrospective frozen-pool diagnostic',cases=summary,
        paths_per_case=6,observations=len(rows),new_fits=0,new_topology_search=0))
    io.write(out/'complete.json',dict(status='complete'))
    io.write(out/'run_lock.json',dict(status='complete',pid=None))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','plan','actual']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](args.out.resolve()),default=lambda v:v.tolist(),allow_nan=False))
