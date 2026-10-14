"""Matched finite-witness master benchmark; never modifies frozen old results."""
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from . import ev_original_study as base
from . import ev_compact_study as study
from . import gradient_price_study as io
from .ev_original_domain_milp import (from_moving_model, exchange_original_domain,
    solve_original_master, evaluate_original_pair)
from .ev_original_ac_check import build_multisite_context


def run(seed=989101,*,seconds=2.):
    began=perf_counter();case=base.make_case(seed,base.FAMILIES[(seed-989101)%4])
    prepared=base.prepare(case,'asset');history=case['history']
    baseline=np.zeros((6,3));baseline[:,0]=2.
    context=build_multisite_context(history,prepared['candidates'],prepared['fits'],case['hm'],case['fm'],
        baseline,-6.,6.,base_kw=base.BASE_KW,line_bounds=base.ASSET_BOUNDS)
    if not context['success']:raise ValueError(context['diagnostics']['failure'])
    pairs=[from_moving_model(m) for m in context['models']]
    offsets=[prepared['fits'][m['b']].loss for m in context['models']]
    rows=[]
    for name,(mobile,discharge) in study.DOMAIN_FLAGS.items():
        domain=base.service_domain(mobile=mobile,discharge=discharge);op=domain.as_dict()
        b=base.realize_control(domain,baseline)
        quality=domain.solve_prices(np.tile([1,9,1,9,1,9],(3,1)).T,time_limit=3.)
        points=[b['x']]
        if domain.check(quality)['feasible']:points.append(quality.x)
        initial=exchange_original_domain(pairs,op['control_lower'],op['control_upper'],
            operation=op,offsets=offsets,initial_operation_points=points,max_iterations=1,total_seconds=1e-6,
            witness_policy='affine_preferred',backend='gurobi')
        record=dict(domain=name,initial_lower=initial['objective_lower'],masters={})
        for form in ('residual_chord','shared_control_square'):
            master=solve_original_master(pairs,initial['witnesses'],op['control_lower'],op['control_upper'],
                operation=op,offsets=offsets,control_segments=4,residual_segments=4,
                time_limit=seconds,backend='gurobi',attained_lower=initial['objective_lower'],
                warm_start_operation=initial['operation_x'],master_form=form)
            if master['u'] is not None:
                qps=[evaluate_original_pair(p,master['u']) for p in pairs]
                lower=float(np.mean([.5*(q['tangent_lower_numeric']-off) for q,off in zip(qps,offsets)]))
                primal=float(np.mean([.5*(q['primal_upper']-off) for q,off in zip(qps,offsets)]))
                master['independent_action_QP']=dict(lower=lower,primal_upper=primal,
                    all_converged=all(q['success'] for q in qps),bounds_valid=master['objective_upper']>=lower-1e-7)
                if not master['independent_action_QP']['bounds_valid']:raise ValueError('master upper below checked action lower')
            record['masters'][form]=master
        rows.append(record)
    return dict(seed=seed,matched_witnesses=True,master_seconds=seconds,rows=rows,
        source_sha256=study.source_hashes(),history=io.observations_record(history),
        historical_fits=[io.fit_record(c,prepared['fits'][c.name]) for c in prepared['candidates']],
        seconds=perf_counter()-began,scope='same finite witnesses and proxy; no global AC claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=989101)
    parser.add_argument('--seconds',type=float,default=2.);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=run(args.seed,seconds=args.seconds)
    io.write_json(args.output,r,compressed=args.output.suffix=='.gz')
    for row in r['rows']:
        print(row['domain'],[(k,m['n_binary'],m.get('n_shared_square_binary',0),
            m['objective_upper'],m['seconds']) for k,m in row['masters'].items()],flush=True)


if __name__=='__main__':main()
