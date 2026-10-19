"""Explicit-only audit of saved nominal AC pair derivatives.

Importing this module performs no fit.  ``run_saved`` reconstructs its entire
information set from saved observed D, meter, forecast and training fits.  It
does not import a case generator or inspect truth, actual future or regime.
Finite-difference controls audit the AC objective in a neighborhood; they are
not proposed executable EV plans and do not change a frozen policy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import gradient_price_study as io
from . import likelihood as lk
from . import ev_mobile_v4_policy as policy
from . import ev_mobile_v4_domain as nested
from .information_context import meter_from_record


DEFAULT_STEPS_KW = (.01, .005)


def declared_directions(shape):
    """Two fixed directions, independent of truth and of fitted gradients."""
    if len(shape)!=2 or shape[0]<2 or shape[1]<2:
        raise ValueError('at least two times and sites required for declared audit directions')
    temporal=np.zeros(shape);temporal[0,0]=1.;temporal[-1,0]=-1.
    spatial=np.zeros(shape);spatial[-1,0]=-1.;spatial[-1,-1]=1.
    return dict(early_late_initial_site=temporal,late_initial_remote_site=spatial)


def _pairs(profile,labels,*,strict):
    expected={(a,b) for a in labels for b in labels if a!=b}
    rows=profile.get('pairs',[])
    keys=[(row['a'],row['b']) for row in rows]
    if len(keys)!=len(set(keys)) or any(key not in expected for key in keys):
        raise ValueError('duplicate or unknown directed pair in finite-difference evidence')
    if strict and (not profile.get('success') or set(keys)!=expected):
        raise ValueError('complete successful saved parent pair profile required')
    return {(row['a'],row['b']):row for row in rows}


def _fit_status(profile):
    """Keep local convergence and physical diagnostics for every AC slot."""
    result=[]
    diagnostic_keys=('failure_kind','total_nfev','active_bound_count',
        'ac_equation_residual','power_balance_residual','global_optimality_certified',
        'line_domain','ratio_constraint_enforced','physical_voltage_feasible')
    for reference in profile.get('references',[]):
        for b,fit in reference.get('fits',{}).items():
            diagnostics=fit.get('diagnostics',{})
            result.append(dict(a=reference['a'],b=b,
                **{name:fit.get(name) for name in ('success','attempted','loss','status',
                    'message','nfev','optimality','gradient_reliable','fixed_source_success','failure')},
                diagnostics={name:diagnostics[name] for name in diagnostic_keys if name in diagnostics}))
    return result


def audit_directed_derivatives(evaluate,power,parent,*,labels,
                               directions=None,steps_kw=DEFAULT_STEPS_KW):
    """Compare each smooth pair gap, retaining every failed derivative slot.

    ``evaluate`` is a nominal AC callable, not a simulator.  The saved parent
    profile supplies the analytic price and is not refitted here.  Returned
    full profiles are kept separately by ``run_saved`` for reproducibility.
    """
    began=perf_counter();u=np.asarray(power,float)
    if u.ndim!=2 or not np.isfinite(u).all():
        raise ValueError('finite [time,site] parent power required')
    labels=tuple(labels)
    if len(labels)<2 or len(labels)!=len(set(labels)):
        raise ValueError('complete distinct candidate labels required')
    base=_pairs(parent,labels,strict=True)
    steps=tuple(float(h) for h in steps_kw)
    if not steps or not np.isfinite(steps).all() or min(steps)<=0 or len(steps)!=len(set(steps)):
        raise ValueError('distinct positive finite difference steps required')
    directions=declared_directions(u.shape) if directions is None else directions
    derivatives=[];profiles={};perturbations=[]
    for direction_name,raw in directions.items():
        direction=np.asarray(raw,float)
        if direction.shape!=u.shape or not np.isfinite(direction).all() or not np.any(direction):
            raise ValueError('finite matching nonzero audit direction required')
        analytic={}
        for key,row in base.items():
            price=np.asarray(row['gradient_per_kw'],float)
            if not row.get('success') or price.shape!=u.shape or not np.isfinite(price).all():
                raise ValueError('complete finite parent pair gradient required')
            analytic[key]=float(np.sum(price*direction))
        for h in steps:
            side_rows={}
            for sign in (-1,1):
                key=f'{direction_name}__h_{h:.12g}__sign_{sign:+d}'
                perturbed=u+sign*h*direction
                try:
                    full=dict(evaluate(perturbed))
                    pairs=_pairs(full,labels,strict=False)
                except Exception as exc:
                    full=dict(success=False,failure=type(exc).__name__+': '+str(exc),pairs=[])
                    pairs={}
                profiles[key]=full;side_rows[sign]=pairs
                perturbations.append(dict(key=key,direction=direction_name,h_kw=h,sign=sign,
                    power_kw=perturbed,success=bool(full.get('success')),
                    score=full.get('score'),failure=full.get('failure'),
                    fit_status=_fit_status(full)))
            for pair in sorted(base):
                minus,plus=side_rows[-1].get(pair),side_rows[1].get(pair)
                valid=bool(minus is not None and plus is not None and
                    minus.get('success') and plus.get('success') and
                    np.isfinite([minus.get('half_gap',np.nan),plus.get('half_gap',np.nan)]).all())
                fd=None if not valid else float((plus['half_gap']-minus['half_gap'])/(2*h))
                difference=None if fd is None else fd-analytic[pair]
                derivatives.append(dict(a=pair[0],b=pair[1],direction=direction_name,
                    h_kw=h,success=valid,analytic_per_kw=analytic[pair],
                    central_difference_per_kw=fd,difference=difference,
                    absolute_error=None if difference is None else abs(difference),
                    error_over_1_plus_abs_price=None if difference is None else
                        abs(difference)/(1+abs(analytic[pair])),
                    failure=None if valid else 'required perturbed local pair fit unavailable or nonfinite'))
    complete=sum(row['success'] for row in derivatives)
    return dict(status='complete' if complete==len(derivatives) else 'partial_failure',
        derivative_slots=len(derivatives),valid_derivative_slots=complete,
        failed_derivative_slots=len(derivatives)-complete,derivatives=derivatives,
        perturbations=perturbations,directions=directions,steps_kw=steps,
        parent_refitted=False,parent_fit_status=_fit_status(parent),
        scope='nominal complete AC directed half-SSE gaps; local fit branches',
        checks_ev_operational_feasibility=False,uses_truth=False,uses_actual_future=False,
        changes_policy=False,seconds=perf_counter()-began,full_profiles=profiles)


def restore_saved_inputs(raw,frozen):
    """Restore only permitted saved inputs; never read simulator-only fields."""
    observed=lk.Observations(**raw['observed'])
    meter=meter_from_record(raw['observed_meter'])
    future_meter=meter_from_record(raw['future_meter'])
    rows=raw['training_result']['fits']
    if not raw['training_result']['all_complete']:
        raise ValueError('saved observed-D reference training was incomplete')
    bank=tuple(lk.RadialCandidate.from_supports(row['name'],observed.n,
        tuple(tuple(s) for s in row['supports'])) for row in rows)
    fits={row['name']:SimpleNamespace(**{**row,
        **{key:np.asarray(row[key],float) for key in ('r','x','source_p','source_q')}})
        for row in rows}
    cfg=raw['config'];study=frozen['config']
    options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds'],
        'multistart':study['multistart_predictive'],'max_nfev':study['max_nfev']}
    forecast=raw['forecast'];base_kw=float(forecast['metadata']['base_kw'])
    context=policy.build_context(observed,meter,bank,fits,future_meter,forecast,
                                base_kw=base_kw,**cfg['source'])
    return context,options,int(raw['spec']['seed'])+3000


def run_saved(path,output):
    """Explicitly spend AC fits on one saved guarded-FC parent audit."""
    path,output=Path(path),Path(output)
    raw=io.read_json(path);frozen=io.read_json(path.parents[1]/'freeze.json')
    if raw['spec']['phase']!='development' or int(raw['spec']['index'])!=0:
        raise ValueError('predeclared audit uses development index 0 only')
    if raw.get('policy_id')!=frozen['policy_id']:
        raise ValueError('saved development policy does not match its freeze')
    context,options,seed=restore_saved_inputs(raw,frozen)
    method=raw['methods']['guarded__fixed_charge'];power=np.asarray(method['power_kw'],float)
    key=nested.action_key(power)
    if key!=method['action_key']:
        raise ValueError('saved selected parent action identity changed')
    parent=raw['reporting_cache'][key]
    objective=policy.CommonACObjective(context,options,seed=seed)
    if parent.get('objective_id')!=objective.objective_id:
        raise ValueError('reconstructed observed-D objective differs from saved parent')
    objective.cache.remember(power,parent)
    def evaluate(perturbed):
        print('AC_DERIVATIVE_PERTURBATION',nested.action_key(perturbed),flush=True)
        return objective.cache.evaluate(perturbed)[1]
    report=audit_directed_derivatives(evaluate,power,parent,
        labels=[candidate.name for candidate in context['bank']])
    full=report.pop('full_profiles')
    profile_path=output.with_name(output.stem+'.profiles.json.gz')
    io.write_json(profile_path,dict(parent=parent,perturbations=full),compressed=True)
    report.update(input_path=str(path.resolve()),input_sha256=sha256(path.read_bytes()).hexdigest(),
        policy_id=frozen['policy_id'],objective_id=objective.objective_id,
        helper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_method='guarded__fixed_charge',parent_action_key=key,
        saved_parent_reused=True,source_fields_read=['observed','observed_meter','future_meter',
            'training_result','config','forecast','spec','methods','reporting_cache','policy_id'],
        fit_options=options,seed=seed,engine_counters=objective.engine.counters(),
        profile_evidence_path=str(profile_path.resolve()),
        profile_evidence_sha256=sha256(profile_path.read_bytes()).hexdigest())
    io.write_json(output,report,compressed=output.suffix=='.gz')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run_saved(args.input,args.output)
    print('DERIVATIVE_AUDIT',report['status'],report['valid_derivative_slots'],
          '/',report['derivative_slots'],flush=True)


if __name__=='__main__':
    main()
