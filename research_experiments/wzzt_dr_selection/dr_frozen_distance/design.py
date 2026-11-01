"""Historical-only dispute, resource and energy-neutral DR catalogue design."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay,DayModule,shift_shape
from research_experiments.wzzt_dr_selection.dr_conic_pricing.resources import resource_contract_data
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem,evaluate_plan


def contrast_shape(gradient,slots=8):
    """Maximizer of the historical linear contrast over two equal-size sets.

    Reduce at the lowest marginal distance, rebound at the highest. The
    rearrangement solution is exact for this bounded unit-shape problem; its
    new-day nonlinear distance is still tested by the stochastic optimizer.
    """
    g=np.asarray(gradient,float)
    if g.shape!=(96,) or not np.isfinite(g).all() or not 0<slots<=48:
        raise ValueError('96 finite historical marginal values and disjoint sets required')
    order=np.argsort(g,kind='stable');shape=np.zeros(96)
    shape[order[:slots]]=-1.;shape[order[-slots:]]=1.
    return shape


def build_design(pool,history,units=5):
    labels=pool['labels'];n=len(labels);base_kw=pool['base_kw']
    coarse,refined=pool['models'];dr=np.asarray(refined['R'])-np.asarray(coarse['R'])
    dx=np.asarray(refined['X'])-np.asarray(coarse['X'])
    days=history['training']+history['validation']
    p=np.stack([s['P_terminal'].to_numpy(float) for s in days]);q=np.stack([s['Q_terminal'].to_numpy(float) for s in days])
    assert len(days)==6 and p.shape==(6,96,n)
    baseline=.2*units;upper=.4*units;modcap=.2*units;elasticity=.1*units;energy=.4*units
    pmin=p.min(axis=(0,1))*base_kw;qmin=q.min(axis=(0,1))*base_kw
    unit_column=np.abs(dr+.35*dx).sum(axis=0)/(n*base_kw)*1e6
    added=pool['dispute_added_supportlabels']
    if len(added)!=1:raise ValueError('one declared historical block dispute is required')
    focus=set(added[0])
    def label_family(model):
        point=model['point'];requested=point.get('full_fit',{}).get('supports',point['supports'])
        return {tuple(sorted(labels[i] for i in block)) for block in requested}
    common=label_family(coarse)&label_family(refined)
    containers=[set(b) for b in common if focus<set(b)]
    if not containers:raise ValueError('historical common containing region missing')
    region=min(containers,key=lambda x:(len(x),tuple(sorted(x))))
    eligible=[i for i in range(n) if pmin[i]>=baseline and qmin[i]>=.35*baseline]
    inside=[i for i in eligible if labels[i] in focus]
    outside=[i for i in eligible if labels[i] in region-focus]
    if not inside or not outside:raise ValueError('historical capacity cannot support declared same-fleet comparison')
    # No test or topology truth enters this ordering. Price may still be zero at
    # either available node; this is resource qualification, not forced spread.
    chosen=[max(part,key=lambda i:(unit_column[i],-labels[i])) for part in (inside,outside)]
    resources=[]
    for identity,index in zip(('focus','partner'),chosen):
        resources.append(dict(resource_id=identity,node_label=int(labels[index]),baseline_kw=baseline,
            power_upper_kw=upper,prefix_delta_lower_kwh=-energy,prefix_delta_upper_kwh=energy,
            available=True,reactive_ratio=.35,incentive_per_kwh=0.,electricity_price_per_kwh=0.,
            declaration_source=f'synthetic aggregate of{units} original0.2kW flexible units already contained in natural load; no behavioral calibration'))
    early=shift_shape(range(24,28),range(28,32));late=shift_shape(range(76,80),range(80,84))
    delta=p.mean(axis=0)@dr.T+q.mean(axis=0)@dx.T
    direction=np.sign(delta)
    modules=[];factors=[];shape_audit=[]
    for identity,index,d in zip(('focus','partner'),chosen,([.15,.05],[.15,-.05])):
        marginal=direction@(dr[:,index]+.35*dx[:,index])/(n*base_kw)
        guided=contrast_shape(marginal)
        kinds=[('early',early),('late',late),('spread',early+late),('opposedlate',-late),('history_contrast',guided)]
        for kind,shape in kinds:
            modules.append(DayModule(identity+'_'+kind,{identity:shape},modcap,
                'historical-only energy-neutral schedule; one full-day random amplitude'))
            factors.append(d)
        shape_audit.append(dict(node_label=labels[index],historical_linear_direction_gain=float(marginal@guided),
            reduction_slots=np.flatnonzero(guided<0).tolist(),rebound_slots=np.flatnonzero(guided>0).tolist(),
            zero_energy=float(.25*guided.sum()),maximum_prefix_hours=float(np.abs(.25*np.cumsum(guided)).max())))
    resource_objects=[ResourceDay(**r) for r in resources]
    partial=resource_contract_data(resource_objects,modules,node_labels=labels,
        elasticities=[elasticity]*len(modules),price_upper=[2.]*len(modules),response_factors=factors)
    # The old model container is reused only for physical/settlement constraints.
    # Cantelli linear-benefit fields are zero and are not the new objective.
    mean=np.r_[p.mean(axis=0).ravel(),q.mean(axis=0).ravel()]*base_kw
    metadata=partial.pop('metadata')
    metadata.update(source='two historical RNJ+W predictors and six historical P/Q days',
        objective='paired exact prediction-distance increase, no futureY in design',
        aggregate_units=units,prototype_elasticity=.1,calibrated=False,
        background_law='uniform draw among6 complete observed historical P/Q days',
        response_law='two independent U[-1,1] factors, shared across resources for whole day',
        relative_response_correlation=.8,baseline_semantics='component contained in existing natural load; not added',
        new_PV=False)
    problem=PriceProblem(mu0=mean,background_factors=np.zeros((len(mean),2)),
        benefit_directions=np.zeros((1,len(mean))),benefit_offsets=[0.],risk_allocations=[.1],budget=1.6,
        metadata=metadata,**partial)
    hp=np.asarray(problem.profiles[:96*n]).reshape(96,n,-1)
    hq=np.asarray(problem.profiles[96*n:]).reshape(96,n,-1)
    delta_columns=(np.einsum('tkj,ik->tij',hp,dr)+np.einsum('tkj,ik->tij',hq,dx)).reshape(96*n,-1)/base_kw
    base_delta=(p@dr.T+q@dx.T).reshape(6,96*n)
    return dict(problem=problem,resources=resources,
        modules=[dict(module_id=m.module_id,profiles={k:v.tolist() for k,v in m.profiles.items()},
            amplitude_upper_kw=m.amplitude_upper_kw,description=m.description) for m in modules],
        delta_columns=delta_columns,base_delta=base_delta,
        qualification=dict(units_per_resource=units,baseline_kw=baseline,power_upper_kw=upper,
            actual_modulation_cap_kw=modcap,prefix_energy_kwh=energy,elasticity=elasticity,
            selected_labels=[labels[i] for i in chosen],dispute_labels=sorted(focus),common_region=sorted(region),
            selection_rule='largest frozen pair column-L1 among eligible points inside dispute and its shared containing region complement; no truth or future',
            nodes=[dict(node_label=labels[i],past_P_min_kw=float(pmin[i]),past_Q_min_kvar=float(qmin[i]),
                unit_response_column_micro=float(unit_column[i]),eligible=i in eligible) for i in range(n)],
            actual_qualification_required=True,synthetic_inventory=True),guided_shape_audit=shape_audit)


def draw_scenarios(design,samples,seed):
    rng=np.random.default_rng(seed);problem=design['problem'];result=[];draws=[]
    for index in range(samples):
        day=int(rng.integers(6));z=rng.uniform(-1,1,2)
        rate=problem.elasticities*(1+problem.response_factors@z)
        result.append(dict(base=design['base_delta'][day],price_coeff=design['delta_columns']*rate,
            normalizer=96*len(problem.metadata['node_labels'])))
        draws.append(dict(index=index,historical_day=day,factors=z.tolist()))
    return result,draws


def plan_payload(design,amplitudes,policy):
    problem=design['problem'];x=np.asarray(amplitudes,float)
    audit=evaluate_plan(problem,x)
    if not audit['numerical_feasible']:raise ValueError('price/device/budget failure')
    if any(np.count_nonzero(x[list(g)]>0)>1 for g in problem.exclusive_groups):
        raise ValueError('inactive types must be exactly zero')
    return dict(status='frozen',policy=policy,resources=design['resources'],modules=design['modules'],
        amplitudes=x.tolist(),prices=(x/problem.elasticities).tolist(),elasticities=problem.elasticities.tolist(),
        response_factors=problem.response_factors.tolist(),budget=1.6,
        expected_payment=audit['expected_cost'],hard_payment_bound=audit['hard_budget_bound'],
        baseline_semantics='already-contained component; increment-only injection',aggregate_units=5,
        probability_scope='90% paired frozen-predictor distance gain, never topology recovery')
