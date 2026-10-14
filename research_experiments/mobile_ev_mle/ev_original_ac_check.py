"""Original absolute-parameter AC proxy, multi-site control and AC checks.

Control order is time-major k=t*n+site. No truth or future noise argument is
accepted by construction/forecast routines. Physical box feasibility, proxy
bounds, and complete AC residual checks remain separate pieces of evidence.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from . import likelihood as lk
from . import gradient_price_study as io
from .symmetric_ac_local import (CHANNELS,baseline_from_history,mean_and_jacobian,
    whiten_observations)
from .ev_ac_moving_cross import (state_vector,vector_state,channel_slices,channel_norms,
    transform_domain,parameter_increment,residual as moving_residual,fixed_action_qp)

STATE=('p','q','source_p','source_q')


def _box(bounds,width,name):
    lo,hi=(np.broadcast_to(np.asarray(x,float),(width,)).copy() for x in bounds)
    if not np.isfinite([lo,hi]).all() or np.any(lo>=hi):raise ValueError(name+' requires finite lower < upper')
    return lo,hi


def multisite_control_map(times,n,base_kw):
    if times<1 or n<1 or not np.isfinite(base_kw) or base_kw<=0:raise ValueError('positive dimensions/base required')
    U=np.zeros((times*(2*n+2),times*n)); k=np.arange(times*n)
    U[(k//n)*(2*n+2)+k%n,k]=1./base_kw
    return U


def _channels(history):
    return CHANNELS+tuple(k for k in ('source_p','source_q') if getattr(history,k) is not None)


def _future(candidate,theta,w,root,meter,channels,with_jacobian=True):
    return mean_and_jacobian(candidate,theta[:candidate.m],theta[candidate.m:],
        w['p'],w['q'],w['source_p'],w['source_q'],root,meter,
        channels=channels,with_jacobian=with_jacobian)


def mixed_control_matrix(candidate,theta,w,root,meter,channels,U,*,step_kw=.001):
    """Mixed derivative for independent-time P controls, including same-time sites.

    Each control acts on exactly one terminal P at one time. Batched +/-
    differences group controls by terminal, then split output rows by time.
    No cross-site or cross-time product is suppressed after construction.
    """
    U=np.asarray(U,float);times=len(w['p']);n=candidate.n;width=2*n+2
    if U.ndim!=2 or U.shape[0]!=times*width or not np.isfinite(U).all() or step_kw<=0:
        raise ValueError('valid state/control map and positive step required')
    identities=[]
    for k in range(U.shape[1]):
        nz=np.flatnonzero(U[:,k])
        if len(nz)!=1 or nz[0]%width>=n:raise ValueError('each control must act on one terminal active-power coordinate')
        identities.append((int(nz[0]//width),int(nz[0]%width)))
    if len(set(identities))!=len(identities):raise ValueError('duplicate controls at same physical coordinate')
    base=state_vector(w); results={}; calls=0
    for site in sorted({j for _,j in identities}):
        ks=[k for k,(_,j) in enumerate(identities) if j==site];direction=U[:,ks].sum(axis=1)
        plus=_future(candidate,theta,vector_state(base+step_kw*direction,n),root,meter,channels)
        minus=_future(candidate,theta,vector_state(base-step_kw*direction,n),root,meter,channels);calls+=2
        dl=(plus['line_jacobian']-minus['line_jacobian'])/(2*step_kw)
        dw=(plus['state_jacobian']-minus['state_jacobian'])/(2*step_kw)
        results[site]=(dl,dw,len(plus['mean']))
    rows=next(iter(results.values()))[2]
    dl_all=np.zeros((U.shape[1],rows,2*candidate.m));dw_all=np.zeros((U.shape[1],rows,times*width))
    for channel,sl in channel_slices(channels,times,n).items():
        cw=n if channel in ('p','q','v') else 1
        for k,(time,site) in enumerate(identities):
            row=slice(sl.start+time*cw,sl.start+(time+1)*cw)
            dl_all[k,row]=results[site][0][row];dw_all[k,row]=results[site][1][row]
    return dict(line=dl_all,state=dw_all,ac_derivative_calls=calls,step_kw=step_kw,
        control_time_site=identities,scheme='central finite difference of complete AC analytic Jacobian')


def product_structure(model,*,tolerance=1e-10):
    At=np.asarray(model['At']);S=np.asarray(model['domain']['shift'])
    raw=np.einsum('krp,ps->ksr',At,S);K=len(At)
    shape=model.get('control_shape'); identities=model.get('control_time_site')
    if identities is None and shape is not None:
        identities=[(k//shape[1],k%shape[1]) for k in range(K)]
    if identities is None:identities=[(k,int(model.get('site',0))) for k in range(K)]
    across=[];same=[]
    for k in range(K):
        for s in range(k+1,K):
            norm=float(np.linalg.norm(raw[k,s]+raw[s,k]))
            item=dict(k=k,s=s,coefficient_norm=norm)
            (across if identities[k][0]!=identities[s][0] else same).append(item)
    return dict(raw=raw,diagonal_norms=[float(np.linalg.norm(raw[k,k])) for k in range(K)],
        cross_time=across,same_time_cross_site=same,
        cross_time_max_norm=max((x['coefficient_norm'] for x in across),default=0.),
        same_time_max_norm=max((x['coefficient_norm'] for x in same),default=0.),
        cross_time_numerically_zero=all(x['coefficient_norm']<=tolerance for x in across),
        tolerance=tolerance,products_discarded=0)


def fixed_absolute_coefficients(model,physical):
    """Exact polynomial of the supplied proxy at one fixed absolute witness."""
    if 'control_curvature_difference' in model:raise ValueError('control-curvature diagnostic requires a separate explicit model')
    xi=np.asarray(physical,float);domain=model['domain'];base=np.asarray(domain['base_center']);S=np.asarray(domain['shift'])
    if xi.shape!=base.shape or not np.isfinite(xi).all():raise ValueError('finite matching physical witness required')
    if np.any(xi<np.asarray(domain['physical_lower'])-1e-10) or np.any(xi>np.asarray(domain['physical_upper'])+1e-10):
        raise ValueError('fixed absolute witness violates original parameter box')
    A0=np.asarray(model['A0']);At=np.asarray(model['At']);B=np.asarray(model['B']);d=xi-base
    a=np.asarray(model['c'])+A0@d
    linear=B+np.einsum('krp,p->rk',At,d)-A0@S
    structure=product_structure(model);raw=structure['raw'];K=B.shape[1]
    diagonal=np.column_stack([-raw[k,k] for k in range(K)])
    pairs=[(k,s) for k in range(K) for s in range(k+1,K)]
    cross=np.column_stack([-(raw[k,s]+raw[s,k]) for k,s in pairs]) if pairs else np.empty((len(a),0))
    return dict(constant=a,linear=linear,diagonal=diagonal,cross_pairs=pairs,cross=cross,
        physical_parameter=xi,structure={k:v for k,v in structure.items() if k!='raw'},
        scope='exact fixed-absolute-witness polynomial of declared proxy, not nonlinear AC')


def polynomial_residual(coefficients,u):
    u=np.asarray(u,float)
    products=np.asarray([u[k]*u[s] for k,s in coefficients['cross_pairs']])
    return (np.asarray(coefficients['constant'])+np.asarray(coefficients['linear'])@u+
        np.asarray(coefficients['diagonal'])@(u*u)+np.asarray(coefficients['cross'])@products)


def original_model_qp(model,u):
    """Original-domain QP plus explicit outer comparison, without changing model."""
    own=fixed_action_qp(model,np.asarray(u,float),domain='original')
    outer=fixed_action_qp(model,np.asarray(u,float),domain='outer')
    return dict(original=own,outer=outer,
        primal_gap_original_minus_outer=float(own['primal_upper']-outer['primal_upper']),
        original_lower_minus_outer_upper=float(own['tangent_lower_numeric']-outer['primal_upper']),
        outer_physical_violation=float(outer['physical_bound_violation']),
        scope='two declared convex proxy QPs; full AC check separate')


def build_multisite_context(history,candidates,fits,hm,fm,reference_kw,lower,upper,
                            base_kw=20.,*,line_bounds=None,state_bounds=None,step_kw=.001):
    """H-only model bank with controls flatten[t*n+site], shared H/N R/X.

    line_bounds: r_bounds/x_bounds, scalar endpoints or per-candidate arrays.
    state_bounds: p_bounds/q_bounds/source_bounds; original absolute box.
    Every required candidate remains present or the whole context fails.
    """
    began=perf_counter();candidates=tuple(candidates);u0=np.asarray(reference_kw,float)
    if u0.ndim!=2 or u0.shape[1]!=history.n or not len(u0) or not np.isfinite(u0).all():raise ValueError('reference_kw must be finite (time,terminal)')
    T,n=u0.shape;K=T*n
    def control_bound(value):
        a=np.asarray(value,float)
        if a.size==K:return a.reshape(K).copy()
        return np.broadcast_to(a,u0.shape).ravel().copy()
    low,high=control_bound(lower),control_bound(upper);ref=u0.ravel()
    if np.any(low>high) or not np.isfinite([low,high]).all() or np.any(ref<low-1e-9) or np.any(ref>high+1e-9):raise ValueError('ordered action box must contain reference')
    if len(candidates)<2 or len({c.name for c in candidates})!=len(candidates):raise ValueError('unique candidate bank required')
    diagnostics=dict(ac_calls=0,uses_truth=False,uses_future_noise=False,controls=K,control_shape=[T,n],
        line_parameters_shared_H_N=True,source_interface=1,EV_in_total_PQ_once=True,failure=None)
    answer=dict(success=False,models=[],forecasts={},diagnostics=diagnostics)
    try:
        if hm.fit_bias or fm.fit_bias:raise ValueError('independently calibrated fixed biases required')
        if any(c.name not in fits or not fits[c.name].success or c.n!=n for c in candidates):raise ValueError('all required historical fits must succeed')
        line_bounds=dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1)) if line_bounds is None else dict(line_bounds)
        sb=dict(p_bounds=(-1.,1.),q_bounds=(-1.,1.),source_bounds=(-.5,.5));sb.update(state_bounds or {})
        ps=_box(sb['p_bounds'],n,'P');qs=_box(sb['q_bounds'],n,'Q');ss=_box(sb['source_bounds'],2,'source')
        state_lo=np.r_[ps[0],qs[0],ss[0]];state_hi=np.r_[ps[1],qs[1],ss[1]]
        channels=_channels(history);obs=whiten_observations(history,hm,channels);hrows=len(obs)
        baseline=baseline_from_history(history,hm,T);U=multisite_control_map(T,n,base_kw)
        contexts={};width=2*n+2
        for c in candidates:
            fit=fits[c.name];theta=np.r_[fit.r,fit.x]
            forecast=dict(p=baseline['p']+u0/base_kw,q=baseline['q'].copy(),root_v=baseline['root_v'].copy(),
                source_p=np.full(T,np.mean(fit.source_p)),source_q=np.full(T,np.mean(fit.source_q)))
            diagnostics['ac_calls']+=1
            hg=mean_and_jacobian(c,fit.r,fit.x,fit.p,fit.q,fit.source_p,fit.source_q,history.root_v,hm,channels=channels)
            diagnostics['ac_calls']+=1;ng=_future(c,theta,forecast,forecast['root_v'],fm,channels)
            mixed=mixed_control_matrix(c,theta,forecast,forecast['root_v'],fm,channels,U,step_kw=step_kw)
            diagnostics['ac_calls']+=mixed['ac_derivative_calls']
            rlo,rhi=_box(line_bounds['r_bounds'],c.m,'R');xlo,xhi=_box(line_bounds['x_bounds'],c.m,'X')
            center=np.r_[theta,state_vector(dict(p=fit.p,q=fit.q,source_p=fit.source_p,source_q=fit.source_q)),state_vector(forecast)]
            abslo=np.r_[rlo,xlo,np.tile(state_lo,history.t+T)];abshi=np.r_[rhi,xhi,np.tile(state_hi,history.t+T)]
            if np.any(center<abslo-1e-9) or np.any(center>abshi+1e-9):raise ValueError('center outside declared physical bounds: '+c.name)
            recost=float(np.sum((hg['mean']-obs)**2))
            if abs(recost-fit.loss)>1e-6*max(1.,fit.loss):raise ValueError('historical cost does not match complete AC')
            contexts[c.name]=dict(hg=hg,ng=ng,mixed=mixed,forecast=forecast,center=center,lo=abslo,hi=abshi)
            answer['forecasts'][c.name]=forecast
        for a in candidates:
            for b in candidates:
                if a.name==b.name:continue
                ca,cb=contexts[a.name],contexts[b.name];hg,ng=cb['hg'],cb['ng'];nr=len(ng['mean'])
                nl=2*b.m;nh=width*history.t;nn=width*T
                C=np.block([[hg['line_jacobian'],hg['state_jacobian'],np.zeros((hrows,nn))],
                    [-ng['line_jacobian'],np.zeros((nr,nh)),-ng['state_jacobian']]])
                B=np.vstack((np.zeros((hrows,K)),(ca['ng']['state_jacobian']-ng['state_jacobian'])@U))
                reference=np.r_[hg['mean']-obs,ca['ng']['mean']-ng['mean']]
                At=np.zeros((K,hrows+nr,nl+nh+nn));At[:,hrows:,:nl]=-cb['mixed']['line'];At[:,hrows:,nl+nh:]=-cb['mixed']['state']
                A0=C-np.einsum('k,krp->rp',ref,At);shift=np.zeros((len(cb['center']),K));shift[nl+nh:]=U
                domain=transform_domain(cb['center'],cb['lo']-cb['center'],cb['hi']-cb['center'],shift,ref,low,high)
                model=dict(name=a.name+'__to__'+b.name,a=a.name,b=b.name,c=reference-B@ref,B=B,A0=A0,At=At,
                    reference_kw=ref,reference_power_kw=u0.copy(),reference_residual=reference,C_at_reference=C,domain=domain,
                    channels=list(channels),history_rows=hrows,history_state_columns=nh,future_state_columns=nn,line_columns=nl,
                    base_kw=base_kw,control_shape=[T,n],control_time_site=[(k//n,k%n) for k in range(K)],
                    future_background=vector_state(state_vector(cb['forecast'])-U@ref,n),future_root_v=baseline['root_v'],
                    source_interface=1,original_domain_preserved_at_fixed_u=True,
                    reference_r=np.asarray(fits[a.name].r),reference_x=np.asarray(fits[a.name].x),
                    reference_background=vector_state(state_vector(ca['forecast'])-U@ref,n),
                    reference_history_cost=float(fits[a.name].loss),competitor_history_cost=float(fits[b.name].loss),
                    default_exchange_domain='original_absolute_parameter_box',global_AC_certificate=False)
                model['product_structure']={k:v for k,v in product_structure(model).items() if k!='raw'}
                answer['models'].append(model)
        diagnostics.update(seconds=perf_counter()-began,pairs=len(answer['models']),line_bounds=line_bounds,
            state_bounds=sb,source_prediction='candidate-specific historical mean at same physical node1')
        answer['success']=True
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        answer['models']=[];diagnostics.update(failure=type(error).__name__+':'+str(error),seconds=perf_counter()-began)
    return answer


def evaluate_multisite_predicted_action(history,candidates,hfits,hm,fm,jm,power_kw,
                                       *,base_kw=20.,fit_options=None,seed=0,weights=None):
    """Every H-only AC forecast refitted by every candidate; uniform mean default."""
    began=perf_counter();power=np.asarray(power_kw,float);candidates=tuple(candidates);names=[c.name for c in candidates]
    if power.ndim!=2 or power.shape[1]!=history.n or not len(power) or not np.isfinite(power).all():raise ValueError('power_kw must be finite (time,terminal)')
    if len(names)<2 or len(set(names))!=len(names):raise ValueError('all unique candidates required')
    ordered=[(a,b) for a in names for b in names if a!=b]
    value=np.full(len(ordered),1/len(ordered)) if weights is None else np.asarray(weights,float)
    if value.shape!=(len(ordered),) or not np.isfinite(value).all() or np.any(value<0) or not np.isclose(value.sum(),1.):raise ValueError('nonnegative task weights must sum to one')
    out=dict(success=False,power_kw=power,weights=value,weight_order=ordered,records=[],pairs=[],failures=[],
        fit_calls=0,fit_ac_calls=0,forecast_calls=0,mean_added_score=None,worst_added_score=None,
        uses_truth=False,uses_future_noise=False,future_added_noise=False,source_interface_shared=True,
        all_candidates_required=True,global_AC_certificate=False,score_meaning='weighted mean task value, not topology probability')
    if any(name not in hfits or not hfits[name].success for name in names):
        out['failures'].append(dict(kind='required_historical_fit_failure'));out['seconds']=perf_counter()-began;return out
    background=baseline_from_history(history,hm,len(power));lossH={name:float(hfits[name].loss) for name in names};allfits={}
    for index,reference in enumerate(candidates):
        a=reference.name;fit=hfits[a]
        forecast=dict(p=background['p']+power/base_kw,q=background['q'],root_v=background['root_v'],
            source_p=np.full(len(power),np.mean(fit.source_p)),source_q=np.full(len(power),np.mean(fit.source_q)))
        record=dict(a=a,forecast_state=forecast,fits=[],failures={},success=False)
        out['records'].append(record)
        try:
            out['forecast_calls']+=1
            phy=lk.ac_predict(reference,fit.r,fit.x,forecast['p'],forecast['q'],forecast['source_p'],forecast['source_q'],forecast['root_v'])
            vals={}
            for channel in (*CHANNELS,'source_p','source_q'):
                if channel not in ('p','q','v') and getattr(history,channel) is None:vals[channel]=None;continue
                vals[channel]=np.asarray(phy[channel])+(getattr(fm,'bias_'+channel) if channel in ('p','q','v') else 0.)
            obs=lk.Observations(**vals,root_v=forecast['root_v']);record['predicted_observations']=io.observations_record(obs)
            fitted,fail,cost=io.fit_all_candidates(candidates,lk.concat_observations(history,obs),jm,
                initial=hfits,seed=seed+1009*index,fit_options=fit_options)
            out['fit_calls']+=cost['fit_calls'];out['fit_ac_calls']+=cost['ac_calls']
            record.update(fits=[io.fit_record(c,fitted[c.name]) for c in candidates if c.name in fitted],failures=fail,cost=cost,
                success=not fail and len(fitted)==len(candidates))
            allfits[a]=fitted
            if not record['success']:out['failures'].append(dict(a=a,kind='required_joint_fit_failure',failures=fail))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            record['failures']['__forecast_or_joint__']=type(error).__name__+':'+str(error);allfits[a]={}
            out['failures'].append(dict(a=a,kind='forecast_or_joint_exception',message=str(error)))
    for k,(a,b) in enumerate(ordered):
        own=allfits.get(a,{}).get(a);comp=allfits.get(a,{}).get(b)
        ok=own is not None and comp is not None and own.success and comp.success
        row=dict(a=a,b=b,weight=float(value[k]),success=ok,added=None,joint=None,self_relative=None)
        if ok:row.update(joint_loss=float(comp.loss),self_loss=float(own.loss),
            added=.5*(comp.loss-lossH[b]),joint=.5*(comp.loss-lossH[a]),self_relative=.5*(comp.loss-own.loss))
        out['pairs'].append(row)
    out['success']=bool(not out['failures'] and all(r['success'] for r in out['pairs']))
    if out['success']:
        for key in ('added','joint','self_relative'):
            scores=np.asarray([r[key] for r in out['pairs']]);out['mean_'+key+'_score']=float(value@scores);out['worst_'+key+'_score']=float(np.min(scores))
        out['score']=out['mean_added_score']
    out.update(seconds=perf_counter()-began,ac_calls=out['fit_ac_calls']+out['forecast_calls'],
        required_fit_states=len(names)**2,returned_fit_states=sum(len(x) for x in allfits.values()))
    return out


def guard_multisite_predicted_actions(history,candidates,hfits,hm,fm,jm,baseline_kw,candidate_kw,
                                      *,base_kw=20.,fit_options=None,seed=0,weights=None,gain_tolerance=.01):
    """Accept proposal by uniform/nonnegative weighted mean H-only AC score."""
    baseline_kw,candidate_kw=np.asarray(baseline_kw,float),np.asarray(candidate_kw,float)
    if baseline_kw.shape!=candidate_kw.shape or gain_tolerance<0:raise ValueError('same-shaped actions and nonnegative tolerance required')
    kwargs=dict(base_kw=base_kw,fit_options=fit_options,seed=seed,weights=weights)
    baseline=evaluate_multisite_predicted_action(history,candidates,hfits,hm,fm,jm,baseline_kw,**kwargs)
    same=bool(np.array_equal(baseline_kw,candidate_kw));proposal=baseline if same else evaluate_multisite_predicted_action(history,candidates,hfits,hm,fm,jm,candidate_kw,**kwargs)
    complete=baseline['success'] and proposal['success'];gain=proposal['mean_added_score']-baseline['mean_added_score'] if complete else None
    accepted=bool(complete and gain>gain_tolerance);unique=[baseline] if same else [baseline,proposal]
    totals={k:sum(x.get(k,0) for x in unique) for k in ('fit_calls','fit_ac_calls','forecast_calls','ac_calls','returned_fit_states')}
    return dict(selected_power_kw=(candidate_kw if accepted else baseline_kw).copy(),accepted=accepted,
        baseline=baseline,candidate=proposal,score_gain=gain,gain_tolerance=gain_tolerance,cache_hit=same,
        evaluated_actions=len(unique),totals=totals,uses_truth=False,uses_future_noise=False,
        acceptance_scope='attained local complete-AC mean task score; not global optimality or future rejection guarantee')


def make_ac_context(model,history,candidates,hfits,hm,fm,*,voltage_limits=(.85,1.15)):
    """Runtime AC context from history only; accepts legacy single-site models."""
    byname={c.name:c for c in candidates};a,b=model['a'],model['b'];n=history.n;width=2*n+2
    T=int(model['future_state_columns'])//width
    if model['future_state_columns']!=T*width or a not in byname or b not in byname:raise ValueError('candidate/state dimensions disagree')
    if a not in hfits:raise ValueError('reference H fit missing')
    fit=hfits[a];background=baseline_from_history(history,hm,T)
    ref_bg=model.get('reference_background') or dict(p=background['p'],q=background['q'],
        source_p=np.full(T,np.mean(fit.source_p)),source_q=np.full(T,np.mean(fit.source_q)))
    return dict(reference=byname[a],competitor=byname[b],reference_r=np.asarray(model.get('reference_r',fit.r)),
        reference_x=np.asarray(model.get('reference_x',fit.x)),reference_background=ref_bg,
        reference_root=np.asarray(model['future_root_v']),history_root=np.asarray(history.root_v),
        history=history,hm=hm,fm=fm,channels=tuple(model['channels']),n=n,T=T,H=history.t,
        future_control_map=np.asarray(model['domain']['shift'])[-int(model['future_state_columns']):],
        white_history=whiten_observations(history,hm,tuple(model['channels'])),
        voltage_limits=tuple(voltage_limits),competitor_H_cost=float(hfits[b].loss),uses_truth=False,uses_future_noise=False)


def _split_absolute(model,physical,n):
    xi=np.asarray(physical,float);nl=int(model['line_columns']);nh=int(model['history_state_columns'])
    return xi[:nl],vector_state(xi[nl:nl+nh],n),vector_state(xi[nl+nh:],n)


def evaluate_ac_witness(model,u,physical,context):
    """Compare proxy and complete AC at the SAME original-domain explanation."""
    u=np.asarray(u,float).ravel();xi=np.asarray(physical,float);domain=model['domain']
    violation=float(max(0.,np.max(np.asarray(domain['physical_lower'])-xi),np.max(xi-np.asarray(domain['physical_upper']))))
    if xi.shape!=np.asarray(domain['base_center']).shape or violation>1e-9:raise ValueError('witness not in original physical box')
    b=context['competitor'];a=context['reference'];theta,wh,wn=_split_absolute(model,xi,context['n'])
    hg=_future(b,theta,wh,context['history_root'],context['hm'],context['channels'],False)
    ng=_future(b,theta,wn,context['reference_root'],context['fm'],context['channels'],False)
    reference_state=vector_state(state_vector(context['reference_background'])+context['future_control_map']@u,context['n'])
    ref=_future(a,np.r_[context['reference_r'],context['reference_x']],reference_state,
        context['reference_root'],context['fm'],context['channels'],False)
    exact=np.r_[hg['mean']-context['white_history'],ref['mean']-ng['mean']]
    z=parameter_increment(model,u,xi);proxy=moving_residual(model,u,z);error=proxy-exact;hrows=model['history_rows']
    hist=channel_norms(error[:hrows],context['channels'],context['H'],context['n'])
    fut=channel_norms(error[hrows:],context['channels'],context['T'],context['n'])
    hR={k:v['norm']/np.sqrt(v['rows']) for k,v in hist.items()};nR={k:v['norm']/np.sqrt(v['rows']) for k,v in fut.items()}
    ranges={key:[float(np.min(abs(g['physical']['voltage']))),float(np.max(abs(g['physical']['voltage'])))]
        for key,g in (('competitor_history',hg),('competitor_future',ng),('reference_future',ref))}
    vlo,vhi=context['voltage_limits'];voltage_valid=all(lo>=vlo and hi<=vhi for lo,hi in ranges.values())
    return dict(physical_box_feasible=violation<=1e-9,physical_box_violation=violation,
        complete_AC_voltage_feasible=voltage_valid,voltage_ranges=ranges,
        complete_AC_residual=exact,proxy_residual=proxy,proxy_minus_AC=error,
        complete_AC_cost=float(exact@exact),proxy_cost=float(proxy@proxy),cost_error=float(proxy@proxy-exact@exact),
        error_norm=float(np.linalg.norm(error)),history_error_norm=float(np.linalg.norm(error[:hrows])),
        future_error_norm=float(np.linalg.norm(error[hrows:])),history_channel_RMS=hR,future_channel_RMS=nR,
        max_channel_RMS=max(*hR.values(),*nR.values()),strict_RMS_025_pass=all(v<=.25 for v in (*hR.values(),*nR.values())),
        complete_AC_H_cost=float(np.sum(exact[:hrows]**2)),complete_AC_N_cost=float(np.sum(exact[hrows:]**2)),
        source_interface=1,source_meter_channels=[k for k in context['channels'] if k.startswith('source_')],
        reference_future_state=reference_state,ac_calls=3,fit_calls=0,
        scope='same absolute witness; neither proxy nor nonlinear inner optimum certified')


def recenter_at_absolute(model,u_anchor,physical,context,*,step_kw=.001):
    """New proxy at a legal H-only explanation; preserve the original box.

    Recentring defines a NEW model and requires new optimization bounds. It
    does not turn the union/minimum of local proxies into an AC certificate.
    """
    anchor=np.asarray(u_anchor,float).ravel();xi=np.asarray(physical,float);domain=model['domain']
    low=np.asarray(domain['physical_lower']);high=np.asarray(domain['physical_upper'])
    if xi.shape!=low.shape or np.any(xi<low-1e-9) or np.any(xi>high+1e-9):raise ValueError('recenter witness outside original box')
    b=context['competitor'];a=context['reference'];theta,wh,wn=_split_absolute(model,xi,context['n'])
    hg=_future(b,theta,wh,context['history_root'],context['hm'],context['channels'])
    ng=_future(b,theta,wn,context['reference_root'],context['fm'],context['channels'])
    reference_state=vector_state(state_vector(context['reference_background'])+context['future_control_map']@anchor,context['n'])
    ag=_future(a,np.r_[context['reference_r'],context['reference_x']],reference_state,
        context['reference_root'],context['fm'],context['channels'])
    U=context['future_control_map'];mixed=mixed_control_matrix(b,theta,wn,context['reference_root'],context['fm'],context['channels'],U,step_kw=step_kw)
    nl=model['line_columns'];nh=model['history_state_columns'];nn=model['future_state_columns'];hrows=model['history_rows'];nr=len(ng['mean'])
    C=np.block([[hg['line_jacobian'],hg['state_jacobian'],np.zeros((hrows,nn))],
        [-ng['line_jacobian'],np.zeros((nr,nh)),-ng['state_jacobian']]])
    B=np.vstack((np.zeros((hrows,len(anchor))),(ag['state_jacobian']-ng['state_jacobian'])@U))
    ref=np.r_[hg['mean']-context['white_history'],ag['mean']-ng['mean']]
    At=np.zeros_like(np.asarray(model['At']));At[:,hrows:,:nl]=-mixed['line'];At[:,hrows:,nl+nh:]=-mixed['state']
    shifted=transform_domain(xi,low-xi,high-xi,np.asarray(domain['shift']),anchor,domain['action_lower'],domain['action_upper'])
    answer=dict(model);answer.update(c=ref-B@anchor,B=B,A0=C-np.einsum('k,krp->rp',anchor,At),At=At,
        reference_kw=anchor,reference_residual=ref,C_at_reference=C,domain=shifted,
        future_background=vector_state(state_vector(wn)-U@anchor,context['n']),
        expanded_at_absolute=xi,expanded_at_control=anchor,
        expansion_history_cost=float(np.sum(ref[:hrows]**2)),
        expansion_history_cost_minus_attained=float(np.sum(ref[:hrows]**2)-context['competitor_H_cost']),
        original_physical_domain_unchanged=True,new_proxy_requires_new_bounds=True,
        expansion_ac_calls=3+mixed['ac_derivative_calls'])
    answer['product_structure']={k:v for k,v in product_structure(answer).items() if k!='raw'}
    return answer
