"""Independent saved-record review: no selector, optimizer, or new AC fitting.

Replays AC by branch currents and forward voltage drops, profiles the saved
linear matrices independently, and keeps scale/floor repetitions grouped by
base seed. Solver bounds remain saved floating-point numerical bounds.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np
from .information_fit_recovery import read_json, write_json, independent_replay
from .information_replay import _obs, _meter
from .ev_flexibility_replay import fit_audits, close, require
from .likelihood import RadialCandidate, concat_observations

HERE=Path(__file__).resolve().parent
METHODS=('none','constant','profile_milp','fixed_mle_milp')
CHANNELS=('p','q','v','master_p','master_q')


def _array(value):return np.asarray(value,float)


def _white(obs,meter):
    return np.concatenate([(_array(getattr(obs,k))/np.broadcast_to(
        _array(getattr(meter,k+'_sigma')),_array(getattr(obs,k)).shape)).ravel() for k in CHANNELS])


def score(model,u):
    residual=_array(model['d'])+_array(model['T'])@_array(u)
    return .5*(float(residual@residual)-float(model['offset']))


def domain_covered(supports,r,x,floor):
    """Physical truth coverage, distinct from local projection feasibility."""
    n=max(max(s) for s in supports)+1
    mask=np.array([1<len(s)<n for s in supports])
    rl=np.where(mask,floor,1e-6);xl=np.where(mask,max(1e-6,2*floor/3),1e-6)
    r,x=_array(r),_array(x)
    require(r.shape==rl.shape and x.shape==xl.shape,'parameter support shape')
    return bool(np.all(r>=rl) and np.all(x>=xl) and np.all(r<=.1) and np.all(x<=.1))


def replay_projection(model,curves):
    full=model['full'];c,B,C=(_array(full[k]) for k in ('c','B','C'))
    d,T=_array(model['d']),_array(model['T'])
    norms=np.linalg.norm(C,axis=0);scale=np.ones(C.shape[1]);scale[norms>0]=1/norms[norms>0]
    A=C*scale
    # Direct least squares for all affine columns together; no producer helper.
    W=np.column_stack((B,c));comp=np.linalg.lstsq(A,-W,rcond=1e-10)[0]
    residual=W+A@comp
    compressed=np.column_stack((T,d))
    close(compressed.T@compressed,residual.T@residual,'profile compressed Gram',atol=2e-7,rtol=2e-8)
    if 'reference_residual' in full:
        close(c+B@_array(model['reference_kw']),full['reference_residual'],'reference intercept',atol=2e-8)
    if C.shape[1]:
        close(full['scale'],scale,'column normalization',atol=1e-12)
        require(len(full['parameter_center'])==C.shape[1],'parameter center dimension')
        require(np.all(_array(full['lower'])<_array(full['upper'])),'ordered increment bounds')
    rows=[]
    for name,u in curves.items():
        u=_array(u);rhs=c+B@u
        step=scale*np.linalg.lstsq(A,-rhs,rcond=1e-10)[0]
        q=float(np.sum((rhs+C@step)**2));compressed_q=float(np.sum((d+T@u)**2))
        close(q,compressed_q,'profile value '+name,atol=2e-7,rtol=2e-8)
        rows.append(dict(method=name,q=q,score=.5*(q-float(model['offset']))))
    return dict(success=True,name=model['name'],rows=rows,residual_rows=len(c),parameter_columns=C.shape[1])


def replay_bounded(model,u,saved):
    full=model['full'];c,B,C=(_array(full[k]) for k in ('c','B','C'))
    z=_array(saved['step']);lo,hi=(_array(full[k]) for k in ('lower','upper'))
    require(z.shape==lo.shape==hi.shape,'bounded step dimensions')
    feasible=bool(np.all(z>=lo-1e-10) and np.all(z<=hi+1e-10))
    require(feasible==saved['feasible'],'bounded feasibility flag')
    residual=c+B@_array(u)+C@z;value=float(residual@residual)
    gradient=2*C.T@residual;corner=np.where(gradient>=0,lo,hi)
    lower=min(value,max(0.,value+float(gradient@(corner-z))))
    close(saved['primal_upper'],value,'bounded feasible upper',atol=2e-7,rtol=2e-8)
    close(saved['tangent_lower_numeric'],lower,'bounded tangent lower',atol=2e-6,rtol=2e-7)
    free=_array(saved['free_step']);free_r=c+B@_array(u)+C@free
    free_q=float(free_r@free_r)
    close(saved['free_value'],free_q,'saved free explanation',atol=2e-7,rtol=2e-8)
    violation=float(max(0.,np.max(lo-free),np.max(free-hi)))
    close(saved['free_bound_violation'],violation,'free physical violation',atol=2e-8)
    close(saved['parameter'],_array(full['parameter_center'])+z,'bounded actual parameter')
    return dict(feasible=feasible,upper=value,lower=lower,free=free_q,
        bounded_minus_free=value-free_q,free_violation=violation)


def replay_parameter_domain(model,candidate,fit,forecast,floor):
    full=model['full'];m=len(candidate['supports']);n=3
    center=np.r_[_array(fit['r']),_array(fit['x']),
        np.column_stack([_array(fit[k]) for k in ('p','q','source_p','source_q')]).ravel(),
        np.column_stack([_array(forecast[k]) for k in ('p','q','source_p','source_q')]).ravel()]
    close(full['parameter_center'],center,'full physical center')
    mask=np.array([1<len(s)<n for s in candidate['supports']])
    lower=np.r_[np.where(mask,floor,1e-6),np.where(mask,max(1e-6,2*floor/3),1e-6),
        np.tile(np.r_[np.full(2*n,-1.),-.5,-.5],len(fit['p'])+len(forecast['p']))]
    upper=np.r_[np.full(2*m,.1),np.tile(np.r_[np.full(2*n,1.),.5,.5],len(fit['p'])+len(forecast['p']))]
    close(_array(full['lower'])+center,lower,'absolute lower bound from increments')
    close(_array(full['upper'])+center,upper,'absolute upper bound from increments')
    require(np.all(center>=lower-1e-9) and np.all(center<=upper+1e-9),'center in original physical domain')
    return True


def replay_solve(models,saved):
    require(saved['u'] is not None,'saved MILP returned no incumbent')
    u=_array(saved['u']);require(u.shape==(4,) and np.all(u>=-1e-7) and np.all(u<=6+1e-7),'MILP power constraints')
    close(float(u.sum()),12.,'MILP energy constraint',atol=1e-7)
    values=np.array([score(m,u) for m in models]);lower=float(values.min())
    close(saved['quadratic_values'],values,'MILP actual quadratic values',atol=2e-7,rtol=2e-8)
    close(saved['objective_lower'],lower,'MILP attained lower',atol=2e-7,rtol=2e-8)
    upper=float(saved['objective_upper']);require(upper>=lower-2e-7,'MILP upper below attained lower')
    close(saved['original_gap'],max(0.,upper-lower),'MILP original gap',atol=2e-7)
    nseg=int(saved['segments']);errors=[];chords=[]
    for m in models:
        d=_array(m['d'])/np.sqrt(2);T=_array(m['T'])/np.sqrt(2)
        lo=d+np.minimum(T,0).sum(axis=1)*6;hi=d+np.maximum(T,0).sum(axis=1)*6
        x=d+T@u;error=0.;total=0.
        for l,h,v in zip(lo,hi,x):
            if abs(h-l)<1e-13:total+=l*l;continue
            knots=np.linspace(l,h,nseg+1)
            total+=float(np.interp(v,knots,knots**2));error+=(h-l)**2/(4*nseg*nseg)
        chords.append(total-.5*float(m['offset']));errors.append(error)
    close(saved['secant_error_by_quadratic'],errors,'secant error bound',atol=2e-7,rtol=2e-8)
    close(saved['chord_value_at_returned_u'],min(chords),'interpolated chord score',atol=2e-7,rtol=2e-8)
    require(min(chords)>=lower-2e-7 and min(chords)-lower<=max(errors)+2e-7,'chord pointwise sandwich')
    return dict(lower=lower,upper=upper,gap=upper-lower,segments=nseg,
        secant_error=max(errors),solver_success=saved['solver_success'],
        scope='independent incumbent/chord arithmetic; saved solver dual bound not re-solved')


def _candidate_map(trial):
    result={}
    for row in trial['proposal']['bank']:
        c=RadialCandidate.from_supports(row['name'],3,row['supports'])
        result[c.name]=dict(name=c.name,supports=[list(s) for s in c.supports])
    require(len(result)==4,'four local candidates retained')
    return result


def _fit_domain(fit,floor):
    r,x=_array(fit['r']),_array(fit['x'])
    if not fit['success'] or not np.isfinite(r).all():return
    mask=np.array([1<len(s)<3 for s in fit['supports']])
    require(np.all(r>=np.where(mask,floor,1e-6)-1e-9),'fit internal R prior')
    require(np.all(x>=np.where(mask,max(1e-6,2*floor/3),1e-6)-1e-9),'fit internal X prior')


def _forward_mean(candidate,r,x,p,q,sp,sq,root,meter):
    """Independent backward/forward AC mean; no production AC solver calls."""
    p,q,sp,sq,root=map(_array,(p,q,sp,sq,root));z=_array(r)+1j*_array(x)
    loads=np.zeros((len(p),candidate.m),complex);loads[:,0]=sp+1j*sq
    loads[:,candidate.terminal_edges]+=p+1j*q
    voltage=np.broadcast_to(root[:,None],loads.shape).astype(complex).copy()
    for _ in range(500):
        current=np.conj(loads/voltage);branch=current.copy()
        for edge in range(candidate.m-1,-1,-1):
            parent=candidate.parents[edge]
            if parent>=0:branch[:,parent]+=branch[:,edge]
        new=np.empty_like(voltage)
        for edge,parent in enumerate(candidate.parents):new[:,edge]=(root if parent<0 else new[:,parent])-z[edge]*branch[:,edge]
        error=float(np.max(abs(new-voltage)));voltage=new
        if error<2e-13:break
    else:raise ValueError('independent future mean AC convergence failure')
    master=root*np.conj(np.sum(np.conj(loads/voltage),axis=1))
    values=dict(p=p+_array(meter.bias_p),q=q+_array(meter.bias_q),
        v=abs(voltage[:,candidate.terminal_edges])+_array(meter.bias_v),
        master_p=master.real,master_q=master.imag,root_v=root,source_p=None,source_q=None)
    return _obs(values)


def _physical_replay(trial,row,fm):
    truth=trial['truth'];u=_array(row['u_kw']);p=_array(truth['p'])[12:].copy();p[:,0]+=u/20
    q=_array(truth['q'])[12:];phy=row['physical']
    close(phy['p'],p,'EV once');close(phy['q'],q,'future Q')
    means={k:_array(phy[k])+(_array(getattr(fm,'bias_'+k)) if k in ('p','q','v') else 0.) for k in CHANNELS}
    obs=_obs(row['observations'])
    for k in CHANNELS:
        close(getattr(obs,k),means[k]+_array(getattr(fm,k+'_sigma'))*_array(trial['white_future'][k]),'future meter '+k,atol=2e-10)
    c=RadialCandidate.from_supports('truth',3,truth['supports'])
    numerical=dict(r=truth['r_pu'],x=truth['x_pu'],p=p,q=q,
        source_p=_array(truth['source_p'])[12:],source_q=_array(truth['source_q'])[12:],predicted=means,
        **{k:getattr(fm,k) for k in ('bias_p','bias_q','bias_v')})
    zero=_obs(dict(**means,root_v=np.ones(4),source_p=None,source_q=None))
    out=independent_replay(c,numerical,zero,fm)
    require(out['success'] and out['full_SSE']<1e-12,'truth physical replay')
    close(row['voltage_range'],[out['voltage_min'],out['voltage_max']],'physical voltage bounds')
    return out


def audit_trial(trial):
    candidates=_candidate_map(trial);h=_obs(trial['history'])
    hm,fm,jm=(_meter(trial['measurement_models'][k]) for k in ('history','future','joint'))
    floor=float(trial['fit_floor']);truth=trial['truth']
    true_c=RadialCandidate.from_supports('truth',3,truth['supports'])
    covered=domain_covered(true_c.supports,truth['r_pu'],truth['x_pu'],floor)
    require(covered==trial['truth_in_parameter_domain'],'truth coverage mismatch')
    require(trial['selection']['uses_truth'] is False and trial['selection']['uses_future_noise'] is False,'selector isolation flags')
    require(trial['old_threshold_applied'] is False,'old threshold reused')
    hist=fit_audits(trial['historical_fits'],candidates,h,hm,trial['historical_failures'],complete=not trial['historical_failures'])
    for f in trial['historical_fits']:_fit_domain(f,floor)
    states=hist['states_replayed'];physical=0;methods=[];curves={}
    require(len(trial['rows'])==4 and {r['method'] for r in trial['rows']}==set(METHODS),'four method denominator')
    for row in trial['rows']:
        method=row['method'];u=_array(row['u_kw']);curves[method]=u
        if method=='none':close(u,np.zeros(4),'no EV curve')
        else:
            require(u.shape==(4,) and np.all(u>=-1e-7) and np.all(u<=6+1e-7),'curve power bound')
            close(u.sum()*.25,3.,'equal grid energy',atol=1e-7)
        public_quality=_array(trial['quality']['v_repeat_counts']).sum(axis=1)
        quality_score=float(public_quality@u);quality_max=float(6*np.sort(public_quality)[-2:].sum())
        result=dict(method=method,status=row['status'],covered=covered,
            public_quality_linear_score=quality_score,public_quality_optimal_score=quality_max,
            matches_public_quality_optimum=bool(method!='none' and np.isclose(quality_score,quality_max,atol=1e-7,rtol=1e-9)))
        if 'observations' in row:
            _physical_replay(trial,row,fm);physical+=1
            joint=concat_observations(h,_obs(row['observations']))
            replay=fit_audits(row['fits'],candidates,joint,jm,row['fit_failures'],complete=row['status']=='complete')
            states+=replay['states_replayed']
            for f in row['fits']:_fit_domain(f,floor)
            if row['status']=='complete':
                loss={f['name']:float(f['loss']) for f in row['fits']};true=row['true_name']
                require(candidates[true]['supports']==[list(s) for s in true_c.supports],'true candidate identity')
                margin=min(v for n,v in loss.items() if n!=true)-loss[true]
                close(row['margin'],margin,'actual full AC margin')
                winner=min(loss,key=loss.get);require(row['winner']==winner,'actual winner')
                result.update(margin=margin,winner=winner,true_name=true,correct_winner=winner==true)
        methods.append(result)
    context=trial.get('proxy_models');proxy=[];solves={};local=[]
    if context:
        require(context['success'],'saved context marked failed')
        for model in context['models']:
            fitb=next(f for f in trial['historical_fits'] if f['name']==model['b'])
            fita=next(f for f in trial['historical_fits'] if f['name']==model['a'])
            replay_parameter_domain(model,candidates[model['b']],fitb,model['forecast_b'],floor)
            replay_parameter_domain(model['self_model'],candidates[model['a']],fita,model['forecast_a'],floor)
            proxy.append(replay_projection(model,curves));proxy.append(replay_projection(model['self_model'],curves))
            selfm=model['self_model']
            require(np.linalg.norm(_array(selfm['T']))<model['self_profile']['tolerance'],'self derivative constant test')
            close(float(_array(selfm['d'])@_array(selfm['d'])),model['offset'],'self score offset')
        for model in context['fixed_models']:proxy.append(replay_projection(model,curves))
        for method,key in [('profile_milp','models'),('fixed_mle_milp','fixed_models')]:
            saved=trial['selection']['solves'].get(method)
            if saved and saved.get('u') is not None:solves[method]=replay_solve(context[key],saved)
        for method,u in curves.items():
            expected=dict(profile=min(score(m,u) for m in context['models']),fixed=min(score(m,u) for m in context['fixed_models']))
            for key,val in expected.items():close(trial['selection']['curve_scores'][method][key],val,'saved curve score')
        bypair={(m['a'],m['b']):m for m in context['models']}
        for row in trial['audit']:
            m=bypair[(row['a'],row['b'])];u=curves[row['method']]
            close(row['free_score'],score(m,u),'audit free score')
            entry=dict(method=row['method'],a=row['a'],b=row['b'],free_score=row['free_score'])
            if 'bounded' in row:
                other=replay_bounded(m,u,row['bounded']);own=replay_bounded(m['self_model'],u,row['bounded_self'])
                low=.5*(other['lower']-own['upper']);high=.5*(other['upper']-own['lower'])
                close(row['bounded_score_lower'],low,'bounded score lower',atol=2e-6)
                close(row['bounded_score_upper'],high,'bounded score upper',atol=2e-6)
                entry.update(bounded_lower=low,bounded_upper=high,
                    competitor_bounded_minus_free=other['bounded_minus_free'],competitor_free_violation=other['free_violation'])
            entry.update(forward_error=row['source_forward_linear_error_norm'],
                forward_relative_error=row['source_forward_relative_linear_error'])
            source=next(f for f in trial['historical_fits'] if f['name']==row['a'])
            forecast=m['forecast_a'];p=_array(forecast['p']).copy();p[:,m['site']]+=(u-_array(m['reference_kw']))/m['base_kw']
            candidate=RadialCandidate.from_supports(row['a'],3,candidates[row['a']]['supports'])
            independent_mean=_forward_mean(candidate,source['r'],source['x'],p,forecast['q'],forecast['source_p'],forecast['source_q'],forecast['root_v'],fm)
            ga=m['geometry_a']['future'];G=_array(ga['state_jacobian'])[:,np.arange(4)*8+m['site']]/m['base_kw']
            exact=_white(independent_mean,fm);base=_array(ga['mean'])
            err=float(np.linalg.norm(exact-base-G@(u-_array(m['reference_kw']))));response=float(np.linalg.norm(exact-base))
            close(row['source_forward_linear_error_norm'],err,'all-condition independent Taylor error',atol=2e-8)
            close(row['source_forward_response_norm'],response,'all-condition independent response norm',atol=2e-8)
            if response>1e-10:close(row['source_forward_relative_linear_error'],err/response,'all-condition relative Taylor error',atol=2e-8)
            if 'predicted_joint_fits' in row:
                predicted=_obs(row['predicted_observations']);joint=concat_observations(h,predicted)
                selected={n:candidates[n] for n in (row['a'],row['b'])}
                audit=fit_audits(row['predicted_joint_fits'],selected,joint,jm,row['predicted_joint_failures'],complete=not row['predicted_joint_failures'])
                states+=audit['states_replayed']
                for f in row['predicted_joint_fits']:_fit_domain(f,floor)
                if not row['predicted_joint_failures']:
                    losses={f['name']:f['loss'] for f in row['predicted_joint_fits']}
                    margin=.5*(losses[row['b']]-losses[row['a']])
                    close(row['nonlinear_predicted_margin_half_sse'],margin,'predicted nonlinear margin')
                    entry['nonlinear_predicted_margin_half_sse']=margin
                # Original source derivative is saved, but the nonlinear forward
                # mean itself is independently checked by branch-current replay.
                source=next(f for f in trial['historical_fits'] if f['name']==row['a'])
                forecast=m['forecast_a'];p=_array(forecast['p']).copy();p[:,m['site']]+=(u-_array(m['reference_kw']))/m['base_kw']
                numeric={**source,'p':p,'q':forecast['q'],'source_p':forecast['source_p'],'source_q':forecast['source_q'],
                    'predicted':{k:getattr(predicted,k) for k in CHANNELS}}
                numeric.pop('loss',None);numeric.pop('loglik',None)
                candidate=RadialCandidate.from_supports(row['a'],3,candidates[row['a']]['supports'])
                replay=independent_replay(candidate,numeric,predicted,fm)
                require(replay['success'] and replay['full_SSE']<1e-12,'predicted source nonlinear mean replay')
                ga=m['geometry_a']['future'];G=_array(ga['state_jacobian'])[:,np.arange(4)*8+m['site']]/m['base_kw']
                exact=_white(predicted,fm);base=_array(ga['mean']);err=float(np.linalg.norm(exact-base-G@(u-_array(m['reference_kw']))))
                response=float(np.linalg.norm(exact-base))
                close(row['source_forward_linear_error_norm'],err,'forward Taylor error',atol=2e-8)
                close(row['source_forward_response_norm'],response,'forward response norm',atol=2e-8)
                if response>1e-10:close(row['source_forward_relative_linear_error'],err/response,'forward relative Taylor error',atol=2e-8)
            local.append(entry)
    rows={m['method']:m for m in methods};delta=None
    if all('margin' in rows[k] for k in ('constant','profile_milp')):delta=rows['profile_milp']['margin']-rows['constant']['margin']
    true_name=next(n for n,c in candidates.items() if c['supports']==[list(s) for s in true_c.supports])
    ranking=sorted(trial['historical_fits'],key=lambda f:(f['loss'],f['name']))
    selected_pair=trial['selection'].get('context_diagnostics',{}).get('selected_pair',[])
    return dict(success=True,seed=trial['seed'],truth_r=trial['truth_r'],floor=floor,covered=covered,
        true_name=true_name,history_true_rank=next(i+1 for i,f in enumerate(ranking) if f['name']==true_name),
        selected_pair=selected_pair,pair_includes_truth=true_name in selected_pair,
        historical_losses={f['name']:f['loss'] for f in ranking},
        selection_fallback=trial['selection']['fallback'],activation_would_occur=trial['selection'].get('activation_would_occur'),
        saved_fit_states_replayed=states,physical_responses_replayed=physical,methods=methods,
        profile_minus_constant_margin=delta,proxies=proxy,milp=solves,local_audit=local,
        condition_seconds=trial['seconds'])


def summarize(reviews):
    require(bool(reviews),'at least one completed condition required')
    counts=Counter(r['seed'] for r in reviews);groups=[]
    for covered in (True,False):
        subset=[r for r in reviews if r['covered']==covered]
        rows=[]
        for method in METHODS:
            items=[next(x for x in r['methods'] if x['method']==method) for r in subset]
            good=[x for x in items if 'margin' in x]
            rows.append(dict(method=method,conditions=len(items),complete=len(good),
                correct_winners=sum(x['correct_winner'] for x in good),
                matches_public_quality_optimum=sum(x.get('matches_public_quality_optimum',False) for x in items),
                mean_margin=float(np.mean([x['margin'] for x in good])) if good else None,
                failures=dict(Counter(x['status'] for x in items if x['status']!='complete'))))
        delta=[r['profile_minus_constant_margin'] for r in subset if r['profile_minus_constant_margin'] is not None]
        groups.append(dict(covered=covered,conditions=len(subset),base_seeds=len({r['seed'] for r in subset}),methods=rows,
            profile_minus_constant=dict(n_conditions=len(delta),positive=sum(x>1e-8 for x in delta),negative=sum(x < -1e-8 for x in delta),
                mean=float(np.mean(delta)) if delta else None)))
    local=[x for r in reviews for x in r['local_audit']] if all('local_audit' in r for r in reviews) else []
    relative=[x['forward_relative_error'] for x in local if x.get('forward_relative_error') is not None]
    return dict(completed_conditions=len(reviews),independent_base_seeds=len(counts),
        repeated_conditions_by_seed=dict(counts),groups=groups,
        selection_fallbacks=sum(r['selection_fallback'] for r in reviews),
        selected_pair_misses_truth=sum(bool(r.get('selected_pair')) and not r.get('pair_includes_truth',False) for r in reviews),
        activation_would_occur=sum(r['activation_would_occur'] is True for r in reviews),
        saved_fit_states_replayed=sum(r['saved_fit_states_replayed'] for r in reviews),
        physical_responses_replayed=sum(r['physical_responses_replayed'] for r in reviews),
        forward_relative_error_range=[float(min(relative)),float(max(relative))] if relative else None,
        nonlinear_mean_comparison_count=sum('nonlinear_predicted_margin_half_sse' in x for x in local),
        statistical_claim='descriptive development conditions; scale/floor rows are repeated base histories')


def floor_monotonicity(trials):
    """Same-data H losses should increase for exact minima as domains shrink.

    A violation by attained local fits is a computational diagnostic. Future
    fitted losses are compared only when both observations and curves match.
    """
    groups=defaultdict(list)
    for t in trials:groups[(t['seed'],t['truth_r'])].append(t)
    checks=[];violations=[]
    for (seed,r),group in groups.items():
        ordered=sorted(group,key=lambda t:t['fit_floor'])
        for left,right in zip(ordered[:-1],ordered[1:]):
            require(left['history']==right['history'],'same-truth H changes across floor')
            pairs=[('history',left['historical_fits'],right['historical_fits'])]
            for a,b in zip(left['rows'],right['rows']):
                if a['method']==b['method'] and a.get('observations')==b.get('observations') and a.get('u_kw')==b.get('u_kw') and 'fits' in a and 'fits' in b:
                    pairs.append((a['method'],a['fits'],b['fits']))
            for method,lf,rf in pairs:
                lm={f['name']:f for f in lf};rm={f['name']:f for f in rf}
                for name in sorted(lm.keys()&rm.keys()):
                    a,b=lm[name],rm[name]
                    if not a['success'] or not b['success']:continue
                    delta=float(b['loss']-a['loss']);tol=1e-6+1e-7*max(abs(a['loss']),abs(b['loss']))
                    item=dict(seed=seed,truth_r=r,method=method,candidate=name,
                        low_floor=left['fit_floor'],high_floor=right['fit_floor'],loss_delta=delta,tolerance=tol,
                        violated=delta < -tol)
                    checks.append(item)
                    if item['violated']:violations.append(item)
    return dict(checks=len(checks),violations=violations,violation_count=len(violations),
        interpretation='nested-domain exact minima are monotone; violations concern attained local fits')


def run_review(directory,output):
    directory=Path(directory);output=Path(output);freeze=read_json(directory/'freeze.json')
    frozen=freeze['hashes'];sources=[]
    for name,expected in frozen.items():
        path=(HERE/name).resolve();actual=hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual==expected,'current source differs from frozen hash:'+name)
        sources.append(dict(name=name,sha256=actual))
    files=sorted(directory.glob('trial_*.json.gz'));reviews=[];filehash=[];trials=[]
    require(bool(files),'no finished trial files')
    for path in files:
        data=path.read_bytes();t=read_json(path)
        require(t['source_sha256']==frozen,'trial frozen source map:'+path.name)
        require(t['config']==freeze['config'],'trial frozen configuration:'+path.name)
        review=audit_trial(t);review['file']=path.name;reviews.append(review);trials.append(t)
        filehash.append(dict(file=path.name,sha256=hashlib.sha256(data).hexdigest()))
    result=dict(success=True,summary=summarize(reviews),conditions=reviews,files=filehash,
        floor_monotonicity=floor_monotonicity(trials),
        source_verification=dict(success=True,count=len(sources),files=sources,
            source_byte_snapshot_present=(directory/'source_snapshot.json.gz').exists(),
            verification='current source hashes and every trial equal pre-run freeze; snapshot presence reported separately'),
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_selector_calls=0,new_fits=0,scope='saved-record independent replay; no formal risk threshold')
    write_json(output/'review.json',result)
    s=result['summary'];lines=['# Continuous-profile MILP independent review','',
        f"Completed {s['completed_conditions']} repeated conditions from {s['independent_base_seeds']} independent base seeds.",
        f"Replayed {s['saved_fit_states_replayed']} saved fit states and {s['physical_responses_replayed']} actual future responses. No new fits or selectors.",
        f"Selection fallbacks: {s['selection_fallbacks']}; H ambiguity activation: {s['activation_would_occur']}.",'',
        '|Covered|Method|Complete / conditions|Correct winner|Mean AC margin|Quality-LP match|','|---|---|---:|---:|---:|---:|']
    for group in s['groups']:
        for row in group['methods']:
            mean='NA' if row['mean_margin'] is None else f"{row['mean_margin']:.6g}"
            lines.append(f"|{group['covered']}|{row['method']}|{row['complete']} / {row['conditions']}|{row['correct_winners']}|{mean}|{row['matches_public_quality_optimum']}|")
    lines+=['','Correct winner means smallest attained AC loss, without a calibrated rejection threshold. Excluded-truth conditions are model-misspecification diagnostics.','',
        '|Seed|Truth R|Fit floor|Covered|NoEV M|Constant M|Profile M|FixedMLE M|Profile minus constant|Fallback|','|---:|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for row in reviews:
        delta='NA' if row['profile_minus_constant_margin'] is None else f"{row['profile_minus_constant_margin']:.6g}"
        values={x['method']:('NA' if 'margin' not in x else f"{x['margin']:.6g}") for x in row['methods']}
        lines.append(f"|{row['seed']}|{row['truth_r']}|{row['floor']}|{row['covered']}|{values['none']}|{values['constant']}|{values['profile_milp']}|{values['fixed_mle_milp']}|{delta}|{row['selection_fallback']}|")
    lines+=['',f"H-selected pair excludes truth in {s['selected_pair_misses_truth']} conditions; all four candidates still enter final AC refits.",
        f"Nested-floor checks: {result['floor_monotonicity']['checks']}; attained-local-fit monotonicity violations: {result['floor_monotonicity']['violation_count']}.",
        f"Fixed-parameter source forward Taylor relative errors: {s['forward_relative_error_range']}. Nonlinear competing-mean refit comparisons: {s['nonlinear_mean_comparison_count']} (first base seed only).",
        'Small source forward error does not measure the approximation error after a competitor changes its shared lines and states. The selected pair can also omit the true candidate; these are separate diagnostic questions.',
        'Quality-LP match means the action attains the best public repeat-count linear score at the same site and energy; it is not a separate fitted strategy.',
        'The JSON retains per-condition proxy scores, bounded/free discrepancies, nonlinear predicted margins, Taylor errors and MILP gaps. No global nonlinear AC certificate is asserted.','']
    output.mkdir(parents=True,exist_ok=True);(output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=HERE/'evidence/ev_profile_milp/v1')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run_review(a.directory,a.output);print(r['summary'])


if __name__=='__main__':main()
