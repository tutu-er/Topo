"""Accepted-contract random codes and observation-only block proposals.

No simulator, optimizer, true baseline or true topology is imported. Four
whole-event codes are not treated as independent quarter-hour experiments.
Numerical proposals are never advertised as confidence/coverage certificates.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np

PROBABILITY=np.array([0.,.5,.75])
WAVE_KW=np.r_[np.full(4,-.2),np.full(4,.2)]
DEVICE=dict(baseline_kw=.2,baseline_kvar=.07,power_upper_kw=.4,
            initial_energy_kWh=0.,terminal_energy_kWh=.4,efficiency=1.,
            interval_hours=.25,non_V2G=True,empirical_calibration=False)


def sample_controls(price_rows,pair_rows,*,seed,dependence='local',base_kw=200.):
    """Draw acceptance first; then independently draw a whole-event orientation.

    The two training rows repeat in two validation events, all independent in
    time. Participants execute a preapproved forward OR reverse task. Partial
    acceptance is never forced into a pair, discarded from cost, or resampled.
    Returned increments are issued controls until execution is separately
    verified; they are not a claim that a physical device exists.
    """
    price=np.asarray(price_rows)
    if price.ndim!=2 or price.shape[0]!=2 or price.shape[1]<2 or not np.isin(price,[0,1,2]).all():
        raise ValueError('two aligned rows of prices0/1/2 required')
    n=price.shape[1];pairs=np.asarray(pair_rows,int)
    if pairs.shape!=(2,2) or any(a==b or min(a,b)<0 or max(a,b)>=n for a,b in pairs):
        raise ValueError('two explicit distinct endpoint pairs required')
    if not np.isfinite(base_kw) or base_kw<=0:raise ValueError('positive power base required')
    if isinstance(seed,bool) or int(seed)!=seed or seed<0:raise ValueError('nonnegative seed required')
    if dependence not in ('local','common'):raise ValueError('unknown response dependence')
    for row,pair in zip(price,pairs):
        if any(v>0 and j not in pair for j,v in enumerate(row)):
            raise ValueError('this primitive offers only the declared pair in each event')
    full_price=price[np.arange(4)%2].astype(int)
    worst=.2*float(full_price.sum())
    if worst>1.6+1e-12:raise ValueError('four-event worst payment exceeds1.6')
    z=np.zeros((4,n),bool);s=np.empty(4,int)
    # Distinct named counter-based streams prevent use of the same random draw
    # for customer acceptance and code direction, even under common response.
    for event in range(4):
        rz=np.random.default_rng(np.random.SeedSequence([int(seed),76001,event]))
        u=rz.random(n) if dependence=='local' else np.repeat(rz.random(),n)
        z[event]=u<PROBABILITY[full_price[event]]
        rs=np.random.default_rng(np.random.SeedSequence([int(seed),76002,event]))
        s[event]=1 if rs.integers(0,2) else -1
    delta=np.zeros((4,8,n))
    for event in range(4):
        a,b=pairs[event%2]
        delta[event,:,a]=s[event]*WAVE_KW*z[event,a]
        delta[event,:,b]=-s[event]*WAVE_KW*z[event,b]
    device_power=.2+delta
    energy=np.cumsum(device_power,axis=1)*.25
    if not ((device_power>=-1e-12).all() and (device_power<=.4+1e-12).all()
            and (energy>=-1e-12).all() and (energy<=.4+1e-12).all()
            and np.allclose(energy[:,-1],.4,atol=1e-12)):
        raise AssertionError('fixed task power or energy violated')
    payments=.2*np.sum(full_price*z,axis=1)
    return dict(prices=full_price,accepted=z,orientation=s,pairs=pairs[np.arange(4)%2],
        delta_p_kw=delta,delta_q_kvar=.35*delta,committed_payment=payments,
        actual_payment=None,delivery_payment_if_executed=payments,
        worst_payment=worst,base_kw=float(base_kw),dependence=dependence,seed=int(seed),
        acceptance_before_code=True,independent_named_random_streams=True,
        code_status='issued; execution must be independently verified',
        whole_event_randomization_units=4,device_declaration=dict(DEVICE),
        no_forced_participation=True,partial_response_cost_retained=True)


def controls_as_samples(draw,event_indices=(0,1),*,samples_per_event=96,window_start=76):
    """Flatten selected observed events; no natural load or voltage is read."""
    indices=[int(e) for e in event_indices]
    if len(set(indices))!=len(indices) or any(e not in range(4) for e in indices):
        raise ValueError('distinct original event IDs required')
    if samples_per_event<8 or window_start<0 or window_start+8>samples_per_event:
        raise ValueError('complete eight-quarter contract window required')
    n=np.asarray(draw['prices']).shape[1];dp=[];dq=[];ids=[];code_weights=[]
    for event in indices:
        p=np.zeros((samples_per_event,n));q=np.zeros_like(p)
        p[window_start:window_start+8]=draw['delta_p_kw'][event]
        q[window_start:window_start+8]=draw['delta_q_kvar'][event]
        w=np.zeros(samples_per_event);w[window_start:window_start+8]=int(draw['orientation'][event])*WAVE_KW
        dp.append(p);dq.append(q);ids.extend([event]*samples_per_event);code_weights.append(w)
    return dict(delta_P_known_kw=np.concatenate(dp),delta_Q_known_kvar=np.concatenate(dq),
        event_ids=np.asarray(ids,int),pair_by_event={e:np.asarray(draw['pairs'][e],int).tolist() for e in indices},
        code_weight_known_kw=np.concatenate(code_weights),
        orientation_by_event={e:int(draw['orientation'][e]) for e in indices},
        accepted_by_event={e:np.asarray(draw['accepted'][e],bool).tolist() for e in indices},
        price_by_event={e:np.asarray(draw['prices'][e],int).tolist() for e in indices},
        base_kw=float(draw['base_kw']),control_status=draw['code_status'],
        acceptance_before_code=bool(draw['acceptance_before_code']))


def _arrays(split,n=None):
    if isinstance(split,(tuple,list)):
        values=[_arrays(v,n) for v in split]
        if not values:raise ValueError('empty training scene list')
        return tuple(np.concatenate([v[j] for v in values]) for j in range(3))
    if not isinstance(split,dict):raise ValueError('P/Q/Y scene dictionary required')
    forbidden={'true_R','true_X','true_K','true_family','true_baseline','p0','q0'}
    if forbidden.intersection(split):raise ValueError('truth/baseline is not an identifier input')
    out=tuple(np.asarray(split[k],float) for k in ('P','Q','Y'))
    if any(v.ndim!=2 for v in out) or any(v.shape!=out[0].shape for v in out):
        raise ValueError('aligned time-by-terminal P/Q/Y required')
    if n is not None and out[0].shape[1]!=n:raise ValueError('terminal count mismatch')
    if not all(np.isfinite(v).all() for v in out):raise ValueError('finite observations required')
    return out


def _get(mapping,event):
    if event in mapping:return mapping[event]
    if str(event) in mapping:return mapping[str(event)]
    raise ValueError('missing explicit control event '+str(event))


def observation_moments(observed,controls):
    """Actual P/Q/Y left-multiplied by externally assigned code weights.

    All partial/no-response outcomes are retained. A price/participation law is
    not needed to compute the moments. This does not make a four-event sample
    unbiased conditional on its realized baseline, nor create new observations.
    Population exogeneity requires the stated pre-acceptance/random-code order
    and an appropriate physical/measurement residual orthogonality condition.
    """
    forbidden={'true_R','true_X','true_K','true_family','true_baseline','p0','q0'}
    if forbidden.intersection(observed) or forbidden.intersection(controls):
        raise ValueError('truth/baseline is not a moment input')
    p,q,y=_arrays(observed['future_train']);record=controls['future_train']
    if not record.get('acceptance_before_code',False):raise ValueError('pre-code acceptance must be recorded')
    ids=np.asarray(record['event_ids'],int)
    if ids.shape!=(len(p),):raise ValueError('event IDs must align with observations')
    base=float(record.get('base_kw',200.))
    if not np.isfinite(base) or base<=0:raise ValueError('positive power base required')
    if 'code_weight_known_kw' not in record:raise ValueError('external code weights must be recorded even when nobody participates')
    w=np.asarray(record['code_weight_known_kw'],float)/base
    dp=np.asarray(record['delta_P_known_kw'],float)/base;dq=np.asarray(record['delta_Q_known_kvar'],float)/base
    if w.shape!=(len(p),) or dp.shape!=p.shape or dq.shape!=p.shape:raise ValueError('code and observed rows differ')
    if not all(np.isfinite(a).all() for a in (w,dp,dq)):raise ValueError('finite code/inputs required')
    events=list(dict.fromkeys(ids.tolist()));pairs=[];rows=[]
    for event in events:
        pair=tuple(map(int,_get(record['pair_by_event'],event)))
        if pair not in pairs:pairs.append(pair)
    for pair in pairs:
        included=[e for e in events if tuple(map(int,_get(record['pair_by_event'],e)))==pair]
        weight=w*np.isin(ids,included)
        scale=float(weight@weight)
        if scale<=0:continue
        weight=weight/scale
        rows.append(dict(pair=list(pair),events=included,P=(weight@p).tolist(),Q=(weight@q).tolist(),Y=(weight@y).tolist(),
            known_control_P_moment=(weight@dp).tolist(),known_control_Q_moment=(weight@dq).tolist(),
            known_control_first_stage_zero=bool(np.max(abs(weight@dp),initial=0.)==0 and np.max(abs(weight@dq),initial=0.)==0),
            includes_partial_and_nonparticipation=True,normalization='sum of assigned code squared, not accepted count'))
    n=p.shape[1]
    return dict(P=np.asarray([r['P'] for r in rows],float).reshape(-1,n),
        Q=np.asarray([r['Q'] for r in rows],float).reshape(-1,n),
        Y=np.asarray([r['Y'] for r in rows],float).reshape(-1,n),rows=rows,
        whole_event_randomization_units=len(events),quarters_iid_assumed=False,
        no_new_observations=True,certificate=False,statistical_coverage=False,
        uses_validation=False,uses_true_baseline=False,
        interpretation='population code moment only under exogeneity; finite random baseline projection remains')


def _families_from_band(center,radius,a,b,max_candidates):
    """All potentially signed clades in a *conditional numerical* interval.

    No lower effect bound is introduced. Entries whose interval straddles zero
    remain unresolved. Overflow returns no truncated candidate certificate.
    """
    lo=center-radius;hi=center+radius;n=len(center)
    proposed=set();sides=[]
    for sign,anchor,other in ((1,a,b),(-1,b,a)):
        lower=lo if sign==1 else -hi;upper=hi if sign==1 else -lo
        forced=set(np.flatnonzero(lower>0).tolist())
        possible=set(np.flatnonzero(upper>0).tolist())
        if forced and anchor not in possible:
            return [],None,dict(status='inconsistent_signed_block_model',forced=sorted(forced),anchor=anchor)
        if other in forced:
            return [],None,dict(status='inconsistent_signed_block_model',wrong_anchor_forced=other)
        possible.discard(other)
        if anchor not in possible:
            sides.append(dict(forced=sorted(forced),possible=sorted(possible),empty_allowed=True));continue
        mandatory=forced|{anchor};optional=sorted(possible-mandatory)
        # Empty signed response is retained if it is not ruled out by the band.
        sides.append(dict(forced=sorted(forced),possible=sorted(possible),empty_allowed=not forced))
        if (1<<len(optional))+len(proposed)>max_candidates:
            return [],None,dict(status='candidate_range_exceeds_declared_cap',sides=sides,max_candidates=max_candidates)
        for size in range(len(optional)+1):
            for extra in combinations(optional,size):
                s=tuple(sorted(mandatory|set(extra)))
                if len(s)>=2:proposed.add(s)
    # A direct support readout is available only when no sign/membership is
    # uncertain. Even then singleton coefficients must be jointly fitted.
    exact_membership=bool(np.all((lo>0)|(hi<0)|((lo==0)&(hi==0))))
    direct=None
    if exact_membership:
        clades=[tuple(np.flatnonzero(center>0)),tuple(np.flatnonzero(center<0))]
        direct=sorted({*(tuple([i]) for i in range(n)),*(s for s in clades if len(s)>=2)},key=lambda s:(len(s),s))
    status='conditional_numeric_readout' if exact_membership else 'unresolved_signed_membership'
    return sorted(proposed,key=lambda s:(len(s),s)),direct,dict(status=status,sides=sides)


def discover_candidates(observed,controls,config):
    """Observation/control-only candidate proposal; validation is never read.

    observed: history_train/future_train dictionaries of P,Q,Y arrays, P/Q pu.
    controls.future_train: flattened known increments and pre-code acceptance.
    config: trusted_exact P/Q, verified_exact control and an explicitly fixed
    Y/model absolute error band are required for a conditional band. Without
    those declarations the default is unresolved, not a guessed certificate.

    Residualizing the coded signal against *observed* remaining P/Q is a partial
    regression calculation. It adds no information to noiseless full raw data,
    and unknown P/Q measurement error makes the elimination non-exact. No
    impedance upper bound is silently used to cover this defect.
    """
    forbidden={'true_R','true_X','true_K','true_family','true_baseline','p0','q0'}
    if forbidden.intersection(observed) or forbidden.intersection(controls) or forbidden.intersection(config):
        raise ValueError('true parameters/topology/baseline must not reach discovery')
    fp,fq,fy=_arrays(observed['future_train']);n=fp.shape[1]
    if 'history_train' in observed:
        hp,hq,hy=_arrays(observed['history_train'],n)
    else:hp=hq=hy=np.empty((0,n))
    p=np.vstack((hp,fp));q=np.vstack((hq,fq));y=np.vstack((hy,fy))
    control=controls['future_train'];base=float(control.get('base_kw',config.get('base_kw',200.)))
    if base<=0 or not np.isfinite(base):raise ValueError('positive power base required')
    dp=np.asarray(control['delta_P_known_kw'],float)/base;dq=np.asarray(control['delta_Q_known_kvar'],float)/base
    ids=np.asarray(control['event_ids'],int)
    if dp.shape!=fp.shape or dq.shape!=fp.shape or ids.shape!=(len(fp),):
        raise ValueError('control rows must match future training exactly')
    if not all(np.isfinite(v).all() for v in (dp,dq)):raise ValueError('finite executed increments required')
    if not control.get('acceptance_before_code',False):raise ValueError('control record must establish acceptance before random direction')
    unclosed=[]
    if config.get('pq_observation_mode')!='trusted_exact':unclosed.append('P/Q error is not bounded after unknown R/X multiplication')
    if config.get('executed_control_mode')!='verified_exact':unclosed.append('known increments are issued rather than verified execution')
    if config.get('y_error_bound') is None or not config.get('band_fixed_before_codes',False):
        unclosed.append('no predeclared Y plus model-error band')
    if config.get('power_unit','pu')!='pu':raise ValueError('current interface requires per-unit P/Q')
    epsilon=None
    if config.get('y_error_bound') is not None:
        bound=np.asarray(config['y_error_bound'],float)
        if bound.shape==fp.shape:
            if len(hp):
                hb=config.get('historical_y_error_bound')
                if hb is None:unclosed.append('historical Y error band missing')
                else:bound=np.vstack((np.broadcast_to(np.asarray(hb,float),hp.shape),bound))
        try:epsilon=np.broadcast_to(bound,y.shape).copy()
        except ValueError:unclosed.append('error band does not cover all training rows')
        if epsilon is not None and (not np.isfinite(epsilon).all() or (epsilon<0).any()):raise ValueError('finite nonnegative error bound required')
    event_values=list(dict.fromkeys(ids.tolist()));pairs=[]
    for event in event_values:
        pair=tuple(map(int,_get(control['pair_by_event'],event)))
        if len(pair)!=2 or pair[0]==pair[1] or min(pair)<0 or max(pair)>=n:raise ValueError('bad event endpoint pair')
        if pair not in pairs:pairs.append(pair)
    diagnostics=[];candidate=set();direct_families=[]
    tol=float(config.get('numeric_tolerance',1e-10));rcond=float(config.get('rcond',1e-12))
    cap=int(config.get('max_candidates',64));allow=bool(config.get('allow_unclosed_heuristic',False))
    for a,b in pairs:
        template=np.zeros_like(p);signal=np.zeros(len(p));selected=[]
        for event in event_values:
            if tuple(map(int,_get(control['pair_by_event'],event)))!=(a,b):continue
            z=np.asarray(_get(control['accepted_by_event'],event),bool)
            if z.shape!=(n,):raise ValueError('accepted-node map mismatch')
            if not(z[a] and z[b]):continue
            rows=np.flatnonzero(ids==event);selected.append(event)
            # Only the balanced known component is selected. Other contracts,
            # partial responses and all natural inputs stay in the nuisance.
            if np.max(abs(dp[rows,a]+dp[rows,b]),initial=0)>tol or np.max(abs(dq[rows]-.35*dp[rows]),initial=0)>tol:
                raise ValueError('accepted pair control is not declared balanced P/Q')
            signal[len(hp)+rows]=dp[rows,a]
            template[len(hp)+rows,a]=dp[rows,a];template[len(hp)+rows,b]=dp[rows,b]
        note=dict(pair=[a,b],paired_accepted_events=selected,total_randomization_events=len(event_values),
            control_units='whole events, not independent quarters',observation_model_unclosed=list(unclosed))
        norm2=float(signal@signal)
        if norm2<=0:
            note.update(status='unresolved_no_paired_acceptance',candidate_supports=[]);diagnostics.append(note);continue
        raw_weights=signal/norm2
        note['raw_code_contrast_Y']=(raw_weights@y).tolist()
        # Nothing here reads a true natural baseline. P - known balanced delta
        # is an observed remaining-input matrix, noisy when P/Q are noisy.
        npower=p-template;nreactive=q-.35*template
        event_columns=[np.r_[np.ones(len(hp)),np.zeros(len(fp))]] if len(hp) else []
        for event in event_values:event_columns.append(np.r_[np.zeros(len(hp)),(ids==event).astype(float)])
        nuisance=np.column_stack((npower,nreactive,*event_columns))
        coefficient=np.linalg.lstsq(nuisance,signal,rcond=rcond)[0]
        residual=signal-nuisance@coefficient;denom=float(residual@signal)
        note.update(nuisance_rank=int(np.linalg.matrix_rank(nuisance)),signal_norm2=norm2,
            remaining_signal_norm2=float(residual@residual),residualized_signal_denominator=denom)
        if denom<=max(np.finfo(float).eps*norm2,(rcond**2)*norm2):
            note.update(status='unresolved_control_absorbed_by_observed_nuisance',candidate_supports=[]);diagnostics.append(note);continue
        weights=residual/denom;center=weights@y
        projected_p=weights@p;projected_q=weights@q
        target=np.zeros(n);target[a]=1.;target[b]=-1.
        closure=max(float(np.max(abs(projected_p-target))),float(np.max(abs(projected_q-.35*target))))
        note.update(residualized_contrast_Y=center.tolist(),observed_P_contrast=projected_p.tolist(),
            observed_Q_contrast=projected_q.tolist(),numeric_input_closure_error=closure,
            weights_l1=float(np.abs(weights).sum()),weights_l2=float(np.linalg.norm(weights)),
            estimator='observed-input partial regression, no new information or confidence guarantee')
        local_unclosed=list(unclosed)
        if closure>tol:local_unclosed.append('numerical remaining-input elimination did not close')
        if local_unclosed and not allow:
            note.update(status='unresolved_observation_model',observation_model_unclosed=local_unclosed,candidate_supports=[])
            diagnostics.append(note);continue
        if local_unclosed:
            proposed=[tuple(np.flatnonzero(center>0)),tuple(np.flatnonzero(center<0))]
            proposed=[s for s in proposed if len(s)>=2]
            direct=None;meta=dict(status='heuristic_only_unclosed_model')
            radius=None
        else:
            # Numerical uncertainty is an observation/readout tolerance, not a
            # positive line-weight restriction. Weak effects stay unresolved.
            radius=np.abs(weights)@epsilon+float(config.get('numeric_readout_band',tol))
            proposed,direct,meta=_families_from_band(center,radius,a,b,cap)
        candidate.update(proposed)
        if direct is not None:direct_families.append(direct)
        note.update(status=meta['status'],signed_band=meta,band_radius=None if radius is None else radius.tolist(),
            candidate_supports=[list(s) for s in proposed],observation_model_unclosed=local_unclosed)
        diagnostics.append(note)
    direct=None
    if len(direct_families)==1 and len(pairs)==1:direct=[list(s) for s in direct_families[0]]
    return dict(status='conditional_or_heuristic_proposals' if candidate else 'unresolved',
        candidate_supports=[list(s) for s in sorted(candidate,key=lambda s:(len(s),s))],direct_family=direct,
        direct_family_requires_joint_nonnegative_fit=True,diagnostics=diagnostics,
        certificate=False,statistical_coverage=False,global_parameter_exclusion=False,
        uses_validation=False,uses_true_baseline=False,uses_true_topology=False,
        zero_and_arbitrarily_weak_weights_not_excluded=True,
        only_R_plus_035X_directly_targeted=True,
        note='Exact linear trusted-input conditional band only; otherwise unresolved or explicitly unverified proposals. WZZT is not the sole estimator that can use these contrasts.')
