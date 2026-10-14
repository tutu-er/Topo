"""Independent replay of saved all-pair EV development records.

No selector, data generator, optimizer, calibration or new AC fit is called.
Saved AC states use a separate branch-current sweep. Proxy QP bounds are
checked by residual arithmetic and the box tangent inequality. Scientific
source bytes and paired-history identities are verified before aggregation.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np

from .information_fit_recovery import read_json, write_json, independent_replay
from .information_replay import _obs, _meter
from .ev_flexibility_replay import fit_audits, close, require
from .likelihood import RadialCandidate, concat_observations

HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q')
DEFAULT_METHODS = ('none','constant','quality_lp','random_price_lp','fixed_mle_milp','cross_raw_milp','cross_guarded_milp')


def arr(v): return np.asarray(v, float)


def json_identity(v):
    import json
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_audit(directory, freeze):
    snapshot = read_json(directory/'source_snapshot.json.gz')
    require(snapshot['schema']=='source_bytes_base64', 'source snapshot schema')
    hashes = freeze['source_sha256']; sources = snapshot['sources']
    require(set(sources)==set(hashes), 'source snapshot coverage')
    rows=[]
    for name, expected in hashes.items():
        path=(HERE/name).resolve(); path.relative_to(HERE.parent.resolve())
        payload=base64.b64decode(sources[name]['bytes_base64'], validate=True)
        require(sources[name]['sha256']==expected and hashlib.sha256(payload).hexdigest()==expected,
                'snapshot source hash:'+name)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,
                'current frozen scientific source differs:'+name)
        rows.append(dict(name=name,sha256=expected))
    return dict(success=True, source_count=len(rows), rows=rows)


def operation_audit(u, config, *, diagnostic=False):
    u=arr(u); require(u.shape==(config['horizon'],) and np.isfinite(u).all(), 'power curve dimensions')
    require(config['base_kw']==20. and config['site']==0, 'declared synthetic units/site changed')
    require(np.all(u>=-1e-7) and np.all(u<=config['power_kw']+1e-7), 'charge power bounds')
    dt=.25; energy=float(dt*u.sum())
    if diagnostic:
        close(u,np.zeros_like(u),'noEV injection',atol=1e-12)
    else:
        close(energy,config['grid_energy_kwh'],'equal grid charging service',atol=2e-6)
        terminal=24.+config['eta']*energy
        close(terminal,config['terminal_battery_kwh'],'terminal battery energy',atol=2e-6)
        soc=np.r_[.4,(24.+config['eta']*dt*np.cumsum(u))/60.]
        require(np.all(soc>=.1-1e-7) and np.all(soc<=.9+1e-7),'SOC range')
    return dict(feasible=True, grid_energy_kwh=energy, power_kw=u.tolist(), diagnostic=diagnostic)


def moving_geometry_audit(model):
    c,B,A0,At=(arr(model[k]) for k in ('c','B','A0','At'))
    u0=arr(model['reference_kw']); D=model['domain']; n=A0.shape[1]
    require(c.ndim==1 and B.shape==(len(c),len(u0)) and A0.shape==(len(c),n)
            and At.shape==(len(u0),len(c),n),'bilinear matrix dimensions')
    close(A0+np.einsum('t,tij->ij',u0,At),model['C_at_reference'],'reference Jacobian closure',atol=2e-8)
    close(c+B@u0,model['reference_residual'],'reference residual closure',atol=2e-8)
    base=arr(D['base_center']);shift=arr(D['shift']);plo,phi=arr(D['physical_lower']),arr(D['physical_upper'])
    ul,uh=arr(D['action_lower']),arr(D['action_upper'])
    slow=np.maximum(shift,0)@ul+np.minimum(shift,0)@uh
    shigh=np.maximum(shift,0)@uh+np.minimum(shift,0)@ul
    close(D['outer_lower'],plo-base-shigh,'common outer lower construction')
    close(D['outer_upper'],phi-base-slow,'common outer upper construction')
    require(model['original_AC_global_certificate'] is False,'proxy labelled as global AC certificate')
    return dict(name=model['name'],rows=len(c),columns=n,scope='declared mixed-term proxy only')


def replay_qp(model,u,saved,*,domain,kind='fixed'):
    """Recompute saved feasible cost and tangent lower without solving a QP."""
    D=model['domain'];u,z=arr(u),arr(saved['z'])
    base=arr(D['base_center']);shift=arr(D['shift'])@u
    if domain=='original': lo,hi=arr(D['physical_lower'])-base-shift,arr(D['physical_upper'])-base-shift
    elif domain=='outer': lo,hi=arr(D['outer_lower']),arr(D['outer_upper'])
    else: raise ValueError('unknown nuisance domain')
    require(z.shape==lo.shape==hi.shape and np.isfinite(z).all(),'saved nuisance dimensions')
    require(np.all(z>=lo-1e-8) and np.all(z<=hi+1e-8),'saved nuisance witness outside domain')
    A=arr(model['A0'])+np.einsum('t,tij->ij',u,arr(model['At']))
    r=arr(model['c'])+arr(model['B'])@u+A@z
    if 'control_curvature_difference' in model:
        r+=.5*arr(model['control_curvature_difference'])@((u-arr(model['reference_kw']))**2)
    close(saved['residual'],r,'saved QP residual',atol=2e-7,rtol=2e-8)
    primal=float(r@r);gradient=2*A.T@r
    correction=float(np.minimum(gradient*(lo-z),gradient*(hi-z)).sum())
    raw=primal+correction
    if kind=='exchange':
        guard=256*np.finfo(float).eps*(1+abs(primal)+abs(correction))
        uk,lk='q_primal_upper','q_tangent_lower'
    else:
        corner=np.where(gradient>=0,lo,hi)
        guard=64*np.finfo(float).eps*max(1.,primal,float(abs(gradient)@(abs(corner)+abs(z))))
        uk,lk='primal_upper','tangent_lower_numeric'
    lower=max(0.,min(primal,raw-guard))
    close(saved[uk],primal,'saved QP primal cost',atol=2e-7,rtol=2e-8)
    close(saved[lk],lower,'saved tangent lower',atol=2e-7,rtol=2e-8)
    require(lower<=primal+2e-7,'invalid QP bounds')
    if 'physical_parameter' in saved:
        physical=base+shift+z
        close(saved['physical_parameter'],physical,'physical moving-center map')
        violation=float(max(0.,np.max(arr(D['physical_lower'])-physical),np.max(physical-arr(D['physical_upper']))))
        close(saved['physical_bound_violation'],violation,'physical domain violation',atol=2e-8)
        if domain=='original': require(violation<=2e-8,'original QP not physically feasible')
    return dict(lower=lower,upper=primal,gap=primal-lower,solver_success=bool(saved['success']),domain=domain)


def exchange_audit(models, saved, history_losses):
    require(saved['required_pairs']==[m['name'] for m in models],'exchange keeps all ordered pairs')
    require(saved['full_curve_enumeration'] is False and saved['original_global_optimum_certified'] is False,
            'exchange scope flags')
    offsets=arr([history_losses[m['b']] for m in models]);scale=float(saved['score_scale'])
    require(scale==.5,'score SSE/loglik scale')
    for m,o in zip(models,offsets): close(saved['offsets'][m['name']],o,'competitor H cost offset')
    entries=[]
    for entry in saved['evaluations']:
        require(len(entry['qp_results'])==len(models),'exchange evaluation drops a competitor')
        audits=[replay_qp(m,entry['u'],q,domain='outer',kind='exchange') for m,q in zip(models,entry['qp_results'])]
        lo=scale*(arr([q['lower'] for q in audits])-offsets)
        hi=scale*(arr([q['upper'] for q in audits])-offsets)
        close(entry['pair_score_lower'],lo,'exchange pair lower')
        close(entry['pair_score_upper'],hi,'exchange pair upper')
        close(entry['score_lower'],min(lo),'exchange min lower')
        close(entry['score_upper'],min(hi),'exchange min upper')
        entries.append(dict(origin=entry['origin'],lower=float(min(lo)),upper=float(min(hi)),
                            qp_success=sum(q['solver_success'] for q in audits)))
    lower=saved['objective_lower'];upper=float(saved['objective_upper'])
    valid_interval=True
    if entries:
        close(lower,max(q['lower'] for q in entries),'best evaluated lower')
        best=int(saved['best_evaluation_index']);close(saved['u'],saved['evaluations'][best]['u'],'returned best curve')
        valid_interval=bool(upper>=lower-1e-9*max(1.,abs(upper),abs(lower)))
        if not valid_interval:
            require(saved['status'] in ('numerical_bound_inconsistency','initial_numerical_bound_inconsistency') and not saved['completed'],
                    'contradictory bounds not retained as numerical failure')
            require(any(f['kind'] in ('numerical_bound_inconsistency','initial_numerical_bound_inconsistency') for f in saved['failures']),
                    'numerical interval failure lacks saved cause')
        close(saved['gap'],max(0.,upper-lower),'proxy recorded gap',atol=2e-7)
        close(saved['raw_gap'],upper-lower,'proxy signed raw gap',atol=2e-7)
        if saved['completed']:
            require(valid_interval and upper-lower<=saved['target_gap']+1e-9*max(1.,abs(upper),abs(lower)),
                    'completed proxy interval exceeds planning gap tolerance')
    previous=float(saved['initial_fixed_witness_upper'])
    global_initial=[];midpoint=[]
    v2=saved.get('initialization')=='all required QP explanations at a feasible control; no midpoint witnesses'
    for index,model in enumerate(models):
        D=model['domain'];lo,hi=arr(D['action_lower']),arr(D['action_upper'])
        middle=.5*(arr(D['outer_lower'])+arr(D['outer_upper']))
        z=arr(saved['evaluations'][0]['qp_results'][index]['z']) if v2 else middle
        def witness_box_score(witness):
            c=arr(model['c'])+arr(model['A0'])@witness
            B=arr(model['B'])+np.einsum('tmn,n->mt',arr(model['At']),witness)
            a=c+np.minimum(B*lo,B*hi).sum(axis=1);b=c+np.maximum(B*lo,B*hi).sum(axis=1)
            distance=float(np.sqrt(np.maximum(a*a,b*b).sum()))
            distance+=128*np.finfo(float).eps*(1+distance)
            return scale*(distance**2-offsets[index])
        global_initial.append(witness_box_score(z));midpoint.append(witness_box_score(middle))
    close(saved['initial_fixed_witness_upper'],min(global_initial),'initial feasible fixed-witness global proxy bound',atol=2e-7,rtol=2e-8)
    if v2: close(saved['midpoint_fixed_witness_upper'],min(midpoint),'old midpoint crude global bound',atol=2e-7,rtol=2e-8)
    for item in saved['iterations']:
        upper_i=float(item['global_upper'])
        require(upper_i<=previous+2e-7*max(1.,abs(previous)),'global proxy upper increases')
        require(item['master'].get('original_global_optimum_certified',False) is False,'master AC certificate label')
        previous=upper_i
    return dict(success=True,evaluations=len(entries),qp_states=len(entries)*len(models),entries=entries,
        objective_lower=lower,objective_upper=upper,gap=saved['gap'],status=saved['status'],completed=saved['completed'],
        valid_global_proxy_interval=valid_interval,raw_gap=saved['raw_gap'],
        initial_witness_upper_replayed=min(global_initial),midpoint_witness_upper_replayed=min(midpoint),
        computational_failure=None if valid_interval else 'saved numerical upper below attained lower; invalid interval preserved',
        scope='saved floating solver bounds for common-outer bilinear proxy; not nonlinear AC distance')


def fixed_score_audit(models,saved,history_losses):
    if saved['u'] is None: return dict(incumbent_available=False,status=saved['status'])
    u=arr(saved['u']);values=[];error=[]
    for model in models:
        d,T=arr(model['d'])/np.sqrt(2.),arr(model['T'])/np.sqrt(2.)
        x=d+T@u;values.append(float(x@x)-.5*history_losses[model['b']])
        lo=d+np.minimum(6*T,0).sum(axis=1);hi=d+np.maximum(6*T,0).sum(axis=1)
        error.append(float(np.sum(((hi-lo)/saved['segments'])**2/4)))
    close(saved['quadratic_values'],values,'fixed-parameter quadratic values',atol=2e-7,rtol=2e-8)
    close(saved['objective_lower'],min(values),'fixed-parameter attained score',atol=2e-7,rtol=2e-8)
    close(saved['secant_error_by_quadratic'],error,'fixed-parameter segment error bounds',atol=2e-7,rtol=2e-8)
    require(saved['objective_upper']>=min(values)-2e-7*max(1.,abs(min(values))),'fixed proxy upper below incumbent')
    require(saved['original_global_optimum_certified'] is False,'fixed proxy AC scope flag')
    return dict(incumbent_available=True,lower=min(values),upper=saved['objective_upper'],gap=saved['original_gap'],
        scope='frozen fixed-parameter quadratic only; saved floating MILP bound')


def quality_audit(trial):
    models={k:_meter(v) for k,v in trial['measurement_models'].items()};hm,fm,jm=(models[k] for k in ('history','future','joint'))
    h,n=12,4;qr=trial['quality'];counts=arr(qr['v_repeat_counts'])
    require(counts.shape==(4,3) and np.all(counts>0),'public counts dimensions')
    for k in CHANNELS:
        hs=getattr(hm,k+'_sigma');fs=getattr(fm,k+'_sigma');js=getattr(jm,k+'_sigma')
        if hs is None:
            require(fs is None and js is None,'absent channel uncertainty');continue
        tail=(3,) if k in ('p','q','v') else ()
        old=np.broadcast_to(arr(hs),(h,*tail))
        repetitions=counts if k=='v' else np.full((n,*tail),5.)
        future=np.broadcast_to(old[0],repetitions.shape)/np.sqrt(repetitions)
        close(fs,future,'future random-error averaging:'+k)
        close(js,np.concatenate((old,future)), 'joint channel sigma:'+k)
    for k in ('p','q','v'):
        close(getattr(hm,'bias_'+k),getattr(fm,'bias_'+k),'fixed calibrated future bias:'+k)
        close(getattr(hm,'bias_'+k),getattr(jm,'bias_'+k),'fixed calibrated joint bias:'+k)
    require(qr['public_plan_uses_truth'] is False and qr['public_plan_uses_future_noise'] is False,'quality isolation flags')
    return models


def physical_audit(trial,row,fm):
    t=trial['truth'];u=arr(row['u_kw']);site=int(row['site']);h=12
    p=arr(t['p'])[h:].copy();q=arr(t['q'])[h:].copy();p[:,site]+=u/trial['config']['base_kw']
    phy=row['physical'];close(phy['p'],p,'EV added to physical total P exactly once');close(phy['q'],q,'EV Q rule')
    means={}
    for k in ('p','q','v','master_p','master_q'):
        raw=arr(phy[k]);bias=arr(getattr(fm,'bias_'+k)) if k in ('p','q','v') else 0.
        means[k]=raw+bias
        predicted=means[k]+np.broadcast_to(arr(getattr(fm,k+'_sigma')),raw.shape)*arr(trial['white_future'][k])
        close(row['observations'][k],predicted,'saved noisy future meter:'+k,atol=2e-10)
    candidate=RadialCandidate.from_supports('saved_generation',3,t['supports'])
    witness=dict(r=t['r_pu'],x=t['x_pu'],p=p,q=q,source_p=arr(t['source_p'])[h:],source_q=arr(t['source_q'])[h:],
        predicted=means,**{key:getattr(fm,key) for key in ('bias_p','bias_q','bias_v')})
    observations=_obs(dict(**means,root_v=np.ones(4),source_p=None,source_q=None))
    replay=independent_replay(candidate,witness,observations,fm)
    require(replay['success'] and replay['full_SSE']<1e-12,'independent future physical AC generation')
    return replay


def guard_action_audit(trial,action,history,hm,fm,jm,candidates):
    """Replay all reference-specific H-only means and attained joint costs."""
    require(action['uses_truth'] is False and action['uses_future_noise'] is False
            and action['future_added_noise'] is False,'guard forecast isolation flags')
    require(action['global_design_optimality_certified'] is False,'guard AC score scope')
    u=arr(action['u']);site=int(action['site']);base=float(action['base_kw'])
    hfits={row['name']:row for row in trial['historical_fits']}
    p0=arr(history.p)-arr(hm.bias_p);q0=arr(history.q)-arr(hm.bias_q)
    background=dict(p=np.tile(p0.mean(axis=0),(4,1)),q=np.tile(q0.mean(axis=0),(4,1)),
                    root_v=np.full(4,np.mean(history.root_v)))
    if 'common_background_forecast' in action:
        for k,v in background.items():close(action['common_background_forecast'][k],v,'guard H-only common forecast:'+k,atol=2e-12)
    record_map={r['source']:r for r in action['records']}
    require(len(record_map)==len(action['records']),'guard duplicate source record')
    if not action.get('failures'):
        require(set(record_map)==set(candidates),'guard missing reference forecast')
    losses={};saved_states=0;forecasts=0;fit_calls=0;fit_ac_calls=0;success_states=0;returned_states=0
    for source,record in record_map.items():
        require(source in candidates,'guard reference outside candidate bank')
        close(record['u'],u,'guard record curve')
        require(record['site']==site,'guard record site')
        cost=record['cost'];fit_calls+=int(cost['fit_calls']);fit_ac_calls+=int(cost['ac_calls'])
        forecasts+=int(record['forecast_calls']);returned_states+=len(record['fits'])
        success_states+=sum(bool(f['success']) and f.get('loss') is not None for f in record['fits'])
        observations=record['predicted_observations']
        if observations is None:
            require(not record['success'] and bool(record['fit_failures']),'missing forecast not retained as failure')
            continue
        obs=_obs(observations);require(obs.t==4,'guard future mean time count')
        f=hfits[source];expected_p=background['p'].copy();expected_p[:,site]+=u/base
        state=dict(p=expected_p,q=background['q'],source_p=np.full(4,arr(f['source_p']).mean()),
                   source_q=np.full(4,arr(f['source_q']).mean()),root_v=background['root_v'])
        for channel,value in state.items():close(record['forecast_state'][channel],value,'guard H-only state:'+channel,atol=2e-12)
        close(obs.p,state['p']+arr(fm.bias_p),'guard EV P added once and meter bias',atol=2e-12)
        close(obs.q,state['q']+arr(fm.bias_q),'guard Q forecast and meter bias',atol=2e-12)
        candidate=RadialCandidate.from_supports(source,3,candidates[source]['supports'])
        witness=dict(r=f['r'],x=f['x'],**{k:state[k] for k in ('p','q','source_p','source_q')},
            predicted={k:getattr(obs,k) for k in CHANNELS if getattr(obs,k) is not None},
            **{k:getattr(fm,k) for k in ('bias_p','bias_q','bias_v')})
        replay=independent_replay(candidate,witness,obs,fm)
        require(replay['success'] and replay['full_SSE']<1e-12,'guard independent noiseless AC means')
        if 'predicted_voltage_range' in record:
            close(record['predicted_voltage_range'],[float(arr(obs.v-fm.bias_v).min()),float(arr(obs.v-fm.bias_v).max())],
                  'guard predicted terminal voltage range',atol=2e-10)
        fail={k:v for k,v in record['fit_failures'].items() if k in candidates}
        audit=fit_audits(record['fits'],candidates,concat_observations(history,obs),jm,fail,complete=record['success'])
        saved_states+=audit['states_replayed']
        losses[source]={r['name']:float(r['loss']) for r in record['fits']
                        if r['success'] and r.get('loss') is not None}
    historical={k:float(v['loss']) for k,v in hfits.items()};required={(a,b) for a in candidates for b in candidates if a!=b}
    pairs=action['pairs'];require(len(pairs)==12 and {(r['a'],r['b']) for r in pairs}==required,'guard retains all twelve pairs')
    added=[];joint=[];self_relative=[]
    for row in pairs:
        a,b=row['a'],row['b'];own=losses.get(a,{}).get(a);comp=losses.get(a,{}).get(b)
        close(row['historical_a'],historical[a],'guard H reference cost')
        close(row['historical_b'],historical[b],'guard H competitor cost')
        if comp is not None:
            av=.5*(comp-historical[b]);jv=.5*(comp-historical[a])
            close(row['joint_loss'],comp,'guard joint competing cost')
            close(row['added'],av,'guard added pair score');close(row['joint'],jv,'guard joint pair score')
            added.append(av);joint.append(jv)
        if own is not None and comp is not None:
            sv=.5*(comp-own);close(row['self_joint_loss'],own,'guard self joint cost')
            close(row['self_relative'],sv,'guard self relative pair score');self_relative.append(sv)
        require(row['success']==(own is not None and comp is not None),'guard pair success classification')
    if action['success']:
        require(len(added)==len(joint)==len(self_relative)==12 and not action['failures'],'guard successful score has failed required pair')
        for key,values in [('added_score',added),('joint_score',joint),('self_relative_score',self_relative)]:
            close(action[key],min(values),'guard minimum '+key)
        close(action['score'],min(added),'guard primary score')
    else:
        require(action['score'] is None,'failed guard action retains comparable minimum score')
    for key,value in [('returned_fit_states',returned_states),('successful_fit_states',success_states),
                      ('fit_calls',fit_calls),('fit_ac_calls',fit_ac_calls),('forecast_calls',forecasts),('ac_calls',fit_ac_calls+forecasts)]:
        close(action[key],value,'guard cost accounting:'+key,atol=0.,rtol=0.)
    return dict(success=action['success'],score=action['score'],added_score=action['added_score'],
        joint_score=action['joint_score'],self_relative_score=action['self_relative_score'],
        saved_fit_states=saved_states,forecast_states=len(losses),pairs=len(pairs),
        fit_calls=fit_calls,fit_ac_calls=fit_ac_calls,forecast_calls=forecasts,ac_calls=fit_ac_calls+forecasts)


def guard_audit(trial,guard,history,hm,fm,jm,candidates):
    baseline=guard_action_audit(trial,guard['baseline'],history,hm,fm,jm,candidates)
    candidate=guard_action_audit(trial,guard['candidate'],history,hm,fm,jm,candidates)
    bu,cu=arr(guard['baseline']['u']),arr(guard['candidate']['u']);same=np.array_equal(bu,cu)
    require(guard['cache_hit']==same,'guard exact action cache classification')
    if same:require(json_identity(guard['baseline'])==json_identity(guard['candidate']),'cached guard actions differ')
    complete=baseline['success'] and candidate['success']
    gain=candidate['score']-baseline['score'] if complete else None
    expected=bool(complete and gain>guard['gain_tolerance'])
    require(guard['accepted']==expected,'guard acceptance rule')
    if complete:close(guard['score_gain'],gain,'guard attained predicted score gain')
    close(guard['selected_u'],cu if expected else bu,'guard selected curve')
    actual={r['method']:r['u_kw'] for r in trial['rows']}
    close(actual['constant'],bu,'guard constant baseline input')
    close(actual['cross_raw_milp'],cu,'guard raw MILP proposal input')
    close(actual['cross_guarded_milp'],guard['selected_u'],'guard executed selected action')
    unique=[guard['baseline']] if same else [guard['baseline'],guard['candidate']]
    for key,value in guard['totals'].items():
        close(value,sum(r[key] for r in unique),'guard unique cost total:'+key,atol=0.,rtol=0.)
    return dict(success=True,accepted=expected,cache_hit=same,reason=guard['reason'],score_gain=gain,
        baseline=baseline,candidate=candidate,
        saved_fit_states=baseline['saved_fit_states']+candidate['saved_fit_states'],
        unique_fit_states=baseline['saved_fit_states']+(0 if same else candidate['saved_fit_states']),
        scope='attained H-only predicted AC score check; no global separation or future rejection guarantee')


def audit_trial(trial):
    config=trial['config'];selection=trial['selection'];models=quality_audit(trial)
    require(trial['development_only'] and config['formal_calibration']==0 and config['formal_evaluation']==0
            and config['old_threshold_applied'] is False,'development scope/threshold')
    require(selection['uses_truth'] is False and selection['uses_future_noise'] is False,'selector flags')
    history=_obs(trial['history']);hm,fm,jm=(models[k] for k in ('history','future','joint'))
    supports={r['name']:r for r in trial['historical_fits']}
    # Passive proposal stores only nontrivial clades; rebuild full canonical supports.
    for row in trial['proposal']['bank']:
        if row['name'] not in supports:
            candidate=RadialCandidate.from_supports(row['name'],3,row['supports'])
            supports[row['name']]=dict(name=row['name'],supports=[list(s) for s in candidate.supports])
    candidates={name:dict(name=name,supports=row['supports']) for name,row in supports.items()}
    require(len(candidates)==4,'all four declared reduced candidates')
    H=fit_audits(trial['historical_fits'],candidates,history,hm,trial['historical_failures'],complete=not trial['historical_failures'])
    losses={r['name']:float(r['loss']) for r in trial['historical_fits'] if r.get('loss') is not None}
    moving=trial.get('moving_models') or []
    if moving:
        pairs={(m['a'],m['b']) for m in moving};required={(a,b) for a in candidates for b in candidates if a!=b}
        require(len(moving)==12 and pairs==required,'all twelve ordered competitors')
        for m in moving:
            moving_geometry_audit(m)
            background=m['future_background'];f=next(r for r in trial['historical_fits'] if r['name']==m['b'])
            for channel in ('p','q'):
                values=arr(getattr(history,channel))-arr(getattr(hm,'bias_'+channel))
                close(background[channel],np.tile(values.mean(axis=0),(4,1)), 'H-only terminal forecast:'+channel,atol=2e-12)
            for channel in ('source_p','source_q'):
                close(background[channel],np.full(4,arr(f[channel]).mean()),'H-only same-interface source forecast:'+channel,atol=2e-12)
            close(m['future_root_v'],np.full(4,np.mean(history.root_v)),'H-only root forecast',atol=2e-12)
            states=np.column_stack([arr(f[k]) for k in ('p','q','source_p','source_q')]).ravel()
            shared=np.r_[arr(f['r']),arr(f['x']),states]
            close(arr(m['domain']['base_center'])[:len(shared)],shared,'shared lines and H states in physical center',atol=2e-12)
    audit_rows=[];qp_count=0
    for method,rows in selection.get('curves_audit',{}).items():
        require(len(rows)==len(moving),'curve audit drops competitor')
        u=next(r['u_kw'] for r in trial['rows'] if r['method']==method)
        bypair={(m['a'],m['b']):m for m in moving}
        for row in rows:
            m=bypair[(row['a'],row['b'])]
            original=replay_qp(m,u,row['original'],domain='original')
            outer=replay_qp(m,u,row['outer'],domain='outer');qp_count+=2
            require(outer['lower']<=original['upper']+2e-7*max(1.,original['upper']), 'outer/original cost interval contradiction')
            expected=[.5*(original['lower']-losses[m['b']]),.5*(original['upper']-losses[m['b']])]
            close(row['added_score_interval'],expected,'original added score interval')
            audit_rows.append(dict(method=method,a=m['a'],b=m['b'],original=original,outer=outer))
    exchange=None
    if 'cross_solver' in selection:
        exchange=exchange_audit(moving,selection['cross_solver'],losses);qp_count+=exchange['qp_states']
    fixed=None
    if trial.get('fixed_models') and 'fixed_solver' in selection:
        fixed=fixed_score_audit(trial['fixed_models'],selection['fixed_solver'],losses)
    rows=[];state_count=H['states_replayed'];physical_count=0
    family=lambda supports:tuple(sorted(tuple(s) for s in supports if 1<len(s)<3))
    true_names=[name for name,c in candidates.items() if family(c['supports'])==family(trial['truth']['supports'])]
    names=[r['method'] for r in trial['rows']];require(len(names)==len(set(names)),'method duplicate')
    require(set(DEFAULT_METHODS)<=set(names),'required baseline/design method omitted')
    for row in trial['rows']:
        out=dict(method=row['method'],status=row['status'],candidate_miss=len(true_names)!=1)
        require(row['candidate_miss']==out['candidate_miss'],'candidate miss label')
        if row['u_kw'] is not None:
            out['operation']=operation_audit(row['u_kw'],config,diagnostic=row['method']=='none')
        if 'physical' in row:
            out['physical']=physical_audit(trial,row,fm);physical_count+=1
            if 'voltage_range' in row:
                close(row['voltage_range'],[out['physical']['voltage_min'],out['physical']['voltage_max']],
                      'saved physical voltage range',atol=2e-10)
                in_range=row['voltage_range'][0]>=.9 and row['voltage_range'][1]<=1.1
                require((row['status']=='physical_voltage_failure')==(not in_range),'physical voltage failure classification')
        if 'fits' in row:
            joint=concat_observations(history,_obs(row['observations']))
            fits=fit_audits(row['fits'],candidates,joint,jm,row['failures'],complete=row['status']=='complete')
            state_count+=fits['states_replayed'];out['fits']=fits
            if row['status']=='complete':
                js={r['name']:float(r['loss']) for r in row['fits']};require(row['losses']==js,'saved joint losses')
                if len(true_names)==1:
                    true=true_names[0];margin=min(v for k,v in js.items() if k!=true)-js[true]
                    close(row['margin'],margin,'actual margin');close(row['loglik_margin'],.5*margin,'actual loglik margin')
                    out.update(margin=margin,truth_is_best=js[true]<=min(js.values())+1e-9)
        rows.append(out)
    guard=selection.get('ac_guard');guard_count=0;guard_result=None
    if guard:
        guard_result=guard_audit(trial,guard,history,hm,fm,jm,candidates)
        guard_count=guard_result['saved_fit_states'];state_count+=guard_count
    return dict(seed=trial['base_seed'],quality=trial['quality']['kind'],success=True,
        history_fit_states=H['states_replayed'],saved_fit_states_replayed=state_count,physical_states_replayed=physical_count,
        proxy_QP_states_replayed=qp_count,guard_fit_states_replayed=guard_count,exchange=exchange,QP_rows=audit_rows,methods=rows,
        guard=guard_result,fixed=fixed,fallback=selection.get('fallback'),correlation=trial['correlation'],seconds=trial['seconds'])


def paired_identity_audit(trials):
    groups=defaultdict(list)
    for t in trials: groups[t['base_seed']].append(t)
    for seed, rows in groups.items():
        labels=[r['quality']['kind'] for r in rows];require(len(labels)==len(set(labels)),'duplicate H-quality condition')
        first=rows[0]
        for row in rows[1:]:
            for key in ('history','historical_fits','historical_failures','truth','white_future','seeds','correlation'):
                require(json_identity(row[key])==json_identity(first[key]),'same-H data or state changed:'+key+':'+str(seed))
    return dict(independent_histories=len(groups),quality_conditions=len(trials),quality_repetitions_not_independent=True)


def summary(audits,planned):
    methods=sorted({r['method'] for a in audits for r in a['methods']});numbers={}
    observed_histories=len({a['seed'] for a in audits})
    for method in methods:
        rows=[r for a in audits for r in a['methods'] if r['method']==method]
        margins=[r['margin'] for r in rows if 'margin' in r]
        numbers[method]=dict(planned_conditions=planned,available_conditions=len(rows),
            statuses=dict(Counter(r['status'] for r in rows)),candidate_miss=sum(r['candidate_miss'] for r in rows),
            margin_available=len(margins),truth_best=sum(r.get('truth_is_best',False) for r in rows),
            mean_margin=None if not margins else float(np.mean(margins)))
    contrasts={}
    for method,baseline in [('cross_raw_milp','constant'),('cross_raw_milp','quality_lp'),
                            ('cross_guarded_milp','constant'),('cross_guarded_milp','cross_raw_milp'),
                            ('fixed_mle_milp','quality_lp')]:
        grouped=defaultdict(list)
        for a in audits:
            rows={r['method']:r for r in a['methods']}
            if 'margin' in rows.get(method,{}) and 'margin' in rows.get(baseline,{}):
                grouped[a['seed']].append(dict(quality=a['quality'],delta=rows[method]['margin']-rows[baseline]['margin']))
        H=[]
        for seed,rows in grouped.items():
            complete={r['quality'] for r in rows}=={'uniform','temporal'}
            if complete:H.append(dict(seed=seed,delta=float(np.mean([r['delta'] for r in rows]))))
        contrasts[method+'_vs_'+baseline]=dict(planned_independent_histories=planned//2,
            observed_independent_histories=observed_histories,
            not_run_histories=planned//2-observed_histories,
            observed_incomplete_paired_histories=observed_histories-len(H),
            available_complete_paired_histories=len(H),unavailable_histories=planned//2-len(H),
            per_history_mean_quality_delta=H,positive=sum(r['delta']>1e-8 for r in H),
            mean_delta=None if not H else float(np.mean([r['delta'] for r in H])))
    return dict(methods=numbers,paired_history_contrasts=contrasts,
        scope='development conditions; pair qualities within each history; no calibrated topology rejection',
        saved_fit_states=sum(a['saved_fit_states_replayed'] for a in audits),
        physical_states=sum(a['physical_states_replayed'] for a in audits),proxy_QP_states=sum(a['proxy_QP_states_replayed'] for a in audits),
        guard_fit_states=sum(a['guard_fit_states_replayed'] for a in audits),
        guard_unique_fit_states=sum(a['guard']['unique_fit_states'] for a in audits if a.get('guard')),
        guard_acceptances=sum(a['guard']['accepted'] for a in audits if a.get('guard')),
        invalid_proxy_global_intervals=sum(not a['exchange']['valid_global_proxy_interval'] for a in audits if a.get('exchange')),
        formal_calibration=0,formal_evaluation=0,new_optimizer_calls=0,new_selection_calls=0,new_AC_fit_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    freeze=read_json(args.directory/'freeze.json');files=sorted(args.directory.glob('trial_*.json.gz'))
    trials=[read_json(f) for f in files];config=freeze['config'];planned=len(config['seeds'])*len(config['quality'])
    require(args.allow_partial or len(trials)==planned,'trial family incomplete; use explicit --allow-partial for pilot review')
    for t in trials:
        require(t['config']==config and t['source_sha256']==freeze['source_sha256'],'trial freeze identity')
    actual={(t['base_seed'],t['quality']['kind']) for t in trials}
    expected={(seed,q) for seed in config['seeds'] for q in config['quality']}
    require(len(actual)==len(trials) and actual<=expected,'declared seed-quality coverage/duplicates')
    require(args.allow_partial or actual==expected,'declared complete seed-quality coverage')
    source=source_audit(args.directory,freeze);pairing=paired_identity_audit(trials)
    audits=[audit_trial(t) for t in trials]
    output=dict(schema='cross_all_independent_review_v1',success=True,source=source,pairing=pairing,
        planned_conditions=planned,available_conditions=len(trials),complete_trial_family=len(trials)==planned,
        numbers=summary(audits,planned),trials=audits,
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reviewed_trial_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    require(not args.output.exists(),'review output exists; choose a new file')
    write_json(args.output,output)
    print(output['numbers'])


if __name__=='__main__': main()
