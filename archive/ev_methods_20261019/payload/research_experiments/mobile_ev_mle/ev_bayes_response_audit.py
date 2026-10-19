"""Read-only replay of saved bounded-line Bayesian EV evidence.

No case generator, optimizer, fit or scientific prediction function is called.
Saved posterior moments feed a separate radial AC sweep and differentiated
nodal equations.  Recorded meter draws, posterior probabilities, coverage,
initial ranking and each full-risk acceptance are independently reconstructed.
GHK replay is opt-in for explicitly named histories and the first candidate;
the expensive bounded integration is not silently repeated for the full batch.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from . import ev_bayes_small_audit as independent
from . import ev_mobile_v4_domain as nested
from . import gradient_price_study as io
from .ev_mobile_service_domain import MobileServiceConfig
from .likelihood import RadialCandidate


HERE = Path(__file__).resolve().parent
DOMAINS = ('fixed_charge','fixed_bidir','mobile_charge','mobile_bidir')
RISK_FAMILIES = ('ranked','joint_oldbc','joint_bc','fixed_bc')
EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError, AssertionError,
              FloatingPointError, RuntimeError, np.linalg.LinAlgError, ArithmeticError)


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _array(value):
    return independent._array(value)


def _close(actual, expected, label, *, atol=3e-7, rtol=3e-7):
    return independent._close(actual, expected, label, atol=atol, rtol=rtol)


def _psd(value, label):
    a = _array(value)
    _require(a.ndim == 2 and a.shape[0] == a.shape[1], label+' square')
    _close(a, a.T, label+' symmetry', atol=1e-10, rtol=1e-10)
    eigen = np.linalg.eigvalsh((a+a.T)*.5)
    _require(eigen.min(initial=0.) >= -1e-8*max(float(abs(eigen).max(initial=0.)),1e-20), label+' PSD')
    return a


def _eta_lines(eta):
    eta = _array(eta).reshape(-1,2)
    return np.exp(eta[:,0]), np.exp(eta.sum(axis=1))


def _line_domain(eta, bounds, label):
    a = _array(eta).reshape(-1,2)
    for v, key in ((a[:,0],'log_R'),(a[:,1],'log_ratio'),(a.sum(axis=1),'log_X')):
        b = _array(bounds[key])
        _require(np.all(v >= b[:,0]-1e-9) and np.all(v <= b[:,1]+1e-9), label+' original '+key)


def restore_joint(mean, covariance, line_mean, line_covariance):
    """Independent Gaussian regression identity, including cross covariance."""
    m, C, lm, LC = map(_array,(mean,covariance,line_mean,line_covariance))
    d = len(lm)
    try:
        A = np.linalg.solve(C[:d,:d],C[:d,d:]).T
    except np.linalg.LinAlgError:
        A = C[d:,:d]@np.linalg.pinv(C[:d,:d],rcond=1e-12)
    S = C[d:,d:]-A@C[:d,d:]
    return np.r_[lm,m[d:]+A@(lm-m[:d])], np.block([[LC,LC@A.T],[A@LC,S+A@LC@A.T]])


def _line_jacobian(candidate,r,x,physical,meter):
    """Independent implicit derivative of radial equations in interleaved eta."""
    V, load = physical['voltage'],physical['load']
    T,m = V.shape
    n = candidate.n
    Z = (candidate.B*(r+1j*x))@candidate.B.T
    coefficient = np.conj(load)/np.conj(V)**2
    K = Z[None]*coefficient[:,None,:]
    eye = np.broadcast_to(np.eye(m),K.shape)
    matrix = np.concatenate((np.concatenate((eye-K.real,-K.imag),axis=2),
                             np.concatenate((-K.imag,eye+K.real),axis=2)),axis=1)
    current = np.conj(load/V)
    branch = current@candidate.B
    rhs = np.zeros((T,m,2*m),complex)
    for e in range(m):
        direction = candidate.B[:,e][None,:]*branch[:,e,None]
        rhs[:,:,2*e] = -(r[e]+1j*x[e])*direction
        rhs[:,:,2*e+1] = -1j*x[e]*direction
    solved = np.linalg.solve(matrix,np.concatenate((rhs.real,rhs.imag),axis=1))
    dV = solved[:,:m]+1j*solved[:,m:]
    dI = -coefficient[:,:,None]*np.conj(dV)
    dmaster = physical['root_v'][:,None]*np.conj(dI.sum(axis=1))
    ids = candidate.terminal_edges
    dmag = (np.conj(V[:,ids,None])*dV[:,ids]).real/abs(V[:,ids,None])
    blocks = []
    for channel in independent.CHANNELS:
        count = n if channel in ('p','q','v') else 1
        if channel == 'v':
            local = dmag
        elif channel in ('master_p','master_q'):
            local = (dmaster.real if channel == 'master_p' else dmaster.imag)[:,None,:]
        else:
            local = np.zeros((T,count,2*m))
        sigma = np.broadcast_to(_array(meter[channel+'_sigma']),(T,n) if count == n else (T,))
        blocks.append((local/sigma.reshape(T,count,1)).reshape(T*count,2*m))
    return np.vstack(blocks)


def _posterior_rows(record, candidates, model):
    rows = []
    prediction = record['prediction']['fixed' if model == 'fixed' else 'joint']
    fits = {v.get('candidate',v.get('name')):v for v in record['training_result']['fits']}
    for name,candidate in candidates.items():
        row = prediction['candidate_updates'][name]
        if model == 'fixed':
            r,x = _array(fits[name]['r']),_array(fits[name]['x'])
            latent = _array(row['future_latent_mean'])
            C = _array(row['future_latent_covariance'])
            d = 0
        else:
            posterior = row['TV_posterior_future']
            d = len(row['line_prior']['mean'])
            r,x = _eta_lines(_array(posterior['mean'])[:d])
            latent = _array(posterior['mean'])[d:].reshape(len(record['methods']['none']['power_kw']),2*candidate.n+2)
            C = _array(posterior['cov']).copy()
            if model == 'no_cross':
                C[:d,d:] = 0.; C[d:,:d] = 0.
            elif model == 'line_known':
                try:A = np.linalg.solve(C[:d,:d],C[:d,d:])
                except np.linalg.LinAlgError:A = np.linalg.pinv(C[:d,:d],rcond=1e-12)@C[:d,d:]
                S = C[d:,d:]-C[d:,:d]@A
                C[:] = 0.; C[d:,d:] = (S+S.T)*.5
        rows.append(dict(candidate=candidate,r=r,x=x,latent=latent,cov=C,line_dimension=d))
    lp = prediction['log_prior_evidence_only' if model == 'evidence_only' else 'log_prior']
    return rows,_array(lp),float(prediction['base_kw'])


def reconstruct_forecast(record,candidates,power,model):
    rows,lp,base_kw = _posterior_rows(record,candidates,model)
    power = _array(power)
    fm = record['future_meter']
    means,covariances,Fs,parts = [],[],[],[]
    for row in rows:
        latent = row['latent'].copy()
        latent[:,:row['candidate'].n] += power/base_kw
        physical = independent._physics(row['candidate'],row['r'],row['x'],latent)
        mean = independent._whiten(physical,fm)
        state = independent._jacobian(row['candidate'],row['r'],row['x'],physical,fm)
        d,C = row['line_dimension'],row['cov']
        if d:
            line = _line_jacobian(row['candidate'],row['r'],row['x'],physical,fm)
            F = np.c_[line,state]
            line_cov = line@C[:d,:d]@line.T
            state_cov = state@C[d:,d:]@state.T
            cross = line@C[:d,d:]@state.T
            cross += cross.T
        else:
            F = state
            line_cov = np.zeros((len(mean),len(mean)))
            state_cov = state@C@state.T
            cross = np.zeros_like(line_cov)
        S = np.eye(len(mean))+line_cov+state_cov+cross
        S = (S+S.T)*.5
        np.linalg.cholesky(S)
        means.append(mean);covariances.append(S);Fs.append(F)
        parts.append(dict(line=line_cov,state=state_cov,cross=cross,latent=latent))
    return dict(bank=dict(means=np.asarray(means)[None],covariances=np.asarray(covariances)[None],
                         prior=np.exp(lp),log_prior=lp,weights=np.ones(1)),joint_jacobian=Fs,parts=parts)


def actual_statistics(bank,clean,white,true_index,horizon,sites,credible_mass=.9):
    """Independent posterior, deterministic tie rule and marginal interval coverage."""
    means,cov,lp = _array(bank['means'])[0],_array(bank['covariances'])[0],_array(bank['log_prior'])
    values = _array(clean)[None]+_array(white)
    post,logpost = independent._logdensity(values,means,cov,lp)
    chosen = np.argmax(post,axis=1)
    order = np.argsort(-post,axis=1,kind='stable')
    ordered = np.take_along_axis(post,order,axis=1)
    selected = np.cumsum(ordered,axis=1)-ordered < credible_mass
    membership = np.zeros_like(selected)
    np.put_along_axis(membership,order,selected,axis=1)
    size = membership.sum(axis=1)
    truth = np.eye(post.shape[1])[true_index]
    mu,var = means[true_index],np.diag(cov[true_index])
    covered = abs(values-mu) <= 1.6448536269514722*np.sqrt(var)
    vi = np.arange(2*horizon*sites,3*horizon*sites)
    samples = dict(correct=(chosen==true_index).astype(int),confidence=post.max(axis=1),
        true_probability=post[:,true_index],posterior=post,brier=np.sum((post-truth)**2,axis=1),
        negative_log_true_probability=-logpost[:,true_index],
        unique_correct=((size==1)&(chosen==true_index)).astype(int),
        wrong_unique=((size==1)&(chosen!=true_index)).astype(int),unresolved=(size>1).astype(int),
        true_rejected=(~membership[:,true_index]).astype(int),credible_size=size,
        meter_interval90_coverage=covered.mean(axis=1),voltage_interval90_coverage=covered[:,vi].mean(axis=1))
    return dict(samples=samples,summary={k:float(np.mean(v)) for k,v in samples.items() if k!='posterior'})


def selector_interface_audit(source):
    """Check explicit planning boundaries in frozen source, not a general sandbox proof."""
    checks = {}
    for filename,functions in {
        'ev_bayes_response_ac.py':('build_context','forecast_bank'),
        'ev_bayes_response_study.py':('risk_family',),
    }.items():
        tree = ast.parse(source[filename])
        for name in functions:
            function = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            args = [a.arg for a in (*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs)]
            _require(not {'truth','actual','white','future_noise','true_lines'}&set(args),filename+'/'+name+' observed-input interface')
            checks[filename+'/'+name] = args
    tree = ast.parse(source['ev_bayes_response_study.py'])
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Objective')
    names = {n.id for n in ast.walk(cls) if isinstance(n,ast.Name)}
    attributes = {n.attr for n in ast.walk(cls) if isinstance(n,ast.Attribute)}
    _require(not {'truth','actual','white','future_noise','true_lines'}&(names|attributes),'Objective contains physical truth or realized future noise')
    checks['Objective'] = 'context + planning-integration noise + predictor only'
    return checks


def audit_record(record,*,replay_integrals=True,spotcheck_ghk=False):
    began=perf_counter();errors=[];counts=Counter(independent_histories=1);maxima=Counter();cache={}
    def attempt(label,fn):
        try:return fn()
        except EXCEPTIONS as exc:
            errors.append(dict(check=label,reason=type(exc).__name__+': '+str(exc)));return None
    def compare(a,b,label,key='numeric_replay',**kw):
        error=_close(a,b,label,**kw);maxima[key]=max(maxima[key],error)
    if record.get('status') not in ('complete','inference_failure'):
        documented=bool(record.get('failure') or record.get('failures') or record.get('training_result'))
        return dict(success=documented,scientific_complete=False,spec=record.get('spec'),status=record.get('status'),
            counts=dict(counts),maxima={},errors=[] if documented else [dict(check='retained_failure',reason='failure reason missing')],
            unverified_due_failure=['inference, plans, predictive probabilities and coverage'],seconds=perf_counter()-began)
    required=('prediction','training_result','future_meter','observed_training','observed_evidence','methods','actions','operation_config')
    missing=[k for k in required if k not in record]
    if missing:
        return dict(success=False,scientific_complete=False,spec=record.get('spec'),status=record.get('status'),counts=dict(counts),maxima={},
                    errors=[dict(check='schema',reason='missing '+','.join(missing))],unverified_due_failure=missing)
    cfg=record['config'];T=int(cfg['horizon']);n=len(record['observed_training']['p'][0]);w=2*n+2
    names=record['candidate_names'];K=len(names);seed=int(record['spec']['seed']);dim=T*(3*n+2)
    fits={v.get('candidate',v.get('name')):v for v in record['training_result']['fits']}
    candidates={name:RadialCandidate.from_supports(name,n,[tuple(s) for s in fits[name]['supports'] if 1<len(s)<n]) for name in names}
    domains=nested.build_nested_domains(MobileServiceConfig(**record['operation_config']))
    counts.update(candidate_slots=K,method_slots=len(record['methods']),unique_action_slots=len(record['actions']))
    planning=np.random.default_rng([seed,2000,20261017]).standard_normal((1,K,int(cfg['planning_samples_per_candidate']),dim))
    independent_noise=np.random.default_rng([seed,3000,20261017]).standard_normal((1,K,int(cfg['independent_samples_per_candidate']),dim))
    white=np.random.default_rng([seed,4000,20261017]).standard_normal((int(cfg['actual_noise_repeats']),dim))
    true_index=names.index(record['true_name'])
    def forecast(power,model):
        key=(nested.action_key(power),model)
        if key not in cache:cache[key]=reconstruct_forecast(record,candidates,power,model)
        return cache[key]
    planning_scores={}
    def risk(power,model):
        key=(nested.action_key(power),model)
        if key not in planning_scores:
            planning_scores[key]=independent._integration(forecast(power,model)['bank'],planning,float(cfg['abstention_cost']))[0]['risk']
        return planning_scores[key]
    def joint_check():
        pred=record['prediction']['joint'];opts=pred['options'];all_log=[];evidence_log=[]
        _require(pred['source_interface']==1 and pred['shared_lines_all_times'],'shared physical source and line scope')
        _require(pred['training_raw_likelihood_count']==1 and pred['fitted_source_pseudo_likelihood_count']==0,'training information used once')
        for name,candidate in candidates.items():
            row=pred['candidate_updates'][name];prior=row['line_prior'];d=2*candidate.m
            for parameter,key in (('r_bounds','log_R'),('x_bounds','log_X'),('ratio_bounds','log_ratio')):
                compare(prior['bounds'][key],np.broadcast_to(np.log(_array(opts[parameter])),(candidate.m,2)),
                        'original declared '+parameter+' '+name,'physical_domain',atol=1e-13,rtol=1e-13)
            _line_domain(row['training_center']['eta'],prior['bounds'],'training center '+name)
            rr,xx=_eta_lines(row['training_center']['eta'])
            compare(rr,fits[name]['r'],'training R center '+name,'physical_domain',atol=1e-12,rtol=1e-12)
            compare(xx,fits[name]['x'],'training X center '+name,'physical_domain',atol=1e-12,rtol=1e-12)
            for column,key in ((-2,'source_p_bounds'),(-1,'source_q_bounds')):
                source=_array(row['training_center']['latent'])[:,column]
                _require(np.all(source>=opts[key][0]-1e-9) and np.all(source<=opts[key][1]+1e-9),'training source original bounds '+name)
            compare(prior['mean'][0::2],np.mean(_array(prior['bounds']['log_R']),axis=1),'public prior a center')
            compare(prior['mean'][1::2],np.mean(_array(prior['bounds']['log_ratio']),axis=1),'public prior b center')
            expected_var=np.empty(d);expected_var[0::2]=np.diff(_array(prior['bounds']['log_R']),axis=1).ravel()**2/12
            expected_var[1::2]=np.diff(_array(prior['bounds']['log_ratio']),axis=1).ravel()**2/12
            compare(prior['cov'],np.diag(expected_var),'public prior covariance')
            for stage in ('T','TV'):
                integration=row[stage+'_line_integration'];diag=integration['diagnostics'];settings=row[stage+'_integration_options']
                _require(integration['status']=='ok' and diag['precision_met'],'bounded precision '+stage+'/'+name)
                _require(np.isfinite(integration['log_mass']) and integration['log_mass']<=1e-10,'bounded log mass '+stage+'/'+name)
                _require(diag['relative_se']<=settings.get('relative_se_tolerance',.01)+1e-14,'bounded mass SE '+stage+'/'+name)
                _require(diag['ess']>=max(settings.get('min_ess',512.),settings.get('min_ess_fraction',.02)*diag['points'])-1e-8,'bounded ESS '+stage+'/'+name)
                _require(diag['mean_standardized_se_max']<=settings.get('mean_se_tolerance',.01)+1e-14,'bounded mean SE '+stage+'/'+name)
                _require(diag['covariance_relative_se_fro']<=settings.get('covariance_se_tolerance',.02)+1e-14,'bounded covariance SE '+stage+'/'+name)
                _line_domain(integration['mean'],prior['bounds'],stage+' bounded mean '+name)
                _psd(integration['cov'],stage+' line covariance '+name)
                untr=row[stage+'_untruncated_future'];bounded=row[stage+'_posterior_future']
                mu,C=restore_joint(untr['mean'],untr['cov'],integration['mean'],integration['cov'])
                compare(bounded['mean'],mu,stage+' restored mean '+name,'joint_mean')
                compare(bounded['cov'],C,stage+' restored covariance '+name,'joint_covariance')
                _psd(bounded['cov'],stage+' joint covariance '+name)
                counts['bounded_moment_stages']+=1
            logH=row['training_Gaussian_log_predictive']+row['logZT']-row['logZprior']
            logV=row['evidence_Gaussian_log_predictive']+row['logZTV']-row['logZT']
            compare(row['training_log_predictive'],logH,'training truncated density ratio '+name,'log_evidence',atol=1e-9,rtol=1e-12)
            compare(row['evidence_log_predictive'],logV,'evidence truncated density ratio '+name,'log_evidence',atol=1e-9,rtol=1e-12)
            compare(row['logZprior'],prior['log_mass'],'proper prior normalization '+name,'log_evidence',atol=1e-12,rtol=1e-12)
            m,C=_array(row['T_untruncated_EF_mean']),_array(row['T_untruncated_EF_covariance'])
            J,offset,y=map(_array,(row['evidence_affine_jacobian'],row['evidence_affine_offset'],row['evidence_observed_whitened']))
            obs=offset+J@m;S=np.eye(len(y))+J@C@J.T;factor=cho_factor((S+S.T)*.5,lower=True)
            compare(row['evidence_predicted_mean'],obs,'evidence predictive mean '+name,'evidence_mean')
            compare(row['evidence_covariance'],S,'evidence predictive covariance '+name,'evidence_covariance')
            cross=C@J.T;res=y-obs;mu=m+cross@cho_solve(factor,res);post=C-cross@cho_solve(factor,cross.T)
            V=int(pred['evidence_count']);ix=np.r_[np.arange(d),d+np.arange(V*w,(V+T)*w)]
            compare(row['TV_untruncated_future']['mean'],mu[ix],'evidence Gaussian updated mean '+name,'update_mean')
            compare(row['TV_untruncated_future']['cov'],post[np.ix_(ix,ix)],'evidence Gaussian updated covariance '+name,'update_covariance')
            logdensity=-.5*(res@cho_solve(factor,res)+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))
            compare(row['evidence_Gaussian_log_predictive'],logdensity,'evidence Gaussian density '+name,'log_evidence',atol=2e-7,rtol=1e-10)
            all_log.append(logH+logV);evidence_log.append(logV)
            counts['evidence_updates']+=1
        compare(pred['log_prior'],np.asarray(all_log)-logsumexp(all_log),'all-history topology posterior','log_prior',atol=2e-9,rtol=1e-11)
        compare(pred['log_prior_evidence_only'],np.asarray(evidence_log)-logsumexp(evidence_log),'evidence-only topology posterior','log_prior',atol=2e-9,rtol=1e-11)
        if spotcheck_ghk:
            from .ev_bayes_response_truncation import bounded_line_moments
            row=pred['candidate_updates'][names[0]];d=2*candidates[names[0]].m
            for stage in ('T','TV'):
                untr=row[stage+'_untruncated_future'];options=dict(row[stage+'_integration_options'],return_samples=False)
                repeated=bounded_line_moments(_array(untr['mean'])[:d],_array(untr['cov'])[:d,:d],row['line_prior']['bounds'],**options)
                _require(repeated['status']=='ok','preselected GHK replay precision')
                for field in ('mean','cov','log_mass'):
                    compare(row[stage+'_line_integration'][field],repeated[field],'preselected GHK '+stage+'/'+field,'GHK_replay',atol=1e-10,rtol=1e-10)
                counts['preselected_GHK_replays']+=1
    if 'joint' in record['prediction']:attempt('joint_posterior',joint_check)
    def methods_check():
        for name,row in record['methods'].items():
            u=_array(row['power_kw']);_require(nested.action_key(u)==row['action_key'],'action identity '+name)
            _require(row['action_key'] in record['actions'],'action present '+name)
            if name=='none':continue
            domain=domains[row['domain']];x=_array(row['operation_x'])
            _require(domain.check(x)['feasible'],'original operation domain '+name)
            compare(domain.decode(x)['u_kw'],u,'decoded power '+name,'operation_power',atol=1e-9,rtol=1e-9)
            counts['operation_plans']+=1
    attempt('operation_plans',methods_check)
    for key,action in record['actions'].items():
        def action_check(key=key,action=action):
            u=_array(action['power_kw']);_require(nested.action_key(u)==key,'unique action identity')
            for regime,actual in action['actual'].items():
                truth=record['true_future_inputs_evaluation_only'][regime]
                base_kw=float(record['provenance']['base_kw'])
                latent=np.c_[_array(truth['p'])+u/base_kw,_array(truth['q']),
                             _array(truth['source_p']),_array(truth['source_q'])]
                physical=independent._physics(candidates[record['true_name']],_array(record['true_lines']['r']),
                                             _array(record['true_lines']['x']),latent,_array(truth['root_v']))
                compare(actual['clean_whitened'],independent._whiten(physical,record['future_meter']),
                        'post-hoc physical AC future '+regime,'physical_future_mean')
                compare(actual['v_min'],physical['v'].min(),'physical future minimum V','physical_voltage')
                compare(actual['v_max'],physical['v'].max(),'physical future maximum V','physical_voltage')
                _require(actual['ac_equation_residual']<1e-8 and actual['power_balance_residual']<1e-8,'saved physical residuals')
                counts['posthoc_physical_action_regimes']+=1
            for model,predicted in action['predictions'].items():
                replay=forecast(u,model);saved=predicted['forecast'];bank=saved['bank']
                for field in ('means','covariances','log_prior'):
                    compare(bank[field],replay['bank'][field],'forecast '+model+'/'+field,'forecast_'+field)
                if model!='fixed':
                    for index,(F,part) in enumerate(zip(replay['joint_jacobian'],replay['parts'])):
                        compare(saved['joint_jacobian'][index],F,'joint AC Jacobian '+model,'joint_AC_jacobian')
                        for component in ('line','state','cross'):
                            compare(saved['physical'][index]['covariance_parts'][component],part[component],'covariance block '+model+'/'+component,'covariance_parts')
                        compare(saved['physical'][index]['latent_inputs_with_EV'],part['latent'],'EV inserted once '+model,'EV_once')
                        _require(saved['physical'][index]['EV_added_once'] and saved['physical'][index]['source_interface']==1,'EV/source flags')
                checked=predicted['independent_score']
                _require(checked['noise_sha256']==sha256(independent_noise.astype('<f8').tobytes()).hexdigest(),'independent integration stream identity')
                if replay_integrals:
                    score,_=independent._integration(replay['bank'],independent_noise,float(cfg['abstention_cost']),float(cfg['credible_mass']))
                    for metric in ('risk','eig','brier','brier_risk','abstention_risk','map_error'):
                        compare(checked[metric],score[metric],'independent score '+model+'/'+metric,'independent_score',atol=2e-8,rtol=2e-7)
                    counts['independent_bank_integrals']+=1
                for regime,actual in action['actual'].items():
                    _require(actual['noise_seed']==[seed,4000,20261017],'paired actual noise stream')
                    computed=actual_statistics(replay['bank'],actual['clean_whitened'],white,true_index,T,n,float(cfg['credible_mass']))
                    observed=actual['models'][model]
                    for field,value in computed['samples'].items():
                        compare(observed['samples'][field],value,'actual '+regime+'/'+model+'/'+field,'actual_posterior',atol=3e-8,rtol=3e-7)
                    for field,value in computed['summary'].items():
                        compare(observed['summary'][field],value,'actual summary '+field,'actual_summary',atol=3e-8,rtol=3e-7)
                    _require(observed['n_independent_histories']==1,'noise replicates are one history')
                    counts['actual_bank_regimes']+=1;counts['conditional_meter_draw_slots']+=len(white)
                counts['forecast_banks']+=1
            if 'response_increment' in action:
                zero=record['methods']['none']['power_kw'];F=forecast(u,'joint');G=forecast(zero,'joint')
                for index,inc in enumerate(action['response_increment']['predictions']):
                    row=record['prediction']['joint']['candidate_updates'][names[index]];d=len(row['line_prior']['mean']);C=_array(row['TV_posterior_future']['cov'])
                    D=F['joint_jacobian'][index]-G['joint_jacobian'][index]
                    total=D@C@D.T;line=D[:,:d]@C[:d,:d]@D[:,:d].T;state=D[:,d:]@C[d:,d:]@D[:,d:].T
                    for field,value in (('variance',np.maximum(np.diag(total),0.)),('line_variance',np.diag(line)),('state_variance',np.diag(state)),('cross_variance',np.diag(total-line-state))):
                        compare(inc[field],value,'shared response '+field,'shared_response_covariance')
                    compare(inc['mean'],F['bank']['means'][0,index]-G['bank']['means'][0,index],'shared response mean','shared_response_mean')
                    counts['shared_response_pairs']+=1
                _require(not action['response_increment']['meter_noise_included'] and action['response_increment']['shared_line_and_background'],'shared-response interpretation')
        attempt('action/'+key,action_check)
    def inheritance_check():
        for family in ('square','square_budget'):
            if family not in record['planning_logs']:continue
            weights=np.broadcast_to(_array(record['future_meter']['v_sigma']),(T,n));least=float(weights.min());weights=(least/weights)**2 if least<=1. else 1./weights**2
            for domain in DOMAINS:
                method=record['methods'][family+'__'+domain];Q=float(np.sum(weights*_array(method['power_kw'])**2))
                compare(record['planning_logs'][family][domain]['Q'],Q,'SQ Q '+family+'/'+domain,'square_Q',atol=1e-8,rtol=1e-9)
                for parent in nested.PARENTS[domain]:
                    oldQ=float(np.sum(weights*_array(record['methods'][family+'__'+parent]['power_kw'])**2))
                    _require(Q>=oldQ-1e-8,'SQ parent objective '+family+'/'+domain)
                    counts['square_inheritance_edges']+=1
        for family in RISK_FAMILIES:
            if family not in record['planning_logs']:continue
            model='fixed' if family=='fixed_bc' else 'joint'
            for domain in DOMAINS:
                log=record['planning_logs'][family][domain]
                pool=[record['methods'][family+'__'+parent]['power_kw'] for parent in nested.PARENTS[domain]]
                pool.extend(record['methods'][name]['power_kw'] for name in ('constant','highlow'))
                pool.append(record['methods']['square__'+domain]['power_kw'])
                if family in ('joint_oldbc','joint_bc'):pool.append(record['methods']['ranked__'+domain]['power_kw'])
                best_power=pool[0];best=risk(best_power,model)
                for power in pool[1:]:
                    value=risk(power,model)
                    if value<best-1e-12:best_power,best=power,value
                compare(log['initial_risk'],best,'risk-ranked initial '+family+'/'+domain,'planning_risk',atol=3e-8,rtol=3e-7)
                accepted=failed=0
                for event in log['events']:
                    compare(event['parent_risk'],best,'step parent risk','planning_risk',atol=3e-8,rtol=3e-7)
                    mip=event['milp'];failed+=not bool(mip.get('success'))
                    if mip.get('operation_x') is None:
                        _require(not event['accepted'],'failed proposal retained')
                        counts['failed_MILP_proposals']+=1;continue
                    x=_array(mip['operation_x']);_require(domains[domain].check(x)['feasible'],'proposal original operation domain')
                    power=domains[domain].decode(x)['u_kw']
                    if 'full_AC_failure' in event:
                        _require(not event['accepted'],'AC failure retained');counts['retained_full_AC_failures']+=1;continue
                    proposed=risk(power,model)
                    compare(event['proposed_risk'],proposed,'proposal full AC risk','planning_risk',atol=3e-8,rtol=3e-7)
                    compare(event['risk_gain'],best-proposed,'proposal gain','planning_risk',atol=3e-8,rtol=3e-7)
                    expected_accept=proposed<best-1e-12
                    # Numerically tied replay is reported as a tie, not forced
                    # across two independently implemented AC factorizations.
                    if abs(proposed-best)>1e-8:
                        _require(bool(event['accepted'])==expected_accept,'full-risk acceptance decision')
                    if event['accepted']:
                        _require(proposed<=best+3e-8,'accepted planning risk not increased')
                        best_power,best=power,proposed;accepted+=1
                    if mip.get('success'):
                        _require(mip['proxy_value']>=mip['frozen_C']-2e-7,'BC proxy upper direction')
                        _require(mip['proxy_actual_error']<=mip['proxy_uniform_error']+2e-7,'BC error certificate')
                        if mip.get('solver_best_bound') is not None:_require(mip['solver_best_bound']<=mip['proxy_value']+2e-7,'MIN solver bound direction')
                        if family=='joint_bc':_require(abs(mip['anchor_proxy_error'])<1e-7,'new proxy parent touching')
                    counts['replayed_proposals']+=1
                compare(log['final_risk'],best,'final planning risk','planning_risk',atol=3e-8,rtol=3e-7)
                compare(record['methods'][family+'__'+domain]['power_kw'],best_power,'selected power after replay','selected_power',atol=1e-8,rtol=1e-8)
                _require(log['accepted_proposals']==accepted and log['failed_proposals']==failed,'proposal denominators '+family+'/'+domain)
                for parent in nested.PARENTS[domain]:
                    pr=risk(record['methods'][family+'__'+parent]['power_kw'],model)
                    compare(log['parent_risks'][parent],pr,'saved direct parent risk','planning_risk',atol=3e-8,rtol=3e-7)
                    _require(best<=pr+3e-8,'inherited same-objective parent '+family+'/'+domain)
                    counts['risk_inheritance_edges']+=1
    attempt('initialization_acceptance_and_inheritance',inheritance_check)
    return dict(success=not errors,scientific_complete=record.get('status')=='complete',spec=record.get('spec'),status=record.get('status'),
        counts=dict(counts),maxima=dict(maxima),errors=errors,unverified_due_failure=list(record.get('failures',{})),
        seconds=perf_counter()-began,GHK_replay_preselected=bool(spotcheck_ghk),no_new_fits_or_actions=True,
        truth_used_posthoc_only=True,probability_scope='local bounded joint moments and stated Gaussian future model')


def run(directory,output,*,phases=('development',),replay_integrals=True,ghk_histories=()):
    directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError(output)
    began=perf_counter();frozen=io.read_json(directory/'freeze.json');snapshot=io.read_json(directory/'source_snapshot.json.gz')
    errors=[];source={}
    for name,expected in frozen['source_sha256'].items():
        try:
            payload=base64.b64decode(snapshot['source_base64'][name],validate=True)
            _require(sha256(payload).hexdigest()==expected,'snapshot SHA256')
            path=HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
            _require(sha256(path.read_bytes()).hexdigest()==expected,'current frozen-source SHA256')
            source[name]=payload.decode('utf-8-sig')
        except (KeyError,ValueError,OSError,UnicodeError) as exc:errors.append(dict(source=name,reason=str(exc)))
    interfaces=None
    try:interfaces=selector_interface_audit(source)
    except EXCEPTIONS as exc:errors.append(dict(source='selector_interfaces',reason=str(exc)))
    rows=[];coverage={};selected=set(ghk_histories)
    for phase in phases:
        present=[]
        for path in sorted((directory/phase).glob('history_*.json.gz')):
            record=io.read_json(path);present.append(record.get('spec'))
            try:
                _require(record.get('policy_id')==frozen['policy_id'],'record/freeze policy identity')
                check=phase+':'+str(record['spec']['index'])
                result=audit_record(record,replay_integrals=replay_integrals,spotcheck_ghk=check in selected)
            except EXCEPTIONS as exc:
                result=dict(success=False,scientific_complete=False,spec=record.get('spec'),status=record.get('status'),counts={},maxima={},
                    errors=[dict(check='record',reason=type(exc).__name__+': '+str(exc))],unverified_due_failure=[])
            result.update(input_file=str(path.resolve()),input_sha256=sha256(path.read_bytes()).hexdigest());rows.append(result)
            print('RESPONSE_AUDIT',phase,path.name,result['success'],flush=True)
        expected=frozen['specifications'].get(phase,[])
        ids=[io.identity(v,{}) for v in present];targets=[io.identity(v,{}) for v in expected]
        coverage[phase]=dict(expected_histories=len(targets),present_histories=len(ids),
            exact_coverage=len(ids)==len(set(ids)) and set(ids)==set(targets),
            missing=[v for v in expected if io.identity(v,{}) not in ids],unexpected=[v for v in present if io.identity(v,{}) not in targets])
    counts,maxima=Counter(),Counter()
    for row in rows:
        counts.update(row['counts'])
        for k,v in row['maxima'].items():maxima[k]=max(maxima[k],v)
    result=dict(schema='ev_bayes_response_independent_audit_v1',input_root=str(directory.resolve()),policy_id=frozen['policy_id'],
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),source_errors=errors,selector_interfaces=interfaces,
        success=not errors and all(v['exact_coverage'] for v in coverage.values()) and all(r['success'] for r in rows),
        scientific_complete_count=sum(r['scientific_complete'] for r in rows),coverage=coverage,counts=dict(counts),maxima=dict(maxima),records=rows,
        status_counts=dict(Counter(r['status'] for r in rows)),error_count=len(errors)+sum(len(r['errors']) for r in rows),
        GHK_preselected_histories=sorted(selected),seconds=perf_counter()-began,no_new_fits_or_actions=True)
    io.write_json(output,result,compressed=output.suffix=='.gz')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',nargs='+',choices=('development','calibration','evaluation'),default=['development'])
    parser.add_argument('--skip-integrals',action='store_true')
    parser.add_argument('--ghk-history',action='append',default=[],help='explicit phase:index; first candidate only')
    args=parser.parse_args()
    result=run(args.input,args.output,phases=tuple(args.phase),replay_integrals=not args.skip_integrals,ghk_histories=args.ghk_history)
    print('RESPONSE_AUDIT_SAVED',result['success'],result['error_count'],flush=True)


if __name__=='__main__':main()
