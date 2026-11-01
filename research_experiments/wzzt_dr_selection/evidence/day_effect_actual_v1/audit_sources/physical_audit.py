"""Post-score independent audit; never generate, solve, fit or change a plan.

Truth/physical packets are opened only after all prediction, score and final
summary references have passed their hash gates. Output contains scalars only.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pickle,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import beta,binom


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('hash mismatch: '+str(p))
    return p
def unpickle(r):
    with checked(r).open('rb') as f:return pickle.load(f)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def array(a):return np.asarray(a.to_numpy() if hasattr(a,'to_numpy') else a,float)


class Checks:
    def __init__(self):self.count=0;self.failed=[];self.maximum={}
    def yes(self,name,value,record=None):
        self.count+=1
        if not bool(value):self.failed.append(dict(check=name,record=record))
    def near(self,name,a,b,tol=1e-10,record=None):
        aa,bb=np.asarray(a),np.asarray(b)
        error=float(np.max(np.abs(aa-bb),initial=0.)) if aa.shape==bb.shape else float('inf')
        if math.isfinite(error):self.maximum[name]=max(self.maximum.get(name,0.),error)
        self.yes(name,math.isfinite(error) and error<=tol,record)


def cp(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def plan_data(plan,labels,z,check,record):
    resources=plan['resources'];positions={r['resource_id']:i for i,r in enumerate(resources)}
    modules=plan['modules'];a=np.asarray(plan['amplitudes']);c=np.asarray(plan['prices']);lam=np.asarray(plan['elasticities']);D=np.asarray(plan['response_factors'])
    check.near('mean_lambda_price',a,lam*c,1e-10,record)
    check.yes('uncapped_DTO',plan.get('budget','missing') is None and plan.get('cap_enforced') is False,record)
    check.yes('contract_domain',np.isfinite(np.r_[a,c,lam,D.ravel()]).all() and np.all(a>=0) and np.all((c>=0)&(c<=2+1e-10)) and np.all(np.abs(D).sum(axis=1)<=1),record)
    profiles=np.zeros((len(modules),len(resources),96));hours=[]
    for j,m in enumerate(modules):
        check.yes('separate_resource_module',len(m['profiles'])==1,record)
        for rid,h in m['profiles'].items():profiles[j,positions[rid]]=np.asarray(h)
        hours.append(.25*np.maximum(-profiles[j],0.).sum())
        check.near('module_zero_energy',.25*profiles[j].sum(axis=1),np.zeros(len(resources)),1e-7,record)
    hours=np.array(hours);mean=np.einsum('j,jrt->rt',a,profiles)
    actual_amplitude=a*(1+D@np.asarray(z));actual=np.einsum('j,jrt->rt',actual_amplitude,profiles)
    expected=float((hours*c)@a);hard=float((hours*lam*(1+np.abs(D).sum(axis=1))*c*c).sum());fee=float((hours*c)@actual_amplitude)
    check.near('declared_expected_fee',expected,plan['expected_payment'],1e-9,record)
    check.near('declared_hard_fee',hard,plan['hard_payment_bound'],1e-9,record)
    check.yes('realized_fee_bound',-1e-12<=fee<=hard+1e-9,record)
    corner_fees=[];max_violation=0.
    for zz in [np.asarray(z),np.zeros(2),(-1,-1),(-1,1),(1,-1),(1,1)]:
        amp=a*(1+D@np.asarray(zz));u=np.einsum('j,jrt->rt',amp,profiles)
        violation=max(float(np.max(-amp,initial=0.)),float(np.max(amp-np.array([m['amplitude_upper_kw'] for m in modules]),initial=0.)))
        for i,r in enumerate(resources):
            base=np.broadcast_to(np.asarray(r['baseline_kw']),(96,));power=base+u[i];prefix=.25*np.cumsum(u[i])
            lo=np.broadcast_to(np.asarray(r.get('power_lower_kw',0.)),(96,));hi=np.broadcast_to(np.asarray(r['power_upper_kw']),(96,))
            elo=np.broadcast_to(np.asarray(r['prefix_delta_lower_kwh']),(96,));ehi=np.broadcast_to(np.asarray(r['prefix_delta_upper_kwh']),(96,))
            available=np.broadcast_to(np.asarray(r.get('available',True),bool),(96,))
            violation=max(violation,float(np.max(lo-power,initial=0.)),float(np.max(power-hi,initial=0.)),float(np.max(elo-prefix,initial=0.)),float(np.max(prefix-ehi,initial=0.)),float(np.max(np.abs(u[i][~available]),initial=0.)),abs(float(prefix[-1])))
            check.yes('original_nonbattery_resource',r.get('battery') is None,record)
        max_violation=max(max_violation,violation);corner_fees.append(float((hours*c)@amp))
    check.yes('mean_actual_four_corner_resources',max_violation<=1e-7,record)
    check.yes('all_support_fees_within_H',max(corner_fees)<=hard+1e-9,record)
    dp=np.zeros((96,len(labels)));dq=np.zeros_like(dp)
    for i,r in enumerate(resources):
        column=labels.index(r['node_label']);dp[:,column]+=actual[i];dq[:,column]+=np.broadcast_to(np.asarray(r.get('reactive_ratio',.35)),(96,))*actual[i]
    return dict(mean=mean,actual=actual,dp=dp,dq=dq,expected=expected,hard=hard,fee=fee,prices=c,
                max_resource_violation=max_violation,zero=not np.any(actual!=0.))


def direct_selection(pool,losses,weight,tau):
    order=pool['selection_order'];models={m['candidate_id']:m for m in pool['models']};index={k:i for i,k in enumerate(order)}
    h={k:models[k]['historical_validation_mae'] for k in order};hscope=[k for k in order if h[k]<=1.03*min(h.values())+1e-12]
    nh=pool['validation_rows'];nd=96;means={k:(nh*h[k]+weight*nd*losses[k])/(nh+weight*nd) for k in order}
    best=min(hscope,key=lambda k:(means[k],index[k]));threshold=means[best]+(.03*nh*h[best]+tau*weight*nd*losses[best])/(nh+weight*nd)+1e-12
    eligible=[k for k in hscope if means[k]<=threshold]
    chosen=min(eligible,key=lambda k:(len(models[k]['point']['supports']),means[k],index[k]))
    return chosen,eligible,means,threshold


def physical_algebra(truth,labels,check,record):
    ac=truth['ac'];V=ac['V_bus_mag'];nodes=list(V.columns);positions={node:i for i,node in enumerate(nodes)}
    T=len(V);localp=np.zeros((T,len(nodes)));localq=np.zeros_like(localp)
    for i,label in enumerate(labels):localp[:,positions[label]]=truth['P_pu'][:,i];localq[:,positions[label]]=truth['Q_pu'][:,i]
    inp=np.zeros_like(localp);inq=np.zeros_like(localp);outp=np.zeros_like(localp);outq=np.zeros_like(localp)
    maximum=0.;loss_error=0.;current_error=0.
    for edge in ac['metadata']['branch_edges']:
        name=edge['label'];u=edge['from_bus'];v=edge['to_bus'];iu,iv=positions[u],positions[v]
        pf=array(ac['branch_p_from_pu'][name]);qf=array(ac['branch_q_from_pu'][name]);pt=array(ac['branch_p_to_pu'][name]);qt=array(ac['branch_q_to_pu'][name])
        inp[:,iv]+=pt;inq[:,iv]+=qt;outp[:,iu]+=pf;outq[:,iu]+=qf
        loss_error=max(loss_error,float(np.max(np.abs(pf-pt-array(ac['branch_loss_p_pu'][name])))),float(np.max(np.abs(qf-qt-array(ac['branch_loss_q_pu'][name])))))
        ell=array(ac['I_branch_mag'][name])**2
        current_error=max(current_error,float(np.max(np.abs(pf*pf+qf*qf-array(V[u])**2*ell))),float(np.max(np.abs(pt*pt+qt*qt-array(V[v])**2*ell))))
    root=ac['metadata']['root_bus'];nonroot=[positions[n] for n in nodes if n!=root]
    p_error=float(np.max(np.abs((inp-outp-localp)[:,nonroot])));q_error=float(np.max(np.abs((inq-outq-localq)[:,nonroot])))
    check.yes('independent_nodal_balance',max(p_error,q_error,loss_error,current_error)<=1e-8,record)
    check.yes('all96_AC_converged',T==96 and np.all(np.asarray(ac['converged'],bool)),record)
    check.yes('physical_voltage_limits',np.isfinite(array(V)).all() and np.all((array(V)>=.9)&(array(V)<=1.1)),record)
    check.near('stored_terminal_voltage',truth['V_pu'],array(V.loc[:,labels]),1e-12,record)
    return dict(nodal_p_error=p_error,nodal_q_error=q_error,branch_loss_identity_error=loss_error,
        current_power_identity_error=current_error,thermal_limits_checked=False)


def run(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('retain existing audit output')
    # These gates precede every truth/physical packet read.
    ended=read(root/'ended.json');checked(ended['summary']);checked(ended['scored'])
    if ended['status']!='complete':raise ValueError('wait for final scoring')
    scored=read(root/'scored.json');pred_path=checked(scored['predictions_frozen']);predictions=read(pred_path)
    if predictions.get('status')!='frozen' or predictions.get('truth_read') is not False:raise ValueError('pre-truth predictive freeze required')
    observations=read(checked(predictions['observations']));freeze=read(checked(observations['freeze']));checked(freeze['protocol']);checked(freeze['source_snapshot']);checked(freeze['preflight'])
    p=read(root/'protocol.json');summary=read(ended['summary']['path']);pool=read(checked(p['pool']));labels=pool['labels'];models={m['candidate_id']:m for m in pool['models']}
    planning_problem=unpickle(p['inputs']['problem'])['problem']
    output.mkdir(parents=True);write(output/'protocol.json',dict(status='frozen',created_utc=now(),script=ref(__file__),
        study_protocol=ref(root/'protocol.json'),score=ref(root/'scored.json'),predictions=ref(pred_path),ended=ref(root/'ended.json'),
        scope='post-score read-only physical/statistical audit; no new generation, AC, fit or optimization'))
    check=Checks();requests={q['record_id']:q for q in freeze['requests']};rows={r['record_id']:r for r in scored['rows']};pre={r['record_id']:r for r in predictions['rows']}
    check.yes('all60_records_retained',len(requests)==len(rows)==len(pre)==60 and set(requests)==set(rows)==set(pre))
    check.yes('full_history',pool['train_rows']==pool['validation_rows']==288)
    with zipfile.ZipFile(checked(freeze['source_snapshot'])) as archive:
        for i,(name,digest) in enumerate(p['source_sha256'].items()):
            check.yes('frozen_source_bytes',hashlib.sha256(archive.read(f'{i:03d}_{Path(name).name}')).hexdigest()==digest)
    identity=[];trials={r['day']:r for r in p['trials']}
    for day in p['trials']:
        identity.append((p['case'],day['profile_index'],day['replicate']))
        expected=np.random.default_rng(np.random.SeedSequence([2026112202,day['day']])).uniform(-1,1,2)
        check.near('prelocked_response_seed',day['factors'],expected,0.)
    check.yes('twelve_distinct_simulated_days',len(identity)==len(set(identity))==12)
    all_data={};audit_rows=[];fees=[]
    for record,q in requests.items():
        row=rows[record];request=read(checked(q['request']));runtime=read(checked(request['protocol']));plan=read(checked(request['plan']))
        day=trials[q['day']];check.yes('physical_identity',(request['case'],request['profile_index'],request['replicate'])==(p['case'],day['profile_index'],day['replicate']),record)
        check.near('paired_latent_response',request['factors'],day['factors'],0.,record)
        check.yes('request_source_binding',request['source_sha256']==p['source_sha256']==runtime['source_sha256'],record)
        check.yes('uncapped_request',request['payment_mode']==runtime['payment_mode']=='frozen_plan_bound' and request['optimizer_payment_cap'] is runtime['optimizer_payment_cap'] is None,record)
        pd=plan_data(plan,labels,request['factors'],check,record)
        check.near('request_H',request['payment_bound'],pd['hard'],1e-9,record);check.near('runtime_H',runtime['payment_bound'],pd['hard'],1e-9,record)
        result=dict(record_id=record,day=q['day'],policy=q['policy'],status=row['status'],full_errors=None,attained=False,
                    model_fee=pd['fee'],expected_fee=pd['expected'],payment_bound=pd['hard'])
        receipt=None
        if row.get('receipt'):
            receipt=read(checked(row['receipt']))
            check.yes('worker_request_hash',receipt['request_sha256_expected']==q['request']['sha256'],record)
            if 'source_sha256' in receipt:check.yes('worker_source_binding',receipt['source_sha256']==p['source_sha256'],record)
            else:check.yes('preaudit_failure_is_unverified',row['status']!='complete',record)
            if receipt.get('actual_payment') is not None:
                for key,value in [('actual_payment',pd['fee']),('expected_payment',pd['expected']),('worst_payment',pd['hard'])]:check.near('worker_'+key,receipt[key],value,1e-9,record)
            else:check.yes('unknown_fee_retained',receipt.get('unknown_payment_events',1)>0,record)
        fees.append(dict(record_id=record,policy=q['policy'],status=row['status'],known_contract_fee=receipt.get('known_actual_payment',0.) if receipt else 0.,
            unknown_payment_events=receipt.get('unknown_payment_events',1) if receipt else 1,
            delivered_fee=receipt['actual_payment'] if row['status']=='complete' and receipt else 0.))
        if row['status']=='complete':
            check.yes('prediction_existed_before_truth',pre[record]['status']=='predictions_frozen',record)
            packet=unpickle(row['packet']);scene=packet['validation'][0]
            folder=Path(row['physical_folder']);private_ref=dict(path=str(folder/'physical_truth.pkl'),sha256=receipt['private_files']['physical_truth.pkl']['sha256'])
            truth=unpickle(private_ref);truth_labels=read(checked(dict(path=str(folder/'truth_evaluation.json'),sha256=receipt['private_files']['truth_evaluation.json']['sha256'])))
            check.yes('observed_only_packet',len(packet['validation'])==1 and packet['metadata']['training_rows']==0 and packet['metadata']['validation_rows']==96,record)
            check.near('executed_delta_kw',truth['delta_kw'],pd['dp'],1e-10,record);check.near('executed_delta_kvar',truth['delta_kvar'],pd['dq'],1e-10,record)
            check.near('one_injection_P',(truth['P_pu']-truth['natural_P_pu'])*p['declared']['base_kw'],pd['dp'],1e-9,record)
            check.near('one_injection_Q',(truth['Q_pu']-truth['natural_Q_pu'])*p['declared']['base_kw'],pd['dq'],1e-9,record)
            check.yes('operation_receipt',receipt['all_operation_checks_pass'] and receipt['checks']['ac_all_96_converged'] and receipt['checks']['electrical_equations_pass'] and receipt['checks']['voltage_limits_pass'],record)
            result['physical_algebra']=physical_algebra(truth,labels,check,record)
            pp=array(scene['P_terminal']);qq=array(scene['Q_terminal']);vv=array(scene['V_terminal']);rootv=array(scene['V_root_observed']);yy=rootv[:,None]**2-vv**2
            losses={k:float(np.abs(yy-pp@np.asarray(m['R']).T-qq@np.asarray(m['X']).T).mean()) for k,m in models.items()}
            M=.6*(models['coarse']['historical_validation_mae']-1.03*models['refined']['historical_validation_mae'])+.4*(losses['coarse']-1.02*losses['refined'])-1e-12
            distance=float(np.abs(pp@(np.asarray(models['coarse']['R'])-np.asarray(models['refined']['R'])).T+qq@(np.asarray(models['coarse']['X'])-np.asarray(models['refined']['X'])).T).mean())
            check.near('direct_M',M,row['M'],1e-13,record);check.near('direct_prediction_distance',distance,row['distance'],1e-13,record)
            saved=read(checked(row['score']));truth_set={tuple(sorted(x)) for x in truth_labels['true_supports']}
            for rule in saved['rules']:
                chosen,eligible,means,threshold=direct_selection(pool,losses,rule['new_day_weight'],rule['new_day_fraction'])
                check.yes('independent_nine_rule_selection',chosen==rule['selected_candidate_id'] and eligible==rule['eligible_candidate_ids'],record)
                check.near('independent_nine_rule_threshold',threshold,rule['eligibility_threshold_mean'],1e-13,record)
            chosen,eligible,_,_=direct_selection(pool,losses,2,.02)
            output_set={tuple(sorted(x)) for x in models[chosen]['frozen_output_support_labels']}
            raw_set={tuple(sorted(labels[i] for i in block)) for block in models[chosen]['point']['supports']}
            errors=len(output_set^truth_set);check.yes('final_complete_wrong_missing',errors==row['full_errors'] and chosen==row['selected_candidate'],record)
            check.yes('coarse_exit',('coarse' not in eligible)==row['coarse_excluded'],record)
            result.update(M=M,distance=distance,full_errors=errors,raw_errors=len(raw_set^truth_set),selected=chosen,
                coarse_excluded='coarse' not in eligible,delivered_fee=receipt['actual_payment'])
            pq=request['pq_sigma'];vs=request['voltage_sigma']
            standardized=dict(P=(pp-truth['P_pu'])/(pq*np.maximum(np.abs(truth['P_pu']),1e-12)),
                Q=(qq-truth['Q_pu'])/(pq*np.maximum(np.abs(truth['Q_pu']),1e-12)),
                V=(vv-truth['V_pu'])/(vs*np.maximum(np.abs(truth['V_pu']),1e-12)),
                root=(rootv-truth['root_pu'])/(vs*np.maximum(np.abs(truth['root_pu']),1e-12)))
            all_data[record]=dict(naturalP=truth['natural_P_pu'],naturalQ=truth['natural_Q_pu'],root=truth['root_pu'],
                observed_root=rootv,standardized=standardized,Pobs=pp,Qobs=qq,Yobs=yy,
                Ptrue=truth['P_pu'],Qtrue=truth['Q_pu'],Ytrue=truth['root_pu'][:,None]**2-truth['V_pu']**2,
                nominal_dp=pd['dp']/p['declared']['base_kw'],nominal_dq=pd['dq']/p['declared']['base_kw'])
        else:
            check.yes('failure_kept_unverified',not row.get('attained',False) and 'full_errors' not in row,record)
        audit_rows.append(result)
    keyed={(r['day'],r['policy']):r for r in audit_rows};original={(r['day'],r['policy']):r for r in scored['rows']}
    for day in range(12):
        zero=keyed[day,'zero'];zero_id=zero['record_id']
        for policy in p['policies']:
            r=keyed[day,policy]
            if r['record_id'] in all_data and zero_id in all_data:
                a=all_data[r['record_id']];b=all_data[zero_id]
                for key in ('naturalP','naturalQ','root','observed_root'):check.near('CRN_'+key,a[key],b[key],0.,r['record_id'])
                for key in ('P','Q','V','root'):check.near('CRN_standardized_'+key,a['standardized'][key],b['standardized'][key],1e-7,r['record_id'])
                target=.2*max(-zero['M'],0.)+1e-10;r.update(target_M=target,delta_M=r['M']-zero['M'],
                    attained=r['M']>=target-1e-13,paired_error_reduction=zero['full_errors']-r['full_errors'])
                check.yes('120percent_event',r['attained']==original[day,policy]['attained'],r['record_id'])
                check.near('120percent_target',target,original[day,policy]['target_M'],1e-13,r['record_id'])
                rf=np.asarray(models['refined']['R']);xf=np.asarray(models['refined']['X']);rb=np.asarray(models['coarse']['R']);xb=np.asarray(models['coarse']['X'])
                hpart=.6*(models['coarse']['historical_validation_mae']-1.03*models['refined']['historical_validation_mae'])
                def margin(y,pp,qq):
                    return hpart+.4*(np.abs(y-pp@rb.T-qq@xb.T).mean()-1.02*np.abs(y-pp@rf.T-qq@xf.T).mean())-1e-12
                dpobs=a['Pobs']-b['Pobs'];dqobs=a['Qobs']-b['Qobs']
                ytransport=b['Yobs']+dpobs@rf.T+dqobs@xf.T
                mt=margin(ytransport,a['Pobs'],a['Qobs'])
                ndp,ndq=a['nominal_dp'],a['nominal_dq']
                nominal_transport=b['Yobs']+ndp@rf.T+ndq@xf.T
                mtn=margin(nominal_transport,b['Pobs']+ndp,b['Qobs']+ndq)
                discrepancy=a['Yobs']-ytransport
                physical_discrepancy=(a['Ytrue']-b['Ytrue'])-(a['Ptrue']-b['Ptrue'])@rf.T-(a['Qtrue']-b['Qtrue'])@xf.T
                contrast=ndp@(rf-rb).T+ndq@(xf-xb).T
                training_delta=[float(.4*(np.abs(np.asarray(t['base_bad_micro'])/1e6+contrast).mean()-np.abs(np.asarray(t['base_bad_micro'])/1e6).mean())) for t in planning_problem['training']]
                r['transport_diagnostic']=dict(delta_M_actual_micro=r['delta_M']*1e6,
                    delta_M_observed_transport_micro=(mt-zero['M'])*1e6,
                    delta_M_nominal_transport_micro=(mtn-zero['M'])*1e6,
                    mean_training_template_predicted_delta_M_micro=float(np.mean(training_delta))*1e6,
                    training_template_delta_M_micro=[x*1e6 for x in training_delta],
                    actual_minus_observed_transport_micro=(r['M']-mt)*1e6,
                    observed_increment_discrepancy_mean_micro=float(discrepancy.mean())*1e6,
                    observed_increment_discrepancy_mae_micro=float(np.abs(discrepancy).mean())*1e6,
                    observed_increment_discrepancy_rms_micro=float(np.sqrt(np.mean(discrepancy**2)))*1e6,
                    physical_increment_model_discrepancy_mae_micro=float(np.abs(physical_discrepancy).mean())*1e6,
                    physical_increment_model_discrepancy_rms_micro=float(np.sqrt(np.mean(physical_discrepancy**2)))*1e6,
                    measurement_increment_discrepancy_rms_micro=float(np.sqrt(np.mean((discrepancy-physical_discrepancy)**2)))*1e6,
                    scope='post-score mechanism only: observed transport isolates reference-residual change; nominal transport versus3old training blocks isolates residual-shape migration; no price redesign')
        for tier in ('low','high'):
            a=keyed[day,'optimized_'+tier];b=keyed[day,'rotation_'+tier]
            check.near('matched_expected_fee_'+tier,a['expected_fee'],b['expected_fee'],1e-9)
            check.near('matched_payment_bound_'+tier,a['payment_bound'],b['payment_bound'],1e-9)
    tables=[];pairs=[];mechanisms=[]
    for policy in p['policies']:
        group=[r for r in audit_rows if r['policy']==policy];valid=[r for r in group if r['full_errors'] is not None];k=sum(r['attained'] for r in group)
        charges=[r for r in fees if r['policy']==policy]
        table=dict(policy=policy,planned_days=12,complete=len(valid),unresolved=12-len(valid),attained=k,
            conservative_unverified_attainment=0,attainment_ci95=cp(k,12),
            full_recoveries=sum(r['full_errors']==0 for r in valid),known_full_errors=sum(r['full_errors'] for r in valid),
            known_contract_fee=sum(r['known_contract_fee'] for r in charges),unknown_payment_events=sum(r['unknown_payment_events'] for r in charges),
            delivered_fee=sum(r['delivered_fee'] for r in charges),failed_or_unverified_known_fee=sum(r['known_contract_fee']-r['delivered_fee'] for r in charges),
            mean_delta_M_micro=float(np.mean([r['delta_M'] for r in valid]))*1e6 if valid and all('delta_M' in r for r in valid) else None,
            new_attainment_vs_zero=sum(r['attained'] and not keyed[r['day'],'zero']['attained'] for r in group),
            lost_attainment_vs_zero=sum(not r['attained'] and keyed[r['day'],'zero']['attained'] for r in group))
        saved=next(t for t in summary['tables'] if t['policy']==policy)
        check.yes('table_counts',table['complete']==saved['complete'] and k==saved['attained'] and table['full_recoveries']==saved['recoveries'] and table['known_full_errors']==saved['known_full_errors'])
        check.near('CP_interval',table['attainment_ci95'],saved['attainment_ci95'],1e-13)
        tables.append(table)
        diag=[r['transport_diagnostic'] for r in valid if 'transport_diagnostic' in r]
        mechanisms.append(dict(policy=policy,paired_days=len(diag),
            means={key:float(np.mean([r[key] for r in diag])) for key in (
                'delta_M_actual_micro','delta_M_observed_transport_micro','delta_M_nominal_transport_micro',
                'mean_training_template_predicted_delta_M_micro','actual_minus_observed_transport_micro',
                'observed_increment_discrepancy_mean_micro','observed_increment_discrepancy_mae_micro',
                'physical_increment_model_discrepancy_mae_micro')} if diag else {},
            pooled_rms={key:float(np.sqrt(np.mean([r[key]**2 for r in diag]))) for key in (
                'observed_increment_discrepancy_rms_micro','physical_increment_model_discrepancy_rms_micro',
                'measurement_increment_discrepancy_rms_micro')} if diag else {}))
    for tier in ('low','high'):
        item=dict(tier=tier,improved=0,worsened=0,tied=0,unresolved=0,error_reduction_known=0,
            optimized_realized_fee=0.,rotation_realized_fee=0.,gained_attainment=0,lost_attainment=0)
        for day in range(12):
            a=keyed[day,'optimized_'+tier];b=keyed[day,'rotation_'+tier]
            if a['full_errors'] is None or b['full_errors'] is None:item['unresolved']+=1;continue
            delta=b['full_errors']-a['full_errors'];item['error_reduction_known']+=delta
            item['improved' if delta>0 else 'worsened' if delta<0 else 'tied']+=1
            item['optimized_realized_fee']+=a['delivered_fee'];item['rotation_realized_fee']+=b['delivered_fee']
            item['gained_attainment']+=int(a['attained'] and not b['attained']);item['lost_attainment']+=int(not a['attained'] and b['attained'])
        saved=next(x for x in summary['pairs'] if x['tier']==tier)
        check.yes('paired_same_fee_error_comparison',all(item[k]==saved[k] for k in ('improved','worsened','tied','unresolved','error_reduction_known')))
        pairs.append(item)
    k=next(t['attained'] for t in tables if t['policy']=='optimized_low');pvalue=float(binom.cdf(k,12,.9))
    check.near('primary_binomial_pvalue',pvalue,summary['primary_one_sided_90pct_test']['pvalue'],1e-13)
    receipt=dict(status='passed' if not check.failed else 'audit_failed',created_utc=now(),checks=check.count,
        passed=check.count-len(check.failed),failures=check.failed,maximum_errors=check.maximum,
        tables=tables,pairs=pairs,mechanism_diagnostics=mechanisms,records=audit_rows,independent_simulated_days=12,planned_records=60,
        completed_records=sum(r['full_errors'] is not None for r in audit_rows),budget=None,optimizer_payment_cap=None,
        primary_test=dict(successes=k,denominator=12,pvalue=pvalue,reject_at_05=pvalue<.05),
        no_new_generation=True,new_AC=0,new_fits=0,new_optimizations=0,new_W_search=0,
        physical_scope='branch-flow nodal/current/loss identities and96AC convergence from saved outputs; impedance-specific checker retained by frozen source/receipt; no branch ampacity ratings',
        inference_scope='twelve prelocked independent synthetic day identities, five paired policies; not sixty independent trials or field evidence',
        selection_scope='historical two-model RNJ+WZZT pool and output mapping fixed; no new W acceptance/search or parameter refits',
        phase_gate='final scored/ended and all predictions frozen before audit truth access')
    write(output/'receipt.json',receipt)
    public={k:v for k,v in receipt.items() if k not in ('records','maximum_errors','failures')}
    public['failed_audit_checks']=len(check.failed);public['protocol_sha256']=sha(output/'protocol.json')
    write(output/'summary_public_candidate.json',public)
    write(output/'ended.json',dict(status=receipt['status'],receipt=ref(output/'receipt.json'),public_summary=ref(output/'summary_public_candidate.json')))
    print(json.dumps(dict(status=receipt['status'],checks=check.count,failed=len(check.failed),tables=tables,pairs=pairs,
        public_summary=ref(output/'summary_public_candidate.json')),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.run_root,args.output)
