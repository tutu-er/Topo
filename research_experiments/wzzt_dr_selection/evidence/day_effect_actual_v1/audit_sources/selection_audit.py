"""Independent frozen-model selection/statistics audit after score completion.

No optimizer, fitter, W search, AC, physical trajectories or simulator is run.
All observed-only predictions are computed before any truth file is opened.
"""
from pathlib import Path
import hashlib,json,pickle,math
from datetime import datetime,timezone
import numpy as np
from scipy.stats import beta,binom

STAGE=Path(__file__).resolve().parents[1];RUN=STAGE/'actual_v1'
POLICIES=['zero','rotation_low','optimized_low','rotation_high','optimized_high']
CHECKS=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(name,ok,detail=None):CHECKS.append(dict(name=name,passed=bool(ok),detail=detail))
def close(name,a,b,tol=2e-12):
    a=np.asarray(a,float);b=np.asarray(b,float)
    error=float(np.max(np.abs(a-b),initial=0.)) if a.shape==b.shape else math.inf
    ck(name,error<=tol,dict(max_abs_error=error,tolerance=tol))
def reference(ref,name):
    p=Path(ref['path']);ck('digest '+name,sha(p)==ref['sha256']);return p
def unpickle(ref,name):
    with reference(ref,name).open('rb') as stream:return pickle.load(stream)
def family(supports):return {tuple(sorted(int(x) for x in s)) for s in supports}
def observe_scene(scene,labels):
    P=np.asarray(scene['P_terminal'],float);Q=np.asarray(scene['Q_terminal'],float)
    V=np.asarray(scene['V_terminal'],float);V0=np.asarray(scene['V_root_observed'],float).reshape(-1)
    ck('observed full day aligned',P.shape==Q.shape==V.shape==(96,len(labels)) and V0.shape==(96,))
    for field in ['P_terminal','Q_terminal','V_terminal']:
        if hasattr(scene[field],'columns'):ck('column order '+field,list(scene[field].columns)==labels)
    y=V0[:,None]**2-V**2
    if 'drop_target' in scene:close('observed target consistency',y,scene['drop_target'])
    return P,Q,y
def predict(model,scene,labels):
    P,Q,y=observe_scene(scene,labels);mu=P@np.asarray(model['R']).T+Q@np.asarray(model['X']).T
    return mu,y,float(np.abs(y-mu).sum())
def cp(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def choose(order,models,means,threshold,scope):
    eligible=[key for key in scope if means[key]<=threshold]
    selected=min(eligible,key=lambda key:(len(models[key]['point']['supports']),means[key],order.index(key)))
    return selected,eligible
def contract(model,calibration,labels):
    point=model['point'];keep=[]
    for s,r,x in zip(point['supports'],point['r'],point['x']):
        if len(s)==1 or calibration['a']*r+calibration['b']*x+1e-10>=.16*calibration['median_root_depth']:keep.append(s)
    close('frozen contract count '+model['candidate_id'],len(keep),model['frozen_output_support_count'],0.)
    ck('fixed .16 output map '+model['candidate_id'],keep==model['frozen_output_supports'])
    labelled=[[labels[k] for k in s] for s in keep]
    ck('fixed .16 label map '+model['candidate_id'],labelled==model['frozen_output_support_labels'])
    return labelled

def main():
    if not (RUN/'ended.json').exists():raise RuntimeError('Refuse actual/truth audit before score-ended receipt exists')
    protocol=read(RUN/'protocol.json');freeze=read(RUN/'freeze.json');ended=read(RUN/'ended.json')
    ck('original protocol digest',sha(RUN/'protocol.json')=='f1e0f5ac0e798481888600c3f37b57aa001efd0a83dce7a5169df6b932659ee7')
    reference(freeze['protocol'],'protocol');reference(ended['summary'],'summary');reference(ended['scored'],'scored')
    observations=read(RUN/'observations_frozen.json');predictive=read(RUN/'predictions_frozen.json')
    reference(observations['freeze'],'observation freeze');reference(predictive['observations'],'predictive observations')
    ck('controller predictions froze before truth',predictive['truth_read'] is False and observations['controller_truth_read'] is False)
    ck('full denominator ledger',observations['planned']==60 and len(observations['rows'])==60)
    pool=read(reference(protocol['pool'],'historical pool'));labels=pool['labels'];order=pool['selection_order'];models={m['candidate_id']:m for m in pool['models']}
    ck('fixed historical sizes',pool['train_rows']==pool['validation_rows']==288 and pool['historical_validation_cells']==288*15)
    packet=unpickle(pool['history_packet'],'old history');history=packet.get('history',packet)
    historical={}
    for key,model in models.items():
        rr=np.zeros((15,15));xx=np.zeros_like(rr)
        for block,r,x in zip(model['point']['supports'],model['point']['r'],model['point']['x']):
            rr[np.ix_(block,block)]+=r;xx[np.ix_(block,block)]+=x
        close('unchanged shared R '+key,rr,model['R']);close('unchanged shared X '+key,xx,model['X'])
        contract(model,pool['calibration'],labels)
        hs=0.
        for scene in history['validation']:
            # Historical data are three complete96-slot days in the same order.
            hs+=predict(model,scene,labels)[2]
        historical[key]=hs;close('complete historical absolute loss '+key,hs,model['historical_validation_abs_sum'])
    hist_cells=pool['historical_validation_cells'];new_cells=96*15
    hmae={key:historical[key]/hist_cells for key in order};hmin=min(hmae.values())
    scope=[key for key in order if hmae[key]<=1.03*hmin+1e-12]
    ours=[];predicted_by_record={x['record_id']:x for x in predictive['rows']};frozen_requests={x['record_id']:x for x in freeze['requests']}
    # Stage1: observations and frozen-model selection only. No truth reads.
    for original in observations['rows']:
        row=dict(day=original['day'],policy=original['policy'],record_id=original['record_id'],status=original['status'],attained=False)
        if original['status']!='observed':ours.append(row);continue
        request=read(reference(frozen_requests[row['record_id']]['request'],'request '+row['record_id']))
        scene_packet=unpickle(original['packet'],'observed packet '+row['record_id']);ck('new day never used for fitting',len(scene_packet.get('training',[]))==0 and len(scene_packet['validation'])==1)
        scene=scene_packet['validation'][0];mu={};dabs={}
        for key in order:mu[key],_,dabs[key]=predict(models[key],scene,labels)
        distance=float(np.abs(mu[order[0]]-mu[order[1]]).mean())
        frozen=predicted_by_record[row['record_id']];s=read(reference(frozen['score'],'saved score '+row['record_id']))
        close('observed prediction distance '+row['record_id'],distance,s['new_prediction_pair_distance_l1'])
        ck('no parameter update '+row['record_id'],s['parameters_changed'] is False and s['calibration_changed'] is False and s['new_fit_calls']==s['new_search_calls']==0)
        for key in order:
            loss=next(x for x in s['loss_rows'] if x['candidate_id']==key)
            close('observed new L1 '+row['record_id']+key,dabs[key],loss['new_validation_abs_sum'])
            close('historical L1 retained '+row['record_id']+key,historical[key],loss['historical_validation_abs_sum'])
        rules=[]
        for weight in (1,2,4):
            cells=hist_cells+weight*new_cells;means={key:(historical[key]+weight*dabs[key])/cells for key in order};ref=min(scope,key=lambda key:(means[key],order.index(key)))
            for tau in (.03,.02,.01):
                rid=f'w{weight}_tau{round(100*tau)}pct'
                threshold=1.03*means[ref]+1e-12 if tau==.03 else means[ref]+.03*historical[ref]/cells+tau*weight*dabs[ref]/cells+1e-12
                selected,eligible=choose(order,models,means,threshold,scope);saved=next(x for x in s['rules'] if x['rule_id']==rid)
                ck('selection '+row['record_id']+rid,selected==saved['selected_candidate_id'] and eligible==saved['eligible_candidate_ids'])
                close('validation threshold '+row['record_id']+rid,threshold,saved['eligibility_threshold_mean'])
                for key in order:close('weighted loss '+row['record_id']+rid+key,means[key],saved['weighted_validation_mae'][key])
                labels_out=contract(models[selected],pool['calibration'],labels)
                ck('selected output '+row['record_id']+rid,labels_out==saved['selected_output_support_labels'])
                rules.append(dict(rule_id=rid,selected=selected,eligible=eligible,output_labels=labels_out))
        plain={key:(historical[key]+dabs[key])/(hist_cells+new_cells) for key in order};best=min(plain.values());chosen,eligible=choose(order,models,plain,1.03*best+1e-12,order)
        ck('original ungated3 percent '+row['record_id'],chosen==s['original_ungated_selection']['selected_candidate_id'] and eligible==s['original_ungated_selection']['eligible_candidate_ids'])
        M=.6*(hmae['coarse']-1.03*hmae['refined'])+.4*(dabs['coarse']/new_cells-1.02*dabs['refined']/new_cells)-1e-12
        close('fixed-pair M '+row['record_id'],M,frozen['M'])
        receipt=read(reference(original['receipt'],'physical receipt '+row['record_id']));plan=read(reference(request['plan'],'executed plan '+row['record_id']))
        u=np.stack([a*np.asarray(next(iter(m['profiles'].values()))) for a,m in zip(plan['amplitudes'],plan['modules'])]);c=np.asarray(plan['prices']);eta=1+np.asarray(plan['response_factors'])@np.asarray(request['factors'])
        model_fee=float(c@(.25*np.maximum(-eta[:,None]*u,0).sum(axis=1)))
        close('delivered model fee '+row['record_id'],model_fee,receipt['actual_payment'],1e-10)
        ck('verified physical state '+row['record_id'],receipt['all_operation_checks_pass'] is True)
        main_rule=next(x for x in rules if x['rule_id']=='w2_tau2pct')
        row.update(status='predictions_complete',M=M,distance=distance,rules=rules,selected=main_rule['selected'],output_labels=main_rule['output_labels'],coarse_excluded='coarse' not in main_rule['eligible'],receipt=receipt,physical_folder=original['physical_folder'],delivered_fee=model_fee,observed_mae={key:dabs[key]/new_cells for key in order})
        ours.append(row)
    ck('all observed predictions before truth',all(x['status']=='predictions_complete' or x['status']!='observed' for x in ours))
    observed_only_count=sum(x['status']=='predictions_complete' for x in ours)
    # Stage2: evaluator topology labels, after all observed-only predictions.
    scored=read(RUN/'scored.json');saved_by_record={x['record_id']:x for x in scored['rows']}
    truth_families=[]
    for row in ours:
        if row['status']!='predictions_complete':continue
        receipt=row.pop('receipt');tf=Path(row.pop('physical_folder'))/'truth_evaluation.json'
        ck('evaluator truth digest '+row['record_id'],sha(tf)==receipt['private_files']['truth_evaluation.json']['sha256'])
        true=family(read(tf)['true_supports']);truth_families.append(true)
        error=len(family(row['output_labels'])^true);saved=saved_by_record[row['record_id']]
        ck('full wrong/missing '+row['record_id'],error==saved['full_errors'])
        ck('final selected identity '+row['record_id'],row['selected']==saved['selected_candidate'] and row['coarse_excluded']==saved['coarse_excluded'])
        for rule in row['rules']:
            err=len(family(rule['output_labels'])^true);old=next(x for x in saved['rule_outcomes'] if x['rule_id']==rule['rule_id'])
            ck('all-rule final errors '+row['record_id']+rule['rule_id'],rule['selected']==old['selected'] and err==old['full_errors'])
        candidate_errors={key:len(family(model['frozen_output_support_labels'])^true) for key,model in models.items()}
        for key,value in candidate_errors.items():ck('candidate truth score '+row['record_id']+key,value==saved['candidate_errors'][key]['full_errors'])
        row.update(status='complete',full_errors=error,recovered=error==0,candidate_errors=candidate_errors)
    if truth_families:ck('same evaluator tree for all policies/days',all(x==truth_families[0] for x in truth_families))
    for row in ours:
        zero=next(x for x in ours if x['day']==row['day'] and x['policy']=='zero')
        if row['status']==zero['status']=='complete':
            target=.2*max(-zero['M'],0.)+1e-10;attained=row['M']>=target-1e-13
            row.update(attained=attained,baseline_M=zero['M'],target_M=target,delta_M=row['M']-zero['M'],distance_gain=row['distance']-zero['distance'],paired_error_reduction=zero['full_errors']-row['full_errors'])
            saved=saved_by_record[row['record_id']]
            ck('actual excitation event '+row['record_id'],attained==saved['attained'])
            for key in ['target_M','baseline_M','delta_M','distance_gain','paired_error_reduction']:close('paired metric '+row['record_id']+key,row[key],saved[key])
    summary=read(RUN/'summary_private.json');tables=[]
    for policy in POLICIES:
        group=[r for r in ours if r['policy']==policy];valid=[r for r in group if r['status']=='complete'];k=sum(r['attained'] for r in group);saved=next(t for t in summary['tables'] if t['policy']==policy)
        ck('twelve-day denominator '+policy,len(group)==saved['planned']==12 and len(valid)==saved['complete'])
        ck('retained structural unknown '+policy,[r.get('full_errors') for r in group]==saved['full_errors'])
        close('exact CP interval '+policy,cp(k,12),saved['attainment_ci95'],1e-13)
        ck('excitation total '+policy,k==saved['attained'])
        ck('recoveries '+policy,sum(r['full_errors']==0 for r in valid)==saved['recoveries'])
        ck('known errors '+policy,sum(r['full_errors'] for r in valid)==saved['known_full_errors'])
        close('delivered fee total '+policy,sum(r['delivered_fee'] for r in valid),saved['delivered_payment'],1e-10)
        new=sum(r['attained'] and not next(z for z in ours if z['day']==r['day'] and z['policy']=='zero')['attained'] for r in group)
        lost=sum(not r['attained'] and next(z for z in ours if z['day']==r['day'] and z['policy']=='zero')['attained'] for r in group)
        ck('new and lost excitation '+policy,new==saved['new_attainment_vs_zero'] and lost==saved['lost_attainment_vs_zero'])
        for key in ['M','delta_M','distance_gain']:
            vals=[r[key] for r in valid if key in r];name={'M':'mean_M_micro','delta_M':'mean_delta_M_micro','distance_gain':'mean_distance_gain_micro'}[key]
            if vals:close('mean '+policy+key,np.mean(vals)*1e6,saved[name],2e-8)
        tables.append(dict(policy=policy,planned=12,complete=len(valid),unknown=12-len(valid),attained=k,attainment_ci95=cp(k,12),recovered=sum(r['full_errors']==0 for r in valid),recovery_ci95=cp(sum(r['full_errors']==0 for r in valid),12),known_full_errors=sum(r['full_errors'] for r in valid),delivered_fee=sum(r['delivered_fee'] for r in valid),new_attainment_vs_zero=new,lost_attainment_vs_zero=lost))
    pairs=[]
    for tier in ('low','high'):
        pair=dict(tier=tier,improved=0,worsened=0,tied=0,unresolved=0,error_reduction_known=0)
        for day in range(12):
            a=next(r for r in ours if r['day']==day and r['policy']=='optimized_'+tier);b=next(r for r in ours if r['day']==day and r['policy']=='rotation_'+tier)
            if a['status']!='complete' or b['status']!='complete':pair['unresolved']+=1;continue
            diff=b['full_errors']-a['full_errors'];pair['error_reduction_known']+=diff;pair['improved' if diff>0 else 'worsened' if diff<0 else 'tied']+=1
        saved=next(x for x in summary['pairs'] if x['tier']==tier);ck('paired structural result '+tier,pair==saved);pairs.append(pair)
    primary=next(x for x in tables if x['policy']=='optimized_low');pvalue=float(binom.cdf(primary['attained'],12,.9));test=summary['primary_one_sided_90pct_test']
    close('primary exact one-sided binomial p',pvalue,test['pvalue'],1e-13);ck('primary declared direction',test['successes']==primary['attained'] and test['denominator']==12 and test['reject_at_05']==(pvalue<.05))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(RUN/'protocol.json'),summary_sha256=sha(RUN/'summary_private.json'),audit_sha256=sha(__file__),checks=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=[x for x in CHECKS if not x['passed']],observed_only_records_completed_before_truth=observed_only_count,planned_records=60,unique_simulated_days=12,qualification=protocol['qualification'],tables=tables,pairs=pairs,primary_pvalue=pvalue,records=[{k:v for k,v in r.items() if k not in ['rules','output_labels']} for r in ours],new_AC=0,new_fits=0,new_optimizations=0,new_W_search=0,scope='independent arithmetic from measured P/Q/V; all fixed .16 maps and9 validation rules replayed; truth opened only after all observed-only selections; five policy records within a day are paired and do not increase the independent denominator; conditional simulated-law probabilities only')
    out=Path(__file__).parent/'actual_receipt.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['checks','passed','failed','unique_simulated_days','tables','pairs','primary_pvalue']},indent=2))
if __name__=='__main__':main()
