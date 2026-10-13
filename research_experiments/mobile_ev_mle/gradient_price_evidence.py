"""Compact export and independent replay of frozen gradient-price evidence.

Only the two explicitly named long singular-value diagnostics are removed.
Actual fitted states are replayed through AC and Gaussian meters; no optimizer
or study summarizer is called. All recorded scientific failures stay visible.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from .likelihood import Observations, RadialCandidate, ac_predict, concat_observations
from .simulation import model
from .single_site_milp import SingleSiteConstraints, SingleSitePlan, check_single_site_plan

HERE=Path(__file__).resolve().parent
DROP_KEYS=frozenset(('local_scaled_singular_values','local_projected_shared_singular_values'))


def read_json(path):
    path=Path(path);b=path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix=='.gz' else b)


def write_json(path,value,compress=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    path.write_bytes(gzip.compress(b,mtime=0) if compress else b)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_value(value):
    if isinstance(value,dict):return {key:compact_value(v) for key,v in value.items() if key not in DROP_KEYS}
    if isinstance(value,list):return [compact_value(v) for v in value]
    return value


def _close(actual,expected,label,errors,atol=1e-6,rtol=5e-8):
    try:
        a,b=np.asarray(actual,float),np.asarray(expected,float)
        good=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol)
    except (ValueError,TypeError):good=False
    if not good:errors.append(label)


def _observations(values):return Observations(**{key:None if value is None else np.asarray(value,float) for key,value in values.items()})


def _family(supports,n):
    return tuple(sorted({tuple(sorted(set(map(int,s)))) for s in supports if 1<len(set(s))<n}))


def replay_fit(row,observations,meter=None):
    """Recompute attained loss from saved physical states, not stored predictions."""
    errors=[];meter=meter or model(observations.n)
    name=row.get('name',row.get('candidate','unknown'))
    try:
        supports=[tuple(sorted(map(int,s))) for s in row['supports']]
        candidate=RadialCandidate.from_supports(name,observations.n,_family(supports,observations.n))
        if len(set(supports))!=candidate.m or set(supports)!=set(candidate.supports):
            raise ValueError('stored complete edge supports required for parameter ordering')
        permutation=[supports.index(s) for s in candidate.supports]
        r=np.asarray(row['r'],float)[permutation];x=np.asarray(row['x'],float)[permutation]
        p,q,sp,sq=(np.asarray(row[k],float) for k in ('p','q','source_p','source_q'))
        finite=all(np.isfinite(a).all() for a in (r,x,p,q,sp,sq))
        if not finite:
            expected_failure=not row['success'] and row.get('loss') is None and row.get('loglik') is None
            return dict(success=expected_failure,name=name,replayed=False,recorded_failed_fit=True,
                        errors=[] if expected_failure else ['nonfinite successful/finite-cost fit'],reason='nonfinite failed fit retained')
        parameter_domain=bool(np.all((r>=1e-6-1e-10)&(r<=.1+1e-10)) and np.all((x>=1e-6-1e-10)&(x<=.1+1e-10))
            and np.all(abs(p)<=1+1e-8) and np.all(abs(q)<=1+1e-8) and np.all(abs(sp)<=.5+1e-8) and np.all(abs(sq)<=.5+1e-8))
        if not parameter_domain:errors.append('parameter_domain')
        physical=ac_predict(candidate,r,x,p,q,sp,sq,observations.root_v)
        cost=normalizer=0.;channel_costs={}
        for channel in ('p','q','v','master_p','master_q','source_p','source_q'):
            observed=getattr(observations,channel)
            if observed is None:continue
            sigma=np.broadcast_to(np.asarray(getattr(meter,channel+'_sigma'),float),observed.shape)
            predicted=np.asarray(physical[channel],float)
            if channel in ('p','q','v'):
                bias=np.broadcast_to(np.asarray(getattr(meter,'bias_'+channel),float),(observations.n,))
                _close(row['bias_'+channel],bias,'bias_'+channel,errors,atol=1e-12,rtol=0)
                predicted=predicted+bias
            residual=(predicted-observed)/sigma
            component=float(np.sum(residual**2));cost+=component
            normalizer+=float(np.sum(np.log(2*np.pi*sigma**2)))
            channel_costs[channel]=component
            if channel in row.get('predicted',{}):
                _close(predicted,row['predicted'][channel],'prediction_'+channel,errors,atol=2e-10,rtol=1e-10)
        loglik=-.5*(cost+normalizer)
        _close(cost,row['loss'],'full_SSE',errors)
        _close(.5*cost,row['half_loss'],'half_SSE',errors)
        _close(loglik,row['loglik'],'loglik',errors)
        voltage=np.abs(physical['voltage']);voltage_domain=bool(voltage.min()>=.85-1e-8 and voltage.max()<=1.15+1e-8)
        if row['success'] and not voltage_domain:errors.append('successful_fit_voltage_domain')
        diagnostics=row.get('diagnostics',{})
        for key in ('ac_equation_residual','power_balance_residual'):
            if physical[key]>2e-10:errors.append(key)
        for key,actual in (('min_voltage',float(voltage.min())),('max_voltage',float(voltage.max()))):
            if key in diagnostics:_close(actual,diagnostics[key],key,errors,atol=2e-10,rtol=1e-10)
        if row.get('global_MLE_certified',False) or diagnostics.get('global_optimality_certified',False):errors.append('unsupported_global_MLE_claim')
        return dict(success=not errors,name=name,replayed=True,recorded_failed_fit=not row['success'],
            errors=errors,full_loss=cost,half_loss=.5*cost,loglik=loglik,channel_costs=channel_costs,
            parameter_domain=parameter_domain,voltage_domain=voltage_domain,
            ac_equation_residual=physical['ac_equation_residual'],power_balance_residual=physical['power_balance_residual'])
    except (ValueError,RuntimeError,KeyError,TypeError,IndexError,FloatingPointError,np.linalg.LinAlgError) as error:
        return dict(success=False,name=name,replayed=False,errors=['replay_exception:'+str(error)])


def _bounds_audit(value,path=''):
    errors=[];count=conflicts=0
    if isinstance(value,dict):
        if value.get('incumbent') is not None and value.get('best_bound') is not None:
            count+=1;gap=float(value['best_bound'])-float(value['incumbent'])
            if value.get('absolute_gap') is not None:_close(gap,value['absolute_gap'],path+'.gap',errors,atol=2e-6,rtol=1e-8)
            if gap < -2e-6:
                conflicts+=1
                if value.get('success') is True:errors.append(path+'.successful_negative_bound_gap')
        for key,item in value.items():
            e,c,f=_bounds_audit(item,path+'.'+key);errors+=e;count+=c;conflicts+=f
    elif isinstance(value,list):
        for i,item in enumerate(value):
            e,c,f=_bounds_audit(item,path+f'[{i}]');errors+=e;count+=c;conflicts+=f
    return errors,count,conflicts


def _policy_check(trial,expected):
    errors=[]
    if expected is None:return errors
    for key in ('config','source_sha256','policy_id'):
        if trial[key]!=expected[key]:errors.append('frozen_'+key)
    computed=hashlib.sha256(json.dumps(dict(config=trial['config'],source_sha256=trial['source_sha256']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if computed!=trial['policy_id']:errors.append('policy_id_digest')
    return errors


def audit_trial(trial,*,calibration=None,expected=None):
    """Replay history and all actual joint fits, then reconstruct trial statistics."""
    began=perf_counter();errors=_policy_check(trial,expected)
    if trial.get('selection_uses_truth') or trial.get('selection_uses_future_noise'):errors.append('selector_information_isolation')
    if 'seeds' in trial:
        streams=np.random.SeedSequence(int(trial['seed'])).spawn(3)
        expected_seeds={name:int(stream.generate_state(1)[0]) for name,stream in zip(('generator','policy','evaluation_noise'),streams)}
        if trial['seeds']!=expected_seeds:errors.append('independent_seed_streams')
    history=_observations(trial['history_observations']);meter=model(history.n)
    hfits=[replay_fit(row,history,meter) for row in trial['historical_fits']]
    for i,result in enumerate(hfits):errors.extend('history['+str(i)+']:'+e for e in result['errors'])
    constraints=SingleSiteConstraints(**trial['device_constraints']);method_audits=[]
    definitions={row['name']:row['supports'] for row in trial['historical_fits']}
    for bank in trial.get('proposal',{}).get('bank',[]):definitions.setdefault(bank['name'],bank['supports'])
    truth=_family(trial['truth']['supports'],history.n)
    actual_true=[name for name in trial['candidate_names'] if name in definitions and _family(definitions[name],history.n)==truth]
    coverage=bool(actual_true)
    if coverage!=trial['true_covered']:errors.append('candidate_coverage')
    if coverage and trial['true_name'] not in actual_true:errors.append('true_candidate_name')
    threshold=None if calibration is None else math.inf if calibration['relative']['vacuous'] else float(calibration['relative']['threshold'])
    all_joint_complete=True;true_gaps=[]
    for record in trial['records']:
        method=record['method'];local=[];replays=[];losses={}
        action=record.get('action')
        if action is not None:
            plan=SingleSitePlan(action['site'],np.asarray(action['p']),np.asarray(action['q']),np.asarray(action['sites']),np.asarray(action['soc']),action.get('solver',{}))
            if len(plan.p)!=trial['config']['future_points']:local.append('action_horizon')
            if method=='none':
                if np.any(plan.p!=0) or np.any(plan.q!=0):local.append('none_injection')
                operation=dict(feasible=not local,same_length_no_EV=True)
            else:
                operation=check_single_site_plan(plan,constraints)
                if record['status']!='operational_failure' and not operation['feasible']:local.append('operation:'+str(operation['issues']))
        else:operation=dict(feasible=False,recorded_failure=True)
        if 'observations' in record:
            future=_observations(record['observations'])
            if future.t!=trial['config']['future_points']:local.append('observation_horizon')
            joint=concat_observations(history,future)
            for row in record['fits']:
                result=replay_fit(row,joint,meter);replays.append(result)
                local.extend(row['name']+':'+e for e in result['errors'])
                if row['success'] and result.get('replayed'):losses[row['name']]=result['full_loss']
        elif record.get('fits'):local.append('fitted_states_without_observations')
        joint_complete=bool(record['status']=='complete' and set(losses)==set(trial['candidate_names']))
        if record['status']=='complete' and not joint_complete:local.append('declared_complete_missing_fit')
        all_joint_complete &= joint_complete
        for name,value in losses.items():_close(value,record['losses'].get(name),name+'.loss_table',local)
        gaps={name:value-min(losses.values()) for name,value in losses.items()} if joint_complete else {}
        if joint_complete:
            for name,value in gaps.items():_close(value,record['gaps'].get(name),name+'.gap',local)
            winner=min(losses,key=lambda name:(losses[name],name))
            if record.get('winner')!=winner:local.append('winner')
            if coverage:
                gap=gaps[trial['true_name']];true_gaps.append(gap)
                _close(gap,record.get('true_gap'),'true_gap',local)
        if threshold is not None:
            retained=[name for name in trial['candidate_names'] if gaps[name]<=threshold] if joint_complete else list(trial['candidate_names'])
            decisions=dict(retained_candidates=retained,false_rejection=bool(coverage and trial['true_name'] not in retained),
                only_truth_retained=bool(coverage and retained==[trial['true_name']]),unresolved=len(retained)>1,
                wrong_candidates_removed=sum(name!=trial['true_name'] and name not in retained for name in trial['candidate_names']) if coverage else None)
            for key,value in decisions.items():
                if record.get(key)!=value:local.append('decision_'+key)
        else:decisions={}
        errors.extend(method+':'+e for e in local)
        method_audits.append(dict(method=method,success=not local,errors=local,recorded_status=record['status'],
                                 operation=operation,fits=replays,losses=losses,joint_complete=joint_complete,decisions=decisions))
    history_complete=bool(not trial['historical_failures'] and len(trial['historical_fits'])==len(trial['candidate_names'])
                          and all(row['success'] and row.get('loss') is not None for row in trial['historical_fits']))
    complete=bool(history_complete and all_joint_complete and coverage)
    if complete!=trial['complete']:errors.append('trial_complete')
    score=max(true_gaps) if complete else math.inf
    if complete:_close(score,trial['relative_score'],'relative_score',errors)
    elif trial.get('relative_score') is not None and not math.isinf(float(trial['relative_score'])):errors.append('failed_calibration_score_not_infinity')
    if trial.get('calibration_score_is_infinite')!=(not complete):errors.append('calibration_infinity_flag')
    if calibration is not None:
        if trial['history_id'] in calibration['calibration_history_ids']:errors.append('calibration_evaluation_overlap')
        any_false=any(m['decisions']['false_rejection'] for m in method_audits)
        if trial.get('any_method_false_rejection')!=any_false:errors.append('any_method_false_rejection')
    bound_errors,count,conflicts=_bounds_audit(trial['selection']);errors+=bound_errors
    fits=hfits+[fit for method in method_audits for fit in method['fits']]
    return dict(success=not errors,errors=errors,history_id=trial['history_id'],policy_id=trial['policy_id'],
        recomputed_complete=complete,recomputed_relative_score=None if math.isinf(score) else score,
        calibration_score_is_infinite=math.isinf(score),candidate_coverage=coverage,
        recorded_failure_count=sum(not row['success'] for row in trial['historical_fits'])+
            sum(m['recorded_status']!='complete' for m in method_audits),
        actual_fit_replay_count=sum(f['replayed'] for f in fits),unreplayable_recorded_fit_count=sum(not f['replayed'] for f in fits),
        milp_bound_checks=count,recorded_bound_conflicts=conflicts,history_fits=hfits,methods=method_audits,seconds=perf_counter()-began)


def audit_calibration(audits,saved,alpha):
    """Order statistic reconstructed directly; failures retain infinity mass."""
    errors=[];n=len(audits);rank=math.ceil((n+1)*(1-alpha))
    scores=sorted(math.inf if a['calibration_score_is_infinite'] else a['recomputed_relative_score'] for a in audits)
    threshold=scores[rank-1] if rank<=n else math.inf
    ids=[a['history_id'] for a in audits]
    if len(set(ids))!=n:errors.append('duplicate_calibration_histories')
    if set(ids)!=set(saved['calibration_history_ids']):errors.append('calibration_history_ids')
    gate=saved['relative']
    if gate['rank']!=rank or gate['sample_count']!=n:errors.append('calibration_rank_or_count')
    if gate['vacuous']!=math.isinf(threshold):errors.append('calibration_vacuity')
    if math.isfinite(threshold):_close(threshold,gate['threshold'],'calibration_threshold',errors)
    failures=sum(a['calibration_score_is_infinite'] for a in audits)
    if failures!=saved['failed_history_count'] or n!=saved['all_history_count']:errors.append('calibration_failure_denominator')
    if saved['alpha']!=alpha:errors.append('calibration_alpha')
    if 'policy_id' in saved and {a['policy_id'] for a in audits}!={saved['policy_id']}:errors.append('calibration_policy_id')
    if 'rank_tail_bound' in gate:_close(max(0.,(n+1-rank)/(n+1)),gate['rank_tail_bound'],'rank_tail_bound',errors,atol=1e-14,rtol=0)
    return dict(success=not errors,errors=errors,all_history_count=n,failed_history_count=failures,rank=rank,
                threshold=None if math.isinf(threshold) else threshold,vacuous=math.isinf(threshold))


def audit_summary(audits,saved,decisions_available):
    """Check the original summary against independently reconstructed counts."""
    errors=[];n=len(audits)
    expected=dict(all_history_count=n,complete_history_count=sum(a['recomputed_complete'] for a in audits),
        failed_history_count=sum(not a['recomputed_complete'] for a in audits),
        candidate_coverage_failures=sum(not a['candidate_coverage'] for a in audits))
    for key,value in expected.items():
        if saved.get(key)!=value:errors.append('summary_'+key)
    declared={m['method']:m for m in saved['method_summary']}
    methods={m['method'] for a in audits for m in a['methods']}
    if set(declared)!=methods:errors.append('summary_method_set')
    any_false=0
    for a in audits:
        if any(m['decisions'].get('false_rejection',False) for m in a['methods']):any_false+=1
    if decisions_available:
        if saved.get('any_method_false_rejections')!=any_false:errors.append('summary_any_method_false_rejections')
        if saved.get('policy_false_rejection_denominator')!=sum(a['candidate_coverage'] for a in audits):errors.append('summary_policy_risk_denominator')
    for method in methods:
        rows=[(a,m) for a in audits for m in a['methods'] if m['method']==method]
        complete=sum(m['joint_complete'] for a,m in rows)
        counts=dict(all_history_count=n,complete_count=complete,failed_count=n-complete,
                    covered_complete_count=sum(a['candidate_coverage'] and m['joint_complete'] for a,m in rows))
        if decisions_available:
            counts.update(false_rejections=sum(m['decisions']['false_rejection'] for a,m in rows if a['candidate_coverage']),
                only_truth_retained=sum(m['decisions']['only_truth_retained'] for a,m in rows if a['candidate_coverage']),
                unresolved=sum(m['decisions']['unresolved'] for a,m in rows if a['candidate_coverage']),
                false_rejection_denominator=sum(a['candidate_coverage'] for a,m in rows),
                incomplete_retained_all_count=sum(a['candidate_coverage'] and not m['joint_complete'] for a,m in rows))
        for key,value in counts.items():
            if declared.get(method,{}).get(key)!=value:errors.append('summary_'+method+'_'+key)
    return dict(success=not errors,errors=errors,recomputed_counts=expected,
                any_method_false_rejections=any_false if decisions_available else None)

def export_phase(output,compact_directory,phase,*,source_root=HERE):
    output=Path(output);compact_directory=Path(compact_directory);raw=output/phase;target=compact_directory/phase
    config=read_json(raw/'config.json');paths=sorted(raw.glob('trial_*.json.gz'))
    if len(paths)!=config['count']:raise ValueError('phase incomplete; export after all planned histories finish')
    expected={key:config[key] for key in ('config','source_sha256','policy_id')}
    if phase!='pilot' and read_json(output/'freeze.json')!=expected:raise ValueError('formal freeze mismatch')
    source_checks={}
    if source_root is not None:
        for name,digest in expected['source_sha256'].items():
            file=Path(source_root)/name
            source_checks[name]=file.exists() and sha(file)==digest
    calibration=read_json(output/'calibration'/'thresholds.json') if phase=='evaluation' else None
    audits=[];manifest=[];errors=[]
    for path in paths:
        original_hash=sha(path);trial=read_json(path);compact=compact_value(trial)
        audit=audit_trial(compact,calibration=calibration,expected=expected);audits.append(audit)
        write_json(target/path.name,compact,compress=True)
        if sha(path)!=original_hash:raise RuntimeError('raw evidence changed during export')
        manifest.append(dict(history_id=trial['history_id'],raw_path=str(path),raw_sha256=original_hash,
            compact_path=path.name,compact_sha256=sha(target/path.name),raw_bytes=path.stat().st_size,
            compact_bytes=(target/path.name).stat().st_size,raw_retained=True))
    ids=[a['history_id'] for a in audits]
    if set(map(int,ids))!=set(config['seeds']):errors.append('phase_seed_list')
    if len(ids)!=len(set(ids)):errors.append('duplicate_phase_history')
    if any(not good for good in source_checks.values()):errors.append('current_source_hash_mismatch')
    calibration_audit=None
    if phase=='calibration':
        saved=read_json(raw/'thresholds.json');calibration_audit=audit_calibration(audits,saved,config['config']['alpha'])
    elif phase=='evaluation':
        cconfig=read_json(output/'calibration'/'config.json')
        cpaths=sorted((output/'calibration').glob('trial_*.json.gz'))
        if len(cpaths)!=cconfig['count']:errors.append('incomplete_calibration_bank')
        caudits=[audit_trial(read_json(p),expected=expected) for p in cpaths]
        calibration_audit=audit_calibration(caudits,calibration,config['config']['alpha'])
        if any(not a['success'] for a in caudits):errors.append('calibration_physical_replay_failed')
        if set(ids)&set(calibration['calibration_history_ids']):errors.append('evaluation_calibration_overlap')
    if calibration_audit is not None and not calibration_audit['success']:errors+=calibration_audit['errors']
    summary_audit=audit_summary(audits,read_json(raw/'summary.json'),phase=='evaluation') if (raw/'summary.json').exists() else None
    if summary_audit is not None and not summary_audit['success']:errors+=summary_audit['errors']
    for file in ('config.json','summary.json','thresholds.json'):
        if (raw/file).exists():
            (target/file).parent.mkdir(parents=True,exist_ok=True);(target/file).write_bytes((raw/file).read_bytes())
    if (output/'freeze.json').exists():
        compact_directory.mkdir(parents=True,exist_ok=True);(compact_directory/'freeze.json').write_bytes((output/'freeze.json').read_bytes())
    if phase=='evaluation':
        (compact_directory/'calibration').mkdir(parents=True,exist_ok=True)
        (compact_directory/'calibration'/'thresholds.json').write_bytes((output/'calibration'/'thresholds.json').read_bytes())
    methods=sorted({m['method'] for a in audits for m in a['methods']})
    counts={method:dict(all_history_count=len(audits),false_rejections=sum(m['decisions'].get('false_rejection',False) for a in audits for m in a['methods'] if m['method']==method),
        only_truth_retained=sum(m['decisions'].get('only_truth_retained',False) for a in audits for m in a['methods'] if m['method']==method),
        unresolved=sum(m['decisions'].get('unresolved',False) for a in audits for m in a['methods'] if m['method']==method),
        recorded_method_failures=sum(m['recorded_status']!='complete' for a in audits for m in a['methods'] if m['method']==method)) for method in methods}
    report=dict(success=not errors and all(a['success'] for a in audits),errors=errors,phase=phase,
        all_history_count=len(audits),passed_history_count=sum(a['success'] for a in audits),
        failed_history_count=sum(not a['success'] for a in audits),recorded_scientific_failure_count=sum(a['recorded_failure_count'] for a in audits),
        actual_fit_replay_count=sum(a['actual_fit_replay_count'] for a in audits),
        unreplayable_recorded_fit_count=sum(a['unreplayable_recorded_fit_count'] for a in audits),
        original_source_sha256=expected['source_sha256'],current_source_checks=source_checks,
        policy_id=expected['policy_id'],calibration_audit=calibration_audit,method_counts=counts,summary_audit=summary_audit,
        verifier_sha256=sha(Path(__file__)),verifier_schema='gradient_price_independent_AC_v1',
        calibrated_decisions_available=phase=='evaluation',
        removed_keys=sorted(DROP_KEYS),all_other_fields_retained=True,raw_full_archive_retained=True,
        manifest=manifest,audits=audits)
    write_json(target/'independent_replay.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compact-directory',type=Path,required=True)
    parser.add_argument('--phase',choices=('pilot','calibration','evaluation'),required=True)
    args=parser.parse_args();report=export_phase(args.output,args.compact_directory,args.phase)
    print(json.dumps({k:report[k] for k in ('success','all_history_count','passed_history_count','failed_history_count','actual_fit_replay_count')},ensure_ascii=False))
    if not report['success']:raise SystemExit(2)


if __name__=='__main__':main()
