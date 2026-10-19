"""Independent B stored-bank probability/calibration audit, without new fits.

Only the completed frozen replay is read. The independent A auditor supplies
Cholesky calculations and history-level bootstrap, not scientific probability
or analysis functions. Raw records, plans, source files and calibration remain
unchanged. No AC solve, GHK, optimization, new action or noise is generated.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
from hashlib import sha256
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path

HERE=Path(__file__).resolve()
PROBABILITY_PATH=HERE.parents[3]/'evidence/ev_ac_repair_v1/scripts/statistical_audit.py'
spec=spec_from_file_location('independent_A_probability_for_B_audit',PROBABILITY_PATH)
prob=module_from_spec(spec);spec.loader.exec_module(prob)
np=prob.np
TARGET_MODELS=('G00','G11','G11_cal')
METRICS=('error','clean_error','false_rejection','unresolved','correct_decisive',
         'wrong_decisive','coverage90','coverage90_terminal_P','coverage90_terminal_Q',
         'coverage90_voltage','coverage90_root_P','coverage90_root_Q',
         'true_predictive_NLL','classification_NLL','brier')


def read(path):return prob.read(path)
def digest(path):return prob.digest(path)
def canonical(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def history_rows(audits):return [row for item in audits for row in item['rows']]


def history_values(rows,metric,plan,model):
    grouped=defaultdict(dict)
    for row in rows:
        if row.get('available') and row['plan']==plan and row['model']==model and metric in row:
            if row['regime'] in grouped[row['index']]:raise ValueError('duplicate history/plan/model/regime')
            grouped[row['index']][row['regime']]=float(row[metric])
    return {index:float(np.mean([values[k] for k in prob.REGIMES])) for index,values in grouped.items()
            if set(values)==set(prob.REGIMES)}


def augment_operational(summary,audits):
    """Add independent all-history bootstrap and explicit 64-task counts."""
    by={(row['index'],row['plan'],row['model'],row['regime']):row
        for item in audits for row in item['rows']}
    fields=('false_rejection','unresolved','correct_decisive','wrong_decisive')
    for key,group in summary['summaries'].items():
        plan,model=key.rsplit('__',1);values={field:[] for field in fields}
        for item in audits:
            within={field:[] for field in fields}
            for regime in prob.REGIMES:
                row=by.get((item['metadata']['index'],plan,model,regime),{})
                available=row.get('available') and all(field in row for field in fields)
                for field in fields:within[field].append(row[field] if available else float(field=='unresolved'))
            for field in fields:values[field].append(float(np.mean(within[field])))
        operational=group['operational_all_histories']
        operational.update(histories=len(audits),tasks=2*len(audits),
            available_tasks=2*len(audits)-operational['fallback_regime_tasks'],
            failed_tasks=operational['fallback_regime_tasks'])
        for field in fields:operational['metrics'][field]['bootstrap95']=prob.bootstrap(values[field])
    return summary


def paired(rows_A,rows_B,count):
    result={}
    for plan in prob.PLANS:
        for ma,mb in (('G00','G00'),('G11','G11'),('G11_cal','G11_cal'),
                      ('AC_fixed','G11_cal'),('AC_joint','G11_cal')):
            detail={}
            for metric in METRICS:
                a=history_values(rows_A,metric,plan,ma);b=history_values(rows_B,metric,plan,mb)
                ids=sorted(set(a)&set(b));diff=[b[i]-a[i] for i in ids]
                detail[metric]=dict(histories=len(ids),total_histories=count,missing_pairs=count-len(ids),
                    paired_indices=ids,A_mean=float(np.mean([a[i] for i in ids])) if ids else None,
                    B_mean=float(np.mean([b[i] for i in ids])) if ids else None,
                    B_minus_A=float(np.mean(diff)) if ids else None,bootstrap95=prob.bootstrap(diff),
                    lower_histories=sum(v<0 for v in diff),tied_histories=sum(v==0 for v in diff),
                    higher_histories=sum(v>0 for v in diff))
                if metric in ('error','clean_error'):
                    detail[metric]['all_history_difference_range']=[
                        max(-1.,(sum(diff)-(count-len(ids)))/count),
                        min(1.,(sum(diff)+(count-len(ids)))/count)]
            result[plan+':A_'+ma+'->B_'+mb]=detail
    return result


def scientific_failures(record,source=None):
    """Classify saved failures, with no change to any scientific status."""
    categories=set();candidate_status=Counter();MAP_status=Counter();posterior_status=Counter()
    index=record['spec']['index']
    if record.get('status')=='process_failure':categories.add('worker_process_failure')
    if record.get('status')=='input_failure':categories.add('observed_input_failure')
    if set(record.get('plans',{}))!=set(prob.PLANS):categories.add('saved_fixed_plan_missing')
    if source is not None and set(source.get('plans',{}))!=set(prob.PLANS):
        categories.add('original_A_plan_missing')
    for row in record.get('history_refresh',{}).get('candidates',{}).values():
        candidate_status[str(row.get('status','missing'))]+=1
        fitted=row.get('MAP',{});post=row.get('posterior',{})
        if fitted:
            MAP_status[str(fitted.get('status','missing'))]+=1
            if not fitted.get('success'):categories.add('historical_MAP_not_accepted')
        if post:
            posterior_status[str(post.get('status','missing'))]+=1
            if fitted.get('success') and not post.get('success'):
                categories.add('historical_posterior_failure')
    for action in record.get('actions',{}).values():
        propagation=action.get('propagation',{})
        if propagation.get('status')=='context_unavailable':categories.add('historical_context_unavailable')
        elif propagation.get('status')!='ok':categories.add('future_propagation_failure')
        banks=propagation.get('banks',{})
        if 'G11' in banks and 'G11' not in prob.eligible_banks(action,dict(
                propagation_mean_MCSE_limit=.2,propagation_covariance_MCSE_relative_limit=.05)):
            categories.add('future_moment_MCSE_gate')
        for regime in prob.REGIMES:
            if action.get('regimes',{}).get(regime,{}).get('values') is None:
                categories.add('saved_future_measurement_missing')
    return dict(index=index,status=record.get('status'),history_status=record.get('history_refresh',{}).get('status'),
        categories=sorted(categories),candidate_status_counts=dict(candidate_status),
        MAP_status_counts=dict(MAP_status),posterior_status_counts=dict(posterior_status))


def stored_score_comparison(record,audited):
    errors=[]
    for row in audited['rows']:
        if row['model'] not in ('G00','G11'):continue
        plan=record.get('plans',{}).get(row['plan'],{})
        action=record.get('actions',{}).get(plan.get('action_key'),{})
        saved=action.get('regimes',{}).get(row['regime'],{}).get('scores',{}).get(row['model'])
        if saved is None:continue
        if bool(saved.get('available'))!=bool(row['available']):
            errors.append('stored score availability differs: '+str((row['plan'],row['model'],row['regime'])))
        if row['available']:
            errors.extend(prob.walk_compare(saved.get('noisy',{}),row,'stored.noisy'))
            errors.extend(prob.walk_compare(saved.get('clean',{}),row.get('clean',{}),'stored.clean'))
    return errors


def fixed_data_comparison(A,B):
    errors=[]
    if A.get('spec')!=B.get('spec'):errors.append('A/B specification differs')
    if B.get('candidate_names') and A.get('candidate_names')!=B.get('candidate_names'):
        errors.append('candidate order differs')
    if B.get('true_name') is not None and A.get('true_name')!=B.get('true_name'):
        errors.append('true label differs')
    for label,p in B.get('plans',{}).items():
        q=A.get('plans',{}).get(label,{})
        key=p.get('action_key')
        if key!=q.get('action_key'):errors.append(label+': action key differs');continue
        aa=A.get('actions',{}).get(key,{});bb=B.get('actions',{}).get(key,{})
        for value in (q.get('power'),p.get('power'),aa.get('power'),bb.get('power')):
            if value is not None and not np.array_equal(np.asarray(value),np.asarray(aa.get('power'))):
                errors.append(label+': declared/action power differs')
        for regime in prob.REGIMES:
            actual=bb.get('regimes',{}).get(regime,{})
            original=aa.get('regimes',{}).get(regime,{})
            for field in ('values','clean'):
                if field in actual and not np.array_equal(actual[field],original.get(field)):
                    errors.append(label+'/'+regime+': '+field+' differs')
    return errors


def freeze_check(root,expected_policy):
    frozen=read(root/'freeze.json');identity={k:v for k,v in frozen.items() if k!='policy_id'};issues=[]
    if frozen['policy_id']!=expected_policy:issues.append('unexpected B policy')
    if canonical(identity)!=frozen['policy_id']:issues.append('B freeze canonical digest mismatch')
    snapshot=read(root/'source_snapshot.json.gz')
    hashes=frozen['source_sha256'];paths=frozen['source_paths']
    if set(snapshot)!=set(hashes):issues.append('primary snapshot source keys differ')
    for name,value in hashes.items():
        if name not in snapshot or sha256(base64.b64decode(snapshot[name])).hexdigest()!=value:
            issues.append('primary snapshot hash differs: '+name)
        if digest(paths[name])!=value:issues.append('live frozen source differs: '+name)
    supplementary=read(root/'runtime_dependency_git_audit.json')
    package=read(root/'runtime_dependency_snapshot.json.gz')
    supplemental_rows=supplementary['rows'];supp_ok=len(supplemental_rows)==7
    work=Path(frozen['source_paths']['B/optimizer.py']).parents[3]
    if {row['path'] for row in supplemental_rows}!=set(package):supp_ok=False
    for row in supplemental_rows:
        if row['path'] not in package or sha256(base64.b64decode(package[row['path']])).hexdigest()!=row['sha256']:
            supp_ok=False
        if digest(work/row['path'])!=row['sha256']:supp_ok=False
    if not supp_ok:issues.append('supplemental runtime dependency package differs')
    seed_registry=prob.verify_specs(frozen)
    if not seed_registry['passed']:issues.append('calibration/evaluation seeds overlap or duplicate')
    return frozen,dict(primary_source_count=len(hashes),supplemental_source_count=len(supplemental_rows),
        primary_and_supplemental_separate=True,supplemental_snapshot_verified=supp_ok,
        seed_registry=seed_registry,issues=issues)


def report_comparison(saved,summary_B,paired_result):
    errors=[];summaries=summary_B['summaries']
    for plan in prob.PLANS:
        for model in TARGET_MODELS:
            key=plan+'__'+model;own=summaries[key];other=saved.get('B',{}).get('summaries',{}).get(key,{})
            if not other:errors.append('report missing B summary '+key);continue
            for field in ('availability',*prob.SUMMARY_METRICS):
                if (field in own)!=(field in other):errors.append(key+': missing/extra metric '+field)
            errors.extend(prob.walk_compare(other,own,'report.B.'+key))
            for channel,field in (('P','coverage90_terminal_P'),('Q','coverage90_terminal_Q'),
                ('V','coverage90_voltage'),('master_P','coverage90_root_P'),('master_Q','coverage90_root_Q')):
                other_channel=saved.get('B_channel_coverage90',{}).get(key+'__'+channel)
                if field in own and other_channel is not None:
                    errors.extend(prob.walk_compare(other_channel,own[field],'report.channel.'+key+'.'+channel))
            operational=saved.get('B_operational_retain_all',{}).get(key,{})
            if operational:
                if operational.get('histories')!=own['availability']['total_histories']:
                    errors.append(key+': operational denominator differs')
                for metric,value in own['operational_all_histories']['metrics'].items():
                    if metric not in operational or not prob.close(operational[metric]['mean'],value['mean']):
                        errors.append(key+': operational '+metric+' differs')
                    elif 'bootstrap95' in value:
                        errors.extend(prob.walk_compare(operational[metric]['bootstrap95'],value['bootstrap95'],
                            key+': operational '+metric+' bootstrap'))
    for name,own in paired_result.items():
        other=saved.get('paired',{}).get(name)
        if other is None:errors.append('report missing pair '+name);continue
        errors.extend(prob.walk_compare(other,own,'report.pair.'+name))
    return errors


def run(root,report_path,*,expected_policy,A_audit_path):
    root=Path(root).resolve();report_path=Path(report_path);issues=[]
    state=read(root/'controller.json')
    if state.get('status')!='statistical_complete':
        raise RuntimeError('wait for B statistical_complete; no intermediate scientific record opened')
    if not report_path.exists():raise FileNotFoundError('completed frozen report required: '+str(report_path))
    frozen,source=freeze_check(root,expected_policy);issues.extend(source['issues'])
    Aroot=Path(frozen['input_A_root']);Afreeze=read(Aroot/'freeze.json')
    if digest(Aroot/'freeze.json')!=frozen['input_A_freeze_sha256']:issues.append('A freeze binding differs')
    if Afreeze['policy_id']!=frozen['input_A_policy_id']:issues.append('A policy differs')
    settings=dict(frozen['settings'],noise_repeats=Afreeze['settings']['noise_repeats'])
    calibration=read(root/'calibration.json');binding=read(root/'evaluation_calibration_binding.json')
    expected_binding=dict(policy_id=frozen['policy_id'],calibration_sha256=digest(root/'calibration.json'),
        G11_multiplier=calibration['G11_multiplier'],input_A_policy_id=frozen['input_A_policy_id'])
    if binding!=expected_binding:issues.append('B evaluation calibration binding differs')
    A_audit=read(A_audit_path)
    if not A_audit.get('passed') or A_audit['policy_id']!=frozen['input_A_policy_id']:
        issues.append('independent A audit not passed or wrong policy')
    audits={};input_hashes={};faults=[];score_errors=[]
    for phase in ('calibration','evaluation'):
        specs=frozen['specifications'][phase]
        if len(specs)!=32:raise ValueError('all32 phase specifications required')
        audited=[]
        for specification in specs:
            index=specification['index'];name=f'history_{index:03d}.json.gz'
            path=root/phase/name;Apath=Aroot/phase/name
            meta=read(path.with_name(f'history_{index:03d}.metadata.json'));B=read(path);A=read(Apath)
            bh,ah=digest(path),digest(Apath);prefix=phase+'/'+str(index)
            input_hashes[phase+'/'+name]=dict(B=bh,A=ah)
            if (B.get('spec')!=specification or A.get('spec')!=specification or meta.get('spec')!=specification):
                issues.append(prefix+': specification differs')
            if meta.get('policy_id')!=frozen['policy_id'] or meta.get('result_sha256')!=bh or not meta.get('science_source_unchanged'):
                issues.append(prefix+': result/source metadata differs')
            if meta.get('source_A_sha256')!=ah or meta.get('input_A_policy_id')!=frozen['input_A_policy_id']:
                issues.append(prefix+': A input metadata differs')
            if meta.get('calibration_binding')!=(binding if phase=='evaluation' else None):
                issues.append(prefix+': worker calibration binding differs')
            if B.get('source_A',{}).get('sha256',ah)!=ah:issues.append(prefix+': inner A hash differs')
            if phase=='calibration':
                if calibration.get('input_B_sha256',{}).get(name)!=bh:issues.append(prefix+': calibration B raw hash differs')
                if calibration.get('input_A_sha256',{}).get(name)!=ah:issues.append(prefix+': calibration A raw hash differs')
            if A_audit.get('raw_inputs',{}).get(phase+'/'+name)!=ah:issues.append(prefix+': independent A input hash differs')
            issues.extend(prefix+': '+e for e in fixed_data_comparison(A,B))
            item=prob.audit_history(B,settings,calibration if phase=='evaluation' else None,calibration_mode=phase=='calibration')
            if not item['metadata']['true_index_agrees']:issues.append(prefix+': true index/name differs')
            if not item['metadata']['noise_repeat_counts_consistent']:issues.append(prefix+': future repetition count differs')
            differences=stored_score_comparison(B,item)
            score_errors.extend(prefix+': '+e for e in differences)
            if phase=='evaluation':faults.append(scientific_failures(B,A))
            audited.append(item)
            del A,B
        audits[phase]=audited
    recalculated=prob.calibration_from_audits(audits['calibration'])
    cal_errors=prob.walk_compare(calibration,recalculated,'B.calibration')
    if calibration.get('policy_id')!=frozen['policy_id'] or calibration.get('histories')!=32:
        cal_errors.append('B calibration policy/count differs')
    issues.extend(cal_errors);issues.extend(score_errors)
    summary_B=augment_operational(prob.summary(audits['evaluation'],calibrated=True),audits['evaluation'])
    summary_B['summaries']={k:v for k,v in summary_B['summaries'].items() if k.rsplit('__',1)[-1] in TARGET_MODELS}
    rows_A=history_rows(A_audit['stages']['evaluation']['histories']);rows_B=history_rows(audits['evaluation'])
    common=paired(rows_A,rows_B,32)
    saved_report=read(report_path);report_errors=report_comparison(saved_report,summary_B,common)
    if (saved_report.get('B_policy')!=frozen['policy_id'] or
            saved_report.get('A_policy')!=frozen['input_A_policy_id'] or
            saved_report.get('phase')!='evaluation' or saved_report.get('histories')!=32):
        report_errors.append('frozen report policy/phase/count differs')
    saved_inputs={row['index']:dict(A=row['A_sha256'],B=row['B_sha256'])
                  for row in saved_report.get('input_hashes',[])}
    own_inputs={spec['index']:input_hashes['evaluation/history_%03d.json.gz'%spec['index']]
                for spec in frozen['specifications']['evaluation']}
    if saved_inputs!=own_inputs:report_errors.append('frozen report exact 32 raw input hashes differ')
    issues.extend(report_errors)
    categories=defaultdict(list)
    for row in faults:
        for category in row['categories']:categories[category].append(row['index'])
    return dict(version='independent_B_probability_statistics_v1',created_UTC=datetime.now(timezone.utc).isoformat(),
        passed=not issues,audit_issues=issues,policy_id=frozen['policy_id'],A_policy_id=frozen['input_A_policy_id'],
        expected_histories=dict(calibration=32,evaluation=32),source=source,
        calibration=dict(recalculated_multiplier=recalculated['G11_multiplier'],available_histories=recalculated['available_histories'],
            total_histories=32,objective=recalculated['objective'],comparison_errors=cal_errors,
            calibration_sha256=digest(root/'calibration.json'),binding=binding,
            available_history_rule='all three saved plans and both regimes have usable G11; partial valid tasks remain in evaluation denominators',
            scope='one total predictive latent covariance multiplier selected by calibration-only proper NLL; not a line-only scale'),
        evaluation=summary_B,common_history_AB_pairs=common,scientific_failures=dict(
            category_history_indices=dict(categories),histories=faults),input_hashes=input_hashes,
        frozen_report_comparison=dict(path=str(report_path.resolve()),sha256=digest(report_path),errors=report_errors),
        stored_raw_score_comparison_errors=score_errors,A_independent_audit_sha256=digest(A_audit_path),
        audit_script_sha256=digest(__file__),independent_Cholesky_script_sha256=digest(PROBABILITY_PATH),
        costs=dict(AC_calls=0,fit_calls=0,QP_calls=0,GHK_calls=0,planning_calls=0,new_measurements=0,new_physical_samples=0),
        statistical_unit='one independent history; average two future regimes and noise repetitions within history',
        inference_scope='complete-AC history MAP candidate, final affine bounded Gaussian and AC future moments; full nonlinear posterior density is not integrated',
        interpretation='inference repair on unchanged fixed plans; does not establish a new MILP planning advantage')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--report',required=True);parser.add_argument('--A-audit',required=True)
    parser.add_argument('--expected-policy',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('preserve existing independent audit')
    value=run(args.root,args.report,expected_policy=args.expected_policy,A_audit_path=args.A_audit)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(passed=value['passed'],issues=len(value['audit_issues']),
        calibrated_a=value['calibration']['recalculated_multiplier'],calibration_available=value['calibration']['available_histories'],output=str(output))))
    return 0 if value['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
