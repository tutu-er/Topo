"""Independent review of the frozen failure-retain-all EV formal study.

The scientific wrapper and its summarizer are never imported. This module
recomputes rejection scores, the rank39 threshold and retained sets, then uses
the stable branch-sweep numeric auditor. It has no selector or fitting calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from scipy.stats import beta
from . import ev_flexibility_replay as numeric
from .information_fit_recovery import read_json,write_json,sha

HERE=Path(__file__).resolve().parent
METHODS=numeric.METHODS
WRAPPERS={'ev_flexibility_formal_safe.py','ev_flexibility_formal_safe_protocol.md'}
STATES=('unique_correct','false_rejection','safe_undecided','compute_failure')


def require(ok,message):
    if not ok:raise ValueError(message)


def losses(row,names):
    if row.get('status')!='complete' or not names or set(row.get('losses',{}))!=set(names):return None
    try:values={name:float(row['losses'][name]) for name in names}
    except (TypeError,ValueError):return None
    return values if all(np.isfinite(v) and v>=0 for v in values.values()) else None


def names_and_rows(trial):
    names=[r['name'] for r in trial.get('candidates',[])];require(len(names)==len(set(names)),'duplicate candidates')
    rows={r['method']:r for r in trial['records']};require(len(rows)==len(trial['records'])==9 and set(rows)==set(METHODS),'9 method Cartesian coverage')
    return names,rows


def history_failed(trial,names):
    fits=trial.get('historical_fits',[]);by={r.get('name'):r for r in fits}
    if trial.get('runner_exception') or trial.get('historical_failures') or not names or len(by)!=len(fits) or set(by)!=set(names):return True
    for row in fits:
        try:value=float(row.get('loss'))
        except (TypeError,ValueError):return True
        if not row.get('success',False) or not np.isfinite(value) or value<0:return True
    return False


def score_history(trial):
    names,rows=names_and_rows(trial);historical=history_failed(trial,names)
    consistent=len({r.get('true_name') for r in rows.values()})==1
    scores={};raw_gaps={};failed={};coverage={}
    for method,row in rows.items():
        known=row.get('true_covered') is True and row.get('true_name') in names
        coverage[method]=known;values=losses(row,names);failure=historical or values is None
        failed[method]=failure
        raw_gaps[method]=None if failure or not known else values[row['true_name']]-min(values.values())
        scores[method]=None if not known else 0. if failure else raw_gaps[method]
    infinite=not consistent or not all(coverage.values())
    return dict(history_id=trial['history_id'],score_is_infinite=infinite,score=None if infinite else max(scores.values()),method_rejection_scores=scores,method_raw_truth_gaps=raw_gaps,failed_methods=failed,coverage=coverage,historical_failure=historical,failure_retained=any(failed.values()),scope='failure0 means no candidate rejection; original fitting losses and failure status remain')


def threshold(scores):
    require(len(scores)==39,'exactly39 independent calibration H')
    alpha,delta=Fraction(1,10),Fraction(1,20);tail=(1-alpha)**39
    require(tail<=delta,'rank39 tolerance check')
    infinite=any(s['score_is_infinite'] for s in scores)
    value=None if infinite else max(s['score'] for s in scores)
    return dict(n=39,rank=39,alpha=str(alpha),delta=str(delta),binomial_tail_exact=str(tail),binomial_tail=float(tail),threshold=value,threshold_is_infinite=infinite,coverage_failure_count=sum(s['score_is_infinite'] for s in scores),failure_retained_history_count=sum(s['failure_retained'] for s in scores))


def retained(names,values,limit,*,failed=False):
    # This deployment rule depends on available losses and failure flags only.
    if failed or values is None:return list(names)
    best=min(values.values());return [name for name in names if values[name]-best<=limit]


def decisions(trial,limit):
    names,rows=names_and_rows(trial);hf=history_failed(trial,names);answer={}
    for method,row in rows.items():
        values=losses(row,names);failed=hf or values is None;kept=retained(names,values,limit,failed=failed)
        covered=row.get('true_covered') is True and row.get('true_name') in names
        state='compute_failure' if failed or not covered else 'false_rejection' if row['true_name'] not in kept else 'unique_correct' if kept==[row['true_name']] else 'safe_undecided'
        selection=row.get('selection',trial.get('selection',{}).get('methods',{}).get(method,{}));activated=bool(trial.get('selection',{}).get('activated'))
        kind=numeric.selection_kind({**row,'selection':selection},trial.get('selection',{}))
        margin=None if failed or not covered else min(v for name,v in values.items() if name!=row['true_name'])-values[row['true_name']]
        answer[method]=dict(state=state,retained=kept,coverage=covered,fit_or_history_failure=failed,status=row['status'],selection_kind=kind,selection_status=selection.get('status'),signed_margin_full_SSE=margin,wrong_candidates_removed=None if not covered else sum(name!=row['true_name'] and name not in kept for name in names),activated=activated)
    return answer


def validate_phase(trials,freeze,phase):
    require(freeze.get('schema')=='ev_service_formal_v2_safe_rejection','safe formal envelope')
    labels={str(row['seed']):row for row in freeze['schedule'][phase]}
    require(len(trials)==len(labels) and {r['history_id'] for r in trials}==set(labels),'complete independent H schedule/duplicates')
    require(tuple(freeze['methods'])==METHODS,'frozen9 method family')
    policy={k:v for k,v in freeze['source_sha256'].items() if k not in WRAPPERS}
    require(len(policy)==21 and len(freeze['source_sha256'])==23,'21 policy plus2 wrapper sources')
    for trial in trials:
        label=labels[trial['history_id']];meta=trial['formal']
        require(meta['policy_id']==freeze['policy_id'] and meta['phase']==phase and meta['labels']==label,'safe formal trial identity')
        require(meta['runner_kind']=='real' and meta['one_quality_per_H'] and meta['old_threshold_applied'] is False,'real single-quality frozen record')
        require(trial['source_sha256']==policy and trial['run_id']==trial['history_id']+'_'+trial['quality'],'scientific policy source/run identity')
        require(trial['case']==label['case'] and trial['quality']==label['quality'],'frozen case/quality')
        require(trial['selector_uses_truth'] is False and trial['selector_uses_future_noise'] is False,'selector isolation')
        names_and_rows(trial)
        if not trial.get('runner_exception'):
            require(trial['config']==freeze['policy_config'],'unchanged policy config')
            require(trial['history_hash']==numeric.fingerprint(trial['history_observations']),'saved H fingerprint')
            require(trial['truth']['supports']==label['family'],'frozen true family')
    return labels


def wrapper_tests(directory):
    evidence=HERE/'evidence/ev_flexibility/safe_wrapper_validation.json';value=read_json(evidence)
    require(value['exit_code']==0 and '37 safe' in value['scope'],'prior37 safe test evidence')
    for name,expected in value['source_sha256'].items():require(sha(HERE/name)==expected,'37-test source bytes:'+name)
    return dict(success=True,evidence_sha256=sha(evidence),safe_tests=37,source_sha256=value['source_sha256'],scope='fake-only validation recorded before the new scientific run; current bytes match')


def interval(k,n):
    return dict(lower=0. if k==0 else float(beta.ppf(.025,k,n-k+1)),upper=1. if k==n else float(beta.ppf(.975,k+1,n-k)),count=k,n=n,confidence=.95)


def evaluation_stats(trials,limit):
    gates=[decisions(t,limit) for t in trials];n=len(trials);methods={}
    for method in METHODS:
        count=Counter(g[method]['state'] for g in gates);selection=Counter(g[method]['selection_kind'] for g in gates)
        require(sum(count.values())==n,'failure/state denominator partition')
        methods[method]=dict(n=n,counts={state:count[state] for state in STATES},selection_counts=dict(selection),activated=sum(g[method]['activated'] for g in gates),wrong_candidates_removed=sum(g[method]['wrong_candidates_removed'] or 0 for g in gates))
    slices={}
    for field in ('case','quality'):
        slices[field]={}
        for label in sorted({t[field] for t in trials}):
            ids=[i for i,t in enumerate(trials) if t[field]==label]
            slices[field][label]=dict(n=len(ids),methods={m:dict(Counter(gates[i][m]['state'] for i in ids)) for m in METHODS})
    comparisons=[]
    for left,right in numeric.PAIRS:
        good=[(t['history_id'],g[left]['signed_margin_full_SSE']-g[right]['signed_margin_full_SSE']) for t,g in zip(trials,gates) if g[left]['signed_margin_full_SSE'] is not None and g[right]['signed_margin_full_SSE'] is not None]
        array=np.asarray([v for _,v in good]);wins=sum(g[left]['state']=='unique_correct' and g[right]['state']!='unique_correct' for g in gates);losses_count=sum(g[right]['state']=='unique_correct' and g[left]['state']!='unique_correct' for g in gates)
        comparisons.append(dict(method=left,baseline=right,planned_H=n,margin_complete_H=len(good),margin_unavailable_H=n-len(good),mean_margin_difference=None if not len(array) else float(array.mean()),margin_SE=None if len(array)<2 else float(array.std(ddof=1)/np.sqrt(len(array))),positive_margin_H=int((array>0).sum()),negative_margin_H=int((array<0).sum()),unique_discordant_wins=wins,unique_discordant_losses=losses_count))
    any_false=sum(any(g[m]['state']=='false_rejection' for m in METHODS) for g in gates)
    return dict(n=n,methods=methods,by_case=slices['case'],by_quality=slices['quality'],comparisons=comparisons,any_method_false_rejections=any_false,any_method_false_rejection_interval=interval(any_false,n),scope='frozen iid simulation mixture; case/quality slices descriptive; no topology probability'),gates



def cost_summary(trials):
    def duration(row,key):
        value=row.get(key)
        return None if value is None else float(value)
    rows=[]
    for trial in trials:
        selection=trial.get('selection',{});config=trial.get('config',{})
        total=duration(trial,'seconds');controller=duration(selection,'seconds')
        rows.append(dict(history_id=trial['history_id'],condition_seconds=total,controller_seconds=controller,
            saved_gate_passed=bool(trial.get('gate_passed',False)),
            condition_time_exceeded=None if total is None else total>config['condition_seconds_gate'],
            controller_time_exceeded=None if controller is None else controller>config['controller_seconds_gate']))
    total=[r['condition_seconds'] for r in rows if r['condition_seconds'] is not None]
    controller=[r['controller_seconds'] for r in rows if r['controller_seconds'] is not None]
    return dict(planned_H=len(rows),saved_gate_exceedances=sum(not r['saved_gate_passed'] for r in rows),
        condition_time_exceedances=sum(r['condition_time_exceeded'] is True for r in rows),
        controller_time_exceedances=sum(r['controller_time_exceeded'] is True for r in rows),
        known_condition_duration_H=len(total),known_controller_duration_H=len(controller),
        total_condition_seconds=sum(total),total_controller_seconds=sum(controller),
        mean_condition_seconds=None if not total else float(np.mean(total)),
        max_condition_seconds=None if not total else max(total),
        mean_controller_seconds=None if not controller else float(np.mean(controller)),
        max_controller_seconds=None if not controller else max(controller),rows=rows,
        scope='all planned H kept; recorded gate also includes fitting failures; duration sums describe this execution environment')


def compare_scientific_summaries(left,right):
    ignored={'input_sha256','seconds','compact_manifest','git_revision_observed_at_review'}
    require({k:v for k,v in left.items() if k not in ignored}=={k:v for k,v in right.items() if k not in ignored},'compact versus original scientific results differ')
    return dict(success=True,scientific_payload_exactly_equal=True,ignored_metadata=sorted(ignored))

def phi_margin(trials):
    out={}
    for method,baseline in [('charge_fixed','constant_fixed'),('charge_mobile','constant_fixed'),('v2g_fixed','charge_fixed'),('charge_staff','charge_mobile')]:
        pairs=[]
        for trial in trials:
            _,rows=names_and_rows(trial);a,b=rows[method],rows[baseline]
            av=a.get('selection',{}).get('predicted_score');bv=b.get('selection',{}).get('information_score') if baseline=='constant_fixed' else b.get('selection',{}).get('predicted_score')
            if av is not None and bv is not None and a.get('signed_margin_full_SSE') is not None and b.get('signed_margin_full_SSE') is not None:
                pairs.append(dict(history_id=trial['history_id'],case=trial['case'],quality=trial['quality'],delta_Phi=av-bv,delta_M=a['signed_margin_full_SSE']-b['signed_margin_full_SSE']))
        x=np.asarray([r['delta_Phi'] for r in pairs]);y=np.asarray([r['delta_M'] for r in pairs]);correlation=None if len(x)<2 or np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
        out[method+'_vs_'+baseline]=dict(planned_H=len(trials),available_H=len(pairs),unavailable_H=len(trials)-len(pairs),Phi_positive_M_positive=sum(r['delta_Phi']>1e-6 and r['delta_M']>0 for r in pairs),Phi_positive_M_negative=sum(r['delta_Phi']>1e-6 and r['delta_M']<0 for r in pairs),Pearson_descriptive=correlation,rows=pairs,scope='same H planned information versus attained AC margin, association only; unavailable or inactive H kept in denominator')
    return out


def export_compact(directory,output,*,phases):
    directory,output=Path(directory),Path(output);require(directory.resolve()!=output.resolve(),'new compact directory required');output.mkdir(parents=True,exist_ok=True)
    for name in ('freeze.json','science_source_snapshot.json.gz',*(phase+'_summary.json' for phase in phases)):
        target=output/name
        if target.exists():require(sha(target)==sha(directory/name),'different prior compact envelope')
        else:shutil.copyfile(directory/name,target)
    files=[]
    for phase in phases:
        for path in sorted((directory/phase).glob('trial_*.json.gz')):
            removed=[];value=numeric.compact_trial(read_json(path),removed=removed)
            encoded=gzip.compress((json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
            relative=path.relative_to(directory);target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(target.read_bytes()==encoded,'different prior compact scientific record')
            else:target.write_bytes(encoded)
            files.append(dict(name=relative.as_posix(),raw_sha256=sha(path),raw_bytes=path.stat().st_size,compact_sha256=sha(target),compact_bytes=target.stat().st_size,removed=removed))
    require(files,'no completed records to compact')
    # Existing cal export stays intact when evaluation is appended.
    manifest=dict(schema='EV_diagnostic_compact_v2',raw_directory=str(directory.resolve()),field_policy='only2 named long SVD diagnostics removed, all other scientific fields retained',files=files,raw_total_bytes=sum(r['raw_bytes'] for r in files),compact_total_bytes=sum(r['compact_bytes'] for r in files),original_files_unchanged=True)
    write_json(output/'compact_manifest.json',manifest);return manifest


def reported_summary(directory,phase):
    # Absolute raw_directory is provenance only: committed compact evidence is sufficient.
    path=Path(directory)/(phase+'_summary.json')
    require(path.is_file(),'missing local frozen '+phase+' summary')
    return read_json(path)


def git_revision():
    executable=shutil.which('git') or r'C:\Users\23761\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
    result=subprocess.run([executable,'-C',str(HERE.parents[1]),'rev-parse','HEAD'],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else None


def run(directory,phase,*,output=None):
    began=perf_counter();directory=Path(directory);freeze=read_json(directory/'freeze.json');tests=wrapper_tests(directory);sources=numeric.source_audit(directory,freeze);manifest=numeric.verify_compact_manifest(directory)
    cal_paths=sorted((directory/'calibration').glob('trial_*.json.gz'));cal=[read_json(p) for p in cal_paths];validate_phase(cal,freeze,'calibration')
    scores=[score_history(t) for t in cal];rule=threshold(scores)
    reported=reported_summary(directory,'calibration')
    require(reported['tolerance']=={**rule,'scope':reported['tolerance']['scope']},'independently reconstructed threshold differs')
    require(reported['independent_history_ids']==[r['history_id'] for r in scores] and len(reported['scores'])==len(scores),'calibration score H coverage/order')
    calibration_counts=dict(case=dict(Counter(t['case'] for t in cal)),quality=dict(Counter(t['quality'] for t in cal)),failure_retained_H=sum(s['failure_retained'] for s in scores),failed_methods=sum(sum(s['failed_methods'].values()) for s in scores),unknown_coverage_H=sum(s['score_is_infinite'] for s in scores))
    require(reported['case_counts']==calibration_counts['case'] and reported['quality_counts']==calibration_counts['quality'],'calibration case/quality denominator')
    require(reported['failure_retained_histories']==calibration_counts['failure_retained_H'] and reported['failed_method_count']==calibration_counts['failed_methods'],'calibration failure denominator')
    for original,recomputed in zip(reported['scores'],scores):
        for key in ('score','score_is_infinite','method_rejection_scores','method_raw_truth_gaps','failure_retained'):require(original[key]==recomputed[key],'calibration score:'+key)
    trials=cal if phase=='calibration' else [read_json(p) for p in sorted((directory/'evaluation').glob('trial_*.json.gz'))]
    if phase=='evaluation':validate_phase(trials,freeze,phase);require(not rule['threshold_is_infinite'],'coverage failure prevents meaningful evaluation')
    numeric_began=perf_counter();replays=[numeric.audit_trial(t) for t in trials];numeric_seconds=perf_counter()-numeric_began;number,method_rows=numeric.summary(trials,replays,quality_count=1)
    number.update(stage='safe_formal_numeric_replay',phase=phase,source_file_count=sources['source_count'],scope='one independent quality condition per H; frozen safe mixture; numeric replay and diagnostic minimum-SSE ranking')
    stats,gates=(None,None) if phase=='calibration' else evaluation_stats(trials,float(rule['threshold']))
    costs=cost_summary(trials)
    if phase=='calibration':require(reported['cost_gate_exceedances']==costs['saved_gate_exceedances'],'calibration cost denominator')
    result=dict(success=True,phase=phase,policy_id=freeze['policy_id'],threshold=rule,independent_H=len(trials),numerical_summary=number,statistics=stats,costs=costs,calibration_counts=calibration_counts,Phi_margin_relations=phi_margin(trials),counts=number['counts'],source_audit=sources,prior37_test_source_audit=tests,compact_manifest=manifest,input_sha256={p.relative_to(directory).as_posix():sha(p) for p in sorted((directory/phase).glob('trial_*.json.gz'))},freeze_sha256=sha(directory/'freeze.json'),review_module_sha256=sha(Path(__file__)),numeric_replay_module_sha256=sha(Path(numeric.__file__)),git_revision_observed_at_review=git_revision(),old_threshold_applied=False,new_fit_calls=0,new_selector_calls=0,whole_research_complete=False,moving_plan_interpretation='equal terminal battery service; moving plans jointly change site, timing, and grid energy replacing driving consumption; report complete executable-plan comparison')
    if phase=='evaluation':
        reported_eval=reported_summary(directory,'evaluation')
        for m in METHODS:require(reported_eval['methods'][m]['counts']==stats['methods'][m]['counts'],'independent decision count:'+m)
        require(reported_eval['any_method_false_rejections']==stats['any_method_false_rejections'],'any9 false rejection count')
        require(reported_eval['cost_gate_exceedances']==costs['saved_gate_exceedances'],'evaluation cost denominator')
    result['seconds']=dict(numerical_replay=numeric_seconds,review_before_output=perf_counter()-began)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True);write_json(output/'summary.json',result);write_json(output/'independent_replay.json.gz',dict(success=True,optimizer_invoked=False,trials=replays,counts=number['counts'],input_sha256=result['input_sha256']))
        gates_by={} if gates is None else {t['history_id']:g for t,g in zip(trials,gates)}
        for row in method_rows:
            g=gates_by.get(row['history_id'],{}).get(row['method'],{})
            row.update(retention_state=g.get('state'),retained_candidates=None if not g else json.dumps(g['retained']),wrong_candidates_removed=g.get('wrong_candidates_removed'))
        with (output/'methods.csv').open('w',encoding='utf-8',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(method_rows[0]));writer.writeheader();writer.writerows(method_rows)
        text=numeric.analysis_text(number).replace('这些H的案例预先固定，表中均值和H间离散程度用于开发判断。方法是否稳定改善排错及误拒，仍需冻结策略后的新独立校准和评价。','本批为冻结策略后的独立校准或评价，每个H仅有一个计量质量条件。校准部分用于阈值，收益判断只使用独立评价部分。')+'\n本文件使用新safe封套独立复算阈值；失败方法保留全部候选，score0仅编码没有拒绝。校准数据不用于独立收益结论。\n\n移动方案保持相同终端电池任务，同时改变位置、时段及补回驾驶耗电的网侧电量。以下对比评价完整可执行方案，位置的单独因果效应需要另设对照。\n'
        text=text.replace('这里只报告开发数据的最小SSE排序和有符号裕量。','前表报告保存计量的最小SSE排序和有符号裕量，正式保留集合在下表。' if phase=='evaluation' else '前表仅为校准数据的最小SSE诊断，阈值收益由独立评价检验。')
        text+=f"\n本批保存成本门超限 {costs['saved_gate_exceedances']}/{costs['planned_H']}；控制器时间超限 {costs['controller_time_exceedances']}，整条件时间超限 {costs['condition_time_exceedances']}。数值回算用时 {numeric_seconds:.3f}s，回算调用拟合器与选择器均为0。\n"
        if stats is not None:
            text+='\n| 方法 | 唯一正确/安全未决/误拒/计算失败 | 分母 |\n|---|---:|---:|\n'
            for method,value in stats['methods'].items():
                c=value['counts'];text+=f"| {method} | {c['unique_correct']}/{c['safe_undecided']}/{c['false_rejection']}/{c['compute_failure']} | {value['n']} |\n"
            text+=f"\n任一9方法误拒 {stats['any_method_false_rejections']}/{stats['n']}；区间与按案例/质量分组统计保存在summary。分组结果描述本批数据。\n"
        (output/'analysis.md').write_text(text,encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,default=HERE/'evidence/ev_flexibility/v2_safe_formal');parser.add_argument('--phase',choices=('calibration','evaluation'),required=True);parser.add_argument('--output',type=Path);parser.add_argument('--compact-output',type=Path)
    args=parser.parse_args();directory=args.directory
    if args.compact_output is not None:
        phases=('calibration',) if args.phase=='calibration' else ('calibration','evaluation');export_compact(directory,args.compact_output,phases=phases);directory=args.compact_output
    output=args.output or directory/args.phase/'safe_independent';result=run(directory,args.phase,output=output)
    print(json.dumps(dict(success=True,phase=args.phase,H=result['independent_H'],threshold=result['threshold'],counts=result['counts'],output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
