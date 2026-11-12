"""Offline topology scoring, solver audits, and independent final-family LPs."""
from __future__ import annotations
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

from .run import ROOT,PARENT,read,write,sha
from .engine import neighborhoods
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_five.analyze import independent_lp,size_errors,supports
from research_experiments.local_rebuild.model import canonical,compatible
from research_experiments.local_rebuild.verify import independent_loss
from research_experiments.rnj_count.engine import criterion
from research_experiments.rnj_count.diagnose import error_type
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def analyze(output):
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    p=read(output/'protocol.json');frozen=read(output/'frozen.json');catalog=read(output/'catalog.json')
    assert frozen['protocol_sha256']==sha(output/'protocol.json')
    checks=1;rows=[];events=[];coverage=[];errors=[];failures=[];lp_checks=[];transitions=[];final_errors=[]
    from research_experiments.rnj_count.report import audit_inputs
    input_receipt=audit_inputs(output)
    checks+=input_receipt['observation_archives']+2*input_receipt['network_directories']
    shutil.copy2(output/'input_audit.json',dest/'input_audit.json')
    for name,digest in p['source_hashes'].items():
        assert sha(output/'source'/name)==digest;checks+=1
    assert not set(p['development'])&set(p['evaluation'])
    for phase in ('development','evaluation'):
        jobs=p[phase];settings=({f'lambda_{v:g}':dict(coefficient=v) for v in p['grid']} if phase=='development' else frozen['settings'])
        for job in jobs:
            common=read(output/'common'/job/'result.json');meta=catalog[job]
            assert sha(output/'common'/job/'result.json')==p['common_hashes'][job]
            data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
            n=data['train'].n;labels=common['labels'];truth=physical_truth(Path(meta['truth_network']),labels)
            pos={v:i for i,v in enumerate(labels)};true={tuple(sorted(pos[v] for v in s)) for s in truth['clades']};cache={}
            true_family=canonical([*true,*((i,) for i in range(n)),tuple(range(n))])
            true_fit=independent_lp(data['train'],dict(supports=true_family,objective=0.))
            true_profile=float(true_fit['objective'])
            cache[true_family]=dict(objective=true_profile,status=true_fit['status'])
            lp_checks.append(dict(phase=phase,job=job,supports=true_family,status=true_fit['status'],objective=true_profile,
                role='oracle true family, offline diagnostic only, never used in search or tuning'));checks+=1
            entries=[]
            for method,config in settings.items():
                file=output/phase/job/method/'result.json'
                if not file.exists():
                    failures.append(dict(phase=phase,job=job,method=method,error=read(file.with_name('error.json'))))
                    entries.append((method,None));continue
                r=read(file);t=r['trajectory']
                assert not r['truth_used'] and not r['test_opened']
                assert r['protocol_sha256']==sha(output/'protocol.json')
                assert r['common_sha256']==sha(output/'common'/job/'result.json')
                assert r['input_hashes']==dict(train=sha(meta['train']),validation=sha(meta['validation']));checks+=4
                if phase=='evaluation':
                    assert r['frozen_sha256']==sha(output/'frozen.json');checks+=1
                previous=float('inf')
                for point in t['path']:
                    basis={*((i,) for i in range(n)),tuple(range(n))}
                    assert basis<=set(canonical(point['supports']));checks+=1
                    value=independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],point)
                    assert abs(value-point['objective'])<1e-8
                    assert criterion(point,n,t['penalty'])<=previous+1e-9
                    previous=criterion(point,n,t['penalty']);checks+=2
                path_lookup={canonical(q['supports']):q for q in t['path']}
                last_forward=None
                for e in t['attempts']:
                    if e['kind']=='wzzt_extension':
                        last_forward=e
                    if e.get('status') is not None:
                        assert e['warm_start']['valid'] and e['bound_consistent_with_known_upper'];checks+=2
                        if e['audit']['has_incumbent']:
                            assert e['audit']['valid'];checks+=1
                    if e.get('regularized_bound') is not None:
                        restored=0
                        if e['kind']=='revocable_local':
                            after_family=(set(map(tuple,e['keep']+e['released']))-set(map(tuple,e['deleted'])))|set(map(tuple,e['added']))
                            restored=sum(1<len(s)<n and tuple(s) not in after_family for s in e['keep'])
                        # A fixed-domain lower bound applies before global zero
                        # cleanup. Restore its fixed count constant for a valid UB.
                        e['cleanup_outside_fixed_domain_count']=restored
                        e['model_scope_decoded_score']=e['decoded_score']+e['penalty']*restored
                        e['bound_scope']='local MILP with original kept structure; excludes later global zero cleanup'
                        assert e['regularized_bound']<=e['model_scope_decoded_score']+1e-8;checks+=1
                    if e['kind']=='revocable_local':
                        assert 3<=len(e['blocks'])<=5;checks+=1
                        if e['accepted']:
                            deleted=set(map(tuple,e['deleted']));added=set(map(tuple,e['added']))
                            before_family=canonical(e['keep']+e['released'])
                            after_family=canonical((set(before_family)-deleted)|added)
                            before_pred=supports(path_lookup[before_family],n)
                            after_pred=supports(path_lookup[after_family],n)
                            f1=lambda pred:2*len(pred&true)/(len(pred)+len(true)) if pred or true else 1.
                            stalled=bool(last_forward is not None and not last_forward.get('accepted',False))
                            certified_stall=bool(stalled and (last_forward.get('structurally_complete',False) or
                                (last_forward.get('regularized_bound') is not None and
                                 last_forward['before_score']-last_forward['regularized_bound']<=last_forward['threshold'])))
                            transitions.append(dict(phase=phase,job=job,method=method,case=meta['case'],
                                gain=e['gain'],raw_gain=e['before']-e['decoded_objective'],
                                F1_before=f1(before_pred),F1_after=f1(after_pred),F1_delta=f1(after_pred)-f1(before_pred),
                                exact_before=before_pred==true,exact_after=after_pred==true,
                                after_stalled_addition=stalled,after_certified_stalled_addition=certified_stall,
                                deleted=len(deleted),added=len(added),deleted_true=len(deleted&true),
                                deleted_false=len({s for s in deleted if 1<len(s)<n}-true),
                                added_true=len(added&true),added_false=len({s for s in added if 1<len(s)<n}-true),
                                released_large=sum(len(s)>5 for s in e['released'])))
                            last_forward=None
                    events.append(dict(phase=phase,job=job,method=method,panel=meta['panel'],case=meta['case'],**e))
                for lp in r['LP']:
                    assert lp['status']==2 and max(lp['max_constraint_violation'],lp['max_bound_violation'])<=1e-7;checks+=1
                for stage,point in t['states'].items():
                    retained=supports(point,n,True);wrong=retained-true
                    cross_wrong={s for s in wrong if error_type(s,true)[0]=='crosses_truth'}
                    declared=neighborhoods(point,n,np.asarray(common['kernel']))
                    opened={s for s in cross_wrong if any(set(s)<set(v) and
                        not any(set(s)<=set(b) for b in blocks) for v,blocks,_ in declared)}
                    coverage.append(dict(phase=phase,job=job,method=method,panel=meta['panel'],case=meta['case'],stage=stage,
                        true_present=len(retained&true),false_compatible=sum(error_type(s,true)[0]=='compatible_overresolution' for s in wrong),
                        false_cross=sum(error_type(s,true)[0]=='crosses_truth' for s in wrong),
                        cross_opened_by_declared_frontiers=len(opened),cross_not_opened_by_declared_frontiers=len(cross_wrong-opened),
                        blocked_true=sum(any(not compatible(s,w) for w in wrong) for s in true)))
                entries.append((method,r))
            if phase=='evaluation':
                prior_joint=ROOT/'outputs/joint_five_rnj_wzzt_20261031'
                old=read(prior_joint/'evaluation'/job/'baseline'/'result.json')
                entries.append(('historical_original',old))
                for cid,label in (('lambda_0','historical_joint_zero'),('lambda_0.003','historical_count')):
                    r=read(PARENT/'evaluation'/job/cid/'result.json')
                    r=dict(r,selected=r['selected_plain']['bank'],selected_path=r['selected_plain']['path'])
                    entries.append((label,r))
            for method,r in entries:
                base=dict(phase=phase,job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],
                    N=meta['N'],voltage_sigma=meta['voltage_sigma'],condition=meta['condition'],method=method)
                if r is None:
                    rows.extend(dict(**base,view=v,model_available=False,clade_f1=0.,exact=False) for v in ('bank','path','endpoint'))
                    continue
                assert r['common_sha256']==sha(output/'common'/job/'result.json');checks+=1
                t=r['trajectory']
                for view,point in (('bank',r['selected']['point']),('path',r['selected_path']['point']),('endpoint',t['point'])):
                    metric=score(point,data,truth,labels,tuple(map(tuple,common['pool'])))
                    penalty=t.get('penalty',0.)
                    rows.append(dict(**base,view=view,**metric,search_seconds=t['seconds'],elapsed_seconds=r['elapsed_seconds'],
                        truth_profile_train_MAE=true_profile,
                        train_J_minus_truth_profile=criterion(point,n,penalty)-(true_profile+penalty*len(true)),
                        truth_comparison_tolerance=1e-10+.001*point['objective'],
                        selected_origin=r['selected'].get('origin',r.get('selected_origin','unknown')) if view=='bank' else view,
                        stop_reason=t['stop_reason']))
                    key=canonical(point['supports'])
                    if key not in cache:
                        check=independent_lp(data['train'],point)
                        assert check['absolute_difference']<2e-7
                        cache[key]=check
                        lp_checks.append(dict(phase=phase,job=job,supports=key,**check));checks+=1
                    assert abs(cache[key]['objective']-point['objective'])<2e-7;checks+=1
                    for q in size_errors(point,true,n):
                        errors.append(dict(**base,view=view,**q))
                    pred=supports(point,n);wrong=pred-true
                    final_errors.append(dict(**base,view=view,false_compatible=sum(error_type(s,true)[0]=='compatible_overresolution' for s in wrong),
                        false_cross=sum(error_type(s,true)[0]=='crosses_truth' for s in wrong),
                        blocked_true=sum(any(not compatible(s,w) for w in wrong) for s in true)))
    frame=pd.DataFrame(rows);frame.to_csv(dest/'per_model.csv',index=False)
    summary=frame.groupby(['phase','panel','view','method']).agg(inputs=('job','size'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),internal=('internal','mean'),train_MAE=('train_mae','mean'),
        validation_MAE=('validation_mae','mean'),test_MAE=('test_mae','mean'),search_median=('search_seconds','median')).reset_index()
    geometry=frame.groupby(['phase','panel','view','method'])[['R_relative','X_relative']].mean().reset_index()
    summary=summary.merge(geometry,on=['phase','panel','view','method'],how='left')
    summary.to_csv(dest/'summary.csv',index=False)
    frame[frame.phase=='evaluation'].groupby(['panel','view','case','method']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum')).to_csv(dest/'cases.csv')
    frame[frame.phase=='evaluation'].groupby(['panel','view','voltage_sigma','N','method']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum')).to_csv(dest/'conditions.csv')
    pd.DataFrame(coverage).to_csv(dest/'stage_coverage.csv',index=False)
    pd.DataFrame(transitions).to_csv(dest/'transitions.csv',index=False)
    pd.DataFrame(errors).to_csv(dest/'size_errors.csv',index=False)
    pd.DataFrame(final_errors).groupby(['phase','panel','view','method'])[['false_compatible','false_cross','blocked_true']].sum().to_csv(dest/'final_error_types.csv')
    oracle=frame[(frame.phase=='evaluation')&(frame.view=='endpoint')].copy()
    oracle['wrong_beats_true_profile']=(~oracle.exact)&(oracle.train_J_minus_truth_profile < -oracle.truth_comparison_tolerance)
    oracle['wrong_worse_than_true_profile']=(~oracle.exact)&(oracle.train_J_minus_truth_profile > oracle.truth_comparison_tolerance)
    oracle.to_csv(dest/'oracle_objectives.csv',index=False)
    if errors:
        pd.DataFrame(errors).groupby(['phase','panel','view','method','size'])[['TP','FP','FN']].sum().to_csv(dest/'size_summary.csv')
    paired=[]
    tuned=frozen['tuned_method']
    for (panel,view),g in frame[frame.phase=='evaluation'].groupby(['panel','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for reference in [c for c in pivot if c!=tuned]:
            d=pivot[tuned]-pivot[reference]
            paired.append(dict(panel=panel,view=view,reference=reference,inputs=len(d),mean_delta=float(d.mean()),
                wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(paired).to_csv(dest/'paired.csv',index=False)
    ef=pd.DataFrame([{k:v for k,v in e.items() if not isinstance(v,(dict,list,tuple))} for e in events])
    ef.to_csv(dest/'solver_events.csv',index=False)
    if 'status' in ef:
        ef.groupby(['phase','method','kind','status'],dropna=False).size().to_csv(dest/'solver_summary.csv')
    write(dest/'verification.json',dict(passed=True,checks=checks,failures=failures,unique_independent_LP=len(lp_checks),
        evaluation_planned=len(p['evaluation'])*len(frozen['settings']),
        evaluation_returned=read(output/'evaluation_completion.json')['returned'],
        analysis_source_sha256=sha(Path(__file__))))
    write(dest/'independent_LP.json',lp_checks)
    write(dest/'event_records.json',events)
    delivery=ROOT/'docs/revocable_rebuild_20261031';delivery.mkdir(exist_ok=True)
    for file in dest.iterdir():
        if file.suffix=='.csv' or file.name in ('verification.json','input_audit.json'):
            shutil.copy2(file,delivery/file.name)
    for name in ('protocol.json','frozen.json','tuning_grid.csv'):
        shutil.copy2(output/name,delivery/name)
    for name in ('unit_tests.json','execution_notes.json'):
        if (output/name).exists():
            shutil.copy2(output/name,delivery/name)
    report(output,delivery,frame,summary,paired,transitions)
    with zipfile.ZipFile(delivery/'records.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ('source','development','evaluation','analysis'):
            for file in (output/directory).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.csv','.log'):
                    archive.write(file,file.relative_to(output))
        for name in ('protocol.json','frozen.json','catalog.json','tuning_grid.csv'):
            archive.write(output/name,name)
        for name in ('unit_tests.json','execution_notes.json'):
            if (output/name).exists():
                archive.write(output/name,name)
        archive.write(Path(__file__),'analysis/analysis_runtime.py')
    print(summary[(summary.phase=='evaluation')&(summary.view=='bank')].to_string(index=False),flush=True)


def report(output,dest,frame,summary,paired,transitions):
    frozen=read(output/'frozen.json');verification=read(output/'analysis/verification.json')
    selected=frozen['selected']['coefficient'];tuned=frozen['tuned_method']
    names=dict(step_locked='逐步MILP 加惩罚 固定旧连接',step_zero='逐步MILP 加惩罚 清理零连接',
        revocable_zero='可撤销五块重构 无惩罚',revocable_tuned='可撤销五块重构 开发集选参',
        historical_original='历史原RNJ加WZZT',historical_joint_zero='历史五块联合 无惩罚',historical_count='历史五块联合 搜索惩罚')
    lines=['# 可撤销局部重构完整矩阵测试','',f'开发集64次搜索；最终参数lambda={selected:g}。正式矩阵108主条件加12压力条件，所有方法沿用完整RNJ初值。',
        f"正式返回{verification['evaluation_returned']}/{verification['evaluation_planned']}，独立重解{verification['unique_independent_LP']}个输入与家族组合，核查{verification['checks']}项。",'',
        '本报告采用v2修正版：单点和全集基础响应项即使权重为零也保持可用，并且不计内部线路惩罚。初版允许零清理删除全集，后续新增阶段可能对全集误计一次代价，偏离所保存的数学模型；其开发与矩阵记录保留在outputs/revocable_rebuild_20261031，不纳入修正版汇总。v2重新进行了开发选择、参数冻结和完整矩阵测试。','',
        '## 主面板与压力面板','', '| 面板 | 方法 | F1 | 完整恢复 | 内部连接数 | 搜索中位秒 |','|---|---|---:|---:|---:|---:|']
    for _,r in summary[(summary.phase=='evaluation')&(summary.view=='bank')].iterrows():
        lines.append(f"| {r.panel} | {names.get(r.method,r.method)} | {r.F1:.4f} | {int(r.exact)}/{int(r.inputs)} | {r.internal:.3f} | {r.search_median:.2f} |")
    lines+=['','## 拟合与拓扑分别核对','',
        '| 方法 主面板 | 最终训练MAE | 最终测试MAE | 未收缩终点F1 | 仅路径F1 |','|---|---:|---:|---:|---:|']
    for _,r in summary[(summary.phase=='evaluation')&(summary.panel=='main')&(summary.view=='bank')].iterrows():
        endpoint=summary[(summary.phase=='evaluation')&(summary.panel=='main')&(summary.view=='endpoint')&(summary.method==r.method)].iloc[0]
        path=summary[(summary.phase=='evaluation')&(summary.panel=='main')&(summary.view=='path')&(summary.method==r.method)].iloc[0]
        lines.append(f"| {names.get(r.method,r.method)} | {r.train_MAE:.7g} | {r.test_MAE:.7g} | {endpoint.F1:.4f} | {path.F1:.4f} |")
    main=frame[(frame.phase=='evaluation')&(frame.panel=='main')&(frame.view=='bank')]
    methods=['historical_original','historical_count','step_locked','step_zero','revocable_zero']
    if tuned not in methods:
        methods.append(tuned)
    lines+=['','## 四网络分别比较','', '每格为平均F1和完整恢复次数；每个网络均为27个主条件。','',
        '| 算例 | '+' | '.join(names.get(m,m) for m in methods)+' |',
        '|---|'+'---:|'*len(methods)]
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        values=[]
        for method in methods:
            g=main[(main.case==case)&(main.method==method)]
            values.append(f'{g.clade_f1.mean():.4f} ({int(g.exact.sum())}/27)')
        lines.append('| '+case+' | '+' | '.join(values)+' |')
    lines+=['','## 范围与公平性','',
        '新运行的逐步固定与逐步清零对照均lambda=0.003，唯一策略差别是清理零响应临时集合。可撤销方法使用五块完整子树frontier，也允许在大分叉中按训练RNJ几何选择至多五个兄弟子树。每个frontier候选并集完整枚举，保留外部权重联合重拟合。未增加正阻抗下界。',
        '每条搜索20秒软预算，RNJ重构预选最多5秒，每次MILP最多2秒，四个单线程工作进程。模型构造和LP可能使软预算小幅超出，实际耗时保留。历史三基线复用上一轮同输入结果，计时不同批，不作为同次速度比较。',
        '修正版正式批次前段与保留的初版诊断批次重叠，最多8个单线程工作进程，机器可见32个逻辑处理器；其后仅修正版4进程。execution_notes.json保留实际执行范围，耗时不作严格受控加速结论。',
        '开发bundle0与1的外层观测用于参数选择；正式bundle2至4的测试观测和真拓扑不参与本次选参。这是此前已分析过的既有矩阵复测，不称为新盲测。最终候选均采用相同公共RNJ库、验证MAE、0.16收缩、0至2权重界；另列仅路径和未收缩终点。路径库也包含未修改初值，实际重构纠错另以逐事件变化统计，避免把RNJ回退或保留初值当作搜索修复。',
        '局部候选至多25个，但20秒只覆盖有限邻域。触时与未排除改进的状态保留为unresolved，未报告全局或全邻域收敛。原始训练MAE与含线路代价的J分别核验，J下降允许MAE上升。',
        '与旧五块联合的搜索域不构成简单包含关系：旧方法在固定粗骨架下同时优化多个区域；当前方法一次优化一个区域，能够释放部分较大边界，但需要多次全局系数重拟合。大分叉辅助邻域仅选择兄弟子树，没有枚举其所有进一步打开方式；stage_coverage.csv另报哪些交叉假块能被本次声明的frontier打开，不能把未覆盖写成所有五块邻域都无效。',
        '全局零清理可能在局部MILP结束后删除其原本固定的外部零支撑。MILP下界只针对清理前的固定结构域；核验时把这些被删外部支撑以零权重恢复，并加回其固定线路数代价，形成同域可行上界。清理后更大的结构域不沿用该局部最优性证书。',
        '','## 配对与结构转化','', '| 面板 | 输出 | 比较基线 | 平均F1差 | 胜 平 负 |','|---|---|---|---:|---|']
    for r in paired:
        if r['view'] in ('bank','path'):
            lines.append(f"| {r['panel']} | {r['view']} | {names.get(r['reference'],r['reference'])} | {r['mean_delta']:+.5f} | {r['wins']} {r['ties']} {r['losses']} |")
    ts=pd.DataFrame(transitions)
    if len(ts):
        g=ts[(ts.phase=='evaluation')&(ts.method==tuned)]
        lines+=['',f"冻结可撤销配置接受{len(g)}次局部改善，其中同时删除和增加连接{int(((g.deleted>0)&(g.added>0)).sum())}次；累计删除假内部连接{int(g.deleted_false.sum())}次、删除真内部连接{int(g.deleted_true.sum())}次、增加真内部连接{int(g.added_true.sum())}次。以上为路径上的事件次数，允许同一输入反复出现，不等于最终净收益。"]
        lines+=[f"接受的局部改善中，支撑F1上升{int((g.F1_delta>1e-10).sum())}次、持平{int((abs(g.F1_delta)<=1e-10).sum())}次、下降{int((g.F1_delta< -1e-10).sum())}次；从不完整变为完整{int((~g.exact_before & g.exact_after).sum())}次，从完整变为不完整{int((g.exact_before & ~g.exact_after).sum())}次。该表直接检查目标改善向拓扑恢复的转化。"]
        lines+=[f"单步新增未接受改善后，重构随后接受改善{int(g.after_stalled_addition.sum())}次；其中前次新增的有效界或结构完备性足以排除过门槛改善的有{int(g.after_certified_stalled_addition.sum())}次。这里比较的是只新增一条的WZZT域，不把它写成所有单候选删除替换算法的停滞证书；前次触时无界的情况也不作为已证明停滞。"]
    oracle=pd.read_csv(output/'analysis/oracle_objectives.csv')
    lines+=['','## 目标偏好与搜索不足分开','',
        '离线把真实支撑集合固定，以相同0至2权重界独立求解训练L1 LP，再加相同线路数代价。真拓扑只在冻结后的诊断中使用，不参与任何搜索或选参。比较门槛沿用1e-10加0.001倍终点MAE。错误终点的J低于真实结构最佳J，表明当前量测与目标确实更偏好该错误结构；错误终点J仍高于真实结构最佳J，则存在未利用的搜索改善空间。两者均不作总体可辨识性结论。','',
        '| 主面板方法 | 错误终点优于真结构 | 错误终点差于真结构 |','|---|---:|---:|']
    for method,g in oracle[oracle.panel=='main'].groupby('method'):
        lines.append(f"| {names.get(method,method)} | {int(g.wrong_beats_true_profile.sum())}/108 | {int(g.wrong_worse_than_true_profile.sum())}/108 |")
    lines+=['','逐例结果、各网络及条件、大小块错误、阶段阻断数、求解器状态见同目录CSV。输入是仓库既有四个论文启发的合成网络，重复条件相关，不视作120次独立随机网络实验。',
        '', '## 复现','', '```powershell','python -m research_experiments.revocable.run freeze outputs/revocable_NEW',
        'python -m research_experiments.revocable.run development outputs/revocable_NEW','python -m research_experiments.revocable.run select outputs/revocable_NEW',
        'python -m research_experiments.revocable.run evaluation outputs/revocable_NEW','python -m research_experiments.revocable.run analyze outputs/revocable_NEW','```','']
    (dest/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g=summary[(summary.phase=='evaluation')&(summary.panel=='main')&(summary.view=='bank')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    methods=list(g.method);x=np.arange(len(g))
    for ax,col,label in zip(axes,('F1','exact'),('Mean support F1','Exact recovery /108')):
        ax.bar(x,g[col],color=['#39739d' if m==tuned else '#9cb6c9' for m in methods]);ax.set_xticks(x,methods,rotation=38,ha='right',fontsize=8)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(dest/'comparison.png',dpi=180);plt.close(fig)
