"""Matched-budget analysis and independent LP verification by input."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import repeat
import json,shutil,zipfile,warnings
import numpy as np,pandas as pd
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import csr_matrix,eye,hstack,vstack
from .run import ROOT,PARENT,read,write,sha,cid
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_five.analyze import size_errors,supports
from research_experiments.local_rebuild.model import canonical
from research_experiments.local_rebuild.verify import independent_loss
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj_count.engine import count,criterion


def independent_profile(data,family):
    p,q,y=data.p[0],data.q[0],data.target[0];n=data.n
    z=np.array([[i in s for i in range(n)] for s in family],float)
    features=csr_matrix(np.column_stack([((a@row)[:,None]*row).ravel() for a in (p,q) for row in z]));m,k=features.shape;I=eye(m,format='csr')
    A=vstack([hstack([features,-I]),hstack([-features,-I])],format='csr');b=np.r_[y.ravel(),-y.ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        result=linprog(np.r_[np.zeros(k),np.ones(m)/m],A_ub=A,b_ub=b,bounds=[(0,2)]*k+[(0,None)]*m,
            method='highs',options={'threads':1})
    if not result.success:raise RuntimeError(result.message)
    violation=float(np.max(A@result.x-b,initial=0.))
    if violation>1e-7:raise RuntimeError('independent LP infeasible')
    return dict(objective=float(result.fun),violation=violation,status=int(result.status))


def metrics(point,truth,labels):
    predicted={frozenset(labels[i] for i in s) for s,r,x in zip(point['supports'],point['r'],point['x'])
        if 1<len(s)<len(labels) and max(r,x)>1e-9}
    true=truth['clades'];tp=len(predicted&true)
    return dict(F1=2*tp/(len(predicted)+len(true)) if predicted or true else 1.,exact=predicted==true,
        TP=tp,FP=len(predicted-true),FN=len(true-predicted),internal=len(predicted))


def ablation(output):
    p=read(output/'protocol.json');cat=read(output/'catalog.json');rows=[]
    assert (output/'frozen.json').exists()
    for job in p['ablation']:
        meta=cat[job];common=read(output/'common'/job/'result.json');labels=common['labels'];truth=physical_truth(Path(meta['truth_network']),labels)
        results={m:read(output/'ablation'/job/m/'result.json') for m in p['ablation_configs']}
        results.update({f'joint_{b}':read(output/'development'/job/f'baseline{b}.json') for b in (20,40)})
        for method,r in results.items():
            t=r['trajectory'];q=r['choices']['strict_0.16']['bank'];events=t['attempts']
            rows.append(dict(job=job,case=meta['case'],method=method,**metrics(q['point'],truth,labels),
                endpoint_F1=metrics(t['point'],truth,labels)['F1'],
                path_F1=metrics(r['choices']['strict_0.16']['path']['point'],truth,labels)['F1'],
                train=t['point']['objective'],seconds=t['seconds'],attempts=len(events),
                local_solves=sum(e['kind']=='revocable_local' and e.get('solve_started',False) for e in events),
                timeouts=sum(e.get('status')==9 for e in events),accepted=sum(e.get('accepted',False) for e in events),
                stop=t['stop_reason']))
    frame=pd.DataFrame(rows);dest=output/'analysis';dest.mkdir(exist_ok=True)
    frame.to_csv(dest/'budget_ablation.csv',index=False)
    summary=frame.groupby('method').agg(inputs=('job','size'),F1=('F1','mean'),exact=('exact','sum'),seconds=('seconds','median'),
        attempts=('attempts','mean'),local_solves=('local_solves','mean'),timeouts=('timeouts','sum'),accepted=('accepted','sum'),train=('train','mean'))
    summary=summary.join(frame.groupby('method')[['endpoint_F1','path_F1']].mean())
    summary.to_csv(dest/'budget_summary.csv');print(summary.to_string(),flush=True)
    return frame


def verify_job(args):
    directory,job=args;output=Path(directory);p=read(output/'protocol.json');frozen=read(output/'frozen.json');meta=read(output/'catalog.json')[job]
    common_file=output/'common'/job/'result.json';common=read(common_file);labels=common['labels'];n=len(labels)
    assert sha(common_file)==p['common_hashes'][job]
    data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
    truth=physical_truth(Path(meta['truth_network']),labels);pos={v:i for i,v in enumerate(labels)}
    true={tuple(sorted(pos[v] for v in s)) for s in truth['clades']};rows=[];errors=[];events=[];checks=0;profiles={};unique=[];states=[]
    choice=f"{frozen['selection']['rule']}_{frozen['selection']['resolution']:g}"
    specs=[];prefix=read(output/'evaluation'/job/'baseline20.json')
    for budget in (20,40):
        r=read(output/'evaluation'/job/f'baseline{budget}.json');t=r['trajectory']
        for view_name,selection in (('matched','strict_0.16'),('tuned',choice)):
            specs.append((f'baseline{budget}',view_name,r['choices'][selection],t['point'],t))
        for e in t['attempts']:
            if e.get('status') is not None:
                assert e['warm_start']['valid'] and e['bound_consistent_with_known_upper'];checks+=2
            events.append(dict(job=job,panel=meta['panel'],case=meta['case'],method=f'baseline{budget}',kind=e['kind'],status=e.get('status'),
                seconds=e.get('seconds'),accepted=e.get('accepted',False),candidate_count=e.get('candidate_count')))
    for config in frozen['evaluation_configs']:
        r=read(output/'evaluation'/job/cid(config)/'result.json');e=r['event']
        assert r['protocol_sha256']==sha(output/'protocol.json') and r['frozen_sha256']==sha(output/'frozen.json')
        assert r['common_sha256']==sha(common_file) and not r['truth_used'] and not r['test_opened'];checks+=4
        assert abs(e['penalty']-config['coefficient']*r['reference_MAE'])<1e-12
        assert set(map(tuple,e['old_pool']))<=set(map(tuple,e['pool']));checks+=2
        for budget,q in r['outputs'].items():
            assert q['audit']['valid']
            assert q['score']<=q['before_score']+1e-8
            if q['bound'] is not None:assert q['bound']<=q['score']+1e-8
            if q['accepted']:assert q['incumbent_at']<=float(budget)
            available=[q['before_score'],*[x['objective'] for x in e['events'] if x['at']<=float(budget)]]
            assert q['raw_score']<=min(available)+1e-8;checks+=5
            assert abs(independent_loss(data['train'].p[0],data['train'].q[0],data['train'].target[0],q['raw_point'])-q['raw_point']['objective'])<1e-8;checks+=1
            label='exit' if config==frozen['config'] else 'exit_matched_lambda'
            method=f'{label}{budget}'
            for view_name,selection in (('matched','strict_0.16'),('tuned',choice)):
                specs.append((method,view_name,q['choices'][selection],q['point'],None))
            events.append(dict(job=job,panel=meta['panel'],case=meta['case'],method=method,kind='pooled_exit',status=e.get('status'),
                seconds=e['seconds'],accepted=q['accepted'],candidate_count=e['pool_count'],bound=q['bound'],
                raw_score=q['raw_score'],score=q['score'],before=q['before_score'],incumbent_at=q['incumbent_at']))
            states.append(dict(job=job,method=method,budget=budget,before=metrics(prefix['trajectory']['point'],truth,labels),
                after=metrics(q['point'],truth,labels),accepted=q['accepted']))
    # Same-input historical comparison; no historical timing used as speedup proof.
    old=read(PARENT/'evaluation'/job/'lambda_0.003'/'result.json')
    specs.append(('historical_joint20','matched',dict(bank=old['selected_plain']['bank'],path=old['selected_plain']['path']),old['trajectory']['point'],old['trajectory']))
    for method,selection_view,choices,endpoint,trajectory in specs:
        for view,point in (('bank',choices['bank']['point']),('path',choices['path']['point']),('endpoint',endpoint)):
            m=score(point,data,truth,labels,tuple(map(tuple,common['pool'])))
            rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],N=meta['N'],condition=meta['condition'],
                voltage_sigma=meta['voltage_sigma'],method=method,selection_view=selection_view,view=view,**m,
                selected_origin=choices['bank'].get('origin','historical') if view=='bank' else view,
                search_seconds=trajectory['seconds'] if trajectory else float(method.replace('exit_matched_lambda','').replace('exit',''))))
            key=canonical(point['supports'])
            if key not in profiles:
                profile=independent_profile(data['train'],key);profiles[key]=profile;unique.append(dict(job=job,family=key,**profile))
            assert abs(profiles[key]['objective']-point['objective'])<2e-7;checks+=1
            for s in size_errors(point,true,n):errors.append(dict(job=job,panel=meta['panel'],case=meta['case'],method=method,selection_view=selection_view,view=view,**s))
    return dict(rows=rows,errors=errors,events=events,states=states,checks=checks,LP=unique)


def analyze(output):
    output=Path(output);p=read(output/'protocol.json');frozen=read(output/'frozen.json');dest=output/'analysis';dest.mkdir(exist_ok=True)
    assert read(output/'evaluation_completion.json')['returned']==120
    assert frozen['protocol_sha256']==sha(output/'protocol.json') and not frozen['truth_used'] and not frozen['evaluation_test_used']
    for name,digest in p['source_hashes'].items():assert sha(output/'source'/name)==digest
    from research_experiments.rnj_count.report import audit_inputs
    audit_inputs(output)
    ablation(output)
    rows=[];errors=[];events=[];states=[];checks=0;LP=[]
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i,r in enumerate(executor.map(verify_job,[(str(output),j) for j in p['evaluation']]),1):
            rows+=r['rows'];errors+=r['errors'];events+=r['events'];states+=r['states'];checks+=r['checks'];LP+=r['LP']
            write(output/'verification_progress.json',dict(planned=120,returned=i,checks=checks))
            if i%20==0:print('verified',i,'/120',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'per_model.csv',index=False)
    summary=frame.groupby(['panel','selection_view','view','method']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum'),
        internal=('internal','mean'),train=('train_mae','mean'),validation=('validation_mae','mean'),test=('test_mae','mean'),search_seconds=('search_seconds','median')).reset_index()
    summary.to_csv(dest/'summary.csv',index=False)
    frame.groupby(['panel','selection_view','view','method','case']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum')).to_csv(dest/'cases.csv')
    frame.groupby(['panel','selection_view','view','method','N','voltage_sigma']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum')).to_csv(dest/'conditions.csv')
    pd.DataFrame(errors).groupby(['panel','selection_view','view','method','size'])[['TP','FP','FN']].sum().to_csv(dest/'size_errors.csv')
    pd.DataFrame(events).to_csv(dest/'solver_events.csv',index=False)
    write(dest/'raw_transitions.json',states);write(dest/'independent_LP.json',LP)
    pairs=[]
    for (panel,sel,view),g in frame.groupby(['panel','selection_view','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for a,b in [('exit20','baseline20'),('exit40','baseline40'),('exit40','exit20')]:
            d=pivot[a]-pivot[b];pairs.append(dict(panel=panel,selection_view=sel,view=view,comparison=f'{a}-{b}',
                delta=float(d.mean()),wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(pairs).to_csv(dest/'paired.csv',index=False)
    write(dest/'verification.json',dict(passed=True,inputs=120,checks=checks,independent_LP=len(LP),failures=[],
        analysis_source_sha256=sha(Path(__file__)),information_completed=(output/'information_completion.json').exists()))
    delivery=ROOT/'docs/joint_exit_20261031';delivery.mkdir(exist_ok=True)
    for f in dest.iterdir():
        if f.suffix=='.csv' or f.name=='verification.json':shutil.copy2(f,delivery/f.name)
    for name in ('protocol.json','frozen.json','tuning_grid.csv','input_audit.json','information_protocol.json','information_completion.json'):
        if (output/name).exists():shutil.copy2(output/name,delivery/name)
    report(output,delivery,summary,frame)
    with zipfile.ZipFile(delivery/'records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('source','development','evaluation','ablation'):
            for file in (output/directory).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.log'):z.write(file,file.relative_to(output))
        for file in dest.iterdir():
            if file.suffix in ('.json','.csv'):z.write(file,file.relative_to(output))
        for name in ('protocol.json','frozen.json','catalog.json','information_protocol.json','information_completion.json'):
            if (output/name).exists():z.write(output/name,name)
        for file in Path(__file__).parent.glob('*.py'):z.write(file,'actual_driver/'+file.name)
    print(summary[(summary.view=='bank')].to_string(index=False),flush=True)


def report(output,dest,summary,frame):
    frozen=read(output/'frozen.json');verification=read(output/'analysis/verification.json')
    lines=['# 多区域联合加入退出机制及末端信息诊断','',
        f"开发16输入、6种退出配置；冻结{frozen['config']}和{frozen['selection']}。正式120输入全部返回，独立重解{verification['independent_LP']}个家族，核查{verification['checks']}项。",'',
        '## 主面板及压力面板','',
        '| 面板 | 共同选模口径 | 方法 | F1 | 完整恢复 | 平均内部连接数 |','|---|---|---|---:|---:|---:|']
    for _,r in summary[summary.view=='bank'].iterrows():
        lines.append(f'| {r.panel} | {r.selection_view} | {r.method} | {r.F1:.4f} | {int(r.exact)}/{int(r.inputs)} | {r.internal:.3f} |')
    lines+=['','matched对所有方法统一使用严格最小验证MAE及0.16收缩。tuned对所有方法统一使用冻结选模规则和阈值，不能把仅改变后处理的收益算成退出搜索收益。历史行仅作同输入参照；新的baseline20和baseline40均重新运行。',
        '','## 时间和扫描消融','',pd.read_csv(output/'analysis/budget_summary.csv').to_markdown(index=False),
        '','8个预设开发输入，同一lambda=0.003。regional40_2相对regional20_2改变总预算；regional40_5在同为40秒时改变单次MIP时间，允许求解深度与覆盖数量发生取舍。均使用严格0.16共同选模，没有根据这些F1重新调参。',
        '','## 退出模型与预算解释','',
        '先运行上轮多区域联合20秒，再将临时内部支撑和候选池放入同一个MILP重选。候选池包含旧联合域与前缀解，全部响应系数联合重新拟合；单点和全集基础项保留，已核实物理先验才强制保留，本批无额外先验。未增加正阻抗下界。',
        '20/40秒是同一次池MILP的壁钟检查点，只使用该时刻已经找到的可行结构及有效界。连续家族闭合重拟合与验证选模作为所有方法的共同后处理，不计为额外结构搜索，其效果单独核查。20秒前缀已用尽预算的输入，20秒增强版本保留前缀。原20秒算法可能有小幅软预算超出。40秒原方法使用新LP bank重新运行，没有读取退出候选。',
        '前缀使用历史已冻结的lambda=0.003，退出阶段lambda由开发观测选择。只对退出阶段的同一个目标核查下降，不声称不同惩罚目标跨阶段单调。求解上下界只针对明确候选池，不覆盖所有可能树。旧单区域方法与新池选择的域也不等同。',
        '','## 末端信息','']
    path=output/'analysis/terminal_summary.csv'
    if path.exists():
        lines+=[pd.read_csv(path).to_markdown(index=False),'',
            '诊断已知真实外部结构，完整枚举至多5终端内的全部局部层级，包含区域根边可有可无。它是oracle条件性诊断，不是部署结果，也没有参与开发选择。linear用干净P/Q与真实R/X合成精确线性响应；clean_ac用干净AC量测；noisy用实际带噪量测。',
            'linear的错误家族近似零剖面间隙提示该设计下的区分不足；clean_ac相对linear的下降体现线性响应模型偏差；noisy进一步叠加量测噪声和有限样本影响。近零间隙按数值容差报告，不作为普遍不可辨识定理。投影后信号与残差比仅是几何诊断，不是显著性概率。',
            '精确反例：P每列都是同一个时间序列，Q=0，R_true=0.1I+0.2 z_12 z_12^T；星形R_star=diag(0.5,0.5,0.1)。两者对该P产生完全相同的Y，且系数都在0至2内。改用P=I时两者响应不同。因此有条件性的固有信息不足，但不能据此断言本批所有末端线都不可辨识。']
    lines+=['','## 范围与复现','',
        '四个网络为仓库论文启发的合成算例；108主条件与12压力条件复用既有归档，不视作120个独立随机拓扑。正式矩阵未参与本次调参，但此前已分析过，不称新盲测。生产默认未修改。原单区域20秒的历史结果在上一报告中，本批时间消融仅覆盖预设8输入。',
        '', '```powershell','python -m research_experiments.joint_exit.run freeze outputs/joint_exit_NEW',
        'python -m research_experiments.joint_exit.run ablation outputs/joint_exit_NEW','python -m research_experiments.joint_exit.run development outputs/joint_exit_NEW',
        'python -m research_experiments.joint_exit.run select outputs/joint_exit_NEW','python -m research_experiments.joint_exit.run evaluation outputs/joint_exit_NEW',
        'python -m research_experiments.joint_exit.run information outputs/joint_exit_NEW','python -m research_experiments.joint_exit.run analyze outputs/joint_exit_NEW','```','']
    (dest/'README.md').write_text('\n'.join(lines),encoding='utf-8')
