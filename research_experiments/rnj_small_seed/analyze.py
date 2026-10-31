"""Independent loss/LP checks and attribution; truth is used only here."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,shutil,zipfile
import numpy as np,pandas as pd
from .run import ROOT,PARENT,read,write,sha
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile,markdown
from research_experiments.local_rebuild.model import canonical,compatible
from research_experiments.joint_five.model import remainder
from research_experiments.joint_five.analyze import size_errors
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def development_diagnostics(output):
    """Truth-based descriptions after the immutable predictive choice only."""
    from research_experiments.joint_exit.analyze import metrics
    output=Path(output);assert (output/'frozen.json').exists();p=read(output/'protocol.json');cat=read(output/'catalog.json');rows=[]
    for job in p['development']:
        labels=read(output/'common'/job/'result.json')['labels'];truth=physical_truth(Path(cat[job]['truth_network']),labels)
        for mode in p['modes']:
            r=read(output/'development'/job/mode/'result.json')
            for resolution in (.16,.24):
                for view in ('bank','path'):
                    q=r['choices'][f'strict_{resolution:g}'][view]
                    rows.append(dict(job=job,case=cat[job]['case'],mode=mode,resolution=resolution,view=view,
                        origin=q.get('origin',view),**metrics(q['point'],truth,labels)))
    dest=output/'analysis';dest.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(dest/'development_metrics.csv',index=False)
    frame.groupby(['resolution','view','mode']).agg(inputs=('job','size'),F1=('F1','mean'),exact=('exact','sum')).to_csv(dest/'development_summary.csv')
    print(frame[(frame.resolution==.24)&(frame['view']=='bank')].groupby('mode').agg(F1=('F1','mean'),exact=('exact','sum')).to_string(),flush=True)


def verify(args):
    output,job=args;output=Path(output);p=read(output/'protocol.json');f=read(output/'frozen.json');meta=read(output/'catalog.json')[job]
    common=read(output/'common'/job/'result.json');labels=common['labels'];n=len(labels);truth=physical_truth(Path(meta['truth_network']),labels)
    pos={v:i for i,v in enumerate(labels)};true={tuple(sorted(pos[v] for v in s)) for s in truth['clades']}
    data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
    assert sha(output/'common'/job/'result.json')==p['common_hashes'][job]
    rows=[];events=[];sizes=[];transitions=[];LP=[];cache={};checks=1
    modes=[f['mode']]+(['none','large'] if job in p['mechanism_controls'] else [])
    specs=[]
    for mode in modes:
        r=read(output/'evaluation'/job/mode/'result.json');t=r['trajectory'];q=r['exit'];e=r['event'];seed=t['seeding']['seeds']
        assert r['protocol_sha256']==sha(output/'protocol.json') and r['frozen_sha256']==sha(output/'frozen.json')
        assert not r['truth_used'] and not r['test_opened'] and r['common_sha256']==sha(output/'common'/job/'result.json');checks+=4
        archived_reference=read(PARENT/'evaluation'/job/'baseline20.json')['trajectory']['reference_MAE']
        assert abs(t['reference_MAE']-archived_reference)<1e-12;checks+=1
        assert q['audit']['valid'] and q['score']<=q['before_score']+1e-8;checks+=2
        if q['accepted']:assert q['incumbent_at']<=40;checks+=1
        if q['bound'] is not None:assert q['bound']<=q['score']+1e-8;checks+=1
        if mode.startswith('small'):
            assert len(seed)<=2 and all(2<=len(B)<=int(mode[-1]) for B in seed)
            assert all(not set(B)&set(C) for i,B in enumerate(seed) for C in seed[i+1:]);checks+=2
        if mode=='none':assert not seed;checks+=1
        before=t['path'][0]
        assert all(len(C) in (1,n) or list(C) in seed or tuple(C) in seed for C in before['supports'])
        assert remainder(before['supports'],n)==t['initial_maximum_remainder'];checks+=2
        for event in t['attempts']:
            if event['kind']=='coarse_wzzt':
                for trial in event['trials']:
                    assert trial['warm_start']['valid'] and trial['bound_consistent_with_known_upper'];checks+=2
                    for C in trial['added']:
                        indices=[B for B in trial['blocks'] if set(B)<=set(C)]
                        assert set(C)=={i for B in indices for i in B}
                        assert trial['minimum_child_blocks']<=len(indices)<len(trial['blocks']);checks+=2
                    if trial['accepted']:assert trial['before_score']-trial['score']>trial['threshold'];checks+=1
        if q['accepted']:
            old={tuple(C) for C in t['point']['supports']};new={tuple(C) for C in q['point']['supports']}
            transitions.append(dict(job=job,mode=mode,deleted_true=len((old-new)&true),deleted_false=len((old-new)-true),
                added_true=len((new-old)&true),added_false=len((new-old)-true)))
        attempts=t['attempts'];coarse=[a for a in attempts if a['kind']=='coarse_wzzt'];tail=[a for a in attempts if a['kind']=='joint_tail']
        added=[C for a in coarse for trial in a['trials'] if trial['accepted'] for C in trial['added']]
        penalty=.003*t['reference_MAE']
        known_rnj=min(point['objective']+penalty*sum(1<len(C)<n for C in point['supports']) for point in common['bank'])
        if q['bound'] is not None:assert q['bound']<=known_rnj+1e-8;checks+=1
        events.append(dict(job=job,panel=meta['panel'],case=meta['case'],mode=mode,seeds=len(seed),seed_correct=sum(tuple(B) in true for B in seed),
            seed_wrong=sum(tuple(B) not in true for B in seed),seed_cross=sum(any(not compatible(B,C) for C in true) for B in seed),initial_max=t['initial_maximum_remainder'],
            coarse_calls=len(coarse),coarse_MIP=sum(len(a['trials']) for a in coarse),coarse_accepted=sum(a['accepted'] for a in coarse),
            coarse_added_small=sum(len(C)<=5 for C in added),coarse_added_large=sum(len(C)>5 for C in added),
            tail_calls=len(tail),tail_MIP=sum(a.get('solve_started',False) for a in tail),tail_accepted=sum(a['accepted'] for a in tail),exit_accepted=q['accepted'],
            prefix_seconds=t['seconds'],exit_seconds=e['seconds'],construction=e['construction_seconds'],elapsed_seconds=r['elapsed_seconds'],
            pool_count=e['pool_count'],true_pool_recall=len(true&set(map(tuple,e['pool'])))/len(true),
            missing_small=sum(len(C)<=5 for C in true-set(map(tuple,e['pool']))),missing_large=sum(len(C)>5 for C in true-set(map(tuple,e['pool']))),
            known_RNJ_J=known_rnj,exit_excess_over_known_RNJ=q['score']-known_rnj,
            prefix_excess_over_known_RNJ=q['before_score']-known_rnj,exit_bound=q['bound'],exit_score=q['score'],
            maximum_remainder=t['maximum_remainder'],stop_reason=t['stop_reason'],exit_final_status=e.get('status')))
        for resolution in (.16,.24):
            key=f'strict_{resolution:g}'
            for view,point,choice in [('bank',r['choices'][key]['bank']['point'],r['choices'][key]['bank']),
                ('path',r['choices'][key]['path']['point'],r['choices'][key]['path']),
                ('prefix_bank',r['prefix_choices'][key]['bank']['point'],r['prefix_choices'][key]['bank']),
                ('prefix_path',r['prefix_choices'][key]['path']['point'],r['prefix_choices'][key]['path']),
                ('endpoint',q['point'],{}),('prefix_endpoint',t['point'],{})]:
                specs.append((mode,resolution,view,point,choice,True))
    oldfile=PARENT/'evaluation'/job/'compact_0.003/result.json';assert sha(oldfile)==p['parent_result_hashes'][job];checks+=1
    if job in p['mechanism_controls'] and (output/'warm_completion.json').exists():
        w=read(output/'warm'/job/'result.json');wp=read(output/'warm_protocol.json');q=w['exit'];primary=read(output/'evaluation'/job/f['mode']/'result.json')
        assert w['prefix_result_sha256']==sha(output/'evaluation'/job/f['mode']/'result.json')
        assert w['protocol_sha256']==sha(output/'warm_protocol.json') and not w['truth_used'] and not w['test_opened']
        assert w['source_hash']==wp['source_hashes']['research_experiments/rnj_small_seed/engine.py']
        assert set(map(tuple,w['event']['pool']))==set(map(tuple,primary['event']['pool']))
        assert q['audit']['valid'] and q['raw_score']<=q['warm_score']+1e-8 and q['score']<=q['before_score']+1e-8;checks+=7
        if q['bound'] is not None:assert q['bound']<=q['score']+1e-8;checks+=1
        if q['accepted']:assert q['incumbent_at']<=40;checks+=1
        original=next(x for x in events if x['mode']==f['mode']);event={**original,'mode':f['mode']+'_known_warm',
            'exit_seconds':w['event']['seconds'],'construction':w['event']['construction_seconds'],'exit_accepted':q['accepted'],
            'elapsed_seconds':None,'exit_excess_over_known_RNJ':q['score']-original['known_RNJ_J'],
            'exit_score':q['score'],'exit_bound':q['bound'],'exit_final_status':w['event'].get('status'),
            'warm_origin':q['warm_origin'],'warm_score':q['warm_score'],'prefix_reused':True}
        events.append(event)
        for resolution in (.16,.24):
            key=f'strict_{resolution:g}'
            for view,point,choice in [('bank',w['choices'][key]['bank']['point'],w['choices'][key]['bank']),
                ('path',w['choices'][key]['path']['point'],w['choices'][key]['path']),('endpoint',q['point'],{})]:
                specs.append((f['mode']+'_known_warm',resolution,view,point,choice,True))
    old=read(oldfile)['outputs']['40'];oldprefix=read(PARENT/'evaluation'/job/'baseline20.json')
    for resolution in (.16,.24):
        key=f'strict_{resolution:g}'
        for view,point,choice in [('bank',old['choices'][key]['bank']['point'],old['choices'][key]['bank']),
            ('path',old['choices'][key]['path']['point'],old['choices'][key]['path']),
            ('prefix_bank',oldprefix['choices'][key]['bank']['point'],oldprefix['choices'][key]['bank']),
            ('prefix_path',oldprefix['choices'][key]['path']['point'],oldprefix['choices'][key]['path']),
            ('endpoint',old['point'],{}),('prefix_endpoint',oldprefix['trajectory']['point'],{})]:
            specs.append(('full_rnj_archived',resolution,view,point,choice,False))
    for mode,resolution,view,point,choice,check_LP in specs:
        m=score(point,data,truth,labels,tuple(map(tuple,common['pool'])))
        rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],N=meta['N'],voltage_sigma=meta['voltage_sigma'],
            mode=mode,resolution=resolution,view=view,selected_origin=choice.get('origin',view),**m))
        if check_LP:
            family=canonical(point['supports'])
            if family not in cache:
                cache[family]=independent_profile(data['train'],family);LP.append(dict(job=job,mode=mode,family=family,**cache[family]))
            assert abs(cache[family]['objective']-point['objective'])<2e-7;checks+=1
        for s in size_errors(point,true,n):sizes.append(dict(job=job,panel=meta['panel'],mode=mode,resolution=resolution,view=view,**s))
    return dict(rows=rows,events=events,sizes=sizes,transitions=transitions,LP=LP,checks=checks)


def analyze(output):
    output=Path(output);p=read(output/'protocol.json');f=read(output/'frozen.json');completion=read(output/'evaluation_completion.json')
    assert completion['planned']==168 and completion['returned']==168
    for name,digest in p['source_hashes'].items():assert sha(output/'source'/name)==digest
    if (output/'warm_completion.json').exists():
        assert read(output/'warm_completion.json')['returned']==24
        for name,digest in read(output/'warm_protocol.json')['source_hashes'].items():assert sha(output/'warm_source'/name)==digest
    assert not f['truth_used'] and not f['evaluation_used'] and f['protocol_sha256']==sha(output/'protocol.json')
    development_diagnostics(output)
    from research_experiments.rnj_count.report import audit_inputs
    audit_inputs(output)
    rows=[];events=[];sizes=[];changes=[];LP=[];checks=0
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i,r in enumerate(executor.map(verify,[(str(output),j) for j in p['evaluation']]),1):
            rows+=r['rows'];events+=r['events'];sizes+=r['sizes'];changes+=r['transitions'];LP+=r['LP'];checks+=r['checks']
            if i%20==0:print('verified',i,'/120',flush=True)
    dest=output/'analysis';dest.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(dest/'per_model.csv',index=False)
    groups=['panel','resolution','view','mode'];aggregate=dict(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum'),internal=('internal','mean'),
        train=('train_mae','mean'),validation=('validation_mae','mean'),test=('test_mae','mean'))
    frame.groupby(groups).agg(**aggregate).to_csv(dest/'summary.csv')
    frame.groupby([*groups,'case']).agg(**aggregate).to_csv(dest/'cases.csv')
    controls=frame[frame.job.isin(p['mechanism_controls'])];controls.groupby(['resolution','view','mode']).agg(**aggregate).to_csv(dest/'controls.csv')
    frame.groupby([*groups,'N','voltage_sigma']).agg(**aggregate).to_csv(dest/'conditions.csv')
    pd.DataFrame(sizes).groupby([*groups,'size'])[['TP','FP','FN']].sum().to_csv(dest/'size_errors.csv')
    ev=pd.DataFrame(events);ev.to_csv(dest/'events.csv',index=False)
    ev.groupby(['panel','mode']).agg(inputs=('job','size'),seeds=('seeds','mean'),seed_correct=('seed_correct','sum'),seed_wrong=('seed_wrong','sum'),seed_cross=('seed_cross','sum'),
        basic_inputs=('coarse_MIP',lambda s:int((s>0).sum())),basic_calls=('coarse_calls','sum'),basic_accepted=('coarse_accepted','sum'),
        added_small=('coarse_added_small','sum'),added_large=('coarse_added_large','sum'),tail_inputs=('tail_calls',lambda s:int((s>0).sum())),
        tail_accepted=('tail_accepted','sum'),exit_accepted=('exit_accepted','sum'),prefix_seconds=('prefix_seconds','median'),exit_seconds=('exit_seconds','median'),
        construction=('construction','median'),elapsed_seconds=('elapsed_seconds','median'),pool_count=('pool_count','mean'),
        true_pool_recall=('true_pool_recall','mean'),missing_small=('missing_small','sum'),missing_large=('missing_large','sum')).to_csv(dest/'stage_summary.csv')
    pd.DataFrame(changes).to_csv(dest/'exit_changes.csv',index=False)
    pairs=[]
    for (panel,resolution,view),g in frame.groupby(['panel','resolution','view']):
        selected=g[g['mode']==f['mode']].set_index('job').clade_f1
        for mode in ('full_rnj_archived','none','large',f['mode']+'_known_warm'):
            reference=g[g['mode']==mode].set_index('job').clade_f1;ids=selected.index.intersection(reference.index);d=selected[ids]-reference[ids]
            if len(d):pairs.append(dict(panel=panel,resolution=resolution,view=view,reference=mode,inputs=len(d),delta=float(d.mean()),
                wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(pairs).to_csv(dest/'paired.csv',index=False)
    write(dest/'independent_LP.json',LP);write(dest/'verification.json',dict(passed=True,inputs=120,searches=168,warm_continuations=24 if (output/'warm_completion.json').exists() else 0,
        checks=checks,independent_LP=len(LP),failures=[],source_hash=sha(Path(__file__))))
    deliver(output)


def deliver(output):
    dest=ROOT/'docs/rnj_small_seed_20261031';dest.mkdir(exist_ok=True);analysis=output/'analysis'
    for path in analysis.iterdir():
        if path.suffix=='.csv' or path.name=='verification.json':shutil.copy2(path,dest/path.name)
    for name in ('protocol.json','frozen.json','development_completion.json','evaluation_completion.json','tuning.csv','input_audit.json',
        'baseline_preservation.json','warm_protocol.json','warm_completion.json'):
        if (output/name).exists():shutil.copy2(output/name,dest/name)
    summary=pd.read_csv(analysis/'summary.csv');controls=pd.read_csv(analysis/'controls.csv');stages=pd.read_csv(analysis/'stage_summary.csv')
    lines=['# RNJ少量小块 → WZZT骨架 → 联合末端 → 退出与收缩','',
        '旧完整RNJ方案原样保留于 docs/joint_exit_20261031（a46691ed）；本报告不覆盖旧结果。模型和边界见 PLAN.md。',
        '',f"冻结开发选择：{read(output/'frozen.json')['mode']}，严格验证；共同收缩0.24，另报0.16。正式168次搜索，覆盖120主/压力输入及24输入上的两种机制对照。",
        '', '## 最终验证输出','',markdown(summary[summary['view']=='bank']),
        '', '## 同一24输入机制对照','',markdown(controls[controls['view']=='bank']),
        '', '已知可行解热启动一行为额外24输入接续消融：复用主试验冻结前缀，同一候选池及名义40秒截止；不改写正式结果，也不是独立重跑完整前缀。',
        '', '## 搜索路径本身','',markdown(summary[summary['view']=='path']),
        '', '## 实际阶段调用及新增大/小块','',markdown(stages),
        '', '## 解释边界','',
        'none/large与选中small共用新的最大区域/大块优先调度及跨界退出池；旧full_rnj是同输入冻结归档参考。新方法对旧方法的差异不只来自RNJ种子，不能将全部变化归因于初始化。',
        '搜索中的原子块限制只在前向阶段成立；跨界窗口和所有终端对在退出中恢复部分被排除的集合。该有限池仍未覆盖全体树。最终收缩只删边，结构替换来自此前的联合选择。',
        '公共完整RNJ树始终参与各方法最终验证选择，路径结果单独报告，回退RNJ不得算作WZZT搜索纠错。',
        '时间预算包含种子筛选、建模、暖启动和原生求解；最终固定家族LP闭合与公共验证后处理另计。前缀及MIP可能有小幅软超出，详细耗时、触时状态和界保留。',
        '', '## 复现','', '```powershell',
        'python -m research_experiments.rnj_small_seed.run freeze outputs/rnj_small_seed_NEW',
        'python -m research_experiments.rnj_small_seed.run development outputs/rnj_small_seed_NEW',
        'python -m research_experiments.rnj_small_seed.run select outputs/rnj_small_seed_NEW',
        'python -m research_experiments.rnj_small_seed.run evaluation outputs/rnj_small_seed_NEW',
        'python -m research_experiments.rnj_small_seed.run analyze outputs/rnj_small_seed_NEW','```','']
    (dest/'README.md').write_text('\n'.join(lines),encoding='utf-8');(dest/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='utf-8')
    with zipfile.ZipFile(dest/'records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('source','development','evaluation','analysis','warm'):
            for file in (output/folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.csv','.log'):z.write(file,file.relative_to(output))
        for file in output.iterdir():
            if file.suffix in ('.json','.csv'):z.write(file,file.name)
        for file in Path(__file__).parent.glob('*.py'):z.write(file,'actual_driver/'+file.name)
        for name in ('engine.py','warm.py'):
            file=output/'warm_source/research_experiments/rnj_small_seed'/name
            if file.exists():z.write(file,file.relative_to(output))
    print(summary[summary['view']=='bank'].to_string(index=False),flush=True)
