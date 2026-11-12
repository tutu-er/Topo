"""Independent verification of 300-second paired runs and short/long controls."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil,zipfile
import numpy as np,pandas as pd
from .run import ROOT,PARENT,read,write,sha
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile,markdown
from research_experiments.local_rebuild.model import canonical,compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def verify(args):
    output,job=args;output=Path(output);p=read(output/'long_protocol.json');meta=read(output/'catalog.json')[job]
    common=read(output/'common'/job/'result.json');labels=common['labels'];n=len(labels);truth=physical_truth(Path(meta['truth_network']),labels)
    positions={v:i for i,v in enumerate(labels)};true={tuple(sorted(positions[v] for v in C)) for C in truth['clades']}
    signature=lambda point:frozenset(tuple(C) for C,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(C)<n and max(r,x)>1e-9)
    rnj_trees={signature(point) for point in common['bank']};frontend=set(map(tuple,common['pool']))
    data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
    rows=[];events=[];LP=[];cache={};checks=0
    for method in p['methods']:
        r=read(output/'long'/job/method/'result.json');t=r['trajectory'];q=r['exit'];e=r['event']
        assert r['protocol_sha256']==sha(output/'long_protocol.json') and r['common_sha256']==sha(output/'common'/job/'result.json')
        assert not r['truth_used'] and not r['test_opened'] and r['limits']==p['limits'];checks+=4
        assert q['audit']['valid'] and q['score']<=q['before_score']+1e-8;checks+=2
        if q['accepted']:assert q['incumbent_at']<=p['limits']['total'];checks+=1
        if q['bound'] is not None:assert q['bound']<=q['score']+1e-8;checks+=1
        penalty=.003*t['reference_MAE'];known=min(v['objective']+penalty*sum(1<len(C)<n for C in v['supports']) for v in common['bank'])
        assert q['raw_score']<=q['warm_score']+1e-8;checks+=1
        available=[q['warm_score'],*[v['objective'] for v in e.get('events',[]) if v['at']<=p['limits']['total']]]
        assert q['raw_score']<=min(available)+1e-8;checks+=1
        if q['bound'] is not None:assert q['bound']<=known+1e-8;checks+=1
        attempts=t['attempts'];coarse=[a for a in attempts if a['kind'] in ('coarse_wzzt','wzzt_extension')]
        tail=[a for a in attempts if a['kind'] in ('joint_tail','wzzt_tail_joint')]
        seeds=t.get('seeding',{}).get('seeds',[]);zero_seeds=set()
        for point in t['path']:
            weights={tuple(C):max(r,x) for C,r,x in zip(point['supports'],point['r'],point['x'])}
            zero_seeds.update(tuple(B) for B in seeds if tuple(B) in weights and weights[tuple(B)]<=1e-9)
        for a in attempts:
            trials=a['trials'] if a['kind']=='coarse_wzzt' else [a]
            for v in trials:
                if v.get('status') is not None:
                    assert v['warm_start']['valid'] and v['bound_consistent_with_known_upper'];checks+=2
        events.append(dict(job=job,panel=meta['panel'],case=meta['case'],method=method,basic_calls=len(coarse),tail_calls=len(tail),
            prefix_seconds=t['seconds'],exit_seconds=e['seconds'],construction=e['construction_seconds'],elapsed_seconds=r['elapsed_seconds'],
            warm_origin=q['warm_origin'],exit_accepted=q['accepted'],exit_status=e.get('status'),exit_score=q['score'],bound=q['bound'],
            known_RNJ_J=known,exit_excess_over_known_RNJ=q['score']-known,pool_count=e['pool_count'],stop_reason=t['stop_reason'],
            true_pool_recall=len(true&set(map(tuple,e['pool'])))/len(true),pool_complete_truth=true<=set(map(tuple,e['pool'])),
            missing_small=sum(len(C)<=5 for C in true-set(map(tuple,e['pool']))),missing_large=sum(len(C)>5 for C in true-set(map(tuple,e['pool']))),
            retained_zero_seeds=len(zero_seeds),retained_zero_cross_seeds=sum(any(not compatible(B,C) for C in true) for B in zero_seeds)))
        for resolution in (.16,.24):
            key=f'strict_{resolution:g}'
            for view,point,choice in [('bank',r['choices'][key]['bank']['point'],r['choices'][key]['bank']),
                ('path',r['choices'][key]['path']['point'],r['choices'][key]['path']),
                ('prefix_bank',r['prefix_choices'][key]['bank']['point'],r['prefix_choices'][key]['bank']),
                ('prefix_path',r['prefix_choices'][key]['path']['point'],r['prefix_choices'][key]['path']),
                ('endpoint',q['point'],{}),('prefix_endpoint',t['point'],{})]:
                m=score(point,data,truth,labels,tuple(map(tuple,e['pool'])))
                rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],N=meta['N'],voltage_sigma=meta['voltage_sigma'],
                    method=method,resolution=resolution,view=view,origin=choice.get('origin',view),topology_in_common_RNJ_bank=signature(point) in rnj_trees,
                    outside_frontend_true=len((set(signature(point))&true)-frontend),outside_frontend_false=len((set(signature(point))-true)-frontend),**m))
                family=canonical(point['supports'])
                if family not in cache:
                    cache[family]=independent_profile(data['train'],family);LP.append(dict(job=job,family=family,**cache[family]))
                assert abs(cache[family]['objective']-point['objective'])<2e-7;checks+=1
    return dict(rows=rows,events=events,LP=LP,checks=checks)


def analyze(output):
    output=Path(output);p=read(output/'long_protocol.json');complete=read(output/'long_completion.json')
    assert complete['returned']==240 and not complete['failures']
    for name,digest in p['source_hashes'].items():assert sha(output/'long_source'/name)==digest
    rows=[];events=[];LP=[];checks=0
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i,r in enumerate(executor.map(verify,[(str(output),j) for j in p['jobs']]),1):
            rows+=r['rows'];events+=r['events'];LP+=r['LP'];checks+=r['checks']
            if i%20==0:print('long verified',i,'/120',flush=True)
    dest=output/'analysis';dest.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(dest/'long_per_model.csv',index=False)
    agg=dict(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum'),internal=('internal','mean'),train=('train_mae','mean'),validation=('validation_mae','mean'),test=('test_mae','mean'))
    groups=['panel','resolution','view','method'];frame.groupby(groups).agg(**agg).to_csv(dest/'long_summary.csv')
    frame.groupby([*groups,'case']).agg(**agg).to_csv(dest/'long_cases.csv')
    frame.groupby([*groups,'N','voltage_sigma']).agg(**agg).to_csv(dest/'long_conditions.csv')
    pd.DataFrame(events).to_csv(dest/'long_events.csv',index=False)
    short=pd.read_csv(dest/'per_model.csv');short['method']=short['mode'].map(lambda s:'short_'+s);long=frame.copy();long['method']='long_'+long['method']
    joined=pd.concat([long,short],ignore_index=True);joined.to_csv(dest/'budget_per_model.csv',index=False)
    joined.groupby(groups).agg(**agg).to_csv(dest/'budget_summary.csv')
    controls=joined[joined.job.isin(read(output/'protocol.json')['mechanism_controls'])]
    controls.groupby(['resolution','view','method']).agg(**agg).to_csv(dest/'long_controls.csv')
    pairs=[]
    for (panel,resolution,view),g in joined.groupby(['panel','resolution','view']):
        for a,b in [('long_small3','long_full_rnj'),('long_small3','short_small3'),('long_small3','short_small3_known_warm'),('long_full_rnj','short_full_rnj_archived')]:
            aa=g[g.method==a].set_index('job').clade_f1;bb=g[g.method==b].set_index('job').clade_f1;ids=aa.index.intersection(bb.index)
            if len(ids):
                d=aa[ids]-bb[ids];pairs.append(dict(panel=panel,resolution=resolution,view=view,comparison=a+'-'+b,inputs=len(d),delta=float(d.mean()),
                    wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(pairs).to_csv(dest/'long_paired.csv',index=False)
    write(dest/'long_independent_LP.json',LP);write(dest/'long_verification.json',dict(passed=True,inputs=120,searches=240,checks=checks,independent_LP=len(LP),failures=[],source_hash=sha(Path(__file__))))
    delivery=ROOT/'docs/rnj_small_seed_20261031'
    for path in dest.iterdir():
        if path.name.startswith(('long_','budget_')) and (path.suffix=='.csv' or path.name=='long_verification.json'):shutil.copy2(path,delivery/path.name)
    for name in ('long_protocol.json','long_completion.json'):shutil.copy2(output/name,delivery/name)
    table=pd.read_csv(dest/'long_summary.csv');control=pd.read_csv(dest/'long_controls.csv')
    text=['# 300秒长时对照','', '单次基础MILP30秒、前缀180秒、总结构预算300秒、LP30秒；新旧方案均使用已有最好可行解热启动。',
        '本页small3是“最大直属区域、半数块优先”的额外调度变体；严格沿用原始基础WZZT的24输入检查另见 ORIGINAL_BASE_RESULTS.md，不能将两者混称为同一实现。',
        '', '## 全120输入最终输出','',markdown(table[table['view']=='bank']),
        '', '## 预设24输入时间与热启动对照','',markdown(control[control['view']=='bank']),
        '', '## 自身路径','',markdown(table[table['view']=='path']),
        '', '长时和短时均保留原始记录。短时small3_known_warm与长时small3的24输入比较使用相同搜索规则、退出池构造及已知解热启动；主要变化是时间上限。',
        '长短预算分别重跑完整前缀，终点改变时实际退出池也可能改变；相同的是候选构造规则，不声称两次实际池完全相同或互相包含。',
        '已知解热启动路径包含完整RNJ候选，需结合origin及topology_in_common_RNJ_bank区分候选库位置与结构发现。',
        '新旧长时方案候选域仍不同；本报告给出相同预算的整体流程比较，不把全部变化归因于RNJ初始化。MIP最优界仅覆盖对应候选域。', '']
    (delivery/'LONG_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    with zipfile.ZipFile(delivery/'long_records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('long_source','long'):
            for file in (output/folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.log'):z.write(file,file.relative_to(output))
        for file in dest.iterdir():
            if file.name.startswith(('long_','budget_')):z.write(file,file.relative_to(output))
        for name in ('long_protocol.json','long_completion.json'):z.write(output/name,name)
        z.write(Path(__file__),'actual_driver/longanalyze.py')
    print(table[table['view']=='bank'].to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('path');a=parser.parse_args();analyze(Path(a.path).resolve())
