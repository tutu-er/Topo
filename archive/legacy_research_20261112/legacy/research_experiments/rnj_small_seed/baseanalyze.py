"""Verify and compare the original-foundation 24-input control."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil,zipfile
import pandas as pd
from .run import ROOT,read,write,sha
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile,markdown
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def verify(args):
    output,job=args;output=Path(output);r=read(output/'base'/job/'result.json');p=read(output/'base_protocol.json');meta=read(output/'catalog.json')[job]
    common=read(output/'common'/job/'result.json');labels=common['labels'];truth=physical_truth(Path(meta['truth_network']),labels)
    data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
    assert r['protocol_sha256']==sha(output/'base_protocol.json') and r['common_sha256']==sha(output/'common'/job/'result.json')
    assert not r['truth_used'] and not r['test_opened'] and r['trajectory']['strategy']=='original'
    q=r['exit'];e=r['event'];t=r['trajectory'];checks=5;cache={};LP=[];rows=[]
    assert q['audit']['valid'] and q['score']<=q['before_score']+1e-8
    available=[q['warm_score'],*[x['objective'] for x in e.get('events',[]) if x['at']<=p['limits']['total']]]
    assert q['raw_score']<=min(available)+1e-8;checks+=3
    if q['accepted']:assert q['incumbent_at']<=p['limits']['total'];checks+=1
    if q['bound'] is not None:assert q['bound']<=q['score']+1e-8;checks+=1
    for a in t['attempts']:
        assert a['kind']!='coarse_wzzt';checks+=1
        if a.get('status') is not None:assert a['warm_start']['valid'] and a['bound_consistent_with_known_upper'];checks+=2
    for resolution in (.16,.24):
        key=f'strict_{resolution:g}'
        for view,point,choice in [('bank',r['choices'][key]['bank']['point'],r['choices'][key]['bank']),
            ('path',r['choices'][key]['path']['point'],r['choices'][key]['path']),
            ('prefix_bank',r['prefix_choices'][key]['bank']['point'],r['prefix_choices'][key]['bank']),
            ('prefix_path',r['prefix_choices'][key]['path']['point'],r['prefix_choices'][key]['path']),
            ('endpoint',q['point'],{}),('prefix_endpoint',t['point'],{})]:
            rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],resolution=resolution,view=view,method='long_small3_original_wzzt',
                origin=choice.get('origin',view),**score(point,data,truth,labels,tuple(map(tuple,e['pool'])))))
            family=canonical(point['supports'])
            if family not in cache:
                cache[family]=independent_profile(data['train'],family);LP.append(dict(job=job,family=family,**cache[family]))
            assert abs(cache[family]['objective']-point['objective'])<2e-7;checks+=1
    event=dict(job=job,case=meta['case'],basic_calls=sum(a['kind']=='wzzt_extension' for a in t['attempts']),tail_calls=sum(a['kind']=='joint_tail' for a in t['attempts']),
        prefix_seconds=t['seconds'],exit_seconds=e['seconds'],exit_status=e.get('status'),exit_accepted=q['accepted'],warm_origin=q['warm_origin'],
        elapsed_seconds=r['elapsed_seconds'],pool_count=e['pool_count'],stop_reason=t['stop_reason'])
    return dict(rows=rows,LP=LP,checks=checks,event=event)


def analyze(output):
    output=Path(output);p=read(output/'base_protocol.json');assert read(output/'base_completion.json')['returned']==24
    for name,digest in p['source_hashes'].items():assert sha(output/'base_source'/name)==digest
    rows=[];LP=[];events=[];checks=0
    with ProcessPoolExecutor(max_workers=8) as executor:
        for r in executor.map(verify,[(str(output),j) for j in p['jobs']]):rows+=r['rows'];LP+=r['LP'];events.append(r['event']);checks+=r['checks']
    dest=output/'analysis';frame=pd.DataFrame(rows);frame.to_csv(dest/'base_per_model.csv',index=False);pd.DataFrame(events).to_csv(dest/'base_events.csv',index=False)
    long=pd.read_csv(dest/'budget_per_model.csv');other=long[long.job.isin(p['jobs'])];joined=pd.concat([other,frame],ignore_index=True)
    agg=dict(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum'),internal=('internal','mean'),train=('train_mae','mean'),validation=('validation_mae','mean'),test=('test_mae','mean'))
    joined.groupby(['resolution','view','method']).agg(**agg).to_csv(dest/'base_comparison.csv')
    joined.groupby(['resolution','view','method','case']).agg(**agg).to_csv(dest/'base_cases.csv')
    write(dest/'base_independent_LP.json',LP);write(dest/'base_verification.json',dict(passed=True,inputs=24,checks=checks,independent_LP=len(LP),failures=[],source_hash=sha(Path(__file__))))
    delivery=ROOT/'docs/rnj_small_seed_20261031';comparison=pd.read_csv(dest/'base_comparison.csv')
    for file in dest.iterdir():
        if file.name.startswith('base_') and (file.suffix=='.csv' or file.name=='base_verification.json'):shutil.copy2(file,delivery/file.name)
    for name in ('base_protocol.json','base_completion.json'):shutil.copy2(output/name,delivery/name)
    text=['# 保持原始基础WZZT的24输入对照','',
        '该检查在发现“半数直属块优先”是额外限制后追加；不改写既有结果，不按正式F1选择参数。沿用同一small3种子、300秒总预算、0.003惩罚及0.16/0.24收缩。',
        '基础步骤直接调用已有rnj_count.original_step：自由选择一个新相容集合，不要求覆盖半数直属块，也不强制以RNJ小块为不可拆原子；RNJ种子仍为可在后续退出的临时层级。',
        '', '## 最终验证输出','',markdown(comparison[comparison['view']=='bank']),
        '', '## 自身路径','',markdown(comparison[comparison['view']=='path']),
        '', '全部行仅为四网络、两个bundle、三档噪声、N48的固定24输入；不外推成108主面板结果。路径可能包含完整RNJ热启动回退，需结合来源解释。','']
    (delivery/'ORIGINAL_BASE_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    with zipfile.ZipFile(delivery/'base_records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('base_source','base'):
            for file in (output/folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.log'):z.write(file,file.relative_to(output))
        for file in dest.iterdir():
            if file.name.startswith('base_'):z.write(file,file.relative_to(output))
        for name in ('base_protocol.json','base_completion.json'):z.write(output/name,name)
        z.write(Path(__file__),'actual_driver/baseanalyze.py')
    print(comparison[(comparison['view']=='bank')&(comparison.resolution==.24)].to_string(index=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('path');a=parser.parse_args();analyze(Path(a.path).resolve())
