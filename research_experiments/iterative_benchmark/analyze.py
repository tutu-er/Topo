"""Post-freeze topology scoring, paired mechanism accounting and LP audits."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import json
import shutil
from time import perf_counter
import numpy as np
import pandas as pd
from .run import ROOT,METHODS,read,write,sha,completion
from research_experiments.local_rebuild.model import canonical,compatible
from research_experiments.rebuild_benchmark.engine import prepare,loss
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile,markdown
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


def solver_records(branch):
    trajectory=branch['trajectory']
    events=trajectory.get('attempts',trajectory.get('events',[]))
    result=[]
    for event in events:
        for record in event.get('trials',[event]):
            if record.get('status') is not None:result.append(dict(stage=event.get('kind'),**record))
    event=branch.get('exit_event',{})
    if event.get('status') is not None:result.append(dict(stage='final_exit',**event))
    return result


def analyze_job(args):
    output,job=args;output=Path(output);protocol=read(output/'protocol.json');meta=read(output/'catalog.json')[job]
    common=read(output/'common'/job/'common.json');labels=common['labels'];n=len(labels)
    assert sha(output/'common'/job/'common.json')==protocol['references'][job]['common_sha256']
    assert {s:sha(meta[s]) for s in ('train','validation','test')}==protocol['input_hashes'][job]
    data={s:prepare(observed(meta[s],meta['N'] if s=='train' else None)) for s in ('train','validation','test')}
    truth=physical_truth(Path(meta['truth_network']),labels);cache={};rows=[];events=[];solver=[];checks=0
    base=dict(job=job,panel=meta['panel'],case=meta['case'],bundle=meta['bundle'],N=meta['N'],
        voltage_sigma=meta['voltage_sigma'],condition=meta['condition'])
    def verify(point):
        nonlocal checks
        family=canonical(point['supports'])
        assert len(family)==len(point['supports']) and all(compatible(a,b) for a,b in combinations(family,2))
        assert min([*point['r'],*point['x']],default=0.)>=-1e-9 and max([*point['r'],*point['x']],default=0.)<=2.+1e-9
        actual=loss(data['train'],point);assert abs(actual-point['objective'])<1e-8
        if family not in cache:cache[family]=independent_profile(data['train'],family)
        assert abs(cache[family]['objective']-actual)<2e-7
        checks+=5
    for method in METHODS:
        result=read(output/'jobs'/job/method/'result.json')
        assert result['protocol_sha256']==sha(output/'protocol.json') and not result['truth_used'] and not result['test_opened']
        assert not result['total_cap_exceeded'] and not result.get('post_LP_failures') and not result.get('LP_failures')
        assert result['input_hashes']=={s:protocol['input_hashes'][job][s] for s in ('train','validation')}
        assert abs(result['total_seconds']-result['fresh_seconds']-result['common_seconds'])<1e-8
        assert result['recovery_attempts']<=1
        for name,branch in result['branches'].items():
            for point in branch['trajectory']['path']:
                verify(point)
                if name=='masked':assert tuple(result['selected_block']) not in canonical(point['supports'])
            solver.extend(dict(**base,method=method,branch=name,**r) for r in solver_records(branch))
        views={**result['views']}
        for name,choice in result['choices_matched'].items():views['branch_'+name]=choice
        if method=='after':views['native_final']=result['native_selection']
        for name,choice in views.items():
            verify(choice['point']);metrics=score(choice['point'],data,truth,labels,canonical(common['pool']))
            assert abs(metrics['validation_mae']-choice['validation'])<1e-10
            rows.append(dict(**base,method=method,view=name,available=True,selected_source=choice.get('source'),
                fresh_seconds=result['fresh_seconds'],common_seconds=result['common_seconds'],total_seconds=result['total_seconds'],**metrics))
        for branch_name,branch in result['branches'].items():
            endpoint=branch['trajectory']['point'];verify(endpoint)
            rows.append(dict(**base,method=method,view='raw_endpoint' if branch_name=='baseline' else 'raw_'+branch_name,
                available=True,selected_source=branch_name,
                train_J=endpoint['objective']+.003*result['prefix']['reference_MAE']*sum(1<len(C)<n for C in endpoint['supports']),
                fresh_seconds=result['fresh_seconds'],common_seconds=result['common_seconds'],total_seconds=result['total_seconds'],
                **score(endpoint,data,truth,labels,canonical(common['pool']))))
        zeros=[dict(branch=name,**e) for name,b in result['branches'].items() for e in b['trajectory'].get('zero_events',[])]
        for e in zeros:
            if e['accepted']:assert e['after']<=e['before']+e['loss_tolerance']
        accepted=[e for e in zeros if e['accepted']]
        forward=[e for e in accepted if e['stage']!='final_exit']
        block=tuple(result['selected_block']) if result.get('selected_block') else None
        position={label:i for i,label in enumerate(labels)};truepos={tuple(sorted(position[v] for v in C)) for C in truth['clades']}
        rec=dict(**base,method=method,zero_checks=len(zeros),zero_release_events=len(accepted),
            zero_released_sets=sum(len(e['removed']) for e in accepted),forward_zero_events=len(forward),
            zero_unique_sets=len({tuple(C) for e in accepted for C in e['removed']}),
            forward_zero_sets=sum(len(e['removed']) for e in forward),
            stalled=bool(result.get('stalled')),deletion_triggered=result['recovery_attempts'],
            deletion_ranking_complete=result.get('ranking',{}).get('complete'),
            release_branch_run='release' in result['branches'],masked_branch_run='masked' in result['branches'],
            final_source=result['views']['final'].get('source'),
            release_adopted=result['views']['final'].get('source')=='release',mask_adopted=result['views']['final'].get('source')=='masked',
            selected_block=json.dumps(result.get('selected_block')),selected_block_size=len(block) if block else None,
            selected_block_is_true=block in truepos if block else None,
            selected_block_was_seed=block in canonical(result['prefix']['seeding']['seeds']) if block else None,
            fresh_seconds=result['fresh_seconds'],total_seconds=result['total_seconds'])
        events.append(rec)
        write(output/'analysis/jobs'/job/(method+'_zero_events.json'),zeros)
    historical=read(output/'common'/job/'historical.json')
    for view,choice in historical['choices']['strict_0.24'].items():
        if view not in ('bank','path'):continue
        verify(choice['point']);rows.append(dict(**base,method='historical300',view='final' if view=='bank' else 'own',available=True,
            selected_source=choice.get('origin'),fresh_seconds=historical['elapsed_seconds'],common_seconds=common['seconds'],
            total_seconds=historical['elapsed_seconds']+common['seconds'],
            **score(choice['point'],data,truth,labels,canonical(common['pool']))))
    record=dict(passed=True,job=job,analyzer_sha256=sha(Path(__file__)),checks=checks,independent_LPs=len(cache),
        profiles=[dict(family=family,**profile) for family,profile in cache.items()],rows=rows,events=events,solver=solver)
    write(output/'analysis/jobs'/job/'verified.json',record)
    return record


def scopes(frame):
    yield 'all',frame
    for panel,group in frame.groupby('panel'):yield panel,group


def paired(frame,left_method,left_view,right_method,right_view,name):
    left=frame[(frame.method==left_method)&(frame.view==left_view)].set_index('job')
    right=frame[(frame.method==right_method)&(frame.view==right_view)].set_index('job')
    jobs=left.index.intersection(right.index);records=[]
    for job in jobs:
        a,b=left.loc[job],right.loc[job]
        records.append(dict(job=job,panel=a.panel,case=a['case'],comparison=name,
            before_F1=a.clade_f1,after_F1=b.clade_f1,delta_F1=b.clade_f1-a.clade_f1,
            before_exact=bool(a.exact),after_exact=bool(b.exact),new_exact=bool(b.exact and not a.exact),lost_exact=bool(a.exact and not b.exact),
            before_validation=a.validation_mae,after_validation=b.validation_mae,delta_validation=b.validation_mae-a.validation_mae,
            delta_train=b.train_mae-a.train_mae,delta_test=b.test_mae-a.test_mae))
    return records


def aggregate(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(keys,key));row.update(inputs=len(g),F1=float(g.clade_f1.mean()),exact=int(g.exact.sum()),
            exact_rate=float(g.exact.mean()),train_MAE=float(g.train_mae.mean()),validation_MAE=float(g.validation_mae.mean()),test_MAE=float(g.test_mae.mean()),
            total_seconds=float(g.total_seconds.sum()),fresh_seconds=float(g.fresh_seconds.sum()),common_seconds=float(g.common_seconds.sum()),
            median_seconds=float(g.total_seconds.median()),max_seconds=float(g.total_seconds.max()))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(output):
    then=perf_counter();p=read(output/'protocol.json');assert completion(output)['returned']==240
    for method in METHODS:
        for name,digest in p['source_hashes'][method].items():assert sha(output/'sources'/method/name)==digest
    m=p['parent_manifest'];assert sha(m['path'])==m['sha256']
    for name,record in m['files'].items():assert sha(Path(m['path']).parent/name)==record['sha256']
    dest=output/'analysis';dest.mkdir(exist_ok=True);cached={};todo=[]
    for job in p['jobs']:
        file=dest/'jobs'/job/'verified.json'
        if file.exists() and read(file).get('analyzer_sha256')==sha(Path(__file__)):cached[job]=read(file)
        else:todo.append((str(output),job))
    with ProcessPoolExecutor(max_workers=6) as pool:
        for record in pool.map(analyze_job,todo):
            cached[record['job']]=record
            print('verified',len(cached),'/120',record['job'],flush=True)
    records=[cached[job] for job in p['jobs']]
    frame=pd.DataFrame([r for record in records for r in record['rows']]);events=pd.DataFrame([r for record in records for r in record['events']])
    solver=[r for record in records for r in record['solver']]
    pairs=[]
    comparisons=[('before','final','after','final','matched_before_after'),('before','own','after','own','matched_own'),
        ('historical300','final','before','final','budget_effect'),('before','final','after','zero_only','zero_path_comparison'),
        ('after','zero_only','after','without_mask','release_increment'),('after','without_mask','after','final','mask_increment'),
        ('after','zero_only','after','final','all_recovery_increment'),('after','native_final','after','final','selection_harmonization')]
    for args in comparisons:pairs.extend(paired(frame,*args))
    pairs=pd.DataFrame(pairs);summary=[];paired_summary=[];event_summary=[];mechanism=[]
    for scope,g in scopes(frame):
        a=aggregate(g,['method','view']);a.insert(0,'scope',scope);summary.append(a)
    for scope,g in scopes(pairs):
        for name,h in g.groupby('comparison'):
            d=h.delta_F1;paired_summary.append(dict(scope=scope,comparison=name,inputs=len(h),delta_F1=float(d.mean()),
                wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum()),
                new_exact=int(h.new_exact.sum()),lost_exact=int(h.lost_exact.sum()),validation_improved=int((h.delta_validation< -1e-12).sum()),
                topology_improved=int((d>1e-10).sum()),validation_improved_F1_worse=int(((h.delta_validation< -1e-12)&(d< -1e-10)).sum())))
    for scope,g in scopes(events):
        for method,h in g.groupby('method'):
            event_summary.append(dict(scope=scope,method=method,inputs=len(h),
                **{name:int(h[name].sum()) for name in ('zero_checks','zero_release_events','zero_released_sets','zero_unique_sets','forward_zero_events','forward_zero_sets',
                    'stalled','deletion_triggered','release_branch_run','masked_branch_run','release_adopted','mask_adopted')},
                inputs_with_zero=int((h.zero_release_events>0).sum()),inputs_with_forward_zero=int((h.forward_zero_events>0).sum())))
    summary=pd.concat(summary,ignore_index=True);paired_summary=pd.DataFrame(paired_summary);event_summary=pd.DataFrame(event_summary)
    detail=pairs.merge(events[events.method=='after'].drop(columns=['method','panel','case']),on='job',how='left')
    for scope,g in scopes(detail):
        for comparison,h in g.groupby('comparison'):
            for subset,selected in [('all',h),('zero_triggered',h[h.forward_zero_events>0]),
                ('deletion_triggered',h[h.deletion_triggered>0]),('mask_ran',h[h.masked_branch_run]),('mask_adopted',h[h.mask_adopted])]:
                if not len(selected):continue
                d=selected.delta_F1
                mechanism.append(dict(scope=scope,comparison=comparison,subset=subset,inputs=len(selected),delta_F1=float(d.mean()),
                    wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum()),
                    new_exact=int(selected.new_exact.sum()),lost_exact=int(selected.lost_exact.sum()),
                    validation_improved=int((selected.delta_validation< -1e-12).sum()),
                    validation_improved_F1_worse=int(((selected.delta_validation< -1e-12)&(d< -1e-10)).sum())))
    tables=dict(per_model=frame,summary=summary,cases=aggregate(frame,['panel','case','method','view']),
        conditions=aggregate(frame,['panel','N','voltage_sigma','method','view']),per_input_events=events,
        events=event_summary,paired_inputs=pairs,paired=paired_summary,mechanism_benefits=pd.DataFrame(mechanism),mechanism_inputs=detail)
    for name,table in tables.items():table.to_csv(dest/(name+'.csv'),index=False)
    write(dest/'solver_records.json',solver)
    solver_frame=pd.DataFrame([dict(job=r['job'],panel=r['panel'],case=r['case'],method=r['method'],branch=r['branch'],stage=r['stage'],
        status=r.get('status'),seconds=r.get('seconds'),gap=r.get('relative_gap'),bound=r.get('bound',r.get('dual_bound',r.get('final_bound'))),
        incumbent=r.get('objective',r.get('final_objective'))) for r in solver])
    solver_frame.to_csv(dest/'solver.csv',index=False)
    solver_summary=solver_frame.groupby(['method','status'],dropna=False).size().reset_index(name='calls');solver_summary.to_csv(dest/'solver_summary.csv',index=False)
    verification=dict(passed=True,returned=240,paired_inputs=120,checks=sum(r['checks'] for r in records),
        independent_LPs=sum(r['independent_LPs'] for r in records),source_hashes_verified=True,parent_files_preserved=len(m['files']),
        analyzer_sha256=sha(Path(__file__)),analysis_seconds=perf_counter()-then)
    write(dest/'verification.json',verification)
    write(dest/'batch_timing.json',dict(batch_wall_seconds=read(output/'state.json').get('batch_wall_seconds'),
        summed_fresh_worker_seconds={method:float(frame[(frame.method==method)&(frame.view=='final')].fresh_seconds.sum()) for method in METHODS},
        summed_method_seconds_including_shared_frontend={method:float(frame[(frame.method==method)&(frame.view=='final')].total_seconds.sum()) for method in METHODS},
        overlapping_method_workers=True,analysis_seconds=perf_counter()-then))
    delivery=ROOT/'docs'/output.name
    for file in dest.iterdir():
        if file.is_file():shutil.copy2(file,delivery/file.name)
    final=summary[(summary.view=='final')];main_pairs=paired_summary[paired_summary.comparison.isin(['matched_before_after','budget_effect','zero_path_comparison','release_increment','mask_increment','all_recovery_increment'])]
    text='# 同120输入：迭代零块释放与一次停滞恢复\n\n240/240新求解完成；原300秒结果仅作预算敏感性参照。每方法/输入总账上限4小时，共享RNJ前端成本计入；新版分支共享预算。所有MILP相对gap统一0.0001，单线程；参数未重调。\n\n'
    text+='## 最终结果\n\n'+markdown(final[['scope','method','inputs','F1','exact','exact_rate','total_seconds','fresh_seconds','median_seconds','max_seconds']])+'\n\n'
    text+='## 配对变化\n\n'+markdown(main_pairs)+'\n\n## 机制触发\n\n'+markdown(event_summary)+'\n\n'
    text+='zero_path_comparison比较旧算法与新版已计算的baseline延续，包含零块处理及其后续路径变化；新版baseline退出时间由分支共享预算分配，因此不是独立满预算零块消融。release_increment比较零块路径与加入非屏蔽恢复分支；mask_increment再比较加入屏蔽分支，均保留相同RNJ验证保护。触发、最终采用、F1变好和新增完整恢复分别报告；验证改善不等于拓扑改善。\n\n'
    text+='统一输出选择沿用旧方法：0.24收缩后联合重拟合、严格验证MAE、平分少内部集合、公共RNJ库及单点回退。native_final另存新版自身原选择，selection_harmonization列出两种选择口径差异。零块计数指新增显式清理事件；旧方法原有最终退出清理保留，其内部次数未伪记为新增前向清理。\n\n'
    text+='## 核验\n\n'+json.dumps(verification,ensure_ascii=False,indent=2)+'\n\n求解器status与gap只对应各候选域子问题；不存在全树最优或恢复保证。全输入逐例数据、时间、阶段事件、paired变化及求解状态见CSV/JSON；观测NPZ由catalog和哈希引用。\n'
    (delivery/'RESULTS.md').write_text(text,encoding='utf-8')
    write(output/'state.json',dict(stage='complete',returned=240,planned=240,verified=True))
    print(markdown(final[['scope','method','inputs','F1','exact','total_seconds']]),flush=True)


if __name__=='__main__':
    import sys
    analyze(Path(sys.argv[1]).resolve())
