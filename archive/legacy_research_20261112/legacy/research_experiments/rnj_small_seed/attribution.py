"""Post-freeze role/candidate diagnostics; no optimization or parameter choice."""
from pathlib import Path
import pandas as pd
from .run import PARENT,read,write,sha
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.local_rebuild.model import compatible


def derive(output):
    output=Path(output);p=read(output/'long_protocol.json');cat=read(output/'catalog.json');rows=[];oracle_rows=[]
    oracle_audits={q['job']:q['sha256'] for q in read(PARENT/'analysis/information_run_audit.json')['jobs']}
    for job in p['jobs']:
        meta=cat[job];common=read(output/'common'/job/'result.json');labels=common['labels'];pos={v:i for i,v in enumerate(labels)}
        true={tuple(sorted(pos[v] for v in C)) for C in physical_truth(Path(meta['truth_network']),labels)['clades']}
        r=read(output/'long'/job/'small3/result.json');t=r['trajectory'];seeds=[tuple(C) for C in t['seeding']['seeds']]
        row=dict(job=job,panel=meta['panel'],case=meta['case'],seeds=len(seeds),seed_true=sum(C in true for C in seeds),
            seed_false=sum(C not in true for C in seeds),seed_cross=sum(any(not compatible(C,D) for D in true) for C in seeds),
            large_trials=0,accepted_large=0,truth_excluded_by_priority=0,accepted_while_truth_excluded=0,
            added_true=0,added_false_cross=0,added_false_compatible=0,added_small=0,added_large=0)
        for event in t['attempts']:
            if event['kind']!='coarse_wzzt':continue
            for trial in event['trials']:
                if trial['large_first']:
                    available=[]
                    for C in true:
                        selected=[B for B in trial['blocks'] if set(B)<=set(C)]
                        if set(C)=={i for B in selected for i in B} and 2<=len(selected)<len(trial['blocks']):available.append(len(selected))
                    excludes=bool(available) and not any(k>=trial['minimum_child_blocks'] for k in available)
                    row['large_trials']+=1;row['accepted_large']+=int(trial['accepted'])
                    row['truth_excluded_by_priority']+=int(excludes);row['accepted_while_truth_excluded']+=int(excludes and trial['accepted'])
                if trial['accepted']:
                    for C in map(tuple,trial['added']):
                        row['added_small' if len(C)<=5 else 'added_large']+=1
                        if C in true:row['added_true']+=1
                        elif any(not compatible(C,D) for D in true):row['added_false_cross']+=1
                        else:row['added_false_compatible']+=1
        rows.append(row)
        # The previous oracle's noisy true_profile is exactly the full true
        # family LP: exterior true supports plus every true support inside V.
        file=PARENT/'information'/job/'result.json';assert sha(file)==oracle_audits[job]
        info=read(file);values=[v['true_profile_MAE'] for v in info['rows'] if v['regime']=='noisy']
        assert values and max(values)-min(values)<1e-10
        reference=t['reference_MAE'];true_J=values[0]+.003*reference*len(true);tolerance=2e-7
        records=[(method,read(output/'long'/job/method/'result.json')) for method in ('full_rnj','small3')]
        if (output/'base_completion.json').exists() and (output/'base'/job/'result.json').exists():
            records.append(('small3_original_wzzt',read(output/'base'/job/'result.json')))
        for method,record in records:
            q=record['exit'];point=q['point'];pool=set(map(tuple,record['event']['pool']));in_pool=true<=pool
            signature={tuple(C) for C,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(C)<len(labels) and max(r,x)>1e-9}
            if in_pool and q['bound'] is not None:assert q['bound']<=true_J+tolerance
            internal=sum(1<len(C)<len(labels) for C in point['supports'])
            oracle_rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],method=method,true_MAE=values[0],true_J=true_J,
                incumbent_J=q['score'],incumbent_MAE=point['objective'],incumbent_internal=internal,true_internal=len(true),
                wrong=signature!=true,true_in_pool=in_pool,wrong_beats_true=signature!=true and q['score']<true_J-tolerance,
                wrong_dominates_fit_and_count=signature!=true and point['objective']<values[0]-tolerance and internal<=len(true),
                true_better_than_incumbent_in_pool=in_pool and q['score']>true_J+tolerance,exit_status=record['event'].get('status'),
                oracle_sha256=oracle_audits[job]))
    dest=output/'analysis';frame=pd.DataFrame(rows);frame.to_csv(dest/'role_diagnostics.csv',index=False)
    frame.groupby('panel').sum(numeric_only=True).to_csv(dest/'role_summary.csv')
    results=pd.read_csv(dest/'long_per_model.csv');events=pd.read_csv(dest/'long_events.csv');pairs=[]
    for panel in ('main','stress'):
        g=results[(results.panel==panel)&(results.resolution==.24)&(results['view']=='bank')]
        status=events[events.panel==panel].pivot(index='job',columns='method',values='exit_status')
        for scope,ids in [('all',status.index),('both_exit_gap_met',status.index[(status.full_rnj==2)&(status.small3==2)])]:
            old=g[(g.method=='full_rnj')&g.job.isin(ids)];new=g[(g.method=='small3')&g.job.isin(ids)]
            pairs.append(dict(panel=panel,scope=scope,inputs=len(ids),old_F1=old.clade_f1.mean(),new_F1=new.clade_f1.mean(),
                old_exact=int(old.exact.sum()),new_exact=int(new.exact.sum())))
    pd.DataFrame(pairs).to_csv(dest/'solver_completion_subset.csv',index=False)
    oracle=pd.DataFrame(oracle_rows);oracle.to_csv(dest/'oracle_objective.csv',index=False)
    oracle.groupby(['panel','method']).agg(inputs=('job','size'),true_in_pool=('true_in_pool','sum'),wrong=('wrong','sum'),
        wrong_beats_true=('wrong_beats_true','sum'),wrong_dominates_fit_and_count=('wrong_dominates_fit_and_count','sum'),
        true_better_than_incumbent_in_pool=('true_better_than_incumbent_in_pool','sum')).to_csv(dest/'oracle_objective_summary.csv')
    if (dest/'base_per_model.csv').exists():
        base=pd.read_csv(dest/'base_per_model.csv');comp=pd.read_csv(dest/'long_per_model.csv');paired=[]
        for resolution in (.16,.24):
            for view in ('bank','path','prefix_path','endpoint'):
                a=base[(base.resolution==resolution)&(base['view']==view)].set_index('job')
                for method in ('full_rnj','small3'):
                    b=comp[(comp.resolution==resolution)&(comp['view']==view)&(comp.method==method)].set_index('job');ids=a.index.intersection(b.index)
                    delta=a.loc[ids].clade_f1-b.loc[ids].clade_f1
                    paired.append(dict(resolution=resolution,view=view,reference=method,inputs=len(ids),delta=float(delta.mean()),
                        wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta< -1e-10).sum()),
                        recovered_new_only=int((a.loc[ids].exact&~b.loc[ids].exact).sum()),recovered_reference_only=int((~a.loc[ids].exact&b.loc[ids].exact).sum())))
        pd.DataFrame(paired).to_csv(dest/'base_paired.csv',index=False)
        sources=[]
        for job in read(output/'base_protocol.json')['jobs']:
            common=read(output/'common'/job/'result.json');n=len(common['labels'])
            signature=lambda point:frozenset(tuple(C) for C,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(C)<n and max(r,x)>1e-9)
            for method,path in [('full_rnj',output/'long'/job/'full_rnj/result.json'),('small3',output/'long'/job/'small3/result.json'),
                ('small3_original_wzzt',output/'base'/job/'result.json')]:
                record=read(path);choice=record['choices']['strict_0.24']['bank'];source=choice['origin'];t=record['trajectory'];q=record['exit']
                if source=='path':
                    index=choice['index']-len(common['bank']);stages=['initial']+[a['kind'] for a in t['attempts'] if a['accepted']]+(['exit'] if q['accepted'] else [])
                    assert len(stages)==len(t['path'])+int(q['accepted']);source=stages[index]
                    if source=='exit' and q['warm_origin']=='complete_rnj' and signature(q['point'])==signature(common['bank'][q['warm_index']]):source='exit_RNJ_warm_structure'
                sources.append(dict(job=job,method=method,library_origin=choice['origin'],selected_stage=source))
        pd.DataFrame(sources).to_csv(dest/'selected_stage_sources.csv',index=False)
    results[results['view']=='bank'].groupby(['panel','resolution','method','origin','topology_in_common_RNJ_bank']).size().rename('inputs').to_csv(dest/'selection_origins.csv')
    write(dest/'attribution_provenance.json',dict(source_hash=sha(Path(__file__)),truth_used_only_after_freeze=True,
        warning='completion subsets are postselected diagnostics; no full-test performance inference from these subsets'))
    delivery=Path(__file__).resolve().parents[2]/'docs/rnj_small_seed_20261031'
    import shutil
    for name in ('role_diagnostics.csv','role_summary.csv','solver_completion_subset.csv','selection_origins.csv','attribution_provenance.json','oracle_objective.csv','oracle_objective_summary.csv'):
        shutil.copy2(dest/name,delivery/name)
    if (dest/'base_paired.csv').exists():shutil.copy2(dest/'base_paired.csv',delivery/'base_paired.csv')
    if (dest/'selected_stage_sources.csv').exists():shutil.copy2(dest/'selected_stage_sources.csv',delivery/'selected_stage_sources.csv')
    print(frame.groupby('panel').sum(numeric_only=True).to_string(),flush=True)
    print(pd.DataFrame(pairs).to_string(index=False),flush=True)


if __name__=='__main__':
    import sys
    derive(Path(sys.argv[1]).resolve())
