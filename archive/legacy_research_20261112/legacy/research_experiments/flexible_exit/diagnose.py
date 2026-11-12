"""Read-only post-fit timing and equivalence diagnostics; no new solves."""
from pathlib import Path
import numpy as np
import pandas as pd
from .run import ROOT, METHODS, read, write
from .model import select_block
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from rnj_wzzt.estimation import laminar_l1_milp as core


def diagnose(output):
    output=Path(output);p=read(output/'protocol.json');catalog=read(output/'catalog.json');rows=[];max_difference=0.
    for job in p['jobs']:
        start=read(output/'parents'/job/'start.json');n=len(start['labels']);reference=None
        for method in ('drop_refit','drop_resume','mask_resume'):
            r=read(output/'jobs'/job/method/'result.json');point=r['guarded']['point']
            rx=np.stack(core.build_matrices_from_atoms(n,point['supports'],point['r'],point['x']))
            if reference is None:reference=rx
            else:max_difference=max(max_difference,float(np.max(abs(rx-reference))))
        parent=read(next(path for path in p['reference_hashes'][job] if Path(path).parent.name=='small3_original_exit'))
        prefix=parent['trajectory']['point'];block=select_block(prefix,n)
        truth=physical_truth(Path(catalog[job]['truth_network']),start['labels']);positions={v:i for i,v in enumerate(start['labels'])}
        true={tuple(sorted(positions[v] for v in C)) for C in truth['clades']}
        rows.append(dict(job=job,stage='prefix_endpoint_before_old_exit',eligible=block is not None,block=block,
            block_true=block in true if block else None,blocked_true=sum(not compatible(block,C) for C in true) if block else 0,
            prefix_block_retained_after_old_exit=bool(block and block in canonical(parent['exit']['point']['supports'])),
            diagnostic_only=True))
    if not p.get('added_after_primary_results',False):assert max_difference<1e-8
    dest=output/'analysis';delivery=Path(p.get('delivery_directory',str(ROOT/'docs/flexible_exit_20261101')))
    pd.DataFrame(rows).to_csv(dest/'earlier_stage_diagnostic.csv',index=False)
    costs=pd.read_csv(dest/'cost_per_input.csv');events=pd.read_csv(dest/'events.csv')
    source=float(costs[costs.method=='historical'].total_seconds.sum());fresh=float(costs[costs.method!='historical'].fresh_seconds.sum())
    payloads=list((output/'jobs').glob('*/*/payload.json'));results=list((output/'jobs').glob('*/*/result.json'))
    write(dest/'batch_budget.json',dict(inputs=24,branches=96,nominal_caps_sum_seconds=96*14400,
        per_branch_recorded_cap_seconds=14400,single_MIP_cap_seconds=3600,
        unique_reused_components_seconds=source,fresh_algorithm_stage_seconds=fresh,unique_components_sum_seconds=source+fresh,
        dispatch_wall_seconds=max(f.stat().st_mtime for f in results)-min(f.stat().st_mtime for f in payloads),
        native_MIPs=len(events[events.status.notna()]),max_MIP_seconds=float(events.seconds.max()),
        component_scope='sum of per-worker algorithm-stage walls; excludes process startup, independent audits and packaging; not CPU or batch elapsed'))
    write(dest/'post_checks.json',dict(passed=True,guarded_three_deletion_arms_RX_max_difference=max_difference,
        earlier_stage_is_diagnostic_only=True,
        note='No new search and no new family fitting. Earlier prefix stage was not run as another intervention arm.'))
    for name in ('earlier_stage_diagnostic.csv','batch_budget.json','post_checks.json'):
        import shutil
        shutil.copy2(dest/name,delivery/name)
    print(read(dest/'batch_budget.json'))
    print(pd.DataFrame(rows).groupby(['eligible','block_true'],dropna=False).agg(inputs=('job','size'),blocked_true=('blocked_true','sum')).to_string())


def compare(primary,earlier):
    primary,earlier=Path(primary),Path(earlier);frames=[]
    for label,folder in (('after_existing_exit',primary),('before_existing_exit_exploratory',earlier)):
        frame=pd.read_csv(folder/'analysis/summary.csv');cost=pd.read_csv(folder/'analysis/cost_per_input.csv')
        timing=cost.groupby('method').agg(fresh_sum_seconds=('fresh_seconds','sum'),total_sum_seconds=('total_seconds','sum'))
        frame=frame.join(timing,on='method');frame['stage']=label;frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(primary/'analysis/timing_comparison.csv',index=False)
    result.to_csv(ROOT/'docs/flexible_exit_20261101/timing_comparison.csv',index=False)
    a,b=(read(folder/'analysis/batch_budget.json') for folder in (primary,earlier))
    assert abs(a['unique_reused_components_seconds']-b['unique_reused_components_seconds'])<1e-8
    write(primary/'analysis/combined_budget.json',dict(unique_inputs=24,primary_branches=96,exploratory_timing_branches=96,
        nominal_caps_sum_seconds=192*14400,shared_old_components_seconds=a['unique_reused_components_seconds'],
        fresh_stages_sum_seconds=a['fresh_algorithm_stage_seconds']+b['fresh_algorithm_stage_seconds'],
        unique_components_sum_seconds=a['unique_reused_components_seconds']+a['fresh_algorithm_stage_seconds']+b['fresh_algorithm_stage_seconds'],
        note='same old mainline charged only once in unique total; per-method deployment costs separately retained'))
    import shutil
    shutil.copy2(primary/'analysis/combined_budget.json',ROOT/'docs/flexible_exit_20261101/combined_budget.json')
    print(result.query('scope == "all24" and view == "guarded"').to_string(index=False))


if __name__=='__main__':
    import sys
    diagnose(Path(sys.argv[1]).resolve())
    if len(sys.argv)>2:compare(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
