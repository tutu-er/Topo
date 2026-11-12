"""Post-freeze attribution and candidate coverage, without altering any model."""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from .run import read,write,cid,sha
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.joint_five.analyze import supports
from research_experiments.local_rebuild.model import compatible
from research_experiments.rnj_count.diagnose import error_type


def derive(output):
    output=Path(output);p=read(output/'protocol.json');frozen=read(output/'frozen.json');cat=read(output/'catalog.json');rows=[];changes=[];initials=[]
    assert (output/'frozen.json').exists()
    for job in p['evaluation']:
        meta=cat[job];common=read(output/'common'/job/'result.json');labels=common['labels'];n=len(labels);pos={v:i for i,v in enumerate(labels)}
        truth=physical_truth(Path(meta['truth_network']),labels)
        true={tuple(sorted(pos[v] for v in s)) for s in truth['clades']};config=frozen['config']
        r=read(output/'evaluation'/job/cid(config)/'result.json');pool=set(map(tuple,r['event']['pool']))
        initial={tuple(s) for s in common['initial'] if 1<len(s)<n};wrong_initial=initial-true
        initials.append(dict(job=job,panel=meta['panel'],case=meta['case'],true_lines=len(true),initial_lines=len(initial),
            TP=len(initial&true),FP=len(wrong_initial),FN=len(true-initial),
            false_compatible=sum(error_type(s,true)[0]=='compatible_overresolution' for s in wrong_initial),
            false_cross=sum(error_type(s,true)[0]=='crosses_truth' for s in wrong_initial),
            blocked_true=sum(any(not compatible(s,w) for w in wrong_initial) for s in true),
            small_FP=sum(len(s)<=5 for s in wrong_initial),small_FN=sum(len(s)<=5 for s in true-initial)))
        baseline=read(output/'evaluation'/job/'baseline20.json')['trajectory']['point'];before=supports(baseline,n)
        for budget,q in r['outputs'].items():
            after=supports(q['point'],n);deleted=before-after;added=after-before
            changes.append(dict(job=job,panel=meta['panel'],case=meta['case'],budget=budget,accepted=q['accepted'],
                deleted_true=len(deleted&true),deleted_false=len(deleted-true),added_true=len(added&true),added_false=len(added-true),
                simultaneous=bool(deleted and added),score_gain=q['before_score']-q['score']))
        for selection,key in [('matched','strict_0.16'),('tuned',f"{frozen['selection']['rule']}_{frozen['selection']['resolution']:g}")]:
            for budget in ('20','40'):
                q=r['outputs'][budget];point=q['choices'][key]['bank']['point'];pred=supports(point,n);wrong=pred-true
                rows.append(dict(job=job,panel=meta['panel'],case=meta['case'],N=meta['N'],voltage_sigma=meta['voltage_sigma'],
                    budget=budget,selection=selection,pool_count=len(pool),true_lines=len(true),true_in_pool=len(pool&true),
                    all_true_in_pool=true<=pool,small_true_missing_pool=sum(len(s)<=5 for s in true-pool),
                    false_compatible=sum(error_type(s,true)[0]=='compatible_overresolution' for s in wrong),
                    false_cross=sum(error_type(s,true)[0]=='crosses_truth' for s in wrong),
                    blocked_true=sum(any(not compatible(s,w) for w in wrong) for s in true)))
    dest=output/'analysis';pd.DataFrame(rows).to_csv(dest/'pool_coverage.csv',index=False);pd.DataFrame(changes).to_csv(dest/'exit_changes.csv',index=False)
    pd.DataFrame(initials).to_csv(dest/'initial_errors.csv',index=False)
    pd.DataFrame(initials).groupby('panel').agg(inputs=('job','size'),**{k:(k,'sum') for k in
        ('TP','FP','FN','false_compatible','false_cross','blocked_true','small_FP','small_FN')}).to_csv(dest/'initial_error_summary.csv')
    pd.DataFrame(rows).groupby(['panel','budget','selection']).agg(inputs=('job','size'),all_true_in_pool=('all_true_in_pool','sum'),
        small_true_missing_pool=('small_true_missing_pool','sum'),false_compatible=('false_compatible','sum'),false_cross=('false_cross','sum'),
        blocked_true=('blocked_true','sum')).to_csv(dest/'pool_summary.csv')
    if (dest/'terminal_regions.csv').exists():
        regions=pd.read_csv(dest/'terminal_regions.csv');signals=pd.read_csv(dest/'terminal_signals.csv')
        regions.groupby(['panel','regime','method','case']).agg(regions=('job','size'),F1=('F1','mean'),exact=('exact','sum'),
            wrong_beats_true=('wrong_beats_true','sum'),near_zero_gap=('near_zero_linear_gap','sum')).to_csv(dest/'terminal_cases.csv')
        unique=regions[(regions.method=='train')].copy()
        unique['P_full_rank']=unique.P_rank==unique['size']
        unique.groupby(['panel','regime']).agg(regions=('job','size'),P_full_rank=('P_full_rank','sum'),
            singular_ratio_median=('P_singular_ratio','median'),wrong_beats_true=('wrong_beats_true','sum'),
            near_zero_linear_gap=('near_zero_linear_gap','sum')).to_csv(dest/'terminal_design.csv')
        signals.groupby(['regime','case','N','voltage_sigma']).agg(lines=('job','size'),contrast_median=('contrast_RMS','median'),
            residual_median=('residual_RMS','median'),contrast_to_residual_median=('contrast_to_residual','median'),
            below_016=('below_016','sum'),below_024=('below_024','sum')).to_csv(dest/'terminal_signal_summary.csv')
    # A pooled solve has one final status; checkpoint-specific evidence is its
    # recorded incumbent time and bound, not that later final solver status.
    events=pd.read_csv(dest/'solver_events.csv');mask=events.kind=='pooled_exit'
    if 'final_solver_status' not in events:
        events['final_solver_status']=np.nan
    events.loc[mask,'final_solver_status']=events.loc[mask,'final_solver_status'].fillna(events.loc[mask,'status'])
    events.loc[mask,'status']=np.nan
    events['status_scope']=np.where(mask,'final pooled solve; see checkpoint bound and incumbent_at','individual completed call')
    events.to_csv(dest/'solver_events.csv',index=False)
    write(dest/'diagnostic_provenance.json',dict(source_hash=sha(Path(__file__)),truth_used_only_for_post_freeze_diagnosis=True))


def verify_information_job(args):
    from research_experiments.local_rebuild.model import canonical
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.wzzt_paper.common import observed
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from .analyze import independent_profile
    output,job=args;output=Path(output);meta=read(output/'catalog.json')[job];r=read(output/'information'/job/'result.json')
    common=read(output/'common'/job/'result.json');labels=common['labels'];n=len(labels);pos={v:i for i,v in enumerate(labels)}
    physical=physical_truth(Path(meta['truth_network']),labels)
    true=canonical(tuple(sorted(pos[v] for v in s)) for s in physical['clades'])
    full=canonical([*((i,) for i in range(n)),tuple(range(n)),*true])
    regions={tuple(q['region']) for q in r['rows']};V=min(regions,key=lambda s:(-len(s),s))
    outside=canonical(s for s in full if len(s)==1 or not set(s)<=set(V))
    noisy=prepare(observed(meta['train'],meta['N']));path=Path(meta['train'])
    clean=prepare(observed(path.with_name('train_truth.npz'),meta['N']))
    linear=core._PreparedScenarios(clean.labels,clean.p,clean.q,
        tuple(pp@physical['R'].T+qq@physical['X'].T for pp,qq in zip(clean.p,clean.q)))
    rows=[]
    for regime,data in [('linear',linear),('clean_ac',clean),('noisy',noisy)]:
        record=next(q for q in r['rows'] if q['regime']==regime and tuple(q['region'])==V)
        wrong=min((q for q in r['profiles'] if q['regime']==regime and tuple(q['region'])==V and not q['true_local']),key=lambda q:q['train'])
        for label,family,reference in [('true',full,record['true_profile_MAE']),
            ('best_wrong',canonical([*outside,*wrong['effective']]),wrong['train'])]:
            checked=independent_profile(data,family)
            difference=abs(checked['objective']-reference)
            if difference>2e-7:raise AssertionError((job,regime,label,difference))
            rows.append(dict(job=job,regime=regime,region=V,profile=label,recorded=reference,
                independent=checked['objective'],difference=difference,violation=checked['violation']))
    return rows


def verify_information(output):
    """Predeclared sample by metadata only; no selection on observed outcomes."""
    output=Path(output);p=read(output/'protocol.json');info=read(output/'information_protocol.json');cat=read(output/'catalog.json')
    selected=[j for j in p['evaluation'] if cat[j]['panel']=='main' and cat[j]['bundle']==2 and cat[j]['N'] in (24,96)]
    assert len(selected)==24
    rule=dict(rule='main, bundle 2, N in {24,96}; largest terminal region, lexical tie break; true and best-wrong profiles in all three regimes',
        jobs=selected,information_protocol_sha256=sha(output/'information_protocol.json'),source_hash=sha(Path(__file__)))
    write(output/'information_verification_protocol.json',rule)
    audits=[]
    for job in p['evaluation']:
        file=output/'information'/job/'result.json';r=read(file)
        assert r['source_hash']==info['source_hash'] and r['information_protocol_sha256']==sha(output/'information_protocol.json')
        assert r['checks']==r['LP_solves']
        audits.append(dict(job=job,region_count=r['region_count'],LP_solves=r['LP_solves'],checks=r['checks'],sha256=sha(file)))
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as executor:
        for r in executor.map(verify_information_job,[(str(output),j) for j in selected]):rows+=r
    pd.DataFrame(rows).to_csv(output/'analysis/information_independent_LP.csv',index=False)
    write(output/'analysis/information_run_audit.json',dict(inputs=len(audits),regions=sum(q['region_count'] for q in audits),
        LP_solves=sum(q['LP_solves'] for q in audits),solver_feasibility_checks=sum(q['checks'] for q in audits),jobs=audits,
        independent_LP=len(rows),max_independent_difference=max(q['difference'] for q in rows),failures=[]))
    print('information verified',len(audits),'inputs;',len(rows),'independent LPs',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('path');parser.add_argument('--verify-information',action='store_true');args=parser.parse_args()
    derive(Path(args.path).resolve())
    if args.verify_information:verify_information(Path(args.path).resolve())
