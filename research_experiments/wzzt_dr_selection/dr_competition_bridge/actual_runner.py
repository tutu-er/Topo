"""Post-planning evaluation protocol: matched single-site control and full costs.

This extension is frozen before any actual future generation. The original
planning files/sources stay untouched. It changes reporting and adds a matched
single-site ablation, not the identifier or price-design calculation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
from . import study

POLICIES=study.POLICIES+('wzzt_single_ablation',)


def summarize(records,planned=None):
    # The original pilot source has one path and four cases. Its 7-policy
    # driver passes56 internally; this extension explicitly has8 policies.
    planned=64
    groups=[]
    for policy in POLICIES:
        all_rows=[r for r in records if r['policy']==policy]
        complete=[r for r in all_rows if r['status']=='complete']
        known=[r['actual_payment'] for r in all_rows if r.get('actual_payment') is not None]
        group=dict(policy=policy,planned=8,records=len(all_rows),complete=len(complete),
            statuses=dict(Counter(r['status'] for r in all_rows)),
            payment_known_records=len(known),payment_unknown_records=len(all_rows)-len(known),
            known_actual_payment=sum(known),
            incomplete_search_records=sum(not r.get('all_local_scans_complete',False) for r in complete))
        if complete:
            group.update(structure_denominator=len(complete),
                wzzt_total_errors=sum(r['evaluation']['multistart_original_rule']['full_block_errors'] for r in complete),
                RNJ_total_errors=sum(r['evaluation']['rnj_bank']['full_block_errors'] for r in complete),
                wzzt_full=sum(r['evaluation']['multistart_original_rule']['full_recovery'] for r in complete))
        else:group.update(structure_denominator=0,wzzt_total_errors=None,RNJ_total_errors=None,wzzt_full=None)
        groups.append(group)
    return dict(status='complete' if len(records)==planned else 'partial',planned=planned,records=len(records),
        statuses=dict(Counter(r['status'] for r in records)),
        unique_fits=len({r['fit_path'] for r in records if r.get('fit_path') and r['status']=='complete'}),
        policy_groups=groups,
        scope='one shared four-day natural realization total; noise, response laws and policies paired; no64 independent samples')


def prepare(planning_root,actual_root):
    planning_root,actual_root=Path(planning_root),Path(actual_root)
    study.verify(planning_root)
    prices=study.read(planning_root/'planning_summary.json')
    if prices['status']!='complete':raise RuntimeError('incomplete planning cannot start actual evaluation')
    if actual_root.exists():
        raise RuntimeError('existing actual root retained; use run after verifying its completed protocol')
    actual_root.mkdir(parents=True)
    # Parent worker verifies unchanged planning producer sources and old inputs.
    (actual_root/'protocol.json').write_bytes((planning_root/'protocol.json').read_bytes())
    local=study.read(planning_root/'planning'/study.read(planning_root/'protocol.json')['local_price_case']/'response_table/plan_result.json')
    sources={str(p):study.sha(p) for p in Path(__file__).parent.glob('*.py')}
    cases=[]
    for case in prices['cases']:
        derived=dict(case,locks={})
        for law,old_lock in case['locks'].items():
            source=study.read(old_lock['path'])
            assert study.sha(old_lock['path'])==old_lock['sha256']
            plans=dict(source['plans'])
            plans['wzzt_single_ablation']=(local['plans']['best_single_within_pair_'+law]
                if case['local_replay_used'] else plans['zero'])
            lock=actual_root/'locks'/f"{case['job_id']}_{law}.json"
            study.write(lock,dict(source,plans=plans,created_utc=study.timestamp(),
                extension='matched WZZT-objective best single within same pair; computed before actual data',
                planning_lock_path=old_lock['path'],planning_lock_sha256=old_lock['sha256']))
            derived['locks'][law]=dict(path=str(lock),sha256=study.sha(lock))
        cases.append(derived)
    study.write(actual_root/'planning_summary.json',dict(prices,cases=cases,policy_count=8,
        derived_from_planning_summary_sha256=study.sha(planning_root/'planning_summary.json')))
    cfg=dict(created_utc=study.timestamp(),status='frozen_before_actual_future',
        source_sha256=sources,planning_root=str(planning_root),
        planning_protocol_sha256=study.sha(planning_root/'protocol.json'),
        planning_summary_sha256=study.sha(planning_root/'planning_summary.json'),
        derived_summary_sha256=study.sha(actual_root/'planning_summary.json'),
        policies=list(POLICIES),planned_records=64,shared_actual_natural_paths=1,
        reporting_fix='costs from all records including failures; missing costs remain unknown; incomplete scans explicit',
        identifier_changed=False,price_model_changed=False,actual_future_read_before_freeze=False,
        actual_domain='original finite-pool RNJ/WZZT adapter coefficient[0,2]',
        planning_domain='nonnegative unbounded; frozen-start local heuristic, not exact same-domain algorithm replica')
    study.write(actual_root/'actual_protocol.json',cfg)
    return cfg


def run(root,max_records=64):
    root=Path(root);cfg=study.read(root/'actual_protocol.json')
    for path,h in cfg['source_sha256'].items():
        if study.sha(path)!=h:raise RuntimeError('actual evaluation source drift '+path)
    if study.sha(root/'planning_summary.json')!=cfg['derived_summary_sha256']:
        raise RuntimeError('frozen actual plan summary changed')
    # Explicitly scoped controls/reporting override; identifier subprocess uses
    # original study.fit_worker unchanged and never receives evaluator truth.
    study.POLICIES=POLICIES
    study.summarize=summarize
    return study.actual(root,max_records)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--planning-root',type=Path);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--max-records',type=int,default=64)
    a=p.parse_args()
    r=prepare(a.planning_root,a.root) if a.mode=='prepare' else run(a.root,a.max_records)
    if r:print(json.dumps({k:v for k,v in r.items() if k in ('status','planned','planned_records','records','statuses','unique_fits')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
