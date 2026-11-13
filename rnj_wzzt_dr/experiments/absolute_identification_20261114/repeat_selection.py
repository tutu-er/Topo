"""Lock repeated-day strata using existing historical qualification only."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from ... import io


def select(source,out):
    destination=out/'repeat44.json';protocol=out/'repeat44_protocol.json'
    if destination.exists() or protocol.exists():raise ValueError('Preserve the locked selection')
    entries=io.read(source)
    if len(entries)!=400 or len({e['id'] for e in entries})!=400:raise ValueError('Expected fixed400 inventory')
    selected=[e for e in entries if e['role'] in ('correct_ambiguous','wrong_present')]
    if Counter(e['role'] for e in selected)!=dict(correct_ambiguous=19,wrong_present=9):raise ValueError('Historical target counts changed')
    for case in sorted({e['case'] for e in entries}):
        for role in ('correct_clear','missing'):
            options=[e for e in entries if e['case']==case and e['role']==role]
            options.sort(key=lambda e:sha256(('repeat20261114|'+e['id']).encode()).hexdigest())
            if len(options)<2:raise ValueError('Insufficient controls')
            selected.extend(options[:2])
    selected.sort(key=lambda e:e['original_index'])
    if len(selected)!=44 or len({e['id'] for e in selected})!=44:raise ValueError('Repeated panel count')
    repeats={6000000+e['original_index']*100+day for e in selected for day in range(32)}
    main={5000000+e['original_index']*100+day for e in entries for day in range(2)}
    if len(repeats)!=1408 or repeats & main:raise ValueError('Overlapping future streams')
    out.mkdir(parents=True,exist_ok=True);io.write(destination,selected)
    summary=dict(histories=len(selected),new_days_per_history=32,new_independent_future_paths=len(repeats),
        physical_history_groups=len({e['physical_history_group'] for e in selected}),
        by_role=dict(Counter(e['role'] for e in selected)),by_case=dict(Counter(e['case'] for e in selected)),
        by_history_partition=dict(Counter(e['history_partition'] for e in selected)),
        case_roles={case:dict(Counter(e['role'] for e in selected if e['case']==case)) for case in sorted({e['case'] for e in selected})},
        partition_roles={part:dict(Counter(e['role'] for e in selected if e['history_partition']==part)) for part in sorted({e['history_partition'] for e in selected})},
        replicate_min=min(repeats),replicate_max=max(repeats),overlap_with_main800=0)
    io.write(protocol,dict(source_script=io.ref(__file__),inventory=io.ref(source),selected_inputs=io.ref(destination),
        rule='All19 historically correct but ambiguous and all9 historically wrong winners with true candidate present. Add2 correct-clear and2 missing-truth controls per network, ranked ascending SHA256(repeat20261114|id).',
        inputs_used='Frozen inventory historical role, case and id only; no future outcome, forecast or plan read.',
        random_stream=dict(study_seed=20261114,replicate_base=6000000,replicate_rule='base + original_index*100 + day',days=32),
        analysis='Supplemental stratified repeated-day evidence. Report per-history and per-role performance; do not pool unweighted with population400 accuracy. Existing history partitions retained.',
        overlap='Fresh replicate range disjoint from main800 at base5000000; same paired random streams across strategies and AC/linear.',summary=summary))
    io.write(out/'repeat44_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(select(a.source.resolve(),a.out.resolve()))
