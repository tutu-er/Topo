"""Post-primary fixed-rule validation view and mechanism/cost diagnostics.

No new search, refit, threshold, or parameter is selected here. This additional
view is explicitly exploratory and was added after primary results were read.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
import shutil
import pandas as pd
from .run import ROOT, FOUR, PORT, read, write, sha
from .model import criterion
from research_experiments.joint_exit.selection import choose
from research_experiments.local_rebuild.model import compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


def diagnose(output):
    output = Path(output); p = read(output / 'protocol.json'); catalog = read(output / 'catalog.json')
    frame = pd.read_csv(output / 'analysis/per_model.csv'); chosen_rows = frame[frame['view'] == 'selected'].set_index(['job', 'resolution', 'method'])
    extra, mechanism, total = [], [], []
    for job in p['jobs']:
        meta = catalog[job]; common = read(FOUR / 'common' / job / 'result.json'); labels = common['labels']; n = len(labels)
        pos = {v: i for i, v in enumerate(labels)}
        true = {tuple(sorted(pos[v] for v in C)) for C in physical_truth(Path(meta['truth_network']), labels)['clades']}
        sig = lambda point: {tuple(C) for C, r, x in zip(point['supports'], point['r'], point['x']) if 1 < len(C) < n and max(r, x) > 1e-9}
        full = read(FOUR / 'evaluation' / job / 'full_rnj_exit/result.json')
        sparse = read(FOUR / 'evaluation' / job / 'small3_original_exit/result.json')
        nj = read(PORT / 'jobs' / job / 'result.json')
        rebuilt = read(output / 'jobs' / job / 'union_count/result.json')
        pure = read(output / 'jobs' / job / 'pure_wzzt/result.json')
        pool = read(output / 'pools' / job / 'result.json')
        accepted = [a for a in pure['trajectory']['attempts'] if a['accepted']]
        false = [a for a in accepted if tuple(a['support']) not in true]
        weights = {tuple(C): max(r, x) for C, r, x in zip(pure['raw_endpoint']['supports'], pure['raw_endpoint']['r'], pure['raw_endpoint']['x'])}
        provisional = {tuple(C) for C in pure['raw_endpoint']['supports'] if 1 < len(C) < n}
        pure_info = dict(accepted_wrong_small=sum(len(a['support']) <= 5 for a in false), accepted_wrong_large=sum(len(a['support']) > 5 for a in false),
            first_false_size=len(false[0]['support']) if false else 0,
            first_false_cross=any(not compatible(tuple(false[0]['support']), C) for C in true) if false else False,
            retained_zero_wrong=sum(C not in true and weights[C] <= 1e-9 for C in provisional),
            zero_wrong_blocks_true=sum(any(C not in true and weights[C] <= 1e-9 and not compatible(C, D) for C in provisional) for D in true))
        warm_signatures = [sig(point) for point in pool['warm']]
        for resolution in p['resolutions']:
            key = f'strict_{resolution:g}'
            q = [dict(full['choices'][key]['bank'], index=0, selected_generator='full_rnj_exit'),
                 dict(sparse['choices'][key]['bank'], index=1, selected_generator='small3_original_exit'),
                 dict(rebuilt['choices'][str(resolution)]['selected'], index=2, selected_generator='union_count')]
            started = perf_counter(); selected = choose(q, 'strict'); elapsed = perf_counter()-started
            assert selected['point'] == q[selected['index']]['point']
            assert selected['validation'] <= min(v['validation'] for v in q[:2])+1e-12
            # Selection happens before looking up any truth-based metric.
            row = chosen_rows.loc[(job, resolution, selected['selected_generator'])].to_dict()
            row.update(job=job, resolution=resolution, method='union_count_validated', selected_generator=selected['selected_generator'],
                       view='selected', exploratory_after_primary=True, extra_selection_seconds=elapsed)
            extra.append(row)
            selected_sig = sig(rebuilt['choices'][str(resolution)]['selected']['point'])
            parent_points = [q[0]['point'], q[1]['point'], nj['choices'][str(resolution)]['candidates']['NJ_matched']['point']]
            parent_signatures = [sig(point) for point in parent_points]
            mechanism.append(dict(job=job, panel=meta['panel'], resolution=resolution,
                raw_rebuild_is_new_training_structure=sig(rebuilt['raw_endpoint']) not in warm_signatures,
                final_rebuild_differs_from_three_parents=selected_sig not in parent_signatures,
                final_rebuild_correct=selected_sig == true,
                new_correct_outside_two=selected_sig == true and all(v != true for v in parent_signatures[:2]),
                new_correct_outside_three=selected_sig == true and all(v != true for v in parent_signatures),
                raw_J_minus_best_final_parent_J=rebuilt['joint']['score']-min(criterion(v, n, rebuilt['penalty']) for v in parent_points),
                **pure_info))
        total.append(dict(job=job, panel=meta['panel'], method='union_count_validated',
            total_seconds=rebuilt['cumulative_seconds']+sum(r['extra_selection_seconds'] for r in extra[-2:]),
            fresh_seconds=rebuilt['fresh_seconds'], budget=14400.,
            budget_scope='same union solve plus unchanged final validation selection; exploratory', timing_kind='recorded upstream plus measured new stages'))
    dest = output / 'analysis'; delivery = ROOT / 'docs/rnj_union_long_20261031'
    e = pd.DataFrame(extra); e.to_csv(dest / 'validation_diagnostic_per_input.csv', index=False)
    aggregate = dict(inputs=('job', 'size'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'), train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean'))
    e.groupby(['panel', 'resolution']).agg(**aggregate).to_csv(dest / 'validation_diagnostic_summary.csv')
    e.groupby(['panel', 'resolution', 'selected_generator']).size().rename('inputs').to_csv(dest / 'validation_diagnostic_sources.csv')
    m = pd.DataFrame(mechanism); m.to_csv(dest / 'mechanism_per_input.csv', index=False)
    m.groupby(['panel', 'resolution']).agg(inputs=('job', 'size'), new_raw=('raw_rebuild_is_new_training_structure', 'sum'),
        novel_final=('final_rebuild_differs_from_three_parents', 'sum'), correct=('final_rebuild_correct', 'sum'),
        new_correct_vs_two=('new_correct_outside_two', 'sum'), new_correct_vs_three=('new_correct_outside_three', 'sum'),
        pure_false_small=('accepted_wrong_small', 'sum'), pure_false_large=('accepted_wrong_large', 'sum'),
        pure_first_false_cross=('first_false_cross', 'sum'), zero_wrong=('retained_zero_wrong', 'sum'), zero_blocks=('zero_wrong_blocks_true', 'sum')).to_csv(dest / 'mechanism_summary.csv')
    existing = pd.read_csv(dest / 'cost_per_input.csv')
    costs = pd.concat([existing, pd.DataFrame(total)], ignore_index=True)
    costs.groupby(['panel', 'method']).agg(inputs=('job', 'size'), total_median=('total_seconds', 'median'), total_p90=('total_seconds', lambda s: s.quantile(.9)),
        total_max=('total_seconds', 'max'), fresh_median=('fresh_seconds', 'median')).to_csv(dest / 'cost_with_validation_diagnostic.csv')
    payloads = list((output / 'jobs').glob('*/*/payload.json')); results = list((output / 'jobs').glob('*/*/result.json'))
    event = pd.read_csv(dest / 'events.csv'); charged_upstream = event[event.method == 'union_count'].upstream_seconds.sum()
    write(dest / 'batch_budget.json', dict(new_jobs=len(results), per_job_cap=14400, gross_nominal_job_caps_seconds=480*14400,
        worker_stage_seconds_sum=float(event.fresh_seconds.sum()), reused_generation_seconds_once=float(charged_upstream),
        cumulative_components_once=float(event.fresh_seconds.sum()+charged_upstream),
        dispatch_to_last_result_seconds=max(f.stat().st_mtime for f in results)-min(f.stat().st_mtime for f in payloads),
        wall_estimate_source='filesystem payload/result timestamps; not a single-method latency',
        note='component seconds sum parallel worker walls; not CPU seconds; repeated union ablations share upstream once in this batch sum'))
    write(dest / 'diagnostic_provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
        added_after_primary_results=True, not_primary_preregistered_comparison=True, rule='existing strict validation, no new parameter',
        new_structural_MILP=0, extra_training_fit=0, truth_not_passed_to_selector=True))
    names = ['validation_diagnostic_per_input.csv', 'validation_diagnostic_summary.csv', 'validation_diagnostic_sources.csv',
             'mechanism_per_input.csv', 'mechanism_summary.csv', 'cost_with_validation_diagnostic.csv', 'batch_budget.json', 'diagnostic_provenance.json']
    for name in names: shutil.copy2(dest / name, delivery / name)
    print(pd.read_csv(dest / 'validation_diagnostic_summary.csv').to_string(index=False))
    print(pd.read_csv(dest / 'mechanism_summary.csv').to_string(index=False))


if __name__ == '__main__':
    import sys
    diagnose(Path(sys.argv[1]).resolve())
