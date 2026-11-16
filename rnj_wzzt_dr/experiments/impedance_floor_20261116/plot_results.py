"""Plot complete frozen-bank floor replays; read results only, never fit or load truth."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / 'outputs/impedance_floor_20261116_v1'
GROUPS = ('history288', 'history10', 'large96_main', 'large96_pressure')
TITLES = {'history288': 'Current 40 histories', 'history10': 'Current 16 histories',
          'large96_main': 'Earlier main study: 240 inputs',
          'large96_pressure': 'Earlier stress study: 160 inputs'}
EXPECTED = dict(zip(GROUPS, (40, 16, 240, 160)))
SCOPES = ('all', 'non_singleton')
COLORS = {'all': '#1463AD', 'non_singleton': '#D66A12'}
FILENAMES = ('rx75_sensitivity.png', 'rx75_sensitivity.svg',
             'ohm_sensitivity.png', 'ohm_sensitivity.svg',
             'plot_data.json', 'plot_receipt.json', 'plot_source_snapshot.py')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path, expected=None):
    r = ref(path)
    if expected is not None and r['sha256'] != expected:
        raise ValueError('Sealed result changed: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf-8')), r


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _key(setting):
    return setting['metric'], setting['scope'], float(setting['threshold'])


def aggregate(run):
    run = Path(run).resolve()
    complete, complete_ref = read(run / 'complete.json')
    if complete.get('status') != 'complete' or not complete.get('source_unchanged'):
        raise ValueError('Require the final complete run with source_unchanged=true')
    protocol, protocol_ref = read(complete['protocol']['path'], complete['protocol']['sha256'])
    entries, entries_ref = read(protocol['inputs']['path'], protocol['inputs']['sha256'])
    counts = Counter(e['source_group'] for e in entries)
    if dict(counts) != EXPECTED or len({e['id'] for e in entries}) != sum(EXPECTED.values()):
        raise ValueError('Expected the locked 40+16+240+160 input ledger')
    if complete['completed_inputs'] != len(entries) or complete['planned_inputs'] != len(entries):
        raise ValueError('The final completion ledger is incomplete')
    records = {r['input_id']: r for r in complete['records']}
    if set(records) != {e['id'] for e in entries}:
        raise ValueError('Completion records differ from the input ledger')
    results, source_refs = {}, [complete_ref, protocol_ref, entries_ref]
    for entry in entries:
        record = records[entry['id']]
        if 'result' not in record:
            results[entry['id']] = dict(status=record['status'], grid=[], baseline=None)
            continue
        result, result_ref = read(record['result']['path'], record['result']['sha256'])
        if result['input_id'] != entry['id']:
            raise ValueError('Result/input identity mismatch')
        source_refs.append(result_ref)
        results[entry['id']] = result
    groups = []
    for group in GROUPS:
        members = [e for e in entries if e['source_group'] == group]
        grid = [('rx75', protocol['relative_grid'])]
        if group in protocol['physical_grid_groups']:
            grid.append(('ohm', protocol['physical_ohm_grid']))
        rows = []
        for metric, values in grid:
            for scope in SCOPES:
                for threshold in values:
                    row = dict(metric=metric, scope=scope, threshold=threshold,
                        planned_inputs=len(members), successful_plans=0, failed_or_unavailable_plans=0,
                        historical=dict(full_correct=0, internal_correct=0),
                        cumulative=dict(available=0, full_correct=0, internal_correct=0),
                        failed_input_ids=[])
                    for entry in members:
                        result = results[entry['id']]
                        matches = [p for p in result.get('grid', [])
                                   if _key(p['setting']) == (metric, scope, float(threshold))]
                        if len(matches) > 1:
                            raise ValueError('Duplicate threshold result for ' + entry['id'])
                        plan = matches[0] if matches else None
                        if (result.get('status') != 'complete' or plan is None
                                or plan.get('status') != 'complete' or plan.get('planning_status') != 'complete'):
                            row['failed_or_unavailable_plans'] += 1
                            row['failed_input_ids'].append(entry['id'])
                            continue
                        row['successful_plans'] += 1
                        outcome = plan['history_selected_topology']
                        row['historical']['full_correct'] += int(outcome['full_exact'])
                        row['historical']['internal_correct'] += int(outcome['internal_exact'])
                        future = plan['cumulative_selected_topology']
                        if future is not None:
                            row['cumulative']['available'] += 1
                            row['cumulative']['full_correct'] += int(future['full_exact'])
                            row['cumulative']['internal_correct'] += int(future['internal_exact'])
                    for criterion in ('full', 'internal'):
                        row['historical'][criterion + '_percent_of_planned'] = 100. * row['historical'][criterion + '_correct'] / len(members)
                        row['cumulative'][criterion + '_percent_of_planned'] = (
                            100. * row['cumulative'][criterion + '_correct'] / len(members)
                            if group in protocol['physical_grid_groups'] else None)
                    rows.append(row)
        # f=0 is a repeated label for the exact same original bank, not new runs.
        baseline = [results[e['id']].get('baseline') for e in members]
        summary = dict(planned_inputs=len(members), available=0, full_correct=0, internal_correct=0)
        for b in baseline:
            if b is not None and b.get('status') == 'complete':
                summary['available'] += 1
                summary['full_correct'] += int(b['history_selected_topology']['full_exact'])
                summary['internal_correct'] += int(b['history_selected_topology']['internal_exact'])
        for row in rows:
            if row['threshold'] == 0. and row['successful_plans'] == summary['available']:
                if any(row['historical'][c + '_correct'] != summary[c + '_correct'] for c in ('full', 'internal')):
                    raise ValueError('Zero-threshold result differs from its original baseline: ' + group)
        groups.append(dict(source_group=group, title=TITLES[group], planned_inputs=len(members),
                           original_baseline=summary, rows=rows))
    return dict(status='complete', run=str(run), groups=groups, source_hashes=source_refs,
        primary_view='Historical validation selection; training is not rescored',
        rate_denominator='All planned inputs within each source group; failed/unavailable plans are never counted as correct',
        full_definition='Exact active support set, including singleton and all-terminal common supports',
        internal_definition='Exact proper internal clades only',
        repeat_note='Thresholds reuse the same observed histories and frozen candidate banks; these are not independent physical days',
        cumulative_note='Historical validation plus existing new day is tabulated only for the current 40/16 inputs, not plotted with historical-only legacy studies',
        no_new_fit_or_AC=True)


def plot_panels(data, metric, out):
    groups = [g for g in data['groups'] if any(r['metric'] == metric for r in g['rows'])]
    if len(groups) not in (2, 4):
        raise ValueError('Expected four relative-floor or two physical-floor panels')
    nrows = 2 if len(groups) == 4 else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12.8, 9.0 if nrows == 2 else 5.5), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        for scope in SCOPES:
            rows = sorted((r for r in group['rows'] if r['metric'] == metric and r['scope'] == scope),
                          key=lambda r: r['threshold'])
            xs = [r['threshold'] for r in rows]
            for criterion, style, marker in (('full', '-', 'o'), ('internal', '--', 's')):
                ys = [r['historical'][criterion + '_percent_of_planned'] for r in rows]
                ax.plot(xs, ys, color=COLORS[scope], linestyle=style, marker=marker,
                        linewidth=1.9, markersize=4.2, markerfacecolor='white' if criterion == 'internal' else COLORS[scope],
                        alpha=.94)
        unavailable = [r['failed_or_unavailable_plans'] for r in group['rows'] if r['metric'] == metric]
        ax.set_title(group['title'], fontsize=12, pad=10)
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xlim(0, .32 if metric == 'rx75' else .02)
        ax.set_xticks([0, .04, .08, .12, .16, .20, .24, .28, .32] if metric == 'rx75'
                      else [0, .005, .010, .015, .020])
        ax.set_xlabel('RX75 normalized branch-strength floor' if metric == 'rx75' else 'Impedance-magnitude floor (ohm)')
        ax.set_ylabel('Exact recovery / planned inputs')
        ax.grid(True, alpha=.22, linewidth=.7)
        ax.text(.02, .035, f"Planned n={group['planned_inputs']}; failed/unavailable per curve point: {min(unavailable)}–{max(unavailable)}",
                transform=ax.transAxes, fontsize=8.3, bbox=dict(facecolor='white', alpha=.82, edgecolor='none', pad=2))
    legend = [Line2D([0], [0], color=COLORS[s], linestyle=style, marker=marker, linewidth=2,
                     label=f"{'All supports' if s == 'all' else 'Non-singleton supports'}: {kind}")
              for s in SCOPES for style, marker, kind in (('-', 'o', 'full support'), ('--', 's', 'internal only'))]
    fig.suptitle('Frozen-bank impedance-floor sensitivity: historical validation', fontsize=14, y=.985)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .953), ncol=2,
               fontsize=9.2, frameon=False)
    fig.text(.5, .018,
        'Solid: exact full active support, including terminal/common terms. Dashed: exact internal topology.\n'
        'Same histories are reused across thresholds. Rates retain every planned input; unresolved cases are listed separately.',
        ha='center', va='bottom', fontsize=9)
    fig.subplots_adjust(top=.84 if nrows == 2 else .74, bottom=.12 if nrows == 2 else .19,
                        left=.075, right=.98, hspace=.37, wspace=.22)
    stem = 'rx75_sensitivity' if metric == 'rx75' else 'ohm_sensitivity'
    fig.savefig(out / (stem + '.png'), dpi=190, facecolor='white')
    svg = out / (stem + '.svg')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    # Hash the final SVG bytes, after whitespace normalization.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if any((out / name).exists() for name in FILENAMES):
        raise FileExistsError('Preserve existing plot artifacts; choose a new output directory')
    data = aggregate(args.run)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    shutil.copyfile(__file__, out / 'plot_source_snapshot.py')
    plot_panels(data, 'rx75', out)
    plot_panels(data, 'ohm', out)
    save_json(out / 'plot_data.json', data)
    # The source is complete before drawing and must remain byte-identical.
    for reference in data['source_hashes']:
        if ref(reference['path'])['sha256'] != reference['sha256']:
            raise ValueError('Result changed while plotting: ' + reference['path'])
    receipt = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source=ref(__file__), source_snapshot=ref(out / 'plot_source_snapshot.py'),
        source_unchanged=True, run_complete=data['source_hashes'][0],
        output_files=[ref(out / name) for name in FILENAMES if name != 'plot_receipt.json'],
        primary_view=data['primary_view'], no_new_fit_or_AC=True)
    save_json(out / 'plot_receipt.json', receipt)
    print(json.dumps(dict(out=str(out), status='complete', plots=4,
                         planned_inputs=sum(g['planned_inputs'] for g in data['groups']))))


if __name__ == '__main__':
    main()
