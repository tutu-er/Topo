"""Plot four predeclared saved EV histories; no fitting or action selection.

Example (from repository root): python -m research_experiments.mobile_ev_mle.ev_compact_plot
--directory <raw-or-slim-development-directory> --seeds 989101 989102 989103 989104
--output <new-figure-directory>. Calibration directories are refused.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .ev_policy_calibration import freeze_identity

METHODS = ('constant', 'refined_v2g_mobile', 'capped_v2g_mobile')
METHOD_LABELS = ('恒充', '精化平均策略', '上限20策略')
SITE_COLORS = ('#2878B5', '#E37942', '#2C956C')
SCHEMA = 'ev_compact_saved_curve_figure_v1'


def layout_parameters(layout='default'):
    parameters = dict(name=layout, width_inches=12., height_inches=8.6, fontsize_pt=9.,
        annotation_fontsize_pt=8., axis_label_fontsize_pt=9., column_title_fontsize_pt=11.,
        legend_fontsize_pt=9., caption_fontsize_pt=8., supertitle_fontsize_pt=13.,
        intended_report_width_inches=6.5,
        margins=dict(left=.10, right=.985, bottom=.085, top=.86, hspace=.44, wspace=.16))
    if layout == 'report':
        parameters.update(width_inches=8., height_inches=8.5, fontsize_pt=10.,
            annotation_fontsize_pt=10., axis_label_fontsize_pt=10., legend_fontsize_pt=9.5,
            caption_fontsize_pt=9.5, supertitle_fontsize_pt=12.,
            margins=dict(left=.12, right=.99, bottom=.13, top=.84, hspace=.55, wspace=.18))
    elif layout != 'default':
        raise ValueError('layout must be default or report')
    scale = parameters['intended_report_width_inches'] / parameters['width_inches']
    parameters['annotation_fontsize_at_report_width_pt'] = parameters['annotation_fontsize_pt'] * scale
    parameters['axis_label_fontsize_at_report_width_pt'] = parameters['axis_label_fontsize_pt'] * scale
    return parameters


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def actual_margin(row, candidates):
    """Recalculate attained actual SSE margin from saved candidate losses."""
    if row.get('status') != 'complete':
        return None, 'actual_joint_fit_incomplete'
    fits = row.get('fits') or []
    losses = {fit['name']: float(fit['loss']) for fit in fits
              if fit.get('success') and _finite(fit.get('loss'))}
    if len(losses) != len(fits) or set(losses) != set(candidates) or row.get('true_name') not in losses:
        return None, 'missing_true_or_competitor_successful_fit'
    true = row['true_name']; value = min(loss for name, loss in losses.items() if name != true) - losses[true]
    if row.get('margin') is not None and abs(value - row['margin']) > 1e-8 * max(1., abs(value)):
        raise ValueError('saved actual margin disagrees with candidate SSE')
    return value, None


def prepare_panel(trial, method):
    physical = trial['config']['physical']; T = physical['horizon']
    n = len(trial['history']['p'][0]); dt = physical['dt_hours']
    rows = {row['method']: row for row in trial['rows']}
    if len(rows) != len(trial['rows']):
        raise ValueError('duplicate saved method row')
    row = rows.get(method) or {}
    selection = trial.get('selection') or {}
    design = (selection.get('methods') or {}).get(method) or {}
    operation = (selection.get('constant_operation') if method == 'constant' else design.get('operation')) or {}
    source = 'constant_plan' if method == 'constant' else design.get('selected_origin')
    candidates = [record['name'] for record in trial['proposal']['bank']]
    margin, margin_reason = actual_margin(row, candidates)
    panel = dict(method=method, status=row.get('status', 'missing_method'), selected_origin=source,
                 actual_M=margin, actual_M_unavailable_reason=margin_reason, slots=[],
                 is_new_MILP_curve=None,
                 operation_feasible=operation.get('feasible'), connection_source='saved_operation.connected',
                 power_unit='kW; positive charging, negative discharging', dt_hours=dt)
    power = row.get('power_kw')
    if power is None:
        panel['curve_unavailable_reason'] = 'no_saved_executed_power'
        return panel
    if len(power) != T or any(len(values) != n or any(not _finite(v) for v in values) for values in power):
        raise ValueError('invalid saved power dimensions or values')
    if source == 'MILP':
        initials = [record['operation']['u_kw'] for record in design.get('initial_candidates') or []
                    if record.get('operation', {}).get('feasible') and record['operation'].get('u_kw') is not None]
        if initials:
            panel['is_new_MILP_curve'] = not any(len(other) == T and all(len(values) == n for values in other)
                and max(abs(power[t][j] - other[t][j]) for t in range(T) for j in range(n)) <= 1e-10
                for other in initials)
    if operation.get('u_kw') is not None:
        if len(operation['u_kw']) != T or any(len(values) != n for values in operation['u_kw']):
            raise ValueError('operation power dimensions')
        if max(abs(power[t][j] - operation['u_kw'][t][j]) for t in range(T) for j in range(n)) > 1e-7:
            raise ValueError('executed curve differs from selected operation')
    connected = operation.get('connected')
    if connected is not None:
        if len(connected) != T or any(len(values) != n or any(v not in (False, True, 0, 1) for v in values) for values in connected):
            raise ValueError('invalid connected occupancy')
    else:
        panel['connection_source'] = 'unavailable; nonzero injection site only'
    activity = ['connected' if connected is not None and any(connected[t]) else 'unknown' for t in range(T)]
    route_seen = set()
    for arc in operation.get('route') or []:
        start, stop, kind = arc['departure'], arc['arrival'], arc['kind']
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= T:
            raise ValueError('invalid saved route interval')
        if kind not in ('connected', 'wait', 'setup', 'travel'):
            raise ValueError('unknown saved route activity')
        for t in range(start, stop):
            if t in route_seen:
                raise ValueError('overlapping saved route intervals')
            route_seen.add(t); activity[t] = kind
    for t, values in enumerate(power):
        injected = [j for j, value in enumerate(values) if abs(value) > 1e-7]
        occupied = [j for j in range(n) if connected is not None and connected[t][j]]
        if len(injected) > 1 or len(occupied) > 1:
            raise ValueError('more than one EV injection/connection site per slot')
        if connected is not None and any(j not in occupied for j in injected):
            raise ValueError('power at an unconnected location')
        if connected is not None and bool(occupied) != (activity[t] == 'connected'):
            raise ValueError('saved route and connected occupancy disagree')
        site = occupied[0] if occupied else injected[0] if injected else None
        panel['slots'].append(dict(slot=t, start_minutes=60 * dt * t, stop_minutes=60 * dt * (t + 1),
                                   site_power_kw=[float(v) for v in values], net_power_kw=float(sum(values)),
                                   connected_site=occupied[0] if occupied else None,
                                   injection_site=injected[0] if injected else None,
                                   plot_site=site, activity=activity[t]))
    panel['operation_metrics'] = {key: operation.get(key) for key in
                                 ('grid_energy_kwh', 'throughput_kwh', 'travel_energy_kwh', 'driver_cost', 'moves')}
    panel['curve_unavailable_reason'] = None
    return panel


def figure_data(directory, seeds, *, kind='auto'):
    directory = Path(directory).resolve(); seeds = list(seeds)
    if len(seeds) != 4 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 4:
        raise ValueError('declare exactly four distinct historical seeds before reading outcomes')
    if kind not in ('auto', 'raw', 'slim'):
        raise ValueError('kind must be auto, raw or slim')
    freeze_bytes = (directory / 'freeze.json').read_bytes(); freeze = json.loads(freeze_bytes)
    if freeze.get('phase') not in ('development', 'evaluation'):
        raise ValueError('plot only explicitly declared development or evaluation samples, not calibration')
    identity = freeze_identity(freeze)
    if not set(seeds) <= set(freeze['seeds']):
        raise ValueError('predeclared figure seed absent from frozen seed plan')
    manifest_path = directory / 'delivery_manifest.json'
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else None
    entries = {row['slim_file']: row for row in manifest['trial_files']} if manifest else {}
    histories = []; inputs = []
    for seed in seeds:
        possible = [directory / f'trial_{seed}.json.gz', directory / f'trial_{seed}.slim.json.gz']
        available = [path for path in possible if path.exists()
                     and (kind == 'auto' or (kind == 'raw') == ('.slim.' not in path.name))]
        if len(available) != 1:
            raise ValueError('each requested seed needs one unambiguous saved raw/slim input')
        path = available[0]; data = path.read_bytes(); sha = digest(data); trial = json.loads(gzip.decompress(data))
        if trial['seed'] != seed or trial['phase'] != freeze['phase'] or freeze_identity(trial) != identity:
            raise ValueError('saved trial seed/phase/policy differs from freeze')
        if '.slim.' in path.name:
            if trial.get('artifact_schema') != 'ev_compact_slim_trace_v1':
                raise ValueError('unknown slim trace schema')
            if manifest and (path.name not in entries or entries[path.name]['slim_sha256'] != sha
                             or entries[path.name]['slim_bytes'] != len(data)):
                raise ValueError('slim input hash/size differs from delivery manifest')
        physical = trial['config']['physical']
        if physical['horizon'] != 6 or len(trial['history']['p'][0]) != 3:
            raise ValueError('this figure layout is for six slots and three sites')
        histories.append(dict(seed=seed, phase=trial['phase'], true_candidate_evaluation_label=trial['rows'][0].get('true_name'),
                              hardware_rated_power_kw=physical['device']['power_kw'],
                              panels=[prepare_panel(trial, method) for method in METHODS]))
        if digest(path.read_bytes()) != sha:
            raise RuntimeError('saved trial changed during figure extraction')
        inputs.append(dict(file=path.name, path=str(path), compressed_sha256=sha, compressed_bytes=len(data),
                           source_raw_sha256=(trial.get('delivery') or {}).get('source_raw_sha256'),
                           kind='slim' if '.slim.' in path.name else 'raw'))
        del trial, data
    if (directory / 'freeze.json').read_bytes() != freeze_bytes:
        raise RuntimeError('freeze changed during figure extraction')
    return dict(schema=SCHEMA, phase=freeze['phase'], predeclared_seeds=seeds, methods=list(METHODS),
                policy_identity=identity, freeze_sha256=digest(freeze_bytes), inputs=inputs, histories=histories,
                plot_source_sha256=digest(Path(__file__).read_bytes()),
                score_definition='actual M = min wrong attained joint AC SSE minus true attained joint AC SSE',
                source_scope='saved executed command curves and operations; no new fit, action or optimization',
                independent_unit='one historical sample; methods share its future noise',
                new_fits=0, new_optimizations=0, new_actions_selected=0)


def _font():
    from matplotlib import font_manager
    for filename in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc'):
        path = Path(filename)
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name(), dict(path=str(path), sha256=digest(path.read_bytes()))
    return 'DejaVu Sans', dict(path=None, note='no bundled Chinese font found; use --font on another host')


def render(data, output, *, font_path=None, layout='default'):
    parameters = layout_parameters(layout)
    output = Path(output)
    if output.exists():
        raise ValueError('use a new figure output directory; prior evidence is not overwritten')
    output.mkdir(parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'topo_ev_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if font_path is None:
        family, font = _font()
    else:
        font_path = Path(font_path); font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        font = dict(path=str(font_path), sha256=digest(font_path.read_bytes()))
    matplotlib.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                                'font.size': parameters['fontsize_pt'], 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(4, 3, figsize=(parameters['width_inches'], parameters['height_inches']), sharex=True, sharey=True)
    limit = max(history['hardware_rated_power_kw'] for history in data['histories'])
    origins = {'MILP': 'MILP返回', 'constant_plan': '恒充预案', 'v2g_fixed': '继承固定V2G',
               'v2g_mobile': '继承移动V2G', 'charge_mobile': '继承移动充电', 'charge_fixed': '继承固定充电'}
    for i, history in enumerate(data['histories']):
        for j, panel in enumerate(history['panels']):
            ax = axes[i, j]; ax.axhline(0., color='#999999', lw=.6, zorder=0)
            ax.set_ylim(-1.12 * limit, 1.12 * limit); ax.set_yticks((-limit, 0., limit))
            if not panel['slots']:
                ax.text(.5, .5, '无已保存执行曲线', transform=ax.transAxes, ha='center', va='center')
            else:
                previous = None
                for slot in panel['slots']:
                    left, right, value = slot['start_minutes'], slot['stop_minutes'], slot['net_power_kw']
                    if slot['activity'] in ('travel', 'setup'):
                        travel = slot['activity'] == 'travel'
                        ax.axvspan(left, right, color='#D9DFE5' if travel else '#F2E5CC', alpha=.8, zorder=-1)
                        ax.text((left + right) / 2, -.92 * limit, '移动' if travel else '接线', ha='center', va='center', fontsize=7)
                    if previous is not None:
                        ax.plot((left, left), (previous, value), color='#555555', lw=.8)
                    site = slot['plot_site']; color = SITE_COLORS[site] if site is not None else '#8A8A8A'
                    ax.plot((left, right), (value, value), color=color, lw=2.0, solid_capstyle='butt')
                    if slot['connected_site'] is not None:
                        ax.plot((left + right) / 2, value, 'o', ms=3.5, color=color)
                    elif slot['injection_site'] is not None:
                        ax.plot((left + right) / 2, value, '^', ms=4, color=color)
                    previous = value
                ax.set_xlim(0., panel['slots'][-1]['stop_minutes'])
            value = panel['actual_M']; score = '未取得' if value is None else f'{value:.2f}'
            origin = origins.get(panel['selected_origin'], panel['selected_origin'] or '来源未保存')
            if panel['selected_origin'] == 'MILP' and panel['is_new_MILP_curve'] is not None:
                origin = 'MILP新曲线' if panel['is_new_MILP_curve'] else 'MILP返回已有曲线'
            ax.text(.015, 1.025, f'M={score}；{origin}', transform=ax.transAxes,
                    fontsize=parameters['annotation_fontsize_pt'], va='bottom')
            ax.grid(axis='y', color='#E9E9E9', lw=.5); ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(METHOD_LABELS[j], fontsize=parameters['column_title_fontsize_pt'], y=1.24, pad=0)
            if j == 0:
                ax.set_ylabel(f'H {history["seed"]}\n网侧功率 / kW', fontsize=parameters['axis_label_fontsize_pt'])
            if i == 3:
                ax.set_xlabel('时间 / 分钟'); ax.set_xticks((0, 15, 30, 45, 60, 75, 90))
    legend = [Line2D([0], [0], color=color, marker='o', lw=2, label=f'接入点 {i+1}') for i, color in enumerate(SITE_COLORS)]
    legend += [Patch(facecolor='#D9DFE5', label='移动空档'), Patch(facecolor='#F2E5CC', label='接线空档')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.54, .955), ncol=5, frameon=False,
               fontsize=parameters['legend_fontsize_pt'])
    fig.suptitle('EV 执行曲线与接入状态：正值充电，负值放电',
                 fontsize=parameters['supertitle_fontsize_pt'], y=.997)
    caption = ('实心点：该时段已连接（含零功率）；底色：移动或接线空档。\n'
               'M：实际联合 AC 的 SSE 裕量；来源区分 MILP 新曲线与继承。') if layout == 'report' else (
               '实心点表示该时段已连接（零功率也保留）；M 为实际联合 AC 的 SSE 裕量。最终来源按保存记录标注。')
    fig.text(.5, .012, caption, ha='center', fontsize=parameters['caption_fontsize_pt'], linespacing=1.35)
    fig.subplots_adjust(**parameters['margins'])
    data = dict(data, font=font, matplotlib_version=matplotlib.__version__, render_parameters=parameters)
    text = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    (output / 'figure_data.json').write_bytes(text)
    fig.savefig(output / 'ev_saved_curves.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'ev_compact_plot'})
    fig.savefig(output / 'ev_saved_curves.png', dpi=180)
    plt.close(fig)
    manifest = dict(schema=SCHEMA, phase=data['phase'], predeclared_seeds=data['predeclared_seeds'],
                    plot_source_sha256=data['plot_source_sha256'], policy_identity=data['policy_identity'],
                    render_parameters=parameters,
                    files={name:dict(sha256=digest((output / name).read_bytes()), bytes=(output / name).stat().st_size)
                           for name in ('figure_data.json', 'ev_saved_curves.pdf', 'ev_saved_curves.png')},
                    source_scope=data['source_scope'], new_fits=0, new_optimizations=0, new_actions_selected=0)
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--seeds', nargs=4, type=int, required=True)
    parser.add_argument('--kind', choices=('auto', 'raw', 'slim'), default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--layout', choices=('default', 'report'), default='default')
    args = parser.parse_args()
    result = render(figure_data(args.directory, args.seeds, kind=args.kind), args.output,
                    font_path=args.font, layout=args.layout)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
