"""Independent fit checks, stage attribution and size-normalized error audit."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from .run import read, write, sha
from .model import remainder
from research_experiments.local_rebuild.model import compatible, canonical
from research_experiments.local_rebuild.verify import independent_loss
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed


def independent_lp(data, point):
    p, q, y = data.p[0], data.q[0], data.target[0]; z = np.zeros((len(point['supports']), data.n))
    for row, support in zip(z, point['supports']):
        row[list(support)] = 1.
    cols = [((p@row)[:, None]*row).ravel() for row in z]+[((q@row)[:, None]*row).ravel() for row in z]
    features = csr_matrix(np.column_stack(cols)); m, k = features.shape; identity = eye(m, format='csr')
    result = linprog(np.r_[np.zeros(k), np.ones(m)/m],
        A_ub=vstack([hstack([features, -identity]), hstack([-features, -identity])], format='csr'),
        b_ub=np.r_[y.ravel(), -y.ravel()], bounds=[(0., 2.)]*k+[(0., None)]*m, method='highs')
    if not result.success:
        raise AssertionError(result.message)
    return dict(status=int(result.status), objective=float(result.fun), saved=point['objective'],
                absolute_difference=abs(float(result.fun)-point['objective']))


def supports(point, n, structural=False):
    return {tuple(s) for s, r, x in zip(point['supports'], point['r'], point['x'])
            if 1<len(s)<n and (structural or max(r, x)>1e-9)}


def size_errors(point, true, n, *, structural=False):
    pred = supports(point, n, structural); result = []
    for size in ('small', 'large'):
        inside = lambda s: (len(s)<=5) == (size=='small')
        p = {s for s in pred if inside(s)}; t = {s for s in true if inside(s)}
        result.append(dict(size=size, predicted=len(p), truth=len(t), TP=len(p&t), FP=len(p-t), FN=len(t-p)))
    return result


def blocking_rows(point, initial, true, n):
    active = supports(point, n, True); initial = supports(initial, n, True); wrong = active-true
    result = []
    for t in sorted(true):
        flags = {k: False for k in ('rnj_small', 'rnj_large', 'wzzt_small', 'wzzt_large')}
        blockers = []
        for s in sorted(wrong):
            if not compatible(s, t):
                origin = 'rnj' if s in initial else 'wzzt'
                size = 'small' if len(s)<=5 else 'large'; flags[origin+'_'+size] = True; blockers.append(s)
        result.append(dict(support=t, terminal_size=len(t), present_structurally=t in active, blocked=bool(blockers),
                           blockers=blockers, **flags))
    return result


def analyze(output):
    output = Path(output); dest = output/'analysis'; dest.mkdir(exist_ok=True)
    protocol = read(output/'protocol.json'); catalog = read(output/'catalog.json')
    checks = 0; rows = []; errors = []; blockings = []; first = []; events = []; cross = []; failures = []
    for name, digest in protocol['source_hashes'].items():
        assert sha(output/'source'/name) == digest; checks += 1
    for job in protocol['evaluation']:
        meta = catalog[job]; common_path = output/'common'/job/'result.json'; common = read(common_path)
        assert sha(common_path) == protocol['frontend_reused'][job]; checks += 1
        data = {s: prepare(observed(meta[s], meta['N'] if s=='train' else None)) for s in ('train', 'validation', 'test')}
        truth = physical_truth(Path(meta['truth_network']), common['labels']); pos = {v:i for i,v in enumerate(common['labels'])}
        true = {tuple(sorted(pos[v] for v in t)) for t in truth['clades']}; n = len(common['labels'])
        for method in protocol['settings']:
            base = dict(job=job, panel=meta['panel'], case=meta['case'], bundle=meta['bundle'], N=meta['N'],
                        condition=meta['condition'], voltage_sigma=meta['voltage_sigma'], method=method)
            file = output/'evaluation'/job/method/'result.json'
            if not file.exists():
                failures.append(dict(**base, error=read(file.with_name('error.json'))))
                for view in ('bank', 'path', 'endpoint'):
                    rows.append(dict(**base, view=view, model_available=False, clade_f1=0., exact=False))
                continue
            r = read(file); t = r['trajectory']; raw = t['states']['rnj']
            assert not r['truth_used'] and not r['test_opened'] and r['protocol_sha256'] == sha(output/'protocol.json')
            assert r['common_sha256'] == sha(common_path)
            assert r['input_hashes'] == dict(train=sha(meta['train']), validation=sha(meta['validation'])); checks += 5
            previous = float('inf')
            for point in t['path']:
                value = independent_loss(data['train'].p[0], data['train'].q[0], data['train'].target[0], point)
                assert abs(value-point['objective'])<1e-8 and value<=previous+1e-9
                previous = value; checks += 2
            for event in t['attempts']:
                if event.get('status') is not None:
                    assert event['warm_start']['valid'] and event['audit']['valid'] and event['bound_consistent_with_known_upper']; checks += 3
                if event['kind'] in ('rnj_pre_joint', 'wzzt_tail_joint'):
                    assert all(len(v['blocks'])<=5 for v in event['regions'])
                    if event['kind']=='wzzt_tail_joint':
                        assert event['maximum_remainder']<=5 and not event['skipped_large_regions']
                    checks += 2
                events.append(dict(**base, **event))
            for lp in r['LP']:
                assert lp['status']==2 and max(lp['max_constraint_violation'], lp['max_bound_violation'])<=1e-7; checks += 1
            for view, point in (('bank', r['selected']['point']), ('path', r['selected_path']['point']), ('endpoint', t['point'])):
                metric = score(point, data, truth, common['labels'], tuple(map(tuple, common['pool'])))
                rows.append(dict(**base, view=view, **metric, search_seconds=t['seconds'], elapsed_seconds=r['elapsed_seconds'],
                    total_estimate_seconds=r['elapsed_seconds']+r['common_seconds'], stop_reason=t['stop_reason'],
                    selected_origin=r['selected_origin'] if view=='bank' else view,
                    pre_invoked=any(e['kind']=='rnj_pre_joint' for e in t['attempts']),
                    pre_accepted=any(e['kind']=='rnj_pre_joint' and e['accepted'] for e in t['attempts']),
                    tail_invoked=any(e['kind']=='wzzt_tail_joint' for e in t['attempts']),
                    tail_accepted=any(e['kind']=='wzzt_tail_joint' and e['accepted'] for e in t['attempts'])))
                for q in size_errors(point, true, n):
                    errors.append(dict(**base, stage=view, structural=False, **q))
            for stage, point in t['states'].items():
                if stage=='endpoint':
                    continue
                for structural in (False, True):
                    for q in size_errors(point, true, n, structural=structural):
                        errors.append(dict(**base, stage=stage, structural=structural, **q))
            for q in blocking_rows(t['point'], t['states']['after_pre'], true, n):
                blockings.append(dict(**base, **q))
            previous_set = supports(raw, n, True); first_new = None
            accepted_events = [e for e in t['attempts'] if e['accepted']]
            for i, point in enumerate(t['path'][1:], 1):
                added = supports(point, n, True)-previous_set
                wrong = added-true
                if wrong and first_new is None:
                    first_new = dict(path_index=i, stage=accepted_events[i-1]['kind'], supports=sorted(wrong), small=sum(len(s)<=5 for s in wrong), large=sum(len(s)>5 for s in wrong))
                previous_set = supports(point, n, True)
            first.append(dict(**base, initial_wrong=len(supports(raw, n, True)-true), first_new_wrong=first_new))
            check = independent_lp(data['train'], r['selected']['point'])
            assert check['absolute_difference']<1e-8; cross.append(dict(**base, **check)); checks += 1
    frame = pd.DataFrame(rows); frame.to_csv(dest/'per_model.csv', index=False)
    summary = frame.groupby(['panel', 'view', 'method']).agg(inputs=('job', 'count'), available=('model_available', 'sum'),
        F1=('clade_f1', 'mean'), exact=('exact', 'sum'), train_MAE=('train_mae', 'mean'), validation_MAE=('validation_mae', 'mean'),
        test_MAE=('test_mae', 'mean'), search_median=('search_seconds', 'median'), pre_invoked=('pre_invoked', 'sum'),
        pre_accepted=('pre_accepted', 'sum'), tail_invoked=('tail_invoked', 'sum'), tail_accepted=('tail_accepted', 'sum')).reset_index()
    summary.to_csv(dest/'summary.csv', index=False)
    frame.groupby(['panel','view','case','method']).agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum')).reset_index().to_csv(dest/'cases.csv',index=False)
    error_frame = pd.DataFrame(errors); error_frame.to_csv(dest/'size_errors.csv', index=False)
    totals = error_frame.groupby(['panel','method','stage','structural','size'])[['predicted','truth','TP','FP','FN']].sum().reset_index()
    totals['false_discovery_rate'] = totals.FP/totals.predicted.replace(0,np.nan)
    totals['miss_rate'] = totals.FN/totals.truth.replace(0,np.nan); totals.to_csv(dest/'size_summary.csv',index=False)
    blocking_frame = pd.DataFrame(blockings); blocking_frame.to_csv(dest/'blocking_rows.csv',index=False)
    blocking_frame.groupby(['panel','method'])[['blocked','rnj_small','rnj_large','wzzt_small','wzzt_large']].sum().reset_index().to_csv(dest/'blocking_summary.csv',index=False)
    small = blocking_frame.rnj_small | blocking_frame.wzzt_small
    large = blocking_frame.rnj_large | blocking_frame.wzzt_large
    blocking_frame['category'] = np.select(
        [blocking_frame.present_structurally, small & ~large, large & ~small, small & large],
        ['present', 'small_only', 'large_only', 'both_sizes'], default='compatible_but_missing')
    blocking_frame.groupby(['panel','method','category']).size().rename('true_line_occurrences').reset_index().to_csv(dest/'blocking_exclusive.csv',index=False)
    pairs=[]
    for (panel,view), g in frame.groupby(['panel','view']):
        pivot=g.pivot(index='job',columns='method',values='clade_f1')
        for method in ('rnj_pre','wzzt_tail','both'):
            d=pivot[method]-pivot.baseline
            pairs.append(dict(panel=panel,view=view,method=method,inputs=len(d),mean_delta=d.mean(),wins=int((d>1e-9).sum()),ties=int((abs(d)<=1e-9).sum()),losses=int((d< -1e-9).sum())))
    pd.DataFrame(pairs).to_csv(dest/'paired.csv',index=False)
    write(dest/'events.json',events);write(dest/'first_errors.json',first)
    write(dest/'crosssolver.json',dict(passed=True,checks=len(cross),records=cross))
    write(dest/'verification.json',dict(passed=True,checks=checks,planned=480,returned=read(output/'completion.json')['returned'],failures=failures,independent_LP=len(cross),analysis_source_sha256=sha(__file__)))
    print(summary.to_string(index=False),flush=True); print('VERIFIED',checks,len(cross),'independent LPs',flush=True)
