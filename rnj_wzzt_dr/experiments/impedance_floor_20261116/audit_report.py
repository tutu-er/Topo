"""Read-only independent checks and summaries of the frozen floor-grid replay.

No optimizer, identifier, AC solver, or production pruning/scoring routine is
called. Every successful grid retains its original candidate denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

from ... import io
from ...observations import Observation, fingerprint, load


ROOT = Path(__file__).resolve().parents[3]


def family(point):
    return tuple(sorted((tuple(sorted(map(int, c))) for c in point['supports']), key=lambda c:(len(c), c)))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def matrix(point, n):
    supports = [tuple(map(int, c)) for c in point['supports']]
    r, x = np.asarray(point['r'], float), np.asarray(point['x'], float)
    if r.shape != (len(supports),) or x.shape != r.shape or len(set(supports)) != len(supports):
        raise ValueError('Invalid coefficient/support dimensions or duplicate supports')
    if not np.isfinite(r).all() or not np.isfinite(x).all() or min(r.min(initial=0.), x.min(initial=0.)) < 0:
        raise ValueError('Invalid response coefficients')
    result = np.zeros((n, 2*n))
    for c, rr, xx in zip(supports, r, x, strict=True):
        if not c or len(c) != len(set(c)) or min(c) < 0 or max(c) >= n:
            raise ValueError('Invalid observed support positions')
        indices = np.asarray(c, int)
        result[np.ix_(indices, indices)] += rr
        result[np.ix_(indices, indices+n)] += xx
    for a in supports:
        for b in supports:
            sa, sb = set(a), set(b)
            if sa & sb and not (sa <= sb or sb <= sa):
                raise ValueError('Non-laminar retained support family')
    return result


def loss(point, obs):
    residual = obs.y - np.c_[obs.p, obs.q]@matrix(point, obs.n).T
    return dict(mse=float(np.mean(residual**2)), mae=float(np.mean(abs(residual))),
                sse=float(np.sum(residual**2)), cells=int(residual.size))


def strict_select(candidates, validation, reference_mse, tie, new_day=None):
    """Independently reproduce scaled MSE, stored-support tie count and order."""
    records = []
    for index, row in enumerate(candidates):
        vh = loss(row['point'], validation)
        nh = None if new_day is None else loss(row['point'], new_day)
        sse, cells = vh['sse'], vh['cells']
        if nh is not None:
            sse += nh['sse']; cells += nh['cells']
        records.append(dict(source_candidate_id=row['source_candidate_id'], order=index,
            validation_mse=vh['mse'], validation_mae=vh['mae'],
            new_day_mse=None if nh is None else nh['mse'],
            new_day_mae=None if nh is None else nh['mae'], scored_mse=sse/cells,
            internal_count=sum(1 < len(c) < validation.n for c in row['point']['supports'])))
    if not records or len({r['source_candidate_id'] for r in records}) != len(records):
        raise ValueError('Candidate IDs are missing or duplicated')
    values = np.asarray([r['scored_mse']/reference_mse for r in records])
    eligible = np.flatnonzero(values <= float(values.min())+tie)
    selected = min(eligible, key=lambda i:(records[i]['internal_count'], values[i], i))
    return dict(selected=records[selected]['source_candidate_id'],
                eligible=[records[i]['source_candidate_id'] for i in eligible],
                scoring_cells=cells, losses=records)


def support_metrics(point, truth, n, zero_tol):
    truth = {tuple(sorted(c)) for c in truth}
    active = {tuple(sorted(c)) for c, r, x in zip(point['supports'], point['r'], point['x'], strict=True)
              if max(r, x) > zero_tol}
    result = dict(active_support_count=len(active), common_active=tuple(range(n)) in active,
                  active_singletons=sum(len(c) == 1 for c in active))
    for name, keep in [('full', lambda c:True), ('internal', lambda c:1 < len(c) < n),
                       ('singleton', lambda c:len(c) == 1), ('common', lambda c:len(c) == n)]:
        a, t = {c for c in active if keep(c)}, {c for c in truth if keep(c)}
        tp, fp, fn = len(a&t), len(a-t), len(t-a)
        result.update({name+'_exact': a == t, name+'_extra':fp, name+'_missing':fn,
                       name+'_tp':tp, name+'_true_count':len(t),
                       name+'_precision':tp/(tp+fp) if tp+fp else float(not t),
                       name+'_recall':tp/(tp+fn) if tp+fn else 1.,
                       name+'_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.})
    return result


def strength(point, setting, calibration, zbase):
    r, x = np.asarray(point['r'], float), np.asarray(point['x'], float)
    if setting['metric'] == 'ohm':
        return np.sqrt(r*r+x*x)*zbase/2.
    return (calibration['a']*r+calibration['b']*x)/calibration['median_root_depth']


def close(actual, expected, label, *, atol=1e-13, rtol=3e-8):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(label+': one value is unavailable')
    elif not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False):
        raise ValueError(f'{label}: {actual!r} != {expected!r}')


def qp_certificate(point, train, reference, bound):
    """Rebuild raw support features and the convex box supporting-plane bound."""
    n, rows = train.n, len(train.y)
    columns = []
    for values in (train.p, train.q):
        for c in point['supports']:
            z = np.zeros(n); z[list(c)] = 1.
            columns.append((values@z)[:, None]*z[None, :])
    F = np.stack(columns, axis=-1).reshape(rows*n, -1) if columns else np.empty((rows*n, 0))
    theta = np.r_[point['r'], point['x']]
    if np.any(theta < 0.) or np.any(theta > bound):
        raise ValueError('Fixed-support coefficients exceed their original box')
    F, target, theta = F.astype(np.longdouble), train.y.ravel().astype(np.longdouble), theta.astype(np.longdouble)
    residual = F@theta-target
    objective = float(np.mean(residual*residual)/reference)
    gradient = 2*F.T@residual/(target.size*reference)
    lower = max(0., objective+float(np.minimum(-gradient*theta, gradient*(bound-theta)).sum()))
    return dict(objective=objective, lower_bound=lower, certificate_gap=objective-lower)


def check_pruning(original, result, receipt, setting, calibration, zbase, cache, train):
    """Replay deletion decisions with saved QP points, never refitting them."""
    if setting['threshold'] == 0.:
        if canonical_json(original) != canonical_json(result):
            raise ValueError('Zero threshold changed a field of the original frozen point')
        if receipt['steps'] or receipt.get('deletions') != 0 or receipt.get('refits') != 0:
            raise ValueError('Zero threshold performed contraction or refitting')
        return dict(deletions=0, removed_singletons=0, removed_common=0)
    f = float(setting['threshold']); tol = f*1e-10
    close(receipt['numerical_absolute_tolerance'], tol, 'threshold tolerance', atol=0.)
    if receipt['observed_labels'] != list(train.labels) or receipt['observations_removed'] or receipt['fixed_supports_added']:
        raise ValueError('Contraction changed the measurement columns or added a basis')
    current = original
    counts = dict(deletions=0, removed_singletons=0, removed_common=0)
    for step in receipt['steps']:
        supports = [tuple(sorted(c)) for c in current['supports']]
        values = strength(current, setting, calibration, zbase)
        bad = [i for i, c in enumerate(supports)
               if (setting['scope'] == 'all' or len(c) > 1) and values[i]+tol < f]
        if not bad:
            raise ValueError('A deletion was made with no violating eligible support')
        k = min(bad, key=lambda i:(float(values[i]), len(supports[i]), supports[i]))
        removed = supports[k]
        if list(removed) != step['removed_support']:
            raise ValueError('Deletion is not the weakest violating support in canonical tie order')
        close(step['removed_strength'], values[k], 'deleted strength')
        remaining = tuple(sorted((c for i,c in enumerate(supports) if i != k), key=lambda c:(len(c),c)))
        if step['support_count_before'] != len(supports) or step['support_count_after'] != len(supports)-1:
            raise ValueError('An iteration did not delete exactly one support')
        if remaining not in cache:
            raise ValueError('An intermediate refit point is absent from the sealed cache')
        nxt = cache[remaining]
        if family(nxt) != remaining:
            raise ValueError('Fixed-support refit reintroduced or lost an unrequested support')
        before_loss, after_loss = loss(current, train), loss(nxt, train)
        for key in ('mse', 'mae'):
            close(step['training_loss_before'][key], before_loss[key], 'trace before '+key)
            close(step['training_loss_after'][key], after_loss[key], 'trace after '+key)
        counts['deletions'] += 1
        counts['removed_singletons'] += len(removed) == 1
        counts['removed_common'] += len(removed) == train.n
        current = nxt
    if canonical_json(current) != canonical_json(result):
        raise ValueError('Final point differs from the last sealed refit')
    values = strength(result, setting, calibration, zbase)
    eligible = [i for i,c in enumerate(result['supports']) if setting['scope'] == 'all' or len(c) > 1]
    if any(values[i]+tol < f for i in eligible):
        raise ValueError('A final retained support is below its threshold after refitting')
    close(receipt['minimum_final_strength'], min((float(values[i]) for i in eligible), default=None),
          'minimum final eligible strength')
    if len(receipt['steps']) > len(original['supports']) or not receipt['floor_satisfied']:
        raise ValueError('Termination or final floor certificate is invalid')
    return counts


def response_metrics(point, Hstar, train, validation, resource_nodes=None):
    if Hstar is None:
        return None
    H = matrix(point, train.n); error = H-Hstar; n = train.n
    result = {}
    for axis, block in [('R', slice(0,n)), ('X', slice(n,2*n))]:
        denom = np.linalg.norm(Hstar[:, block])
        result[axis+'_relative_error'] = float(np.linalg.norm(error[:, block])/denom) if denom else None
    for name, obs in [('training', train), ('validation', validation)]:
        result[name+'_response_error_RMS'] = float(np.sqrt(np.mean((np.c_[obs.p,obs.q]@error.T)**2)))
    result['resource_columns'] = {}
    if resource_nodes:
        positions = {label:i for i,label in enumerate(train.labels)}
        for axis, offset in [('P',0), ('Q',n)]:
            labels = sorted(set(resource_nodes[axis])); idx = [positions[x]+offset for x in labels]
            denom = np.linalg.norm(Hstar[:, idx])
            result['resource_columns'][axis] = dict(labels=labels, count=len(idx),
                absolute_error_norm=float(np.linalg.norm(error[:,idx])),
                relative_error=float(np.linalg.norm(error[:,idx])/denom) if denom else None,
                scope='registered device columns; no device-action or AC replay')
    return result


class Audit:
    def __init__(self, run):
        self.run = Path(run).resolve(); self.refs = {}; self.errors = []; self.cells = []; self.inputs = []
        self.qp_count = 0; self.trace_count = 0; self.calibration_count = 0
        self.point_loss_cache = {}

    def checked(self, reference):
        p = Path(reference['path']).resolve(); key = str(p)
        digest = reference['sha256']
        if key in self.refs and self.refs[key]['sha256'] != digest:
            raise ValueError('Conflicting expected hashes: '+key)
        if key not in self.refs:
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError('Source/hash mismatch: '+key)
            self.refs[key] = dict(path=key, sha256=digest)
        return p

    def read(self, reference):
        return io.read(self.checked(reference))

    def local(self, name):
        reference = io.ref(self.run/name); return self.read(reference)

    def observation(self, reference, interval, labels):
        whole = load(self.checked(reference)); start, stop = interval
        if list(whole.labels) != labels or not 0 <= start < stop <= whole.points:
            raise ValueError('Saved observation time range or label order differs')
        return Observation(whole.p[start:stop], whole.q[start:stop], whole.y[start:stop], whole.labels)

    def truth(self, entry):
        meta = self.read(entry['evaluation_case_refs']['metadata.json'])
        with self.checked(entry['evaluation_case_refs']['branches.csv']).open(encoding='utf-8', newline='') as stream:
            edges = [r for r in csv.DictReader(stream) if str(r.get('status','True')).lower() in ('true','1')]
        parents = {}
        for row in edges:
            child, parent = int(row['to_bus']), int(row['from_bus'])
            if child in parents: raise ValueError('Physical branch table has multiple parents')
            parents[child] = parent
        root = int(meta['root_bus']); descendants = defaultdict(set)
        for i, label in enumerate(entry['labels']):
            cursor, visited = int(label), set()
            while cursor != root:
                if cursor in visited or cursor not in parents:
                    raise ValueError('Measured node has no unique root path')
                visited.add(cursor); descendants[cursor].add(i); cursor = parents[cursor]
        supports = {tuple(sorted(v)) for v in descendants.values()}
        Hstar = None
        if entry.get('Hstar_ref'):
            with np.load(self.checked(entry['Hstar_ref']), allow_pickle=False) as z:
                Hstar = np.asarray(z['Hstar'], float)
                if list(map(int,z['labels'])) != entry['labels']:
                    raise ValueError('True response labels changed')
            # Reconstruct coefficients from branch data; old pressure response
            # values are intentionally unavailable and never substituted here.
            zbase = float(meta['base_kv'])**2/float(meta['base_mva']); n=len(entry['labels'])
            direct = np.zeros((n,2*n))
            for row in edges:
                c = sorted(descendants.get(int(row['to_bus']), ()))
                if c:
                    direct[np.ix_(c,c)] += 2*float(row['r_ohm'])/zbase
                    direct[np.ix_(c,[v+n for v in c])] += 2*float(row['x_ohm'])/zbase
            close(Hstar, direct, 'physical response reconstruction', atol=1e-12)
        return supports, Hstar

    def resources(self, entry):
        if entry['source_group'] not in ('history10','history288'): return None
        inventory = self.read(entry['source_index_ref'])
        rows = inventory['inputs'] if isinstance(inventory,dict) else inventory
        owner = next(r for r in rows if r['id'] == entry['original_id']) if entry['source_group'] == 'history288' else next(r for r in rows if r['case']==entry['case'])
        spec = self.read(owner['equipment_spec'])
        devices = spec['devices']
        return dict(P=[int(d['node_label']) for d in devices],
                    Q=[int(d['node_label']) for d in devices if d['kind'] in ('pv','battery')])

    def verify_selection(self, saved, candidates, obs, reference, tie, new=None):
        direct = strict_select(candidates, obs, reference, tie, new)
        for key in ('selected','eligible','scoring_cells'):
            if saved['selection'][key] != direct[key]:
                raise ValueError('Strict validation/cumulative selection mismatch: '+key)
        if len(saved['losses']) != len(direct['losses']): raise ValueError('Scored candidate denominator changed')
        for actual, expected in zip(saved['losses'], direct['losses'], strict=True):
            if actual['source_candidate_id'] != expected['source_candidate_id']:
                raise ValueError('Candidate order changed in scoring')
            for key in ('validation_mse','validation_mae','new_day_mse','new_day_mae','scored_mse','internal_count'):
                close(actual[key], expected[key], 'scoring '+key)
        return direct

    def evaluate(self, entry, plan, evaluated, train, val, new, truth, Hstar, resource_nodes, seal, baseline=None):
        setting = plan['setting']; candidates = plan['candidates']
        base = dict(input_id=entry['id'], source_group=entry['source_group'], case=entry['case'],
                    source_condition=entry.get('source_condition'), condition_value=entry.get('condition_value'),
                    phase_condition=entry.get('phase_condition'), physical_history_group=entry.get('physical_history_group'),
                    training_points=train.points, validation_points=val.points, new_points=0 if new is None else new.points,
                    metric=setting['metric'], scope=setting['scope'], threshold=setting['threshold'], setting_id=setting['id'])
        if plan['status'] != 'complete':
            if evaluated['status'] != 'unavailable_failed_candidate_bank':
                raise ValueError('Failed candidate bank was nevertheless selected')
            for phase in ('historical', 'cumulative') if new is not None else ('historical',):
                self.cells.append(dict(base, phase=phase, status='failed', error=plan.get('error'),
                                       failed_candidates=len(plan['failed_candidate_ids'])))
            return
        topology = {r['source_candidate_id']:support_metrics(r['point'],truth,train.n,seal['zero_tol']) for r in candidates}
        for stored in evaluated['candidate_topologies']:
            direct = topology[stored['source_candidate_id']]
            for key,value in stored.items():
                if key != 'source_candidate_id' and value != direct[key]: raise ValueError('Recorded topology metric differs: '+key)
        coverage = dict(bank_has_internal_truth=any(v['internal_exact'] for v in topology.values()),
                        bank_has_full_truth=any(v['full_exact'] for v in topology.values()))
        for key,value in coverage.items():
            if evaluated[key] != value: raise ValueError('Candidate coverage mismatch: '+key)
        for phase, newobs in [('historical',None)]+([] if new is None else [('cumulative',new)]):
            direct = self.verify_selection(evaluated[phase], candidates,val,seal['reference_mse'],seal['validation_tie'],newobs)
            if phase == 'historical': self.verify_selection(plan['historical_scoring'],candidates,val,seal['reference_mse'],seal['validation_tie'])
            chosen = next(r for r in candidates if r['source_candidate_id']==direct['selected'])
            selected = topology[direct['selected']]
            key = 'history_selected_topology' if phase == 'historical' else 'cumulative_selected_topology'
            if any(evaluated[key][k] != selected[k] for k in evaluated[key]): raise ValueError('Selected topology differs')
            scored = next(r for r in direct['losses'] if r['source_candidate_id']==direct['selected'])
            row = dict(base, phase=phase, status='complete', candidate_count=len(candidates),
                       selected=direct['selected'], **coverage, **selected,
                       train_mse=loss(chosen['point'],train)['mse'], **{k:scored[k] for k in ('validation_mse','new_day_mse','scored_mse')},
                       response=response_metrics(chosen['point'],Hstar,train,val,resource_nodes),
                       processing_seconds=plan.get('seconds',0.))
            if baseline is not None:
                for kind in ('internal','full'):
                    before = baseline[phase][kind+'_exact']; after=selected[kind+'_exact']
                    row.update({kind+'_newly_recovered':not before and after, kind+'_lost':before and not after})
            self.cells.append(row)

    def one(self, entry, complete_record):
        directory = Path('inputs')/entry['id']
        result = complete_record if complete_record['status']=='worker_failed' else (
            self.read(complete_record['result']) if complete_record.get('result') else self.local(directory/'result.json'))
        if result['status'] != 'complete':
            self.inputs.append(dict(input_id=entry['id'],source_group=entry['source_group'],case=entry['case'],
                                    status='failed',error=result.get('error'),planned_grid_cells=complete_record['planned_grid_cells']))
            for setting in [dict(id='original_frozen',metric=None,scope=None,threshold=None),*self.settings(entry)]:
                for phase in ('historical','cumulative') if entry.get('new_day_ref') else ('historical',):
                    self.cells.append(dict(input_id=entry['id'],source_group=entry['source_group'],case=entry['case'],
                        source_condition=entry.get('source_condition'),condition_value=entry.get('condition_value'),
                        metric=setting['metric'],scope=setting['scope'],threshold=setting['threshold'],setting_id=setting['id'],
                        phase=phase,status='failed',error=result.get('error')))
            return
        seal = self.read(result['seal']); banks=self.read(seal['banks']); calibration_receipt=self.read(seal['calibration'])
        qp_records=self.read(seal['qp_records']); cache_rows=self.read(seal['qp_cache_points'])
        for ref in seal['original_source_refs']+result['evaluation_source_refs']+entry.get('executed_source_refs',[]): self.checked(ref)
        native = self.read(banks['original_native']); original = {r['index']:r['point'] for r in native['frozen_candidates']}
        raw_native=self.read(entry['native_ref']); raw_native=raw_native[entry['native_wrapper_key']] if entry['native_wrapper_key'] else raw_native
        if canonical_json(native) != canonical_json(raw_native): raise ValueError('Copied original native bank changed')
        order=[r['index'] for r in native['frozen_candidates']]
        train=self.observation(entry['train_ref'],entry['train_rows'],entry['labels'])
        val=self.observation(entry['validation_ref'],entry['validation_rows'],entry['labels'])
        new=None if entry.get('new_day_ref') is None else self.observation(entry['new_day_ref'],entry['new_day_rows'],entry['labels'])
        if fingerprint(train)!=seal['training_sha256'] or fingerprint(val)!=seal['validation_sha256']:
            raise ValueError('Historical input fingerprints changed')
        if seal['truth_used'] or seal['new_day_used']: raise ValueError('Candidate seal used excluded data')
        if seal['reference_mse'] != native['reference_mse'] or seal['validation_tie']!=native['config']['validation_tie']:
            raise ValueError('Original scoring scale or tie tolerance changed')
        calibration=calibration_receipt['calibration']
        if calibration is not None:
            if calibration['training_sha256']!=fingerprint(train) or calibration['labels']!=list(train.labels):
                raise ValueError('Calibration belongs to another input')
            close(calibration['a'],.75/calibration['mu_r'],'calibration a')
            close(calibration['b'],.25/calibration['mu_x'],'calibration b')
            if calibration['median_root_depth']<=0: raise ValueError('Nonpositive relative scale')
            self.calibration_count+=1
        cache={family(row['point']):row['point'] for row in cache_rows}
        if len(cache)!=len(cache_rows): raise ValueError('Duplicate fixed-support cache family')
        if any(family(row['point'])!=tuple(tuple(c) for c in row['family']) for row in cache_rows):
            raise ValueError('Cache family label differs from point supports')
        records_by_family={tuple(tuple(c) for c in r['family']):r for r in qp_records}
        qp_maxgap=0.
        for key,p in cache.items():
            direct=qp_certificate(p,train,seal['reference_mse'],seal['coefficient_bound']); record=records_by_family[key]
            close(p['objective'],direct['objective'],'cached QP objective',atol=1e-10)
            close(p['mse'],direct['objective']*seal['reference_mse'],'cached QP raw MSE')
            for name in ('objective','lower_bound','certificate_gap'):
                close(record[name],direct[name],'QP '+name,atol=1e-8)
            if direct['certificate_gap']>1.1e-6*(1+direct['objective']): raise ValueError('Cached QP certificate is not closed to declared tolerance')
            qp_maxgap=max(qp_maxgap,direct['certificate_gap']); self.qp_count+=1
        truth,Hstar=self.truth(entry); resource_nodes=self.resources(entry)
        baseline=banks['baseline']
        if [c['source_candidate_id'] for c in baseline['candidates']]!=order: raise ValueError('Original candidate order changed')
        for row in baseline['candidates']:
            if canonical_json(row['point'])!=canonical_json(original[row['source_candidate_id']]): raise ValueError('Original baseline point changed')
        self.evaluate(entry,baseline,result['baseline'],train,val,new,truth,Hstar,resource_nodes,seal)
        base_rows={r['phase']:r for r in self.cells if r['input_id']==entry['id'] and r['setting_id']=='original_frozen'}
        if base_rows['historical']['selected']!=native['selected']['index']: raise ValueError('Original strict validation winner not reproduced')
        expected=self.settings(entry)
        if [p['setting'] for p in banks['grid']]!=expected or [r['setting'] for r in result['grid']]!=expected:
            raise ValueError('Grid cells are missing, duplicated or reordered')
        failures=0; deleted=0
        for plan,evaluated in zip(banks['grid'],result['grid'],strict=True):
            if [c['source_candidate_id'] for c in plan['candidates']]!=order: raise ValueError('Pruning removed a source competitor')
            failed_ids=[]
            for c in plan['candidates']:
                if c['status']=='complete':
                    counts=check_pruning(original[c['source_candidate_id']],c['point'],c['pruning'],plan['setting'],calibration,seal['zbase_ohm'],cache,train)
                    deleted+=counts['deletions']; self.trace_count+=1
                else:failed_ids.append(c['source_candidate_id'])
            if failed_ids!=plan['failed_candidate_ids'] or bool(failed_ids)!=(plan['status']!='complete'):
                raise ValueError('Failed candidate ledger differs from complete-grid declaration')
            if plan['status']!='complete':
                failures+=1
            self.evaluate(entry,plan,evaluated,train,val,new,truth,Hstar,resource_nodes,seal,base_rows)
        if failures!=result['failed_grid_cells'] or len(expected)-failures!=result['successful_grid_cells']:
            raise ValueError('Recorded successful/failed grid denominator mismatch')
        self.inputs.append(dict(input_id=entry['id'],source_group=entry['source_group'],case=entry['case'],status='complete',
            planned_grid_cells=len(expected),failed_grid_cells=failures,successful_grid_cells=len(expected)-failures,
            calibration_status=calibration_receipt['status'],calibration_seconds=result['calibration_seconds'],
            seconds=result['seconds'],unique_QP_fits=len(qp_records),qp_certificate_maxgap=qp_maxgap,
            qp_seconds=sum(r.get('seconds',0.) for r in qp_records),
            qp_soft_timeouts=sum(bool(r.get('soft_cap_exceeded')) for r in qp_records),deletion_events_across_grid=deleted))

    def settings(self, entry):
        metrics=[('rx75',self.protocol['relative_grid'])]
        if entry['source_group'] in self.protocol['physical_grid_groups']: metrics.append(('ohm',self.protocol['physical_ohm_grid']))
        return [dict(id=f'{metric}__{scope}__f{str(value).replace(".","p")}',metric=metric,scope=scope,threshold=value)
                for metric,values in metrics for scope in self.protocol['scopes'] for value in values]

    def execute(self):
        self.protocol=self.local('protocol.json'); complete=self.local('complete.json')
        if complete['status']!='complete': raise ValueError('The run has not finished; no final denominator audit')
        entries=self.read(self.protocol['inputs']); records={r['input_id']:r for r in complete['records']}
        if len(entries)!=self.protocol['planned_inputs'] or len({r['id'] for r in entries})!=len(entries): raise ValueError('Input ledger denominator mismatch')
        if set(records)!={e['id'] for e in entries} or len(records)!=len(complete['records']): raise ValueError('Completion ledger is incomplete or duplicated')
        for ref in self.protocol['sources']:
            self.checked(ref)
            relative=Path(ref['path']).resolve().relative_to(ROOT)
            self.checked(dict(path=str(self.run/'source_snapshot'/relative),sha256=ref['sha256']))
        planned=sum(len(self.settings(e)) for e in entries)
        if planned!=self.protocol['planned_grid_cells']: raise ValueError('Grid denominator differs from protocol')
        for i,entry in enumerate(entries):
            try: self.one(entry,records[entry['id']])
            except Exception as exc:
                self.errors.append(dict(input_id=entry['id'],error=type(exc).__name__+': '+str(exc)))
            if (i+1)%20==0: print(json.dumps(dict(audited_inputs=i+1,planned_inputs=len(entries),audit_errors=len(self.errors))),flush=True)
        successful=sum(r.get('successful_grid_cells',0) for r in self.inputs)
        if not self.errors and (successful!=complete['successful_grid_cells'] or planned-successful!=complete['failed_grid_cells']):
            self.errors.append(dict(error='Root completion successful/failed grid totals disagree'))
        return dict(status='passed' if not self.errors else 'failed',source_run=str(self.run),
            source_protocol=io.ref(self.run/'protocol.json'),planned_inputs=len(entries),audited_inputs=len(self.inputs),
            planned_grid_cells=planned,successful_grid_cells=successful,failed_grid_cells=planned-successful,
            independent_qp_certificates=self.qp_count,independent_pruning_traces=self.trace_count,
            calibration_scale_checks=self.calibration_count,
            calibration_scope='Checks frozen training identity and scalar normalization; does not rerun ordered sensitivity QP',
            errors=self.errors,inputs=self.inputs,source_inputs=list(self.refs.values()),
            postprocessing_sources=[io.ref(Path(__file__)),io.ref(Path(io.__file__))],
            no_new_optimization=True,no_new_AC=True)


def summarize(cells):
    groups=defaultdict(list)
    dimensions=('source_group','case','source_condition','condition_value','metric','scope','threshold','phase')
    for row in cells:
        key=tuple(row.get(k) for k in dimensions); groups[key].append(row)
        aggregate=(row['source_group'],'ALL',None,None,row.get('metric'),row.get('scope'),row.get('threshold'),row['phase'])
        groups[aggregate].append(row)
    out=[]
    for key,rows in groups.items():
        successful=[r for r in rows if r['status']=='complete']
        r=dict(zip(dimensions,key));r.update(planned=len(rows),completed=len(successful),failed=len(rows)-len(successful))
        for flag in ('internal_exact','full_exact','bank_has_internal_truth','bank_has_full_truth','internal_newly_recovered','internal_lost','full_newly_recovered','full_lost'):
            r[flag]=sum(bool(v.get(flag)) for v in successful)
        for kind in ('full','internal','singleton','common'):
            for metric in ('extra','missing','tp'): r[kind+'_'+metric]=sum(v[kind+'_'+metric] for v in successful)
            tp,fp,fn=(r[kind+'_'+s] for s in ('tp','extra','missing'))
            r[kind+'_pooled_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else (1. if successful else None)
        for name in ('train_mse','validation_mse','new_day_mse','scored_mse','processing_seconds'):
            values=[v[name] for v in successful if v.get(name) is not None]
            r[name+'_mean']=float(np.mean(values)) if values else None
        for axis in ('R','X'):
            values=[v['response'][axis+'_relative_error'] for v in successful if v.get('response') and v['response'][axis+'_relative_error'] is not None]
            r[axis+'_relative_error_count']=len(values); r[axis+'_relative_error_mean']=float(np.mean(values)) if values else None
        out.append(r)
    return sorted(out,key=lambda r:tuple(str(r.get(k)) for k in dimensions))


def markdown(audit,report):
    lines=['# 阻抗下限逐条收缩：独立核验与历史重放结果','',
        f"核验状态：**{audit['status']}**。输入 {audit['audited_inputs']}/{audit['planned_inputs']}；计划格 {audit['planned_grid_cells']}，成功 {audit['successful_grid_cells']}，失败 {audit['failed_grid_cells']}。",'',
        '本报告重用已用历史与封存新日；没有新增结构搜索、AC或灵活性采购。不同批次、网络、训练长度和压力条件分别统计。零阈值在两种指标和范围下重复出现，属于同一个原始基线，不增加独立样本数。', '',
        '“内部恢复”只比较 proper 内部支持；“完整恢复”比较全部活动支持，包含公共枝和单节点枝。删除真实单节点仍计遗漏，其观测列始终保留。', '',
        '每格全部原生候选处理完成才允许选模；任一候选失败则该格保留失败。QP上下界仅针对固定支持拟合，逐条贪心删除没有全局拓扑最优保证。', '',
        '## 分批总体结果','',
        '|批次|评分|指标/范围|下限|成功/计划|内部恢复|完整恢复|内部新恢复/丢失|完整新恢复/丢失|完整多余/遗漏|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['groups']:
        if r['case']!='ALL':continue
        name='原始基线' if r['metric'] is None else r['metric']+'/'+r['scope']
        f='—' if r['threshold'] is None else f"{r['threshold']:g}"
        lines.append(f"|{r['source_group']}|{r['phase']}|{name}|{f}|{r['completed']}/{r['planned']}|{r['internal_exact']}|{r['full_exact']}|{r['internal_newly_recovered']}/{r['internal_lost']}|{r['full_newly_recovered']}/{r['full_lost']}|{r['full_extra']}/{r['full_missing']}|")
    lines+=['','## 数值与证书范围','',
        f"独立重算 {audit['independent_pruning_traces']} 条候选轨迹、{audit['independent_qp_certificates']} 个缓存固定支持QP的原始损失、梯度支持平面下界与最终支路门槛。",'',
        'RX75 校准核对训练哈希、标签及固定标量关系；本次核验不重新求解预树灵敏度QP。当前56输入另外给出R/X矩阵Frobenius误差、历史P/Q响应误差、已登记设备节点的P/Q列误差；旧400的压力参数未提供对应真实响应，参数误差记为不可用。设备列误差不等于具体采购动作误差。', '',
        '完整逐输入/网格/评分结果在 cells.csv 和 cells.jsonl；分网络、训练量、噪声/压力条件聚合在 report.json。表中累计评分按验证与新日总观测单元数加权，旧400没有独立新日。', '',
        '错误、失败、软超时和运行时间列于 audit.json / report.json；未按结果丢弃输入或竞争候选。']
    if audit['errors']:
        lines+=['','## 核验错误','']+[f"- {r.get('input_id','全局')}: {r['error']}" for r in audit['errors']]
    return '\n'.join(lines)+'\n'


def execute(run,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing audit/report: '+str(out))
    out.mkdir(parents=True); began=perf_counter(); worker=Audit(run)
    audit=worker.execute(); audit['seconds']=perf_counter()-began
    report=dict(status=audit['status'],source_audit=io.ref(Path(__file__)),
        interpretation='development/history replay; no new identification, AC or procurement',
        planned_inputs=audit['planned_inputs'],planned_grid_cells=audit['planned_grid_cells'],
        groups=summarize(worker.cells),inputs=audit['inputs'],
        input_failures=[r for r in audit['inputs'] if r['status']!='complete'],
        qp_soft_timeouts=sum(r.get('qp_soft_timeouts',0) for r in audit['inputs']))
    io.write(out/'audit.json',audit); report['source_audit']=io.ref(out/'audit.json')
    io.write(out/'report.json',report)
    with (out/'cells.jsonl').open('w',encoding='utf-8') as stream:
        for row in worker.cells:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    keys=sorted({k for row in worker.cells for k in row if k!='response'})
    with (out/'cells.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r.get(k) for k in keys} for r in worker.cells)
    (out/'REPORT.md').write_text(markdown(audit,report),encoding='utf-8')
    io.write(out/'complete.json',dict(status=audit['status'],audit=io.ref(out/'audit.json'),report=io.ref(out/'report.json'),
        records=io.ref(out/'cells.jsonl'),csv=io.ref(out/'cells.csv'),markdown=io.ref(out/'REPORT.md')))
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);result=execute(args.run,args.out)
    print(json.dumps({k:result[k] for k in ('status','audited_inputs','planned_inputs','successful_grid_cells','failed_grid_cells')},ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
