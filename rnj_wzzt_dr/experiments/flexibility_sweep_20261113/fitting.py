"""Call the unchanged identifier with observed train/validation files only."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ...history import fit_observed, freeze_history
from ...identification.algorithm import SearchConfig
from ..posterior_mse_20261112.scoring import score, independent_check, active_topology, topology_id
from .config import FIT_BUDGET


def native_json_record(native):
    """Preserve nonfinite failed-solve diagnostics without accepting bad models."""
    for row in native.get('frozen_candidates', []):
        arrays = [row[k] for k in ('R', 'X') if k in row]
        arrays += [row.get('point', {})[k] for k in ('r', 'x') if k in row.get('point', {})]
        if any(not np.isfinite(np.asarray(a, float)).all() for a in arrays):
            raise ValueError('Frozen model coefficients must remain finite')
    fields = []
    def visit(value, path):
        if isinstance(value, np.ndarray):
            return visit(value.tolist(), path)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            fields.append(dict(path=path, value=repr(value)))
            return None
        if isinstance(value, dict):
            return {k: visit(v, path+'.'+str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(v, path+f'[{i}]') for i, v in enumerate(value)]
        return value
    record = visit(native, 'native')
    record['_nonfinite_native_fields'] = fields
    return record


def pack_score(native, training, validation):
    s = score(native, training, validation)
    s['independent_verification'] = independent_check(native, training, validation, s)
    frozen = freeze_history(native, validation)
    s['parameters'] = []
    for row in s['models']:
        i = row['index']
        s['parameters'].append(dict(index=i, point=native['frozen_candidates'][i]['point'],
            H=frozen.H[i], history_sse=float(frozen.sse[i])))
    s['raw_map_index'] = s['map_index']
    s['map_index'] = next(i for i, m in enumerate(s['models']) if m['topology_id'] == s['map_topology_id'])
    s['reference_mse'] = native['reference_mse']
    s['validation_tie'] = native['config']['validation_tie']
    s['native_bank'] = [dict(H=frozen.H[i], history_sse=float(frozen.sse[i]), internal=int(frozen.internal[i]),
        topology_id=topology_id(active_topology(row['point'], training.labels, native['config']['zero_tol'])),
        active_clades=active_topology(row['point'], training.labels, native['config']['zero_tol']))
        for i, row in enumerate(native['frozen_candidates'])]
    s['representative_H_sha256'] = io.contract_hash(np.asarray([p['H'] for p in s['parameters']]))
    return s


def calibrated_means(s, validation):
    """The existing history-residual formula with an explicit validation split."""
    H = np.asarray([p['H'] for p in s['parameters']])
    power = np.column_stack((validation.p, validation.q))
    mean = np.einsum('tc,knc->ktn', power, H)
    residual = validation.y[None]-mean
    factor = 1+np.sum(mean*residual, axis=1)/(np.sum(mean*mean, axis=1)+len(power)*s['sigma2'])
    return factor[:, :, None]*H


def fit_one(task):
    began = perf_counter()
    directory = Path(task['directory'])
    if (directory/'fit_complete.json').exists():
        return io.read(directory/'fit_complete.json')
    if (directory/'fit_started.json').exists():
        raise RuntimeError('Unfinished fit lock requires review; do not reset its budget')
    io.write(directory/'fit_started.json', dict(pid=os.getpid(), budget=FIT_BUDGET, input=task['observed_refs']))
    try:
        for name, ref in task['observed_refs'].items():
            if io.ref(ref['path'])['sha256'] != ref['sha256']:
                raise ValueError('Observed fit input changed: '+name)
        train = load(task['observed_refs']['train']['path'])
        valid = load(task['observed_refs']['validation']['path'])
        native = fit_observed(train, valid, method='101', config=SearchConfig(**FIT_BUDGET))
        s = pack_score(native, train, valid)
        io.write(directory/'native_result.json', native_json_record(native))
        io.write(directory/'score.json', s)
        receipt = dict(id=task['id'], status='fitted', score=io.ref(directory/'score.json'),
            native=io.ref(directory/'native_result.json'), training_points=train.points, validation_points=valid.points,
            seconds=perf_counter()-began, method_cap_exceeded=native['total_cap_exceeded'],
            native_stop_reason=native['stop_reason'], search_budget_reached='budget' in native['stop_reason'],
            native_bound_conflicts=native['native_bound_conflicts'],
            truth_used_by_identifier=False, test_used_by_identifier=False)
    except Exception as exc:
        receipt = dict(id=task['id'], status='failed', error=repr(exc), traceback=traceback.format_exc(),
                       seconds=perf_counter()-began)
    io.write(directory/'fit_complete.json', receipt)
    return receipt
