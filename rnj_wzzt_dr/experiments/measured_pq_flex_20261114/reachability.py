"""Recompute relaxed power-box bounds from sealed DFS planning inputs only.

No optimizer, AC simulation, parameter fitting, or future observation is run.
Example (from the repository root)::

    python -m rnj_wzzt_dr.experiments.measured_pq_flex_20261114.reachability \
      --run outputs/measured_pq_dfs_enriched_20261114_v1 \
      --output outputs/measured_pq_dfs_enriched_reachability_20261114_v1.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy

from ..absolute_identification_20261114.solver import SymmetricRecovery
from ..customer_flexibility_20261113.placement import CandidateEvidence
from .domain import IncrementCase


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Sealed source changed: {path}')
    return dict(path=str(path), sha256=digest)


def checked(ref):
    return reference(ref['path'], ref['sha256'])


def run(directory, fractions=(.1, .25), base_kw=200., gamma=2.):
    directory = Path(directory).resolve()
    protocol = read(directory/'protocol.json')
    inputs_ref = checked(protocol['inputs'])
    original_sources = [checked(ref) for ref in protocol['sources']]
    entries = read(inputs_ref['path'])
    completion = read(directory/'complete_fraction0.10.json')
    if completion['failed_tasks'] or completion['completed_forecasts'] != completion['total_forecasts']:
        raise ValueError('Original forecast batch is incomplete')
    rows, mappings, original_quotes = [], [], []
    input_refs = []
    for entry in entries:
        refs = {key: checked(entry[key]) for key in ('score', 'resource_context', 'native_copy')}
        input_refs.append(dict(id=entry['id'], **refs))
        score, context = read(refs['score']['path']), read(refs['resource_context']['path'])
        H = np.asarray([p['H'] for p in score['parameters']], float)
        target = int(score['map_index'])
        competitors = [j for j in range(len(H)) if j != target]
        if not competitors:
            raise ValueError('Reachability comparison requires a competitor')
        nearest = min(competitors, key=lambda j: score['parameters'][j]['history_sse'])
        mapping = dict(id=entry['id'], representative_target=target,
            representative_nearest_competitor=nearest,
            target_native_index=int(score['models'][target]['index']),
            nearest_native_index=int(score['models'][nearest]['index']),
            raw_points=score['raw_points'], representative_points=len(H),
            selection='Highest historical score among different-topology representatives; no truth selection',
            representatives=[dict(representative_index=i, native_index=int(m['index']),
                topology_id=m['topology_id'], member_indices=m['member_indices'])
                for i, m in enumerate(score['models'])])
        folders = sorted((directory/'fraction0.10'/entry['id']).glob('forecast*'))
        if len(folders) != protocol['forecasts']:
            raise ValueError('Missing sealed forecast')
        for folder in folders:
            seal = read(folder/'seal.json')
            sealed_refs = {key: checked(seal[key]) for key in ('planner', 'plans', 'protocol', 'score')}
            if sealed_refs['score'] != refs['score']:
                raise ValueError('Forecast score differs from entry score')
            with np.load(sealed_refs['planner']['path'], allow_pickle=False) as arrays:
                forecast = arrays['forecast'].copy()
            quotes = read(sealed_refs['plans']['path'])
            original_quotes.append(dict(id=entry['id'], forecast=folder.name,
                seal=reference(folder/'seal.json'), sealed_inputs=sealed_refs,
                plans=quotes,
                shared_initial_receipt=reference(folder/'shared_initial_receipt.json')))
            for fraction in fractions:
                case = IncrementCase(context, score['labels'], forecast, base_kw, fraction)
                evidence = CandidateEvidence(case, H, target, competitors,
                    forecast, score['sigma2'], base_kw)
                gain_upper = evidence.power_box_upper()
                # Appending mean guards cannot change these nominal J/h/box values.
                # Use the guard-free constructor to avoid even reading residual Y.
                symmetric = SymmetricRecovery(case, score, None, forecast,
                    base_kw, mode='symmetric')
                k = next(k for k, pair in enumerate(symmetric.pairs)
                    if set(pair) == {target, nearest})
                h, z, tie = float(symmetric.h[k]), float(symmetric.z), float(symmetric.tie)
                # For h < tie, J >= 0 has only the upper feasible branch.
                required = (z+math.sqrt(z*z-2*(h-tie)))**2/2 if h < tie else None
                upper = float(gain_upper[competitors.index(nearest)])
                J0, J_upper = float(symmetric.c[k]), float(symmetric.upper[k])
                if not np.isclose(J0+upper, J_upper, rtol=1e-10, atol=1e-10):
                    raise ArithmeticError('Independent absolute and incremental bounds disagree')
                rows.append(dict(id=entry['id'], forecast=folder.name,
                    contract_fraction=float(fraction), target=target, competitor=nearest,
                    J0=J0, gamma=gamma, gain_box_upper=upper,
                    gamma_infeasible_by_bound=bool(upper < gamma-1e-8),
                    all_nominal_gain_upper=[dict(competitor=j, upper=float(u))
                        for j, u in zip(competitors, gain_upper, strict=True)],
                    symmetric_targets=symmetric.targets,
                    symmetric_alpha=float(symmetric.alpha), symmetric_z=z,
                    symmetric_history_offset=h, symmetric_tie=tie,
                    J_box_upper=J_upper, J_required=required,
                    symmetric_infeasible_by_bound=bool(required is not None and J_upper < required-1e-8),
                    upper_identity_absolute_error=abs(J0+upper-J_upper),
                    pure_P_resources=bool(np.all(case.qratio == 0)),
                    resource_node_labels=sorted({int(r['node_label']) for r in case.resources})))
        D = H[target]-H[nearest]
        n = len(score['labels'])
        nodes = sorted(set(map(int, case.node)))
        mapping['response_difference'] = dict(P_frobenius=float(np.linalg.norm(D[:, :n])),
            Q_frobenius=float(np.linalg.norm(D[:, n:])),
            controlled_P_columns_frobenius=float(np.linalg.norm(D[:, nodes])),
            controlled_node_indices=nodes,
            units='Frozen squared-voltage-response coefficients, not ohms',
            interpretation='Coefficient norms alone do not measure actual P/Q contribution; bounds use the full frozen forecast.')
        mappings.append(mapping)
    summary = []
    for fraction in fractions:
        selected = [row for row in rows if row['contract_fraction'] == fraction]
        summary.append(dict(contract_fraction=fraction, forecast_problems=len(selected),
            gain_box_upper_range=[min(r['gain_box_upper'] for r in selected), max(r['gain_box_upper'] for r in selected)],
            J_box_upper_range=[min(r['J_box_upper'] for r in selected), max(r['J_box_upper'] for r in selected)],
            gamma_infeasible_count=sum(r['gamma_infeasible_by_bound'] for r in selected),
            symmetric_infeasible_count=sum(r['symmetric_infeasible_by_bound'] for r in selected)))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        type='Analytic reachability diagnostic of existing sealed planning inputs',
        run_directory=str(directory), protocol=reference(directory/'protocol.json'),
        completion=reference(directory/'complete_fraction0.10.json'), inputs=inputs_ref,
        input_refs=input_refs, script=reference(__file__), original_sources=original_sources,
        original_source_hashes_verified=True, original_git_commit=protocol['git_commit'],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        base_kw=base_kw, fractions=list(fractions), gamma=gamma,
        conditions=dict(
            no_new_optimization=True, no_new_AC=True, no_parameter_fitting=True,
            actual_future_observations_read=False, resources='Original catalog and clock windows; only contract fraction changes',
            bound='Per-output interval maxima; relax resource energy/time coupling and all evidence/mean guards',
            absolute_condition='h + J - z*sqrt(2*J) >= tie; reported lower J threshold applies when h < tie',
            scope='Infeasibility of the declared frozen-forecast QCQP requirements; no AC or statistical claim',
            base_kw_source='Explicit common 200 kW unit of the sealed experiment; no hidden network parameters read'),
        mappings=mappings, original_quotes=original_quotes, rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=reference(args.output), summary=result['summary'])))


if __name__ == '__main__':
    main()
