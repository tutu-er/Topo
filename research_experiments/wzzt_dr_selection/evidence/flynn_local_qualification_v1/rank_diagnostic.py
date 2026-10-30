"""Saved measured-input rank diagnostic only; never a DR pricing score."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pickle
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'rank_receipt.json'
    protocol_path = ROOT / 'rank_protocol.json'
    if output.exists() or protocol_path.exists() or (ROOT / 'rank_failure.json').exists():
        raise FileExistsError('do not overwrite independent diagnostic')
    inputs = [RUN / 'history_and_forecast.pkl', RUN / 'baseline_request' / 'result.json']
    protocol = {'status': 'frozen_before_calculation', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'inputs': {str(p): sha(p) for p in inputs},
        'scope': 'Rank/SVD of complete measured P/Q and selected fixed-family features; no price score or recovery claim',
        'threshold': 'numpy matrix_rank default: max(shape)*float64_eps*largest singular value',
        'new_LP_calls': 0, 'new_fits': 0, 'true_topology_or_future_read': False}
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    try:
        history = pickle.loads(inputs[0].read_bytes())['history']
        fit = json.loads(inputs[1].read_text(encoding='utf-8'))
        point = fit['models']['multistart_original_rule']['selected_uncontracted']
        n = len(history['labels'])
        observations = {}
        for role in ('training', 'validation'):
            p = np.concatenate([d['P_terminal'].to_numpy(dtype=float) for d in history[role]])
            q = np.concatenate([d['Q_terminal'].to_numpy(dtype=float) for d in history[role]])
            assert p.shape == q.shape == (288, n)
            observations[role] = (p, q)
        observations['all_history'] = tuple(np.concatenate([observations['training'][j], observations['validation'][j]]) for j in range(2))
        def stats(a):
            singular = np.linalg.svd(a, compute_uv=False)
            tolerance = float(singular[0] * max(a.shape) * np.finfo(np.float64).eps)
            rank = int(np.sum(singular > tolerance))
            assert rank == int(np.linalg.matrix_rank(a))
            return {'shape': list(a.shape), 'rank': rank, 'default_tolerance': tolerance,
                'largest_singular_value': float(singular[0]), 'smallest_singular_value': float(singular[-1]),
                'smallest_over_largest': float(singular[-1] / singular[0]),
                'condition_2': float(singular[0] / singular[-1]) if singular[-1] else None,
                'singular_values': singular.tolist()}
        records = {}
        for role, (p, q) in observations.items():
            columns = []
            for values in (p, q):
                for support in point['supports']:
                    feature = np.zeros_like(p)
                    feature[:, support] = values[:, support].sum(axis=1)[:, None]
                    columns.append(feature.ravel())
            features = np.column_stack(columns)
            records[role] = {'measured_P_Q': stats(np.column_stack((p, q))),
                             'fixed_family_RX_features': stats(features)}
        for p, expected in protocol['inputs'].items():
            assert sha(Path(p)) == expected
        receipt = {'status': 'complete', 'protocol_sha256': sha(protocol_path),
            'source_sha256': protocol['source_sha256'], 'records': records,
            'chosen_family_support_labels': [[history['labels'][i] for i in s] for s in point['supports']],
            'chosen_family_support_count': len(point['supports']), 'separate_RX_parameter_count': 2 * len(point['supports']),
            'new_LP_calls': 0, 'new_fits': 0, 'true_topology_or_future_read': False,
            'interpretation': ['actual measured input is not the exact common-input four-terminal example',
                'full column rank excludes nonzero exact fixed-family feature-null directions at the stated numerical threshold',
                'it does not exclude nonunique L1 optima, near-optimal compensation, uncertainty or competing-tree fits',
                'P/Q meter noise can create measured rank; this is not a certificate of latent physical excitation',
                'rank and condition number are diagnostics only, never a price target or topology-recovery guarantee'],
            'environment': {'python': sys.version, 'numpy': np.__version__}}
        output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': 'complete', 'records': {r: {k: {j: v[j] for j in ('shape', 'rank', 'smallest_over_largest', 'condition_2')} for k, v in values.items()} for r, values in records.items()}, 'output': str(output)}))
    except Exception as exc:
        (ROOT / 'rank_failure.json').write_text(json.dumps({'status': 'failed', 'message': str(exc),
            'traceback': traceback.format_exc(), 'new_LP_calls': 0, 'new_fits': 0}, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
