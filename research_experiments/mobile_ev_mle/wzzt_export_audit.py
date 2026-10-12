"""Read-only probe of the existing history-only WZZT proposal contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _hash_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        digest.update(name.encode('utf-8')); digest.update(str(array.shape).encode('ascii'))
        digest.update(str(array.dtype).encode('ascii')); digest.update(array.tobytes())
    return digest.hexdigest()


def run_audit(root, input, out, *, limit=6):
    root, input, out = Path(root), Path(input), Path(out)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research_experiments.mobile_ev_mle.candidates import propose
    from research_experiments.mobile_ev_mle.likelihood import Observations
    from research_experiments.mobile_ev_probing import passive_bridge_v9 as bridge
    from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
    paths = [root/'research_experiments/mobile_ev_mle/candidates.py',
             root/'research_experiments/mobile_ev_mle/adaptive_history.py',
             root/'research_experiments/mobile_ev_mle/likelihood.py',
             root/'research_experiments/mobile_ev_probing/passive_bridge_v9.py',
             root/'research_experiments/mobile_ev_probing/wzzt_adapter.py',
             root/'rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_source.py']
    source_before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records = []
    for filename in sorted(input.glob('trial_[0-9]*_observations.npz'))[:limit]:
        # Only history channels enter proposal construction. No true state,
        # action, future observation, generator seed or future noise is read.
        with np.load(filename, allow_pickle=False) as archive:
            data = {key: np.array(archive['history_' + key]) for key in ('p', 'q', 'v', 'root_v')}
        history = Observations(**data)
        captures, rnj = [], []
        original_wzzt, original_rnj = bridge.fit_source_aware_wzzt, bridge.fit_projected_sensitivity

        def capture_wzzt(*args, **kwargs):
            result = original_wzzt(*args, **kwargs)
            selected = result.selected
            current = dict(source_inputs_is_None=kwargs.get('source_inputs') is None,
                source_p_error_max=float(np.max(kwargs.get('source_p_error', 0.))),
                source_q_error_max=float(np.max(kwargs.get('source_q_error', 0.))),
                result_summary=result.summary(), summary_keys=list(result.summary()),
                selected_solution_fields=list(selected.__dataclass_fields__) if selected else [],
                returned_path_count=len(result.path), selected_solution=None)
            if selected is not None:
                target = data['root_v'][:, None]**2 - data['v']**2
                prediction = data['p'] @ selected.r_matrix.T + data['q'] @ selected.x_matrix.T
                residual = target - prediction
                current['selected_solution'] = dict(support_indices=selected.support_indices,
                    r_atom_coefficients=selected.r_values, x_atom_coefficients=selected.x_values,
                    r_sensitivity_matrix=selected.r_matrix, x_sensitivity_matrix=selected.x_matrix,
                    h=selected.h, g_r=selected.g_r, g_x=selected.g_x,
                    raw_residual_reconstructed_for_audit=residual,
                    reconstructed_MAE=float(np.mean(np.abs(residual))), train_mae=selected.train_mae,
                    source_variables_active=kwargs.get('source_inputs') is not None,
                    raw_residual_is_exported_by_bridge=False)
                if kwargs.get('source_inputs') is None and (
                    np.any(selected.h) or np.any(selected.g_r) or np.any(selected.g_x)):
                    raise AssertionError('inactive source returned nonzero source response')
            captures.append(current)
            return result

        def capture_rnj(*args, **kwargs):
            result = original_rnj(*args, **kwargs)
            r, x, r2, condition = result
            geometry = sensitivity_geometry(r, np.zeros_like(r))
            rnj.append(dict(mode=args[0][0]['name'], estimated_r=r, estimated_x=x,
                r2=r2, condition=condition, root_depths=geometry.root_depths,
                shared_paths=geometry.shared_paths, exported_by_bridge=False))
            return result

        bridge.fit_source_aware_wzzt, bridge.fit_projected_sensitivity = capture_wzzt, capture_rnj
        try:
            bank, proposal = propose(history, run_wzzt=True, wzzt_time_limit=.3, max_candidates=8)
        finally:
            bridge.fit_source_aware_wzzt, bridge.fit_projected_sensitivity = original_wzzt, original_rnj
        passive = proposal['passive']
        supports = [row['supports'] for row in passive['candidates']]
        union = {tuple(clade) for family in supports for clade in family}
        intersection = set.intersection(*(set(map(tuple, family)) for family in supports)) if supports else set()
        def by_origin(origin_name):
            return [candidate['supports'] for candidate in passive['candidates']
                    if origin_name in candidate['proposal_origins']]
        raw_rnj = by_origin('rnj:raw_effective_pq:tolerance=0')
        centered_rnj = by_origin('rnj:time_centered_effective_pq:tolerance=0')
        final_wzzt = [support for support in passive['diagnostics']['wzzt'].get('support_indices', [])
                      if 1 < len(support) < history.n]
        trial_name = filename.name.replace('_observations.npz', '.json')
        # Truth is read only after the proposal is frozen, for offline coverage.
        trial = json.loads((input / trial_name).read_text(encoding='utf-8'))
        true_index = trial.get('true_index')
        true_name = trial['candidate_names'][true_index] if true_index is not None else None
        records.append(dict(history_id=str(trial['history_id']), history_sha256=_hash_arrays(data),
            history_shape=list(data['p'].shape), proposal=proposal,
            adapter_capture=captures, rnj_internal_capture=rnj,
            observed_structure_disagreement_clades=sorted(union - intersection),
            structural_disagreement_scope='all replaceable proposals, including generated neighborhoods',
            origin_specific_families=dict(raw_RNJ_tolerance0=raw_rnj,
                centered_RNJ_tolerance0=centered_rnj, WZZT_selected_normalized=final_wzzt),
            passive_exported_keys=list(passive), passive_candidate_keys=list(passive['candidates'][0]) if passive['candidates'] else [],
            uses_structure=True, uses_impedance_initial=False, estimates_source=False,
            source_claim_scope='actual default proposal branch, not all optional adapter branches',
            offline_evaluator_only=dict(true_name=true_name, truth_topology=trial.get('truth_topology'),
                final_n3_bank_names=[candidate.name for candidate in bank],
                true_name_in_final_bank=true_name in [candidate.name for candidate in bank] if true_name else False)))
    source_after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if source_before != source_after:
        raise ValueError('audited source changed during probe')
    summary = dict(history_count=len(records), pilot_only=True, source_sha256=source_before,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selector_input_whitelist=['history_p', 'history_q', 'history_v', 'history_root_v'],
        truth_read_only_after_proposal=True, proposal_uses_truth=False,
        uses_structure=True, uses_impedance_initial=False, estimates_source=False,
        WZZT_and_AC_same_history_extra_likelihood=False,
        bridge_exports_matrices=False, bridge_exports_atom_weights=False,
        bridge_exports_pointwise_residuals=False, bridge_exports_source_response=False,
        adapter_solution_has_matrices_weights_and_source_fields=True,
        topology_global_optimality=False, n3_repair_bank_complete_only_reduced_class=True,
        expansion_or_weight_warmstart_implemented=False, records=records)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wzzt_export_audit.json').write_text(json.dumps(_finite(summary), ensure_ascii=False,
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=6)
    args = parser.parse_args()
    result = run_audit(args.root, args.input, args.out, limit=args.limit)
    print(json.dumps(dict(history_count=result['history_count'], uses_structure=True,
        uses_impedance_initial=False, estimates_source=False)))
