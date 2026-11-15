"""Original probability laws for the historical native winner alone."""
from __future__ import annotations

import numpy as np

from ...dr.selection_mse import choose_mse
from ..absolute_identification_20261114.solver import target_set
from ..distribution_gate_20261114.evidence import (
    DistributionEvidence, METER_RELATIVE_SD, FORECAST_RELATIVE_SD,
)


def historical_target(score, validation):
    """Verify the original strict validation/tie choice without truth or refitting."""
    bank = score['native_bank']
    cells = int(score['cells'])
    if cells != np.asarray(validation.y).size or cells <= 0:
        raise ValueError('Historical validation cells differ from the frozen score')
    selected = score['native_index']
    if isinstance(selected, bool) or int(selected) != selected or not 0 <= int(selected) < len(bank):
        raise ValueError('Invalid historical native_index')
    selected = int(selected)
    history = np.asarray([p['history_sse'] for p in bank], float)
    counts = np.asarray([p['internal'] for p in bank])
    power = np.column_stack((validation.p, validation.q))
    # Match freeze_history: mean residual square multiplied by observation cells.
    direct = np.asarray([float(np.square(validation.y-power@np.asarray(p['H'], float).T).mean())*cells
                         for p in bank])
    scale = max(float(np.max(np.abs(history), initial=0.)), np.finfo(float).tiny)
    if not np.allclose(direct, history, rtol=1e-11, atol=64*np.finfo(float).eps*scale):
        raise ValueError('Direct historical validation SSE differs from the frozen bank')
    options = dict(K=1/float(score['reference_mse']), epsilon_scaled=float(score['validation_tie']))
    saved_choice = choose_mse(history/cells, counts, **options)
    direct_choice = choose_mse(direct/cells, counts, **options)
    if saved_choice['selected'] != selected or direct_choice['selected'] != selected:
        raise ValueError('native_index differs from strict historical MSE/tie selection')
    return selected, dict(
        selected_native=selected, saved_history_selected=saved_choice['selected'],
        direct_history_selected=direct_choice['selected'],
        eligible_native_indices=saved_choice['eligible'], validation_cells=cells,
        maximum_history_sse_reconstruction_error=float(np.max(np.abs(direct-history), initial=0.)),
        rule='Original strict validation MSE/reference_MSE; numerical ties by internal count, scaled MSE, then native order',
        data_scope='Frozen historical validation P/Q/Y and native coefficients only; no truth, future observations or refit',
    )


class SingleTargetEvidence(DistributionEvidence):
    """Filter unchanged full-bank formulas; keep only the winner's forward guards."""

    def __init__(self, case, score, validation, forecast, base_kw,
                 mode='mgf_forecast', target_mass=.99, lambda_start=.5, *,
                 meter_relative_sd=METER_RELATIVE_SD,
                 forecast_relative_sd=FORECAST_RELATIVE_SD):
        native, selection = historical_target(score, validation)
        representatives = [int(p['index']) for p in score['parameters']]
        if native not in representatives:
            raise ValueError('Historical native winner is not a frozen representative; no substitution allowed')
        representative = representatives.index(native)
        # Enumerate all unchanged rows first, so a numerical-tie winner outside
        # the former mass set still has every required original formula.
        super().__init__(case, score, validation, forecast, base_kw, mode=mode,
            target_mass=1., lambda_start=lambda_start, meter_relative_sd=meter_relative_sd,
            forecast_relative_sd=forecast_relative_sd)
        old_targets, _ = target_set(self.history, self.sigma2, target_mass)
        kept = np.asarray([k for k, row in enumerate(self.rows) if row['target_native'] == native], dtype=int)
        expected = [j for j, topology in enumerate(self.native_topology)
                    if topology != self.native_topology[native]]
        if [self.rows[k]['competitor_native'] for k in kept] != expected:
            raise ValueError('Single target must retain every different-topology native point')
        all_rows, all_guards = len(self.rows), len(self.guards)
        original_rows = sum(r['target_rep'] in old_targets for r in self.rows)
        original_guards = sum(g[0] in old_targets or g[1] in old_targets for g in self.guards)
        self.rows = [self.rows[k] for k in kept]
        fields = ['mean_matrices', 'variance_matrices', 'mu_q', 'mu_a', 'mu_c',
                  'variance_q', 'variance_a', 'variance_c', 'alphas', 'z', 'variance_upper']
        if mode != 'normal_fixed':
            fields += ['gate_q', 'gate_a', 'gate_c', 'gate_upper']
            self.effective_matrices = [self.effective_matrices[k] for k in kept]
            self.logdet_terms = [self.logdet_terms[k] for k in kept]
        for field in fields:
            setattr(self, field, np.asarray(getattr(self, field))[kept].copy())
        self.guards = [g for g in self.guards if g[0] == representative]
        self.pairs = [(i, j) for i, j in self.pairs if representative in (i, j)]
        self.targets = [representative]
        self.target_mass = float(target_mass)
        self.selection = dict(selection, selected_representative=representative,
            previous_mass_parameter=float(target_mass), previous_mass_targets=old_targets,
            selected_was_in_previous_mass_targets=representative in old_targets,
            full_rows_before_filter=all_rows, retained_probability_rows=len(self.rows),
            dropped_full_probability_rows=all_rows-len(self.rows),
            previous_mass_probability_rows=original_rows,
            full_guards_before_filter=all_guards, retained_forward_guards=len(self.guards),
            dropped_full_guards=all_guards-len(self.guards),
            previous_mass_guard_count=original_guards,
            target_rule='Single historical native winner; mass parameter retained only for baseline lineage',
            competitor_rule='All different-topology native parameter points; no representative-only pruning',
            guard_rule='Only original representative guards with left equal to the selected target; reverse and other-target guards removed')
        self.metadata.update(target_mass=float(target_mass), targets=self.targets,
            target_native_indices=[native], pair_count=len(self.rows), guard_count=len(self.guards),
            gate_box_upper=self.gate_upper, variance_box_upper=self.variance_upper,
            historical_target_selection=self.selection,
            guard_scope=self.selection['guard_rule'],
            confidence_scope='Conditional on the historical winning target and its unchanged scalar-calibrated mean; original fixed Gaussian laws, all foreign native points; no actual-AC guarantee')
