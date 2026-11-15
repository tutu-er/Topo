"""Thin single-target adapter; original solver, objective and seed search unchanged."""
from __future__ import annotations
from pathlib import Path

from ..distribution_gate_20261114 import solver as engine
from ..distribution_gate_20261114.evidence import METER_RELATIVE_SD, FORECAST_RELATIVE_SD
from ..flexibility_extended_solve_20261114.solver import _reference
from .evidence import SingleTargetEvidence


def plan(case, score, validation, forecast, base_kw, *, mode='mgf_forecast',
         seconds=600., env, seed_points=None, target_mass=.99, record_directory=None,
         meter_relative_sd=METER_RELATIVE_SD, forecast_relative_sd=FORECAST_RELATIVE_SD):
    constructed = []

    def factory(*args, **kwargs):
        evidence = SingleTargetEvidence(*args, **kwargs)
        constructed.append(evidence)
        return evidence

    receipt, point = engine.plan(case, score, validation, forecast, base_kw,
        mode=mode, seconds=seconds, env=env, seed_points=seed_points,
        target_mass=target_mass, record_directory=record_directory,
        meter_relative_sd=meter_relative_sd, forecast_relative_sd=forecast_relative_sd,
        evidence_factory=factory)
    evidence = constructed[0]
    receipt.update(historical_target_selection=evidence.selection,
        evidence_metadata=evidence.metadata, single_target=True,
        base_evidence_source=receipt.get('evidence_source'),
        evidence_source=_reference(Path(__file__).with_name('evidence.py')),
        single_target_adapter_source=_reference(__file__),
        original_scalar_guards_retained=False, original_forward_scalar_guards_retained=True,
        original_reverse_scalar_guards_retained=False,
        single_target_scope='Only historical target selection and row/guard filtering changed; coefficients, error laws, risk allocations, lambda, activation objective, resources and common seeds unchanged')
    return receipt, point
