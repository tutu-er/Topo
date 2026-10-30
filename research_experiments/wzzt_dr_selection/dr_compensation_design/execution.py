"""Private simulation boundary and three-method observed-data execution.

Importing this module does not fit, simulate or read any saved trajectory.
Only ``generate_future`` accepts simulator R/X. ``identify_three`` accepts
observations only and retains the frozen fitting, selection and output rules.
Frozen project modules are reused without editing or monkey-patching them.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from time import perf_counter
import traceback

REPO = Path(os.environ.get('WZZT_REPO_ROOT', r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
STAGE = Path(__file__).resolve().parent
METHODS = ('current_rebuilt_rnj', 'current_pool_wzzt', 'current_root_augmented_rnj')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime():
    for path in (str(REPO), str(REPO / 'rnj_wzzt_core')):
        if path not in sys.path:
            sys.path.insert(0, path)
    from research_experiments.wzzt_dr_selection.dr_reference_noise import reference_noise_driver
    from research_experiments.wzzt_dr_selection.dr_random_observation_gate import study
    return reference_noise_driver, study


def _controls():
    """Load this stage's explicitly frozen controls, never an unrelated module."""
    path = STAGE / 'controls.py'
    if not path.is_file():
        raise FileNotFoundError('stage controls.py must be frozen before execution')
    name = '_wzzt_compensation_stage_controls'
    old = sys.modules.get(name)
    if old is not None:
        if Path(old.__file__).resolve() != path.resolve():
            raise RuntimeError('controls module identity mismatch')
        return old
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _regime(cfg):
    if isinstance(cfg.get('regime'), dict):
        result = dict(cfg['regime'])
    elif isinstance(cfg.get('noise_regime'), dict):
        result = dict(cfg['noise_regime'])
    elif len(cfg.get('noise_regimes', ())) == 1:
        result = dict(cfg['noise_regimes'][0])
    elif {'pq_relative', 'voltage_relative'} <= set(cfg):
        result = {k: cfg[k] for k in ('pq_relative', 'voltage_relative')}
        result['id'] = cfg.get('noise_id', 'explicit_noise_regime')
    else:
        raise ValueError('cfg must identify one explicit noise regime; no silent choice')
    if not {'pq_relative', 'voltage_relative'} <= set(result):
        raise ValueError('noise regime requires both P/Q and voltage rates')
    if any(float(result[k]) < 0 for k in ('pq_relative', 'voltage_relative')):
        raise ValueError('negative observation error rate')
    return result


def _seed(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(value)


def generate_future(R, X, menu, behavior_seed, measurement_seed, cfg, dependence='local'):
    """Simulate four complete events; truth is confined to this boundary.

    Returns ``(observed_packet, execution_receipt)``. The receipt is evaluator
    material and is not an input to ``identify_three`` or the price planner.
    Natural loads and task amplitudes remain the frozen n3 mechanism case.
    """
    import numpy as np
    reference, original = _runtime()
    controls = _controls()
    future_cfg = deepcopy(cfg)
    behavior_seed = _seed(behavior_seed, 'behavior_seed')
    measurement_seed = _seed(measurement_seed, 'measurement_seed')
    seeds = dict(future_cfg.get('seeds', {}))
    if seeds.get('natural') != 884001:
        raise ValueError('the original natural-load stream 884001 must be retained')
    if float(future_cfg.get('base_kw', 0)) != 200.:
        raise ValueError('frozen execute_future assumes the 200 kW power base')
    if float(future_cfg.get('kappa', -1)) != .35:
        raise ValueError('frozen task retains Q/P=.35')
    seeds['measurement'] = measurement_seed
    seeds['acceptance_and_independent_code'] = behavior_seed
    future_cfg['seeds'] = seeds
    regime = _regime(future_cfg)
    r = np.asarray(R, dtype=float)
    x = np.asarray(X, dtype=float)
    if r.shape != (3, 3) or x.shape != (3, 3) or not (np.isfinite(r).all() and np.isfinite(x).all()):
        raise ValueError('finite n3 simulator matrices required')
    draw = controls.sample_menu(menu, seed=behavior_seed, dependence=dependence)
    if float(draw.get('base_kw', 200.)) != 200.:
        raise ValueError('controls must use the same 200 kW power base')
    packet, events = original.execute_future(r, x, draw, regime, future_cfg)
    total = float(sum(row['payment'] for row in events))
    worst = float(draw['worst_payment'])
    if worst > 1.6 + 1e-12 or total > worst + 1e-12:
        raise AssertionError('execution exceeded the original worst-payment budget')
    if not all(all(event['checks'].values()) for event in events):
        raise AssertionError('declared complete task failed execution')
    parameter_hash = hashlib.sha256(r.tobytes() + x.tobytes()).hexdigest()
    receipt = dict(status='executed', created_utc=_now(), dependence=dependence,
                   behavior_seed=behavior_seed, measurement_seed=measurement_seed,
                   natural_seed=884001, noise_regime=regime, menu=deepcopy(menu),
                   controls=draw, events=events, actual_payment=total, worst_payment=worst,
                   packet_sha256=hashlib.sha256(pickle.dumps(packet, protocol=4)).hexdigest(),
                   simulator_parameter_sha256=parameter_hash,
                   source_sha256={str(Path(__file__).resolve()): _sha(__file__),
                                  str(Path(original.__file__).resolve()): _sha(original.__file__),
                                  str(STAGE / 'controls.py'): _sha(STAGE / 'controls.py')},
                   full_event_roles=['training', 'training', 'validation', 'validation'],
                   simulation_receipt_for_evaluator_only=True,
                   empirical_response_calibration=False, AC_validation=False)
    return packet, original.plain(receipt)


def _observed_packet(packet, *, future):
    if not isinstance(packet, dict) or set(packet) - {'training', 'validation', 'metadata'}:
        raise ValueError('identifier packets contain observed splits and metadata only')
    if not all(isinstance(packet.get(role), list) for role in ('training', 'validation')):
        raise ValueError('training and validation scene lists required')
    meta = packet.get('metadata', {})
    if not meta.get('observed_only', False):
        raise ValueError('observed_only declaration required')
    forbidden = {'truth', 'true_supports', 'physical_network', 'R_true', 'X_true',
                 'natural_P', 'natural_Q', 'true_family', 'true_K', 'p0', 'q0'}
    if forbidden & set(meta):
        raise ValueError('latent simulator information cannot enter identifier metadata')
    if future and meta.get('contains_history') is not False:
        raise ValueError('future packet must exclude old history; this function prepends it once')


def _source_hashes(reference):
    result = reference.base._fixed_source_hashes()
    for path in (Path(__file__).resolve(), Path(reference.__file__).resolve()):
        result[str(path)] = _sha(path)
    return result


def identify_three(history_packet, future_packet, history_state, settings, budget, outpath):
    """Fit three unchanged methods on one shared observed bank/calibration.

    ``budget`` is the controller's frozen gate_driver.Budget. Every method has
    its own PoolLP context. No true topology/parameter argument is accepted.
    Interrupted/finished records are never overwritten or implicitly retried.
    """
    reference, _ = _runtime()
    base = reference.base
    out = Path(outpath).resolve()
    current_file = out.with_name(out.stem + '_current_state.json')
    if out.exists() or current_file.exists():
        raise FileExistsError('existing fit receipt/checkpoint retained; controller must resume explicitly')
    started = perf_counter()
    result = dict(status='running', started_utc=_now(), methods={}, truth_input_allowed=False,
                  current_state_path=str(current_file), settings=deepcopy(settings),
                  separate_method_LP_contexts=True, shared_cross_method_LP_cache=False,
                  shared_current_native_bank=True, calibration_snapshot='current_observed',
                  price_is_external_frozen_policy=True,
                  no_price_optimization_inside_identifier=True)
    base.write(out, result)
    try:
        reference.validate_settings(settings)
        _observed_packet(history_packet, future=False)
        _observed_packet(future_packet, future=True)
        if history_state.get('status') != 'complete':
            raise ValueError('complete historical observed state required')
        training = history_packet['training'] + future_packet['training']
        validation = history_packet['validation'] + future_packet['validation']
        train, valid, days, hashes, counts = base._observations(training, validation)
        labels = list(train[0].columns)
        if labels != history_state['terminal_labels']:
            raise ValueError('history/current terminal order mismatch')
        if len(labels) != 3:
            raise ValueError('this bounded execution wrapper is n3 only')
        result.update(input_sha256=hashes, source_sha256=_source_hashes(reference),
                      day_counts=counts, terminal_labels=labels,
                      training_rows=len(train[0]), validation_rows=len(valid[0]),
                      historical_calibration=history_state['calibration'])
        base.write(out, result)
        current = base.prepare_observed_state(dict(training=training, validation=validation),
                    settings, budget, current_file, context='compensation_all_current_observed_training')
        result['current_state_status'] = current['status']
        result['shared_snapshot_preprocessing_seconds'] = current['seconds']
        result['shared_preprocessing_LP_solves'] = current['LP_solves']
        if current['status'] != 'complete':
            result['status'] = 'snapshot_state_unresolved'
            for name in METHODS:
                result['methods'][name] = dict(status='not_started_current_state_unresolved', LP_solves=[])
        else:
            result.update(applied_calibration=current['calibration'], current_pool=current['pool'],
                          historical_pool=history_state['pool'])
            for name in METHODS:
                try:
                    budget.check()
                    if name == 'current_root_augmented_rnj':
                        method = reference.run_augmented(current, train, valid, settings, budget, day_ids=days)
                    else:
                        method = base._run_method(name, current, train, valid, current['pool'],
                                                  settings, budget, day_ids=days, current_state=current)
                    result['methods'][name] = method
                except Exception as exc:
                    is_budget = isinstance(exc, RuntimeError) and ('budget' in str(exc) or 'limit' in str(exc))
                    result['methods'][name] = dict(status='not_started_budget' if is_budget else 'unresolved',
                        error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), LP_solves=[])
                base.write(out, result)
            result['status'] = 'complete_record'
    except Exception as exc:
        result.update(status='snapshot_unresolved', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        for name in METHODS:
            result['methods'].setdefault(name, dict(status='not_started_snapshot_unresolved', LP_solves=[]))
    finally:
        result.update(ended_utc=_now(), seconds=perf_counter()-started,
                      method_statuses=dict(Counter(v['status'] for v in result['methods'].values())),
                      all_methods_complete=all(result['methods'].get(name, {}).get('status') == 'complete' for name in METHODS),
                      all_W_scans_complete=result['methods'].get('current_pool_wzzt', {}).get('all_scans_complete'),
                      per_method_seconds_exclude_shared_preprocessing=True,
                      global_topology_probability_claim=False)
        base.write(out, result)
    return result
