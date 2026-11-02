"""Thin, explicit-frozen-gate adapter to the existing reference AC producer.

Importing this file creates no network and no trajectories.  Runtime/context
construction and background generation are explicit post-freeze operations.
No resource limit causes clipping, redraw, rescaling, or an early AC skip.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRODUCER_MODULES = (
    'rnj_wzzt.scenario.simulation', 'rnj_wzzt.scenario.profiles',
    'rnj_wzzt.scenario.settings', 'rnj_wzzt.data.paper_style_case_bank',
    'rnj_wzzt.data.paper_style_terminal_lv', 'rnj_wzzt.models.network',
    'rnj_wzzt.models.ac_powerflow',
)
NOISE_KEYS = ('none', 'v', 'pq', 'both')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(reference):
    if not isinstance(reference, dict) or not {'path', 'sha256'} <= set(reference):
        raise ValueError('explicit path/sha256 reference required')
    path = Path(reference['path']).resolve()
    if _sha(path) != reference['sha256']:
        raise ValueError('frozen reference hash mismatch: ' + str(path))
    return path


def open_runtime(source_root, metadata_ref, *, repo_root, protocol_ref,
                 labels=tuple(range(101, 116)), case='paper15',
                 v_sigma=.0005, pq_sigma=.005):
    """Verify the frozen protocol/source closure, then construct original net.

    metadata_ref is {path, sha256, case_name}; the metadata belongs under
    outputs/.../networks/paper15/reference, not inside the source/ folder.
    The returned dict is private runtime state, not a serializable payload.
    """
    protocol_path = _checked(protocol_ref)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'frozen':
        raise ValueError('physical runtime requires an already frozen protocol')
    sources = protocol.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('protocol must bind explicit source closure')
    checked_sources = {}
    for file, digest in sources.items():
        path = Path(file)
        if not path.is_absolute():
            raise ValueError('source manifest keys must be absolute paths')
        path = path.resolve()
        if _sha(path) != digest:
            raise ValueError('source drift before physical runtime: ' + str(path))
        checked_sources[str(path)] = digest
    repo_root, source_root = Path(repo_root).resolve(), Path(source_root).resolve()
    legacy_path = repo_root/'research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py'
    checker_path = repo_root/'research_experiments/wzzt_dr_selection/end_to_end/physical_case.py'
    required = [Path(__file__).resolve(), legacy_path, checker_path]
    required += [source_root.joinpath(*m.split('.')).with_suffix('.py') for m in PRODUCER_MODULES]
    if any(str(p.resolve()) not in checked_sources for p in required):
        raise ValueError('source closure omits AC bridge, physical checker or original producer')
    metadata_path = _checked(metadata_ref)
    metadata_json = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata_json.get('source', {}).get('case_name') != metadata_ref.get('case_name'):
        raise ValueError('case metadata identity differs from frozen reference')
    base_kw = float(metadata_json['base_mva'])*1000.
    root_label = int(metadata_json['root_bus'])
    if not np.isfinite([base_kw, v_sigma, pq_sigma]).all() or base_kw <= 0 or min(v_sigma, pq_sigma) < 0:
        raise ValueError('invalid physical base or meter standard deviation')
    spec = importlib.util.spec_from_file_location('_conditional_original_physical_bridge', legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('existing physical bridge unavailable')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    lock = dict(case=case, base_kw=base_kw, root_label=root_label, source_metadata=dict(metadata_ref))
    metadata = legacy.frozen_source_metadata(lock, case)
    sim, runtime_sha = legacy._simulation_runtime(source_root)
    for module in PRODUCER_MODULES:
        path = source_root.joinpath(*module.split('.')).with_suffix('.py').resolve()
        if runtime_sha[module] != checked_sources[str(path)]:
            raise ValueError('loaded runtime differs from frozen source: ' + module)
    checker, loaded_checker = legacy._physical_check_function(repo_root)
    if _sha(loaded_checker) != checked_sources[str(checker_path.resolve())]:
        raise ValueError('physical checker changed')
    context = sim._prepare_case_context(case, 96, scenario_suite='reference',
        root_observation='noisy', root_meter_noise_rel=float(v_sigma), root_sigma=.003,
        impedance_scale=1.)
    legacy.validate_context_metadata(context, metadata, list(labels))
    context.net.check_terminal_load_only(strict=True)
    return dict(sim=sim, context=context, checker=checker, legacy=legacy,
        metadata=metadata, labels=list(labels), source_sha256=checked_sources,
        protocol_ref=dict(protocol_ref), metadata_ref=dict(metadata_ref), case=case,
        source_root=str(source_root), voltage_sigma=float(v_sigma), pq_sigma=float(pq_sigma))


def build_background(runtime, profile_index, replicate):
    """Generate one predeclared untouched original background; never retry."""
    if isinstance(profile_index, bool) or profile_index not in (0, 1, 2):
        raise ValueError('original profile_index 0/1/2 required')
    if isinstance(replicate, bool) or not isinstance(replicate, (int, np.integer)) or replicate < 0:
        raise ValueError('nonnegative original replicate required')
    return runtime['sim']._build_base_trajectory(runtime['context'], int(profile_index), int(replicate))


def execute_policy(runtime, background, u, Z, *, resource_labels=(103, 101),
                   cv=np.sqrt(.025/3), reactive_ratio=.35,
                   component_baseline_kw=1., component_lower_kw=0., component_upper_kw=2.,
                   elasticity=.5, dt=.25, day_offset=6, name='conditional_ac_day'):
    """Deliver the frozen signed Gaussian curve and solve original AC once.

    Return layout: scenes={none/v/pq/both: observed scene or None}, summary=
    JSON-safe audit, private=arrays/AC. Component, terminal and voltage violations
    remain in the result; only missing/nonfinite/nonconverged AC blocks observed
    scenes. Production equations/measurement code are called without rewriting.
    """
    labels, context, sim = list(runtime['labels']), runtime['context'], runtime['sim']
    resource_labels = tuple(resource_labels)
    if len(resource_labels) != 2 or len(set(resource_labels)) != 2 or any(v not in labels for v in resource_labels):
        raise ValueError('two distinct declared resource terminals required')
    u, Z = np.asarray(u, float), np.asarray(Z, float)
    if u.shape != (2, 96) or Z.shape != u.shape or not np.isfinite(u).all() or not np.isfinite(Z).all():
        raise ValueError('finite frozen mean and one independent Gaussian draw of shape(2,96) required')
    cv = np.broadcast_to(np.asarray(cv, float), u.shape)
    if not np.isfinite(cv).all() or np.any(cv < 0):
        raise ValueError('finite nonnegative response coefficient required')
    if not np.isfinite([reactive_ratio, component_baseline_kw, component_lower_kw, component_upper_kw, elasticity, dt]).all():
        raise ValueError('finite fixed contract coefficients required')
    if elasticity <= 0 or dt <= 0 or component_lower_kw > component_upper_kw:
        raise ValueError('invalid contract domain')
    p0, q0 = background.p_base.copy(), background.q_base.copy()
    if p0.shape != (96, len(labels)) or q0.shape != p0.shape or list(p0.columns) != labels or list(q0.columns) != labels:
        raise ValueError('unchanged original 96-row background and terminal order required')
    if not p0.index.equals(q0.index) or not p0.index.equals(background.root_true.index):
        raise ValueError('physical background and root timestamps differ')
    base_kw = float(runtime['metadata']['base_kw'])
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('positive frozen power base required')
    delivered = u*(1 + cv*Z)  # Signed multiplier; no abs(u), clip or redraw.
    dp, dq = np.zeros(p0.shape), np.zeros(p0.shape)
    for r, node in enumerate(resource_labels):
        j = labels.index(node)
        dp[:, j] += delivered[r]
        dq[:, j] += reactive_ratio*delivered[r]
    pf, qf = p0 + dp/base_kw, q0 + dq/base_kw
    component = component_baseline_kw + delivered
    component_bad = (component < component_lower_kw-1e-10) | (component > component_upper_kw+1e-10)
    increment_error = max(float(np.max(np.abs((pf-p0).to_numpy()*base_kw-dp), initial=0.)),
                          float(np.max(np.abs((qf-q0).to_numpy()*base_kw-dq), initial=0.)))
    if increment_error > 1e-9:
        raise AssertionError('the inherited physical load received the increment more than once')
    summary = dict(status='started', case=runtime['case'], profile_index=int(background.scenario_index),
        replicate=int(background.replicate), profile=str(background.profile), profile_seed=int(background.shifted_seed),
        root_seed=int(background.shifted_seed)+1000, meter_seed=int(background.shifted_seed)+2000,
        root_meter_seed=70_000_019+100_003*int(background.replicate)+int(background.scenario_index),
        resource_labels=list(resource_labels), resource_violation=bool(component_bad.any()),
        resource_violation_cells=int(component_bad.sum()), component_min_kw=float(component.min()),
        component_max_kw=float(component.max()), terminal_negative_cells=int((pf.to_numpy() < 0).sum()),
        terminal_min_kw=float(pf.to_numpy().min()*base_kw), max_increment_error_kw=increment_error,
        nominal_mean_box_violation=float(max(0., np.max(np.abs(u))-1.)),
        planned_cost=float(dt*np.sum(u*u/elasticity)), planned_cost_contract='prepaid bidirectional mean-volume',
        actual_net_energy_kwh=(dt*delivered.sum(axis=1)).tolist(), physical_sample_count=96,
        no_clipping=True, no_redraw=True, resource_limit_does_not_skip_AC=True,
        root_sigma=.003, voltage_sigma=float(runtime['voltage_sigma']), pq_sigma=float(runtime['pq_sigma']),
        impedance_scale=1., new_fit_calls=0, new_search_calls=0, ac_calls=0, observed_noise_versions=0)
    private = dict(natural_P_pu=p0, natural_Q_pu=q0, physical_P_pu=pf, physical_Q_pu=qf,
        root_true=background.root_true.copy(), mean_u=u.copy(), Z=Z.copy(), delivered_kw=delivered,
        node_increment_kw=dp, node_increment_kvar=dq, resource_component_kw=component,
        resource_violation_mask=component_bad)
    scenes = {key: None for key in NOISE_KEYS}
    try:
        summary['ac_calls'] = 1
        # This is exactly the production call inside sim._solve_physical_ac.
        # Calling its already-imported function directly retains nonconvergence
        # arrays instead of throwing them away at that wrapper's final guard.
        ac = sim.solve_ac_power_flow_timeseries(context.net, pf, qf,
            v_root=background.root_true, max_iter=100, tol=1e-10)
        private['AC'] = ac
        voltage = ac['V_bus_mag'].loc[:, labels]
        private['voltage_true'] = voltage
        checks = runtime['checker'](context.net, pf, qf, ac, (.9, 1.1), 1e-8)
        private['electrical_checks'] = checks
        converged = int(np.asarray(ac['converged'], bool).sum())
        summary.update(ac_converged_samples=converged, ac_nonconverged_samples=96-converged,
            electrical_equations_pass=bool(checks['electrical_equations_pass']),
            voltage_limits_pass=bool(checks['voltage_limits_pass']),
            voltage_violation_cells=int(checks['voltage_violation_cells']),
            voltage_min_pu=float(voltage.to_numpy().min()), voltage_max_pu=float(voltage.to_numpy().max()),
            max_nodal_p_residual_pu=float(checks['max_nodal_p_residual_pu']),
            max_nodal_q_residual_pu=float(checks['max_nodal_q_residual_pu']),
            max_distflow_identity_error_pu2=float(checks['max_distflow_identity_error_pu2']),
            thermal_limits_checked=False)
        if converged != 96 or not checks['electrical_equations_pass']:
            summary.update(status='physical_unresolved', all_operation_checks_pass=False,
                           failure_retained=True)
            return dict(scenes=scenes, summary=summary, private=private)
        measured = sim._observe_terminal_channels(context, background, pf, qf, voltage,
            pq_noise_rel=runtime['pq_sigma'], v_noise_rel=runtime['voltage_sigma'])
        private['both_measured_channels'] = measured
        for key in NOISE_KEYS:
            power_noise = key in ('pq', 'both')
            voltage_noise = key in ('v', 'both')
            pp = measured['p_measured'] if power_noise else pf
            qq = measured['q_measured'] if power_noise else qf
            vv = measured['voltage_measured'] if voltage_noise else voltage
            rr = measured['root_observed'] if voltage_noise else background.root_true
            scenes[key] = runtime['legacy'].scene(pp.to_numpy(), qq.to_numpy(), vv.to_numpy(),
                rr.to_numpy(), labels, name+'_'+key, int(day_offset))
        summary.update(status='complete', observed_noise_versions=4, observation_rows_per_noise=96,
            all_operation_checks_pass=bool(not component_bad.any() and checks['voltage_limits_pass']),
            failed=False, failure_retained=False)
    except Exception as error:
        summary.update(status='physical_failed', error_type=type(error).__name__,
            error=str(error), all_operation_checks_pass=False, failed=True, failure_retained=True)
    return dict(scenes=scenes, summary=summary, private=private)
