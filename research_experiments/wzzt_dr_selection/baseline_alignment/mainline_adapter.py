"""Observed-data RNJ -> WZZT adapter using the current production defaults.

The CLI pipeline generates synthetic data internally. This research adapter
replaces only that data preparation: RNJ selection calls the production stage,
and MILP arguments match ``pipeline._fit_milp_variants``. It never accepts a
physical network or topology truth. Its outputs retain the v1 fit schema.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from research_experiments.wzzt_dr_selection.end_to_end import wzzt_adapter as observed
from rnj_wzzt import pipeline

core = observed.core
_CORE_DEFAULTS = {k:p.default for k,p in inspect.signature(core.fit_laminar_l1_sensitivity).parameters.items()
                  if p.default is not inspect.Parameter.empty
                  and k not in ('progress_callback','validation_scenarios','initial_supports')}
_OPTION_NAMES = ('training_replicate', 'bootstrap_replicates', 'block_length',
                 'confidence_threshold', 'maximum_candidate_count', 'tolerance_factor',
                 'time_limit', 'coefficient_bound', 'milp_solver')


def mainline_defaults():
    """Read the authoritative defaults rather than maintaining a second set."""
    parameters = inspect.signature(pipeline.run).parameters
    return {name:parameters[name].default for name in _OPTION_NAMES}


def _options(overrides):
    supplied = {} if overrides is None else dict(overrides)
    unknown = set(supplied) - set(_OPTION_NAMES)
    if unknown:
        raise ValueError('unsupported mainline override keys: ' + ', '.join(sorted(unknown)))
    effective = {**mainline_defaults(), **supplied}
    pipeline._validate_solver(effective['milp_solver'])
    pipeline._integer(effective['training_replicate'], 'training_replicate', 0)
    pipeline._validate_run_options(
        scenario_suite='reference', samples_per_scenario=96, scenario_count=3,
        root_observation='noisy',
        **{name:effective[name] for name in _OPTION_NAMES
           if name not in ('training_replicate','milp_solver')})
    # Validation above checks algorithm options, not external data row counts.
    return effective, supplied


def _family(family):
    return [sorted(clade) for clade in sorted(family, key=lambda c:(len(c),tuple(sorted(c))))]


def _selection_record(selection):
    return {'full_clades':_family(selection.full_clades),
            'full_cherries':_family(selection.full_cherries),
            'selected':[sorted(c) for c in selection.selected],
            'bootstrap_frequencies':[{'support':sorted(c),'frequency':v}
                for c,v in sorted(selection.confidence.items(),key=lambda item:(len(item[0]),tuple(sorted(item[0]))))],
            'r2_score':selection.r2_score,'condition':selection.condition,
            'elapsed_seconds':selection.elapsed,
            'meaning':'bootstrap selection frequency, not topology correctness probability'}


def _input_hash(scenarios):
    digest = hashlib.sha256()
    for scenario in scenarios:
        digest.update(scenario['name'].encode())
        for name in ('P_terminal','Q_terminal','V_terminal','V_root_observed','drop_target'):
            value = scenario[name]
            digest.update(name.encode())
            digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
            if isinstance(value,pd.DataFrame):
                digest.update(repr(value.columns.tolist()).encode())
    return digest.hexdigest()


def fit_mainline_observed(training, validation, root_label, output_dir, overrides=None):
    """Fit the complete production selection/search sequence to observed inputs.

    All exposed overrides use ``pipeline.run`` names. Defaults come from that
    entrypoint; other core parameters are left unpassed, exactly as in the
    production hybrid call. The root label is an observed-label convention.
    """
    effective, supplied = _options(overrides)
    train = observed._normalize_observed(training, 'training')
    valid = observed._normalize_observed(validation, 'validation')
    labels = train[0]['P_terminal'].columns
    if not valid[0]['P_terminal'].columns.equals(labels):
        raise ValueError('training and validation terminal order must agree')
    if root_label in labels:
        raise ValueError('observed root must not be a terminal label')
    if {s['name'] for s in train} & {s['name'] for s in valid}:
        raise ValueError('training and validation scenario names must be disjoint')
    output = Path(output_dir)
    checkpoint = output/'wzzt.json'
    if checkpoint.exists():
        raise FileExistsError('refusing to replace an existing mainline fit: ' + str(checkpoint))
    output.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    source_files = [Path(__file__).resolve(), Path(observed.__file__).resolve(),
                    *Path(pipeline.__file__).parent.rglob('*.py')]
    effective_core = {**_CORE_DEFAULTS,
        'r_upper_bound':effective['coefficient_bound'],'x_upper_bound':effective['coefficient_bound'],
        'time_limit':effective['time_limit'],'solver':effective['milp_solver']}
    # Compatible field used by v1 truth evaluation; no new core override.
    fit_config = {'zero_tolerance':effective_core['zero_tolerance'],
                  'validation_blocks':effective_core['validation_blocks'],
                  'solver':effective['milp_solver'],'time_limit_seconds':effective['time_limit']}
    receipt = {'status':'running','stage':'rnj','has_model':False,
        'entrypoint':'production _select_case_rnj -> fit_laminar_l1_sensitivity',
        'initializer':'rx75_rnj_plus_milp','rnj_stage_called':True,
        'production_default_pipeline_called':False,'production_rnj_wzzt_stages_called':True,
        'external_observations_replace_synthetic_data_preparation':True,
        'truth_input_allowed':False,'global_tree_optimality_claim':False,
        'support_search_mode':'unrestricted', 'terminal_labels':labels.tolist(),'root_label':root_label,
        'training_scenarios':[s['name'] for s in train], 'validation_scenarios':[s['name'] for s in valid],
        'training_time_rows':sum(len(s['P_terminal']) for s in train),
        'validation_time_rows':sum(len(s['P_terminal']) for s in valid),
        'input_sha256':{'training':_input_hash(train),'validation':_input_hash(valid)},
        'units':train[0]['units'],'observed_target':'V_root_observed^2 - V_terminal^2; raw; no intercept',
        'mainline_options':effective,'explicit_overrides':supplied,'core_defaults_and_forwarded_options':effective_core,
        'config':fit_config, 'bootstrap_seed':20_260_903+effective['training_replicate'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},
        'core_file':str(Path(core.__file__).resolve()),
        'core_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()}
    observed._write_json(checkpoint,receipt)
    case = pipeline._CaseData(name='external_observed',training=train,validation=valid,
        terminals=labels.tolist(),root=root_label,truth_clades=set(),truth_cherries=set())
    events=[]
    last_snapshot={}

    def progress(state):
        nonlocal last_snapshot
        events.append({'event':state['event'],'elapsed_seconds':perf_counter()-started,
            'path_length':len(state['path']),'attempt_count':len(state['attempted_extensions']),
            'bound_expansions':state['bound_expansions'],'stop_reason':state['stop_reason']})
        last_snapshot={'events':events,
            'last_certified_path':[observed._path(p,state['path'][i-1] if i else None) for i,p in enumerate(state['path'])],
            'attempted_extensions':[{**observed._plain(a),'diagnostics':observed._diagnostic(a.diagnostics)} for a in state['attempted_extensions']]}
        observed._write_json(checkpoint,{**receipt,**last_snapshot})

    try:
        selection = pipeline._select_case_rnj(case,
            **{name:effective[name] for name in ('bootstrap_replicates','block_length','confidence_threshold','maximum_candidate_count','tolerance_factor')},
            seed=receipt['bootstrap_seed'])
        receipt['rnj'] = _selection_record(selection)
        observed._write_json(output/'rnj.json',receipt['rnj'])
        positions = {label:index for index,label in enumerate(labels)}
        supports = [*pipeline._leaf_singletons(len(labels)),
                    *(tuple(positions[label] for label in sorted(clade)) for clade in selection.selected)]
        receipt.update({'stage':'wzzt','initial_support_indices':supports})
        observed._write_json(checkpoint,receipt)
        fitted = pipeline.fit_laminar_l1_sensitivity(
            case.training,validation_scenarios=case.validation,initial_supports=supports,
            r_upper_bound=effective['coefficient_bound'],x_upper_bound=effective['coefficient_bound'],
            time_limit=effective['time_limit'],solver=effective['milp_solver'],progress_callback=progress)
        train_mae,_,_=core.evaluate_l1_matrices(train,fitted.r_matrix,fitted.x_matrix)
        valid_mae,valid_se,valid_blocks=core.evaluate_l1_matrices(valid,fitted.r_matrix,fitted.x_matrix,
            blocks_per_scenario=effective_core['validation_blocks'])
        if not np.isclose(train_mae,fitted.train_mae,atol=1e-10,rtol=1e-7):
            raise RuntimeError('selected training loss does not reproduce')
        if not np.isclose(valid_mae,fitted.validation_mae,atol=1e-10,rtol=1e-7):
            raise RuntimeError('selected validation loss does not reproduce')
        incomplete = fitted.stop_reason in {'extension_not_proven_optimal','extension_without_incumbent','bound_expansion_limit'}
        result = {**receipt,'stage':'finished','status':'incomplete' if incomplete else 'complete','has_model':True,
            'elapsed_seconds':perf_counter()-started,'events':events,'summary':fitted.summary(),
            'selected':observed._path(fitted.path[fitted.selected_path_index]),
            'path':[observed._path(p,fitted.path[i-1] if i else None) for i,p in enumerate(fitted.path)],
            'attempted_extensions':[{**observed._plain(a),'diagnostics':observed._diagnostic(a.diagnostics)} for a in fitted.attempted_extensions],
            'selected_loss_recomputed':{'train_mae_pu2':train_mae,'validation_mae_pu2':valid_mae,
                'validation_se_pu2':valid_se,'validation_block_means_pu2':valid_blocks.tolist()},
            'validation_rule':'production one-standard-error rule on stored path; no truth input',
            'topology_recovery_evaluated':False}
    except Exception as exc:
        result={**receipt,**last_snapshot,'status':'failed','has_model':False,
            'exception_type':type(exc).__name__,'exception':str(exc),
            'elapsed_seconds':perf_counter()-started,'events':events}
    result=observed._plain(result)
    observed._write_json(checkpoint,result)
    return result
