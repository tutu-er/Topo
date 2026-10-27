"""Recompute validation and select from saved WZZT paths and complete RNJ fits.

No LP, MILP, RNJ, new observations, or price planning is executed. All saved
weights are evaluated on their unchanged training and validation observations.
Truth is used only in the last evaluation after both choices are fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import scipy

from . import pooled_selection as pooled
from .run_observed_baseline import load_inputs
from .audit_rnj_family import recovery

core, observed = pooled.core, pooled.observed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def recompute_point(record, training, validation, *, validation_blocks=4):
    prepared = core._prepare_scenarios(training)
    positions = {label:i for i,label in enumerate(prepared.labels)}
    labels = record.get('support_labels', record.get('optimized_support_labels'))
    supports = tuple(tuple(positions[label] for label in support) for support in labels)
    diagnostic_fields = {f.name for f in fields(core.SolverDiagnostics)}
    diagnostic = core.SolverDiagnostics(**{k:v for k,v in record['solver'].items()
                                           if k in diagnostic_fields})
    if not core.solver_diagnostics_prove_optimality(diagnostic):
        raise ValueError('stored candidate is not a numerically optimal accepted fit')
    train_mae = record.get('train_mae', record.get('train_mae_pu2'))
    solution = core.FixedSupportSolution(supports, np.asarray(record['r_values']),
        np.asarray(record['x_values']), train_mae, diagnostic)
    point = core._make_path_point(iteration=record.get('iteration', -1), prepared=prepared,
        solution=solution, validation_scenarios=validation, validation_blocks=validation_blocks,
        accepted_gain=record.get('accepted_gain'), solver=diagnostic)
    actual_train, _, _ = core.evaluate_l1_matrices(training, point.r_matrix, point.x_matrix)
    expected = {'train_mae':train_mae,
        'validation_mae':record.get('validation_mae', record.get('validation_mae_pu2')),
        'validation_se':record.get('validation_se', record.get('validation_se_pu2'))}
    actual = {'train_mae':actual_train, 'validation_mae':point.validation_mae,
              'validation_se':point.validation_se}
    for name,value in actual.items():
        if not np.isclose(value, expected[name], atol=1e-10, rtol=1e-7):
            raise ValueError(f'saved {name} does not reproduce: {expected[name]} vs {value}')
    return point


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--rnj-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('pooled replay already exists; refusing to overwrite')
    audit = read(args.rnj_audit)
    if audit['completed_lp_count'] != 2:
        raise ValueError('both original complete RNJ fits must be available')
    # Candidate LPs and WZZT must have been formed with identical bounds.
    if audit['options']['r_upper_bound'] != 2 or audit['options']['x_upper_bound'] != 2:
        raise ValueError('recorded candidate bounds differ from locked baseline')
    core_hash = sha(core.__file__)
    if core_hash not in audit['source_sha256'].values():
        raise ValueError('core source differs from the saved complete RNJ fits')
    hashes = {str(args.rnj_audit.resolve()):sha(args.rnj_audit)}
    records = []
    for candidate_record in audit['records']:
        name = candidate_record['case']
        folder = args.runs / name
        for filename,expected_hash in candidate_record['input_sha256'].items():
            if sha(filename) != expected_hash:
                raise ValueError('original observation or RNJ selection changed: '+filename)
            hashes[filename] = expected_hash
        if name == 'N0_default_reference':
            if not (folder/'completion.json').exists():
                raise ValueError('original reference default search must be complete')
            prepared_path = folder/'prepared_case.pkl'
            with prepared_path.open('rb') as stream:
                case,_ = pickle.load(stream)
            training,validation = case.training,case.validation
            truth = set(case.truth_clades)
            saved_path = folder/'fit.json'
            saved = read(saved_path)
            original_index = saved['selected_path_index']
            algorithm_selection = read(folder/'selection.json')
        elif name == 'N1_iflex_no_reward':
            training,validation,all_truth,input_hashes = load_inputs(args.source_run)
            truth = {frozenset(s) for s in all_truth
                     if 1 < len(s) < len(training[0]['P_terminal'].columns)}
            hashes.update(input_hashes)
            saved_path = folder/'wzzt.json'
            saved = read(saved_path)
            original_index = saved['summary']['selected_path_index']
            algorithm_selection = read(folder/'rnj.json')
        else:
            raise ValueError('unexpected recorded baseline condition: '+name)
        hashes[str(saved_path.resolve())] = sha(saved_path)
        labels = list(training[0]['P_terminal'].columns)
        expected_family = {frozenset(s) for s in algorithm_selection['full_clades']}
        expected_family.update(frozenset([label]) for label in labels)
        if expected_family != {frozenset(s) for s in candidate_record['optimized_support_labels']}:
            raise ValueError('candidate is not the complete algorithm-generated RNJ family')
        path = [recompute_point(p,training,validation) for p in saved['path']]
        rnj = recompute_point(candidate_record,training,validation)
        selection = pooled.select_with_complete_rnj(path,rnj)
        if selection['original_wzzt_selected_path_index'] != original_index:
            raise ValueError('recomputed original WZZT choice differs from saved choice')
        # Topology truth is first consumed here, after final selection.
        selected = selection['selected']
        original = observed._plain(path[original_index])
        selected_recovery = recovery(selected['support_labels'],selected['r_values'],
                                     selected['x_values'],truth,labels)
        original_recovery = recovery(original['support_labels'],original['r_values'],
                                     original['x_values'],truth,labels)
        records.append({'condition':name, 'training_rows':sum(len(s['P_terminal']) for s in training),
            'validation_rows':sum(len(s['P_terminal']) for s in validation),
            'validation_blocks_per_scenario':4, 'recomputed_candidate_count':len(path)+1,
            'selection':selection, 'original_wzzt_recovery':original_recovery,
            'final_selection_recovery':selected_recovery,
            'selection_inputs_contain_truth':False})
    result = {'status':'complete', 'purpose':__doc__,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'condition_count':len(records), 'recomputed_candidate_count':sum(r['recomputed_candidate_count'] for r in records),
        'new_lp_calls':0, 'new_milp_calls':0, 'new_price_conditions':0,
        'original_wzzt_full_recoveries':sum(r['original_wzzt_recovery']['complete_effective_family'] for r in records),
        'pooled_full_recoveries':sum(r['final_selection_recovery']['complete_effective_family'] for r in records),
        'selection_by_origin':{origin:sum(r['selection']['selected_origin']==origin for r in records)
                               for origin in ('wzzt_path','complete_rnj')},
        'selection_is_not_a_wzzt_search_repair':True,
        'source_sha256':{str(Path(f).resolve()):sha(f) for f in (__file__,pooled.__file__,core.__file__)},
        'input_sha256':hashes,
        'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                       'numpy':np.__version__,'scipy':scipy.__version__}, 'records':records}
    observed._write_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','condition_count','new_lp_calls','new_milp_calls',
        'original_wzzt_full_recoveries','pooled_full_recoveries','selection_by_origin')},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
