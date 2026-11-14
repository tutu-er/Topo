"""Auxiliary means from one original fixed-support QP on known history.

Only training and historical validation P/Q/Y are opened. This module never
loads a physical network, true parameters, the future day or executions.
Original scoring matrices and the native candidate bank remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from ... import io
from ...observations import Observation, load
from ...identification.optimization import QPFitter, errors, response_matrices
from ...identification.trees import canonical
from ...identification import optimization


def checked(reference):
    path = Path(reference['path'])
    if io.ref(path)['sha256'] != reference['sha256']:
        raise ValueError('Changed sealed source: '+str(path))
    return path


def combined_history(train_reference, validation_reference):
    train, validation = load(checked(train_reference)), load(checked(validation_reference))
    if train.labels != validation.labels or train.rows != 192 or validation.rows != 96:
        raise ValueError('Expected the fixed192-row training and96-row historical validation split')
    combined = Observation(np.r_[train.p, validation.p], np.r_[train.q, validation.q],
                           np.r_[train.y, validation.y], train.labels)
    return train, validation, combined


def members_for(score):
    if len(score['models']) != len(score['parameters']):
        raise ValueError('Representative model/parameter ordering mismatch')
    return [dict(representative_index=i, native_index=int(row['index']),
                 supports=score['parameters'][i]['point']['supports'])
            for i, row in enumerate(score['models'])]


def fit_from_run(source_run, out, *, input_id=None):
    """Reproduce the fixed recipe in a NEW output directory, when requested."""
    source_run, out = Path(source_run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep completed means; choose a new output directory')
    if io.read(source_run/'complete.json')['status'] != 'complete':
        raise ValueError('Completed historical source run required')
    source_protocol = io.read(source_run/'protocol.json')
    entries = io.read(checked(source_protocol['inputs']))
    if input_id is not None:
        entries = [e for e in entries if e['id'] == input_id]
    identities = {(e['score']['sha256'], e['files']['train.npz']['sha256'],
                   e['files']['validation.npz']['sha256']) for e in entries}
    if not entries or len(identities) != 1:
        raise ValueError('Select exactly one historical problem, allowing repeated resource variants')
    entry = entries[0]
    score, native = io.read(checked(entry['score'])), io.read(checked(entry['native_copy']))
    train, validation, history = combined_history(entry['files']['train.npz'], entry['files']['validation.npz'])
    members = members_for(score)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=io.ref(__file__),
        fit_source=io.ref(optimization.__file__), source_run=io.ref(source_run/'complete.json'),
        source_protocol=io.ref(source_run/'protocol.json'), score=entry['score'], native=entry['native_copy'],
        train=entry['files']['train.npz'], validation=entry['files']['validation.npz'],
        train_rows=192, validation_rows=96, combined_rows=288, members=members,
        statistical_rule='One original unregularized BVLS fit per frozen representative support on all known history; no structural search or hyperparameter selection',
        reference_mse=score['reference_mse'], coefficient_bound=native['config']['response_bound'], qp_soft_seconds=30.,
        role='Auxiliary conditional planning means. Frozen scoringHi, sigma, historical SSE and native-bank final selection are unchanged.',
        validation_role='Historical validation is used in the auxiliary fit and is not an independent validation set for these means',
        physical_truth_used=False, new_day_or_execution_inputs_used=False, bootstrap_replicates=0, AC_calls=0)
    out.mkdir(parents=True)
    io.write(out/'protocol.json', protocol)
    fitter = QPFitter(history, reference_mse=score['reference_mse'],
                      bound=native['config']['response_bound'], qp_seconds=30.)
    started = perf_counter(); means = []; paths = []
    try:
        for member in members:
            point = fitter.fit(member['supports'])
            R, X = response_matrices(point, history.n)
            K = np.c_[R, X]
            zeros = [support for support, r, x in zip(point['supports'], point['r'], point['x'], strict=True)
                     if 1 < len(support) < history.n and max(r, x) <= native['config']['zero_tol']]
            row = dict(**member, point=point, auxiliary_mean=K, QP=fitter.records[-1],
                history_fit_error=errors(history, point), train_fit_error=errors(train, point),
                validation_fit_error=errors(validation, point),
                proper_supports_with_both_coefficients_at_numerical_zero=zeros,
                support_list_retained_without_cleanup=canonical(point['supports']) == canonical(member['supports']))
            path = out/f"fit_rep{member['representative_index']:02d}.json"
            io.write(path, row); paths.append(io.ref(path)); means.append(K)
    except Exception as exc:
        io.write(out/'failed.json', dict(status='failed', completed_fits=paths,
            error=repr(exc), source_unchanged=io.ref(__file__)['sha256'] == protocol['source']['sha256']))
        raise
    np.savez_compressed(out/'auxiliary_means.npz', means=np.asarray(means),
        native_indices=np.array([m['native_index'] for m in members]), labels=history.labels)
    receipt = dict(status='complete', protocol=io.ref(out/'protocol.json'),
        means=io.ref(out/'auxiliary_means.npz'), fits=paths, seconds=perf_counter()-started,
        original_score_unchanged=io.ref(entry['score']['path'])['sha256'] == entry['score']['sha256'],
        physical_truth_used=False, new_day_or_execution_inputs_used=False)
    io.write(out/'fit_complete.json', receipt)
    io.write(out/'complete.json', dict(status='complete', fit=io.ref(out/'fit_complete.json')))
    return receipt


def register_existing(existing, out):
    """Audit old artifacts and record this entry point; never rerun their QPs."""
    existing, out = Path(existing).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve previous migration records')
    receipt = io.read(existing/'fit_complete.json')
    if receipt['status'] != 'complete':
        raise ValueError('Completed means required')
    protocol = io.read(checked(receipt['protocol']))
    checked(protocol['source']); checked(protocol['fit_source'])
    score, native = io.read(checked(protocol['score'])), io.read(checked(protocol['native']))
    train, validation, history = combined_history(protocol['train'], protocol['validation'])
    members = members_for(score)
    if protocol['members'] != members or protocol['reference_mse'] != score['reference_mse']:
        raise ValueError('Existing family/reference differs from reproduction recipe')
    if protocol['coefficient_bound'] != native['config']['response_bound'] or protocol['qp_soft_seconds'] != 30.:
        raise ValueError('Existing fitting bounds/budget differ from reproduction recipe')
    with np.load(checked(receipt['means']), allow_pickle=False) as saved:
        means = saved['means'].copy(); indices = saved['native_indices'].copy(); labels = saved['labels'].copy()
    if means.shape != (len(members), history.n, 2*history.n):
        raise ValueError('Unexpected auxiliary matrix shape')
    np.testing.assert_array_equal(indices, [m['native_index'] for m in members])
    np.testing.assert_array_equal(labels, history.labels)
    rows = []
    for member, reference, K in zip(members, receipt['fits'], means, strict=True):
        old = io.read(checked(reference)); point = old['point']
        if canonical(point['supports']) != canonical(member['supports']):
            raise ValueError('Stored mean has a changed candidate support')
        R, X = response_matrices(point, history.n); rebuilt = np.c_[R, X]
        np.testing.assert_array_equal(K, rebuilt)
        values = errors(history, point)
        error = abs(values['mse']-old['history_fit_error']['mse'])
        if error > 1e-14:
            raise ArithmeticError('Stored historical SSE replay failed')
        rows.append(dict(representative_index=member['representative_index'], native_index=member['native_index'],
            stored_fit=reference, exact_matrix_reconstruction=True, history_mse_replay_difference=error))
    migration = dict(status='verified_without_refitting', created_utc=datetime.now(timezone.utc).isoformat(),
        reproducible_entrypoint=io.ref(__file__), original_generator_preserved=protocol['source'],
        original_fit_receipt=io.ref(existing/'fit_complete.json'), original_fit_protocol=receipt['protocol'],
        existing_means=receipt['means'], original_fitter_source=protocol['fit_source'],
        current_fitter_source=io.ref(optimization.__file__),
        original_fitter_source_matches_current=io.ref(optimization.__file__)['sha256']==protocol['fit_source']['sha256'],
        fit_recipe_matches=True, fit_replayed=False, fitted_mean_arrays_rebuilt_exactly=True,
        data_scope='Only192training+96historical-validation rows; fixed supports/bound/reference_mse; original scoring bank unchanged',
        retrospective_truth_check='Legacy offline_validation.json remains separate. It is neither read nor used by this module.',
        new_identifier_fits=0, new_calibration_fits=0, optimizer_calls=0, AC_calls=0, checks=rows)
    out.parent.mkdir(parents=True, exist_ok=True); io.write(out, migration)
    return migration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fitting = commands.add_parser('fit', help='Run the fixed historical recipe into a new directory')
    fitting.add_argument('--source-run', type=Path, required=True)
    fitting.add_argument('--out', type=Path, required=True)
    fitting.add_argument('--input-id')
    migration = commands.add_parser('register-existing', help='Verify and map existing means without fitting')
    migration.add_argument('--existing', type=Path, required=True)
    migration.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        result = fit_from_run(args.source_run, args.out, input_id=args.input_id)
    else:
        result = register_existing(args.existing, args.out)
    print(result['status'])
