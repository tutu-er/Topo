"""Locked exploratory rescoring of OLD completed runs; numpy only.

No fitting, solver, AC or current stability-study results are imported/read.
Full-native H stays frozen. Centering changes the loss and profiles one common
offset per time. The old sigma is not used to publish a new probability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ALLOWED_RUNS = {'flexibility_extended_paired900_20261114_v1',
                'flexibility_extended_fresh900_20261114_v1'}
TRUTH_NATIVE_INDEX = 4  # Previously verified offline internal-topology label.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Sealed input changed: '+str(path))
    return dict(path=str(path), sha256=digest)


def check(reference):
    return ref(reference['path'], reference['sha256'])


def center(value):
    return value-value.mean(axis=-1, keepdims=True)


def losses(H, X, Y):
    residual = Y[None]-np.einsum('tc,knc->ktn', X, H)
    original = np.sum(residual**2, axis=(1, 2))
    projected = np.sum(center(residual)**2, axis=(1, 2))
    profiled = original-residual.shape[-1]*np.sum(residual.mean(axis=-1)**2, axis=1)
    alternate = np.sum((center(Y)[None]-np.einsum('tc,knc->ktn', X, H-H.mean(axis=1, keepdims=True)))**2, axis=(1, 2))
    error = max(float(np.max(np.abs(projected-profiled))), float(np.max(np.abs(projected-alternate))))
    if error > 2e-12:
        raise ArithmeticError('Centering, offset profiling and projected-H loss disagree')
    return original, projected, error


def choose(sse, internal, cells, reference_mse, tie):
    scaled = sse/(cells*reference_mse)
    eligible = np.flatnonzero(scaled <= scaled.min()+tie).tolist()
    selected = min(eligible, key=lambda i:(internal[i], scaled[i], i))
    return dict(selected=int(selected), eligible=eligible,
                exact_minimum_indices=np.flatnonzero(sse == sse.min()).tolist())


def load_observations(path):
    # Deliberately avoid true_P/true_Q/true_V/true_root and delivered actions.
    with np.load(path, allow_pickle=False) as data:
        return {key:data[key].copy() for key in ('P', 'Q', 'Y', 'V', 'root', 'labels')}


def run(out):
    out = Path(out).resolve(); protocol_ref = ref(out/'PROTOCOL.json')
    protocol = read(protocol_ref['path'])
    if protocol['status'] != 'locked_before_centered_rescoring':
        raise ValueError('A pre-rescoring lock is required')
    if {Path(item['path']).name for item in protocol['source_runs']} != ALLOWED_RUNS:
        raise ValueError('Only the two specifically locked OLD completed runs are allowed')
    if protocol['strategies'] != ['no_dfs', 'symmetric_guard', 'symmetric_guard_rotation']:
        raise ValueError('The locked full strategy list changed')
    rows, contexts, sources = [], [], []
    max_identity, max_original_error, max_root_error, max_common_H_error = 0., 0., 0., 0.
    for source in protocol['source_runs']:
        root = Path(source['path']); check(source['protocol']); check(source['complete'])
        complete = read(source['complete']['path'])
        if complete['status'] != 'complete' or not complete['source_unchanged']:
            raise ValueError('Source run was not completed with unchanged code')
        config = read(source['protocol']['path']); inputs_ref = check(config['inputs'])
        if config['forecasts'] != 4 or config['repeats'] != 8:
            raise ValueError('Expected every four-by-eight old execution path')
        for entry in read(inputs_ref['path']):
            if entry['case'] != 'pengwah18' or entry['start_date'] != '2024-10-14':
                raise ValueError('This diagnostic is restricted to the locked Pengwah history')
            score_ref = check(entry['score']); native_ref = check(entry['native_copy'])
            validation_ref = check(entry['files']['validation.npz'])
            score, native = read(score_ref['path']), read(native_ref['path'])
            H = np.asarray([row['H'] for row in score['native_bank']], float)
            if H.shape != (5, 18, 36):
                raise ValueError('Expected all five frozen native models')
            labels = list(map(int, score['labels'])); n = len(labels)
            ids = [row['topology_id'] for row in score['native_bank']]
            internal = [row['internal'] for row in score['native_bank']]
            common = np.zeros_like(H); rebuilt = np.zeros_like(H); full_common_active = []
            for k, item in enumerate(native['frozen_candidates']):
                point = item['point']; active = False
                for support, r, x in zip(point['supports'], point['r'], point['x'], strict=True):
                    for offset, value in ((0, r), (n, x)):
                        block = np.ix_(support, np.asarray(support)+offset)
                        rebuilt[k][block] += value
                        if len(support) == n:
                            common[k][block] += value
                    if len(support) == n and max(r, x) > native['config']['zero_tol']:
                        active = True
                full_common_active.append(active)
            if not np.allclose(rebuilt, H, atol=1e-13, rtol=1e-13):
                raise ArithmeticError('All-native matrix rebuilding differs')
            CH = H-H.mean(axis=1, keepdims=True)
            common_error = float(np.max(np.abs(CH-((H-common)-(H-common).mean(axis=1, keepdims=True)))))
            max_common_H_error = max(max_common_H_error, common_error)
            symmetry = max(float(np.max(np.abs(H[:,:,:n]-H[:,:,:n].transpose(0,2,1)))),
                           float(np.max(np.abs(H[:,:,n:]-H[:,:,n:].transpose(0,2,1)))))
            val = load_observations(validation_ref['path'])
            Xh = np.c_[val['P'], val['Q']]
            historical, historical_c, error = losses(H, Xh, val['Y']); max_identity = max(max_identity, error)
            history_error = float(np.max(np.abs(historical-[item['history_sse'] for item in score['native_bank']])));
            max_original_error = max(max_original_error, history_error)
            if history_error > 1e-12: raise ArithmeticError('Stored historical SSE differs')
            pair_rows = []
            for i, j in itertools.combinations(range(len(H)), 2):
                D = H[i]-H[j]; DC = CH[i]-CH[j]; dh = Xh@D.T; dhc = center(dh)
                denominator = float(np.square(D).sum()); denominator_h = float(np.square(dh).sum())
                pair_rows.append(dict(pair=[i,j], same_internal_topology=ids[i] == ids[j],
                    original_H_difference_norm=float(np.linalg.norm(D)),
                    centered_H_difference_norm=float(np.linalg.norm(DC)),
                    centered_H_distance_squared_fraction=float(np.square(DC).sum()/denominator) if denominator else None,
                    history_distance_squared_fraction=float(np.square(dhc).sum()/denominator_h) if denominator_h else None,
                    full_matrix_projection_collision=bool(np.linalg.norm(DC) <= 1e-12*max(1.,np.linalg.norm(H[i]),np.linalg.norm(H[j])))))
            ctx_id=root.name+'::'+entry['id']
            contexts.append(dict(id=ctx_id, source_run=root.name, entry_id=entry['id'],
                score=score_ref, native=native_ref, validation=validation_ref,
                sigma2_unchanged_reference_only=score['sigma2'], covariance_or_probability_recalibrated=False,
                native_models=5, unique_internal_topologies=len(set(ids)), internal_topology_ids=ids,
                truth_native_index=TRUTH_NATIVE_INDEX, truth_topology_id=ids[TRUTH_NATIVE_INDEX],
                all_candidates_have_active_global_support=all(full_common_active),
                global_support_active_by_candidate=full_common_active,
                candidate_matrix_symmetry_error=symmetry, centering_removes_global_H_error=common_error,
                historical_original_sse=historical.tolist(), historical_centered_sse=historical_c.tolist(),
                historical_original_choice=choose(historical,internal,val['Y'].size,score['reference_mse'],score['validation_tie']),
                historical_centered_choice=choose(historical_c,internal,val['Y'].size,score['reference_mse'],score['validation_tie']),
                pair_separation=pair_rows))
            for draw in range(4):
                folder=root/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
                if read(folder/'complete.json')['status'] != 'complete':
                    raise ValueError('Unfinished source forecast')
                seal_ref=ref(folder/'seal.json'); seal=read(seal_ref['path'])
                check(seal['planner']); check(seal['plans']); check(seal['protocol'])
                if check(seal['score']) != score_ref: raise ValueError('Score seal mismatch')
                for repeat in range(8):
                    for strategy in protocol['strategies']:
                        target=folder/f'execution{repeat:02d}'/strategy
                        observation_ref, receipt_ref = ref(target/'observations.npz'), ref(target/'receipt.json')
                        receipt=read(receipt_ref['path'])
                        if receipt['status'] != 'complete':
                            raise ValueError('A failed source record requires explicit preserved-failure handling')
                        observed=load_observations(observation_ref['path'])
                        if observed['labels'].tolist() != labels: raise ValueError('Label order changed')
                        X=np.c_[observed['P'],observed['Q']]
                        original, projected, error=losses(H,X,observed['Y']);max_identity=max(max_identity,error)
                        root_identity=center(observed['Y'])+center(observed['V']**2)
                        root_error=float(np.max(np.abs(root_identity)));max_root_error=max(max_root_error,root_error)
                        if root_error > 1e-13: raise ArithmeticError('Measured shared-root cancellation failed')
                        with np.load(observation_ref['path'],allow_pickle=False) as data:
                            error=float(np.max(np.abs(original-data['native_new_sse'])));
                            total_error=float(np.max(np.abs(historical+original-data['native_total_sse'])))
                        max_original_error=max(max_original_error,error,total_error)
                        if max(error,total_error)>1e-12: raise ArithmeticError('Stored native scores differ')
                        original_total,centered_total=historical+original,historical_c+projected
                        cells=val['Y'].size+observed['Y'].size
                        old=choose(original_total,internal,cells,score['reference_mse'],score['validation_tie'])
                        new=choose(centered_total,internal,cells,score['reference_mse'],score['validation_tie'])
                        old_correct=ids[old['selected']]==ids[TRUTH_NATIVE_INDEX]
                        new_correct=ids[new['selected']]==ids[TRUTH_NATIVE_INDEX]
                        if old['selected']!=receipt['native_selected_index'] or old_correct!=receipt['native_map_correct']:
                            raise ArithmeticError('Independent original selection/correctness differs')
                        rows.append(dict(context=ctx_id,source_run=root.name,entry_id=entry['id'],
                            forecast=draw,repeat=repeat,strategy=strategy,
                            observation=observation_ref, original_receipt=receipt_ref,
                            original_choice=old,centered_choice=new,original_correct=old_correct,centered_correct=new_correct,
                            wrong_to_correct=bool(not old_correct and new_correct),correct_to_wrong=bool(old_correct and not new_correct),
                            original_total_sse=original_total.tolist(),centered_total_sse=centered_total.tolist(),
                            original_true_vs_historical_target_sse_margin=float(original_total[2]-original_total[4]),
                            centered_true_vs_historical_target_sse_margin=float(centered_total[2]-centered_total[4]),
                            original_full_support_exact=receipt['native_full_support_exact'],
                            inferred_parameters_or_supports_changed=False))
            sources.append(dict(run_protocol=source['protocol'],inputs=inputs_ref,score=score_ref,native=native_ref,validation=validation_ref))
    if len(rows)!=protocol['expected_execution_records']:
        raise ValueError('Unexpected number of records; preserve the locked whole batch')
    groups=[]
    for context in contexts:
        selected=[row for row in rows if row['context']==context['id']]
        zero={(row['forecast'],row['repeat']):row for row in selected if row['strategy']=='no_dfs'}
        for strategy in protocol['strategies']:
            items=[row for row in selected if row['strategy']==strategy]
            group=dict(source_run=context['source_run'],entry_id=context['entry_id'],strategy=strategy,n=len(items),
                original_correct=sum(row['original_correct'] for row in items),centered_correct=sum(row['centered_correct'] for row in items),
                criterion_wrong_to_correct=sum(row['wrong_to_correct'] for row in items),criterion_correct_to_wrong=sum(row['correct_to_wrong'] for row in items),
                original_selected_index_counts=dict(Counter(row['original_choice']['selected'] for row in items)),
                centered_selected_index_counts=dict(Counter(row['centered_choice']['selected'] for row in items)),
                centered_multiple_numerical_minima=sum(len(row['centered_choice']['eligible'])>1 for row in items))
            for kind in ('original','centered'):
                group[kind+'_action_gains_vs_zero']=sum(not zero[(r['forecast'],r['repeat'])][kind+'_correct'] and r[kind+'_correct'] for r in items)
                group[kind+'_action_losses_vs_zero']=sum(zero[(r['forecast'],r['repeat'])][kind+'_correct'] and not r[kind+'_correct'] for r in items)
            groups.append(group)
    if any(name.startswith('gurobipy') for name in sys.modules): raise AssertionError('Optimizer import prohibited')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol_ref,source=ref(__file__),
        scope='Exploratory change of scoring criterion on all locked old observations; no new independent experiment or formal metric change',
        truth_scope='Previously independently verified Pengwah internal-topology label: native index 4; only used after scoring',
        source_inputs=sources,contexts=contexts,groups=groups,rows=rows,
        checks=dict(records=len(rows),original_selections_reproduced=len(rows),all_native_points=5,
            maximum_loss_identity_error=max_identity,maximum_original_saved_score_error=max_original_error,
            maximum_shared_root_cancellation_error=max_root_error,maximum_full_common_H_removal_error=max_common_H_error,
            new_optimization=False,new_AC=False,new_fitting=False,new_probabilities=False,formal_current_results_read=False))
    destination=out/'REPORT.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    shutil.copyfile(__file__,out/'centered_diagnostic.py')
    print(json.dumps(dict(report=ref(destination),checks=result['checks'],groups=groups),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
