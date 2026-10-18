"""Post-decision numerical checks and compact evidence; truth never selects a rule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from .bounds import TreeSpec, linear_voltage_matrix
from .experiment import read, save, scientific_source_hashes


def audit(directory, destination):
    directory, destination = Path(directory), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read(directory/'manifest.json')
    assets = read(directory/'public_assets.json')
    current_hashes = scientific_source_hashes()
    rows, histories, issues = [], [], []
    for item in manifest['histories']:
        meter = read(directory/item['meter_file'])
        truth = read(directory/item['truth_file'])
        record = read(directory/'results'/(item['id']+'.json'))
        if record['scientific_source_hashes'] != current_hashes:
            issues.append([item['id'], 'science_hash_mismatch'])
        for key, filename in [('meter_sha256','meter_file'),('truth_sha256','truth_file')]:
            actual = hashlib.sha256((directory/item[filename]).read_bytes()).hexdigest()
            if item[key] != actual:
                issues.append([item['id'], key])
        t, n = np.asarray(meter['p']).shape
        original = next(c for c in assets['candidates'] if c['id']==truth['generating_candidate_id'])
        tree = TreeSpec(original['root'], original['edges'], original['observed_nodes'])
        by_edge = {frozenset((e['from_bus'],e['to_bus'])):e for e in truth['branch_table']}
        real = np.array([by_edge[frozenset(e)]['r_ohm']/assets['zbase_ohm'] for e in original['edges']])
        imag = np.array([by_edge[frozenset(e)]['x_ohm']/assets['zbase_ohm'] for e in original['edges']])
        h = linear_voltage_matrix(tree, real, imag)
        project = np.eye(n)
        if meter['source_projection']:
            project -= np.ones((n,n))/n
        eta = ((np.asarray(meter['root_v'])[:,None]**2-np.asarray(truth['v'])**2
                -np.c_[truth['p'],truth['q']]@h.T)@project.T)
        actual_eta = float(np.linalg.norm(eta))
        history = dict(id=item['id'], samples=t, source_mode=meter['source_mode'],
            actual_loss_remainder_norm_post_truth_audit=actual_eta,
            read_truth_only_after_decisions=True)
        for slot in record['candidate_results']:
            if not slot['success']:
                issues.append([item['id'],slot['candidate_id'],'candidate_failure'])
                continue
            r = slot['result']
            L, U = r['qp_support_lower'], r['qp_primal']
            if L > U+1e-8*(1+abs(U)):
                issues.append([item['id'],slot['candidate_id'],'lower_above_primal'])
            b = (max(np.sqrt(L)-r['remainder_norm'],0)/r['lipschitz_k'])**2
            if not np.isclose(b,r['ac_sse_lower'],rtol=1e-10,atol=1e-10):
                issues.append([item['id'],slot['candidate_id'],'AC_bound_recompute'])
            if bool(b>r['chi_square_threshold']) != r['reject']:
                issues.append([item['id'],slot['candidate_id'],'raw_decision'])
            if bool(L>r['residual_chi_threshold']) != r['residual_chi_reject']:
                issues.append([item['id'],slot['candidate_id'],'residual_decision'])
            if r['diagnostics']['qp_domain_violation'] > 1e-7:
                issues.append([item['id'],slot['candidate_id'],'qp_domain_violation'])
            row = dict(history=item['id'],samples=t,source_mode=meter['source_mode'],
                candidate=slot['candidate_id'],
                sqrt_lower=float(np.sqrt(L)),qp_gap=U-L,
                AC_loss_envelope=r['remainder_norm'],
                raw_noise_radius=r['lipschitz_k']*np.sqrt(r['chi_square_threshold']),
                residual_noise_radius=r['residual_chi_noise_radius'],
                AC_sse_lower=b,raw_chi_square=r['chi_square_threshold'],
                raw_reject=r['reject'],residual_reject=r['residual_chi_reject'],
                no_loss_residual_reject_DIAGNOSTIC=bool(L>r['residual_chi_noise_radius']**2),
                feasible_point_upper=slot['ac_feasible_point'].get('whitened_AC_sse_upper'),
                seconds=slot['seconds'])
            rows.append(row)
            if slot['candidate_id']==truth['generating_candidate_id']:
                noisy_at_true=((np.asarray(meter['root_v'])[:,None]**2-np.asarray(meter['v'])**2
                                -np.c_[meter['p'],meter['q']]@h.T)@project.T)
                if L > np.sum(noisy_at_true**2)+1e-8:
                    issues.append([item['id'],'QP_lower_above_true_linear_point'])
                if actual_eta > r['remainder_norm']+1e-8:
                    issues.append([item['id'],'loss_envelope_failure'])
                history.update(AC_loss_envelope=r['remainder_norm'],
                    envelope_to_actual_ratio=r['remainder_norm']/max(actual_eta,1e-30),
                    true_candidate_sqrt_lower=np.sqrt(L),
                    true_candidate_no_loss_rejected_DIAGNOSTIC=row['no_loss_residual_reject_DIAGNOSTIC'])
        histories.append(history)
    with (destination/'candidate_bounds.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='passed' if not issues else 'failed',issues=issues,
        candidate_count=len(rows),history_count=len(histories),
        max_QP_primal_minus_support_lower=max(r['qp_gap'] for r in rows),
        loss_envelope_range=[min(r['AC_loss_envelope'] for r in rows),max(r['AC_loss_envelope'] for r in rows)],
        residual_norm_range=[min(r['sqrt_lower'] for r in rows),max(r['sqrt_lower'] for r in rows)],
        no_loss_rejections_DIAGNOSTIC=sum(r['no_loss_residual_reject_DIAGNOSTIC'] for r in rows),
        no_loss_true_rejections_DIAGNOSTIC=sum(r['true_candidate_no_loss_rejected_DIAGNOSTIC'] for r in histories),
        source_hashes=current_hashes,histories=histories)
    save(destination/'independent_audit.json',summary)
    with zipfile.ZipFile(destination/'final_records.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(directory.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(directory).as_posix())
    save(destination/'archive_manifest.json',dict(
        filename='final_records.zip',bytes=(destination/'final_records.zip').stat().st_size,
        sha256=hashlib.sha256((destination/'final_records.zip').read_bytes()).hexdigest(),
        base_commit='3ff97936cd6cca9e34f7dea6fdaf60add3e8d0d4',
        note='Includes separate noisy meters/truth, frozen public prior, all 80 decisions and post-decision audits.'))
    # Restore exact bytes independently of Git's line-ending settings.
    from . import compatibility
    root=Path(__file__).resolve().parents[3]
    files={Path(__file__).resolve(),Path(compatibility.__file__).resolve(),
           root/'terminal_case33/data/case33bw_raw.py'}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(root) and path.suffix=='.py':files.add(path)
    source_hashes={}
    with zipfile.ZipFile(destination/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(files):
            relative=path.relative_to(root).as_posix()
            z.write(path,relative)
            source_hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(destination/'source_snapshot_manifest.json',dict(source_hashes=source_hashes,
        archive_sha256=hashlib.sha256((destination/'source_snapshot.zip').read_bytes()).hexdigest(),
        note='Loaded project dependency graph plus AST-read static line table; original scientific hashes above retained.'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();out=audit(a.directory,a.destination)
    print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','histories')}))
    if out['issues']:raise SystemExit(1)
