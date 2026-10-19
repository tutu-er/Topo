"""Small self-contained retained-state evidence for the three-event audit.

Pure projection/roundtrip checks. No simulation, AC, fits or EV planning.
Optimizer iterates, Jacobians and repeated residual arrays remain in D raw.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path

from . import gradient_price_study as io


def small_solver(value):
    if value is None:return None
    return {k:value[k] for k in ('candidate','loss','loglik','success','status','message','nfev','optimality') if k in value}


def retained_stage(stage, *, state=None):
    keys=('loss','loglik','normalizer','success','optimization_failure','selected_origin','witness_origin',
        'optimizer_error','physical_voltage_feasible','source_box_feasible','line_domain_feasible',
        'usable','ac_equation_residual','power_balance_residual','seconds','global_optimality_certified',
        'loss_scope')
    out={k:stage[k] for k in keys if k in stage}
    out['state']=stage.get('state',state)
    out['blocks']={name:{k:v for k,v in block.items() if k in ('loss','channel_loss','time_count')}
                   for name,block in stage.get('blocks',{}).items()}
    for name in ('optimizer','future_fit'):
        if name in stage:out[name]=small_solver(stage[name])
    return out


def compact_history(meta,raw):
    histories={}
    for name,h in meta['histories'].items():
        train=h['training_fit']
        histories[name]=dict(cache_fingerprint=h['fingerprint'],
            training_fit=dict(**small_solver(train),state={k:train[k] for k in ('r','x','source_p','source_q')},
                direct_PQ=train.get('diagnostics',{}).get('P_Q_inputs_fixed'),
                ratio_constraint_enforced=train.get('diagnostics',{}).get('ratio_constraint_enforced')),
            fixed_pre=retained_stage(h['fixed_pre'],state=h['fixed_state']),
            joint_pre=retained_stage(h['joint_pre'],state=h['joint_state']),
            original_preparation_seconds=h['seconds'])
    return dict(bank=raw['physical_bank'],index=raw['index'],seed=meta['seed'],case_id=raw['case_id'],
        base_kw=raw['config']['base_kw'],
        family_index=raw['family_index'],true_name=meta['true_name'],candidates=raw['candidates'],
        raw_source=dict(path=meta['raw_path'],sha256=meta['raw_sha256']),
        H=meta['training'],V=meta['validation'],H_meter=meta['training_meter'],V_meter=meta['validation_meter'],
        E_meter=meta['future_meter'],fit_options=meta['options'],histories=histories,
        source_interface=1,forecast_inputs=meta['predicted'],offline_true_inputs=meta['true'],
        saved_actions=meta['power_kw'],original_noisy_M=meta['noisy_actual_M'],
        block_order=['training','validation','future'],
        block_lengths=[len(meta['training']['p']),len(meta['validation']['p']),len(meta['predicted']['p'])])


def compact_target(target):
    rows={}
    for name,r in target['candidates'].items():
        small={k:r[k] for k in ('seed','future_source_seed','joint_fit_seed','joint_success','fixed_success',
            'joint_loss','fixed_loss','seconds','counters','failure','traceback') if k in r}
        c=r.get('comparison')
        if c:
            small.update(joint=retained_stage(c['joint']['post']),fixed=retained_stage(c['fixed']['post']),
                         checks=c['checks'],cost=c['cost'])
        rows[name]=small
    return dict(method=target['method'],state=target['state'],factors=target['factors'],
        observations=target.get('observations'),target_sha256=target.get('target_sha256'),
        power_kw=target.get('power_kw'),AC=target.get('AC'),target_seconds=target.get('target_seconds'),
        target_failure=target.get('target_failure'),traceback=target.get('traceback'),candidates=rows)


def pack(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Preserve existing compact evidence')
    freeze=io.read_json(source/'freeze.json');summary=io.read_json(source/'summary.json')
    histories=[];targets=[];files=[];raw_total=0
    for bank,index in freeze['events']:
        folder=source/f'{bank}_{index:03d}';meta=io.read_json(folder/'history.json.gz')
        raw_path=Path(meta['raw_path'])
        if sha256(raw_path.read_bytes()).hexdigest()!=meta['raw_sha256']:raise ValueError('Original raw hash mismatch')
        raw=io.read_json(raw_path);histories.append(compact_history(meta,raw));del raw
        for path in sorted(folder.glob('scalar_*.json.gz')):
            data=path.read_bytes();target=io.read_json(path);small=compact_target(target)
            small.update(bank=bank,index=index,source_sha256=sha256(data).hexdigest())
            targets.append(small);raw_total+=len(data)
            files.append(dict(path=str(path),sha256=sha256(data).hexdigest(),bytes=len(data)))
    original_validation=io.read_json(source/'validation.json')
    record=dict(schema='mobile_v3_complete_score_compact_evidence_v1',freeze=freeze,summary=summary,
        validation={k:v for k,v in original_validation.items() if k!='files'},histories=histories,targets=targets,
        projection_tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),target_sources=files,
        replay_scope='Retained-state AC rescoring possible from candidate supports, H/V/E observations/meters and one joint/fixed R/X plus HVE source state; no optimizer rerun required.',
        omitted=['optimizer iterations','optimizer Jacobians','repeated fitted predictions','residual arrays'],
        interpretation='Every target includes original H+V. No separate H-only versus H+V ablation was performed. Background contribution +72.5 in n3 index20 is retained.')
    io.write_json(output,record,compressed=True)
    restored=io.read_json(output)
    if io.serial(record)!=restored:raise ValueError('Compact evidence JSON roundtrip differs')
    validation=dict(success=True,target_count=len(targets),history_count=len(histories),
        retained_mode_states=sum(2 for t in targets for r in t['candidates'].values() if r.get('comparison') is not None or ('joint' in r and 'fixed' in r)),
        full_target_compressed_bytes=raw_total,compact_bytes=output.stat().st_size,
        compact_sha256=sha256(output.read_bytes()).hexdigest(),
        no_AC_or_fit_calls=True,lossless_roundtrip=True)
    if len(targets)!=48 or len(histories)!=3 or validation['retained_mode_states']!=384:
        raise ValueError('Compact evidence denominator incomplete')
    io.write_json(output.with_suffix('.validation.json'),validation)
    return validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('outputs/mobile_v3_score_attribution_v1'))
    p.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_score_attribution_compact.json.gz'))
    a=p.parse_args();print(pack(a.source,a.output),flush=True)


if __name__=='__main__':main()
