"""Pack observed-only inputs and unchanged saved measurements for B replay.

No fitting, action selection, scientific simulation, or new noise is performed.
Each packed history must reproduce the original preparation and evaluation
fingerprints. Packed bytes have their own hash, distinct from original A raw.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from .. import io


PLAN_KEYS=('action_key','domain','power','operation_x','decoded','physicalcheck')
INITIAL_KEYS=('r','x','source_p','source_q')
TRAIN_KEYS=('candidate','name','success','r','x','source_p','source_q','p','q')


def _pick(value,keys):
    return {key:deepcopy(value[key]) for key in keys if key in value}


def thin_record(raw):
    """Explicit field allowlist; never copy posterior or future fitted states."""
    result=dict(replay_input_schema='observed_fixed_plan_B_replay_v1')
    for key in ('status','spec','failures','observed_training','observed_evidence',
                'train_meter','evidence_meter','future_meter','candidate_names',
                'candidate_structures','refit_options','operation_config','true_name','true_index'):
        if key in raw:result[key]=deepcopy(raw[key])
    if 'config' in raw:result['config']=_pick(raw['config'],('horizon','posterior_options'))
    if 'refit_history' in raw:
        saved=raw['refit_history'];history=_pick(saved,('observed_history_only','fit_options','complete','failures'))
        if 'candidates' in saved:
            history['candidates']={}
            for name,row in saved['candidates'].items():
                entry=_pick(row,('usable','success','failure','status'))
                if 'state' in row:
                    entry['state']=_pick(row['state'],INITIAL_KEYS) if isinstance(row['state'],dict) else deepcopy(row['state'])
                history['candidates'][name]=entry
        result['refit_history']=history
    if 'training_result' in raw:
        result['training_result']=dict(fits=[_pick(row,TRAIN_KEYS) for row in raw['training_result'].get('fits',[])])
    if 'plans' in raw:
        result['plans']={name:_pick(row,PLAN_KEYS) for name,row in raw['plans'].items()}
    if 'actions' in raw:
        keys=dict.fromkeys(row['action_key'] for row in result.get('plans',{}).values() if 'action_key' in row)
        actions={}
        for key in keys:
            if key not in raw['actions']:continue
            action=raw['actions'][key];row=_pick(action,('methods','power','failures'))
            row['regimes']={name:_pick(value,('values','clean')) for name,value in action.get('regimes',{}).items()}
            actions[key]=row
        result['actions']=actions
    return result


def verify_inputs(original,packed,*,base_kw=20.,future_root_v=1.):
    """Both successful and failed preparation outcomes must be preserved."""
    began=perf_counter();prepared={};errors={};payload={};evaluation_errors={}
    for label,value in (('original',original),('packed',packed)):
        try:prepared[label]=io.prepare_history(value,base_kw=base_kw,future_root_v=future_root_v)
        except Exception as exc:errors[label]=dict(type=type(exc).__name__,message=str(exc))
        if label in prepared:
            try:payload[label]=io.evaluation_payload(value,prepared[label])
            except Exception as exc:evaluation_errors[label]=dict(type=type(exc).__name__,message=str(exc))
    if errors:
        preparation_equal=set(errors)=={'original','packed'} and errors['original']==errors['packed']
        preparation=dict(available=False,equivalent=preparation_equal,errors=errors)
    else:
        hashes={label:p['observed_input_sha256'] for label,p in prepared.items()}
        preparation=dict(available=True,equivalent=hashes['original']==hashes['packed'],sha256=hashes)
    if not prepared:
        evaluation=dict(available=False,equivalent=preparation['equivalent'],scope='no historical input; original failure preserved')
    elif evaluation_errors:
        equal=set(evaluation_errors)=={'original','packed'} and evaluation_errors['original']==evaluation_errors['packed']
        evaluation=dict(available=False,equivalent=equal,errors=evaluation_errors)
    else:
        hashes={label:io.fingerprint(value) for label,value in payload.items()}
        evaluation=dict(available=True,equivalent=hashes['original']==hashes['packed'],sha256=hashes)
    physical_original=dict(plans={name:_pick(row,PLAN_KEYS) for name,row in original.get('plans',{}).items()},
                           operation_config=original.get('operation_config'))
    physical_packed=dict(plans=packed.get('plans',{}),operation_config=packed.get('operation_config'))
    physical_hashes=dict(original=io.fingerprint(physical_original),packed=io.fingerprint(physical_packed))
    physical_equal=physical_hashes['original']==physical_hashes['packed']
    return dict(passed=preparation['equivalent'] and evaluation['equivalent'] and physical_equal,
        preparation=preparation,evaluation=evaluation,physical_fixed_plans=dict(equivalent=physical_equal,sha256=physical_hashes),
        seconds=perf_counter()-began,new_optimizations=0,new_scientific_samples=0,new_noise=0,
        integration_scope='original prepare_history may recompute its diagonal prior normalization; no posterior/GHK integration')


def pack_file(source,destination,*,base_kw=20.,future_root_v=1.):
    began=perf_counter();source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('packed destination must differ from original raw')
    if destination.exists():raise FileExistsError('preserve existing packed history: '+str(destination))
    original=io.load_record(source);packed=thin_record(original)
    proof=verify_inputs(original,packed,base_kw=base_kw,future_root_v=future_root_v)
    io.save_record(destination,packed)
    return dict(source=str(source),packed=str(destination),spec=original.get('spec'),
        original_scientific_status=original.get('status'),original_raw_sha256=sha256(source.read_bytes()).hexdigest(),
        packed_raw_sha256=sha256(destination.read_bytes()).hexdigest(),original_bytes=source.stat().st_size,
        packed_bytes=destination.stat().st_size,verification=proof,passed=proof['passed'],
        seconds=perf_counter()-began,new_optimizations=0,new_cases=0,new_actions=0,new_noise=0)


README='''# 精简输入与外部复现

这些gzip记录保存同一历史观测、公共模型设置、历史初值、固定EV计划和原未来计量。
每条都核对原始与精简记录的 observed_input_sha256、evaluation_payload 指纹及固定物理计划指纹。
科学失败保持在分母中。数据不含历史后验矩阵、传播样本、未来联合拟合初值、真R/X或未来潜变量。

首先恢复发布的主165份源码快照和补充7个依赖，并安装记录中的Python/NumPy/SciPy/Gurobi环境。
在恢复源码的工作目录下，可以直接调用原B函数重放一个文件：

```python
from research_experiments.mobile_ev_mle.ac_history_refresh import io, replay
f = io.load_record("B_freeze.json")
s = f["settings"]
replay.replay_file(
    "replay_inputs/evaluation/history_000.json.gz",
    "reproduced_B/evaluation/history_000.json.gz",
    draws=s["draws"], replicates=s["replicates"], base_kw=s["base_kw"],
    future_root_v=s["future_root_v"], solver_options=s["solver_options"],
    truncation_options=s["truncation_options"], variance_multiplier=1.)
```

对全部32份校准输入按原规则重新校准，再评价32份评价输入；或只复算指定文件的原始预测。
输出耗时和source_A路径会改变，量测与模型输入的等价性由manifest的双重指纹证明。
精简文件有自己的字节SHA256，不能作为原A raw通过原metadata的字节绑定。
请把复现输出与新元数据另存，保留发布的原freeze、校准绑定和原始hash不变。
'''


def pack_batch(Aroot,output_root,*,phases=('calibration','evaluation'),base_kw=20.,future_root_v=1.,B_freeze=None):
    Aroot=Path(Aroot);output_root=Path(output_root)
    if len(set(phases))!=len(phases):raise ValueError('packing phases must be unique')
    if io.load_record(Aroot/'controller.json').get('status') not in ('statistical_complete','evaluation_complete'):
        raise RuntimeError('batch packing waits for A evaluation completion; no scientific raw opened')
    if (output_root/'manifest.json').exists():raise FileExistsError('preserve existing complete replay package')
    freeze=io.load_record(Aroot/'freeze.json');rows=[];candidate_meta={};began=perf_counter()
    output_root.mkdir(parents=True,exist_ok=True)
    if B_freeze is not None:
        B_freeze=Path(B_freeze);bound=io.load_record(B_freeze)
        if bound['input_A_policy_id']!=freeze['policy_id'] or bound['input_A_freeze_sha256']!=sha256((Aroot/'freeze.json').read_bytes()).hexdigest():
            raise ValueError('B freeze must bind the same original A freeze')
        with (output_root/'B_freeze.json').open('xb') as stream:stream.write(B_freeze.read_bytes())
    for phase in phases:
        if phase not in ('calibration','evaluation'):raise ValueError('calibration/evaluation only')
        for spec in freeze['specifications'][phase]:
            name='history_%03d.json.gz'%spec['index']
            try:row=pack_file(Aroot/phase/name,output_root/phase/name,base_kw=base_kw,future_root_v=future_root_v)
            except Exception as exc:row=dict(spec=spec,passed=False,packing_failure=type(exc).__name__+': '+str(exc))
            if 'packed' in row:
                if row['spec']!=spec:row.update(passed=False,specification_mismatch=True)
                thin=io.load_record(row['packed']);meta=_pick(thin,('candidate_names','candidate_structures'))
                if meta:
                    key=io.fingerprint(meta);candidate_meta[key]=meta;row['candidate_metadata_sha256']=key
            rows.append(row)
            with (output_root/'packing_progress.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps(io._serial(row),ensure_ascii=False,allow_nan=False)+'\n')
            print(phase,spec['index'],'PASS' if row['passed'] else 'REVIEW',flush=True)
    manifest=dict(schema='observed_fixed_plan_B_replay_pack_v1',A_policy_id=freeze['policy_id'],
        A_freeze_sha256=sha256((Aroot/'freeze.json').read_bytes()).hexdigest(),phases=list(phases),records=rows,
        history_denominator=sum(len(freeze['specifications'][p]) for p in phases),packed_histories=sum('packed_bytes' in r for r in rows),
        original_scientific_failures=sum(r.get('original_scientific_status') not in (None,'complete') for r in rows),
        packing_or_equivalence_failures=sum(not r['passed'] for r in rows),passed=all(r['passed'] for r in rows),
        packed_total_bytes=sum(r.get('packed_bytes',0) for r in rows),original_total_bytes=sum(r.get('original_bytes',0) for r in rows),
        new_cases_actions_noise=False,new_optimizations=0,seconds=perf_counter()-began,
        packer_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    io.save_record(output_root/'manifest.json',manifest)
    io.save_record(output_root/'candidate_metadata.json',candidate_meta)
    with (output_root/'README.md').open('x',encoding='utf-8') as stream:stream.write(README)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    one=sub.add_parser('one');one.add_argument('--input',required=True);one.add_argument('--output',required=True)
    one.add_argument('--manifest',required=True)
    batch=sub.add_parser('batch');batch.add_argument('--A-root',required=True);batch.add_argument('--output-root',required=True)
    batch.add_argument('--B-freeze')
    batch.add_argument('--phases',nargs='+',choices=('calibration','evaluation'),default=['calibration','evaluation'])
    args=parser.parse_args()
    if args.command=='one':
        record=pack_file(args.input,args.output);io.save_record(args.manifest,record)
        print(json.dumps(dict(passed=record['passed'],packed_bytes=record['packed_bytes'])))
    else:
        result=pack_batch(args.A_root,args.output_root,phases=args.phases,B_freeze=args.B_freeze)
        print(json.dumps(dict(passed=result['passed'],histories=result['history_denominator'],packed_bytes=result['packed_total_bytes'])))


if __name__=='__main__':main()
