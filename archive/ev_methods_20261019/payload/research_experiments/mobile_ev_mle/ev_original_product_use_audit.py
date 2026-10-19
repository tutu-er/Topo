"""Read-only actual witness/product use from saved original-domain trials.

No science module is imported and no fitting, simulation or selection runs.
Witness entries are counted as saved (including repeated stages), not as
independent experiments or unique explanations across methods.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,json,platform,sys
from pathlib import Path
from collections import Counter
from time import perf_counter
import numpy as np

METHODS=('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_trial(path):
    before=sha(path)
    with gzip.open(path,'rt',encoding='utf-8') as f:record=json.load(f)
    selection=record['selection'];models=selection.get('context_models',[])
    shifts=[np.asarray(m['domain']['shift'],float) for m in models]
    rows=[]
    for method in METHODS:
        saved=selection.get('methods',{}).get(method,{})
        results=[('final',saved.get('raw'))]
        if 'before_recenter' in saved:results.append(('before_recenter',saved['before_recenter']))
        for stage,result in results:
            row=dict(method=method,stage=stage,status='unavailable',witnesses=0,nonzero_G=0,
                max_abs_G=None,fixed_absolute_witnesses=0,global_validation_missing_or_false=0,
                masters=[],failures=[],policy=None)
            rows.append(row)
            if result is None:
                row['failures'].append('missing_exchange_result');continue
            row['policy']=result.get('witness_policy')
            if len(result.get('witnesses',[]))!=len(shifts) or not shifts:
                row['failures'].append('required_pair_witness_group_count_mismatch');continue
            maximum=0.
            for S,group in zip(shifts,result['witnesses']):
                if not group:row['failures'].append('empty_required_pair_witness_group')
                for witness in group:
                    T=np.asarray(witness['physical_slope'],float)
                    if T.shape!=S.shape or not np.isfinite(T).all():
                        row['failures'].append('invalid_witness_slope');continue
                    value=float(np.max(abs(T-S),initial=0.));maximum=max(maximum,value)
                    row['witnesses']+=1;row['nonzero_G']+=bool(value!=0.)
                    row['fixed_absolute_witnesses']+=witness.get('origin','').startswith('fixed_absolute')
                    row['global_validation_missing_or_false']+=witness.get('validation',{}).get('valid') is not True
            row['max_abs_G']=maximum
            for iteration in result.get('iterations',[]):
                master=iteration['master'];keys=master.get('product_keys')
                product_binary=master.get('n_product_binary');residual_binary=master.get('n_residual_binary')
                fields_complete=keys is not None and product_binary is not None and residual_binary is not None
                row['masters'].append(dict(product_key_count=None if keys is None else len(keys),
                    product_binary=product_binary,residual_binary=residual_binary,fields_complete=fields_complete,
                    all_first_level_products_absent=bool(fields_complete and len(keys)==0 and product_binary==0),
                    status=master.get('status'),message=master.get('message')))
            row['affine_rejection_records']=sum(f.get('kind')=='affine_witness_rejected' for f in result.get('failures',[]))
            row['status']='complete' if not row['failures'] else 'incomplete'
    if sha(path)!=before:raise RuntimeError('frozen trial changed during read-only audit')
    return dict(file=path.name,input_sha256=before,input_unchanged=True,rows=rows)


def aggregate(rows):
    complete=[r for r in rows if r['status']=='complete'];masters=[m for r in complete for m in r['masters']]
    residual=[m['residual_binary'] for m in masters if m['residual_binary'] is not None]
    witnesses=sum(r['witnesses'] for r in complete);nonzero=sum(r['nonzero_G'] for r in complete)
    bad_validation=sum(r['global_validation_missing_or_false'] for r in complete)
    return dict(required_stages=len(rows),complete_stages=len(complete),unavailable_or_incomplete_stages=len(rows)-len(complete),
        witness_entries=witnesses,nonzero_G_witness_entries=nonzero,max_abs_G=max((r['max_abs_G'] for r in complete),default=None),
        all_observed_witnesses_T_equals_S=bool(witnesses and nonzero==0),
        global_validation_missing_or_false=bad_validation,
        all_observed_witnesses_global_validation_passed=bool(witnesses and bad_validation==0),
        fixed_absolute_witness_entries=sum(r['fixed_absolute_witnesses'] for r in complete),
        affine_rejection_records=sum(r.get('affine_rejection_records',0) for r in complete),
        master_iterations=len(masters),masters_with_first_level_products=sum(not m['all_first_level_products_absent'] for m in masters),
        master_fields_missing=sum(not m['fields_complete'] for m in masters),
        all_observed_masters_first_level_products_absent=bool(masters and all(m['all_first_level_products_absent'] for m in masters)),
        no_master_stages=sum(not r['masters'] for r in complete),
        residual_binary_min=min(residual,default=None),residual_binary_max=max(residual,default=None),
        policies=dict(Counter(r['policy'] for r in complete)))


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path,required=True);args=p.parse_args();began=perf_counter()
    source=Path(__file__).read_bytes();paths=sorted(args.directory.glob('trial_*.json.gz'))
    if not paths:raise ValueError('no saved trials')
    trials=[audit_trial(path) for path in paths];rows=[r for t in trials for r in t['rows']]
    final=aggregate([r for r in rows if r['stage']=='final']);all_stages=aggregate(rows)
    result=dict(schema='original_actual_product_use_v1',scope='read-only saved results; no fits, noise or policy changes',
        directory=str(args.directory.resolve()),independent_histories=len(trials),methods=list(METHODS),
        witness_count_meaning='saved entries; repeated stages and methods counted as stored, not independent samples',
        final=final,including_before_recenter=all_stages,trials=trials,
        interpretation='T=S only for retained affine witnesses; the original inner physical domain is unchanged. Residual-square MILP remains active.',
        source_script=dict(name=Path(__file__).name,sha256=hashlib.sha256(source).hexdigest(),bytes_base64=base64.b64encode(source).decode('ascii')),
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),seconds=perf_counter()-began)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# 保存主例实际使用的乘积项核查','',
        f"只读检查{len(trials)}条历史、每条4策略；0次新拟合/模拟/选择。输入trial哈希与检查脚本原字节见同名JSON。",'',
        '| 范围 | 阶段 | 保存见证条目 | T−S非零 | 主MILP次数 | 有第一层产品的主MILP |',
        '|---|---:|---:|---:|---:|---:|',
        f"| 最终结果 | {final['complete_stages']}/{final['required_stages']} | {final['witness_entries']} | {final['nonzero_G_witness_entries']} | {final['master_iterations']} | {final['masters_with_first_level_products']} |",
        f"| 含重展开前 | {all_stages['complete_stages']}/{all_stages['required_stages']} | {all_stages['witness_entries']} | {all_stages['nonzero_G_witness_entries']} | {all_stages['master_iterations']} | {all_stages['masters_with_first_level_products']} |",'',
        f"全阶段max|T−S|={all_stages['max_abs_G']}；固定绝对解释条目{all_stages['fixed_absolute_witness_entries']}，仿射解释拒绝记录{all_stages['affine_rejection_records']}，全域验证缺失/不通过条目{all_stages['global_validation_missing_or_false']}。",'',
        f"残差平方分段二进制范围：{all_stages['residual_binary_min']}–{all_stages['residual_binary_max']}。缺失主问题字段{all_stages['master_fields_missing']}；不完整阶段{all_stages['unavailable_or_incomplete_stages']}。",'',
        '实际入库的共移动解释若T=S，则G=T−S=0，其响应残差中的控制二次乘积严格消失。这只描述有限见证库，不约束整个原内层物理参数域。原q(u)没有因此变成凸最大化问题；残差平方弦与操作二进制仍参与主MILP。', '',
        '通用非零乘积的分段能力已实现并另行测试；本主例若第一层产品计数为0，应明确写为未实际触发该部分，而由合法共移动见证退化为仿射残差。保存见证条目跨方法/阶段重复，不作为统计样本量。','']
    with args.markdown.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(final=final,including_before_recenter=all_stages),ensure_ascii=False))


if __name__=='__main__':main()
