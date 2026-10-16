"""Arithmetic-only fixed/shared-line loss decomposition on all saved histories."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import statistics


def decomposition(fixed_F,fixed_J,mobile_F,mobile_J,true,rival):
    """Keep the same rival under both controls; line opening includes source refit."""
    actions={}
    for name,F,J in (('fixed_action',fixed_F,fixed_J),('mobile_action',mobile_F,mobile_J)):
        if set(F)!=set(J) or true not in F or rival not in F:raise ValueError('matching complete candidate losses required')
        C={k:float(F[k]-J[k]) for k in F}
        if any(c < -1e-7*(1+abs(F[k])) for k,c in C.items()):raise ValueError('shared fit worse than protected fixed-line witness')
        fixedgap=F[rival]-F[true];relative=C[rival]-C[true];jointgap=J[rival]-J[true]
        error=jointgap-(fixedgap-relative)
        if abs(error)>1e-8*(1+abs(jointgap)):raise ValueError('within-action loss identity failed')
        actions[name]=dict(F=F,J=J,C=C,fixed_gap=fixedgap,relative_compensation=relative,joint_gap=jointgap,identity_error=error)
    f,m=actions['fixed_action'],actions['mobile_action']
    delta=dict(joint_gap=m['joint_gap']-f['joint_gap'],fixed_gap=m['fixed_gap']-f['fixed_gap'],
        relative_compensation=m['relative_compensation']-f['relative_compensation'])
    delta['identity_error']=delta['joint_gap']-delta['fixed_gap']+delta['relative_compensation']
    if abs(delta['identity_error'])>1e-8*(1+abs(delta['joint_gap'])):raise ValueError('cross-action loss identity failed')
    return dict(true=true,rival=rival,actions=actions,delta=delta)


def _stats(v):
    v=list(v)
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v)) if v else dict(n=0)


def summarize(rows):
    good=[r for r in rows if r['success']]
    return dict(denominator=len(rows),complete=len(good),failed=len(rows)-len(good),
        joint_gap_change=_stats(r['decomposition']['delta']['joint_gap'] for r in good),
        fixed_gap_change=_stats(r['decomposition']['delta']['fixed_gap'] for r in good),
        relative_compensation_change=_stats(r['decomposition']['delta']['relative_compensation'] for r in good),
        fixed_gap_rises_joint_gap_falls=sum(r['decomposition']['delta']['fixed_gap']>1e-7 and r['decomposition']['delta']['joint_gap']<-1e-7 for r in good),
        compensation_increases=sum(r['decomposition']['delta']['relative_compensation']>1e-7 for r in good),
        same_rival_under_fixed_joint_nearest=sum(r['fixed_nearest_joint_rival']==r['mobile_nearest_joint_rival'] for r in good))


def run(audit_path,output):
    audit_path,output=Path(audit_path),Path(output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError(output)
    content=audit_path.read_bytes();audit=json.loads(content);rows=[];rawchecks=[]
    for bank,b in audit['banks'].items():
        for h in b['rows']:
            row={k:h[k] for k in ('case_id','index','seed','true_name','H','key')};row.update(bank=bank,success=False)
            try:
                def loss(mode,tag):return h['modes'][mode]['score_details'][tag+'_loss_'+tag+'_cut']['losses']
                FF,FJ,MF,MJ=loss('fixed_total','fixed'),loss('joint_total','fixed'),loss('fixed_total','mobile'),loss('joint_total','mobile')
                true=h['true_name'];rival=min((k for k in MJ if k!=true),key=lambda k:MJ[k])
                row.update(success=True,mobile_nearest_joint_rival=rival,
                    fixed_nearest_joint_rival=min((k for k in FJ if k!=true),key=lambda k:FJ[k]),
                    original_delta_M=h['delta_M'],
                    decomposition=decomposition(FF,FJ,MF,MJ,true,rival))
                if 'focus_raw_evidence' in h:
                    path=Path(h['focus_raw_evidence']['raw_path']);data=path.read_bytes()
                    if sha256(data).hexdigest()!=h['focus_raw_evidence']['raw_sha256']:raise ValueError('focused raw hash changed')
                    raw=json.loads(gzip.decompress(data));methods={m['name']:m for m in raw['methods']}
                    checks=[]
                    for name,F,J in (('scalar_fixed',FF,FJ),('scalar_mobile',MF,MJ)):
                        comparisons=raw['future_records'][methods[name]['action_key']]['comparisons']
                        for k,c in comparisons.items():
                            if abs(c['fixed']['fixed_total']-F[k])>1e-8*(1+abs(F[k])) or abs(c['joint']['joint_total']-J[k])>1e-8*(1+abs(J[k])):
                                raise ValueError('projection vs original fixed/joint loss mismatch')
                            checks.append(dict(method=name,candidate=k,fixed_total=c['fixed']['fixed_total'],joint_total=c['joint']['joint_total']))
                    rawchecks.append(dict(key=h['key'],path=str(path),sha256=sha256(data).hexdigest(),success=True,checks=checks))
                    row['focused_raw_verified']=True
            except Exception as exc:row.update(success=False,failure=type(exc).__name__+': '+str(exc))
            rows.append(row)
    bybank={bank:summarize([r for r in rows if r['bank']==bank]) for bank in ('n3','n4')}
    result=dict(schema='mobile_v3_fixed_shared_compensation_identity_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        denominator=96,summary=bybank,rows=rows,focused_raw_checks=rawchecks,
        formula=dict(F='F_k(u): saved fixed training R/X full SSE, source fitted at every observed time.',
            J='J_k(u): saved shared training+validation+EV line/source refit full SSE.',
            C='C_k(u)=F_k(u)-J_k(u)>=0 via retained fixed-line witness.',
            gap='G_b(u)=J_b(u)-J_true(u)=F_b(u)-F_true(u)-[C_b(u)-C_true(u)].',
            difference='Delta G=Delta fixed_gap-Delta relative_compensation; same b=mobile nearest wrong joint model under both controls.'),
        interpretation='C measures the gain from opening shared R/X and jointly reoptimizing source. It is not the signed R/X cancellation diagnostic and is not an isolated causal line-error estimate.',
        nearest_scope='Changing the nearest rival can make Delta G differ from Delta M. Both rivals and the original Delta M are saved.',
        fit_scope='Saved attained local fits only; identities are arithmetic and do not require global optimization.',
        source=dict(path=str(audit_path.resolve()),sha256=sha256(content).hexdigest()),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),no_AC_fit_planning=True)
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf8')
    render(result,output.with_suffix('.md'));return result


def render(data,target):
    lines=['# 开放共享线路参数后的补偿收益：精确损失恒等式','',
        r'固定一条已保存 EV 动作 $u$，令 $F_k(u)$ 为训练 R/X 固定、逐时源已拟合的完整损失，$J_k(u)$ 为历史、验证与 EV 共用线路、重新联合拟合线路和源后的损失。定义 $C_k=F_k-J_k\ge0$。',
        '',r'对真候选 $a$ 与同一个错误竞争者 $b$：',
        '',r'$$G_b(u)=J_b(u)-J_a(u)=[F_b(u)-F_a(u)]-[C_b(u)-C_a(u)].$$',
        '',r'比较已保存固定点动作与移动动作，得到精确恒等式：',
        '',r'$$\Delta G_b=\Delta(F_b-F_a)-\Delta(C_b-C_a).$$',
        '',
        '第一项是固定训练线路时的实际计量损失差变化；第二项是开放线路后错误模型相对真模型增加的补偿收益。源在固定线路阶段已经允许逐时调整；C 表示增加线路自由度后重新联合拟合的总收益，包含线路与源的相互作用。它与 δR、δX 异号抵消是两个不同问题。','',
        '每条历史将 b 固定为移动动作下最近的错误 joint 模型，再用同一 b 比较固定动作。若固定动作的最近错误候选不同，下面 ΔG 与原 ΔM 会不同，JSON 明确保留两者。全部 96 条小投影均使用，三条主规则变化例另核对原 raw 的所有 F/J 字段。','',
        '|银行|分母/完整|平均 Δ固定线路差|平均 Δ相对补偿|平均 Δ共享线路差|固定差升但共享差降|','|---|---:|---:|---:|---:|---:|']
    for bank,s in data['summary'].items():lines.append(f"|{bank}|{s['denominator']}/{s['complete']}|{s['fixed_gap_change']['mean']:.3f}|{s['relative_compensation_change']['mean']:.3f}|{s['joint_gap_change']['mean']:.3f}|{s['fixed_gap_rises_joint_gap_falls']}|")
    lines.extend(['','## 三条判别变化例','',
        '|历史/竞争者|固定动作 F差|固定动作 C差|固定动作 G|移动动作 F差|移动动作 C差|移动动作 G|ΔG|','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in data['rows']:
        if not r.get('focused_raw_verified'):continue
        d=r['decomposition'];f,m=d['actions']['fixed_action'],d['actions']['mobile_action']
        lines.append(f"|{r['key']}/{d['rival']}|{f['fixed_gap']:.6f}|{f['relative_compensation']:.6f}|{f['joint_gap']:.6f}|{m['fixed_gap']:.6f}|{m['relative_compensation']:.6f}|{m['joint_gap']:.6f}|{d['delta']['joint_gap']:.6f}|")
    lines.extend(['','逐候选 F、J、C、各恒等式的数值误差及不同动作的最近竞争者保存在 JSON。该分解使用同一批带噪计量的已保存局部拟合，精确解释开放线路后得分如何变化；它没有将预测背景误差、计量噪声或优化局部性分别确定为因果来源。',''])
    Path(target).write_text('\n'.join(lines),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,default=Path('outputs/mobile_v3_scalar_evidence_audit.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_compensation_decomposition.json'))
    args=parser.parse_args();x=run(args.audit,args.output);print(json.dumps(x['summary']),flush=True)


if __name__=='__main__':main()
