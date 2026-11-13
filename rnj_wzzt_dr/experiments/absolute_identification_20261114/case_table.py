"""Public per-history counts and mean fees, with no individual-day records."""
import argparse
from pathlib import Path
from ... import io
from .report import label


def write(source,destination):
    report=io.read(source);histories=report['per_history']
    methods=[m for m in report['overall']['AC']['strategies'] if not m.endswith('_rotation')]
    days=sorted({h['days'] for h in histories.values()})
    if len(days)!=1:raise ValueError('Use separate tables for unequal repeat counts')
    roles=dict(correct_clear='明确正确',correct_ambiguous='弱优势',wrong_present='原胜者错误但含真',missing='缺真候选')
    paid=[m for m in methods if m!='no_dfs']
    lines=[f'# {len(histories)}个历史的{days[0]}日完整结果','',
        f"每行分母均为{days[0]}条新日；表中为AC下完整原生拓扑恢复次数，未决或不可行报价回退零动作，仍保留在分母中。",'',
        '费用为含零动作和回退日的平均实际激活补偿，无服务费用均为0，不代表一次非零报价或完整市场运营费。只公开逐历史汇总，原始P/Q/V、拟合参数及逐日行动和报价留私有输出。','']
    if any(m.startswith('pq_trust') for m in methods):
        lines+=['本轮使用查看此前全部结果后设计的方法和新的随机流。原开发／原留出仅为来源标记，原留出历史已用于诊断，不作为本轮未接触的历史验证集。','']
    fee_label='／'.join(label(m) for m in paid)
    lines+=['|历史输入|网络|历史资格／原分组|'+'|'.join(label(m)+'恢复' for m in methods)+'|平均费用（元/日）：'+fee_label+'|',
        '|---|---|---|'+'|'.join('---:' for _ in methods)+'|---:|']
    for identity,h in histories.items():
        own=h['physics']['AC'];name=identity.replace('main__input','主').replace('pressure__input','压力').replace('__101','')
        split='原留出' if h['history_partition']=='history_holdout' else '原开发'
        lines.append('|'+name+'|'+h['case']+'|'+roles[h['role']]+'／'+split+'|'+'|'.join(str(own[m]['native_correct']) for m in methods)+'|'+'／'.join(f"{own[m]['mean_settled_fee']:.3f}" for m in paid)+'|')
    totals=report['overall']['AC']['strategies']
    lines+=['','本固定面板总体：'+'；'.join(f"{label(m)}{totals[m]['native_correct']}/{totals[m]['planned_days']}" for m in methods)+'。',
        '', '同费轮换、配对新增／丢失恢复、超时、初值保留与费用边界见[完整报告](REPORT.md)。','']
    destination.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();write(a.report.resolve(),a.out.resolve())
