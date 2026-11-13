"""Publish aggregate32-day outcomes for every preselected repeat history."""
import argparse
from pathlib import Path
from ... import io


def write(report,destination):
    data=io.read(report)
    if data['sample_design']!='stratified_repeat44' or len(data['per_history'])!=44:raise ValueError('Expected locked repeat44 report')
    methods=['no_dfs','old_gamma2','symmetric_guard','weighted_nominal']
    roles=dict(correct_clear='明确正确',correct_ambiguous='弱优势',wrong_present='原胜者错误但含真',missing='缺真候选')
    lines=['# 44个固定历史的32日完整结果','',
        '每行均包含32条预先锁定的新日，失败或未决报价执行零动作，保留在分母中。这里只列AC下完整原生拓扑恢复；物理线性结果另见各批汇总。44个历史根据既有资格选定，未按新日效果挑选。',
        '', '“旧”指期望证据增量方法，“保护”指多目标绝对恢复加方向保护，“分配”指按竞争难度分配误判额度。费用为32日平均实际激活补偿，包含未采购和回退的零费日；无服务费用全部为0。这不是完整市场运营费，也不是一次非零服务的报价。',
        '', '**正例：** 历史留出的Pengwah18主129，恢复由18/32提高至保护方案27/32，同费轮换24/32；保护方案平均激活费12.54元/日，30/32日实际采购。旧方法23/32，分配方法20/32。',
        '', '**下降例：** 历史留出的Soumalas11压力042，无服务15/32，旧方法13/32，保护方案14/32，分配方法15/32。',
        '', '**概率判断反例：** Soumalas11压力043两种绝对恢复方法32天均判定零动作已充分，模型下界至少98.55%，实际只有3/32恢复。逐日约束、初值和原生选择已独立复算；偏差分解见[补充批核验](repeat44/VERIFY.md)。',
        '', '|历史输入|网络|历史资格／分组|无服务恢复|旧恢复|保护恢复|分配恢复|平均费用：旧／保护／分配（元/日）|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for identity,history in data['per_history'].items():
        if history['days']!=32:raise ValueError('Unexpected per-history denominator')
        own=history['physics']['AC'];label=identity.replace('main__input','主').replace('pressure__input','压力').replace('__101','')
        split='留出' if history['history_partition']=='history_holdout' else '开发'
        counts='|'.join(str(own[m]['native_correct']) for m in methods)
        fees='／'.join(f"{own[m]['mean_settled_fee']:.2f}" for m in methods[1:])
        lines.append(f"|{label}|{history['case']}|{roles[history['role']]}／{split}|{counts}|{fees}|")
    lines+=['','按固定补充面板汇总，无服务948/1408，旧方法955，保护方案959，分配方法951。保护方案净新增11次恢复，集中于3个历史，另1个历史下降。主400历史总体结果另见[主批报告](confirmation400/REPORT.md)，不与本表未经加权合并。',
        '', '逐历史的条件二项区间与配对日bootstrap留在本地REPORT.json；32次重复支持对固定条件的评估，其区间不代表跨网络、跨历史的普遍保证。公开表格只含汇总次数和平均费用，原始P/Q/V、拟合参数、逐日价格及行动轨迹留在私有输出。','']
    destination.write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();write(a.report.resolve(),a.out.resolve())
