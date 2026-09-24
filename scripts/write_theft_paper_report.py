import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'theft_wzzt/outputs'
s1=json.loads((BASE/'theft_paper/summary.json').read_text());s2=json.loads((BASE/'theft_paper_v2/summary.json').read_text());s3=json.loads((BASE/'theft_paper_v3/summary.json').read_text())
assert all(s['status']=='complete' and not s['errors'] for s in [s1,s2,s3])
def rate(r):return f"{r['k']}/{r['n']}"
def percent(v):return f'{100*v:.1f}%'
def interval(r):return f"[{100*r['ci95'][0]:.1f}%, {100*r['ci95'][1]:.1f}%]"
def link(label,path):return f'[{label}]({(ROOT/path).as_posix()})'
rows=[r for r in s3['localization'] if r['method']=='shrinkage']
lines=['# theft_wzzt：论文容量评估与自主改进结果（2026-09-19）','',
'## 1. 判断','',
'原来六组算例加 M7 足以证明机制能运行，不能单独支撑可靠定位、统计可信度和现场适用性的完整主张。本轮已补齐三网络、独立开发/校准/测试、消融、压力与失败边界，获得了可组织成方法论文初稿的结果。**最合适的主线是未计量正负荷的区域集合辨识与校准可靠性；唯一母线定位、任意漂移鲁棒性和真实偷电判定均不成立。**','',
'能写成论文初稿不等于已达到投稿竞争力：三个网络仍是自建合成算例；没有标准三相不平衡/实测数据上的验证，也没有完成最接近已有方法的同数据复现比较。大批重复试验不能替代这些证据。','',
'## 2. 实际完成了什么','',
f"- 第一轮 {s1['completed']} 组场景：固定位置的自由 R/X 重估 LP、固定 R/X 基线、总表基线、电压增益；普通独立秩与分裂校准。",
f"- 第二轮 {s2['completed']} 组全新场景：加入向共同漂移收缩的逐边 R/X 先验，开发集冻结 λ=10；更保守的第195顺序统计量及复合零假设。",
f"- 第三轮 {s3['completed']} 组全新场景：模型及 λ 不变，仅改为独立校准分数的最大值，并重新验证误报、漏检、区域大小与幅值区间。",
'- 另外完成220组持续时间/位置切换边界试验；原测试与研究扩展共32项通过。历史134个受保护文件哈希未变，原 MILP、M7 校准及辨识树缓存保留。',
'- 每个正式场景均保存单独 JSON，可按冻结协议续跑；前两轮不理想结果全部保留。',
'',
'三个网络分别为 paper15（22母线/15终端）、soumalas11（16母线/11终端）、flynn16（23母线/16终端）。名称来自本地合成算例库，不能称为原作者或 IEEE 标准算例复现。三棵干净训练得到的约简树支持集合与真值的精确率/召回率均为1；这也意味着本轮尚未检验拓扑辨识错误下的鲁棒性。','',
'三轮正式试验之间种子完全分离。各条件内部的100条随机留出序列使用不同复制种子；条件间有意配对相同种子，不把全部场景数当作独立样本总量计算置信区间。时间点之间也不当作独立样本。','',
'## 3. 最终结果：第三轮新测试集','',
'下面每个网络使用100条独立无偷电测试序列和100条随机事件测试序列。事件位置均匀抽取非根节点，幅值从2/4/8 kW抽取，持续6点，起始点在固定32点窗口内随机。告警使用整个预设零假设库，无需知道真实压力条件。','',
'| 网络 | 正常误报 | 随机事件检出 | 点定位正确 | 区域集合覆盖 | 平均候选数/全部 | 整窗幅值覆盖 |',
'|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 f=next(v for v in s3['groups'] if v['case']==r['case'] and v['condition']=='nominal' and v['method']=='shrinkage')
 lines.append(f"| {r['case']} | {rate(f['composite_fpr'])} | {rate(r['composite_detection'])} | {rate(r['point'])} | {rate(r['region_coverage'])} | {r['region_size_mean']:.2f}/{r['candidate_count']} | {rate(r['amplitude_window_coverage'])} |")
lines += ['', '误报与覆盖必须连同统计误差阅读：', '', '| 网络 | 正常误报95%区间 | 检出率95%区间 | 区域覆盖95%区间 | 幅值半径(kW) | 幅值平均相对绝对误差 |', '|---|---:|---:|---:|---:|---:|']
for r in rows:
 f=next(v for v in s3['groups'] if v['case']==r['case'] and v['condition']=='nominal' and v['method']=='shrinkage')
 lines.append(f"| {r['case']} | {interval(f['composite_fpr'])} | {interval(r['composite_detection'])} | {interval(r['region_coverage'])} | ±{r['amplitude_radius_kw']:.3f} | {percent(r['amplitude_active_relative_mae'])} |")
lines += ['', '采用逐单元精确二项（Clopper–Pearson）区间，未宣称这些区间在全部表格上同时成立。幅值半径在0处截断；整窗覆盖要求32个时间点全部覆盖，不能与旧线损包络的逐活动点覆盖率直接当成同一种统计量比较。', '', '## 4. 压力条件与保守告警的代价', '',
'paper15 每个压力条件另有199条校准、100条无偷电测试，以及内部节点2/终端108各20条8 kW事件。正常阈值迁移、已知条件阈值和整个库的统一阈值全部报告。后者才是无需真实条件标签的最终告警。', '',
'| 条件 | 仅正常最大阈值误报 | 已知条件最大阈值误报 | 全库最大阈值误报 | 全库检出/40 | 点定位/40 |', '|---|---:|---:|---:|---:|---:|']
for cond in ['nominal','drift10','hetero10','loss_under25']:
 f=next(v for v in s3['groups'] if v['case']=='paper15' and v['condition']==cond and v['method']=='shrinkage')
 grid=[v for v in s3['grid'] if v['case']=='paper15' and v['condition']==cond and v['method']=='shrinkage' and v['amp_kw']==8]
 lines.append(f"| {cond} | {rate(f['transferred_nominal_fpr'])} | {rate(f['null_fpr'])} | {rate(f['composite_fpr'])} | {sum(v['composite_detection']['k'] for v in grid)}/40 | {sum(v['point']['k'] for v in grid)}/40 |")
lines += ['', 'drift10 是统一阻抗+10%；hetero10 是逐边独立±10%且每条序列重新抽取；loss_under25 是L1估计器输出下调25%的压力扰动。这里只保证有限的已声明分数生成机制，不是任意连续阻抗/线损误差区间上的全局鲁棒性。', '', '弱事件的检出也完整保留：', '', '| 网络 | 2 kW检出/40 | 4 kW检出/40 | 8 kW检出/40 |', '|---|---:|---:|---:|']
for case in ['paper15','soumalas11','flynn16']:
 counts=[]
 for amp in [2,4,8]:
  cell=[v for v in s3['grid'] if v['case']==case and v['condition']=='nominal' and v['method']=='shrinkage' and v['amp_kw']==amp]
  counts.append(f"{sum(v['composite_detection']['k'] for v in cell)}/40")
 lines.append('| '+case+' | '+' | '.join(counts)+' |')
lines += ['', '上述两位置的40条为配对场景汇总，未额外把它当40条完全独立样本做显著性主张。', '', '## 5. 哪些改进成功，哪些没有成功', '',
'1. **简化计算成功。** 位置和活动状态固定后，原二元乘积能消元，枚举全部候选的L1 LP即可精确解该受限模型。增添收缩先验后仍是LP。它不是对原逐时自由位置MILP保持问题不变的加速。',
f"2. **计算可扩展。** 第三轮正则化候选枚举每组中位时间{s3['timing']['shrinkage']['median']:.3f}秒，95分位{s3['timing']['shrinkage']['p95']:.3f}秒；这是推断部分，不含训练树和全部仿真。并行运行下的本机时间不能作为跨硬件速度保证。",
'3. **全边自由重估并不自动改善定位。** 第一轮随机点定位为70/74/75%，区域覆盖88/94/95%。第二轮正则化在同批新测试上的点定位78/74/68%，自由重估76/76/73%；逐网络配对精确McNemar检验均不显著（p约0.815/0.815/0.405）。不能拿不同批次70%与78%之差声称显著改进。',
'4. **低维倍率也有失败边界。** 一到两个公共R/X倍率能适应统一漂移，但在逐边变化的开发集明显变差。最终保留逐边变量并施加共同漂移先验，λ只由开发集选择。',
'5. **普通校准对一次小样本尾部估计不够稳定。** 第二轮flynn16区域覆盖90/100，线损压力误报11/100，均完整保留。第三轮不继续调整模型，而用更严格的最大分数界及全新测试。收益须与候选集合大小、幅值宽度和弱事件漏检一起评价。',
'6. **总表基线不可省略。** 它在本任务的较强事件中已经很容易告警；复杂优化的主要作用是区域解释和参数失配处理。平衡增益与输入幅值同源，不能作为独立证据再算一次。',
'7. **短脉冲/位置切换仍是边界。** 第一轮边界测试中，内部节点1点事件区域覆盖13/20；持续32点为15/20；切换2→108的20例中，固定位置集合没有一次同时覆盖两个真实区域。第三轮没有重新验证这些分布，不将第一轮数字冒充第三轮性能。',
'', '## 6. 论文可以怎样组织', '',
'建议题目方向：**基于辨识约简拓扑与校准候选集的配电网未计量正负荷区域定位**。', '',
'| 正文部分 | 应回答的问题 | 现有支撑 |', '|---|---|---|',
'| 问题与辨识边界 | 可判断存在、区域还是唯一母线？ | 约简树标签、隐藏支路等价性、位置/阻抗混淆 |',
'| 测量与线损 | 幅值从哪里来，误差是什么？ | L1测量线损、带符号残差、旧包络失败与新区间 |',
'| 优化方法 | 为什么可精确解，哪些限制必需？ | 固定窗口位置、H0并入H1、有限LP枚举、收缩先验 |',
'| 校准与集合 | 置信度到底保证什么？ | 独立零分布、复合零假设、最大分数容忍限 |',
'| 实验与消融 | 是否优于简单方法、哪里失败？ | 三网络、三轮留出、固定/自由R/X与总表基线、压力和边界 |',
'| 局限 | 什么尚不能外推？ | 三相、实测、错误拓扑、持续/移动多源、全天多重检验 |',
'', '这足以形成约8–10页方法与验证初稿的实质内容；页数只是组织建议。有限枚举、L1正则和秩/容忍限均是已有工具，不能包装为新数学理论。贡献应集中在测量约束下的区域辨识、误差混淆揭示和可审查的可靠性输出。', '',
'最近邻工作已包含拓扑、总表与电压联合优化定位及混合整数凸模型：[Carmona-Pardo等PSCC 2026官方论文页](https://pscc.epfl.ch/modules/request.php?action=summary.php&id=34&module=oc_program)。已核对其正文III–V节和式(1)：它假定拓扑/导线类型/相别已知，并在改编的三相四线53母线网络上模拟偷电。本项目的可能差异是数据辨识约简树、R/X误差处理及独立区域/幅值校准，尚未完成同数据数值对照。', '',
'## 7. 投稿前仍需补的关键证据', '',
'- 首要：在公开标准三相不平衡网络或可信实测数据上复现最接近的模型驱动方法，并使用相同测量、候选空间、噪声及预算。现有固定R/X、总表等是内部消融，不是领先文献复现。',
'- 验证辨识树错误、未知合法未计量负荷、长期持续偷电、移动/多源、不同功率因数，以及全天窗口选择的多重检验。当前不能把有限零假设库外的异常都判为偷电。',
'- 增加校准集重复抽样的外层研究，衡量不同校准集产生的阈值/集合大小波动；当前三轮结果已经显示这比继续堆同一阈值的测试样本更重要。',
'', '## 8. 验证、失败记录与文件', '',
f"第三轮{ s3['completed']}组全部完成，无求解失败。共记录{s3['diagnostics']['lp_count']}个主方法LP证书，最大直接目标误差{s3['diagnostics']['objective_error']:.3g}、最大标准化约束残差{s3['diagnostics']['scaled_constraint_violation']:.3g}、最大原始—对偶差{s3['diagnostics']['primal_dual_gap']:.3g}。这些是声明模型及数值容差下的证书，不是物理世界全局唯一恢复证明。",
'', '开发期间发生过相对路径与网络字段名两个启动错误，已修复且保留日志；没有删除失败研究变体。PDF Python依赖缺失时使用已安装的原生pdftotext完成原文核对。', '',
link('方法、公式与执行命令','docs/theft_paper_method_and_execution_20260919.md'), '',
link('最终推荐测量入口','theft_wzzt/detect_conservative.py'), '',
link('第三轮完整统计JSON','theft_wzzt/outputs/theft_paper_v3/summary.json'), '',
link('第三轮冻结协议','theft_wzzt/outputs/theft_paper_v3/frozen_protocol.json'), '',
link('第一轮完整结果','theft_wzzt/outputs/theft_paper/summary.json')+'；'+link('第二轮完整结果','theft_wzzt/outputs/theft_paper_v2/summary.json'), '',
'!['+'第三轮结果图'+']('+str((BASE/'theft_paper_v3/paper_results.png').as_posix())+')']
(ROOT/'docs/theft_paper_readiness_20260919.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(ROOT/'docs/theft_paper_readiness_20260919.md')
