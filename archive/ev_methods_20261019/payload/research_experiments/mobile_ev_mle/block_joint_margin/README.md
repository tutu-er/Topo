# 联合拟合优势EV小试验

阅读顺序：report.md（结果）→math.md（本版推导）→protocol.md（事前验收）→followup.md（后续未实现方案）。旧实验未修改。

运行单位测试，在Topo源码根目录：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle/block_joint_margin -q -o addopts='' -p no:cacheprovider
```

新空输出目录中顺序运行冻结版本：

```powershell
python -m research_experiments.mobile_ev_mle.block_joint_margin.study --phase development --output <新目录>
python -m research_experiments.mobile_ev_mle.block_joint_margin.study --phase evaluation --output <同一目录>
```

评价要求两份开发价格门槛通过。已有原始记录保持不可覆盖；复现使用新目录，续接已完成记录可用--resume。该版本的全局checkpoint是串行接口，并行时每个worker须使用独立输出目录，不能共享checkpoint临时文件。

只读复算本轮统计：

```powershell
python research_experiments/mobile_ev_mle/block_joint_margin/analysis/summarize.py --output D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/ev_block_joint_margin_v1
```

八份评价原始数据已封存；后续方案不使用它们做新的独立性能声明。驻点审计源码作为.txt存档，实际可执行脚本保存在D盘原始输出目录；它只使用历史及假想响应，先保存两次试步，再以--append-analytic-step追加解析步，不修改策略。
