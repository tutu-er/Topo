# 参数域与短支路压力检查

2026-10-14。主策略68份科学文件和asset独立评价不变。本检查属于开发消融，不新设统计拒绝阈值。

## 固定条件

1. 将已完成的四条pilot_v2历史988301–988304在numeric域重新运行：逐边R/X范围均为[1e-6,.1]pu。真线路、历史计量、未来噪声及设备条件与原asset pilot完全相同，改变的只有竞争拟合域及其产生的策略。
2. 运行原驱动预置的2m内部支路压力：988501/cherry01及988502/cherry02，分别使用asset和numeric域。四个条件全部保留，不按结果挑选。
3. 所有九方法、全部四候选及12有向任务继续参加。优化预算、AC点检查、回退和局部拟合预算保持冻结值。

## 解释与验收

asset域来自公开电缆表与声明25–150m长度族；2m真支路违反该长度先验。因此结构在候选表中与真实参数在允许域内分别记录，域外不能写成真模型覆盖。

逐历史核对asset/numeric的真实参数、原历史、未来噪声确切一致；两域分别保存源码快照、完整配置、求解界、各候选R/X、实际M和失败。分析比较域对恒充/设计的共同影响，以及设计减恒充的配对增量，不把缩窄参数域的效果计为EV效果。

独立评价仍仅指原asset evaluation_v2的12条历史。numeric复用pilot不增加独立评价分母；两条压力历史不是风险校准样本。若宽域下设计没有优势、AC门失败或拟合失败，保留原结果。

运行入口按顺序为：

```
python -m research_experiments.mobile_ev_mle.ev_original_study --phase pilot --domain numeric --output research_experiments/mobile_ev_mle/evidence/ev_original_domain/pilot_numeric_v2
python -m research_experiments.mobile_ev_mle.ev_original_study --phase stress --domain asset --output research_experiments/mobile_ev_mle/evidence/ev_original_domain/stress_asset_v2
python -m research_experiments.mobile_ev_mle.ev_original_study --phase stress --domain numeric --output research_experiments/mobile_ev_mle/evidence/ev_original_domain/stress_numeric_v2
```

等待主12H与电气位置消融完成后串行运行，避免同时的大量AC计算影响计时解释。
