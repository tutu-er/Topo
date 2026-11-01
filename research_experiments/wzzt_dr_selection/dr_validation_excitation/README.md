# 共同调权后的验证排除激励

- MODEL.md：完整数学模型及概率口径。
- REPORT.md：保持3%和原病例的实际观测重分析、均值反号、10场景小试验。
- profile_margin.py：复用PoolLP的全族调权、验证排除裕量与最优面核验。
- audit_existing.py：重读既有8观测，不生成物理未来。
- forecast_gate.py / scenario_gate.py：历史均值和整段残差参考门控。
- test_profile_margin.py：4个可手算的非唯一/训练验证隔离检查。
- ../evidence/validation_excitation_v1：仅含标量/离散支持的公开证据。

当前没有经过验证的新连续价格优化器；原SOCP/MISOCP仍属固定线性激励版本。本目录没有改生产RNJ/WZZT、3%验证或.16收缩。
