# A完整审计归档

各 `.json.gz` 是同名原JSON的无损gzip压缩；`archive_manifest.json`同时保存原文件及压缩文件SHA256、大小。解压后的原JSON未删字段。原记录与完整审计输出留在D盘执行目录；本目录约0.8MB，用于Git交付。

先读研究目录的 `ev_ac_repair_results.md`。统计单位为32份独立评价历史，未来两情景在同一历史内平均。所有失败保留。A原150份科学源码及数据未修改；B在读取A评价内容之前已冻结。

- `main_statistical_audit_v1`：64份校准/评价绑定、独立Cholesky概率评分、校准倍率、失败分母与配对统计。
- `main_oracle_mean_audit_v1`：同一固定计划分别换真实线路、负荷、源的诊断；真值未进入实际选择器。
- `main_score_decomposition_v1`：联合分数与条件电气分数的离线分解，0新AC/优化/计划。
- `A_main_physical_audit_20261017`：完整AC物理回放与原始文件哈希。
- `A_main_refit_cost_audit_20261017`：SSE、P/Q/源、EV只计一次、同噪声、MILP操作、缓存及费用。
- `A_main_audit_compact_20261017`：两项AC审计与费用简表。
- 其余两文件保存执行命令、脚本/产物哈希及运行状态。

独立审计脚本位于上一级 `scripts/`；原源码快照、策略和开发检查不在此重复归档。
