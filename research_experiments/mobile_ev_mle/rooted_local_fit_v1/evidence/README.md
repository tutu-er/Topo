# 复现与数据边界

主结果是rooted_local_main_v2：64条件、416训练槽；独立历史单位为2网×4种子，窗口配对。`configuration.json`保存生效配置和28份源码hash，`scientific_sources.zip`保存这些原字节，避免Git换行差异。

- `all_rows.csv`：全部训练、带噪留出、共同AC评价及封存后的真值诊断，失败/未驻点保留。
- `aggregate.json`：分网络、长度、上下文和方法的统计；参数均值只对明确合格的base给出分母。
- `rankings.json`、`ranking_summary.json`：四候选、数值并列、每次可用性和逐对顺序。最小J不叫正确概率。
- `audit.json`：576次独立AC、256次响应及128个QP界复算；留出线路和训练封存核对。
- `record_hashes.json`：64组完整训练/留出及输入文件hash；全量原始数据留在工作区`outputs/rooted_local_main_v2/`。
- `representative_compact/`：事先固定的两网首种子、7天FULL base，以及**全部4个未达驻点的训练解**；保留必要状态和可复算参数，不重复上传整个原始批次。NPZ只有当时可用的带噪观测，真值由固定seed事后恢复。
- `initial_batch_checkpoint.json`、`development_solver_comparison.json`、`numerical_version_comparison.json`：初版中止、独立数值修订依据及43个配对求解结果。初版未完成槽没有写成完成。
- `tests.xml`：68项相关测试。`package_manifest.json`锚定后处理代码和精简代表包的版本。
- `compact_audit.json`：精简代表重新进行15次AC、5次响应复算，最大损失差5.46×10⁻¹¹；`postprocessing_sources.zip`保留两份后处理脚本的原字节。

环境：Python3.11.15、NumPy2.4.6、SciPy1.17.1、项目已有Gurobi13.0.3，Windows。每个进程设置OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS为1；进程数4。

从任务分支仓库根目录执行，每次选择新的输出路径：

```powershell
python -m research_experiments.mobile_ev_mle.rooted_local_fit_v1.experiment --output outputs/rooted_reproduce --workers 4
python -m research_experiments.mobile_ev_mle.rooted_local_fit_v1.finish --input outputs/rooted_reproduce --evidence outputs/rooted_reproduce_evidence
python -m research_experiments.mobile_ev_mle.rooted_local_fit_v1.package --input outputs/rooted_reproduce --evidence outputs/rooted_reproduce_evidence
python -m pytest research_experiments/mobile_ev_mle/rooted_local_fit_v1/test_models.py research_experiments/mobile_ev_mle/rooted_local_fit_v1/test_audit.py research_experiments/mobile_ev_mle/physical_local_v1/test_physics.py research_experiments/mobile_ev_mle/local_response_score/tests/test_scoring.py research_experiments/mobile_ev_mle/reference_benchmark/test_representation.py -q --basetemp=outputs/rooted_reproduce_tmp
```

主拟合器只接收观测、候选和声明域；留出固定R/X。`all_rows.csv`中的clean/true/exact_RPXQ等列是**封存后的仿真诊断**，没有进入初值、权重或选拓扑规则。`holdout_J`是各方法自己的带噪留出评分，`common_AC_holdout_J`把所有训练线路统一带入完整AC，不再估计R/X。

RP+XQ的leaf列使用真实接收端支路功率隔离参数误差；`focus_cumulative_RPXQ_relative`按原根到末端路径求和，`path_compensation_ratio`是有符号路径误差范数与逐边绝对误差累积范数之比。`leaf_unique_X_voltage_RMS_pu`是去掉P方向后Q变化造成的一阶X电压量级，仅诊断信息量。它们都不是额外量测或概率。

旧[0.25,1.5]范围只作为OLD-RATIO控制，实际新拟合入口默认[0.05,2]。旧实验模块常量及原PDF保持原样。若要复现原始字节hash，使用归档源码；跨环境比较先检查数值数组、可行性和目标，耗时与压缩容器hash可能变化。
