# 固定计划AC修正的复现

本目录B修正的是历史拟合及预测，不重新规划EV。固定计划来自A，全部候选、计量、未来噪声及物理域保持不变。A/B各32校准、32评价；历史是独立统计单位。

## 1. 源码与环境

`evidence/main_frozen_protocol.json`、`main_frozen_source.json.gz`锁定A150+B15份科学源码。原归档的键以A/B开头，不是可直接拼接到文件系统的安全相对路径；按冻结manifest的映射恢复，并检查目标仍在新工作目录内。另需 `runtime_dependency_snapshot.json.gz` 中7个原有导入文件；其工作树和Git内容核对见 `runtime_dependency_git_audit.json`。不得把最初的165份快照称为完整导入闭包。

环境见 `evidence/environment_packages.json`。本次使用Python3.11.15、NumPy2.4.6、SciPy1.17.1、Gurobi13.0.3；线性代数线程数均为1。`source_restore_audit.json`核对恢复内容，`restored_source_tests_v3.xml`记录恢复目录的55项测试通过；早期缺依赖的失败收据也保留。

`cold_actual_replay_audit.json`进一步记录：在实际恢复的165+7源码目录中，用原开发0完整重放，160个非时间数值量与原结果完全相同。这是同一份开发数据的复算，不增加独立样本数。

## 2. 精简输入

`evidence/replay_inputs_v1/`保留64份历史观测、公共模型/初值、候选、固定计划、原未来计量与无噪声计量。64份均逐项核对历史准备、评价payload和计划指纹；50份原complete、14份原partial_failure保留。它省略后验大矩阵、传播样本和真R/X等诊断数据；64个gzip约2.24MB，原A压缩raw约796MB仍留执行目录。

精简文件有自己的字节SHA，不能冒充原A raw。原冻结控制器的原始字节绑定继续指向D盘原文件；外部重放应另存输出及元数据。使用当前分支源码时先核对冻结科学文件SHA；也可从上述快照恢复。示例：

```python
from pathlib import Path
from research_experiments.mobile_ev_mle.ac_history_refresh import io, replay

bundle = Path("research_experiments/mobile_ev_mle/ac_history_refresh/evidence/replay_inputs_v1")
settings = io.load_record(bundle / "B_freeze.json")["settings"]
replay.replay_file(
    bundle / "evaluation/history_000.json.gz",
    Path("outputs/external_B_replay/evaluation/history_000.json.gz"),
    draws=settings["draws"], replicates=settings["replicates"],
    base_kw=settings["base_kw"], future_root_v=settings["future_root_v"],
    solver_options=settings["solver_options"],
    truncation_options=settings["truncation_options"], variance_multiplier=1.)
```

复现全流程时，先用32校准输入复算原倍率规则并保存新的输入hash绑定，再处理32评价输入。原控制器、原calibration及raw不覆盖。全部候选无法完成时保留该历史及失败，执行规则为保留全部候选。

## 3. 完整原始记录与独立审计

执行根：`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`。

- A：`outputs/ev_ac_repair_frozen_v1`，policy `fc21118dee52dd44d0b2a1a22dd2fc60de636936d5cddfdcb3caadf304931e13`。
- B：`outputs/ev_ac_history_refresh_B_v1`，policy `1b9d8f5d22ae6d3ca0c07f57a26466bc1bf9b83a0bde7cb80ee8688285e7d316`。
- A的原始统计、完整AC/费用、响应分解归档：`../evidence/ev_ac_repair_v1/completed_A/`。

B必须达到statistical_complete，再用冻结 `report.py` 同历史、同动作核对。`evidence/history_refresh_audit.py`核对全部保存MAP初值、试点、接受轨迹、原目标、物理域与数值驻点，不新增优化或GHK。`evidence/scripts/b_statistical_audit.py`独立Cholesky复算概率、负对数密度、覆盖、校准与失败分母。`evidence/predictive_risk_audit.py`仅积分已保存预测模型，随机数是数值积分点，不是新增物理数据。

## 4. 解释结果

G00/G11均传播所声明的近似历史后验；G11只是未来完整AC采样的均值/协方差，不是完整非线性历史后验的精确积分。B刷新了完整AC历史中心，最终仍用一个局部Gaussian。条件AC_joint用同一R/X解释历史和EV，与潜变量P/Q的Bayes模型分开报告。物理R/X后验方案见 `physical_rx_next_math.md`，未接入本批或MILP。

先看 `../ev_ac_repair_results.md` 和本目录 `results.md`。核对完整分母、配对区间、未决/误拒及费用，再判断修正是否有效；不以模型自预测风险替代实际独立评价错误率。
