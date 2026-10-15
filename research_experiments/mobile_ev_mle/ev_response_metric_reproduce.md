# EV响应指标：复算、证据与续接

2026-10-16。结论见 `ev_response_metric_results.md`。研究检出为 `C:/Users/23761/.codex/worktrees/36a1/Topo`，分支 `codex/ev-train-validation-design`。实际计算在独立D阶段目录进行；101份正式科学源码与研究检出逐字节核对。原报告及其他任务的文件不属于本次提交。

## 1. 哪些是独立评价

正式版本 `strict_binary_fixed_site_response_metric_v1`，policy_id：

`6961123743036dea395101d13d8884c2136fd4712cfed957af5ed4573871f37d`

数学/实现提交 `58e9c7a7`，冻结说明提交 `a29ddecf`。4开发为994101—994104；18校准为994301—994318；8独立评价为994201—994208。严格候选对star/clade_01、固定site0、统一精度、9种方法均预先固定。每个结构族校准9份、评价4份，统计单位为独立历史。

旧993系列用于误差分解，降为开发诊断。条件未来积分和一元平方距离MILP仅使用新的4开发历史，没有进入上述冻结策略或正式评价。不同阶段的样本数与策略版本分别保存。

## 2. 正式数据恢复与统计重算

证据目录 `evidence/ev_response_metric/`。`raw.tar.xz` 为9,092,260字节，SHA256：

`f51e88eae28e70f3564e7fa9e9db6ab94936a1fcffb35d325c5a7de6e7169f09`

它保存development/calibration/evaluation三个阶段的38个原文件，包括101份原字节科学源码快照、配置、环境、全部方法和失败分母。恢复目标必须为新目录或空目录：

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_response_metric/raw.tar.xz --output outputs/response_recovered
python -m research_experiments.mobile_ev_mle.ev_response_metric_analyze --development outputs/response_recovered/development --output outputs/response_dev_recomputed
python -m research_experiments.mobile_ev_mle.ev_response_metric_analyze --calibration outputs/response_recovered/calibration --output outputs/response_recomputed
python -m research_experiments.mobile_ev_mle.ev_response_metric_analyze --calibration outputs/response_recovered/calibration --evaluation outputs/response_recovered/evaluation --output outputs/response_recomputed
```

本次已实际恢复38/38个原文件，SHA全部一致。恢复后重新计算的数值校准阈值、逐历史候选决策和计数全部一致。报告包含绝对输入路径和由此生成的来源ID，换恢复目录后阈值文件及整份报告字节会变化；`replay_comparison.json` 明确记录这一点。统计重算不调用拟合、选择器或潮流。

首次阈值在读取评价前保存，其SHA256为 `29e7d7d7e161a0818055ebcef167d6b12760c93a0a512d0d637c0324e8f14686`，原件未覆盖。

## 3. 独立核查

```powershell
python -m research_experiments.mobile_ev_mle.ev_response_metric_review --directory outputs/response_recovered/development --output outputs/response_dev_review.json.gz
python -m research_experiments.mobile_ev_mle.ev_response_metric_review --directory outputs/response_recovered/calibration --output outputs/response_cal_review.json.gz
python -m research_experiments.mobile_ev_mle.ev_response_metric_review --directory outputs/response_recovered/evaluation --output outputs/response_eval_review.json.gz
```

本次30份历史全部核查通过：1,200个拟合状态、300个AC有向模型比较、440个价格坐标、420个对比模型、2,940次固定方向AC评分及46次严格增分接受。β与源导数的有限差分最大误差分别约1.33e−9、4.81e−9。核查使用独立潮流重建，不重新求参数或选择动作。旧四候选核查器的schema仅通过显式只读view适配；原记录和冻结源码不改，原schema错误保留在审计记录内。

## 4. 重跑冻结实验

使用 `D:/apps/miniconda3/envs/Topo/python.exe`，环境版本保存在freeze中。先设置数值库单线程，最多三批并行；下面给串行命令，新输出不得覆盖旧结果：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m research_experiments.mobile_ev_mle.ev_response_metric_study --output outputs/response_new_dev --seeds (994101..994104)
python -m research_experiments.mobile_ev_mle.ev_response_metric_study --output outputs/response_new_cal --seeds (994301..994318)
python -m research_experiments.mobile_ev_mle.ev_response_metric_analyze --calibration outputs/response_new_cal --output outputs/response_new_analysis
python -m research_experiments.mobile_ev_mle.ev_response_metric_study --output outputs/response_new_eval --seeds (994201..994208)
python -m research_experiments.mobile_ev_mle.ev_response_metric_analyze --calibration outputs/response_new_cal --evaluation outputs/response_new_eval --output outputs/response_new_analysis
```

正式运行校准分为301—309、310—318两批，按trial哈希合并。评价可独立计算，但结果在阈值保存前保持未读。启动、合并和阈值时间证据均随交付保存。35秒规划软预算与2秒MILP预算按方法独立，依赖计划与不可中断调用耗时另记。

## 5. 开发诊断的单独复算

`ranking_diagnostics.zip` 保存本轮旧数据误差分解，逐文件SHA在对应manifest中。`conditional_development.zip` 保存4份开发历史的32路径积分、104份源码和输入哈希；它以仿真真实过去状态条件化，属于机制诊断。

```powershell
python -m research_experiments.mobile_ev_mle.ev_response_conditional_diagnostic --directory outputs/response_recovered/development --output outputs/response_conditional_new
```

一元距离MILP的数学见 `ev_scalar_distance_math.md`，实现见 `ev_scalar_distance_milp.py`，独立开发入口为 `ev_scalar_distance_probe.py`。其输入限定4份994开发历史，参数仍固定，新增方法需要另行冻结和校准后才能报告独立排错率。运行细节、源QP界、网格、MILP界、完整AC接受及实际结果保存在独立开发证据，不与9法正式评价合并。

```powershell
python -m research_experiments.mobile_ev_mle.ev_scalar_distance_probe --directory outputs/response_recovered/development --output outputs/scalar_distance_new_dev
```

`scalar_development.zip` 保存105源码的独立冻结、4份开发记录及全部网格结果。18项标量测试、220项相关流程集成测试、10项独立多节点/移动审计测试共248项通过，分别保存XML。3/5/9/17网格各次MILP限2秒，累计35秒仅记录，用于开发网格审查，不写成严格同预算的正式策略对比。

```powershell
python -m pytest research_experiments/mobile_ev_mle/test_ev_scalar_distance_milp.py research_experiments/mobile_ev_mle/test_ev_scalar_distance_audit.py -q
```

## 6. 下一步验收

先检查一元MILP是否在相同求解预算下缩小冻结代理间隙，及完整AC是否接受其计划；再用训练可获得的负荷预测分布比较平均距离，避免把5条压力情景当预测覆盖。若对强平方基线仍只有偶发改进，应记录原因，再决定增加可识别的训练激励或滚动预测。固定点结果稳定后才检验移动收益，并扣除接线、驾驶电量和占用时段。新方法校准前保留所有开发失败，不追溯修改正式结果。
