# 固定计划AC修正：续接

2026-10-17。源码分支 codex/ev-train-validation-design；实现阶段提交 6ed82322 已推送。执行副本 D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015。原冻结实验不再运行。

## 正在运行

开发输出 outputs/ev_ac_repair_development_v3；控制器PID 27724，policy 596c5f8d902792d5f01b9c58a694492e1d2a52b8bb7ed19239ac67ad52c87ba0。8份开发，2个worker；每份历史固定恒充、平方移动双向、原贝叶斯移动双向计划。计算期不要修改冻结Python与数学稿。

先读 controller.json、controller.stderr.log、development/history_*.progress.json。沙箱可能看不到外部进程，应在完整权限下核对CIM进程命令行；不能仅因沙箱Get-Process无结果重复启动。普通沙箱后台启动PID46364已核实消失；实际运行是完整权限启动PID27724。控制器有活动锁、逐worker启动登记及失败保存。

## 顺序

1. 等8份开发完成，读取 development_summary.json，检查G00/G10/G01/G11数值精度及失败、fixed/joint实际排序和费用。开发初值不足、积分失败、源越界均保留。
2. 只有开发资料用于修正程序。若改源码或数值配置，保留v3原记录并开新版本。数学稿中“两个具名初值”应在下一版改成“最多两个；Bayes中心不可用时只有history初值”，实际记录starts是准据。
3. 冻结最终科学源码、参数及32校准/32评价种子。每历史两情景各1次噪声、另有clean诊断；不把这些重复算成独立历史。
4. 校准只选G11的全局协方差倍率和条件AC的SSE保留阈值。完整32校准记录、失败分母、policy和输入哈希都要保存，才可启动独立评价。评价不能选倍率或修改方法。
5. 比较同域平方与原BC固定计划在修正推断下的结果。当前尚未更换MILP梯度；预测校准通过后再决定是否接入。完整AC传播只修正既有近似历史后验的未来传播，历史后验本身仍是单中心局部模型。
6. 独立回放AC、求解状态、源及线路域，报告均值修正、协方差修正、参数重拟合的效果与代价，保留失败、未决和配对历史数。阶段验证后提交推送授权分支，保护PDF和pyproject。

## 既有开发证据

旧预算pilot在 outputs/ev_ac_repair_development_v1/development/history_000.json.gz 与 v2/history_001.json.gz。两者实际继承GHK max_power16，均后验积分失败，缺BC与G传播。dev0 SQ固定判错而joint判对；dev1恒充固定判对而joint判错。128个保存状态独立AC重放通过；它们用于开发定位，不能当新版独立评价。

## 实现与验证

数学 ev_ac_repair_math.md；传播 ev_ac_repair_propagation.py；条件重拟合 ev_ac_repair_refit.py；单历史 ev_ac_repair_study.py；校准统计 ev_ac_repair_analysis.py；阶段控制 ev_ac_repair_control.py。71项相关测试通过。新控制器最多2worker，校准/评价强制完整32规格，校准文件与评价绑定。所有成本保留，源和末端电压越界不筛除后重算。
