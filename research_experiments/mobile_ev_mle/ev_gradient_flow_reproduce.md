# EV梯度方法：完整流程复算与续接

2026-10-15。研究分支 `codex/ev-train-validation-design`。数学、实现与冻结提交依次为 `5c3fec5e`、`c7574044`、`e17a5457`。结论见 `ev_gradient_flow_results.md`；这批正式结果没有用于修改策略。原LaTeX/PDF未修改。

## 1. 实际生效版本与证据

正式策略版本 `end_to_end_gradient_EV_v2_canonical_square`，policy_id：

`fc223c9f45ac45b7e76fa7a5e96d1556c49874910254827d34ea46161fecce8f`

`evidence/ev_gradient_flow/` 保存原始记录归档、清单、逐字节恢复报告、正式统计、独立物理核查、源码对应关系、启动配置和测试结果。`delivery_manifest.json` 为交付文件的SHA256清单。压缩的JSON均保留原文件解压字节；不需要重新拟合即可读取。

`raw.tar.xz` 共34,419,396字节，SHA256：

`8773cfe3700bacb97f850ad2914172311aa82c3abd900d2475d6397ddff2c686`

归档包括四阶段：pilot（1份旧开发，同一个993101，不增加独立历史分母）、development（4份v2开发）、calibration（36份）、evaluation（16份），共68个原文件。归档中的 `source_snapshot.json.gz` 保存97份实际运行科学源码的原字节和哈希；正式阶段三份源码快照一致。分析和只读核查工具独立记录各自哈希，不参与冻结策略ID。

## 2. 恢复原始记录，重算统计

在研究检出根目录运行，恢复目标必须为新目录或空目录：

```powershell
python -m research_experiments.mobile_ev_mle.ev_dynamic_delivery restore --archive research_experiments/mobile_ev_mle/evidence/ev_gradient_flow/raw.tar.xz --output outputs/gradient_flow_recovered
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_analyze --calibration outputs/gradient_flow_recovered/calibration --evaluation outputs/gradient_flow_recovered/evaluation --output outputs/gradient_flow_recomputed
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_analyze --development outputs/gradient_flow_recovered/development --output outputs/gradient_flow_development_recomputed
```

本次已实际完成68/68原文件逐字节恢复，继而从恢复目录重新计算。`thresholds.json` 和 `evaluation.json` 均与首次统计逐字节一致，对应SHA256：

|文件|SHA256|
|---|---|
|thresholds.json|6f9b181a3a0891423b900a9bf5babf384d494dde99b16dd39a3ced814dd2b9f7|
|evaluation.json|8bf85a1bb4f1432c4648f08e4d081afb99465daedd1a39c32377654ab332c0e5|

复算统计不运行AC求解、拟合或选择器。分析输出包含目录路径，所以不同恢复位置的整个 `summary.json` 不要求逐字节相同；上表两个统计文件要求一致。

## 3. 独立物理与梯度核查

```powershell
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_review --directory outputs/gradient_flow_recovered/calibration --output outputs/gradient_flow_calibration_recheck.json.gz
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_review --directory outputs/gradient_flow_recovered/evaluation --output outputs/gradient_flow_evaluation_recheck.json.gz
```

核查器读取保存的固定参数、源状态和动作，使用独立潮流重建与有限差分，不调用新拟合、选择器或生产AC预测器。正式52份历史共4,084个AC有向模型对、2,208个I²保存对、6,640个价格坐标、247个严格提高AC预测分数的已接受步骤通过核查；34,728次独立前向批计算属于核查成本，未伪装为策略运行成本。I²保存对的检查是其自身数值一致性检查，不声称它等于完整AC。

正式统计另独立复算7,072项成本和次数关系，全部通过。校准中的3份历史保存有约1.07e-5至1.60e-5的最大归一化源一阶残差；评价最大约3.40e-6。局部源拟合及线性MILP界均不作为非线性全局最优证明。

## 4. 重跑冻结流程

本次使用 `D:/apps/miniconda3/envs/Topo/python.exe`；Python 3.11.15，gzip/zlib 1.3.2。Python、NumPy和SciPy版本保存在冻结记录的环境信息中。重跑前检查97份科学源码是否与冻结快照一致，避免把新版本结果混入本批。可在独立研究检出调用已有 `ev_fixed_train_source_restore.audit_source_bytes(directory, expected_count=97)` 默认只读检查；不要覆盖生产检出。

先设置单线程数值库，最多四批并行。下面展示串行命令，所有目标目录必须新建：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_study --output outputs/gradient_flow_new_dev --seeds (993101..993104)
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_study --output outputs/gradient_flow_new_cal --seeds (993301..993336)
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_study --output outputs/gradient_flow_new_eval --seeds (993201..993216)
python -m research_experiments.mobile_ev_mle.ev_gradient_flow_analyze --calibration outputs/gradient_flow_new_cal --evaluation outputs/gradient_flow_new_eval --output outputs/gradient_flow_new_analysis
```

每种结构由种子序列顺序循环配置。本次正式运行是3个连续12份校准批和1个16份评价批；各批起点均对齐四结构循环，合并时按原文件哈希核对。阈值在读取评价结果前保存。正式13种方法、35秒独立规划软预算、2秒主MILP、最多2初值×2轮均在冻结配置中；不可中断调用耗时和依赖计划成本分别记录。没有跨方法共享AC评分缓存。

本次完成127项相关测试，命令：

```powershell
python -m pytest research_experiments/mobile_ev_mle/test_ev_fixed_response.py research_experiments/mobile_ev_mle/test_ev_fixed_gradient.py research_experiments/mobile_ev_mle/test_ev_fixed_gradient_math.py research_experiments/mobile_ev_mle/test_ev_gradient_flow_math.py research_experiments/mobile_ev_mle/test_ev_gradient_flow_design.py research_experiments/mobile_ev_mle/test_ev_gradient_flow_study.py research_experiments/mobile_ev_mle/test_ev_gradient_flow_analyze.py research_experiments/mobile_ev_mle/test_ev_gradient_flow_review.py -q
```

## 5. 下一阶段的边界

本批已完成理论、选择器及完整独立评价。下一批应先在新开发历史上检查固定训练参数对EV响应的预测误差，以及移动损失的两个计量时段，再决定改进目标或调度。若要检验严格的“两候选宇宙”，需预先固定候选对并重新校准；本批pairac仅在规划中取训练前二，最终始终评价四候选。滚动多轮、现场数据、扩网和全局最优证明均未在本批完成。
