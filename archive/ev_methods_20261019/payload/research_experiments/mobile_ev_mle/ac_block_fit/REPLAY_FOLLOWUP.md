# 复原本轮后续核查

本包复现数值求解及固定端点核查，保留原失败与全部分母。它不新增拓扑辨识的科学样本，也不修改原冻结批次或已有 `numerical_evidence_v1.zip`。

包内包括历史白名单、共同梯度诊断、50/60位AC复算、32个原拟合端点、32个预算内修整端点、8次单独声明预算的交叉初值核查、case33坐标映射与物理AC曲率分解、固定端点的有限EV注入完整AC核查、当前源码、数学说明、报告和检查点。旧 `83e8bdee...` 修整源码按原字节随原证据保存；wrapper记录修订另有版本与测试receipt。

## 核对与运行

1. 对照外部 `package_receipt.json` 核对归档SHA256。解压到新的空目录；`manifest.json` 给出每个成员的大小及SHA256。
2. `source/` 是可独立运行的源码根。它含当前模块及测试实际导入的原冻结依赖，依赖字节对照 `dependencies/original_freeze.json` 核验。完整GPU运行库和旧实验库不在包内。
3. CPU环境使用原记录的Python、NumPy、SciPy和pytest版本；50/60位复算使用Python标准库Decimal。设置 `PYTHONDONTWRITEBYTECODE=1`，BLAS/OpenMP线程数均为1，`PYTHONPATH` 包含解压后的 `source/` 及其 `rnj_wzzt_core/`。在 `source/` 内运行：

```powershell
python -m pytest research_experiments/mobile_ev_mle/ac_block_fit -q -p no:cacheprovider -c <独立pytest.ini> --rootdir . --ignore=research_experiments/mobile_ev_mle/ac_block_fit/test_gpu.py --basetemp <全新测试目录>
```

独立 `pytest.ini` 只需一行 `[pytest]`。测试临时目录的父目录须已存在；临时目录本身用新名字。保留原输出，不覆盖研究结果。

打包脚本先在开发执行副本收集CPU测试实际导入的依赖，核对原冻结源码，再解压归档到独立目录运行全部CPU测试。它检查所有项目导入均来自解压根、导入文件哈希与归档一致，最后再次核对原893份科学源码。实际测试数量和通过情况以 `verification/restored_cpu_tests/test_receipt.json` 为准。

第一次准备运行发现父目录的pytest配置把rnj导入指向主工作树；28个文件统一换行后源码文本均相同。该次来源检查失败的证据保留。最终验证用独立配置隔离父目录设置，并要求所有项目导入都来自解压根。

独立解压的首轮测试为155/156通过，一项固定线路的二维源QP触发活动集迭代上限；同源码、同环境的唯一完整复验为156/156通过。首失败和短探针均保留，原因尚未解释。最终包状态明确记录这项计算不稳定性，未将它写成全程无失败，也未据此启用QR替代原方法。

## 数据和旧版本

`data/history_only_v1.json.gz`、`data/candidate_history_v1.json.gz` 仅含允许用于这些核查的已到达历史。回放这些核查无需读取未来量测、真线路参数或完整全年负荷数据。

`replay_inputs/cpu_precision_v1/` 仅保存50/60位诊断必须读取的两个原端点。需要重放该诊断时，将它们复制到新源码根的 `outputs/cpu_precision_v1/`，并将对应计划和结果复制到新的 `outputs/common_gradient_diagnosis_v1/`。使用旧源码快照运行时，保留其原始输入哈希。数值任务中记录的原执行目录是来源信息；迁移到新目录时应在任务副本中显式替换执行根并记录映射，保留原任务文件。

每批的计划、输入哈希、源码哈希、日志、失败结果、源快照和完整轨迹保存在 `evidence/`。重放某批旧程序时，按该批自己的源码快照覆盖到另一个新源码根，再检查其记录的哈希。当前源码不会替代旧版本解释旧结果。

50/60位复算仅验证已保存端点附近的数值变化；FP64微小容差带是数值接受规则。8次交叉初值拟合使用单独声明的核查预算。有限EV注入核查固定已保存的线路及源，不重新拟合，不据此声称EV规划或拓扑辨识效果改善。

## 重新生成证据包

待各核查的完成记录、最终报告和检查点已保存后，使用原执行副本运行：

```powershell
python -m research_experiments.mobile_ev_mle.ac_block_fit.followup_package package --output <新的归档目录> --finalize
```

该命令检查32/32/8个端点分母和有限注入核查完成状态，运行CPU测试、构造小型依赖闭包、重新解压并测试。它不会启动原科学worker，也不会启动新的拟合批次。

本次归档在首轮恢复失败后停止了自动封存。经明确批准的一次完整复验通过后，以包内 `verification/packaging_completion/publish_instability.py` 保存两轮原记录并完成封存；该步骤不追加测试或修改数值源码。权威状态以外置 `package_receipt.json` 和 `replay_receipt.json` 为准。
