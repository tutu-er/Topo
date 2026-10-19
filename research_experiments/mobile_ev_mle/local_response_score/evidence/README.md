# 可复现证据

本包约8.94 MB，数据均为声明的项目合成算例，无现场私人计量。主批原始输出保留在`outputs/local_response_main_v1`，全文件hash见`full_local_manifest.json`。

- 16条件、8个既有网络/种子族；3/7天配对，四法各64候选训练全部合格。
- `summary.json`保存全部分母、排序及计时摘要，`benchmark.json`是新配对256次重放，全部损失复现。
- `audit.json`为全部16条件、286个AC状态独立重算；`physics_independent_audit.json`为128个余项QP及拒绝门槛的独立核查。
- `physics_summary.json`保留全部128个primal/dual/gap、阈值及共同结构，删去重复的逐时大矩阵；原76 MB物理记录保留在本地。
- `source_snapshot.zip`保存28份冻结科学源码及本轮独立审计、数学和复算文件。
- `representative_records.zip`保留两网种子33001的3/7天4条件、四法完整状态、含噪输入及配对旧AC记录。
- `restore_receipt.json`记录实际从归档源码恢复paper15的3/7天2条件并重算通过；不是声称全部16条件优化重跑。
- `unit_tests.xml`记录142项回归通过。最初临时目录权限错误和物理中心对齐错误的版本在原始本地目录保留，文件hash可追溯。

没有用真P/Q或真线路计算部署评分，没有以点排名替代拒绝证书。全部候选保留，内部二元分组未决。原PDF及旧冻结科学文件未改。
