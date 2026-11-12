# 旧代码归档边界

主线抽取后，旧代码和文档已按原路径移入 `archive/legacy_research_20261112/`。当前实际保留清单见根目录 [KEEP.md](../KEEP.md)，归档位置和恢复方法见 [本次归档](../archive/legacy_research_20261112/README.md)。运行中的任务、未提交研究代码、私人轨迹和原始结果资产保持原位。

## 当前运行闭包

辨识、冻结评分、零动作核算和单对报价只导入本目录文件及环境依赖。`identification` 四文件逐字节复用现有独立引擎；DR 的功能片段从交接工作树抽取，并去掉旧批量驱动、动态 `run/vendor` 注入和历史 `.24` 输入限制。`PROVENANCE.json` 记录原文件哈希、抽取符号及迁移修改。

原辨识器目录须待当前严格 96 点任务及其后处理完整封存后，才可整体搬到历史代码区；同时保留观测、网络与研究记录。新目录不带 48 组 P/Q/V 和试验 ZIP；CLI 接收明确路径。读旧输入属于数据依赖，不是旧辨识器代码依赖。

## 分别保留什么

| 原位置／类别 | 后续处理 |
|---|---|
| `topology_identification_standalone` 的四个引擎文件 | 本包已有哈希一致副本；原目录保留历史版本与其独立试验工具 |
| `topology_identification_standalone/data, cases, experiments` | 独立保存数据和完成证据；本包不复制大数据，也不替这些旧批次重跑 |
| `miqp_loss_benchmark_20261111`、`mse_exit_tuning_20261111` | 旧算法和 `.24/.001` 结果谱系，可归档；当前运行无导入 |
| DR 工作树的旧建池、Hbar、均值、VAE/copula、重复批量和绘图脚本 | 历史研究，可归档；当前使用其已抽取积分／设备代码 |
| 原 `outputs` 内的冻结物理 `source/`、网络 metadata、历史输入／原协议 | 原 AC 桥和复现仍需要；移动后要更新明确路径并重做源闭包核验 |
| 私有恢复根中的 P/Q/V、报价 NPZ、完成记录、源快照 | 留私有，保持协议／哈希／随机流和所有失败／限时记录 |
| 当前 Git 未提交改动、活动批次的锁和完成状态 | 由各任务处理；此次不纳入提交和归档 |

## 原 AC 接口保留的外部资产

`compat/ac_bridge.py`、`compat/research_experiments/wzzt_dr_selection/dr_competition_bridge/physical_bridge.py`、`.../end_to_end/physical_case.py` 为原文件原字节副本。以 `compat/` 为 `repo_root`，仍使用原 `open_runtime(source_root, metadata_ref, protocol_ref=..., ...)`。它按协议逐文件校验，再加载原模拟器。

实际执行前需准备新路径下的冻结协议：必须绑定**本包桥文件和辅助文件的新绝对路径及哈希**，以及原冻结物理源闭包、网络元数据、噪声、背景流。旧完成协议保留原样，不改写其历史哈希。原桥加载 `rnj_wzzt` 物理包，需原完整冻结源及其环境依赖（含 pandas/networkx），本包不替换原物理模拟器。仅 `fit/zero/score/plan` 不加载该桥，也不访问这些外部资产。

归档时可把旧代码及原源快照整包搬走；在新位置通过原协议恢复核验后再清理旧位置。不能仅复制本目录就宣称重跑全部旧 AC 实验。

## 移动前的最小核验

1. `run.py doctor` 核对包内代码和资源。
2. `run.py check --solver` 核对原评分、方向、冻结、设备费用及小模型。
3. 将目录复制改名，用 `python -I /new/name/run.py doctor` 与 `check` 验证隔离运行。
4. 对实际使用的数据／原 AC 资产保存私人路径、哈希和恢复记录；禁止把拟合矩阵或报价轨迹加入公共 Git。

文档 `evidence/dr_handoff` 是原交接快照，其相对链接和工作树描述属于原位置。当前操作入口以本目录 README/MODEL 为准。
