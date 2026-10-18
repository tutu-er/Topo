# 复算

基于研究提交3ff97936，隔离分支codex/label-free-ac-bounds-20261018。环境Python3.11.15、NumPy2.4.6、SciPy1.17.1、Gurobi13.0.3（凸QP；无许可可回退SciPy加独立support-LP界）。不需要现场数据、已知真拓扑校准库或网络下载。

在仓库根目录运行，输出另设新目录：

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle/label_free_bounds -q
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.label_free_bounds.experiment --output outputs/label_free_replay --mode all --workers 2
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.label_free_bounds.audit --directory outputs/label_free_replay --destination outputs/label_free_replay_evidence
```

原执行目录D:/0-github_workspace/Topo/outputs/label_free_case33_bounds_20261018/final。pilot和main保留迭代来源；final是统一源码哈希的8组正式结果，只有final进入主表。source源场景与无源共享负荷/噪声种子，统计单位为4条独立基础历史，不把8组视为8个独立重复。

evidence/final_records.zip含8份meter、8份独立truth、10候选公开域、80个决定和后审计。evidence/source_snapshot.zip含21个实际加载的依赖文件及AST读取的静态case33数据；按source_snapshot_manifest.json校验原始字节，可避免Git换行转换造成hash差异。

evidence/compatibility_records.zip含首历史实际使用的三个输入（无truth）、10份AC解释和独立前推回代复核。全部候选使用同一个算法、两个起点和100次函数评价预算，阈值由M=9216、alpha=0.05直接计算。原来的SQ/G-SQ及其已知标签校准阈值均未使用。

复算补充AC解释：把上述compatibility包的inputs恢复到任意目录，传入实际文件路径：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.label_free_bounds.compatibility --meters outputs/label_free_replay/meters/case33_T96_seed20261018071_none.json.gz --assets outputs/label_free_replay/public_assets.json --qp-results outputs/label_free_replay/results/case33_T96_seed20261018071_none.json --output outputs/label_free_replay_compatibility --max-nfev 100
```

全部压缩包CRC、成员及sourcehash已核对，final包实际恢复后重算10候选，数值差为0，见evidence/recovery_audit.json。该文件还保存独立AC逐点复核。压缩证据约4.1MB；依赖版本和每个原始输入哈希在manifest中。
