# 复原本轮数值证据

1. 核对evidence/package_receipt.json中的numerical_evidence_v1.zip SHA256，解压外层归档；manifest.json列出每个成员的大小与SHA256。完整GPU依赖不在归档内。
2. 在一个新的空执行目录中解压dependencies/reference_source_subset.zip，保持原相对路径。将current_source中的文件复制到research_experiments/mobile_ev_mle/ac_block_fit。依赖包仅含这些求解器及测试实际导入的原版本文件，不夹带其他旧实验；其哈希可对照dependencies/original_freeze.json。
3. CPU环境采用原记录的Python3.11.15、NumPy2.4.6、SciPy1.17.1、pytest。设置PYTHONDONTWRITEBYTECODE=1，OMP/MKL/OPENBLAS/NUMEXPR线程数均为1；PYTHONPATH为新执行根及其rnj_wzzt_core。

归档中的history_only_v1.json.gz已经是严格历史白名单，重放数值比较无需重新读取未来、真实线路或外部全年负荷。每个任务的task.json保留候选、历史前缀、初值规则、种子、重复编号和容差。历史包及任务哈希必须一致。

```powershell
python -m pytest research_experiments/mobile_ev_mle/ac_block_fit -q -p no:cacheprovider --ignore=research_experiments/mobile_ev_mle/ac_block_fit/test_gpu.py --basetemp <新建的测试临时目录>
python -m research_experiments.mobile_ev_mle.ac_block_fit.benchmark run --whitelist <白名单绝对路径> --output <新的结果目录> --prefixes 192 --algorithms ref_sparse,ref_jac,block_reference_gap,block_jac_gap,fixed_ref,fixed_batch --repeats 3 --joint-timeout 60 --fixed-timeout 30
python -m research_experiments.mobile_ev_mle.ac_block_fit.summarize --output <新的结果目录>
```

先创建测试临时目录的父目录，临时目录本身须尚不存在；避免pytest自动清理已有目录。原结果目录不要覆盖。

第一轮cpu_screen_v1归档保存精确旧源码overlay，按其source_snapshot_receipt.json和同目录source_snapshot.zip复原；第二轮和容差检查各自也有源码快照。首轮两个超时任务没有个体源码指纹，这个缺口按原样保存。

GPU复现先把gpu_requirements.txt安装到独立的outputs/gpu_deps，保持原NumPy/SciPy先导入，再调用gpu.configure_dependency_path。该函数仅在当前进程设置独立CUDA发现、临时目录和编译缓存；不需要修改系统/conda环境。按gpu_fit_v1中的记录重放gpu_fit_benchmark。要重现首次冷编译成本，应使用新的、事先声明的空缓存目录，保留原缓存与结果。

每次源码、数据、硬件和CPU竞争都要记录。原研究worker并行运行时的计时不是硬件恒定基准；应比较交替顺序的配对重复，并同时核对损失、物理状态及共同梯度，不能只复算速度比。
