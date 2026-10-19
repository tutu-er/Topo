# 复现与证据

本包是物理资产与根路径检查，不含新MLE/EV性能结果。

- `configuration.json`：固定4主种子、所有线路库、噪声规则、22份运行依赖的SHA256。
- `results.json`、`conditions.csv`：48条件、96个配对前缀行，失败分母及全部物理检查。
- `aggregate.json`：按布局/线路库/窗口聚合；3/7天不当作独立历史相加。
- `networks/`：12套完整资产、原长度和型号分配。
- `independent_audit.json`：48份保存状态的AC及噪声生成恢复，源hash核对。
- `scientific_sources.zip`：22份运行依赖原字节；逐成员恢复核对，避免Git换行规则改变字节hash。
- `all_raw_hashes.json`：全部48个原始NPZ的hash；全量保留在工作区`outputs/physical_lv_main_v1/raw/`。
- `representative_raw/`：两份可公开的合成代表记录，单位见配置；真值仅为审计材料。
- `tests.xml`：最终回归；`initial_environment_errors.xml`保留首次临时目录权限错误。

从本任务分支仓库根目录运行，Python3.11.15（完整版本见配置），NumPy2.4.6；每个命令使用全新的输出目录。设置OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS为1。

```powershell
python -m research_experiments.mobile_ev_mle.physical_local_v1.experiment --output outputs/physical_lv_reproduce
python -m research_experiments.mobile_ev_mle.physical_local_v1.finish --input outputs/physical_lv_reproduce --evidence outputs/physical_lv_reproduce_audit
python -m pytest research_experiments/mobile_ev_mle/physical_local_v1/test_physics.py research_experiments/mobile_ev_mle/reference_benchmark/test_networks.py research_experiments/mobile_ev_mle/reference_benchmark/test_representation.py research_experiments/mobile_ev_mle/reference_benchmark/test_cases.py -q -rs --basetemp=outputs/physical_lv_reproduce_tmp
```

重新运行的耗时、平台字段和NPZ容器hash不要求跨环境相同；先核对科学源码、种子、数组和物理误差。当前归档hash用于核对本批原始字节。原标准网络源码的可选audit字节文件可从原reference_benchmark证据恢复；缺少它们的三个测试会明确skip，其余测试仍执行。

新代码入口：`physical_local_v1.assets.build_network(name, profile="cable_v1")`；可用paper15、paper11（soumalas11/samem11别名）、flynn16、pengwah18。case33/CIGRE保持原参数。旧模块入口不自动改变，历史批次可复现。
