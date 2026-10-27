# iFlex / SimBench 数据校准入口

本阶段确认原始数据并检验简单价格条件均值。先读REPORT.md；校准失败记录不能删去，也不能与旧A2合成实验合并。

## 本地数据与源码分离

原始和家庭级派生数据缓存位于：

`C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_data_calibration`

`raw/`包含4个已校验原始文件、选定负荷缓存、带split的评估轨迹档案；`deps/`仅含本次隔离安装的pyreadr/narwhals；`packages/`及`simbench_source/`保存冻结网络数据包与原始CSV。这些目录不入Git。重新取得数据时，设置自己的缓存目录即可。pickle仅由本任务从已校验原始文件生成；不要加载来历不明的pickle。

在本目录运行，PowerShell示例：

```powershell
$env:WZZT_DR_DATA_DIR = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_data_calibration'
python acquire.py
python inspect_data.py
python calibrate.py
python verify_calibration.py
python control_training_audit.py
python topology_audit.py --help
```

首次安装使用独立Python环境，依赖见requirements-calibration.txt。当前实测环境Python3.13.12、numpy2.4.4、pandas3.0.3、pyreadr0.5.8、narwhals2.27.1。取得网络包可使用`python -m pip download --no-deps simbench==1.6.3`，校验SHA-256 `8b48bdab2445286a96649eb18bb09476d000a0e212fd57fdba83399df54af4de`，解压其中`simbench/networks/1-complete_data-mixed-all-0-sw/*.csv`（无需Profile表）。`topology_audit.py`支持独立指定源目录、包目录和输出文件。

## 结果索引

- `protocol.json`：在拟合之前固定的均值模型、日期边界、指标及接受条件。
- `outputs/raw_manifest.json`：发布MD5实算验证和本地SHA-256。
- `outputs/initial_data_audit.json`、`data_quality.json`：对象、字段、编码差异、完整性与完整分母。
- `outputs/calibration_run.json`：实际代码、协议、输入哈希、依赖和运行状态。
- `outputs/mean_factors.csv`、`normal_day_factors.csv`：参数，不称因果弹性。
- `outputs/split_manifest.csv`、`group_day_results.csv`、`split_metrics.csv`、`menu_metrics.csv`：59组日及两个模型的训练/测试结果。
- `outputs/whole_day_residual_diagnostics.csv`：描述性波动诊断，非已校准的随机生成器。
- `outputs/verification.json`：独立聚合重算与实际轨迹核验。
- `outputs/topology_audit.json`：固定来源、原选中网络行、开关收缩后图、物理块及接口限制。
- `protocol_control_training.json`及`outputs/control_training*`：失败后的一次训练期同期对照诊断，含原始执行源码和控制台编码失败回执；没有二次测试期模型搜索。

当前禁止将测试轨迹作为设计样本，禁止用此模型宣称新价格更优或WZZT恢复提升。原网络与iFlex属于不同来源，后续整合只能称基于实测需求响应的公共网络仿真。
