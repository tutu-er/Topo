# AC拟合的分块求解开发

先读[当前数学模型](current_model.md)、[本轮共同梯度与多候选核对](FOLLOWUP_REPORT.md)和[续接检查点](followup_checkpoint.json)。新增预算内驻点修整，开发156项CPU测试通过；独立恢复曾有1项未解释失败、唯一完整复验通过，原失败保留。充分收敛的8候选及5候选对的参数/非零功率AC导数一致。case33另8次同起点严格拟合仍未通过，已定位一个由R/alpha映射切点主导的拒绝步。原TRF继续默认，QR暂不接替后续科学实验；物理R/X凸约束GN修正仅完成推导。

新证据的复算见[REPLAY_FOLLOWUP.md](REPLAY_FOLLOWUP.md)，原单候选/GPU证据及下列旧分母保持原样。

保留完整AC、原R/X参数域及条件输出损失，改进数值求解。执行目录为D:/0-github_workspace/Topo/outputs/ev_ac_block_stage_20261018，原reference_benchmark冻结批次继续使用自己的执行副本，不接入本目录改动。

- fixed_source.py：固定线路，逐时二维有界源拟合；保留各期失败，最后整历史完整AC回放。
- batched_source.py：源状态仍逐时独立，AC/导数/二维步批量计算，避免逐行Python调用开销。
- blocks.py：直接保存共享线路及当期源导数块，避免全局源零块。
- linear.py：有界阻尼线性子问题，活动集与逐时QR消源；内层KKT及误差界独立记录。
- nonlinear.py：新的有界阻尼Gauss--Newton外层，完整AC决定步接受；与原SciPy TRF比较同一目标及域，而非要求轨迹相同。
- reference_variants.py：原TRF加相同列尺度的强对照；gpu.py支持FP64完整AC/导数，QR仍在CPU。
- math.md、protocol.md：公式、等价前提、验收条件和数据边界。

当前116项CPU组件测试及4项GPU测试通过。117个CPU计时条件、12个CPU/GPU拟合及2个共同精度检查完成；结果见REPORT.md，证据复原见REPRODUCE.md。paper11的QR原尺度相对原TRF配对中位快约9倍；GPU在32末端受控例约快1.33倍、在CIGRE更慢。共同严格梯度目标未通过，原冻结批次保持原实现。数值提速不等于新增线路信息。

测试命令（单线程BLAS、PYTHONDONTWRITEBYTECODE=1、PYTHONPATH为执行根及rnj_wzzt_core）：

```powershell
python -m pytest research_experiments/mobile_ev_mle/ac_block_fit/test_blocks.py research_experiments/mobile_ev_mle/ac_block_fit/test_linear.py research_experiments/mobile_ev_mle/ac_block_fit/test_fixed_source.py research_experiments/mobile_ev_mle/ac_block_fit/test_nonlinear.py -q -p no:cacheprovider
```

GPU依赖仅安装到执行目录outputs/gpu_deps；不改Topo conda环境，不提交大型依赖。CPU对照必须先载入原NumPy/SciPy，再插入GPU依赖目录，以保持同一CPU库。
