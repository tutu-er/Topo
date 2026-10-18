# AC拟合的分块求解开发

保留完整AC、原R/X参数域及条件输出损失，改进数值求解。执行目录为D:/0-github_workspace/Topo/outputs/ev_ac_block_stage_20261018，原reference_benchmark冻结批次继续使用自己的执行副本，不接入本目录改动。

- fixed_source.py：固定线路，逐时二维有界源拟合；保留各期失败，最后整历史完整AC回放。
- blocks.py：直接保存共享线路及当期源导数块，避免全局源零块。
- linear.py：有界阻尼线性子问题，活动集与逐时QR消源；内层KKT及误差界独立记录。
- nonlinear.py：新的有界阻尼Gauss--Newton外层，完整AC决定步接受；与原SciPy TRF比较同一目标及域，而非要求轨迹相同。
- math.md、protocol.md：公式、等价前提、验收条件和数据边界。

当前79项小例测试通过；真实开发历史的端到端速度和GPU验证尚待完成，见checkpoint.json。求解器ftol/xtol成功和严格梯度验收分开保存；数值提速不等于新增线路信息。

测试命令（单线程BLAS、PYTHONDONTWRITEBYTECODE=1、PYTHONPATH为执行根及rnj_wzzt_core）：

```powershell
python -m pytest research_experiments/mobile_ev_mle/ac_block_fit/test_blocks.py research_experiments/mobile_ev_mle/ac_block_fit/test_linear.py research_experiments/mobile_ev_mle/ac_block_fit/test_fixed_source.py research_experiments/mobile_ev_mle/ac_block_fit/test_nonlinear.py -q -p no:cacheprovider
```

GPU依赖仅安装到执行目录outputs/gpu_deps；不改Topo conda环境，不提交大型依赖。CPU对照必须先载入原NumPy/SciPy，再插入GPU依赖目录，以保持同一CPU库。
