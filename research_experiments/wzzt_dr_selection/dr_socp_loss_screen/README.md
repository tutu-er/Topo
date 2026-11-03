# SOCP原损失排除条件：先核查，不盲目优化

已完成结果见 REPORT.md，数学与适用范围见 MODEL.md，预锁计划见 PLAN.md。

- kernel.py：共享高斯因子的精确方差、原L1差的单侧修正、固定报价最优行风险分配、末端能量检查。
- screen.py：冻结输入/hash、12报价有界核查、逐次结果、确定性重算。
- test_kernel.py：8项不同核验，含独立密集协方差、数值积分及小SLSQP参照。

已完成run_v2：12/12报价、48/48界，0通过；338项重算通过。没有随机抽样、价格优化、新AC、拟合或W搜索。所提安全界过于保守，而且原误差模型未经执行控制不保持末端能量，未进入新价格优化阶段。

在本机恢复仅需读取REPORT和evidence/socp_loss_screen_v1；不要重启已完成记录。代码需原私有冻结输入；新批如获授权，可使用以下入口（在本目录运行）：

    python -m unittest -v test_kernel
    python screen.py prepare --repo <原工作树> --base <旧冻结run_v1> --output <全新私有目录>
    python screen.py run --output <该目录>
    python screen.py audit --output <该目录>

协议将源码、设备库及输入哈希冻结。公开不包含P/Q/V、完整R/X或报价数组。固定报价的行风险分配优化不等于已经实现联合定价SOCP。
