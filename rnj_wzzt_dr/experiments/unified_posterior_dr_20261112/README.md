# 统一后验判据与DR实验

本批已按预算执行，包含未取得报价和求解超时。先读[最终结果](results/final/REPORT.md)；启动前冻结的[方案](../../plans/unified_posterior_dr_20261112/PLAN.md)保留原文，[执行状态](../../plans/unified_posterior_dr_20261112/STATUS.md)另记。所有新数据和拟合参数留在忽略目录`outputs/unified_posterior_dr_20261112_v1/`；本目录只发布代码、概率及必要汇总。

101表示RNJ提供候选＋WZZT逐步加线＋结束前联合重选连接；111额外增加末端局部重选。三位开关依次为RNJ引导（R）、末端局部重构（T）、最后一次联合重构（E）；WZZT加线在八种配置中始终存在。T/E属于训练阶段的结构搜索，两种方法最后都按验证MSE选择输出。复用冻结训练参数，不修改原生产默认。

## 已完成的阶段

两批400个原96点输入、800个101/111候选库均已按原64训练＋32验证重评分。未重新拟合辨识器。先审查来源、计算并封存全部权重，再关联原真族。1628个来源／观测／结果文件核对原档案SHA通过；800/800与原独立审计的覆盖和原生首选正确性一致，逐支持预测与高精度概率独立复算的最大概率误差约1.01e-13。

| 方法／输入比较 | 库含真 | MAP完整恢复 | 90%正确唯一输出 | 90%错误唯一输出 | 90%真族误拒 |
|---|---:|---:|---:|---:|---:|
| 主240，101 | 117 | 110 | 100 | 88 | 2 |
| 主240，111 | 112 | 110 | 105 | 101 | 1 |
| 压力160，101 | 33 | 31 | 27 | 74 | 0 |
| 压力160，111 | 35 | 34 | 33 | 86 | 0 |

800次比较复用观测，不能当作800次独立试验。当前800次后验MAP均与原生首选拓扑相同。503次库缺真；另有12次库含真但MAP错误。因此，本批已包含旧100库回溯中未出现的“库含真而首选错误”情形。权重是声明残差模型和有限候选库内的支持程度。

按预定结构强度排序及离线样本资格选择14个输入、28个条件目标，每目标2048条工作模型路径，完成零DR开发检查。输出门槛依预定、只读模型结果的规则在独立新日前锁为90%。90%门槛下，28目标中6个有零动作充分证书、15个有剩余概率缺口的数值证据，其中5个目标确为真拓扑，来自4个网络。代码不将目标真值标签交给报价器。

工作模型保留原5%峰值功率误差、AR(.7)及同节点P/Q共享因素；剩余电压响应误差采用事前约定、冻结的共同sigma²。它与实际AC量测／设备法则分别评价。

历史统计、模型开发表与门槛图见[阶段交付](results/development/REPORT.md)。随后选取4个主要输入和2个零费用对照，各对历史前两个目标尝试报价，共12次：4个目标零费已足够；7个目标无法满足所用指数概率界；1个非零求解在30分钟预算内未找到可行点。原后验概率问题的非零可行性仍未解决。

独立模型确认每目标2048条，未发现频率区间与解析下界冲突。原交流潮流生成器先重放6个旧日，P/Q/V及根量测全部复现；再生成6×32＝192条独立新日。主要难例的无DR正确唯一输出为31/32、9/32、2/32、9/32；两个零费对照均为32/32。仅执行了有合格报价的零动作策略，未执行的非零策略明确留空。全部1280行有效评分复用这些物理观测，不扩大独立样本数。本批没有非零DR改善证据。

从整库后验到概率界、多个竞争者及容量检查的[推导](DERIVATION.md)给出公式意义和保证范围。指数概率界无法满足，只能排除本次保守保证；未据此宣称原概率问题不可行。原合同支付按名义报价计，实际交付的费用函数值单列为资源损耗代理。

## 入口与验证

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
& $py -B -m unittest rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.test_posterior rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.test_prospective -v
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.history_scan --output outputs/a_new_private_run
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.development --run outputs/a_new_private_run
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.pricing --run outputs/a_new_private_run
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.physical --run outputs/a_new_private_run
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.confirmation --run outputs/a_new_private_run
& $py -B -m rnj_wzzt_dr.experiments.unified_posterior_dr_20261112.finish_report --run outputs/a_new_private_run --public outputs/a_new_private_run/public_final
```

需要同机保留原large96两批输出和records两卷档案；输入、封存、源码和运行环境写入私人协议。上述命令是复现顺序，已有完成结果不重跑。使用新输出目录，禁止覆盖旧实验。23项前阶段检查与5项报价／执行检查均通过；另对1280行实际评分独立逐支持复算全部候选权重、结构错漏和四个门槛决定。原始新日及真实功率日志分别保存，评分端只读取观测。验证摘要见[记录](results/final/VALIDATION.json)。

文件职责：`posterior.py`只负责冻结尺度的累加评分；`prospective.py`计算概率界和模型路径；`pricing.py`在同一个原设备图中加全部主要竞争约束；`physical.py`接原交流潮流与设备执行器；`confirmation.py`冻结策略后生成独立路径；`finish_report.py`独立核验并汇总。旧辨识器和设备模型继续复用。
