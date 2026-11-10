# paper15 概率报价的精简入口

主流程只使用两个入口文件。

| 文件 | 输入 → 输出 |
|---|---|
| `full_price.py` | 完整历史、已发布预测、原资源及可选起点 → 统一费用优化、概率/设备核验、冻结报价 |
| `full_evaluate.py` | 冻结报价及原封存自然日 → 独立抽样、同费对照、原AC和冻结候选最终错漏 |

共用的`devices.py`、`continuous.py`、`gauss_frequency.py`、`full_adjoint.py`继续作为设备/数学库复用。它们不是新的求解入口。`test_*.py`、公式及结果审计脚本用于核验，历史试验入口`run.py`仅复用其数据读取函数，不启动旧两变量优化。

## 模型

`x`是60资源×24小时的归一化请求，`u=Devices.nominal(x)`是60资源×96槽的kW名义交付。模型预先计算两候选的`a0`与影响矩阵`G`，搜索时只计算`a=a0+G*u`。费用沿用原二次合同奖励。概率积分保留同单元两候选的共同电压误差，约束为原损失差`D>9.179191188409`的数值概率不低于.90。

全部历史、冻结R/X、原设备和资源、w2/common3%/.16均不变。当前规划仅积分给定功率预测均值和名义交付后的独立电压创新。完整功率与DR随机性仍在报价冻结后评价。

## 统一求解

默认起点是原旧90合同，也可明确传入同一信息集下的冻结报价。所有有容量的1290小时变量都自由。统一过程为

1. 检查零合同和起点。
2. 以固定旧合同费用缩放目标及梯度，直接用SLSQP最小费用，允许起点不满足90%，默认最多60迭代/300秒。
3. 若仍无可行返回点，接最多10次求值/30秒的原概率梯度可行性恢复。
4. 全部已核验可行返回点中取费用最低者，强制16/32/64阶概率重算，核对设备和合同费，再冻结。

所有阶段共用一份trace/checkpoint，失败保留。数值达标严格使用`p>=.90`，`1e-7`只用于积分加密。局部返回不证明费用最优。

旧的`full_price_continue.py`、`full_price_scaled.py`、`full_price_restore.py`已从活动实现移除，其已完成批次可由Git `78eb5b17`和私有源快照恢复。既有225次求值、556秒、43.401926报价和AC成绩属于原批。精简后的求解路径发生改变，本轮仅核验模型计算与冻结价一致，没有再次优化或仿真。

## 调用

以下`ARCHIVE`是既有`wzzt_dr_p90_stage/run_v1`，`PUBLIC`是`wzzt_dr_forecast_day_v1/run_v1/prepared/public`，`SEALED`是后者的`run_v1`。它们均位于私有scratch，输出也应放私有目录。

```text
python -B full_price.py --archive ARCHIVE --public PUBLIC --protocol README.md --out NEW_PRIVATE_RUN
```

核验已有冻结报价，不进行优化或AC

```text
python -B full_price.py --archive ARCHIVE --public PUBLIC --protocol README.md --warm-quote QUOTE.pkl --check-only --out NEW_PRIVATE_CHECK
```

独立评价仅在报价冻结后启动。`--quote`必须对应优化目录中的完成凭据和冻结记录，不能直接送入任意数组。

```text
python -B full_evaluate.py --archive ARCHIVE --public PUBLIC --sealed SEALED --quote NEW_PRIVATE_RUN/quote.pkl --protocol FULL_PRICE_PLAN.md --out NEW_PRIVATE_EVALUATION
```

本次不执行上述新评价批。规划和评价分别冻结来源指纹，评价结果不得回传价格。

## 调度与理论

- 当前精简数学正文为`../dr_paper15_delivery_margin/pricing_model/main.tex`，详细逐式证明保留在Git `78eb5b17`的同文件及89项`formula_verification.json`。
- `SCHEDULE.md`、`schedule_summary.csv`、`schedule.png`给出当前冻结合同的节点、时段及资源调度。完整报价和资源×时段数组保持私有。
- `FULL_PRICE_REPORT.md`与`full_price_results.json`保留原独立结果，`refactor_check.json`仅记录本轮模型/报价重放与测试。两者不合并实验分母。
