# 独立 wzzT Benders 原型

日期：2026-09-22。

已在 `experiments/wzzt_benders_prototype.py` 实现当前 **无候选池、单次新增支撑** MILP 的 Benders 分解。此原型不导入生产核心，不替换现有入口，不改变核心文件、旧实验或校准链。

## 实现范围

- 主问题：新支撑 z、上三角 y=z zᵀ、与旧支撑的三类层次关系，以及损失下估计 θ。保留全部合法非空、非重复、与旧族相容的支撑。
- LP 子问题：全部旧权重及新权重 r/x、乘积 u、场景/终端截距和 L1 残差。跨场景共享线路参数。
- 割：统一提升 LP 的对偶给出的全域仿射下界，不采用候选池或 no-good 排除。
- 证书：独立保存全局下界、可行上界、绝对 gap 和逐轮 trace。`proven_optimal=True` 只表示当前固定边界单步问题达到指定数值容差。
- 求解器：SciPy `milp` 主问题 + `linprog(method="highs")` 子问题。采用外循环重求解主问题，尚无回调、割筛选或热启动优化。

旧支撑在一次扩展中不变，旧权重会重新拟合。该接口没有实现整条前向路径、1-SE 选择、候选生成或自动扩界，也不宣称整树全局最优。

数学推导见 [完整域 Benders 推导](wzzt_full_domain_benders_derivation_20260922.md)。

## 运行示例

在仓库根目录：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/run_wzzt_benders_demo.py
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B -m pytest -p no:cacheprovider tests/test_wzzt_benders_prototype.py -q
```

`-B` 禁止写入导入模块的字节码；测试关闭 pytest 缓存写入。demo 默认输出 `outputs/wzzt_benders_prototype_20260922/demo_report.json`，可以使用 `--output` 指定另一份报告以保留旧结果。

直接调用：

```python
from experiments.wzzt_benders_prototype import solve_best_laminar_extension_benders

# scenarios: 每项包含 P_terminal、Q_terminal、drop_target，形状均为 (T, n)。
# supports: 当前已有支撑的终端位置索引，例如三终端时 [(0,), (1,), (2,)]。
result = solve_best_laminar_extension_benders(
    scenarios,
    supports,
    r_upper_bound=2.0,
    x_upper_bound=2.0,
    max_iterations=200,
    time_limit=60.0,
    absolute_tolerance=1e-8,
    relative_tolerance=1e-8,
)
print(result.support, result.objective)
print(result.proven_optimal, result.stop_reason)
print(result.lower_bound, result.upper_bound, result.absolute_gap)
```

参数上界沿用灵敏度原子系数单位；并不直接代表欧姆。DataFrame 按终端列标签对齐，时间索引不一致时明确报错。`r_values/x_values` 的最后一项对应新增支撑，之前依输入次序对应旧支撑；`intercepts` 形状为 `(场景数,n)`。

到达迭代/时间预算时可以返回已经找到的可行解，但必须检查 `proven_optimal`。没有合法扩展时返回 `stop_reason="no_feasible_extension"`、空支撑和空目标。时间预算按整次调用累计，子求解器只获剩余时间；这是软件层预算，不是外部进程硬杀时限。

## 数值证书处理

为了避免把未最优 LP 的原始目标误用为割锚点，割使用实际对偶下界的常数和梯度。对残差对偶检查/修正截距零和条件及 L1 盒约束，对有限边界变量计算拉格朗日下确界补偿。仍以浮点数值容差解释其证书，不是精确有理数证明。

每个最终可行上界都用取整后的合法支撑和有界权重独立重建预测再计算，避免将乘积可行性误差误计为结构收益。若下界/上界明显矛盾、主问题结构非法、LP 未解出或出现数值停滞，停止并保留失败状态，不作成功认证。

## 已完成的小规模检查

最终版本通过 **29 项测试（1.59 秒）**，记录为 `outputs/wzzt_benders_prototype_20260922/pytest_report.xml`。额外核验包括非秩一分数 y 的割、极小总时间预算、未最优 LP 不生成割，以及上下界倒置超过请求容差时拒绝认证。

独立测试使用自行构建的固定支撑 dense LP 和全部合法支撑穷举，不依赖原型或核心的建模函数。覆盖 3/4 终端、正负 P/Q、噪声、有旧支撑、多场景共享参数、割的全域有效性及生成点紧性、无合法扩展、未完成迭代和非法输入。

首轮另有四个与原核心 `solve_best_laminar_extension_l1(candidate_supports=None)` 的匹配对照，均有两个场景。相同数据、相同权重边界、相同平均 L1 目标：

| 案例 | 原 MILP 秒 | Benders 秒 | Benders 迭代 | 目标绝对差 |
|---|---:|---:|---:|---:|
| n=3，无噪、无旧支撑 | 0.140 | 0.375 | 5 | 1.66e-17 |
| n=3，有噪、旧单点支撑 | 0.260 | 0.307 | 3 | 1.55e-15 |
| n=4，有噪、旧双分支 | 0.430 | 0.234 | 2 | 1.24e-16 |
| n=4，有噪、旧嵌套支撑 | 0.406 | 0.351 | 3 | 5.99e-15 |

四项均完成数值最优性认证，选择的新支撑一致；独立重建预测 MAE 的差异均小于 1.9e-14。当前数据只是正确性示例，不能据此判断大规模普遍提速，尤其前两个例子 Benders 更慢。

最终版本再次运行四项对照，全部通过，最大目标差仍为 5.99e-15。完整输入、输出、每轮上下界、割、原型源码哈希和环境版本存于 `outputs/wzzt_benders_prototype_20260922/demo_report_final.json`；首轮报告与上表计时保留在 `demo_report.json`。运行环境为 Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1。

demo 对 28 个生产核心 `.py` 文件在运行前后计算 SHA256，全部一致；主估计文件的 SHA256 也匹配实现前记录：

```text
B5155BAF7EE86471E4C721122C04A2BD00431CBD195C9054183640C97EC5D208
```

## 文件

- 原型：`experiments/wzzt_benders_prototype.py`
- 独立测试：`tests/test_wzzt_benders_prototype.py`
- 与原核心对照：`scripts/run_wzzt_benders_demo.py`
- 最终对照结果：`outputs/wzzt_benders_prototype_20260922/demo_report_final.json`
- 数学推导：`docs/wzzt_full_domain_benders_derivation_20260922.md`

后续应先扩大终端数与样本数，在统一总预算下测量主问题时间、LP 时间、割数和未认证率，再决定是否值得替换单步求解器。当前原型保留为独立实验入口。
