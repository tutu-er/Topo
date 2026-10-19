# 保存数据的EV效果摘要

实现：`ev_compact_effect_summary.py`。只读既有分析CSV及collector风险JSON，不运行选择、拟合、优化或重新校准。

## 配对量与分母

令 \(i\) 表示一条独立历史，\(m\) 表示方法。实际裕量为

\[
M_{im}=\min_{b\ne a_i}J_{im}(b)-J_{im}(a_i),
\]

其中 \(a_i\) 是离线评价使用的真候选，\(J\) 是完整AC联合拟合取得的白化SSE。正裕量表示真候选取得更小误差。无噪声裕量仅去除未来噪声，保留该历史。

对主方法 \(m\) 和对照 \(c\)，保存每历史的

\[
\Delta_i=M_{im}-M_{ic},\qquad
\overline\Delta=\frac1{n_{mc}}\sum_{i\in I_{mc}}\Delta_i.
\]

\(I_{mc}\) 是两方法均有完整裕量的历史，\(n_{mc}=|I_{mc}|\)。每项对照另报全部已执行历史数、有效配对数、缺失或失败数。四种对照来自同一组历史，合起来仍只有原来的独立历史数。未执行历史单列为pending，不填零。

R/X列沿用真候选模型下的相对参数误差。它们可能在其他候选拟合失败时仍可计算，因此各指标保存自己的有效数。配对差统一取“主方法减对照”：M增加有利，R/X误差下降有利。所有统计均为描述性统计；不由M正负定义统计误拒。

## 曲线来源

从同一历史的所有方法CSV行读取`selected_origin`，沿父方法追到MILP、质量规则、恒充或随机终点。区别为：

- `direct_new_MILP_count`：本层保存的新增MILP曲线标志为真。
- `ancestral_MILP_count`：至少经过一个父方法，最终来源为MILP。
- `from_MILP_count`：终点为MILP，包含直接与继承。
- `ancestral_new_MILP_count`：继承路径的终端MILP保存了新增曲线标志。

因此一个`refined → v2g_fixed → MILP`方案可以本层新增数为0、继承MILP数为1。MILP终点但未新增曲线的情况继续保留终端标志；缺父条目、缺来源、循环保持unknown，主方法经过随机来源单列。两套递归可通过`crosscheck_lineages(rows, reference_trace)`逐条核对，不依赖重新读大trial。

## 校准结果的接入

可选输入是`ev_compact_collect`保存的完整风险JSON。助手核对标准小记录的内容哈希、policy、阶段、历史ID集合，以及分析元数据存在时的原trial压缩文件哈希。逐方法与两主方法共同阈值结果分开读取。

误拒、唯一正确、排除错误候选个数、未决和计算失败来自保存的校准决策；不从裕量或设计分数另作判定。漏真历史的误拒事件未定义，单列漏真数及含真分母，全部执行历史仍留在失败和未决分母。开发数据未附风险结果；校准阶段仅列阈值，不计作评价结论。

## 调用

```powershell
python -m research_experiments.mobile_ev_mle.ev_compact_effect_summary `
  --analysis <ev_compact_analysis输出目录> `
  --risk <collector风险JSON，可省略> `
  --output <新的效果摘要目录>
```

输出`summary.json/md`、逐历史`paired_primary.csv`和`risk_decisions.csv`。JSON保存输入字节哈希与助手源码哈希；不使用这些结果调整冻结策略。
