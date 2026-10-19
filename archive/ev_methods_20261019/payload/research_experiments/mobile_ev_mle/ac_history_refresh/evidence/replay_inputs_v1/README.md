# 精简输入与外部复现

这些gzip记录保存同一历史观测、公共模型设置、历史初值、固定EV计划和原未来计量。
每条都核对原始与精简记录的 observed_input_sha256、evaluation_payload 指纹及固定物理计划指纹。
科学失败保持在分母中。数据不含历史后验矩阵、传播样本、未来联合拟合初值、真R/X或未来潜变量。

首先恢复发布的主165份源码快照和补充7个依赖，并安装记录中的Python/NumPy/SciPy/Gurobi环境。
在恢复源码的工作目录下，可以直接调用原B函数重放一个文件：

```python
from research_experiments.mobile_ev_mle.ac_history_refresh import io, replay
f = io.load_record("B_freeze.json")
s = f["settings"]
replay.replay_file(
    "replay_inputs/evaluation/history_000.json.gz",
    "reproduced_B/evaluation/history_000.json.gz",
    draws=s["draws"], replicates=s["replicates"], base_kw=s["base_kw"],
    future_root_v=s["future_root_v"], solver_options=s["solver_options"],
    truncation_options=s["truncation_options"], variance_multiplier=1.)
```

对全部32份校准输入按原规则重新校准，再评价32份评价输入；或只复算指定文件的原始预测。
输出耗时和source_A路径会改变，量测与模型输入的等价性由manifest的双重指纹证明。
精简文件有自己的字节SHA256，不能作为原A raw通过原metadata的字节绑定。
请把复现输出与新元数据另存，保留发布的原freeze、校准绑定和原始hash不变。
