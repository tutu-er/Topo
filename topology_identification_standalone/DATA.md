# 本地数据和算例生成

本目录附带四个物理树的完整定义，以及近期连续一天96点面板的全部48个历史输入。默认回放只访问本目录文件；不读取原项目的`outputs/`，不下载数据。

## 附带的历史面板

| 算例 | 物理母线数 | 线路数 | 观测终端数 | 输入数 |
|---|---:|---:|---:|---:|
| paper15 | 22 | 21 | 15 | 12 |
| soumalas11 | 16 | 15 | 11 | 12 |
| flynn16 | 23 | 22 | 16 | 12 |
| pengwah18 | 26 | 25 | 18 | 12 |

每个算例有4组种子、3档电压与根表相对噪声：0.0002、0.0005、0.001。每个输入包括：

- 训练：连续一天96个15分钟时点。
- 验证：独立3天，共288点。
- 测试：独立一天96点。

共有144个`*_observed.npz`，合计7,891,367字节。它们与历史生成文件逐字节一致，不是这次重新模拟的数据。`catalog.json`中的文件路径均相对于本目录；`provenance.json`记录原记录的哈希与来源，来源字符串仅作审计，不在运行时访问。`physical_seeds.json`和每输入的`diagnostics.json`保留原始种子、条件与数据诊断。

四个网络共享同一组随机种子，因此不能将全部48输入当成48个独立随机重复。全部是本项目中的文献风格合成网络；文件名不是对相应论文原始数据或完整算法的复现声明。

## 单位与量测定义

`cases/<case>/`保存`buses.csv`、`branches.csv`、`metadata.json`。四网的基准均为0.4kV、0.2MVA，即200kW/kvar。原物理线路表的R/X单位为欧姆。

观测文件包含：

- `P`、`Q`：按200kW/kvar归一化的终端有功与无功，正号表示消费；形状为`天数 × 96 × 终端数`。
- `V`：终端电压幅值，标幺值。
- `root`：同一时点的根表电压幅值，标幺值。
- `labels`：与数组列一一对应的终端母线标签。

加载后逐日展平，不抽样、不丢弃时点，构造

\[
Y_{ti}=V_{{\rm root},t}^2-V_{ti}^2.
\]

同一根表样本被用于该时点所有终端目标，因此根表误差会造成终端间相关误差。P/Q相对量测噪声标准差为0.005；电压和根表噪声按上述三档配对。历史根物理电压变化幅值设置为0.003，它与根表测量噪声不同。实际电压范围、设计矩阵秩和信噪比见各输入诊断文件。

`Observation`只保存允许用于辨识的观测：`p/q/y/labels`以及可选`voltage/root`。真实树独立保存在物理网络文件中。算法入口只应接收训练和验证`Observation`；`case_truth`与测试观测只供事后评价。

## 数据接口

在这个独立目录内运行Python：

```python
from data import load_catalog, load_observations, case_truth, verify_bundled_data

catalog = load_catalog()
job, meta = next(iter(catalog.items()))
train = load_observations(meta["train"])
validation = load_observations(meta["validation"])
print(train.p.shape, validation.p.shape)  # (96, n), (288, n)
print(verify_bundled_data())

# 下面两项只在拟合与选模结束后读取
test = load_observations(meta["test"])
truth = case_truth(meta["case"], train.labels)
```

`case_truth`以零起始列索引返回每条物理线路的下游终端集合。评估内部块时需排除单点和全集。`load_input(job)`是评价脚本的便利函数，返回`train, validation, test, truth, metadata`；算法内部不要调用它。

`data/manifest.json`给出全部附带网络和数据文件的SHA-256清单。新目录可以整体复制到任意位置，默认路径通过`data.py`自身位置解析。

## 自包含的新算例生成器

```python
from data import generate_synthetic

train, validation, test, truth, metadata = generate_synthetic(
    case="paper15", seed=7, points=96,
    voltage_sigma=0.0005, pq_sigma=0.005, root_sigma=0.003,
)
```

这个新生成器只依赖NumPy和本地网络表。它用于可复现扩展测试，不声称重现历史面板的随机轨迹。

负荷采用正消费的连续日曲线、节点相位差和AR(1)变化；有功与无功具有额外独立变化。训练、验证与测试使用独立随机流，验证包含三天。默认根电压以1.02pu为中心，含幅值参数0.003的物理变化。P/Q、终端电压及根表分别加相对高斯量测误差；同一个根表误差用于全部终端目标。

平衡径向AC潮流使用可读的后向电流累加、前向电压更新：

\[
I_i=\overline{S_i/V_i}+\sum_{j\in\mathrm{children}(i)}I_j,
\qquad V_j=V_i-Z_{ij}I_j.
\]

阻抗基准为$Z_{\rm base}=V_{\rm base}^2/S_{\rm base}$。隐藏母线零注入，给定负荷位于观测终端；源电压给定。每个样本收敛至电压更新小于$10^{-10}$，最多200次；异常或未收敛会显式报错。

已用独立的复数Ybus节点功率平衡方程和SciPy求根核验四个网络的名义潮流，电压幅值最大差低于$6.5\times10^{-13}$pu；相同种子的新观测完全重现。记录见`data/generator_validation.json`。这项核验只针对本地新生成器；历史数据另以原字节哈希校验。
