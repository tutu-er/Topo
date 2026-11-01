# R/T/E完整结构消融

主设计为8个配置×同一批120个观测输入，共960次独立配置运行。
R是RNJ结构指导（少量种子、RNJ/NJ几何候选和RNJ热启动），T是≤5直属块联合末端，E是一次最终结构退出。
及时清零固定开启；所有输出只比较自身路径和单点候选，不使用完整RNJ树备用择优。

| 配置 | RNJ结构指导R | 联合末端T | 最终退出E |
|---|---:|---:|---:|
| R0T0E0 | 0 | 0 | 0 |
| R1T0E0 | 1 | 0 | 0 |
| R0T1E0 | 0 | 1 | 0 |
| R0T0E1 | 0 | 0 | 1 |
| R1T1E0 | 1 | 1 | 0 |
| R1T0E1 | 1 | 0 | 1 |
| R0T1E1 | 0 | 1 | 1 |
| R1T1E1 | 1 | 1 | 1 |

R关闭时不接收RNJ支持集合或完整树热启动；只保留基础响应列、自身生成的支持集合，以及E允许的通用成对候选和局部并集。
标量f0、dense-QP几何标定与kernel共同固定，所以R0T0E0准确名称为“WZZT基座（公共校准固定）”，区别于此前无正则的纯WZZT。
全部组合使用λ=0.003f0、响应界[0,2]、0.24收缩和严格验证规则。原有方案不修改。

每配置/输入总预算14400秒（含公共前端记账），单步WZZT≤3600秒，固定结构LP≤300秒，后处理预留600秒。
所有配置采用相同前向阶段时限：≤7200秒且≤剩余结构预算一半；E关闭不会增加前向预算。
统一相对MIP gap=0.0001，单线程，8个worker。名义上限3840方法小时；实际成本按逐次记录报告。
各步达到gap后立即结束，不耗满时限。失败、未完成、超时及未决停止均保留分母。

先冻结输入/源码和协议，再运行并后分析：

```powershell
$env:PYTHONPATH='D:/0-github_workspace/Topo/rnj_wzzt_core'
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.run freeze outputs/full_ablation_20261102
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.run run outputs/full_ablation_20261102
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.analyze outputs/full_ablation_20261102
```

批次运行期间也可启动一次性后处理等待器；它只等待完成回执，然后评分、绘图并压缩已有记录，不新增优化求解：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.full_ablation.finish outputs/full_ablation_20261102
```

运行器可重入，跳过已有结果和错误回执；不覆盖失败记录。只有后分析打开真实拓扑和测试量测。
完整8配置支持全部6种模块加入顺序的边际比较与交互分解；这不改变单次算法内RNJ预选、联合末端和最终退出的执行顺序。
E的候选池也覆盖部分小块结构，因此T/E可能替代，不预设各模块的收益为正。

计划输出包括F1、完整恢复率、真实总耗时、各网络和噪声条件结果、12个条件模块增量、6种加入顺序、平均边际贡献及交互项。
这批120输入已用于历史研究，本轮固定参数和协议，不宣称新盲测。
