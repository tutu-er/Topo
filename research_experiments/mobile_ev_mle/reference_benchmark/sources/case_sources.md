# 本轮论文与标准算例来源核查

核查日期：2026-10-18。范围是算例来源、改造边界和当前程序兼容性；未运行辨识、EV规划或效果试验。使用 research-literature-review；按 hku-library 流程先复用本地文本和公开原始资料，未触发订阅访问，也未新下载论文PDF。

## 结论

本轮适合分为三类，分别报告，避免将来源不同的数据混为“真实完整馈线”：

1. **标准参数网络**：完整 MATPOWER case33bw；保留全部32个负荷母线和32条闭合线路，另声明生成的时间变化、DER和计量噪声。
2. **更真实的低压网络与负荷联合数据**：优先核验 SINTEF/CINELDI 2024 年四个匿名230V网络，或使用 SimBench 小型LV1及其完整年度曲线；源有功、无功及原始资料的生成方式逐项登记。
3. **历史兼容组**：paper15、soumalas11、flynn16、pengwah18属于本项目自建论文风格网络，应继续以合成组命名，而非原论文参数复现。

IEEE European LV 是与 Pengwah 原文关联最直接的标准低压例，但原始三相、互耦、变压器和仅一天负荷曲线超出目前平衡末端树的直接接口。可作为第二阶段的明确分相/正序派生测试，或使用独立三相AC仿真做模型失配审查；不能直接把平衡派生结果称为原始三相完整馈线测试。

## 1. 四个旧“论文风格”例到底是什么

实际核查源码：

- `C:/Users/23761/.codex/worktrees/36a1/Topo/rnj_wzzt_core/rnj_wzzt/data/paper_style_case_bank.py`
- `C:/Users/23761/.codex/worktrees/36a1/Topo/rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py`
- `C:/Users/23761/.codex/worktrees/36a1/Topo/terminal_case33/data/paper_style_case_bank.py`

下表由上述构造器只读加载统计，无AC或优化调用。

|键|母线/边/计量末端|根度数|静态P/Q总量(kW/kvar)|基准|线路参数变化|
|---|---:|---:|---:|---|---|
|paper15|22/21/15|3|130.5/44.9|0.4kV、0.2MVA|R/X同量级的合成库，全部阻抗乘5|
|soumalas11|16/15/11|3|76.4/25.7|同上|同合成库，乘5.2|
|flynn16|23/22/16|2|119.3/40.4|同上|同合成库，乘4.8|
|pengwah18|26/25/18|4|139.9/48.8|同上|同合成库，乘4.5|

其隐藏内节点全部零注入，所有负荷都在计量末端；没有内置现场P/Q/V历史。所谓“unbalanced”描述支路/分组不均，并不是三相不平衡潮流。主干 `0.42+j0.34 Ω/km`、服务线 `0.55+j0.38 Ω/km` 是为数值辨识构造的库，代码明言先前较小X导致X灵敏度偏弱，所以提升X/R并整体乘倍数。这些额定电压与线长不能一起用于声称现场电缆的真实参数。

资源分配构造器的 `profile_source_candidate=NREL/NSRDB/WIND Toolkit` 只是候选来源标签，不证明已经读取或采用这些公开曲线。

### 与真正原论文的区别

本次阅读的是已有完整PDF抽取正文缓存，不是重新获得或视觉复核原PDF；书目信息另与官方机构/DOI页面交叉核对。实际正文范围：

- Soumalas 2017：本地 `docs/paper_notes/soumalas_2017.txt`，重点IV节、Table I/II及Fig.4附近，第5页。原文7个随机12–26节点网络，线长30m或28–32m；举例主干 `0.284+j0.083 Ω/km`，服务线 `1.380+j0.082 Ω/km`。它们与上述11/15末端自建库明显不同。原文使用同步P/Q/V，假定调压后的近恒定根电压，并利用导线类型/线长信息。DOI：[10.1109/PTC.2017.7981061](https://doi.org/10.1109/PTC.2017.7981061)。本文没有核实可下载原始拓扑/负荷表。
- Flynn 2023：本地 `docs/paper_notes/flynn_2023.txt`，VI–VII节，期刊3855–3858页附近。随机20–30节点网络及50–80节点扩展；另有澳大利亚Victoria DNSP四台三相变压器下的12个分相网络，每网14–29客户、约一个月10000记录。随机例线路取IEEE European LV阻抗范围，负荷增量独立正态，根电压AR模型。现有flynn16不是作者原网；作者现场数据本次未取得。DOI：[10.1109/TSG.2023.3239650](https://doi.org/10.1109/TSG.2023.3239650)。
- Pengwah 2024：本地 `docs/paper_notes/pengwah_2024.txt`，IV-A与IV-G，期刊996–1000页。随机10–50节点、每规模100个网络，P独立均匀(均值1kW、±0.5kW)，PF另抽；标准例用IEEE European LV55客户，A/B/C分别21/19/15，70%智能表，其余只有较慢能量表。原文取PES的一天P曲线，重复同一天P并重采PF增加样本。该操作可扩充电气情景，不能提供独立多天历史的统计分母。本轮若用完整P/Q/V表计应声明信息条件强于其70%混合表计试验。[Monash官方摘要](https://research.monash.edu/en/publications/topology-identification-of-distribution-networks-with-partial-sma/)与[DOI](https://doi.org/10.1109/TPWRD.2024.3354292)。

## 2. 完整case33bw：可以立即使用，但需保留内点负荷

[MATPOWER官方源码](https://matpower.org/docs/ref/matpower6.0/case33bw.html)正文及全数据表已读。原始33母线、12.66kV；32条闭合边、5条常开联络边；总负荷3715kW/2300kvar。线路表单位Ω、负荷表kW/kvar。官方基准100MVA，项目本地原始适配器采用10MVA；Ω和kW保持一致时只是重新标幺，必须写明本轮基准。

本地完整表：`terminal_case33/data/case33bw_raw.py`。此前 `ev_physical_cases.case33_local_standard()` 只保留5条线路和4/19/23三负荷，清零2/3内点负荷并删除其他母线，是局部派生例，不能充当本轮完整33节点算例。

完整33网的大多数负荷位于内部节点。当前候选树接口要求全部有量测的负荷在叶端，适配选项：

- 最准确：允许“可观测内部注入节点”，保留原物理节点、线路和量测关联。
- 最小兼容：将每个内点量测作为同电压的虚拟零阻抗叶，计算和真值评价时收缩虚拟边；必须支持零边，不能给虚拟叶添加任意正阻抗制造可辨识性。
- 将内点负荷搬到新的有正阻抗末端，会改变电气模型，应称“case33参数派生末端化网”，与完整原网结果分表。

单台约6kW EV仅为单个标准负荷的约1.4%–13.3%，总网负荷的0.16%。此尺度可产生“电气扰动很弱”的真实失败，不能为保证优势而无记录放大EV。可事先另列单车11kW/22kW接入能力或多车条件，各方案保持相同能力与能量任务；MV接入须解释为经变压器折算的LV充电站，不将车辆直接接12.66kV。

## 3. 推荐真实/标准低压扩展

### 首选：CINELDI四个230V匿名网

[SINTEF/Zenodo官方记录14528192](https://zenodo.org/records/14528192)，v1，2024-12-20，官方说明正文已读；未下载ZIP。8.5MB `dataset.zip`，四网名含39/56/50/80，分别半城市/乡村；含MATPOWER母线/支路、支路附加属性和baseMVA，所有负荷节点的小时P/Q表。它比跨国住宅曲线映射到虚拟网更直接，但P/Q是否直接实测、建模或由PF生成，必须读ZIP内说明/元数据后定论。本次只确认有P/Q文件，未将Q称为已核实实测。

适合优先从39网做端到端开发，再处理其余网。保持小时分辨率；如做15分钟控制，不能把4个复制/插值点视为4份独立真实背景或历史观测。

### 标准小型LV：SimBench LV1

[SimBench官方数据入口](https://simbench.de/en/download/datasets/)与[官方软件文档](https://simbench.readthedocs.io/en/stable/networks.html)已读；官方说明有全年负荷、发电与储能曲线。建议 `1-LV-rural1--0-sw`，0.4kV、13供电点、160kVA变压器；选择它是为了保持完整网络且便于现有MILP规模，后续可加 `1-LV-semiurb4--0-sw` 39供电点。上述规模见 [SimBench作者论文的Table1](https://simbench.de/wp-content/uploads/2025/11/meinecke_et_al-simbench_paper.pdf)（本次只读可提取的该表和对应数据说明，未阅读全文）。

年度曲线适合严格时间切分、预测误差和独立日期评价；它是经构造的公开基准，不等于相同馈线现场全年PQV。具体P/Q列、时间区、profile映射及来源需在实际导入时留哈希与字段核验。原有PV可作为已知量测净负荷的一部分，不要把其功率再添加一次。

### 与原论文直接关联：IEEE European LV

Pengwah IV-G确认55客户、分相和一天P曲线。EPRI官方[Examples, Docs and Test Cases](https://opendss.epri.com/ExamplesDocsandTestCases.html)确认OpenDSS分发含IEEE测试馈线；IEEE资源入口本次公共工具不可达，未取得原始欧洲LV文件，不能声称已经核验其逐条线路/负荷表。

本轮若时间有限，可先交付SimBench完整平衡LV测试及明确来源的旧paper-style兼容组，把EuropeanLV原始三相作为已声明未完成范围。若使用分相约化例，必须逐项保存相别、收缩映射、互耦/零线处理、保留支路、单位和与三相AC的电压误差。

### 已有工业数据备选：Norwegian 2023 v2.3

[Zenodo官方10361330](https://zenodo.org/records/10361330)说明正文及文件清单已读。原实网匿名/简化，两条径向馈线、45客户、2019-03-01至2022-03-16小时用电需求；44.5MB。本地C目录目前仅 `SOURCE.txt`，没有bus/branch/客户曲线文件，故旧适配器不是即用数据。

旧适配 `norwegian_industrial.py` 把多电压设备表示为公共标幺平衡等值，保留理想零阻抗变压器；`canonicalize_norwegian_terminal_radial` 会给零终端边补 `0.0005+j0.0004pu` 服务阻抗。此补线是合成改造，不能称原数据参数。推荐收缩不可分设备层级，保留聚合计量映射，或将补服务阻抗单独作为派生敏感性组。此集没有已核实Q/V，V仍需AC生成、Q仍需明确模型。

## 4. 源项与当前接口的关键不兼容

`mobile_ev_probing/ac_general_v9.py:RadialCandidate.from_supports` 总会加一个全末端共同上游边，源固定其下游node1。上述四个paper-style根已直接分叉，原网没有这一共同干线。仅从clade导入并令共同边为正，会额外增加一段上游线路；若放源也构成新改造，必须登记。

保持物理来源的处理：

1. 有公开变压器阻抗/容量时，可将其作为显式已知或待估上游支路，根放在变压器一次侧或受控边界，源接口选有物理意义的固定母线；每个竞争解释使用同一接口，不能随竞争拓扑自由换点。
2. 若根为已知二次侧电压，根上游变化通过已测根电压进入，不再另加可吸收任意电压误差的自由源。
3. 末端PV等已包含在表计净P/Q中的DER，只影响总输入，不另作为未计量源重复添加。未计量源是额外实验条件，P/Q均应建模，容量/位置/时序来源单列。
4. 功率正号统一用消耗为正：`P_net=P_demand-P_PV+P_EV`；`Q_net=Q_demand-Q_DER+Q_EV`。EV若单位功率因数设Q_EV=0；若提供无功服务须有视在功率能力并公平施加到所有对照。

## 5. 本轮建议的最小充分矩阵与边界

建议优先四项：**完整case33、SimBench LV1、CINELDI 39网、旧paper-style四网兼容组**。EuropeanLV为下一阶段三相核验目标，不先把它的原一天曲线重复成多天后声称独立校准。

每项至少区分“原网络/原曲线”、“固定资产加测得多天P形状的半合成”、“加入声明DER后的模拟”。所有新增Q、源、预测及仪表噪声单独保留生成记录。网络资产和新增源的规模先固定，再看算法表现；不能以评价结果调整线路下界、负荷倍率或EV额定功率。

数据切分须以连续日期/用户群为单位，先冻结开发、校准、评价窗口，再提取历史和未来。尺度和预测只由当前历史或更早开发数据估计。小时数据不伪装15分钟实测，论文的一天曲线重复不增加独立历史数，缺少真实Q的模型不写成实测Q。

当前核查没有新的辨识性能结论。全文阅读范围、数据下载缺口与模型变换在上述各项明确列出。

## 6. 随后实际实现的快速标准对照

主代理选择本轮先接入完整case33、CIGRE居民LV子网以及paper11/paper15的去倍率兼容组。上述SimBench/CINELDI/EuropeanLV仍是来源建议，不写成已经导入并完成测试。

物理导出实现于 `research_experiments/mobile_ev_mle/reference_benchmark/networks.py`，独立测试15项通过，仅涉及数据/结构/来源核查，无AC拟合。统一导出 `buses[{bus_id,pd_kw,qd_kvar}]`、`branches[{from_bus,to_bus,r_ohm,x_ohm}]`、`root_bus/base_kv/base_mva/base_kw/observation_bus_ids/source_bus` 及来源/物理内容哈希；内点负荷保留。编码代理另行处理精确零阻抗虚拟量测叶。

完整case33另取 [MATPOWER7.1官方源文件](https://raw.githubusercontent.com/MATPOWER/matpower/7.1/data/case33bw.m)，4866字节，SHA-256 `b40831eeb444669ae876e2996f0dda9f05cd83e81b314b8dfca51e4890cca95d`，保存为本目录 `case33bw_matpower_7.1.m`。逐条核对33个母线P/Q和32条闭合线原Ω表，全部与本地适配一致；未执行MATLAB代码。

CIGRE源来自 [pandapower v3.2.1 官方模块](https://raw.githubusercontent.com/e2nIEE/pandapower/v3.2.1/pandapower/networks/cigre_networks.py)，公开源码29516字节，SHA-256 `04817b6bdb11074310fc1e59aa91a27bf5b335a3c1236cb56b655fc73a6a1480`；保存为本目录 `cigre_networks_pandapower_v3.2.1.py`。实际阅读和AST核验 `create_cigre_network_lv` 的居民母线、17条居民线路、6个居民负荷、UG1/UG3库和R0–R1变压器数据。未导入或执行该下载模块、未安装pandapower。

CIGRE导出完整居民部分，共19母线、18支路、6有负荷量测点；R1保留190kW/62.449980kvar内点负荷，总负荷383.8kW。原R0–R1变压器500kVA、20/0.4kV、vkr=1%、vk=4.123106%、空载损耗和电流为零、零分接档。把理想变比和统一相移折入LV根参考后保留低压侧串联 `R=0.0032Ω`、`X≈0.0128Ω`。这一步保留平衡正序P/Q、幅值和损耗，不加入人为共同线路；工业/商业另外两网明确未保留，名称为“居民子网”。

源接口导出是明确的物理母线：case33为2，CIGRE为R1，paper11/15为原根1；实验冻结前可统一另行声明，竞争解释不得逐模型移动源接口。源在paper原根时其末端电压作用自然为零，这是接口物理性质，不能靠添线改变而不记录。
