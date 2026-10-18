# 更真实算例：负荷、源与单位审查

2026-10-18。审查范围为公开原始说明、官方实现和现有本地数据；没有运行新的规划或 AC 拟合，没有改动科学程序。结论用于选择下一批数据口径。已考虑 HKU 全文获取；本轮关键来源均有合法公开正文或官方代码，无需登录或订阅下载。

## 建议

主实验优先采用 **SimBench 原低压网及它原来分配的 P/Q 时序**；把原论文网络和 IEEE 33 节点保留为独立网络组，保持原额定电压、额定负荷和线路阻抗。OPSD 的建筑进口曲线继续作为实测形状敏感性数据，不再把 8 条曲线复制成 18/33 个“独立用户”。标准网静态参数加上外部时序是改造基准，报告须写出两者来源，不称为标准网实测时序。

单辆 EV 在 MW 级中压网作用较弱是合理的结果。不能把 IEEE 33 的电压改成 0.4 kV、把总负荷缩成若干 kW，却继续沿用原线路并称原标准算例。若扩大 EV 功率，应单列为车队或移动储能装置，不能仍解释为普通单车。

## 当前 OPSD 数据能支持什么

本地文件：`D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018/data_external/opsd_household_20200415/`。

2020-04-15 版官方 README、datapackage.json 与 CSV 表头/行数本次实际读取。CSV 共 153,810 行、71 列，UTC 范围 2014-12-11 17:45 至 2019-05-01 22:00；11 个 grid_import、3 个 grid_export、9 个 PV 通道，没有无功列。CSV 的原 SHA256 为 `eca7dee5ac6e53373cd0209bc05ba3383cd860bfa5d5e995f74fa78db8857c58`。许可证 CC BY 4.0。

官方说明采用累计 kWh，缺口曾按线性或前日数据填补，有对应插值标记。有效 15 分钟相邻差分除以 0.25 h 得平均 kW；差分两端的缺失、插值、复位及不连续时间必须一并筛除。官方数据不是同一已知馈线的同步 P/Q/V/线路量测。

`grid_import` 是单向电网进口，不是可正可负的净进口。住宅 3/4/6 另有 grid_export；有完整边界时，有符号净进口为进口减出口。若重建总用电，还需 PV、储能等完整账目；只有一条进口列时不能凭空反演总消费，再声称准确添加了同一户 PV。上一批把进口曲线作为合成馈线的非负负荷形状，这一口径可以复现，但没有检验真实反送电和原生 Q。

来源：[OPSD Household Data 2020-04-15](https://data.open-power-system-data.org/household_data/2020-04-15/)，本地 README 第 1–45 行、Sources/Field documentation 和完整 CSV 表头已读。没有新增数据下载。

## 官方可用的增强数据

|数据/网络|本次实际阅读范围|可用内容|适用边界与是否需下载|
|---|---|---|---|
|SimBench profiles|官方 profiles 文档全部接口说明；官方 2019 时序预印本 §2.1–2.3、§3.2|全年 15 min、单独 P/Q、负荷及发电时序，已有节点 profile 对应关系|本地未安装 simbench/pandapower，需下载并固定版本。时序由不同来源年份对齐到 2016，包含日期调整，不能称同一真实馈线同步量测|
|CIGRE LV|pandapower 官方 CIGRE 文档 LV 节和官方 create_cigre_network_lv 代码|44 bus、37 line、15 load、3 trafo 的静态基准；多个低压供电支路|官方该构造函数未提供对应时序。居民支路可作为低压主网，若只取居民支路要明确是子网，保留该支路原参数|
|IEEE 33 / Baran–Wu|MATPOWER 官方 case33bw.m 的负荷、线路、单位转换|33 bus、12.66 kV、额定静态 P/Q、逐边 ohm|原版没有时序和分布式 PV；增加负荷曲线/源属于显式改造。可作规模与弱注入试验，不能把不显著激励当实现错误|

SimBench 官方算法按 `net.load.profile` 绑定曲线，以额定列乘相对 profile；P 与 Q 后缀不同。其设计依据可用于算法论文的公开基准仿真，但这些证据不能替代现场验证，也不能自动保证任何目标期刊的特定要求。

来源与读取范围：

- [SimBench 官方 profiles 文档](https://simbench.readthedocs.io/en/stable/profiles.html)：get_absolute_profiles_from_relative_profiles、get_absolute_values、_pload/_qload 后缀及 profile 映射说明已读。
- [SimBench 官方数据入口](https://simbench.de/en/download/datasets/)：网格代码、CSV/Python 入口已读。可从 `1-LV-rural1--0-sw` 或相应无开关表示开始；最终采用的具体网应先核对节点、负荷和径向状态，不依名称猜规模。
- [Spalthoff 等，SimBench 时序预印本](https://simbench.de/wp-content/uploads/2025/11/time_series_preprint.pdf)：§2.1–2.3、§3.2 正文已读。公开时序含实测基础与构造部分，应分别标注；不是纯原始现场数据。
- [SimBench 许可证](https://github.com/e2nIEE/simbench/blob/develop/LICENSE)：数据库 ODbL 1.0、内容 DbCL 1.0、代码 BSD-3-Clause。保存版本、来源、哈希及许可，发布处理后数据库时同时核对相应许可要求。
- [pandapower 官方 CIGRE 文档](https://pandapower.readthedocs.io/en/stable/networks/cigre.html)：引用 CIGRE TB 575 (2014)；本次未取得 TB 575 正文，表格/参数主张依据官方实现，不假称原报告逐表核对。
- [pandapower 官方 CIGRE 实现](https://raw.githubusercontent.com/e2nIEE/pandapower/develop/pandapower/networks/cigre_networks.py)：create_cigre_network_lv 及其参数构造已定位。
- [MATPOWER case33bw](https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m)：原始 kW/kvar、ohm、12.66 kV 与 per-unit 转换代码已读。其引用为 Baran & Wu (1989), DOI 10.1109/61.25627；本次未重新阅读全文。

## 最小建模口径

### 1. 负荷保留标准额定量级

令 `p_i^0`、`q_i^0` 为标准网节点 i 的额定 kW/kvar；`a_i(t)`、`b_i(t)` 为对应 profile 的相对有功、无功值：

\[
P_i^L(t)=p_i^0 a_i(t),\qquad Q_i^L(t)=q_i^0 b_i(t).
\]

SimBench 原网直接采用它原来的额定列和 profile 绑定。P/Q 分开读取，避免把全部节点设为固定同一 Q/P，导致 R/X 难以分开估计。它已有的公共波动和 profile 复用应保留；报告不同 profile 数与跨节点相关性，不能以节点数量声称同数量的独立轨迹。

外部 profile 加到静态 IEEE/CIGRE 网时，先按节点的住宅/商业/工业类别分配，并事前冻结。若需要归一化，只用独立训练年或规划时已到达的历史，不能用该评价窗未来最大值/均值。单个住宅曲线放大为数百 kW 不会自然得到真实聚合负荷：它仍保留单户尖峰。主方案应采用标准原 profile 映射；实测扩展则用多个真实不同住户求和，记录住户成员、权重及是否跨节点复用。同一历史对应的多拓扑/多策略条件按日期成簇统计。

### 2. 源和 EV 在同一计量边界上记账

约定吸收功率为正。已计量末端的 AC 条件输入为：

\[
P_i=P_i^L-P_i^{PV}+P_i^{EV},\qquad
Q_i=Q_i^L-Q_i^{PV}+Q_i^{EV}.
\]

这些净 P/Q 到达后已经含 EV 和本边界内的 PV，AC 再加入 EV 会重复计数。仿真生成器按组成量计算一次净值，推断端只收到净表计值。原标准网已有的发电不能在扣除一次后再作为“未知源”扣一次。

未计量源另定义为固定物理接口 h 的 `(P_h^S(t),Q_h^S(t))`，历史与未来共享同一个 h，P/Q 都进 AC。不同候选不能选择各自最有利的虚拟注入点。最简实现先只加一个源；多个已计量 PV 可以包含在各末端净 P/Q 中。若设多处未计量源，则应独立说明模型扩展与可识别性成本，不能仍声称旧单源模型已覆盖。

PV 有功采用标准原容量乘 PV profile。无功先明确采用单位功率因数还是固定功率因数/Volt–VAR；固定功率因数与控制参数须各策略一致、在开发前固定，不能由真拓扑为每个样本挑参数。若采用 Volt–VAR，Q 依电压变化，AC/灵敏度必须对同一控制律求解和求导。

### 3. EV 无功和接入类型

首版若保持 Q_EV=0，应标为单位功率因数近似；这不是所有车辆的保证。对非单位功率因数，取已声明硬件或试验假设；独立无功优化仅用于明确支持该功能的双向变流器，并约束 `P_EV^2+Q_EV^2 <= S_rated^2`。不能从 AC 壁盒的功率规格推断车载充电机具有独立 Q 控制。

[ABB 官方 Terra AC 规格](https://new.abb.com/docs/librariesprovider54/publikationer/9akk107680a2257_rev-d_terra_ac_wallbox_brochure.pdf?sfvrsn=c4070c17_2) 本次读取技术表，支持单相至 7.4 kW、三相至 22 kW 的功率量级，未以该表证明 V2G 或 Q 控制。若现有求解器是平衡单相等值，单相 7.4 kW EV 无法用于检验相间不平衡，主实验宜采用三相平衡等值的 11/22 kW；随后另做三相外部校验。

### 4. 单位与电气量级核对

平衡三相等值统一使用总三相 S_base 和线电压 V_LL,base，`Z_base=V_LL,base^2/S_base`。若内部使用相电压，转换为 V_LL/√3，不能混用。网格原线路长度/R/X、变压器额定参数、供电电压和负荷额定值全部保存。转换之后，应在至少轻/中/重负荷和 PV 反送电点与独立标准 AC 实现逐点比对电压、根 P/Q、损耗；未通过前不跑大量方法优劣比较。

## 数据边界与验收

1. 冻结原数据/标准参数版本、文件哈希、逐节点 profile/容量/单位、缺失和插值规则，列出重复绑定。生成器持有真网和未来曲线；规划器不接触其对象。
2. 使用按时间划分的训练、开发、校准、评价块。同一天的多个网/源/策略配对属于一个簇。标准 profile 自身已做全年归一化属于公开基准给定规格，不能把这当作在线预测可用的未来统计；在线预测重新标准化时只用历史。
3. 时序预测至少支持前一日/同类日与递归滞后模型。若历史不足一天，日周期对照不可伪装为已测试。按真实到达时间提供过去数据；未来天气若采用，只能是明确的已发布预测，不能输入事后天气真值。
4. 源模式不能通过配置标签泄露给选择器或分层阈值。源预测来自过去可用计量/拟合序列；若无可用气象预测，按相同信息边界比较持久/日周期方案。
5. 输出 forecast P/Q/source 误差、真实运行电压/负载率/反送比例、每辆 EV 对本地/全网功率占比、动作是否合法、候选漏真及失败。保留弱注入和功率预测失真样本。
6. 能支持的首要结论是“公开标准网络、公开/有来源时序驱动下的可复现模拟表现”。要提升到实测台区可靠性，还缺真实线路表、净 P/Q/V 同步计量及可执行 EV 现场数据。本次不以公开 household 曲线替代这些信息。

## 下载状态

初次只读审查未新增下载。随后按本轮数据准备授权，实际取得以下最小公开输入，均未安装软件、没有认证阻塞：

1. SimBench 1.6.3 官方 PyPI 轮包的场景 0 `Load.csv`、`LoadProfile.csv`、`RESProfile.csv`、LICENSE、RECORD。采用 HTTPS Range 只取所需 ZIP 条目，合计约 23.5 MB 压缩网络量，未下载约 91 MB 整包。每条 CRC32、长度与官方 wheel RECORD SHA256 全部一致。文件及逐条哈希在 `data_external/simbench/acquisition.json`；整轮包的 PyPI SHA256 仅作为预期值保存，不能称已经整包校验。
2. [SINTEF/CINELDI 官方数据](https://zenodo.org/records/14528192)，DOI `10.5281/zenodo.14528192`，v1，CC BY 4.0。`dataset.zip` 为 8,539,525 bytes，官方 MD5 `f248e6cb206bf96c12920c088fdeddb8` 核验一致，SHA256 `e3babc3c8faeef086518a82b4912a7bd33c10a903ab5192125649391e72adf41`。来源原 JSON、条目清单与许可保存在 `data_external/sintef_cineldi_14528192/`。没有把数据重采样为更细间隔。

实际文件与 2019 预印本存在版本差异：SimBench 1.6.3 有 96 对 P/Q profiles、35,136 条 15 分钟记录；时间标签实际包含 2016 年德国夏令时缺口与重复。加载器显式以 Europe/Berlin 解释 CET/CEST 后转 UTC，保留每个物理间隔，不删除重复本地小时或插值。原文明确描述的普通负荷类别为 27 个 profile。**HS0..HS27 不是 household 的缩写**：本次继续核验官方 Load.csv，HS/HSexp 共 386 条绑定全部用于 EHV1、voltLvl=1，额定 pLoad 为 2.15943–1255.9 MW；不可移作住宅独立曲线。真正 H0-A/B/C/G/L 的 28,777 个绑定用于 LV、额定 1–12 kW，仅 5 种；H0-H 列存在但该 Load.csv 未引用它。事实与例子保存在 `audit/household_profile_origin.json`。32 母线并不要求 32 独立曲线；若采用官方复用方式就报告复用和秩，而不是把 EHV 的平滑聚合曲线当住户来凑数量。

SINTEF 官方记录正文及 ZIP 全部 CSV 表头和数值结构本次已读：4 个基于真实匿名化 230 V 径向网的参考数据，逐小时 8,760 点。39/56 半城镇网 Q/P 全部精确为 0.2030586606（功率因数 0.98），不是独立 Q 激励；50/80 农村网 Q/P 随时间变化。某母线包含多个真实客户列，不能把 pandas 自动改名的 `30.1` 误认为新母线；加载器按 `load_bus_extra` 的客户顺序校验，并对同一母线求和。39/50/56/80 网分别聚合为 28/21/44/32 个负荷母线；Base MVA 分别为 0.05、0.0344、0.05、0.015967，均不等于 1。39/56 静态 bus.Pd 全零；50/80 的静态 bus.Pd 合计分别为 0.023、0.02601，但静态列并非时序单位声明，不能用其数值接近程度定单位。官方公开说明没有明确分开负荷 CSV 的 MW/pu 单位，故保留 raw，**不根据电压效果猜单位进入该原网 AC**。可待获得原作者单位说明后再用；当前原网测试不受它阻塞。

`data_sources.py` 已实现并通过 13 个独立单元测试：P/Q 分别额定缩放、母线顺序、重复曲线明示、改变未来不影响过去、DST 间隔、同母线客户合并、哈希/单位声明以及原采样保留。`audit/data_inventory.json` 记录全部本地来源的统计，不作为规划器特征或评价调参输入。来源加载步骤没有 AC、拟合或规划调用。
