# 单独采购无功能否称为配电灵活性服务

核查日期：2026-11-14。范围限于术语、独立无功采购、费用结构和逆变器可用容量。采用5项官方公开资料的相关正文；未开展全面综述、设备询价或新数值实验。已按 `research-literature-review` 与 `hku-library` 流程先复用本项目资源依据，再查公开正文；所需证据均可公开读取，未使用HKU订阅。论文中的称谓和模型取舍属于下文明确标出的研究建议。

**结论：可以把单独采购Q称为配电灵活性服务中的无功服务，建议具体写作“reactive-power flexibility procurement”。没有为满足这个名称而同时购买P的必要。** 是否能参加某一现有市场，要看该市场采购的具体产品；有些常规配网招标采购的是kW级有功调节，不能直接把kvar合同视为同一个产品。

## 核实的来源与阅读位置

| 来源、版本 | 本次读到的位置 | 支持的结论与适用范围 |
|---|---|---|
| [UK Power Networks / National Grid ESO，Power Potential Market Procedures v7](https://d1oyzg0jo3ox9g.cloudfront.net/app/uploads/2023/10/Market_Procedures_v7_October-_2020.pdf)，2020-10-13重发，30页 | 正文第5–10页，第16–20页，第22页；重点第9页流程、第16–17页§3.3.2.1、第20页§3.3.2.3。正文页码与PDF页码一致。 | 有功和无功分别申报；无功报价包含可用性价与调用价，并按报价值结算。无功可用量根据设备P–Q能力图和有功运行预测确定。第22页另列有功市场，因此提供P预测并不等于同时购买P调节。文件是试点规则，不能写成所有地区现行统一规则。 |
| [ENA Innovation Portal / NGED（原WPD），Q-Flex，NIA_WPD_072](https://smarter.energynetworks.org/projects/nia_wpd_072/)，项目期2022-09至2023-08，完成报告入口标记2024-07-30 | 官方网页“Summary”“Benefits”“Learnings—Outcomes”“Lessons Learnt”WP1–WP4正文；未下载其全部附件 | 项目明确研究无功灵活性市场、无功资源采购及其可用性支付。成果强调节点位置有效性、接入控制和参与者机会成本。主要研究电压等级为33、66、132kV，不据此声称本研究低压末端服务已普遍商业化。 |
| [SP Electricity North West，Flexibility services](https://www.enwl.co.uk/future-energy/flexibility-services/)，2026-11-14访问的现行网页 | “Introduction to flexibility services”“How to get involved”两节正文 | 该入口的产品要求调节发电/用电量，列出至少10kW的参与容量，可自行或聚合提供。这给出具体市场采用有功产品的实例；10kW不能换称10kvar，也不能证明其所有未来产品都排除Q。 |
| [Narang, Mahmud, Ingram, Hoke，An Overview of Issues Related to IEEE Std 1547-2018 Requirements Regarding Voltage and Reactive Power Control](https://docs.nlr.gov/docs/fy21osti/77156.pdf)，NREL/TP-5D00-77156，2021-09 | 正文§2.4，第7–9页；§3，第10–11页及表2；§4第15–16页。PDF对应第15–19、23–24页。 | 支持逆变器吸收/注入Q、恒Q控制，以及Q优先时可能需要牺牲P输出。能力要求与运营商是否启用分开。这里依据NREL对标准的技术解读，未将其表述为已审查IEEE标准全部条款。 |
| [EPRI OpenDSS，Inverter Capability Curve](https://opendss.epri.com/InverterCapabilityCurve.html)，网页版权标记2024，2026-11-14访问 | “The following properties…”下的`kVA`、`kvarMax`、`kvarMaxAbs`、`%CutIn`、`%CutOut`、`%PminNoVars`、`%PminkvarMax`及`varFollowInverter`说明 | 能力取决于视在容量、正负无功上限及开关机/低有功条件。软件默认值不是现场设备额定值；不能用一个默认开关状态假定所有PV夜间均可提供Q。 |

另查过2018年Power Potential合同纲要，但其首页标明工作草案，本文采用更新的2020年市场程序。公开检索还发现OneNet和EUniversal的Q灵活性材料；相关PDF在本次工具中未完整打开，未将搜索摘要当作正文证据。无需为本窄问题继续下载大批文献。

## 怎样使用“Distribution Flexibility Services”

建议论文先给出自己的工作定义：运营商从配电侧资源采购可控的有功和/或无功调整。本研究模型允许两类动作，本批成本目标下主要选择无功。随后把实际使用的产品写清为**用于增强拓扑辨识信息的无功灵活性服务**。

官方项目证明无功可以作为独立服务采购，支持这种资源建模。现有来源主要以电压、拥塞和损耗为服务用途；**为拓扑辨识专门购买激励，是本研究提出的用途**，不应写成这些市场已在购买拓扑辨识服务。也不宜把主要靠SVG动作的结果笼统写成居民移峰或需求响应收益。

不需要增设“必须采购一定有功”的约束来使名称成立。若研究还要比较P和Q的作用，可事先安排P-only、Q-only和P+Q对照；这个选择应服务于研究问题，不能因当前解只买Q而事后加入凑量约束。

## 费用：调用量与可用容量分开

Power Potential第20页把无功支付分为两部分：接受的可用容量和实际无功响应。第17页明确允许零报价，但这只说明合同可以提交零价，不证明设备预留没有成本。按其费用结构，可使用下面的研究简化式：

\[
C=\sum_{i,h}\left[f_{i,h}K_{i,h}+c^+_{i,h}(u_{Q,i,h})_++c^-_{i,h}(-u_{Q,i,h})_+\right]\Delta t.
\]

其中，\(K\)是合同约定的可用容量，\(f\)以元/(kvar·h)计；\(c^+,c^-\)分别是两方向的调用价，以元/kvarh计；\(u_Q\)是相对基线的调整，\(\Delta t\)以小时计。**合同容量如何定义须先约定**：Power Potential按能力图的最大无功范围计算某些可用性支付，不能不加说明就把双向±5kvar合同一律按5或10kvar计预留费。

本项目现用0.05元/kvarh相当于50元/Mvarh，是示例调用价，未由上述资料校准。当前预留费为零，报价应继续称“激活费用”；新增设备投资、控制改造、容量预留和辅助用电未被这笔费用全部覆盖。无需为了让总价显得较高而任意补固定费。若同一批设备和预留容量对所有方案固定，加入相同容量费会改变总费用，不会自行改变调用方案的排序。

可追溯的原文数值：Power Potential第19页表3用5Mvar、可用性价£1.50或£1/(Mvar·h)、调用价£10或£5/Mvarh演示位置有效性；同页脚注明这些价格任意选取，**不是预期投标价格**。第8/10/22页的£150/MWh是有功调试测试补偿，也不能拿来校准无功价格。上述数值仅说明计费结构和单位，不作为本低压合同价格依据。

## 用户侧逆变器与独立补偿器的边界

在设备三相平衡、用视在容量近似电流上限时，逆变器的简化域可写为

\[
(p_i+u_{P,i})^2+(q_i+u_{Q,i})^2\le S_i^2,
\qquad -Q_i^{\rm inj}\le q_i+u_{Q,i}\le Q_i^{\rm abs}.
\]

这表达同一逆变器的P和Q共享容量；边界是凸二次约束，可保留QCQP形式。沿用本项目净吸收为正的符号，\(Q^{\rm abs}\)与\(Q^{\rm inj}\)分别表示吸收和注入上限的非负数值。电压、电流、温度、控制模式和接入协议可能进一步收紧该域。[EPRI能力参数](https://opendss.epri.com/InverterCapabilityCurve.html)

单买Q时可以令\(u_P=0\)，但Q上限随实际设备P运行点变化。例如10kVA逆变器在8kW、基线Q=0时，仅由容量圆得到|Q|≤6kvar；这是\(\sqrt{10^2-8^2}\)的算术示例，不是原文设备报价或实测配置。若目标Q超过剩余容量，才需要降低P、扩大设备容量或改用独立补偿器。这里降低P是物理容量的需要，与是否能称为灵活性无关。[NREL，第10–11页](https://docs.nlr.gov/docs/fy21osti/77156.pdf)

设备P不是节点总净P；PV登记kW也不是逆变器kVA。没有分项运行点、视在额定值及控制权限，就不能从源负荷净值算出可靠的逆变器Q余量。夜间和低有功时是否有Q能力，应以实际设备控制、供电和合同可用性为准。[EPRI开关机及低功率参数](https://opendss.epri.com/InverterCapabilityCurve.html)

本批若继续解释为独立SVG/可控补偿器，则独立双向Q盒约束仍是合理的简化，且无需购买P调节。装置自身有功损耗和网络损耗应另行核算；该解释依赖声明的合成设备部署，不能据FeederBW的PV登记推断现场已经装有这些补偿器。
