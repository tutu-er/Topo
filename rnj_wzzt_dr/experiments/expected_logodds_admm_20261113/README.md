# 期望对数后验赔率DR与ADMM

本目录只接冻结辨识结果，不改变辨识器、历史、尺度、资源或量测规则。主目标是使指定竞争者的期望对数后验赔率相对同一个无DR新日增加至少`log(2)`，并最小化原合同费用。它表示平均证据增强，不是90%后验达标概率保证。

本轮开发测试已完成，详见[完整结果](results/REPORT.md)、[逐起点记录](results/attempts.csv)和[核验记录](results/VALIDATION.json)。原整数设备版ADMM和直接QCQP均为0/4合格报价；明确的凸设备子域版为2/4，两份报价来自同一个Flynn输入的相反条件方向。全部实际ADMM起点均到时间上限，保留了合格中间点，尚无迭代收敛或全局最优证明。

Flynn真目标的声明模型平均后验从95.0930%升至95.9602%；独立AC模拟从94.5683%升至94.7988%，正确唯一输出从29/32升至30/32，完整恢复仍为30/32。AC平均证据增量的95%均值区间包含0；Paper两个目标无合格报价。13项解析、控制器回放和后验核验测试通过。原始数据分别保存在忽略的v1、v2输出目录。

## 精确二次目标

条件于冻结目标t为生成模型，`Y=H_t P+e`、`E[e|P]=0`，历史共同sigma²固定。对每个竞争者j：

\[
L_j=L_j^h+\frac{\|(H_t-H_j)P\|^2+2e^\top(H_t-H_j)P}{2\sigma^2}.
\]

规划功率均值是`p0+B u`，背景误差法则固定。因此同样加入一个新日时，DR的期望增量为

\[
G_j(u)=\mathbb E L_j(u)-\mathbb E L_j(0)
=\frac{\|D_j(p_0+Bu)\|^2-\|D_jp_0\|^2}{2\sigma^2}.
\]

把各设备贡献写为`F_jr u_r`，基准为b_j，则`G_j=||b_j+sum_r F_jr u_r||²-||b_j||²`。本批约束为`G_j>=log(2)`，数值求解内部另留1e-5余量，验收仍核原log(2)。常数缩放仅改善计算条件，不改变证据或费用规则。

## 两个块如何协调

给每个贡献设一个副本`Z_jr=F_jr u_r`。目标分成设备费用和球外约束：

\[
\min\sum_r C_r(u_r),\quad u_r\in U_r,\quad
Z_{jr}=F_{jr}u_r,\quad
\|b_j+\sum_r Z_{jr}\|\ge\sqrt{\|b_j\|^2+\gamma}.
\]

ADMM每轮先分别求各设备的费用加平方偏差，再更新协调副本与乘子：

\[
u_r^{k+1}\in\arg\min_{U_r}
C_r(u_r)+\frac\rho2\sum_j\|F_{jr}u_r-Z_{jr}^k+d_{jr}^k\|^2,
\]
\[
v_{jr}=F_{jr}u_r^{k+1}+d_{jr}^k,\quad
s_j=b_j+\sum_r v_{jr},\quad
\widehat s_j=\Pi_{\|s\|\ge r_j}(s_j),
\]
\[
Z_{jr}^{k+1}=v_{jr}+(\widehat s_j-s_j)/R_j,\quad
d_{jr}^{k+1}=d_{jr}^k+F_{jr}u_r^{k+1}-Z_{jr}^{k+1}.
\]

R_j是对该比较确有非零影响的资源数。最近球外点：已在球外则不变，否则沿原方向移到球面；零点指定确定方向。在固定总修正下，平均分到副本使平方偏差最小，因此该协调步骤是精确求解，不是切线近似。

## 凸性、初值和保证范围

若设备域凸，设备步骤是凸QP，另一步是非凸球外投影。**原小时控制器含截断和方向整数逻辑，所以本批实际设备步骤是小型MIQP，并设置求解时限；不能称为纯凸ADMM或精确设备更新。** 复用`dr/device_graph.py`，按资源切片，保留原控制域而不偷偷替换为连续放松。

该结构让非凸距离更新很便宜，但不自动保证迭代收敛或全局最优。解析小例同时保留成功和失败：对称两竞争者在零初值可停滞；某些乘子参数下，即便有可行初值，也可能只保留初值。原始可行性以重放后的设备交付和完整H预测复核，不能只看副本投影或残差。

研究参考：[Boyd等ADMM原文第9节](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)。此处拆分与验证针对本项目目标，不据该综述宣称非凸整数情形有普遍收敛定理。

## 本批固定范围

- 输入：已有Flynn `main__input054__101`、Paper `main__input121__101`；每输入固定历史前两个目标，合计4个条件目标。两例曾被研究，不能称新输入盲测。
- 竞争者：复用历史权重确定的主要集合，分别含1或2个；不使用实际未来或真线路。完整原库用于最终后验评分，理论增量只对列入约束的竞争者成立。
- ADMM：零请求及固定1/4幅轮换两个起点，各30秒、至多300轮，包含模型构造及初值成本；rho从10开始，每10轮按残差调整。每设备原生求解最多0.05秒，所有求解器单线程。
- 直接QCQP：同一原设备域、同一增量要求，从零开始60秒，不借ADMM的报价作初值。分别报告可行点、原生界、gap和停机原因。
- 规划冻结后：2048条新的声明模型配对路径，报告期望对数赔率增量、平均目标后验、80/90/95/99%达标频率及费用；再对可执行报价做32条原交流潮流配对新日。无报价的策略留空。
- 原始数据、数值参数、日志和交付轨迹留在忽略的`outputs/expected_logodds_admm_20261113_v1/`，公开只保留必要汇总和归一化费用。源代码、输入引用、预算与随机流写入事前协议。

## 入口

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
& $py -X utf8 -B -m unittest rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.test_admm -v
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.run --output outputs/a_new_expected_logodds_run
```

只使用新输出目录，保留所有失败、超时和起点结果。本轮已有完成标记，不重跑已有批次；上面的命令供新的、事前固定范围的实验使用。

## 第二个原型：明确的凸设备子域

第一批已完成：4个条件目标，原整数设备ADMM均未取得合格报价，同预算直接QCQP也均未取得报价。每个30秒起点只完成约十余轮原设备更新；这批失败保留在`outputs/expected_logodds_admm_20261113_v1/`。

第二批保持相同输入、目标、竞争者、增量和资源数值，只测试一个声明清楚的搜索子域：小时原始计划直接满足全部原功率、能量、未来可行区间、重置和截止要求，不触发名义投影截断；电池充放电方向固定为两小时交替。非电池保留允许的正负方向。它是原设备域的子集，可能排除更好的原域方案，任何子域下界都不能充当原QCQP下界。

这样每个设备更新成为真正的凸QP，球外协调步骤不变。另在零点用二次式的全局线性下界求一次凸QP初值；即使零点未满足增强量，该线性问题若可行，也能给出原二次条件可行点。先保存初值费用及生成时间，再分别从零点、该初值运行ADMM。若最后只是保留初值，就明确标为`input_supporting_plane_seed`，不称为ADMM找到的新方案。

第二批每目标总预算60秒，含至多10秒的初值QP，余额分给两个起点。原域直接QCQP基线复用第一批的同目标60秒结果，不重复求解。新模型/物理确认仍须在全部报价冻结后开启；第二批新流与第一批预留流分开。

```powershell
& $py -X utf8 -B -m unittest rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.test_convex -v
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.convex_run --output outputs/a_new_convex_subset_run
```

## 报价冻结后的确认与汇总

先完成全部目标的规划并冻结策略，再运行`confirm`。它分别保存2048条条件模型路径和每输入32条AC新日，比较无DR、同费轮换和可执行的ADMM报价。失败报价保留为未执行；相反目标共用的实际新日不重复计为独立样本。`audit_model`复算已有路径，不新开测试随机流；`report`独立重算全部实际后验和结构错漏，再生成公开汇总。

```powershell
& $py -X utf8 -B -m unittest rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.test_admm rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.test_convex rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.test_confirm -v
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.confirm --help
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.audit_model --run outputs/a_completed_convex_subset_run
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.expected_logodds_admm_20261113.report --run outputs/a_completed_convex_subset_run --public outputs/a_new_public_summary
```

本轮`results/RECEIPT.json`保存公开文件哈希；`VALIDATION.json`记录完整预算、初值来源、超时和独立复算范围。模型与AC确认使用规划冻结后的新路径，输入本身仍是已有开发输入。
