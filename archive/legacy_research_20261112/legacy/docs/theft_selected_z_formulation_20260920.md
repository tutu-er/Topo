# 偷电定位：从已辨识的 z 集合中作整数选择

日期：2026-09-20。按用户本轮纠正，研究主线使用既有 wzzᵀ 辨识结果的离散选择表述，并沿用原设计的 R/X 联合重估。2026-09-19 的固定窗口 LP 枚举仍作为历史受限模型；其数值结果与校准不得改名为本主线的结果。本说明不修改原 MILP、原校准、原测试或任何历史输出。

## 1. 固定什么，选择什么

正常数据先产生已辨识字典

\[
\mathcal Z_{id}=\{z_1,\ldots,z_E\},\qquad
R=\sum_e r_ez_ez_e^\top,\quad X=\sum_e x_ez_ez_e^\top.
\]

偷电辨识时固定字典、终端次序及其层级包含关系，不新增任意二元支撑，不重新辨识拓扑。原设计仍允许 r/x 在给定范围内联合重估；“固定 z”本身并不意味着固定线路权重。

对于当前辨识树，每个非根候选节点 h 均对应一个已有支撑 z_h（入边的下游终端集合）。其标签可能是实测终端编号，也可能是约简区域编号。选择变量为

\[
s_{ht}\in\{0,1\},\qquad \sum_hs_{ht}\le1.
\]

无源允许 s=0；若某时刻的额外负荷已作为可信输入确认存在，才使用等号。幅值为零时不报告位置。原始逐时 MILP 允许位置和活动状态随时间变化，不新增“整个窗口位置固定”的假设。

## 2. 选一个 z，需要激活其祖先链

定义完全由已辨识字典确定的常数矩阵

\[
A_{eh}=\mathbf1\{\operatorname{supp}(z_h)\subseteq\operatorname{supp}(z_e)\},
\qquad b_{et}=\sum_h A_{eh}s_{ht}.
\]

由于已有支撑构成合法的层级树，A 的第 h 列正是根节点至候选 h 的路径。每时刻至多选择一个 h 时，b 也是二元量。

**s 选的是已有 z 对应的接入区域；b 选的是该区域上游实际增加潮流的线路链。** 不能让每条线路的 b 独立选择，也不能只给被选中的单个 z 添加一项而遗漏祖先线路。

例如已有支撑 {A,B,C}、{A,B}、{A}，在 {A} 对应节点增加负荷会同时影响这三条入边；在 {A,B} 对应节点增加负荷则只影响前两条。此处是模型中的候选接入节点，并非认为集合中每个终端各自增加一份相同负荷。

## 3. 同一个 wzzᵀ 模型的扩展列

把单个未计量源作为额外一列，定义

\[
\widetilde z_{e,t}=\begin{bmatrix}z_e\\b_{et}\end{bmatrix},\qquad
\widetilde R_t=\sum_e r_e\widetilde z_{e,t}\widetilde z_{e,t}^\top,
\quad \widetilde X_t=\sum_e x_e\widetilde z_{e,t}\widetilde z_{e,t}^\top.
\]

原观测块 R_OO/X_OO 的支撑不变，新增源与观测端的耦合列由选中的祖先链给出：

\[
R_{Oh,t}=\sum_e r_e b_{et}z_e,\qquad
X_{Oh,t}=\sum_e x_e b_{et}z_e.
\]

上式下标 h 表示这一时刻选择的单源列；s=0 时该列为零。额外源自身的电压不参与当前观测拟合，也不由此保证恢复真实物理母线。

固定总分表与线损估计得到的幅值后，终端平方电压降预测为

\[
\mu_t=\sum_e z_e\left[r_ez_e^\top p_t+x_ez_e^\top q_t
+\widehat a_t r_e b_{et}+\widehat q_t^a x_e b_{et}\right].
\]

因此只剩二元与有界连续变量乘积。引入 v^r_et=r_e b_et、v^x_et=x_e b_et，并作原设计的精确 big-M 线性化；L1 数据损失得到 **MILP**。这里的 L1 是绝对值损失，不是 LP 求解器，也不同于名为 L1 的线损估计器。

如果另行固定 r/x 和幅值，则所有候选预测都是常数，单源情况下可以直接逐候选评分；这一版本不需要为每个候选求连续 LP，但它不是原设计要求的 R/X 联合重估版本。

## 4. 对当前代码的核对与执行入口

- `theft_wzzt/theft_wzzt/theft/identified_tree.py:to_theft_tree`：已有支撑形成节点和父子关系，每个非根候选对应一个已有 z。
- `theft_wzzt/theft_wzzt/theft/theft_model.py:TheftTree.incidence`：z 是已有支撑，c 是上述 A。c 从路径构造，与支撑包含关系一致。
- `theft_wzzt/theft_wzzt/theft/theft_model.py:fit_theft_milp`：s 为整数变量，b=c*s，R/X 可调整，已有精确乘积线性化和直接校验。**此模型已实现，不需要再写一套仅换符号的求解器。**
- `theft_wzzt/detect_simple.py`：已有仅用测量通道的 H0/H1 MILP 入口；该入口使用自己匹配的校准。无校准时输出未校准状态。
- `theft_wzzt/paper_study_v2/regularized.py`、`detect_conservative.py`：历史的固定窗口位置枚举与连续 LP 参数重估模型。它也只枚举已有候选，没有将 z 连续化；但固定窗口位置、活动状态以及额外参数惩罚改变了模型范围，不能替代原逐时整数选择主线。

H0/H1 应使用相同的权重范围、测量通道、损失与误差处理。校准必须匹配实际推断流程；历史 V3 告警阈值及区域覆盖率不自动适用于此主线。修改求解表述不能改善数据中的可辨识性，也不能把约简区域保证变成唯一物理节点保证。

## 5. 本轮验证与未验证项

新增 `scripts/verify_theft_selected_z.py`，只读取三个既有辨识树缓存，核对：

1. 每个候选确实对应已有支撑，支撑向量未改变。
2. 对所有候选，集合包含矩阵 A 与原代码路径矩阵 c 完全一致。
3. 对原权重及另一组固定随机正权重，将扩展 wzzᵀ 矩阵乘以负荷所得的观测电压，与原 `predict` 直接计算逐候选逐时比较；包含零幅值时刻。

结果记录于 `theft_wzzt/outputs/theft_selected_z/formulation_audit_20260920.json`。这验证的是数学表达与实现的一致性，不是新的定位准确率实验。本轮没有改求解器，也没有重跑整套 MILP 校准或把 LP 实验结果迁移到 MILP。

复现命令（输出文件已存在时拒绝覆盖，复现时使用新的工作区内文件名）：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' 'D:\0-github_workspace\Topo\scripts\verify_theft_selected_z.py' --output 'D:\0-github_workspace\Topo\theft_wzzt\outputs\theft_selected_z\formulation_audit_20260920.json'
```
