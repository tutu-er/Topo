# 支路下限检查与逐条收缩

按[锁定协议](PROTOCOL.md)比较旧RX75相对强度和实际欧姆阻抗下限。
每次只删除一条不达标支路，重新训练拟合，再检查至稳定；全部候选
处理后用原验证规则重新选择。保留零阈值原结果、误删、多遗漏、失败
和历史求解状态。

输入为当前56组及之前系统测试的400组101结果；不新造观测、不改变
噪声或线路真值、不进行新结构搜索或DR。

**全部456输入、10,240计划格已完成并通过独立核验，37项测试通过。**
具体下限建议、恢复增加和真实误删见[完整结果说明](RESULTS.md)。
当前8天拟合主设置建议先用全支路0.003Ω；不能将此数值直接推广到较短历史和弱线压力。

支路以它下游的观测节点集合 $C$ 表示。原模型为

$$Y=PR^T+QX^T+E,\qquad
R=\sum_C r_Cz_Cz_C^T,\quad X=\sum_C x_Cz_Cz_C^T.$$

这里 $r_C,x_C$ 是平方电压响应系数，物理线路欧姆值为它们的
$Z_{base}/2$ 倍。原固定支持QP允许系数为零，系数上界仍为2。

检查两种下限，数值不能混用：

- 物理下限：$|Z_C|=(Z_{base}/2)\sqrt{r_C^2+x_C^2}\ge z_{min}$，单位欧姆。
  本批当前两网 $Z_{base}=0.8\ \Omega$。这是阻抗幅值下限，不要求R和X各自超过同一正数。
- 旧RX75相对下限：$s_C=(a r_C+b x_C)/m\ge\tau$。
  $a=0.75/\mu_R,b=0.25/\mu_X$；$\mu_R,\mu_X$ 是训练段前置矩阵的正节点间距离均值，
  $m$ 是相应混合矩阵对角线的中位数。三者只用训练数据算一次，删线时不更新。

每个正阈值都从原候选重新开始，选最弱且不达标的一条支路删除，再按原训练平方损失
拟合剩余系数，重新检查，直到没有违规支路。这样得到的保留支路满足下限；删除顺序是
贪心规则，未求解带下限的全域结构最优问题。固定支持的连续QP证书单独保存。

`all`作用于每条线路；`non_singleton`只豁免末端单节点接入线，是旧规则的对照。
两者都检查全部用户共用的公共上游项。豁免末端时，不应声称全部物理线路都满足下限。
删除接入线表示该线零长度收缩，观测节点和数据列仍保留，删错真实支路会计入遗漏。

全部候选处理完后，按原严格验证MSE重新选择。真值仅用于事后评价，不能决定删哪条线。
阈值0逐值保留原候选参数和原选择，作为成对基线。某候选失败会使该计划格失败，
保留原分母，不从库中静默去掉失败候选。

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m pytest -q rnj_wzzt_dr/experiments/impedance_floor_20261116
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m rnj_wzzt_dr.experiments.impedance_floor_20261116.run --out outputs/impedance_floor_20261116_v1 --workers 3 --prepare-only
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m rnj_wzzt_dr.experiments.impedance_floor_20261116.run --out outputs/impedance_floor_20261116_v1 --resume
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m rnj_wzzt_dr.experiments.impedance_floor_20261116.audit_report --run outputs/impedance_floor_20261116_v1 --out outputs/impedance_floor_audit_20261116_v1
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m rnj_wzzt_dr.experiments.impedance_floor_20261116.plot_results --run outputs/impedance_floor_20261116_v1 --out outputs/impedance_floor_plots_20261116_v1
```

原始观测、每候选系数、QP证书和完整删线轨迹留在ignored `outputs`。
各输入的`banks.json`保存完整新库及原候选ID，`original_native.json`保留原识别器记录。
可直接读取这些新候选做后续研究；本入口没有修改生产辨识器默认值。
