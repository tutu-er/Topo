# 共享功率平方 upper：两条 H-only 完整搜索 pilot

共享 upper 在本次两条固定 H 上都缩小模型、获得可用全域代理上界，并减少选择器耗时。收益排序仍未被确定：H1 的共享选址代理下界比原版低，H2 比原版高；选中动作之间的有效区间两条 H 均重叠。不能把这次 pilot 写成一致的辨别收益提升。

## 固定协议与实现差异

协议、验收与 code/input SHA256 均先保存。只读 sanitized H_710009、H_720001 的六个已知历史通道；没有读原 trial、实际未来、模拟真拓扑或噪声。各历史只重建一次用于搜索的 context，fresh seeds=28101301/28101302；max_centers=2、attempts=3、multistart=2、nfev=300、β±30%、source±0.05、mean forecast。两者均得到四候选、每候选两中心、24 pair；源规则允许保留未优化的 feasible low-loss seeds，它们不是置信样本。

两方法固定相同 pair 指纹、EV 运行模型（8 slots、6kWh、一个 EV）与仅两个 initial vectors（constant site0 3kW、none）。搜索均为 iterations=3、solver=3s、total=25s、segments=2→max8、relative_gap=0.001、absolute_gap=1e−4。types.FunctionType 直接复用原 adaptive_search 的 __code__/defaults/kwdefaults，copy globals；共有 pass-through QP 记录，政策差异只有 upper_master target。原函数 globals 与原 LinearMILP.solve 在每次运行后自动断言恢复，15 个核心源码哈希前后不变。

各 pair 的实际 B 通过精确时段支撑检查，没有 near-zero 删除。共同日志仅留内存记录和 model 引用，快照/文件 IO 在 search 返回后进行；共享后端自身的 snapshot 构造仍属于其实际调用耗时。随着两方法产生不同上界候选，后续 cuts/见证池自然不同，后续轮模型大小不是“固定同一见证池”的额外比较；此前单 upper 比较已覆盖固定池验证。

## 搜索结果

| H | upper backend | 已得动作代理 LB | 全域代理 UB | raw global gap | selector wall 秒 | QP calls | 各轮 upper 变量数 |
|---|---|---:|---:|---:|---:|---:|---|
| 710009 | original | 52.9025436350 | 52.9026627400 | 0.00011910503 | 10.7696 | 120 | [23194, 57754] |
| 710009 | shared | 52.8899573311 | 52.9025728672 | 0.012615536 | 4.1538 | 168 | [322, 466] |
| 720001 | original | 51.1076214152 | 51.1490587530 | 0.041437338 | 6.9915 | 96 | [17434] |
| 720001 | shared | 51.1336659007 | 51.1489626805 | 0.01529678 | 2.1541 | 120 | [322] |

四次搜索均保存 status=proxy_gap_met。原版三次 upper 均 HiGHS time-limit、无 incumbent/best bound；共享三次 upper 均返回 incumbent/best bound。direction-lower master 的 solver bound只界定有限方向代理，不能充当全域平方代理上界。一次共享 upper 的 status=0 仍有非零 mip_gap，原值保存，不按 status 置零。设置的 3s 是 solver 参数，部分原版 Python solver call超过 3s；单机单次耗时不是跨硬件或总体效率定理。

四次 raw gap 都大于 1e−4，停止依赖 0.001×总代理 LB。该分母包含历史代价，不能读成相对新增收益精度。旧/new H1 的活动 AC witness future share分别约 2.28e−7 / 2.06e−7；H2 约 4.13e−6 / 6.09e−6，均由历史代价占据。这是保存活动见证的描述，不是统一非线性 infimum 的分解。

## 内层区间与动作收益

| H / backend | 相对 constant 的动作代理增益区间 |
|---|---|
| 710009 original | [0.0000113656, 0.10868240] |
| 710009 shared | [−0.01257494, 0.10868335] |
| 720001 original | [−0.04127440, 0.15875763] |
| 720001 shared | [−0.01522992, 0.15877361] |

这些均为 selected action q_LB−baseline q_UB 到 selected action q_UB−baseline q_LB。旧 H1 按原始数值区间有一个很小的正下端约 1.14e−5；应完整报告，但其尺度不能证明实际未来辨别改善。共享对 constant 的两条区间都跨 0。共享选中动作减原版选中动作的区间，H1 约 [−0.01258808,0.0000027350]，H2 约 [−0.01524423,0.04130470]，也都跨 0；不能只比较记录的下界就判定真实代理值优劣。

QP success没有被替换成精确最优性。四个选中动作最大 raw primal−tangent gap分别约 1.78e−6、0.01258904、0.04388442、0.01625903；全部逐 pair 保存。共享 master 闭合后仍可受 nuisance QP 界宽度和基于总代理代价的停止尺度限制，本次没有修改这些冻结部件。

## H-only full-AC 回代与数值检查

四个所选动作共 96 pair 全部回代，均 physically_feasible=true；actual 与 affine cost、Taylor norm、AC 方程残差、功率平衡残差及历史/未来见证占比逐 pair 保存。每个物理可行回代值仅是该固定动作、该 pair 非线性 nuisance infimum 的可行见证上界，不能升格为非线性设计界、实际排错或拓扑正确概率。

没有严格负功率、微小非零或严格同时站点功率；原 -0.0 完整保留，共享 H1 的 6.000000000000001 也未裁成 6。EV运行 checks 使用既定2e−6容差。搜索 LB 只按声明运行容差标为可用数值候选界，不给严格区间算术/现场电网安全证书。

为使 AC 证据可离线复查，额外仅以同种子重建两条已授权 H context并保存拟合 latent-H 状态；与原 pilot 的 pair 指纹精确相同，没有重跑 selector。随后 verifier 只读取保存状态/矩阵，用冻结 AC helper 回放，没有调用 fit optimizer 或搜索：696 条 QP primal/tangent 记录、16 个 MILP snapshot、96 个 AC witness 全部通过。primal/tangent 最大差均 0；snapshot 最大行违约1.18e−9；AC cost最大差9.09e−13，Taylor norm最大差7.11e−15。

主要入口为 honly_search_pilot.py；protocol.json 与 code_and_input_hashes_before/after.json 保存协议和冻结来源；各 H 子目录保存 pair NPZ、初始动作 QP、original/shared 全过程、solver snapshots、QP traces及所有 AC 回代。verify_Honly_search_saved.py 进行无优化器重放；AC replay使用已保存 H 拟合状态和冻结 helper，不需要原 trial 或未来数据。evidence_manifest.json 给新交付文件完整 SHA256；本目录 .gitattributes 只保护归档字节，避免 Git 换行转换破坏哈希。

本次支持共享上界表示在这两条 H 的数值构造与计算效果，未验证新策略的校准风险、整体识别效率、真实未来分离或一般场景性能。原 C/D 策略、门限和结果均未改变。
