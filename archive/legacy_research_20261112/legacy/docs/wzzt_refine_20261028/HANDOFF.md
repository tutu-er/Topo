# 给其他对话的最新策略交接

工作区 D:\0-github_workspace\Topo；本批平均 F1 最高的已测版本为 `ridge_pool`，基础策略仍为 `multistart_original_rule`。请先读取同目录 README.md、recommended_strategy.json。

三项请求均已实现：core.py::backward、regularized_pool、two_step；worker.py::refine_bank 统一返回 backward/ridge_pool/two_step/combined 四个对照及过程记录。该接口接收训练/验证观测、原观测生成的模型库和候选池、固定校准量，不接收真簇。

当前主增强仅启用 ridge_pool：在原多初值策略的模型库上，选取原规则给出的收缩前模型；用 raw/demean、lambda=0.001/0.01 的前置有序 R/X 正则化结果补充 NJ/RNJ 候选簇；保留原池及原校准，在扩展池上从该模型追加最多 25 秒交叉替换搜索；原模型库加新路径仍按 3% 简洁规则选点，再执行最终 0.16 收缩，不作最终重估。

alpha=lambda×trace(DᵀD)/(2n)，D=[P Q]。原响应系数界 [0,2]、绝对残差 L1、raw 拟合及独立验证/测试划分保持不变。正则化只作用于新增候选生成，不替换最终参数拟合。

固定 80 组输入中，ridge_pool 平均 F1=0.9319361847，对原多初值提升 2 项、退化 0 项，完整恢复 65/80。原多初值平均 0.9252397562，baseline25 平均 0.9194371947。后向删除和 combined 的平均 F1 下降，two_step 单独使用最终 F1 不变，保留为可调用实验组件。

当前未完整恢复的噪声—数据量表见 ridge_unidentified_noise.csv；旧方法的 15 项与八项变化分别见 previous_unidentified.csv、previous_changes_vs_baseline25.csv。高电压噪声为 P/Q 0.5%、末端/根电压 0.1% 的高斯相对标准差；其他含噪条件电压 0.02%。N 为三个物理日总快照数。

直接读取现有模型库：D:\0-github_workspace\Topo\docs\wzzt_refine_20261028\catalog.sqlite。继续使用 research_experiments.wzzt_budget.records.find/load/load_trace，variant='ridge_pool'。已有 480 条记录，含全部三种参照差值、实际噪声、支持及系数、原输入路径和完整动作日志。

原最优策略执行顺序详见 docs/wzzt_rnj_best_strategy_20261028/README.md，当前报告是在它之上新增扩池修正；请勿把本目录 ridge_pool 与早期叶初值 WZZT_pool 或组合补丁 multistart75 混用。新数据仍需先构造观测驱动的基模型库，refine_bank 本身是增量增强入口，不是从任意原始文件直接完成整条旧流水线的包装器。

本批未新增随机重复。复现使用新目录，保留所有基线与退化，并使用冻结 source/producer_source.zip；交付已绑定输入、过程、代码和评价哈希。
