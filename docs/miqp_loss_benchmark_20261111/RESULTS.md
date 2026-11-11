# L1 与平方误差的独立配对核查

新MIQP核验 96/96，复用L1核验 96/96。完整状态：True。

每种损失、每种结构固定48输入；试运行和未完成结果保留原分母。两种验证视图使用同一输出库，不增加拟合次数。

主结果见 summary.csv，配对差异见 paired_summary.csv。源文件、数据、父分析与逐结果SHA链保存在 verification.json 和 jobs/*/verified.json。

独立QP只核查固定家族的平方损失最优性；结构搜索证书仍按各原生MIQP的实际上下界和状态报告。
