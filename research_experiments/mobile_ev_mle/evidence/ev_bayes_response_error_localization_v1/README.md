# 既有误判来源的复现证据

本目录只保存原冻结评价的后处理结果，不更换策略，不新增AC、拟合、MILP、未来量测或动作。原策略policy为6b956e4661c4ee060972f732ba3f529f95b80635304cbdb35dadac1445e435b9；140份冻结科学源码核对一致。

- result.json：128历史的完整/失败列表、分层、配对结果、176个可得历史—情景的评分拆分、预定历史0的独立概率重放及输入/脚本哈希。
- verification.json：独立检查。128份历史中88完整、40失败；条件计量错误共1565次，其中1486次来自无噪声已判错的48个历史—情景。条件重复不增加独立历史样本量。
- ../../ev_bayes_response_error_localization_results.md：简明结果。
- ../../ev_bayes_response_error_localization_math.md：符号、推导和解释范围。
- ../../ev_bayes_response_error_localization.py：仅依赖NumPy、SciPy及标准库的后处理程序。

当前交付对应执行目录outputs/ev_bayes_response_error_localization_v4。v1/v2/v3前次运行仍保留在D盘；v4是审阅与数学排版调整后的独立重放，24项旧汇总检查通过，概率与Gaussian条件分解重放通过。运行约13.25秒，无后处理失败。

复现时从研究工作树根目录运行下列命令；输出目录必须尚不存在。analysis和diagnostics是上批已交付的汇总文件，也保存在旧postprocess_evidence.zip；原始历史0已随旧证据交付。各输入以result.json内SHA256为准。

    D:/apps/miniconda3/envs/Topo/python.exe research_experiments/mobile_ev_mle/ev_bayes_response_error_localization.py --analysis D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/ev_bayes_response_finish_v1/steps/02_analysis_evaluation/attempt_001/result/summary.json --diagnostics D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/ev_bayes_response_finish_v1/steps/04_diagnostics_evaluation/attempt_001/result/diagnostics.json --raw-audit research_experiments/mobile_ev_mle/evidence/ev_bayes_response_final_v1/evaluation_000.json.gz --output outputs/ev_error_localization_rerun

统一拓扑权重、真候选均值替换、去P/Q边际评分均为事后敏感性检查。真均值替换使用未来真值，不作为可部署策略或性能上界；仅历史0的结果不外推到88份历史。已存AC联合重拟合与贝叶斯目标不同，单噪声、局部解且SSE差很小；同时保留移动方案纠正和恒充反向判错的情况。
