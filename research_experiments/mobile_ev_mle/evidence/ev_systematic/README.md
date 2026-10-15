# 本轮证据

总入口为上两层的`ev_systematic_results.md`、`ev_systematic_math.md`和`ev_systematic_reproduce.md`。

- `development/calibration/evaluation/*_bundles`：共5包、31,109,348字节，每包不超过20MiB。全部244个记录槽有analysis-lite，包含逐方法曲线、操作、分数、参数误差、成本和来源。开发另有每个场景首历史、共16个完整compact代表；其计量及全部固定/联合拟合状态可回放。
- `analysis`：三套独立规则及开发描述。大JSON用gzip保存，解压后与原JSON逐字节一致；原字节与交付文件SHA同时列于`evidence_index.json`。原阈值保持不变，名义alpha和实际误拒分别报告。
- `audits`：全部244条原raw的物理/成本、代理算术及独立统计审计；另含16代表compact的实际回放和130项测试XML。开发审计在`development/independent_audits`中。
- `manifests`：原raw SHA、冻结/阈值/启动记录和总交付索引。`v1_interrupted_logs`保留早期记账异常，不混入v2比较。

analysis-lite没有完整计量和拟合状态。原818,933,600字节raw及未交付的选择器中间数组仍保留在`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/`；可用冻结源码和种子重新运行，新的有限时限MILP不保证字节相同动作。16代表的compact已另行实际回放通过；它们不冒充全部244条物理状态。

开发scalar审计器曾修正compact-schema读取；其失败中间文件保留在D。发布使用`independent_scalar_audit_completed.json`，最终manifest明确版本边界。该修正不改变科学代码、输入或结果。

`evidence_index.json`逐项记录发布文件的SHA和原路径。目录恢复后，来源绝对路径会变化；统计复核比较分数、阈值、逐历史保留集合及分母，并保留原SHA映射。
