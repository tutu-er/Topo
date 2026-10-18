# 标准线路与公开P/Q时序测试

本轮承接已完成的full_pipeline；先读protocol.md、math.md及sources中的来源审查。阶段：来源核验、精确内点表示、稀疏AC组件验证与开发pilot。正式校准/评价尚未开始。

已逐条核对原MATPOWER case33bw和CIGRE居民子网。旧paper11/paper15只是项目自建论文风格网络，恢复原放大前阻抗并单列变压器扩展；尚未取得之前五篇论文作者最终单相等效逐线数值，具体缺口保留。

SimBench主数据使用其原生P/Q额定值与适当的LV聚合曲线。源P/Q在同一物理接口，输入与未来真值分开。SINTEF/CINELDI小时数据已取得，但原负荷CSV单位未证实，暂不猜单位用于原网AC。

新代码与证据只在本目录；旧批次、原PDF和pyproject不修改。源码分支codex/ev-train-validation-design，执行副本D:/0-github_workspace/Topo/outputs/ev_reference_stage_20261018。
