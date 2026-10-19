# 条件贝叶斯EV小规模验证证据

2条开发、8条冻结评价，完整10条记录均保留。原gzip共67,118,601字节，无损LZMA ZIP共36,146,442字节；未舍入、删除数组或筛除失败。

从项目根目录先读research_experiments/mobile_ev_mle/ev_bayes_small_results.md与ev_bayes_small_reproduce.md。collection_manifest.json给出10个归档的精确路径、原始文件与规范内容SHA；delivery_manifest.json核对本交付目录所有文件。

恢复与核查：

~~~powershell
python -m research_experiments.mobile_ev_mle.ev_bayes_small_evidence unpack-set research_experiments/mobile_ev_mle/evidence/ev_bayes_small --output-dir outputs/bayes_restored_fresh
Copy-Item -LiteralPath research_experiments/mobile_ev_mle/evidence/ev_bayes_small/freeze.json -Destination outputs/bayes_restored_fresh/freeze.json
Copy-Item -LiteralPath research_experiments/mobile_ev_mle/evidence/ev_bayes_small/source_snapshot.json.gz -Destination outputs/bayes_restored_fresh/source_snapshot.json.gz
python -m research_experiments.mobile_ev_mle.ev_bayes_small_audit --input outputs/bayes_restored_fresh --output outputs/bayes_reaudit.json --phase development evaluation
~~~

完整恢复已经执行，restore_manifest.json记录10/10的类型和浮点位相同，restore_audit.json记录恢复后10条历史全部独立AC/概率/运行检查通过。源码133文件快照仍需与当前安装版本一致，恢复工具只恢复数据。

实验单位为8条独立评价历史。regular/shift和每条件32次计量噪声均为同一历史的配对回放；report/results.md报告全部方法、失败及未决，report/diagnostics.md拆分模型内部检查与真实AC模拟。Bayes与同域平方方法的实际错误率相同，固定训练线路造成的响应失配仍存在。

final_tests.xml为77项通过记录；pytest_sandbox_temp_failure.xml单独保存系统临时目录权限导致的8项setup错误，改用工作区临时目录后测试通过。initial_development_failure保留旧开发序列化失败，不计入正式评价。
