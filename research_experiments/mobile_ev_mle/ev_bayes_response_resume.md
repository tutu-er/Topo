# 线路响应不确定性研究续接

本批研究、后处理、人工交叉核对及恢复验证均完成。先读ev_bayes_response_final_report.md和完成包manifest；不再启动本批实验或后处理。

- 研究源：C:/Users/23761/.codex/worktrees/36a1/Topo，分支codex/ev-train-validation-design。
- 执行副本：D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015。
- 当前唯一冻结运行：执行副本/outputs/ev_bayes_response_frozen_v1。
- policy_id：6b956e4661c4ee060972f732ba3f529f95b80635304cbdb35dadac1445e435b9。
- 实现已提交推送：c4418db3；数学文件换行校正65ed88e0。更早方案提交f172dd41。
- 初始controller PID14788已在原生诊断故障后被恢复controller2848接替；健康worker22536保留。当前PID与阶段以controller.json、controller.lock及runtime_recovery_v1/attempts.jsonl和完整用户令牌进程核对为准。两worker、所有数值库单线程。

## 当前边界

旧133份科学源码保持不变；新增5份科学Python文件及数学/协议共7项在freeze中。原PDF与pyproject的已有修改不提交。不要修改当前冻结源码、参数、种子、先验或评价规则；校准与评价不用于调参。

主程序依次完成16开发、64校准、128评价历史，每阶段结束进行完整独立审计；审计失败停止下一阶段。矩阵为三/四末端、4真结构、训练32/96、普通/相关负荷、原/三倍创新。每历史regular/shift和32噪声为配对重复，不增加独立历史。27计划、5同动作推断模型均按协议保留；16评价历史有原共享线路AC局部重拟合审计。

Bayes已替换规划风险、完整AC接受和最终概率评价；AC均值及导数底座与EV运行域沿用。MILP冻结协方差，用AC均值Jacobian构造Bhattacharyya二次距离和分段上界代理；完整风险梯度包含动作引起的协方差变化，目前仅独立诊断。父计划在四组嵌套运行域中继承，新旧BC和固定线路BC统一提交父操作MIP-start。

## 已通过的开发验收

- 新模型/统计/工具及旧small Bayes回归167项通过，证据执行副本/outputs/ev_bayes_response_tests_v1.xml。
- 独立线性有界生成式2048历史，失败0；90%线路覆盖90.19%、未来精确/矩Gaussian区间91.41%、预测/实际错误20.38%/20.65%。脚本ev_bayes_response_linear_calibration.py，原JSON在outputs/ev_bayes_response_linear_calibration_v1.json。仅验证已知超参数线性生成式。
- 旧开发v2的0/1只作开发：完整AC/协方差/后验独立审计通过；首候选GHK复核通过。两例响应排序仍有严重失配，八角分解主要指向线路响应。开发0后验R误差变大；开发1变小但响应误差仍大；二者响应区间过窄。完整历史固定线源profile检查也发现部分后验线路中心的条件SSE明显升高。不要把加入线路方差写成已经校准。
- 两开发共5个预定非零提案方向核查，3个无提案保留。协方差项抵消约8%–61%均值BC梯度，完整方向有限差分误差不超过1.97e-7。原risk score估计方差大，路径导数较稳定；均未改变冻结规划策略。

## 计算完成后的工作

1. 检查controller.status和3阶段audit、所有种子/源码hash/失败分母。statistical_complete表示计算及阶段审计结束，尚需交付。
2. 运行新增ev_bayes_response_finish CLI（先看--help），它只允许statistical_complete后进行统计、物理响应拆分、预定16历史支持审计、最终测试及费用/hash汇总；逐步骤保存checkpoint和日志，失败产物保留。
3. 审查cal/eval报告、配对区间、置信分箱、90%排真与预测区间、强对照及初始化/新提案贡献。报告三/四末端大样本批次，不称大型网络验证。
4. 核对完整AC收益与规划风险排序，分别归因线路/源/负荷模型、线性化、积分与求解预算。当前负例提示下一步检查非线性历史支持与单中心近似；不以任意扩大噪声或提高阻抗下界修饰结果。
5. 整理简明数学及结果报告、全部raw hash、source snapshot、配置/种子、代表性完整raw的可复现包装。大体积原始记录保留D盘，远端只放必要且经过恢复核查的证据。不改长LaTeX/PDF。
6. 源码/报告/证据只放研究目录，按阶段精确暂存、commit、正常push已批准远端分支。保护PDF SHA FE89B241285FD81F570A921FE99A49BFDB1D6BCA6E0CED0D498A45EF81D5DBFA 与pyproject SHA 67D21B6A3DA4629C5CBA5C445404FC5ECDE154BF6B472EE4FA06F75CEB760E1C。
7. 最终回答当前用户：不确定性与预测校准的证据、Bayes改变了哪些指标、MILP梯度来源及与完整梯度的区别、同域基线/放电/移动比较、适用与失败条件。区分推导/实现/验证，不把未完成写完成。完成核对与推送后删除本轮续接自动化，不启动新科学批次。

## 遇到问题

先读取controller.stderr.log、worker日志和失败JSON。进程仍存活时不重启。controller停止但worker仍活时等待其保存，禁止重叠执行。统计/审计工具的错误可修复后换新输出目录重算；冻结科学模型若确有错误，保存失败、暂停该批推进并说明后续修订方案，不能把修改混进已有policy。普通权限或网络问题保留本地提交，不重复请求已批准的远端授权。

worker每5分钟由faulthandler输出一次当前栈，日志中的“Timeout (0:05:00)”表示诊断栈打印，不会终止计算；退出状态及失败JSON才决定是否失败。record.seconds不含记录序列化/压缩，费用报告还须给controller实际起止与并发数。开发004已保留一次joint积分精度失败：524288点、ESS约5.71、质量相对SE25.1%，不删除该竞争者或用较低精度分数继续规划。

## 最近用户明确的主问题

最终报告以贝叶斯方法能否改善EV的MILP规划目标及实际计划质量为主结论。分别量化同动作推断变化、计划池重新排序、MILP新提案及真实排错收益。后验校准和模型内风险只是支撑证据；不把评分改善当计划改善，不把同域基线收益算作MILP增量。保持当前冻结评价，不新增或改动策略。

进程核对须注意：受限令牌对已提权启动的controller调用OpenProcess可能返回False。应使用已授权完整用户令牌的Get-Process核对PID，避免误判退出并重复启动。

## 2026-10-16 20:09 UTC运行恢复

校准worker32188在定时打印Python栈时原生崩溃，最终记录0、2、4保留，6无最终记录。已保留Windows事件、匹配符号解析和日志；仅更换运行入口及编排，140份科学源码、policy和所有种子不变。恢复controller2848接管仍存活的22536，先补校准6，再完成缺失校准/评价。新进程每次一份历史，仅无最终记录的原生失败允许一次重试；已保存科学失败不重试。运行入口5项及编排28项测试通过，开发16份既有独立审计通过。详见ev_bayes_response_runtime_recovery.md。

新worker关闭原五分钟诊断定时器；健康旧worker22536沿原入口继续。续接先查上述锁和attempt日志，不要因为旧controller.json尚在读取大记录时更新延迟而重复启动。全部完成后，finish须纳入运行恢复证据及费用；603秒是原worker完整历时，包含已保存历史，不是可额外相加的丢失时间。

## 2026-10-17 01:42 UTC后处理

统计controller2848已正常退出，controller.status=statistical_complete。三阶段审计均success、error_count=0；完整推断为12/16、45/64、88/128，失败分别4、19、40份，全部保留。审计成功验证实现重放与失败保存，最终物理收益和概率校准仍待解释。

唯一finish进程PID45564；输出outputs/ev_bayes_response_finish_v1/checkpoint.json，日志在其同级ev_bayes_response_finish_v1.stdout.log及.stderr.log，PID在同级.pid文件。先核对finish及checkpoint中的child_pid，存活时不重启。finish开始时对全部原始记录/源码建立输入封存，首次checkpoint可能稍后才出现；finish.lock已记录PID。依次完成inventory、cal/eval统计、cal/eval响应诊断、预定16份历史支持和回归测试。不要在finish运行中修改其工具或测试源码，否则输入封存会变化。

最终必须分开：同动作推断、ranked相对平方基线的初始化排序、新BC相对ranked的新MILP增量；条件均值同时报可用/128和缺失结果范围。旧开发v2的非零完整梯度验证与当前冻结开发的7零方向/1无提案分开。完整AC预测校准独立于已知超参数线性2048历史校准。先检查诊断失败与历史支持失败，再写最终结论。现有统计源码未变，不用评价调参。

## 2026-10-17 最终交付

finish于03:10 UTC完成，7步各一次成功；279项测试通过。评价88/128可完整推断、40失败保留，36为积分精度、4为源中心越域。独立核验1307个文件及全部140科学源码；ZIP875成员、2份成功/失败样本实际恢复且独立二次核对通过，原8.17GB raw留D盘。

最终结论：新BC较旧代理更省时、对仅排序策略有小幅新提案收益，但未超过同域平方基线；预测仍过度自信，线路响应校准是优先问题。移动与放电在共同88历史中有价值，不专属于贝叶斯。主MILP仍冻结协方差，完整贝叶斯梯度仅独立诊断。

最终报告、指标/物理摘录、独立验证、完成包均在研究目录。原PDF和pyproject保持原SHA。完成阶段普通commit/push并核对远端后删除topo-ev续接；该动作之外不再生成新批次。
