# WZZT与同能量相反移峰：有界试验

先读EXPERIMENT.md。实现隔离于本目录，不改生产或旧实验。本轮第一门为paper15 b8/.001；b8/b9×四原噪声面板保留，余项不自动启动。

程序入口：

- coded_planner.py prepare/run：历史竞争选址、三模板共同900秒/6000LP预算、完整条件价格积分。未完成表不生成价格锁。
- coded_model.py、coded_replay.py：同资源曲线、共同调权、交叉删除、实际接受与最终输出。
- coded_physical.py、actual_worker.py：固定模拟非V2G任务的执行检查；价格冻结后才生成新自然负荷与AC，观测/真值分开。
- controller.py prepare/run：首事件21h与最终四日两检查点，同实际输入去重，完整失败/未知费用记录。
- qualify_device.py、integration_check.py：本轮只读资格与纯数组独立核验源码，原始执行路径及哈希另存证据。

先在独立私有目录prepare，再run；严禁把源已改变的旧protocol重跑。Topo环境下PYTHONPATH包含仓库、rnj_wzzt_core及本目录。命令细节见CONTROLLER.md；规划例：

    python -B coded_planner.py prepare --root <new-private-planning-root> --repo-root <repo> --qualification-root <private-qualification-root>
    python -B coded_planner.py run --root <same-planning-root> --repo-root <repo>

只有case_plans.json完整且price locks已冻结，才可controller prepare/run。actual使用新replicate>=20000、参与seed90410000；只有一条共享四日自然路径，不将32检查点记录写成独立成功率。

代码可运行与合同在给定模型内可行，均不代表真实设备已校准或DR已改善恢复；阶段结果见已完成的REPORT.md及STATUS.md。

本轮完整价门未决后，实际使用显式controls_only路径：fixed_controls.py绑定原未决协议/结果hash，冻结7个原已声明控制策略。controller按28条计数，另列原价格方法4条price_unresolved；它们没有zero价格回退。原规划源已先封存，接口修复后另冻actual源，无价表重跑。

本轮已经结束：28控制记录/17不同拟合完成，另4全价法项未决；early均1错漏、final均0，无奖励相同。6条记录WZZT未全扫，全部边界见REPORT。没有待续跑记录或自动扩批。公开证据在evidence/coded_dr_v1，数据轨迹仍私有。
