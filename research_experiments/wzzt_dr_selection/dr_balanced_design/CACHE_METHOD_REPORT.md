# LP方法与完整观测缓存隔离

2026-10-31。修复仅涉及缓存身份与方法传递，不改变辨识目标或策略规则，不重新运行研究批。

## 问题

fit_observed已显式区分lp_method，但SharedFitter此前只按观察、来源、root及search_seconds生成key。若调用者用partial将方法绑定为2，却沿用方法1的共享缓存，可能静默返回另一个数值配置的结果。原预算批都使用默认1；两次完整数值重放直接调用fit_observed且使用独立目录，没有发生此混用。

## 修复

- observation_identity及SharedFitter显式接收lp_method（默认1），将它加入key与metadata。
- 非默认方法传给现有fit函数；旧默认自定义fake签名兼容。
- 新结果、磁盘结果、缓存metadata及返回前均核对方法记录；冲突直接报错。
- partial绑定2却请求默认1会拒绝；非默认方法没有任何方法记录也会拒绝。
- 默认旧fake缺记录时仅标legacy_default_unrecorded，不当成实际求解方法已验证。

这保证记录与请求一致；自定义fit函数的内部计算仍由其自身实现负责。

## 测试与边界

新增8项和既有matrix4项共12个不同用例通过，另7个参数子测试不重复扩充分母。全部使用fake拟合，真实fit/LP/QP/AC调用均0；没有重复旧研究数据或改变参照选择。

初次测试夹具在受限系统TEMP出现WinError5，原XML保留。只将新夹具改为pytest tmp_path、使用另一个已核验私有basetemp后全部通过；生产代码没有因夹具失败增加其他逻辑。

与前一数值API的6项新测试、7项既有adapter测试合计25个不同用例，研究恢复证据仍以原64观察与另2个数值重放分别报告。

## 恢复

源：dr_pool_v2/matrix_runner.py；标准测试：research_experiments/tests/test_wzzt_dr_pool_matrix_lp_method.py及test_wzzt_dr_pool_matrix_runner.py。

公开evidence/shared_fitter_lp_method_v1/summary.json含测试、源码与失败回执索引；原XML/私有候选留scratch/source。此修改改变source hash，旧冻结协议须使用各自精确源码快照；不为继续旧缓存而跳过校验。
