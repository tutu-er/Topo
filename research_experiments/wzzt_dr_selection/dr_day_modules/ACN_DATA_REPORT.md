# ACN会话数据资格核验

2026-10-31。仅核验字段与信息时序，没有新增调度、拓扑或运行试验。

## 来源与范围

已读[官方字段说明](https://ev.caltech.edu/dataset)、[官方教程](https://acnportal.readthedocs.io/en/latest/tutorials/acnsim/lesson1.html)、少量作者测试源码及API小样本。手工测试fixture未当作实测。[作者静态仓库](https://github.com/tongxin-li/ACN-Data-Static)和[软件许可](https://github.com/zach401/acnportal/blob/master/LICENSE.txt)也已核对；软件许可不替代数据条款。

## 5/38字段资格样本

固定Caltech站点、洛杉矶本地2019-05-01全天，即UTC 2019-05-01 07:00至次日07:00（右端不含）。按connectionTime排序，只读第1页最多5条，不翻页、不取timeseries。API报告窗口共38条，本次保存5条、3526 bytes。

原响应SHA-256：6e154240611f07448f69b85540bbd5c03f6d9855b1918706146f0e06fca2aa10。

该样本只证明字段可用，不能当成代表性校准、未来性能留出或价格响应试验。

## 字段与信息时序

- 事后记录：connectionTime、disconnectTime、doneChargingTime、kWhDelivered、timezone。
- 用户声明：5/5含kWhRequested、requestedDeparture、minutesAvailable、modifiedAt。
- 5/5满足connectionTime≤modifiedAt<requestedDeparture，且modifiedAt<实际断开。首个声明在连接后13–29秒提供。使用个体声明的报价/控制不得早于modifiedAt；15分钟控制不能回填此前时段。
- 5/5请求能量大于已交付能量；两者不能互换，交付/请求比不是DR参与率。样本预计离站早于实际断开也不是未来可用时间保证。
- 请求能量除以“声明到预计离站的剩余小时”，得到所需平均充电率1.0014–7.4722 kW。它不是实测最大功率，未据此判可行。

## 可用模型及缺口

会话可支持服务能量任务：
\[
E^{del}_{t+1}=E^{del}_t+P_t\Delta t,\quad
0\le P_t\le v_t\bar P,\quad E^{del}_{d}\ge E^{req}.
\]
E仅记录交付服务能量，不是真实SOC；计量侧口径、功率上限和效率仍需另有声明。

数据缺少初始SOC、容量、真实车辆Pmax、DR报价/接受过程及馈线P/Q/V。当前v3 Battery域不能无假设直接接入；应另建能量任务接口，或保留明确的合成电池参数标签，不称实测SOC/参与价格校准。

下一步先锁历史训练期、预测及无激励基线规则。日前规划只用历史预测；到站并出现声明后再更新，后续修改不回填。实际断开、交付偏差和任务未完成用于事后核算，不能预知。映射到模拟末端仍是“真实会话参数＋合成网络/节点映射”，不是现场拓扑试验。

## 安全白名单

可复制本报告与public_schema_summary.json；后者只含来源、schema、窗口、聚合核验和hash，没有认证字段、原始ID或单会话轨迹。

public_schema_summary.json的SHA-256：269d81c3dc8f718082445bc0e2bb09644b395708cbeeb45cd6a0b590a48330b3。

原响应、逐条字段记录及获取辅助脚本留私有目录。未核实到独立的数据再分发条款，因此原始会话不纳入Git。
