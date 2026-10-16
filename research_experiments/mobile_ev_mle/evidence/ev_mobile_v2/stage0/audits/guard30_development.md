# Guard开发完整回放

全部28个开发槽保留，对应16个基础seed；n3为20槽/8 seed，n4为8槽/8 seed。未读取正式评价。
原版与guard比较同时受到计算负载、30秒软预算和MILP有限时限解的影响，不能全部归因于保护约束。

表中“原始正确”是完整最终拟合SSE唯一第一，不是校准后唯一识别。规划失败即使采用合法回退也照计。

## n3：20槽/8 seed

|方法|规划失败|最终联合拟合未完成|原始正确|平均M|平均规划秒|实际移动|
|---|---:|---:|---:|---:|---:|---:|
|none|0|0|16|7.596|0.00|0|
|constant|0|0|17|23.755|0.00|0|
|highlow|0|0|17|69.770|0.00|0|
|square_fixed|0|0|16|147.577|1.03|0|
|square_mobile|0|0|17|155.262|5.26|8|
|scalar_fixed|0|0|16|152.188|7.60|0|
|scalar_mobile|0|0|20|162.625|17.98|10|
|cap_fixed|0|0|17|146.601|9.14|0|
|cap_mobile|0|0|17|152.286|21.03|2|
|jointgrad_fixed|1|0|16|145.897|18.43|0|
|jointgrad_mobile|1|0|16|145.897|29.38|0|

B移动相对B固定点：{'win': 0, 'tie': 20, 'loss': 0, 'incomplete': 0, 'valid_pairs': 20}；驾驶分钟均值0.0。
B移动梯度MILP调用10、接受0、最终直接梯度计划0。
保护请求20、启用19、最终检查19、通过19、缺失1；{'initial_guard_checked': 25, 'initial_guard_rejected_complete': 4, 'initial_guard_incomplete': 0, 'gradient_attempts_not_accepted': 10, 'proposal_guard_checked': 0, 'proposal_guard_rejected_complete': 0, 'proposal_guard_incomplete': 0}。

原版→guard中未改变算法的基线也应核对曲线：
- square_fixed：曲线相同20/20；规划失败状态变化0。
- square_mobile：曲线相同12/20；规划失败状态变化0。
- scalar_fixed：曲线相同18/20；规划失败状态变化0。
- scalar_mobile：曲线相同20/20；规划失败状态变化0。
- cap_fixed：曲线相同20/20；规划失败状态变化0。
- cap_mobile：曲线相同18/20；规划失败状态变化0。

## n4：8槽/8 seed

|方法|规划失败|最终联合拟合未完成|原始正确|平均M|平均规划秒|实际移动|
|---|---:|---:|---:|---:|---:|---:|
|none|0|0|6|4.464|0.00|0|
|constant|0|0|6|5.732|0.00|0|
|highlow|0|0|7|12.473|0.00|0|
|square_fixed|0|0|7|23.805|2.21|0|
|square_mobile|0|0|7|23.805|5.23|0|
|scalar_fixed|0|0|7|24.899|16.72|0|
|scalar_mobile|0|0|8|39.443|30.18|7|
|cap_fixed|0|0|7|24.927|20.04|0|
|cap_mobile|0|0|8|23.655|33.74|4|
|jointgrad_fixed|7|0|7|24.047|30.39|0|
|jointgrad_mobile|7|0|7|24.047|43.86|0|

B移动相对B固定点：{'win': 0, 'tie': 8, 'loss': 0, 'incomplete': 0, 'valid_pairs': 8}；驾驶分钟均值0.0。
B移动梯度MILP调用0、接受0、最终直接梯度计划0。
保护请求8、启用1、最终检查1、通过1、缺失7；{'initial_guard_checked': 1, 'initial_guard_rejected_complete': 0, 'initial_guard_incomplete': 0, 'gradient_attempts_not_accepted': 0, 'proposal_guard_checked': 0, 'proposal_guard_rejected_complete': 0, 'proposal_guard_incomplete': 0}。

原版→guard中未改变算法的基线也应核对曲线：
- square_fixed：曲线相同8/8；规划失败状态变化0。
- square_mobile：曲线相同8/8；规划失败状态变化0。
- scalar_fixed：曲线相同8/8；规划失败状态变化0。
- scalar_mobile：曲线相同4/8；规划失败状态变化0。
- cap_fixed：曲线相同8/8；规划失败状态变化0。
- cap_mobile：曲线相同6/8；规划失败状态变化0。

