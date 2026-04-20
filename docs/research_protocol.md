# 量化打板隔日溢价研究规程

## 研究目标

本项目研究 A 股股票触及涨停板后的隔日开盘溢价。核心问题不是复现参考论文的高频收益预测，而是在其“高频数据 + 机器学习 + 严格样本外评估”的框架下，改写为适合打板交易场景的事件研究和预测任务。

目标变量定义为：

```text
next_open_return = (next_open - limit_price) / limit_price
```

其中 `limit_price` 使用未复权的真实交易价格计算，因为涨停价、逐笔成交价、盘口价格均是交易所原始价格口径。后复权数据和复权因子可用于辅助技术特征，但不用于判定是否触板。

## 参考论文对应关系

参考论文 `wfb.pdf` 的基本结构是：

- 数据：2021-02-23 至 2023-05-29 的中国 A 股高频 Level-2 数据。
- 任务：用高频交易和盘口变量预测短期收益。
- 方法：LASSO、Ridge、Elastic Net、Random Forest、LightGBM 等机器学习模型。
- 评估：严格按时间滚动训练和样本外测试，报告预测能力和特征重要性。

本项目保留其可交代的研究骨架：

- 使用真实高频数据，而非本地样例数据。
- 使用时间顺序切分训练集和测试集。
- 使用机器学习模型预测收益。
- 报告数据清理、缺失数据、样本构造和模型评估结果。

本项目的核心改动是：

- 研究对象从“一般高频收益预测”改为“触及涨停板事件后的隔日溢价预测”。
- 预测点从固定秒级时间点改为首次触板事件。
- 目标收益从秒级收益改为隔日开盘相对涨停价收益。

## 数据口径

当前服务器数据目录：

```text
/mnt/nvme_raid0/experiment_data
```

主要数据源：

```text
day:        /mnt/nvme_raid0/experiment_data/day/day_all_a_raw_from_tick_20210223_20230529.parquet
tick:       /mnt/nvme_raid0/experiment_data/tick/stock_tick_month
l2_order:   /mnt/nvme_raid0/experiment_data/l2/order
minute_hfq: /mnt/nvme_raid0/experiment_data/min/1m_hfq
adj_factor: /mnt/nvme_raid0/experiment_data/adj_factor
```

已验证日线数据：

```text
date range: 2021-02-24 to 2023-05-29
rows:       2,544,765
symbols:    5,061
```

## 数据清理要求

数据清理必须可报告、可追溯，不允许静默丢数据。

当前规则：

- 缺失 tick：事件不能确认首次触板，剔除并记录。
- 缺失分钟数据：事件保留，分钟技术特征为空，由后续特征清洗决定是否可用于模型。
- 缺失 L2：事件保留，L2 相关特征为空或置零，同时记录缺失。
- L2 文件按交易日期匹配对应月份分片，不能误读未来月份文件。
- 训练前删除含有 `NaN`、`null`、非有限数值的样本。
- 训练前删除当前样本内全为非有限值或无方差的特征列。

缺失报告输出：

```text
/mnt/nvme_raid0/experiment_data/logs/missing_data_report_summary.csv
/mnt/nvme_raid0/experiment_data/logs/missing_data_report_detail.csv
```

Smoke 验证报告：

```text
/mnt/nvme_raid0/experiment_data/logs/missing_data_smoke_summary.csv
/mnt/nvme_raid0/experiment_data/logs/missing_data_smoke_detail.csv
```

## 事件定义

对每个股票和交易日：

1. 使用原始日线 `pre_close` 计算涨停价。
2. 主板默认 10%，创业板和科创板默认 20%。
3. 先用日内最高价筛选可能触板股票。
4. 再读取 tick 数据，找到首次满足以下任一条件的时刻：

```text
current >= limit_price
or a1_p >= limit_price
or b1_p >= limit_price
```

5. 以首次触板时刻前后窗口构造事件特征。
6. 使用下一交易日开盘价计算目标变量。

## 特征体系

当前特征分为几组：

- 价格动态：收益率、收益速度、波动率、价格位置。
- 成交量和交易强度：成交量比率、成交量突增、交易强度。
- 盘口特征：买卖一档比例、订单不平衡、盘口深度比例。
- 主动买卖和资金流：主动买入比例、大单比例、净流入。
- 技术特征：均线斜率、短期 RSI、动量。
- 论文仿写特征：成交广度、带符号成交量、交易不平衡、报价价差、盘口不平衡、接近涨停程度等。

新增特征原则：

- 优先通过 `extra_feature_registry` 或独立特征函数加入。
- 不直接破坏主实验流程。
- 每个特征必须说明使用的数据源、时间窗口和经济含义。

## 模型与实验流程

基础模型：

- LASSO
- Ridge
- Elastic Net
- Random Forest
- LightGBM

当前 smoke 已验证 LASSO 可在服务器真实数据上跑通。

建议正式实验流程：

1. 先跑 1 周窗口，确认缺失比例和运行时间。
2. 再跑 1 个月窗口，检查模型结果是否稳定。
3. 最后跑完整窗口：

```text
train: 2021-02-23 to 2022-12-30
test:  2023-01-03 to 2023-05-29
```

如果全 A 运行时间过长，可在论文中采用两级实验：

- 主实验：沪深 A 股全样本或流动性过滤后的全样本。
- 稳健性：较短窗口、不同板块、不同涨停制度、不同模型。

## 已完成的服务器 smoke 结果

Smoke 配置：

```text
train: 2021-03-01
test:  2021-03-02
```

结果：

```text
candidate_rows:          8,352
daily_limit_candidates:    174
events_built:              173
missing_tick:                0
missing_minute:             14
missing_l2:                 82
```

模型成功完成：

- 事件构建
- 缺失数据报告
- LASSO 训练
- 样本外评估

## 可复现命令

服务器项目目录：

```bash
cd /home/busanbusi/experiment
```

Smoke 命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1 \
/home/busanbusi/.virtualenvs/experiment/bin/python -u main.py \
  --config config_smoke.yaml \
  --model lasso
```

正式配置命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1 \
/home/busanbusi/.virtualenvs/experiment/bin/python -u main.py \
  --config config.yaml \
  --model lasso
```

后台运行建议：

```bash
cd /home/busanbusi/experiment
nohup env MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1 \
  /home/busanbusi/.virtualenvs/experiment/bin/python -u main.py \
  --config config.yaml \
  --model lasso \
  > /mnt/nvme_raid0/experiment_data/logs/full_run_lasso.log 2>&1 &
echo $! > /mnt/nvme_raid0/experiment_data/logs/full_run_lasso.pid
```

查看进度：

```bash
tail -f /mnt/nvme_raid0/experiment_data/logs/full_run_lasso.log
```

## 论文写作建议

论文主线建议写成：

1. 引言：打板交易在 A 股市场中的现实意义，涨停制度造成的流动性和情绪集聚。
2. 文献综述：高频收益预测、订单流不平衡、A 股涨跌停制度、机器学习资产定价。
3. 数据与样本：说明 tick、L2、分钟、日线数据，给出清理流程和缺失报告。
4. 事件构造：定义首次触板事件和隔日开盘溢价。
5. 特征工程：按价格、成交量、订单流、盘口、技术指标分组。
6. 模型方法：说明 LASSO 等模型和时间切分。
7. 实证结果：报告预测性能、特征重要性、缺失数据和稳健性。
8. 结论：打板隔日溢价是否可预测、哪些微观结构特征最有效、局限与改进。

老师最容易追问的地方要提前准备：

- 为什么事件判定不用后复权。
- 为什么缺失 L2 不直接删除所有事件。
- 如何避免未来函数。
- 训练集和测试集是否严格按时间切分。
- 样本筛选是否会造成选择偏差。
- 全样本和子样本结果是否一致。
