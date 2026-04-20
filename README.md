# 打板隔日溢价实验

本项目研究沪深 A 股打板事件的隔日开盘溢价。当前主线是：

1. 用日线先找“可能触及涨停”的股票和日期。
2. 用 L2 `LastPrice >= 涨停价` 确认真实首次触板时间。
3. 计算触板前后的盘口、板位、时间、市场热度因子。
4. 预测“次日开盘价相对打板买入价”的收益。
5. 比较线性模型、随机森林、LightGBM、XGBoost 和集成模型。

## 推荐阅读顺序

- `README.md`：先看这里，了解怎么跑。
- `docs/factor_guide.md`：看怎么加因子。
- `config_morning_board_eventstudy.yaml`：当前主实验配置。
- `data_processor.py`：事件识别和样本构建。
- `feature_engineer.py`：因子计算。
- `tools/run_event_study.py`：无条件事件研究。
- `tools/run_model_suite.py`：模型组实验。

## 当前主实验

主配置文件：

```bash
config_morning_board_eventstudy.yaml
```

数据路径在服务器上：

```bash
/mnt/nvme_raid0/experiment_data
```

训练期：

```text
2021-06-01 到 2022-12-30
```

测试期：

```text
2023-01-03 到 2023-05-29
```

事件窗口：

```text
09:30:00 到 11:30:00
```

触板口径：

```text
L2 LastPrice >= 涨停价
```

## 一键复现实验

在服务器 `/home/busanbusi/experiment` 下运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/busanbusi/.virtualenvs/experiment/bin/python tools/run_event_study.py \
  --config config_morning_board_eventstudy.yaml \
  --run-id morning_board_final \
  --out-dir /mnt/nvme_raid0/experiment_data/logs/metrics
```

然后复用事件样本跑模型组：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/busanbusi/.virtualenvs/experiment/bin/python tools/run_model_suite.py \
  --config config_morning_board_eventstudy.yaml \
  --run-id morning_board_final_models \
  --events-path /mnt/nvme_raid0/experiment_data/logs/metrics/morning_board_final_events.parquet \
  --out-dir /mnt/nvme_raid0/experiment_data/logs/metrics
```

最后做组合分层检查：

```bash
/home/busanbusi/.virtualenvs/experiment/bin/python tools/analyze_prediction_outputs.py \
  --metrics-dir /mnt/nvme_raid0/experiment_data/logs/metrics \
  --run-id morning_board_final_models \
  --models random_forest xgboost tree_ensemble ridge \
  --top-frac 0.1
```

## 主要输出

事件研究：

```text
/mnt/nvme_raid0/experiment_data/logs/metrics/*_event_study_summary.csv
/mnt/nvme_raid0/experiment_data/logs/metrics/*_events.parquet
```

模型比较：

```text
/mnt/nvme_raid0/experiment_data/logs/metrics/*_model_comparison.csv
/mnt/nvme_raid0/experiment_data/logs/metrics/*_suite_metrics.json
```

组合检查：

```text
/mnt/nvme_raid0/experiment_data/logs/metrics/*_portfolio_check.csv
```

缺失数据报告：

```text
/mnt/nvme_raid0/experiment_data/logs/missing_data_*_summary.csv
/mnt/nvme_raid0/experiment_data/logs/missing_data_*_detail.csv
```

## 当前核心因子

时间和事件因子：

- `touch_time_minutes`
- `touch_seconds_from_open_norm`
- `touch_time_bucket`

板位和市场高度：

- `prior_limit_up_streak`
- `board_position`
- `prev_market_max_close_limit_streak`

盘中市场热度：

- `market_prior_touch_count`
- `market_prior_touch_ratio`
- `market_total_count`

盘口和资金流：

- `active_buy_ratio`
- `net_inflow`
- `large_order_ratio`
- `l2_limit_count_ratio`
- `l2_limit_volume_ratio`
- `l2_buy_count_ratio`
- `l2_buy_volume_ratio`
- `l2_net_order_flow`
- `l2_large_order_volume_ratio`

## 加因子的位置

优先看：

```text
docs/factor_guide.md
```

一般原则：

- 盘口窗口因子加在 `feature_engineer.py`。
- 日级上下文因子加在 `data_processor.py` 的 `_add_daily_event_context()`。
- 盘中事件排序因子加在 `data_processor.py` 的 `_add_intraday_event_context()`。
- 是否启用因子由 `config_*.yaml` 控制。

不要把日终才知道的信息直接放进盘中预测模型。

## 本地验证

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s ./tests -t . -v
```

当前测试覆盖：

- 板位和市场高度计算。
- 事件时间桶和日级上下文接入。
- “此前已有多少票触板”的盘中可交易口径。
