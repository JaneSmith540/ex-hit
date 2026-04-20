# 因子添加说明

本项目把事件样本构建和因子计算分开：

- `data_processor.py`：负责找打板事件、准备盘口窗口、准备日级上下文。
- `feature_engineer.py`：负责把窗口和上下文变成模型因子。
- `config_*.yaml`：决定本次实验实际启用哪些因子。

## 最推荐的加法

普通盘口因子优先加在 `feature_engineer.py`：

1. 写一个 `_calculate_xxx(...)` 函数。
2. 在 `_register_default_features()` 里注册：

```python
self.register_feature("my_factor", self._calculate_my_factor)
```

3. 在实验配置的对应组里加名字，例如：

```yaml
feature:
  flow_features:
    - my_factor
```

## 事件上下文因子

如果因子依赖事件时刻、L2 窗口、涨停价、板位、市场热度，放进 `calculate_event_features()`。

当前已经有这些可交易口径因子：

- `touch_time_minutes`：触板时刻，分钟数。
- `touch_seconds_from_open_norm`：触板时刻相对开盘归一化。
- `touch_time_bucket`：触板半小时段。
- `prior_limit_up_streak`：该票事件日前连续收盘涨停次数。
- `board_position`：当前触板是第几板。
- `market_prior_touch_count`：当天此事件前已经有多少只票真实触板。
- `market_prior_touch_ratio`：上述数量除以当日股票池数量。
- `prev_market_max_close_limit_streak`：上一交易日市场最高收盘连板高度。

## 注意

不要把日终才知道的信息直接放进可交易模型，例如“今天最终上涨多少只票”。这类字段可以做描述性统计或分组解释，但不能作为盘中预测因子。
