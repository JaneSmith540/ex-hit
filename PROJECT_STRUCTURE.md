# 项目结构

这个项目尽量保持一条清楚主线：先构建打板事件，再计算因子，再训练和评估模型。

## 核心文件

```text
config_morning_board_eventstudy.yaml   当前主实验配置
data_processor.py                      数据加载、L2 触板确认、事件样本构建
feature_engineer.py                    因子计算和因子注册
main.py                                实验主类，负责训练/测试切分和模型输入
evaluator.py                           MSE、MAE、R2、IC、RankIC 等指标
models/                                线性模型、树模型、集成模型
tools/run_event_study.py               无条件事件研究
tools/run_model_suite.py               多模型横向比较
tools/analyze_prediction_outputs.py    模型打分分层和组合检查
tests/                                 关键逻辑单元测试
docs/factor_guide.md                   加因子说明
```

## 数据流程

```text
日线数据
  -> 筛选可能触及涨停的股票-日期
  -> 读取对应 L2 数据
  -> 用 LastPrice 确认首次真实触板时间
  -> 构造事件窗口
  -> 计算因子
  -> 生成事件级样本
  -> 事件研究 / 模型训练
```

## 主实验配置

使用：

```text
config_morning_board_eventstudy.yaml
```

重要设置：

```text
event_trigger_mode: last_price_only
event_min_touch_time: 09:30:00
event_max_touch_time: 11:30:00
```

这表示：只用 L2 成交/最新价确认真实触板，不用挂单价提前触发。

## 输出文件

所有主实验输出建议放在服务器：

```text
/mnt/nvme_raid0/experiment_data/logs/metrics
```

常见文件：

```text
*_events.parquet                 事件级样本
*_event_study_summary.csv        事件研究汇总
*_model_comparison.csv           模型指标对比
*_suite_metrics.json             模型详细结果和特征重要性
*_portfolio_check.csv            模型分层组合检查
```

## 维护边界

- 改事件定义：只改 `data_processor.py`。
- 加普通因子：优先改 `feature_engineer.py`。
- 加板位/市场高度这类上下文：改 `data_processor.py`，再在 `feature_engineer.py` 注册。
- 启用/关闭因子：只改 `config_*.yaml`。
- 加模型：放到 `models/`，并在 `models/__init__.py` 注册。

这样导师后续加商用因子时，不需要理解整套工程，只要按 `docs/factor_guide.md` 操作。
