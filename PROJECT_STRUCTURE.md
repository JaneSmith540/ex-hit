# 项目架构与用户文档

## 1. 项目目标

本项目用于研究：

- 事件：个股盘中首次触及涨停价
- 目标：预测该事件对应股票在次日开盘时，相对触板买入价的溢价或折价
- 方法：基于日线、tick、L2、分钟线构建事件级特征，并使用机器学习模型做回归预测

当前推荐的主评估方式是 `walk-forward` 滚动训练/滚动测试，因为它更接近论文式时序实验，不会偷看未来。

## 2. 推荐使用方式

安装依赖：

```bash
pip install -r requirements.txt
```

默认配置文件：

- [config.yaml](/D:/experiment/config.yaml)

最推荐的运行命令：

```bash
python main.py --model random_forest --walk-forward
```

如果需要做模型横向比较：

```bash
python main.py --compare
```

## 3. 整理后的模块划分

### 配置层

- [settings.py](/D:/experiment/settings.py)
  负责配置对象、默认参数、模型注册和路径规范化。
- [config.yaml](/D:/experiment/config.yaml)
  负责本地数据路径、训练测试区间和默认模型参数。

### 数据层

- [data_processor.py](/D:/experiment/data_processor.py)
  负责加载日线、tick、L2、分钟线数据，识别首触涨停事件，构建事件样本，并做训练前清洗。

### 特征层

- [feature_engineer.py](/D:/experiment/feature_engineer.py)
  负责基础事件特征和窗口特征计算。
- [feature_engineer.py](/D:/experiment/feature_engineer.py)
  同时负责基础因子和论文风格的微观结构扩展特征，统一在一个文件内注册和维护。

### 模型层

- [models/base.py](/D:/experiment/models/base.py)
  统一模型接口。
- [models/linear_models.py](/D:/experiment/models/linear_models.py)
  线性模型集合。
- [models/tree_models.py](/D:/experiment/models/tree_models.py)
  树模型集合。
- [models/ensemble_models.py](/D:/experiment/models/ensemble_models.py)
  集成模型集合。

当前主模型建议使用 `random_forest`。

### 评估层

- [evaluator.py](/D:/experiment/evaluator.py)
  负责指标计算、结果汇总和图形输出。

### 入口层

- [main.py](/D:/experiment/main.py)
  作为项目主入口，负责组织数据处理、特征筛选、模型训练、静态评估、滚动评估和模型比较。

## 4. 当前主流程

固定切分流程：

```text
config.yaml
  -> DataProcessor.build_event_dataset()
  -> FeatureEngineer.calculate_event_features()
  -> load_and_process_data()
  -> train_single_model()
  -> evaluate_model()
```

滚动评估流程：

```text
config.yaml
  -> load_and_process_data()
  -> walk_forward_evaluate_model()
      -> 每个测试日只使用此前最近 N 个交易日训练
      -> 训练
      -> 预测当日
      -> 汇总整个测试期指标
```

## 5. 整理后的目录

```text
D:\experiment
├─ settings.py
├─ config.yaml
├─ data_processor.py
├─ feature_engineer.py
├─ evaluator.py
├─ main.py
├─ models\
├─ results\
├─ saved_models\
├─ wfb.pdf
├─ 特征工程与目标.txt
├─ PROJECT_STRUCTURE.md
└─ README.md
```

## 6. 后续维护边界

为了避免项目再次变乱，后续建议按下面的边界维护：

- 新特征优先直接加到 [feature_engineer.py](/D:/experiment/feature_engineer.py) 并在内部注册。
- 数据字段适配只改 [data_processor.py](/D:/experiment/data_processor.py)。
- 新模型只放到 [models](/D:/experiment/models) 目录。
- 新评估口径优先改 [main.py](/D:/experiment/main.py) 和 [evaluator.py](/D:/experiment/evaluator.py)。
- 用户说明只维护 [PROJECT_STRUCTURE.md](/D:/experiment/PROJECT_STRUCTURE.md) 和 [README.md](/D:/experiment/README.md)。

## 7. 当前交付建议

如果现在的目标是继续做成论文，建议只保留一条主线：

1. 主任务：预测次日开盘溢价/折价
2. 主特征：基础事件特征 + 论文风格微观结构特征
3. 主评估：`walk-forward`
4. 主模型：`random_forest`

这样结构最清楚，也最适合后续继续补经济意义、分层表现和论文写作。
