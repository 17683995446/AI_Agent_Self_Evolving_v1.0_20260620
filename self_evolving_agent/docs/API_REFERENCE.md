# Agent系统 API 参考文档

> 自动生成于: 2026-06-06T00:35:22.386896

## 一、核心接口定义

### IDecision

决策引擎接口

- `decide_and_execute(self, state_vector: Dict[str, float], drives: Dict[str, float]) -> Any`
  - 基于状态和驱力进行决策并执行
Args:
    state_vector: 状态向量
    drives: 驱力字典，键名遵循 DriveKeys 契约
Returns:
    ActionResult 对象

### IDriveEngine

驱力引擎接口

- `compute(self, state_vector: Dict[str, float], tasks_completed: int, tasks_failed: int, prediction_error: float) -> Any`
  - 计算驱力状态
Args:
    state_vector: 状态向量，键名遵循 StateKeys 契约
    tasks_completed: 累计已完成任务数（非负整数）
    tasks_failed: 累计失败任务数（非负整数）
    prediction_error: 预测误差（非负浮点数）
Returns:
    DriveState 对象

- `get_exploration_ratio(self) -> float`
  - 返回当前探索比例 [0.0, 1.0]

- `should_explore(self) -> bool`
  - 返回是否应进行探索

### IErrorEvaluator

误差评估接口

- `evaluate(self, predicted: Any, actual: Any) -> Dict[str, Any]`
  - 评估预测误差
Args:
    predicted: 预测值（字典或列表）
    actual: 实际值（字典或列表）
Returns:
    误差报告字典，键名遵循 ErrorReportKeys 契约

### IMemory

记忆系统接口

- `close(self) -> None`
  - 关闭记忆系统

- `get_stats(self) -> Dict[str, Any]`
  - 获取统计信息

- `offline_review(self) -> Dict[str, Any]`
  - 离线复盘

- `record_experience(self, state: Dict[str, Any], action: str, reward: float, next_state: Dict[str, Any], td_error: float) -> None`
  - 记录经验

### IOptimizer

优化器接口

- `offline_optimize(self) -> Any`
  - 离线优化

- `online_optimize(self, predicted: Dict[str, float], actual: Dict[str, float], error_report: Dict[str, Any]) -> Any`
  - 在线优化

### IPerception

环境感知接口

- `get_state_vector(self) -> Dict[str, float]`
  - 获取标准化状态向量 [0.0, 1.0]
返回值必须包含 StateKeys 中定义的键

### IPrediction

预测引擎接口

- `online_learn(self, actual: Dict[str, float]) -> None`
  - 根据实际值进行在线学习

- `predict(self) -> Any`
  - 返回预测结果对象（PredictionResult）

- `update(self, state: Dict[str, float]) -> None`
  - 更新历史状态
Args:
    state: 必须包含 MetricKeys 中定义的键

## 二、标准键名契约

### StateKeys

状态向量标准键名 —— 所有模块必须使用这些键名

- `cpu_percent`
- `memory_percent`
- `disk_percent`
- `error_rate`
- `load_avg`
- `model_accuracy`
- `agent_cpu`
- `agent_memory`
- `disk_io`
- `net_io`

### MetricKeys

预测指标标准键名

- `cpu`
- `memory`
- `disk`

### DriveKeys

驱力字典的标准键名

- `safe`
- `gain`
- `eff`

### ErrorReportKeys

误差报告返回值的标准键名

- `mse`
- `mae`
- `mean_relative_error`
- `error_entropy`
- `free_energy`
- `mean_error`
- `std_error`
- `surprise_count`
- `surprise_rate`
- `moving_average`
- `history_std`
- `trend`
- `reconstruction_signal`
- `sample_count`
- `timestamp`

## 三、模块实现文档

### EnvironmentPerception

**模块**: `modules.perception`

环境感知器 —— 全维度状态采集

- `__init__(self) -> None`
- `collect_file_metrics(*args, **kwargs)`
- `collect_log_metrics(*args, **kwargs)`
- `collect_self_metrics(*args, **kwargs)`
- `collect_system_metrics(*args, **kwargs)`
- `get_state_vector(*args, **kwargs)`
- `start(self) -> None`
- `stop(self) -> None`

### DriveEngine

**模块**: `modules.drive_engine`

驱力引擎主类 —— 统一接口，调度三大驱力计算

- `__init__(self) -> None`
- `compute(*args, **kwargs)`
- `get_drive_summary(*args, **kwargs)`
- `get_exploration_ratio(*args, **kwargs)`
- `should_explore(*args, **kwargs)`

### DecisionEngine

**模块**: `modules.decision`

决策引擎主类 —— 整合动作空间、策略和执行器

- `__init__(self) -> None`
- `decide_and_execute(*args, **kwargs)`
- `get_status(*args, **kwargs)`
- `reset_stats(*args, **kwargs)`

### PredictionEngine

**模块**: `modules.prediction`

预测引擎主类 —— 协调三层预测器，支持在线学习和自动重置

- `__init__(self, base_window: int = 10, mid_lookback: int = 20, high_window: int = 60) -> None`
- `get_confidence_report(*args, **kwargs)`
- `online_learn(*args, **kwargs)`
- `predict(*args, **kwargs)`
- `predict_all(*args, **kwargs)`
- `update(*args, **kwargs)`

### ErrorEvaluator

**模块**: `modules.error_eval`

误差评估器主类 —— 整合所有误差评估功能

- `__init__(self, history_size: int = 1000) -> None`
- `compute_error_entropy(*args, **kwargs)`
- `compute_mae(*args, **kwargs)`
- `compute_mse(*args, **kwargs)`
- `compute_relative_error(*args, **kwargs)`
- `evaluate(*args, **kwargs)`
- `get_history_stats(*args, **kwargs)`
- `reset(*args, **kwargs)`

### Optimizer

**模块**: `modules.optimizer`

优化器主类 —— 整合梯度下降、超参数搜索和元学习

- `__init__(self, learning_rate: float = 0.01, variant: str = 'adam') -> None`
- `get_current_params(*args, **kwargs)`
- `get_history(*args, **kwargs)`
- `meta_optimize(*args, **kwargs)`
- `offline_optimize(*args, **kwargs)`
- `online_optimize(*args, **kwargs)`
- `optimize(*args, **kwargs)`
- `reset(*args, **kwargs)`
- `search_hyperparameters(*args, **kwargs)`

### MemorySystem

**模块**: `modules.memory`

记忆系统主类 —— 整合短期、长期、工作记忆

- `__init__(self, stm_capacity: int = 100, ltm_capacity: int = 1000, wm_capacity: int = 7) -> None`
- `clear(*args, **kwargs)`
- `close(*args, **kwargs)`
- `consolidate(*args, **kwargs)`
- `forget(*args, **kwargs)`
- `get_stats(*args, **kwargs)`
- `offline_review(*args, **kwargs)`
- `record_experience(*args, **kwargs)`
- `retrieve(*args, **kwargs)`
- `retrieve_by_tag(*args, **kwargs)`
- `sample_for_training(*args, **kwargs)`
- `store(*args, **kwargs)`
