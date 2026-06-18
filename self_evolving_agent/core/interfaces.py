"""
核心接口定义模块 —— 定义所有模块的抽象基类和接口契约

此模块明确规范：
1. 每个模块必须实现的接口
2. 数据结构的精确类型定义
3. 状态向量的标准键名
4. 返回值字典的标准键名
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 状态向量标准键名契约
# ============================================================
class StateKeys:
    """状态向量标准键名 —— 所有模块必须使用这些键名"""
    CPU_PERCENT = "cpu_percent"          # CPU使用率 [0.0, 1.0]
    MEMORY_PERCENT = "memory_percent"    # 内存使用率 [0.0, 1.0]
    DISK_PERCENT = "disk_percent"        # 磁盘使用率 [0.0, 1.0]
    ERROR_RATE = "error_rate"            # 错误率 [0.0, 1.0]
    LOAD_AVG = "load_avg"                # 平均负载 [0.0, +inf)
    MODEL_ACCURACY = "model_accuracy"    # 模型准确率 [0.0, 1.0]
    AGENT_CPU = "agent_cpu"              # Agent自身CPU使用率 [0.0, 1.0]
    AGENT_MEMORY = "agent_memory"        # Agent自身内存使用率 [0.0, 1.0]
    DISK_IO = "disk_io"                  # 磁盘IO强度 [0.0, 1.0]
    NET_IO = "net_io"                    # 网络IO强度 [0.0, 1.0]

    @classmethod
    def all_keys(cls) -> List[str]:
        return [
            cls.CPU_PERCENT, cls.MEMORY_PERCENT, cls.DISK_PERCENT,
            cls.ERROR_RATE, cls.LOAD_AVG, cls.MODEL_ACCURACY,
            cls.AGENT_CPU, cls.AGENT_MEMORY, cls.DISK_IO, cls.NET_IO,
        ]


# ============================================================
# 预测值标准键名契约
# ============================================================
class MetricKeys:
    """预测指标标准键名"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"

    @classmethod
    def all_keys(cls) -> List[str]:
        return [cls.CPU, cls.MEMORY, cls.DISK]


# ============================================================
# 误差报告标准键名契约
# ============================================================
class ErrorReportKeys:
    """误差报告返回值的标准键名"""
    MSE = "mse"                          # 均方误差
    MAE = "mae"                          # 平均绝对误差
    MEAN_RELATIVE_ERROR = "mean_relative_error"
    ERROR_ENTROPY = "error_entropy"
    FREE_ENERGY = "free_energy"
    MEAN_ERROR = "mean_error"
    STD_ERROR = "std_error"
    SURPRISE_COUNT = "surprise_count"
    SURPRISE_RATE = "surprise_rate"
    MOVING_AVERAGE = "moving_average"
    HISTORY_STD = "history_std"
    TREND = "trend"
    RECONSTRUCTION_SIGNAL = "reconstruction_signal"
    SAMPLE_COUNT = "sample_count"
    TIMESTAMP = "timestamp"

    @classmethod
    def all_keys(cls) -> list:
        return [
            cls.MSE, cls.MAE, cls.MEAN_RELATIVE_ERROR, cls.ERROR_ENTROPY,
            cls.FREE_ENERGY, cls.MEAN_ERROR, cls.STD_ERROR, cls.SURPRISE_COUNT,
            cls.SURPRISE_RATE, cls.MOVING_AVERAGE, cls.HISTORY_STD, cls.TREND,
            cls.RECONSTRUCTION_SIGNAL, cls.SAMPLE_COUNT, cls.TIMESTAMP,
        ]


# ============================================================
# 驱力字典标准键名契约
# ============================================================
class DriveKeys:
    """驱力字典的标准键名"""
    SAFE = "safe"
    GAIN = "gain"
    EFF = "eff"

    @classmethod
    def all_keys(cls) -> list:
        return [cls.SAFE, cls.GAIN, cls.EFF]


# ============================================================
# 抽象基类定义
# ============================================================
class IPerception(ABC):
    """环境感知接口"""

    @abstractmethod
    def get_state_vector(self) -> Dict[str, float]:
        """获取标准化状态向量 [0.0, 1.0]
        返回值必须包含 StateKeys 中定义的键
        """
        pass


class IPrediction(ABC):
    """预测引擎接口"""

    @abstractmethod
    def update(self, state: Dict[str, float]) -> None:
        """更新历史状态
        Args:
            state: 必须包含 MetricKeys 中定义的键
        """
        pass

    @abstractmethod
    def predict(self) -> Any:
        """返回预测结果对象（PredictionResult）"""
        pass

    @abstractmethod
    def online_learn(self, actual: Dict[str, float]) -> None:
        """根据实际值进行在线学习"""
        pass


class IDriveEngine(ABC):
    """驱力引擎接口"""

    @abstractmethod
    def compute(self, state_vector: Dict[str, float],
                tasks_completed: int, tasks_failed: int,
                prediction_error: float) -> Any:
        """计算驱力状态
        Args:
            state_vector: 状态向量，键名遵循 StateKeys 契约
            tasks_completed: 累计已完成任务数（非负整数）
            tasks_failed: 累计失败任务数（非负整数）
            prediction_error: 预测误差（非负浮点数）
        Returns:
            DriveState 对象
        """
        pass

    @abstractmethod
    def get_exploration_ratio(self) -> float:
        """返回当前探索比例 [0.0, 1.0]"""
        pass

    @abstractmethod
    def should_explore(self) -> bool:
        """返回是否应进行探索"""
        pass


class IDecision(ABC):
    """决策引擎接口"""

    @abstractmethod
    def decide_and_execute(self, state_vector: Dict[str, float],
                           drives: Dict[str, float]) -> Any:
        """基于状态和驱力进行决策并执行
        Args:
            state_vector: 状态向量
            drives: 驱力字典，键名遵循 DriveKeys 契约
        Returns:
            ActionResult 对象
        """
        pass


class IErrorEvaluator(ABC):
    """误差评估接口"""

    @abstractmethod
    def evaluate(self, predicted: Any, actual: Any) -> Dict[str, Any]:
        """评估预测误差
        Args:
            predicted: 预测值（字典或列表）
            actual: 实际值（字典或列表）
        Returns:
            误差报告字典，键名遵循 ErrorReportKeys 契约
        """
        pass


class IOptimizer(ABC):
    """优化器接口"""

    @abstractmethod
    def online_optimize(self, predicted: Dict[str, float],
                        actual: Dict[str, float],
                        error_report: Dict[str, Any]) -> Any:
        """在线优化"""
        pass

    @abstractmethod
    def offline_optimize(self) -> Any:
        """离线优化"""
        pass


class IMemory(ABC):
    """记忆系统接口"""

    @abstractmethod
    def record_experience(self, state: Dict[str, Any], action: str,
                          reward: float, next_state: Dict[str, Any],
                          td_error: float) -> None:
        """记录经验"""
        pass

    @abstractmethod
    def offline_review(self) -> Dict[str, Any]:
        """离线复盘"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭记忆系统"""
        pass
