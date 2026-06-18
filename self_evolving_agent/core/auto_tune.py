"""
自动调参模块 —— 基于运行表现自适应调整系统参数

设计原则（第一性原理）：
1. 误差驱动：参数调整的方向由预测误差梯度决定
2. 局部最优：小步长迭代，避免参数震荡
3. 安全边界：任何参数调整不得突破安全约束
"""

import os
import sys
import time
import math
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.config import config_manager


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    # 目标：minimize_error / maximize_reward / maximize_efficiency
    objective: str = "minimize_error"


class AdaptiveTuner:
    """自适应调参器 —— 根据运行表现调整关键参数"""

    def __init__(self):
        self.config = config_manager.config
        self._parameters: Dict[str, ParameterSpace] = {}
        self._performance_history: Dict[str, List[float]] = {}
        self._last_tune_time = 0.0
        self.tune_interval = 60.0  # 每60秒评估一次
        self._initialize_parameters()
        logger.info("AUTO_TUNE", "自适应调参器初始化", module="auto_tune")

    def _initialize_parameters(self) -> None:
        """初始化可调参数空间"""
        # 预测窗口大小
        self._parameters["prediction_window"] = ParameterSpace(
            name="prediction_window",
            current_value=10.0,
            min_value=5.0,
            max_value=50.0,
            step_size=1.0,
            objective="minimize_error",
        )
        # 探索比例（epsilon）
        self._parameters["exploration_ratio"] = ParameterSpace(
            name="exploration_ratio",
            current_value=0.3,
            min_value=0.05,
            max_value=0.5,
            step_size=0.02,
            objective="maximize_reward",
        )
        # 记忆巩固阈值
        self._parameters["consolidation_threshold"] = ParameterSpace(
            name="consolidation_threshold",
            current_value=0.6,
            min_value=0.3,
            max_value=0.9,
            step_size=0.05,
            objective="maximize_efficiency",
        )
        # 预测置信度衰减
        self._parameters["confidence_decay"] = ParameterSpace(
            name="confidence_decay",
            current_value=0.95,
            min_value=0.8,
            max_value=0.99,
            step_size=0.01,
            objective="minimize_error",
        )

    def record_performance(self, metric_name: str, value: float) -> None:
        """记录性能指标"""
        if metric_name not in self._performance_history:
            self._performance_history[metric_name] = []
        self._performance_history[metric_name].append(value)
        # 保留最近100个数据点
        if len(self._performance_history[metric_name]) > 100:
            self._performance_history[metric_name] = self._performance_history[metric_name][-100:]

    def _compute_gradient(self, values: List[float]) -> float:
        """计算性能变化梯度（最近5个点 vs 前5个点）"""
        if len(values) < 10:
            return 0.0
        recent = sum(values[-5:]) / 5
        previous = sum(values[-10:-5]) / 5
        if previous == 0:
            return 0.0
        return (recent - previous) / abs(previous)

    def _should_increase(self, param: ParameterSpace, gradient: float) -> bool:
        """根据目标和梯度判断是否应该增加参数"""
        if param.objective == "minimize_error":
            # 误差梯度为正（误差在增加），应调整参数方向
            # 但不知道哪个方向有效，使用随机扰动试探
            return gradient > 0
        elif param.objective == "maximize_reward":
            return gradient > 0
        elif param.objective == "maximize_efficiency":
            return gradient > 0
        return False

    def tune(self, force: bool = False) -> Dict[str, Any]:
        """执行参数调优"""
        now = time.time()
        if not force and (now - self._last_tune_time) < self.tune_interval:
            return {"tuned": False, "reason": "间隔不足"}

        self._last_tune_time = now
        adjustments = {}

        for name, param in self._parameters.items():
            # 查找对应的性能指标
            metric_key = self._get_metric_for_param(name)
            if metric_key not in self._performance_history:
                continue

            history = self._performance_history[metric_key]
            if len(history) < 10:
                continue

            gradient = self._compute_gradient(history)
            if abs(gradient) < 0.01:  # 变化太小，不调整
                continue

            # 决定调整方向
            old_value = param.current_value
            should_increase = self._should_increase(param, gradient)

            # 应用调整（带安全边界）
            if should_increase:
                new_value = min(param.max_value, old_value + param.step_size)
            else:
                new_value = max(param.min_value, old_value - param.step_size)

            if new_value != old_value:
                param.current_value = new_value
                adjustments[name] = {
                    "old": old_value,
                    "new": new_value,
                    "gradient": gradient,
                    "reason": f"梯度={gradient:.4f}，{'增加' if should_increase else '减少'}参数",
                }
                logger.info("AUTO_TUNE", f"参数调整: {name}", module="auto_tune",
                            extra={"old": old_value, "new": new_value, "gradient": gradient})

        return {
            "tuned": len(adjustments) > 0,
            "adjustments": adjustments,
            "timestamp": now,
        }

    def _get_metric_for_param(self, param_name: str) -> str:
        """参数对应的性能指标"""
        mapping = {
            "prediction_window": "prediction_error",
            "exploration_ratio": "task_reward",
            "consolidation_threshold": "memory_efficiency",
            "confidence_decay": "prediction_error",
        }
        return mapping.get(param_name, "")

    def get_parameter(self, name: str) -> float:
        """获取当前参数值"""
        if name in self._parameters:
            return self._parameters[name].current_value
        return 0.0

    def set_parameter(self, name: str, value: float) -> bool:
        """手动设置参数值（受边界约束）"""
        if name not in self._parameters:
            return False
        param = self._parameters[name]
        param.current_value = max(param.min_value, min(param.max_value, value))
        return True

    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """获取所有参数状态"""
        return {
            name: {
                "current": p.current_value,
                "min": p.min_value,
                "max": p.max_value,
                "objective": p.objective,
            }
            for name, p in self._parameters.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取调参统计"""
        return {
            "parameters": self.get_all_parameters(),
            "performance_metrics": {k: len(v) for k, v in self._performance_history.items()},
            "last_tune_time": self._last_tune_time,
        }


class ExplorationBalancer:
    """探索-利用平衡器 —— 动态调整探索比例"""

    def __init__(self, initial_epsilon: float = 0.3, min_epsilon: float = 0.05,
                 max_epsilon: float = 0.5, adaptation_rate: float = 0.01):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.adaptation_rate = adaptation_rate
        self._success_history: List[bool] = []
        self._exploration_history: List[bool] = []
        logger.info("AUTO_TUNE", f"探索平衡器初始化: epsilon={initial_epsilon}", module="auto_tune")

    def should_explore(self) -> bool:
        """决定是否探索"""
        import random
        is_explore = random.random() < self.epsilon
        self._exploration_history.append(is_explore)
        if len(self._exploration_history) > 100:
            self._exploration_history = self._exploration_history[-100:]
        return is_explore

    def report_result(self, success: bool, was_exploration: bool) -> None:
        """报告动作结果，用于调整探索比例"""
        self._success_history.append(success)
        if len(self._success_history) > 100:
            self._success_history = self._success_history[-100:]

        # 如果探索动作成功率高，可以适当降低探索（已经找到好策略）
        # 如果利用动作成功率低，需要增加探索
        if len(self._success_history) >= 20:
            recent_success_rate = sum(self._success_history[-20:]) / 20
            if was_exploration:
                # 探索成功 -> 可能找到了新策略，稍微降低探索
                if success:
                    self.epsilon = max(self.min_epsilon, self.epsilon - self.adaptation_rate * 0.5)
                else:
                    # 探索失败 -> 增加探索以找到更好策略
                    self.epsilon = min(self.max_epsilon, self.epsilon + self.adaptation_rate)
            else:
                # 利用动作失败 -> 需要更多探索
                if not success:
                    self.epsilon = min(self.max_epsilon, self.epsilon + self.adaptation_rate * 2)
                else:
                    # 利用成功 -> 稍微降低探索，更多利用已知策略
                    self.epsilon = max(self.min_epsilon, self.epsilon - self.adaptation_rate * 0.3)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = len(self._success_history)
        success_rate = sum(self._success_history) / total if total > 0 else 0.0
        explore_rate = sum(self._exploration_history) / len(self._exploration_history) if self._exploration_history else 0.0
        return {
            "epsilon": self.epsilon,
            "success_rate": success_rate,
            "explore_rate": explore_rate,
            "total_actions": total,
        }
