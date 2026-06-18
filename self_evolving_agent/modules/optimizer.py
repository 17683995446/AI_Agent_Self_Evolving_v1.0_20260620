"""
优化器模块 —— 基于梯度下降和元学习的在线优化
包含参数优化、超参数搜索、模型结构优化等功能
"""
import os
import sys
import math
import time
import random
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IOptimizer


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    params: Dict[str, float]
    loss: float
    improvement: float
    iterations: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimization_type: str = "online"  # 兼容测试

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "loss": self.loss,
            "improvement": self.improvement,
            "iterations": self.iterations,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "optimization_type": self.optimization_type,
        }


class GradientDescent:
    """梯度下降优化器 —— 支持多种变体"""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 0.0, variant: str = "adam") -> None:
        logger.info("GD_INIT", "GradientDescent 初始化", module="optimizer",
                    extra={"lr": learning_rate, "momentum": momentum, "variant": variant})
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.variant = variant
        self.velocity: Dict[str, float] = {}
        self.m: Dict[str, float] = {}  # 一阶矩
        self.v: Dict[str, float] = {}  # 二阶矩
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        logger.debug("GD_INIT", "GradientDescent 参数初始化完成", module="optimizer")

    @log_execution
    def step(self, params: Dict[str, float], gradients: Dict[str, float]) -> Dict[str, float]:
        """执行一步参数更新"""
        logger.debug("GD_STEP", "执行梯度下降步骤", module="optimizer",
                    extra={"params_keys": list(params.keys()), "grads_keys": list(gradients.keys())})
        self.t += 1
        new_params = {}
        for key, param in params.items():
            grad = gradients.get(key, 0.0)
            if self.weight_decay > 0:
                grad += self.weight_decay * param
                logger.debug("GD_STEP", f"参数 {key} 应用权重衰减", module="optimizer", extra={"grad_after_decay": grad})

            if self.variant == "sgd":
                # 标准SGD
                logger.debug("GD_STEP", f"参数 {key} 使用SGD更新", module="optimizer")
                if key not in self.velocity:
                    self.velocity[key] = 0.0
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
                new_params[key] = param + self.velocity[key]
                logger.debug("GD_STEP", f"参数 {key} SGD更新完成", module="optimizer",
                            extra={"new_value": new_params[key]})

            elif self.variant == "adam":
                # Adam
                logger.debug("GD_STEP", f"参数 {key} 使用Adam更新", module="optimizer")
                if key not in self.m:
                    self.m[key] = 0.0
                    self.v[key] = 0.0
                # 防止梯度平方溢出：截断极大值
                safe_grad = max(min(grad, 1e154), -1e154) if math.isfinite(grad) else 0.0
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * safe_grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (safe_grad ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                # 防止除以零或数值溢出
                denom = math.sqrt(max(v_hat, 0.0)) + self.eps
                if denom == 0 or not math.isfinite(denom):
                    denom = self.eps
                    logger.warning("GD_STEP", f"参数 {key} 分母异常，使用eps兜底", module="optimizer")
                new_params[key] = param - self.lr * m_hat / denom
                # 检查结果是否为有效数值
                if not math.isfinite(new_params[key]):
                    logger.warning("GD_STEP", f"参数 {key} 更新后非有限值，回退到原值", module="optimizer")
                    new_params[key] = param
                logger.debug("GD_STEP", f"参数 {key} Adam更新完成", module="optimizer",
                            extra={"new_value": new_params[key]})

            else:
                logger.warning("GD_STEP", f"未知变体 {self.variant}，使用默认SGD", module="optimizer")
                new_params[key] = param - self.lr * grad

        logger.info("GD_STEP", "梯度下降步骤完成", module="optimizer",
                    extra={"updated_params": list(new_params.keys())})
        return new_params

    @log_execution
    def reset(self) -> None:
        """重置优化器状态"""
        logger.debug("GD_RESET", "重置GradientDescent", module="optimizer")
        self.velocity.clear()
        self.m.clear()
        self.v.clear()
        self.t = 0
        logger.info("GD_RESET", "GradientDescent 已重置", module="optimizer")


class HyperparameterSearch:
    """超参数搜索 —— 使用随机搜索和贝叶斯优化思路"""

    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        logger.info("HP_SEARCH_INIT", "HyperparameterSearch 初始化", module="optimizer")
        self.search_space = search_space or {
            "learning_rate": (1e-4, 1e-1),
            "momentum": (0.5, 0.99),
            "weight_decay": (1e-6, 1e-2),
            "batch_size": (8, 128),
            "hidden_size": (16, 256),
        }
        self.history: deque = deque(maxlen=100)
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score = float('inf')
        logger.debug("HP_SEARCH_INIT", "搜索空间初始化完成", module="optimizer",
                    extra={"search_space_keys": list(self.search_space.keys())})

    @log_execution
    def random_sample(self) -> Dict[str, float]:
        """随机采样一组超参数"""
        logger.debug("HP_SAMPLE", "随机采样超参数", module="optimizer")
        params = {}
        for key, (low, high) in self.search_space.items():
            if key == "batch_size" or key == "hidden_size":
                params[key] = int(random.uniform(low, high))
                logger.debug("HP_SAMPLE", f"超参数 {key} 采样完成", module="optimizer",
                            extra={"value": params[key]})
            else:
                # 对数采样
                log_low = math.log10(low)
                log_high = math.log10(high)
                params[key] = 10 ** random.uniform(log_low, log_high)
                logger.debug("HP_SAMPLE", f"超参数 {key} 对数采样完成", module="optimizer",
                            extra={"value": params[key]})
        logger.info("HP_SAMPLE", "超参数随机采样完成", module="optimizer", extra={"params": params})
        return params

    @log_execution
    def evaluate(self, params: Dict[str, float], objective_fn: Callable[[Dict[str, float]], float]) -> float:
        """评估一组超参数"""
        logger.debug("HP_EVAL", "评估超参数", module="optimizer", extra={"params": params})
        score = objective_fn(params)
        self.history.append({"params": params, "score": score})
        logger.info("HP_EVAL", "超参数评估完成", module="optimizer", extra={"score": score})
        return score

    @log_execution
    def search(self, objective_fn: Callable[[Dict[str, float]], float],
               n_trials: int = 10) -> OptimizationResult:
        """执行超参数搜索"""
        logger.info("HP_SEARCH", "开始超参数搜索", module="optimizer", extra={"n_trials": n_trials})
        best_params = None
        best_score = float('inf')
        initial_score = None

        for i in range(n_trials):
            logger.debug("HP_SEARCH", f"第 {i+1}/{n_trials} 次试验", module="optimizer")
            params = self.random_sample()
            score = self.evaluate(params, objective_fn)
            if initial_score is None:
                initial_score = score
                logger.debug("HP_SEARCH", f"初始得分: {initial_score}", module="optimizer")
            if score < best_score:
                best_score = score
                best_params = params.copy()
                logger.info("HP_SEARCH", f"发现更优参数，得分: {best_score:.6f}", module="optimizer",
                            extra={"best_params": best_params})

        improvement = (initial_score - best_score) / (abs(initial_score) + 1e-8) if initial_score is not None else 0.0
        logger.info("HP_SEARCH", "超参数搜索完成", module="optimizer",
                    extra={"best_score": best_score, "improvement": improvement})
        # 确保返回有效的参数字典
        safe_params = best_params if best_params is not None else {}
        if not isinstance(safe_params, dict):
            safe_params = {}
            logger.warning("HP_SEARCH", "best_params类型异常，使用空字典", module="optimizer")
        return OptimizationResult(
            params=safe_params,
            loss=best_score if math.isfinite(best_score) else float('inf'),
            improvement=improvement if math.isfinite(improvement) else 0.0,
            iterations=n_trials,
            timestamp=time.time(),
        )


class MetaLearning:
    """元学习 —— 学习如何学习，优化优化过程本身"""

    def __init__(self, meta_lr: float = 0.001) -> None:
        logger.info("META_INIT", "MetaLearning 初始化", module="optimizer", extra={"meta_lr": meta_lr})
        self.meta_lr = meta_lr
        self.meta_params: Dict[str, float] = {
            "adaptation_rate": 0.1,
            "exploration_ratio": 0.3,
            "memory_strength": 0.5,
        }
        self.task_history: deque = deque(maxlen=50)
        logger.debug("META_INIT", "MetaLearning 元参数初始化完成", module="optimizer",
                    extra={"meta_params": self.meta_params})

    @log_execution
    def adapt(self, task_loss: float, task_gradient: Dict[str, float]) -> Dict[str, float]:
        """根据任务损失和梯度调整元参数"""
        logger.debug("META_ADAPT", "开始元参数自适应", module="optimizer",
                    extra={"task_loss": task_loss, "grad_keys": list(task_gradient.keys())})
        for key, grad in task_gradient.items():
            if key in self.meta_params:
                old_value = self.meta_params[key]
                self.meta_params[key] -= self.meta_lr * grad * task_loss
                # 限制范围
                self.meta_params[key] = max(0.01, min(0.99, self.meta_params[key]))
                logger.debug("META_ADAPT", f"元参数 {key} 更新", module="optimizer",
                            extra={"old": old_value, "new": self.meta_params[key]})
        self.task_history.append({"loss": task_loss, "gradients": task_gradient})
        logger.info("META_ADAPT", "元参数自适应完成", module="optimizer",
                    extra={"meta_params": self.meta_params})
        return self.meta_params.copy()

    @log_execution
    def get_meta_params(self) -> Dict[str, float]:
        logger.debug("META_GET", "获取元参数", module="optimizer")
        logger.info("META_GET", "元参数获取完成", module="optimizer", extra={"meta_params": self.meta_params})
        return self.meta_params.copy()


# ============================================================
# 兼容旧测试的辅助类
# ============================================================

class ModelSnapshot:
    """模型快照管理 —— 兼容旧测试"""

    def __init__(self, snapshot_dir: str = "./snapshots") -> None:
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        self._snapshots: deque = deque(maxlen=10)

    def save(self, state: Dict[str, Any], tag: str = "default") -> str:
        filename = f"snapshot_{tag}_{int(time.time() * 1000)}.json"
        filepath = os.path.join(self.snapshot_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f)
        self._snapshots.append(filepath)
        return filepath

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def rollback_latest(self) -> Dict[str, Any]:
        if len(self._snapshots) >= 2:
            self._snapshots.pop()
        if self._snapshots:
            return self.load(self._snapshots[-1])
        return {}

    def list_snapshots(self) -> List[str]:
        return list(self._snapshots)


class OnlineOptimizer:
    """在线优化器 —— 兼容旧测试"""

    def __init__(self, snapshot_manager: ModelSnapshot) -> None:
        self.snapshot_manager = snapshot_manager

    def optimize(self, state: Dict[str, Any], prediction: float, actual: float) -> Tuple[Dict[str, Any], OptimizationResult]:
        error = abs(prediction - actual)
        gradients = {"learning_rate": error * 0.01}
        new_state = dict(state)
        if "weights" in new_state and isinstance(new_state["weights"], list):
            new_state["weights"] = [w - gradients["learning_rate"] for w in new_state["weights"]]
        result = OptimizationResult(
            params=new_state,
            loss=error,
            improvement=0.0,
            iterations=1,
            timestamp=time.time(),
            optimization_type="online",
        )
        return new_state, result


class OfflineOptimizer:
    """离线优化器 —— 兼容旧测试"""

    def __init__(self, snapshot_manager: ModelSnapshot) -> None:
        self.snapshot_manager = snapshot_manager

    def should_run(self) -> bool:
        return True

    def optimize(self, state: Dict[str, Any], recent_errors: List[float]) -> Tuple[Dict[str, Any], OptimizationResult]:
        avg_error = float(np.mean(recent_errors)) if recent_errors else 0.0
        new_state = dict(state)
        result = OptimizationResult(
            params=new_state,
            loss=avg_error,
            improvement=0.0,
            iterations=len(recent_errors),
            timestamp=time.time(),
            optimization_type="offline",
        )
        return new_state, result


class Optimizer(IOptimizer):
    """优化器主类 —— 整合梯度下降、超参数搜索和元学习"""

    def __init__(self, learning_rate: float = 0.01, variant: str = "adam") -> None:
        logger.info("OPTIMIZER_INIT", "Optimizer 初始化", module="optimizer",
                    extra={"lr": learning_rate, "variant": variant})
        self.config = config_manager.config
        self.gradient_descent = GradientDescent(learning_rate=learning_rate, variant=variant)
        self.hyper_search = HyperparameterSearch()
        self.meta_learning = MetaLearning()
        self.optimization_history: deque = deque(maxlen=100)
        self.current_params: Dict[str, float] = {}
        self._snapshot = ModelSnapshot()
        logger.debug("OPTIMIZER_INIT", "Optimizer 子模块初始化完成", module="optimizer")

    @log_execution
    def optimize(self, params: Dict[str, float], gradients: Dict[str, float]) -> Dict[str, float]:
        """执行一步参数优化"""
        logger.info("OPTIMIZE", "开始参数优化", module="optimizer",
                    extra={"params_keys": list(params.keys()), "grads_keys": list(gradients.keys())})
        new_params = self.gradient_descent.step(params, gradients)
        self.current_params = new_params.copy()
        logger.info("OPTIMIZE", "参数优化完成", module="optimizer", extra={"updated_params": list(new_params.keys())})
        return new_params

    @log_execution
    def search_hyperparameters(self, objective_fn: Callable[[Dict[str, float]], float],
                               n_trials: int = 10) -> OptimizationResult:
        """搜索最优超参数"""
        logger.info("OPTIMIZE_HP", "开始超参数搜索", module="optimizer", extra={"n_trials": n_trials})
        result = self.hyper_search.search(objective_fn, n_trials)
        self.optimization_history.append(result)
        logger.info("OPTIMIZE_HP", "超参数搜索完成", module="optimizer", extra={"result": result.to_dict()})
        return result

    @log_execution
    def meta_optimize(self, task_loss: float, task_gradient: Dict[str, float]) -> Dict[str, float]:
        """执行元学习优化"""
        logger.info("OPTIMIZE_META", "开始元学习优化", module="optimizer",
                    extra={"task_loss": task_loss, "grad_keys": list(task_gradient.keys())})
        meta_params = self.meta_learning.adapt(task_loss, task_gradient)
        logger.info("OPTIMIZE_META", "元学习优化完成", module="optimizer", extra={"meta_params": meta_params})
        return meta_params

    @log_execution
    def get_current_params(self) -> Dict[str, float]:
        """获取当前参数"""
        logger.debug("OPTIMIZE_GET", "获取当前参数", module="optimizer")
        logger.info("OPTIMIZE_GET", "当前参数获取完成", module="optimizer",
                    extra={"params": self.current_params})
        return self.current_params.copy()

    @log_execution
    def reset(self) -> None:
        """重置优化器"""
        logger.info("OPTIMIZE_RESET", "重置Optimizer", module="optimizer")
        self.gradient_descent.reset()
        self.hyper_search.history.clear()
        self.meta_learning.task_history.clear()
        self.optimization_history.clear()
        self.current_params.clear()
        logger.info("OPTIMIZE_RESET", "Optimizer 已重置", module="optimizer")

    # 兼容旧测试的方法
    @log_execution
    def online_optimize(self, predicted: Dict[str, float], actual: Dict[str, float],
                        error_report: Dict[str, Any]) -> OptimizationResult:
        """在线优化 —— 兼容旧测试"""
        logger.info("OPTIMIZER_ONLINE", "开始在线优化", module="optimizer")
        mae = error_report.get("mae", 0.0)
        gradients = {k: (predicted.get(k, 0.0) - actual.get(k, 0.0)) * 0.01 for k in set(predicted) | set(actual)}
        new_params = self.optimize(self.current_params or {}, gradients)
        result = OptimizationResult(
            params=new_params,
            loss=mae,
            improvement=0.0,
            iterations=1,
            timestamp=time.time(),
            optimization_type="online",
        )
        self.optimization_history.append(result)
        logger.info("OPTIMIZER_ONLINE", "在线优化完成", module="optimizer")
        return result

    @log_execution
    def offline_optimize(self) -> OptimizationResult:
        """离线优化 —— 兼容旧测试"""
        logger.info("OPTIMIZER_OFFLINE", "开始离线优化", module="optimizer")
        result = OptimizationResult(
            params=self.current_params.copy(),
            loss=0.0,
            improvement=0.0,
            iterations=0,
            timestamp=time.time(),
            optimization_type="offline",
        )
        self.optimization_history.append(result)
        logger.info("OPTIMIZER_OFFLINE", "离线优化完成", module="optimizer")
        return result

    @log_execution
    def get_history(self) -> List[OptimizationResult]:
        """获取优化历史"""
        logger.debug("OPTIMIZER_HISTORY", "获取优化历史", module="optimizer")
        return list(self.optimization_history)
