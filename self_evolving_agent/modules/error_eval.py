"""
误差评估模块 —— 基于自由能原理和预测编码理论的预测误差全面评估

核心概念：
- 自由能原理：智能体通过最小化自由能（预测误差 + 模型复杂度）来感知和行动
- 预测编码：大脑通过预测下一时刻的感官输入，并仅传递预测误差来高效编码信息
- 惊奇（Surprise）：当预测误差远超预期时，表明模型需要更新
"""

import os
import sys
import math
import time
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IErrorEvaluator, ErrorReportKeys
from core.exceptions import ErrorEvalException


class ErrorType(Enum):
    """预测误差类型枚举"""
    PREDICTION = "prediction"
    PERCEPTION = "perception"
    ACTION = "action"
    MODEL = "model"
    EXTERNAL = "external"


@dataclass
class PredictionError:
    """预测误差数据类 —— 包含误差值、误差类型、时间戳"""
    value: float
    error_type: ErrorType
    timestamp: float
    predicted: Optional[float] = None
    actual: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        logger.debug("PREDICTION_ERROR_INIT", "PredictionError 初始化", module="error_eval",
                    extra={"value": self.value, "type": self.error_type.value, "timestamp": self.timestamp})

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("PREDICTION_ERROR_DICT", "PredictionError 转换为字典", module="error_eval")
        return {
            "value": self.value,
            "error_type": self.error_type.value,
            "timestamp": self.timestamp,
            "predicted": self.predicted,
            "actual": self.actual,
            "metadata": self.metadata,
        }


class ErrorHistory:
    """误差历史管理 —— 维护最近N次误差，计算统计指标"""

    def __init__(self, max_size: int = 1000) -> None:
        logger.info("ERROR_HISTORY_INIT", "ErrorHistory 初始化", module="error_eval",
                    extra={"max_size": max_size})
        self.max_size = max_size
        self.errors: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
        logger.debug("ERROR_HISTORY_INIT", "ErrorHistory 缓冲区创建完成", module="error_eval")

    @log_execution
    def add(self, error: PredictionError) -> None:
        logger.debug("ERROR_HISTORY_ADD", "添加误差到历史", module="error_eval",
                    extra={"value": error.value, "type": error.error_type.value})
        self.errors.append(error.value)
        self.timestamps.append(error.timestamp)
        logger.info("ERROR_HISTORY_ADD", "误差已添加到历史", module="error_eval",
                    extra={"current_size": len(self.errors), "max_size": self.max_size})

    @log_execution
    def get_recent(self, n: int = 100) -> List[float]:
        logger.debug("ERROR_HISTORY_RECENT", f"获取最近 {n} 次误差", module="error_eval")
        recent = list(self.errors)[-n:]
        logger.info("ERROR_HISTORY_RECENT", "最近误差获取完成", module="error_eval",
                    extra={"requested": n, "returned": len(recent)})
        return recent

    @log_execution
    def moving_average(self, window: int = 100) -> float:
        logger.debug("ERROR_HISTORY_MA", f"计算滑动平均, window={window}", module="error_eval")
        if len(self.errors) == 0:
            logger.warning("ERROR_HISTORY_MA", "误差历史为空，返回0", module="error_eval")
            return 0.0
        recent = list(self.errors)[-window:]
        avg = float(np.mean(recent))
        logger.info("ERROR_HISTORY_MA", "滑动平均计算完成", module="error_eval",
                    extra={"window": len(recent), "average": avg})
        return avg

    @log_execution
    def std(self, window: int = 100) -> float:
        logger.debug("ERROR_HISTORY_STD", f"计算标准差, window={window}", module="error_eval")
        if len(self.errors) < 2:
            logger.warning("ERROR_HISTORY_STD", "误差历史不足2个，返回0", module="error_eval")
            return 0.0
        recent = list(self.errors)[-window:]
        std_val = float(np.std(recent, ddof=1))
        logger.info("ERROR_HISTORY_STD", "标准差计算完成", module="error_eval",
                    extra={"window": len(recent), "std": std_val})
        return std_val

    @log_execution
    def trend(self, window: int = 100) -> float:
        """计算误差趋势：正值表示误差在增大，负值表示在减小"""
        logger.debug("ERROR_HISTORY_TREND", f"计算误差趋势, window={window}", module="error_eval")
        if len(self.errors) < 2:
            logger.warning("ERROR_HISTORY_TREND", "误差历史不足2个，返回0", module="error_eval")
            return 0.0
        recent = list(self.errors)[-window:]
        if len(recent) < 2:
            logger.warning("ERROR_HISTORY_TREND", "窗口内数据不足，返回0", module="error_eval")
            return 0.0
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = float(np.polyfit(x, y, 1)[0])
        logger.info("ERROR_HISTORY_TREND", "误差趋势计算完成", module="error_eval",
                    extra={"window": len(recent), "slope": slope})
        return slope

    @log_execution
    def is_increasing(self, window: int = 100, threshold: float = 0.0) -> bool:
        """判断误差是否持续增大"""
        logger.debug("ERROR_HISTORY_INCREASING", f"判断误差是否持续增大", module="error_eval",
                    extra={"window": window, "threshold": threshold})
        slope = self.trend(window=window)
        increasing = slope > threshold
        logger.info("ERROR_HISTORY_INCREASING", "判断完成", module="error_eval",
                    extra={"slope": slope, "increasing": increasing})
        return increasing

    @log_execution
    def entropy(self, window: int = 100, bins: int = 10) -> float:
        """计算误差分布的熵（不确定性）"""
        logger.debug("ERROR_HISTORY_ENTROPY", f"计算误差熵", module="error_eval",
                    extra={"window": window, "bins": bins})
        if len(self.errors) < 2:
            logger.warning("ERROR_HISTORY_ENTROPY", "误差历史不足，返回0", module="error_eval")
            return 0.0
        recent = np.array(list(self.errors)[-window:])
        if np.all(recent == recent[0]):
            logger.warning("ERROR_HISTORY_ENTROPY", "所有误差值相同，熵为0", module="error_eval")
            return 0.0
        hist, _ = np.histogram(recent, bins=bins, density=True)
        hist = hist[hist > 0]
        ent = float(-np.sum(hist * np.log(hist)))
        logger.info("ERROR_HISTORY_ENTROPY", "误差熵计算完成", module="error_eval",
                    extra={"entropy": ent})
        return ent

    @log_execution
    def clear(self) -> None:
        logger.debug("ERROR_HISTORY_CLEAR", "清空误差历史", module="error_eval")
        self.errors.clear()
        self.timestamps.clear()
        logger.info("ERROR_HISTORY_CLEAR", "误差历史已清空", module="error_eval")


class SurpriseDetector:
    """惊奇检测器 —— 检测异常大的预测误差"""

    def __init__(self, sigma_multiplier: float = 2.0) -> None:
        logger.info("SURPRISE_DETECTOR_INIT", "SurpriseDetector 初始化", module="error_eval",
                    extra={"sigma_multiplier": sigma_multiplier})
        self.sigma_multiplier = sigma_multiplier
        self.surprise_count = 0
        self.last_surprise_time: Optional[float] = None
        logger.debug("SURPRISE_DETECTOR_INIT", "SurpriseDetector 初始化完成", module="error_eval")

    @log_execution
    def detect(self, error_value: float, mean: float, std: float) -> bool:
        """当误差超过 mean + sigma_multiplier * std 时标记为惊奇事件"""
        logger.debug("SURPRISE_DETECT", "开始惊奇检测", module="error_eval",
                    extra={"error": error_value, "mean": mean, "std": std})
        if std <= 0:
            logger.warning("SURPRISE_DETECT", "标准差为0，无法检测惊奇", module="error_eval")
            return False
        threshold = mean + self.sigma_multiplier * std
        is_surprise = error_value > threshold
        if is_surprise:
            self.surprise_count += 1
            self.last_surprise_time = time.time()
            logger.warning("SURPRISE_DETECT", "检测到惊奇事件", module="error_eval",
                          extra={"error": error_value, "threshold": threshold, "count": self.surprise_count})
        else:
            logger.debug("SURPRISE_DETECT", "未检测到惊奇", module="error_eval",
                        extra={"error": error_value, "threshold": threshold})
        return is_surprise

    @log_execution
    def get_surprise_rate(self, total_errors: int) -> float:
        logger.debug("SURPRISE_RATE", "计算惊奇率", module="error_eval",
                    extra={"surprise_count": self.surprise_count, "total_errors": total_errors})
        if total_errors <= 0:
            logger.warning("SURPRISE_RATE", "总误差数为0，返回0", module="error_eval")
            return 0.0
        rate = self.surprise_count / total_errors
        logger.info("SURPRISE_RATE", "惊奇率计算完成", module="error_eval",
                    extra={"rate": rate})
        return rate

    @log_execution
    def reset(self) -> None:
        logger.debug("SURPRISE_RESET", "重置惊奇检测器", module="error_eval")
        self.surprise_count = 0
        self.last_surprise_time = None
        logger.info("SURPRISE_RESET", "惊奇检测器已重置", module="error_eval")


class FreeEnergyCalculator:
    """自由能计算器 —— 基于KL散度近似

    自由能近似 = 预测误差 + 模型复杂度惩罚
    其中预测误差用均方误差近似，模型复杂度惩罚用参数数量或熵近似
    """

    def __init__(self, complexity_penalty_weight: float = 0.1) -> None:
        logger.info("FE_CALCULATOR_INIT", "FreeEnergyCalculator 初始化", module="error_eval",
                    extra={"complexity_penalty_weight": complexity_penalty_weight})
        self.complexity_penalty_weight = complexity_penalty_weight
        logger.debug("FE_CALCULATOR_INIT", "FreeEnergyCalculator 初始化完成", module="error_eval")

    @log_execution
    def kl_divergence_approx(self, p_mean: float, p_std: float, q_mean: float, q_std: float) -> float:
        """计算两个高斯分布之间的KL散度（近似）"""
        logger.debug("FE_KL", "计算KL散度近似", module="error_eval",
                    extra={"p_mean": p_mean, "p_std": p_std, "q_mean": q_mean, "q_std": q_std})
        if p_std <= 0 or q_std <= 0:
            logger.warning("FE_KL", "标准差非正，返回大值", module="error_eval")
            return 1e6
        kl = math.log(q_std / p_std) + (p_std**2 + (p_mean - q_mean)**2) / (2 * q_std**2) - 0.5
        logger.info("FE_KL", "KL散度计算完成", module="error_eval",
                    extra={"kl": kl})
        return kl

    @log_execution
    def complexity_penalty(self, model_params: Optional[int] = None, entropy: float = 0.0) -> float:
        """模型复杂度惩罚"""
        logger.debug("FE_COMPLEXITY", "计算模型复杂度惩罚", module="error_eval",
                    extra={"model_params": model_params, "entropy": entropy})
        if model_params is not None:
            penalty = self.complexity_penalty_weight * math.log1p(model_params)
        else:
            penalty = self.complexity_penalty_weight * entropy
        logger.info("FE_COMPLEXITY", "复杂度惩罚计算完成", module="error_eval",
                    extra={"penalty": penalty})
        return penalty

    @log_execution
    def free_energy(self, prediction_error: float, model_params: Optional[int] = None, entropy: float = 0.0) -> float:
        """计算自由能近似值"""
        logger.debug("FE_CALCULATE", "计算自由能", module="error_eval",
                    extra={"prediction_error": prediction_error})
        penalty = self.complexity_penalty(model_params=model_params, entropy=entropy)
        fe = prediction_error + penalty
        logger.info("FE_CALCULATE", "自由能计算完成", module="error_eval",
                    extra={"free_energy": fe, "error": prediction_error, "penalty": penalty})
        return fe


class ErrorEvaluator(IErrorEvaluator):
    """误差评估器主类 —— 整合所有误差评估功能"""

    def __init__(self, history_size: int = 1000) -> None:
        logger.info("ERROR_EVALUATOR_INIT", "ErrorEvaluator 初始化", module="error_eval",
                    extra={"history_size": history_size})
        self.history = ErrorHistory(max_size=history_size)
        self.surprise_detector = SurpriseDetector(sigma_multiplier=2.0)
        self.free_energy_calculator = FreeEnergyCalculator(complexity_penalty_weight=0.1)
        self.reconstruction_signal = False
        logger.debug("ERROR_EVALUATOR_INIT", "ErrorEvaluator 子模块初始化完成", module="error_eval")

    @log_execution
    def compute_mse(self, predicted: List[float], actual: List[float]) -> float:
        """计算均方误差（MSE）"""
        logger.debug("EVAL_MSE", "开始计算MSE", module="error_eval",
                    extra={"pred_len": len(predicted), "actual_len": len(actual)})
        if len(predicted) != len(actual) or len(predicted) == 0:
            logger.error("EVAL_MSE", "预测值和真实值长度不匹配或为空", module="error_eval")
            raise ErrorEvalException("预测值和真实值长度必须相同且非空")
        mse = float(np.mean([(p - a) ** 2 for p, a in zip(predicted, actual)]))
        logger.info("EVAL_MSE", "MSE计算完成", module="error_eval",
                    extra={"mse": mse})
        return mse

    @log_execution
    def compute_mae(self, predicted: List[float], actual: List[float]) -> float:
        """计算平均绝对误差（MAE）"""
        logger.debug("EVAL_MAE", "开始计算MAE", module="error_eval",
                    extra={"pred_len": len(predicted), "actual_len": len(actual)})
        if len(predicted) != len(actual) or len(predicted) == 0:
            logger.error("EVAL_MAE", "预测值和真实值长度不匹配或为空", module="error_eval")
            raise ErrorEvalException("预测值和真实值长度必须相同且非空")
        mae = float(np.mean([abs(p - a) for p, a in zip(predicted, actual)]))
        logger.info("EVAL_MAE", "MAE计算完成", module="error_eval",
                    extra={"mae": mae})
        return mae

    @log_execution
    def compute_relative_error(self, predicted: List[float], actual: List[float], eps: float = 1e-8) -> List[float]:
        """计算相对误差（误差/真实值）"""
        logger.debug("EVAL_REL", "开始计算相对误差", module="error_eval",
                    extra={"pred_len": len(predicted), "actual_len": len(actual)})
        if len(predicted) != len(actual) or len(predicted) == 0:
            logger.error("EVAL_REL", "预测值和真实值长度不匹配或为空", module="error_eval")
            raise ErrorEvalException("预测值和真实值长度必须相同且非空")
        rel_errors = [abs(p - a) / max(abs(a), eps) for p, a in zip(predicted, actual)]
        logger.info("EVAL_REL", "相对误差计算完成", module="error_eval",
                    extra={"count": len(rel_errors), "mean_rel_error": float(np.mean(rel_errors))})
        return rel_errors

    @log_execution
    def compute_error_entropy(self, errors: List[float], bins: int = 10) -> float:
        """计算误差熵（误差的分布不确定性）"""
        logger.debug("EVAL_ENTROPY", "开始计算误差熵", module="error_eval",
                    extra={"error_count": len(errors), "bins": bins})
        if len(errors) < 2:
            logger.warning("EVAL_ENTROPY", "误差数量不足，返回0", module="error_eval")
            return 0.0
        arr = np.array(errors)
        if np.all(arr == arr[0]):
            logger.warning("EVAL_ENTROPY", "所有误差值相同，熵为0", module="error_eval")
            return 0.0
        # 自动调整bins数量，避免数据点太少
        actual_bins = min(bins, len(errors))
        if actual_bins < 2:
            logger.warning("EVAL_ENTROPY", "数据点不足，无法计算熵", module="error_eval")
            return 0.0
        try:
            hist, _ = np.histogram(arr, bins=actual_bins, density=False)
        except ValueError as e:
            logger.warning("EVAL_ENTROPY", f"直方图计算失败: {e}", module="error_eval")
            return 0.0
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum == 0:
            logger.warning("EVAL_ENTROPY", "直方图总和为0，熵为0", module="error_eval")
            return 0.0
        probs = hist / hist_sum
        probs = probs[probs > 0]
        ent = float(-np.sum(probs * np.log(probs)))
        logger.info("EVAL_ENTROPY", "误差熵计算完成", module="error_eval",
                    extra={"entropy": ent})
        return ent

    @log_execution
    def evaluate(self, predicted: Any, actual: Any,
                 error_type: ErrorType = ErrorType.PREDICTION,
                 model_params: Optional[int] = None) -> Dict[str, Any]:
        """全面评估预测误差 —— 支持字典或列表输入"""
        logger.info("EVALUATE", "开始全面误差评估", module="error_eval",
                    extra={"error_type": error_type.value, "predicted_type": type(predicted).__name__})

        # 统一转换为列表
        if isinstance(predicted, dict) and isinstance(actual, dict):
            logger.debug("EVALUATE", "输入为字典，转换为列表", module="error_eval")
            keys = sorted(set(predicted.keys()) & set(actual.keys()))
            predicted_list = [predicted[k] for k in keys]
            actual_list = [actual[k] for k in keys]
            logger.debug("EVALUATE", f"字典转换完成，keys={keys}", module="error_eval")
        elif isinstance(predicted, list) and isinstance(actual, list):
            predicted_list = predicted
            actual_list = actual
        else:
            logger.error("EVALUATE", "输入类型不匹配", module="error_eval",
                        extra={"predicted_type": type(predicted).__name__, "actual_type": type(actual).__name__})
            raise TypeError("predicted 和 actual 必须同为字典或同为列表")

        if len(predicted_list) == 0 or len(actual_list) == 0:
            logger.warning("EVALUATE", "输入为空，返回默认结果", module="error_eval")
            return {
                ErrorReportKeys.MSE: 0.0, ErrorReportKeys.MAE: 0.0, ErrorReportKeys.MEAN_RELATIVE_ERROR: 0.0,
                ErrorReportKeys.ERROR_ENTROPY: 0.0, ErrorReportKeys.FREE_ENERGY: 0.0, ErrorReportKeys.MEAN_ERROR: 0.0,
                ErrorReportKeys.STD_ERROR: 0.0, ErrorReportKeys.SURPRISE_COUNT: 0, ErrorReportKeys.SURPRISE_RATE: 0.0,
                ErrorReportKeys.MOVING_AVERAGE: 0.0, ErrorReportKeys.HISTORY_STD: 0.0, ErrorReportKeys.TREND: 0.0,
                ErrorReportKeys.RECONSTRUCTION_SIGNAL: False, ErrorReportKeys.SAMPLE_COUNT: 0,
                ErrorReportKeys.TIMESTAMP: time.time(),
            }

        # 基础指标
        mse = self.compute_mse(predicted_list, actual_list)
        logger.debug("EVALUATE", "MSE计算完成", module="error_eval", extra={"mse": mse})
        mae = self.compute_mae(predicted_list, actual_list)
        logger.debug("EVALUATE", "MAE计算完成", module="error_eval", extra={"mae": mae})
        rel_errors = self.compute_relative_error(predicted_list, actual_list)
        logger.debug("EVALUATE", "相对误差计算完成", module="error_eval", extra={"rel_errors_count": len(rel_errors)})
        mean_rel_error = float(np.mean(rel_errors))
        logger.debug("EVALUATE", "平均相对误差计算完成", module="error_eval", extra={"mean_rel_error": mean_rel_error})

        # 逐点误差
        point_errors = [abs(p - a) for p, a in zip(predicted_list, actual_list)]
        logger.debug("EVALUATE", "逐点误差计算完成", module="error_eval", extra={"point_errors_count": len(point_errors)})

        # 误差熵
        error_entropy = self.compute_error_entropy(point_errors)
        logger.debug("EVALUATE", "误差熵计算完成", module="error_eval", extra={"error_entropy": error_entropy})

        # 添加到历史
        current_time = time.time()
        for err, pred, act in zip(point_errors, predicted_list, actual_list):
            pred_error = PredictionError(
                value=err,
                error_type=error_type,
                timestamp=current_time,
                predicted=pred,
                actual=act,
            )
            self.history.add(pred_error)
        logger.debug("EVALUATE", "误差已添加到历史", module="error_eval", extra={"history_size": len(self.history.errors)})

        # 自由能
        mean_error = float(np.mean(point_errors))
        logger.debug("EVALUATE", "平均误差计算完成", module="error_eval", extra={"mean_error": mean_error})
        free_energy = self.free_energy_calculator.free_energy(
            prediction_error=mean_error,
            model_params=model_params,
            entropy=error_entropy,
        )
        logger.debug("EVALUATE", "自由能计算完成", module="error_eval", extra={"free_energy": free_energy})

        # 惊奇检测
        std_error = float(np.std(point_errors, ddof=1)) if len(point_errors) > 1 else 0.0
        logger.debug("EVALUATE", "误差标准差计算完成", module="error_eval", extra={"std_error": std_error})
        surprises = []
        for err in point_errors:
            is_surprise = self.surprise_detector.detect(err, mean_error, std_error)
            surprises.append(is_surprise)
        surprise_count = sum(surprises)
        logger.debug("EVALUATE", "惊奇检测完成", module="error_eval", extra={"surprise_count": surprise_count})

        # 历史统计
        moving_avg = self.history.moving_average(window=min(100, len(self.history.errors)))
        logger.debug("EVALUATE", "滑动平均计算完成", module="error_eval", extra={"moving_avg": moving_avg})
        history_std = self.history.std(window=min(100, len(self.history.errors)))
        logger.debug("EVALUATE", "历史标准差计算完成", module="error_eval", extra={"history_std": history_std})
        trend = self.history.trend(window=min(100, len(self.history.errors)))
        logger.debug("EVALUATE", "误差趋势计算完成", module="error_eval", extra={"trend": trend})

        # 重构信号：误差持续增大
        self.reconstruction_signal = self.history.is_increasing(window=min(100, len(self.history.errors)), threshold=0.0)
        if self.reconstruction_signal:
            logger.warning("EVALUATE", "误差持续增大，触发模型重构信号", module="error_eval",
                          extra={"trend": trend, "moving_avg": moving_avg})
        else:
            logger.debug("EVALUATE", "误差未持续增大，无需重构", module="error_eval")

        result = {
            ErrorReportKeys.MSE: mse,
            ErrorReportKeys.MAE: mae,
            ErrorReportKeys.MEAN_RELATIVE_ERROR: mean_rel_error,
            ErrorReportKeys.ERROR_ENTROPY: error_entropy,
            ErrorReportKeys.FREE_ENERGY: free_energy,
            ErrorReportKeys.MEAN_ERROR: mean_error,
            ErrorReportKeys.STD_ERROR: std_error,
            ErrorReportKeys.SURPRISE_COUNT: int(surprise_count),
            ErrorReportKeys.SURPRISE_RATE: float(surprise_count / len(point_errors)) if point_errors else 0.0,
            ErrorReportKeys.MOVING_AVERAGE: moving_avg,
            ErrorReportKeys.HISTORY_STD: history_std,
            ErrorReportKeys.TREND: trend,
            ErrorReportKeys.RECONSTRUCTION_SIGNAL: self.reconstruction_signal,
            ErrorReportKeys.SAMPLE_COUNT: len(predicted_list),
            ErrorReportKeys.TIMESTAMP: current_time,
        }

        logger.info("EVALUATE", "全面误差评估完成", module="error_eval",
                    extra={"mse": mse, "mae": mae, "free_energy": free_energy,
                           "reconstruction_signal": self.reconstruction_signal})
        return result

    @log_execution
    def get_history_stats(self) -> Dict[str, Any]:
        """获取误差历史统计信息"""
        logger.debug("EVAL_HISTORY_STATS", "获取误差历史统计", module="error_eval")
        size = len(self.history.errors)
        stats = {
            "history_size": size,
            "moving_average": self.history.moving_average() if size > 0 else 0.0,
            "std": self.history.std() if size > 1 else 0.0,
            "trend": self.history.trend() if size > 1 else 0.0,
            "entropy": self.history.entropy() if size > 1 else 0.0,
        }
        logger.info("EVAL_HISTORY_STATS", "误差历史统计完成", module="error_eval", extra=stats)
        return stats

    @log_execution
    def reset(self) -> None:
        """重置评估器状态"""
        logger.debug("EVAL_RESET", "重置ErrorEvaluator", module="error_eval")
        self.history.clear()
        self.surprise_detector.reset()
        self.reconstruction_signal = False
        logger.info("EVAL_RESET", "ErrorEvaluator 已重置", module="error_eval")
