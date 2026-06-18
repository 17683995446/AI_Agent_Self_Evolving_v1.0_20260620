"""
多层预测模块 —— 实现三层预测模型：底层、中层、高层
用于预测系统状态（CPU/内存/磁盘）并给出策略建议
"""
import sys
import os
import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IPrediction, MetricKeys


@dataclass
class PredictionResult:
    """预测结果数据类 —— 包含预测值、置信度、预测层、时间戳"""
    predicted_values: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    layer: str = ""
    timestamp: float = field(default_factory=time.time)
    strategy_advice: Optional[str] = None
    trend_direction: Optional[str] = None
    error_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        logger.debug("PREDICTION_RESULT_INIT", "PredictionResult 初始化完成", module="prediction",
                    extra={"layer": self.layer, "confidence": self.confidence})


class BaseLayerPredictor(IPrediction):
    """底层预测器 —— 基于最近N个状态向量的滑动平均 + 简单线性回归，预测下一时刻的CPU/内存/磁盘"""

    def __init__(self, window_size: int = 10) -> None:
        logger.info("BASE_LAYER_INIT", "底层预测器初始化", module="prediction",
                    extra={"window_size": window_size})
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.metrics_keys = [MetricKeys.CPU, MetricKeys.MEMORY, MetricKeys.DISK]
        self.accuracy_history: deque = deque(maxlen=50)
        self.confidence = 0.5
        logger.debug("BASE_LAYER_INIT", "历史队列和指标键初始化完成", module="prediction",
                    extra={"metrics_keys": self.metrics_keys})

    @log_execution
    def update(self, state: Dict[str, float]) -> None:
        """更新历史状态"""
        logger.debug("BASE_LAYER_UPDATE", "开始更新历史状态", module="prediction",
                    extra={"state_keys": list(state.keys())})
        filtered = {k: state.get(k, 0.0) for k in self.metrics_keys}
        logger.debug("BASE_LAYER_UPDATE", "状态过滤完成", module="prediction",
                    extra={"filtered": filtered})
        self.history.append(filtered)
        logger.info("BASE_LAYER_UPDATE", "历史状态更新完成", module="prediction",
                    extra={"history_len": len(self.history)})

    @log_execution
    def predict(self) -> PredictionResult:
        """基于滑动平均 + 简单线性回归预测下一时刻状态"""
        logger.debug("BASE_LAYER_PREDICT", "开始底层预测", module="prediction",
                    extra={"history_len": len(self.history)})
        if len(self.history) < 2:
            logger.warning("BASE_LAYER_PREDICT", "历史数据不足，返回默认值", module="prediction",
                          extra={"history_len": len(self.history)})
            result = PredictionResult(
                predicted_values={k: 0.0 for k in self.metrics_keys},
                confidence=0.0,
                layer="base",
                trend_direction="unknown"
            )
            logger.info("BASE_LAYER_PREDICT", "底层预测完成（数据不足）", module="prediction")
            return result

        # 滑动平均预测
        logger.debug("BASE_LAYER_PREDICT", "计算滑动平均", module="prediction")
        avg_values = {}
        for k in self.metrics_keys:
            values = [h[k] for h in self.history]
            avg = sum(values) / len(values)
            avg_values[k] = avg
            logger.debug("BASE_LAYER_PREDICT", f"指标 {k} 滑动平均计算完成", module="prediction",
                        extra={"metric": k, "avg": avg})

        # 简单线性回归（最小二乘法）
        logger.debug("BASE_LAYER_PREDICT", "开始线性回归计算", module="prediction")
        n = len(self.history)
        x = list(range(n))
        lr_values = {}
        overall_slope = 0.0
        for k in self.metrics_keys:
            y = [h[k] for h in self.history]
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            logger.debug("BASE_LAYER_PREDICT", f"指标 {k} 均值计算完成", module="prediction",
                        extra={"x_mean": x_mean, "y_mean": y_mean})
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            denominator = sum((xi - x_mean) ** 2 for xi in x)
            if denominator == 0:
                slope = 0.0
                logger.warning("BASE_LAYER_PREDICT", f"指标 {k} 分母为零，斜率置零", module="prediction")
            else:
                slope = numerator / denominator
                logger.debug("BASE_LAYER_PREDICT", f"指标 {k} 斜率计算完成", module="prediction",
                            extra={"slope": slope})
            intercept = y_mean - slope * x_mean
            next_val = slope * n + intercept
            # 结合滑动平均和线性回归
            blended = 0.6 * avg_values[k] + 0.4 * next_val
            lr_values[k] = max(0.0, min(100.0, blended))
            logger.debug("BASE_LAYER_PREDICT", f"指标 {k} 预测值混合完成", module="prediction",
                        extra={"avg": avg_values[k], "lr": next_val, "blended": lr_values[k]})
            overall_slope += slope
        overall_slope = overall_slope / len(self.metrics_keys) if self.metrics_keys else 0.0

        # 动态调整置信度
        logger.debug("BASE_LAYER_PREDICT", "调整置信度", module="prediction",
                    extra={"accuracy_history_len": len(self.accuracy_history)})
        if len(self.accuracy_history) > 0:
            recent_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
            self.confidence = 0.3 + 0.7 * recent_accuracy
            logger.info("BASE_LAYER_PREDICT", "置信度基于历史准确率更新", module="prediction",
                        extra={"recent_accuracy": recent_accuracy, "confidence": self.confidence})
        else:
            self.confidence = 0.5
            logger.debug("BASE_LAYER_PREDICT", "无历史准确率，置信度设为默认值", module="prediction",
                        extra={"confidence": self.confidence})

        result = PredictionResult(
            predicted_values=lr_values,
            confidence=self.confidence,
            layer="base",
            trend_direction="stable" if abs(overall_slope) < 0.01 else ("up" if overall_slope > 0 else "down")
        )
        logger.info("BASE_LAYER_PREDICT", "底层预测完成", module="prediction",
                    extra={"predicted_values": lr_values, "confidence": self.confidence})
        return result

    @log_execution
    def online_learn(self, actual: Dict[str, float]) -> None:
        """在线学习：根据真实值更新模型参数"""
        logger.debug("BASE_LAYER_LEARN", "开始在线学习", module="prediction",
                    extra={"actual": actual})
        if len(self.history) == 0:
            logger.warning("BASE_LAYER_LEARN", "历史为空，跳过在线学习", module="prediction")
            return
        last_predicted = self.predict()
        logger.debug("BASE_LAYER_LEARN", "获取上次预测值", module="prediction",
                    extra={"last_predicted": last_predicted.predicted_values})
        errors = []
        for k in self.metrics_keys:
            pred = last_predicted.predicted_values.get(k, 0.0)
            act = actual.get(k, 0.0)
            error = abs(pred - act) / 100.0
            errors.append(1.0 - min(error, 1.0))
            logger.debug("BASE_LAYER_LEARN", f"指标 {k} 误差计算完成", module="prediction",
                        extra={"pred": pred, "act": act, "error": error})
        avg_accuracy = sum(errors) / len(errors)
        self.accuracy_history.append(avg_accuracy)
        logger.info("BASE_LAYER_LEARN", "在线学习完成", module="prediction",
                    extra={"avg_accuracy": avg_accuracy, "accuracy_history_len": len(self.accuracy_history)})


class MidLayerPredictor(IPrediction):
    """中层预测器 —— 趋势检测 + 变化点检测，预测未来K步的状态变化方向"""

    def __init__(self, lookback: int = 20, future_steps: int = 5) -> None:
        logger.info("MID_LAYER_INIT", "中层预测器初始化", module="prediction",
                    extra={"lookback": lookback, "future_steps": future_steps})
        self.lookback = lookback
        self.future_steps = future_steps
        self.history: deque = deque(maxlen=lookback)
        self.metrics_keys = [MetricKeys.CPU, MetricKeys.MEMORY, MetricKeys.DISK]
        self.accuracy_history: deque = deque(maxlen=50)
        self.confidence = 0.5
        self.change_points: List[int] = []
        logger.debug("MID_LAYER_INIT", "中层预测器参数初始化完成", module="prediction")

    @log_execution
    def update(self, state: Dict[str, float]) -> None:
        """更新历史状态"""
        logger.debug("MID_LAYER_UPDATE", "开始更新历史状态", module="prediction",
                    extra={"state_keys": list(state.keys())})
        filtered = {k: state.get(k, 0.0) for k in self.metrics_keys}
        self.history.append(filtered)
        logger.info("MID_LAYER_UPDATE", "历史状态更新完成", module="prediction",
                    extra={"history_len": len(self.history)})

    @log_execution
    def detect_trend(self, values: List[float]) -> str:
        """检测趋势：上升/下降/平稳"""
        logger.debug("MID_LAYER_TREND", "开始趋势检测", module="prediction",
                    extra={"values_len": len(values)})
        if len(values) < 3:
            logger.warning("MID_LAYER_TREND", "数据不足，返回平稳", module="prediction")
            return "stable"
        # 使用简单斜率判断趋势
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        if denominator == 0:
            slope = 0.0
            logger.warning("MID_LAYER_TREND", "分母为零，斜率置零", module="prediction")
        else:
            slope = numerator / denominator
            logger.debug("MID_LAYER_TREND", "斜率计算完成", module="prediction",
                        extra={"slope": slope})
        threshold = 0.05
        if slope > threshold:
            trend = "up"
        elif slope < -threshold:
            trend = "down"
        else:
            trend = "stable"
        logger.info("MID_LAYER_TREND", "趋势检测完成", module="prediction",
                    extra={"trend": trend, "slope": slope})
        return trend

    @log_execution
    def detect_change_points(self, values: List[float]) -> List[int]:
        """检测变化点（使用简单差分阈值法）"""
        logger.debug("MID_LAYER_CHANGE", "开始变化点检测", module="prediction",
                    extra={"values_len": len(values)})
        if len(values) < 3:
            logger.warning("MID_LAYER_CHANGE", "数据不足，返回空变化点", module="prediction")
            return []
        changes = []
        threshold = np.std(values) * 0.8 if len(values) > 1 else 5.0
        logger.debug("MID_LAYER_CHANGE", "变化点阈值计算完成", module="prediction",
                    extra={"threshold": threshold})
        for i in range(1, len(values)):
            diff = abs(values[i] - values[i - 1])
            if diff > threshold:
                changes.append(i)
                logger.debug("MID_LAYER_CHANGE", f"发现变化点", module="prediction",
                            extra={"index": i, "diff": diff})
        self.change_points = changes
        logger.info("MID_LAYER_CHANGE", "变化点检测完成", module="prediction",
                    extra={"change_points_count": len(changes)})
        return changes

    @log_execution
    def predict(self) -> PredictionResult:
        """预测未来K步的状态变化方向"""
        logger.debug("MID_LAYER_PREDICT", "开始中层预测", module="prediction",
                    extra={"history_len": len(self.history)})
        if len(self.history) < 5:
            logger.warning("MID_LAYER_PREDICT", "历史数据不足，返回默认值", module="prediction")
            return PredictionResult(
                predicted_values={k: 0.0 for k in self.metrics_keys},
                confidence=0.0,
                layer="mid",
                trend_direction="unknown"
            )

        trends = {}
        change_points_map = {}
        for k in self.metrics_keys:
            values = [h[k] for h in self.history]
            trend = self.detect_trend(values)
            trends[k] = trend
            logger.debug("MID_LAYER_PREDICT", f"指标 {k} 趋势检测完成", module="prediction",
                        extra={"trend": trend})
            cps = self.detect_change_points(values)
            change_points_map[k] = cps
            logger.debug("MID_LAYER_PREDICT", f"指标 {k} 变化点检测完成", module="prediction",
                        extra={"change_points": cps})

        # 基于趋势外推未来K步
        logger.debug("MID_LAYER_PREDICT", "开始趋势外推", module="prediction",
                    extra={"future_steps": self.future_steps})
        predicted = {}
        for k in self.metrics_keys:
            values = [h[k] for h in self.history]
            last_val = values[-1]
            trend = trends[k]
            if trend == "up":
                delta = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0
                predicted[k] = last_val + delta * self.future_steps
                logger.debug("MID_LAYER_PREDICT", f"指标 {k} 上升趋势外推", module="prediction",
                            extra={"delta": delta, "predicted": predicted[k]})
            elif trend == "down":
                delta = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0
                predicted[k] = last_val + delta * self.future_steps
                logger.debug("MID_LAYER_PREDICT", f"指标 {k} 下降趋势外推", module="prediction",
                            extra={"delta": delta, "predicted": predicted[k]})
            else:
                predicted[k] = last_val
                logger.debug("MID_LAYER_PREDICT", f"指标 {k} 平稳趋势外推", module="prediction",
                            extra={"predicted": predicted[k]})
            predicted[k] = max(0.0, min(100.0, predicted[k]))
            logger.debug("MID_LAYER_PREDICT", f"指标 {k} 预测值裁剪完成", module="prediction",
                        extra={"clipped": predicted[k]})

        # 置信度计算
        logger.debug("MID_LAYER_PREDICT", "计算置信度", module="prediction")
        if len(self.accuracy_history) > 0:
            recent_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
            self.confidence = 0.4 + 0.6 * recent_accuracy
            logger.info("MID_LAYER_PREDICT", "置信度基于历史准确率更新", module="prediction",
                        extra={"recent_accuracy": recent_accuracy, "confidence": self.confidence})
        else:
            self.confidence = 0.5
            logger.debug("MID_LAYER_PREDICT", "无历史准确率，置信度设为默认值", module="prediction")

        # 综合趋势方向
        trend_counts = {"up": 0, "down": 0, "stable": 0}
        for t in trends.values():
            trend_counts[t] = trend_counts.get(t, 0) + 1
        overall_trend = max(trend_counts, key=trend_counts.get)
        logger.info("MID_LAYER_PREDICT", "综合趋势方向确定", module="prediction",
                    extra={"overall_trend": overall_trend, "trend_counts": trend_counts})

        result = PredictionResult(
            predicted_values=predicted,
            confidence=self.confidence,
            layer="mid",
            trend_direction=overall_trend
        )
        logger.info("MID_LAYER_PREDICT", "中层预测完成", module="prediction",
                    extra={"predicted": predicted, "confidence": self.confidence})
        return result

    @log_execution
    def online_learn(self, actual: Dict[str, float]) -> None:
        """在线学习：根据真实值更新模型参数"""
        logger.debug("MID_LAYER_LEARN", "开始在线学习", module="prediction",
                    extra={"actual": actual})
        if len(self.history) == 0:
            logger.warning("MID_LAYER_LEARN", "历史为空，跳过在线学习", module="prediction")
            return
        last_predicted = self.predict()
        errors = []
        for k in self.metrics_keys:
            pred = last_predicted.predicted_values.get(k, 0.0)
            act = actual.get(k, 0.0)
            error = abs(pred - act) / 100.0
            errors.append(1.0 - min(error, 1.0))
            logger.debug("MID_LAYER_LEARN", f"指标 {k} 误差计算完成", module="prediction",
                        extra={"pred": pred, "act": act, "error": error})
        avg_accuracy = sum(errors) / len(errors)
        self.accuracy_history.append(avg_accuracy)
        logger.info("MID_LAYER_LEARN", "在线学习完成", module="prediction",
                    extra={"avg_accuracy": avg_accuracy})


class HighLayerPredictor(IPrediction):
    """高层预测器 —— 模式识别 + 周期性检测，给出策略建议"""

    def __init__(self, pattern_window: int = 60) -> None:
        logger.info("HIGH_LAYER_INIT", "高层预测器初始化", module="prediction",
                    extra={"pattern_window": pattern_window})
        self.pattern_window = pattern_window
        self.history: deque = deque(maxlen=pattern_window)
        self.metrics_keys = [MetricKeys.CPU, MetricKeys.MEMORY, MetricKeys.DISK]
        self.accuracy_history: deque = deque(maxlen=50)
        self.confidence = 0.5
        self.detected_periods: Dict[str, Optional[int]] = {}
        self.patterns: Dict[str, List[str]] = {}
        logger.debug("HIGH_LAYER_INIT", "高层预测器参数初始化完成", module="prediction")

    @log_execution
    def update(self, state: Dict[str, float]) -> None:
        """更新历史状态"""
        logger.debug("HIGH_LAYER_UPDATE", "开始更新历史状态", module="prediction",
                    extra={"state_keys": list(state.keys())})
        filtered = {k: state.get(k, 0.0) for k in self.metrics_keys}
        self.history.append(filtered)
        logger.info("HIGH_LAYER_UPDATE", "历史状态更新完成", module="prediction",
                    extra={"history_len": len(self.history)})

    @log_execution
    def detect_periodicity(self, values: List[float]) -> Optional[int]:
        """检测周期性模式（使用自相关函数）"""
        logger.debug("HIGH_LAYER_PERIOD", "开始周期性检测", module="prediction",
                    extra={"values_len": len(values)})
        if len(values) < 10:
            logger.warning("HIGH_LAYER_PERIOD", "数据不足，无法检测周期性", module="prediction")
            return None
        # 简化的自相关检测
        n = len(values)
        mean_val = sum(values) / n
        logger.debug("HIGH_LAYER_PERIOD", "均值计算完成", module="prediction",
                    extra={"mean": mean_val})
        autocorr = []
        for lag in range(1, min(n // 2, 30)):
            c = sum((values[i] - mean_val) * (values[i - lag] - mean_val) for i in range(lag, n))
            autocorr.append((lag, c))
            logger.debug("HIGH_LAYER_PERIOD", f"滞后 {lag} 自相关计算完成", module="prediction",
                        extra={"lag": lag, "corr": c})
        if not autocorr:
            logger.warning("HIGH_LAYER_PERIOD", "自相关列表为空", module="prediction")
            return None
        # 寻找峰值
        max_corr = max(autocorr, key=lambda x: x[1])
        period = max_corr[0]
        logger.info("HIGH_LAYER_PERIOD", "周期性检测完成", module="prediction",
                    extra={"period": period, "max_corr": max_corr[1]})
        return period if max_corr[1] > 0 else None

    @log_execution
    def recognize_pattern(self, values: List[float]) -> str:
        """识别模式：周期性/突发/平稳/累积"""
        logger.debug("HIGH_LAYER_PATTERN", "开始模式识别", module="prediction",
                    extra={"values_len": len(values)})
        if len(values) < 5:
            logger.warning("HIGH_LAYER_PATTERN", "数据不足，返回未知模式", module="prediction")
            return "unknown"
        period = self.detect_periodicity(values)
        if period is not None and period > 1:
            logger.info("HIGH_LAYER_PATTERN", "识别为周期性模式", module="prediction",
                        extra={"period": period})
            return "periodic"
        # 检测突发
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
        max_diff = max(diffs) if diffs else 0.0
        logger.debug("HIGH_LAYER_PATTERN", "差分统计完成", module="prediction",
                    extra={"avg_diff": avg_diff, "max_diff": max_diff})
        if max_diff > avg_diff * 3 and max_diff > 10.0:
            logger.info("HIGH_LAYER_PATTERN", "识别为突发模式", module="prediction")
            return "spike"
        # 检测累积
        if len(values) > 3:
            first_half = sum(values[:len(values) // 2]) / (len(values) // 2) if len(values) // 2 > 0 else 0.0
            second_half = sum(values[len(values) // 2:]) / (len(values) - len(values) // 2) if (len(values) - len(values) // 2) > 0 else 0.0
            logger.debug("HIGH_LAYER_PATTERN", "累积检测统计完成", module="prediction",
                        extra={"first_half": first_half, "second_half": second_half})
            if second_half > first_half * 1.3:
                logger.info("HIGH_LAYER_PATTERN", "识别为累积模式", module="prediction")
                return "accumulating"
        logger.info("HIGH_LAYER_PATTERN", "识别为平稳模式", module="prediction")
        return "stable"

    @log_execution
    def generate_strategy(self, patterns: Dict[str, str], predicted: Dict[str, float]) -> str:
        """基于模式和预测生成策略建议"""
        logger.debug("HIGH_LAYER_STRATEGY", "开始生成策略建议", module="prediction",
                    extra={"patterns": patterns, "predicted": predicted})
        advices = []
        for k in self.metrics_keys:
            pattern = patterns.get(k, "unknown")
            pred_val = predicted.get(k, 0.0)
            logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 策略分析", module="prediction",
                        extra={"pattern": pattern, "pred_val": pred_val})
            if pattern == "periodic":
                advices.append(f"{k}: 检测到周期性波动，建议按周期预分配资源")
                logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 周期性策略生成", module="prediction")
            elif pattern == "spike":
                advices.append(f"{k}: 检测到突发模式，建议增加缓冲容量")
                logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 突发策略生成", module="prediction")
            elif pattern == "accumulating":
                advices.append(f"{k}: 检测到累积增长，建议提前扩容")
                logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 累积策略生成", module="prediction")
            else:
                if pred_val > 80.0:
                    advices.append(f"{k}: 预测值较高({pred_val:.1f}%)，建议优化或扩容")
                    logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 高负载策略生成", module="prediction")
                elif pred_val < 20.0:
                    advices.append(f"{k}: 预测值较低({pred_val:.1f}%)，资源利用率不足")
                    logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 低负载策略生成", module="prediction")
                else:
                    advices.append(f"{k}: 状态平稳，维持当前配置")
                    logger.debug("HIGH_LAYER_STRATEGY", f"指标 {k} 平稳策略生成", module="prediction")
        strategy = "; ".join(advices)
        logger.info("HIGH_LAYER_STRATEGY", "策略建议生成完成", module="prediction",
                    extra={"strategy": strategy})
        return strategy

    @log_execution
    def predict(self) -> PredictionResult:
        """高层预测：识别周期性模式，给出策略建议"""
        logger.debug("HIGH_LAYER_PREDICT", "开始高层预测", module="prediction",
                    extra={"history_len": len(self.history)})
        if len(self.history) < 10:
            logger.warning("HIGH_LAYER_PREDICT", "历史数据不足，返回默认值", module="prediction")
            return PredictionResult(
                predicted_values={k: 0.0 for k in self.metrics_keys},
                confidence=0.0,
                layer="high",
                trend_direction="unknown",
                strategy_advice="数据不足，无法生成策略建议"
            )

        patterns = {}
        periods = {}
        predicted = {}
        for k in self.metrics_keys:
            values = [h[k] for h in self.history]
            pattern = self.recognize_pattern(values)
            patterns[k] = pattern
            logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 模式识别完成", module="prediction",
                        extra={"pattern": pattern})
            period = self.detect_periodicity(values)
            periods[k] = period
            logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 周期性检测完成", module="prediction",
                        extra={"period": period})
            # 基于模式给出预测值
            if pattern == "periodic" and period and period > 0:
                idx = len(values) % period
                idx = max(0, min(idx, len(values) - 1))
                predicted[k] = values[idx] if values else 0.0
                logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 周期性预测值计算", module="prediction",
                            extra={"predicted": predicted[k]})
            elif pattern == "accumulating":
                trend = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0
                predicted[k] = values[-1] + trend * 3
                logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 累积预测值计算", module="prediction",
                            extra={"predicted": predicted[k]})
            else:
                predicted[k] = sum(values[-5:]) / min(5, len(values)) if values else 0.0
                logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 默认预测值计算", module="prediction",
                            extra={"predicted": predicted[k]})
            predicted[k] = max(0.0, min(100.0, predicted[k]))
            logger.debug("HIGH_LAYER_PREDICT", f"指标 {k} 预测值裁剪完成", module="prediction",
                        extra={"clipped": predicted[k]})

        self.detected_periods = periods
        self.patterns = patterns

        # 置信度
        logger.debug("HIGH_LAYER_PREDICT", "计算置信度", module="prediction")
        if len(self.accuracy_history) > 0:
            recent_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)
            self.confidence = 0.3 + 0.7 * recent_accuracy
            logger.info("HIGH_LAYER_PREDICT", "置信度基于历史准确率更新", module="prediction",
                        extra={"recent_accuracy": recent_accuracy, "confidence": self.confidence})
        else:
            self.confidence = 0.5
            logger.debug("HIGH_LAYER_PREDICT", "无历史准确率，置信度设为默认值", module="prediction")

        strategy = self.generate_strategy(patterns, predicted)
        result = PredictionResult(
            predicted_values=predicted,
            confidence=self.confidence,
            layer="high",
            trend_direction="periodic" if any(p == "periodic" for p in patterns.values()) else "mixed",
            strategy_advice=strategy
        )
        logger.info("HIGH_LAYER_PREDICT", "高层预测完成", module="prediction",
                    extra={"predicted": predicted, "confidence": self.confidence, "strategy": strategy})
        return result

    @log_execution
    def online_learn(self, actual: Dict[str, float]) -> None:
        """在线学习：根据真实值更新模型参数"""
        logger.debug("HIGH_LAYER_LEARN", "开始在线学习", module="prediction",
                    extra={"actual": actual})
        if len(self.history) == 0:
            logger.warning("HIGH_LAYER_LEARN", "历史为空，跳过在线学习", module="prediction")
            return
        last_predicted = self.predict()
        errors = []
        for k in self.metrics_keys:
            pred = last_predicted.predicted_values.get(k, 0.0)
            act = actual.get(k, 0.0)
            error = abs(pred - act) / 100.0
            errors.append(1.0 - min(error, 1.0))
            logger.debug("HIGH_LAYER_LEARN", f"指标 {k} 误差计算完成", module="prediction",
                        extra={"pred": pred, "act": act, "error": error})
        avg_accuracy = sum(errors) / len(errors)
        self.accuracy_history.append(avg_accuracy)
        logger.info("HIGH_LAYER_LEARN", "在线学习完成", module="prediction",
                    extra={"avg_accuracy": avg_accuracy})


class PredictionEngine(IPrediction):
    """预测引擎主类 —— 协调三层预测器，支持在线学习和自动重置"""

    def __init__(self, base_window: int = 10, mid_lookback: int = 20, high_window: int = 60) -> None:
        logger.info("ENGINE_INIT", "预测引擎初始化", module="prediction",
                    extra={"base_window": base_window, "mid_lookback": mid_lookback, "high_window": high_window})
        self.base_predictor = BaseLayerPredictor(window_size=base_window)
        logger.debug("ENGINE_INIT", "底层预测器创建完成", module="prediction")
        self.mid_predictor = MidLayerPredictor(lookback=mid_lookback)
        logger.debug("ENGINE_INIT", "中层预测器创建完成", module="prediction")
        self.high_predictor = HighLayerPredictor(pattern_window=high_window)
        logger.debug("ENGINE_INIT", "高层预测器创建完成", module="prediction")
        self.error_history: deque = deque(maxlen=20)
        self.reset_threshold = 3
        self.consecutive_errors = 0
        self.metrics_keys = [MetricKeys.CPU, MetricKeys.MEMORY, MetricKeys.DISK]
        logger.info("ENGINE_INIT", "预测引擎初始化完成", module="prediction")

    @log_execution
    def update(self, state: Dict[str, float]) -> None:
        """更新所有层的历史状态"""
        logger.debug("ENGINE_UPDATE", "开始更新所有预测器", module="prediction",
                    extra={"state_keys": list(state.keys())})
        self.base_predictor.update(state)
        logger.debug("ENGINE_UPDATE", "底层预测器更新完成", module="prediction")
        self.mid_predictor.update(state)
        logger.debug("ENGINE_UPDATE", "中层预测器更新完成", module="prediction")
        self.high_predictor.update(state)
        logger.debug("ENGINE_UPDATE", "高层预测器更新完成", module="prediction")
        logger.info("ENGINE_UPDATE", "所有预测器更新完成", module="prediction")

    @log_execution
    def predict(self, layer: Optional[str] = None) -> PredictionResult:
        """执行预测，可选择指定层"""
        logger.debug("ENGINE_PREDICT", "开始预测", module="prediction",
                    extra={"requested_layer": layer})
        if layer == "base":
            result = self.base_predictor.predict()
            logger.info("ENGINE_PREDICT", "底层预测完成", module="prediction")
            return result
        elif layer == "mid":
            result = self.mid_predictor.predict()
            logger.info("ENGINE_PREDICT", "中层预测完成", module="prediction")
            return result
        elif layer == "high":
            result = self.high_predictor.predict()
            logger.info("ENGINE_PREDICT", "高层预测完成", module="prediction")
            return result
        else:
            # 默认返回中层预测（平衡速度和准确性）
            logger.debug("ENGINE_PREDICT", "未指定层，使用默认中层预测", module="prediction")
            result = self.mid_predictor.predict()
            logger.info("ENGINE_PREDICT", "默认中层预测完成", module="prediction")
            return result

    @log_execution
    def predict_all(self) -> Dict[str, PredictionResult]:
        """执行所有层的预测"""
        logger.debug("ENGINE_PREDICT_ALL", "开始执行所有层预测", module="prediction")
        base_result = self.base_predictor.predict()
        logger.debug("ENGINE_PREDICT_ALL", "底层预测完成", module="prediction")
        mid_result = self.mid_predictor.predict()
        logger.debug("ENGINE_PREDICT_ALL", "中层预测完成", module="prediction")
        high_result = self.high_predictor.predict()
        logger.debug("ENGINE_PREDICT_ALL", "高层预测完成", module="prediction")
        results = {
            "base": base_result,
            "mid": mid_result,
            "high": high_result
        }
        logger.info("ENGINE_PREDICT_ALL", "所有层预测完成", module="prediction",
                    extra={"layers": list(results.keys())})
        return results

    @log_execution
    def online_learn(self, actual: Dict[str, float]) -> None:
        """在线学习：根据真实值更新所有层模型"""
        logger.debug("ENGINE_LEARN", "开始在线学习", module="prediction",
                    extra={"actual": actual})
        self.base_predictor.online_learn(actual)
        logger.debug("ENGINE_LEARN", "底层在线学习完成", module="prediction")
        self.mid_predictor.online_learn(actual)
        logger.debug("ENGINE_LEARN", "中层在线学习完成", module="prediction")
        self.high_predictor.online_learn(actual)
        logger.debug("ENGINE_LEARN", "高层在线学习完成", module="prediction")

        # 计算整体误差
        logger.debug("ENGINE_LEARN", "计算整体误差", module="prediction")
        base_pred = self.base_predictor.predict()
        errors = []
        for k in self.metrics_keys:
            pred = base_pred.predicted_values.get(k, 0.0)
            act = actual.get(k, 0.0)
            error = abs(pred - act) / 100.0
            errors.append(error)
            logger.debug("ENGINE_LEARN", f"指标 {k} 误差", module="prediction",
                        extra={"error": error})
        avg_error = sum(errors) / len(errors)
        self.error_history.append(avg_error)
        logger.info("ENGINE_LEARN", "整体误差计算完成", module="prediction",
                    extra={"avg_error": avg_error, "error_history_len": len(self.error_history)})

        # 检查是否需要重置
        logger.debug("ENGINE_LEARN", "检查是否需要模型重置", module="prediction")
        self._check_reset()

    @log_execution
    def _check_reset(self) -> None:
        """检查预测误差是否持续增大，自动触发模型重置"""
        logger.debug("ENGINE_CHECK_RESET", "开始检查重置条件", module="prediction",
                    extra={"error_history_len": len(self.error_history), "consecutive_errors": self.consecutive_errors})
        if len(self.error_history) < 5:
            logger.debug("ENGINE_CHECK_RESET", "误差历史不足，跳过检查", module="prediction")
            return
        recent_errors = list(self.error_history)[-5:]
        logger.debug("ENGINE_CHECK_RESET", "获取最近误差", module="prediction",
                    extra={"recent_errors": recent_errors})
        increasing = all(recent_errors[i] <= recent_errors[i + 1] for i in range(len(recent_errors) - 1))
        logger.debug("ENGINE_CHECK_RESET", "误差趋势判断完成", module="prediction",
                    extra={"increasing": increasing})
        if increasing and recent_errors[-1] > 0.2:
            self.consecutive_errors += 1
            logger.warning("ENGINE_CHECK_RESET", "误差持续增大", module="prediction",
                          extra={"consecutive_errors": self.consecutive_errors})
            if self.consecutive_errors >= self.reset_threshold:
                logger.error("ENGINE_CHECK_RESET", "触发模型重置", module="prediction",
                            extra={"consecutive_errors": self.consecutive_errors, "threshold": self.reset_threshold})
                self._reset_models()
                self.consecutive_errors = 0
                logger.info("ENGINE_CHECK_RESET", "模型重置完成", module="prediction")
        else:
            if self.consecutive_errors > 0:
                logger.debug("ENGINE_CHECK_RESET", "误差趋势缓解，重置连续错误计数", module="prediction")
            self.consecutive_errors = 0

    @log_execution
    def _reset_models(self) -> None:
        """重置所有预测模型"""
        logger.info("ENGINE_RESET", "开始重置所有预测模型", module="prediction")
        self.base_predictor = BaseLayerPredictor(window_size=self.base_predictor.window_size)
        logger.debug("ENGINE_RESET", "底层预测器已重置", module="prediction")
        self.mid_predictor = MidLayerPredictor(lookback=self.mid_predictor.lookback, future_steps=self.mid_predictor.future_steps)
        logger.debug("ENGINE_RESET", "中层预测器已重置", module="prediction")
        self.high_predictor = HighLayerPredictor(pattern_window=self.high_predictor.pattern_window)
        logger.debug("ENGINE_RESET", "高层预测器已重置", module="prediction")
        self.error_history.clear()
        logger.debug("ENGINE_RESET", "误差历史已清空", module="prediction")
        logger.info("ENGINE_RESET", "所有预测模型重置完成", module="prediction")

    @log_execution
    def get_confidence_report(self) -> Dict[str, float]:
        """获取各层置信度报告"""
        logger.debug("ENGINE_CONFIDENCE", "开始生成置信度报告", module="prediction")
        report = {
            "base": self.base_predictor.confidence,
            "mid": self.mid_predictor.confidence,
            "high": self.high_predictor.confidence,
            "overall": (self.base_predictor.confidence + self.mid_predictor.confidence + self.high_predictor.confidence) / 3.0
        }
        logger.info("ENGINE_CONFIDENCE", "置信度报告生成完成", module="prediction",
                    extra={"report": report})
        return report
