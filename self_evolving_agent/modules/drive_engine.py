"""
驱力引擎模块 —— 计算三大原生驱力与内在动机
包含安全驱力（Reward_safe）、收益驱力（Reward_gain）、能效驱力（Reward_eff）
"""
import os
import sys
import time
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

# 仅在直接运行时添加路径（避免副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.config import config_manager
from core.interfaces import IDriveEngine, StateKeys, DriveKeys


def _safe_float(val, default: float = 0.0) -> float:
    """安全转换为float，处理None、NaN、Inf等异常值"""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


@dataclass
class DriveState:
    """驱力状态数据类"""
    safety: float = 0.0
    gain: float = 0.0
    efficiency: float = 0.0
    intrinsic_motivation: float = 0.0
    exploration_ratio: float = 0.30
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety": self.safety,
            "gain": self.gain,
            "efficiency": self.efficiency,
            "intrinsic_motivation": self.intrinsic_motivation,
            "exploration_ratio": self.exploration_ratio,
            "timestamp": self.timestamp,
        }


class SafetyDrive:
    """安全驱力计算 —— 基于CPU/内存/磁盘/错误率等风险指标"""
    
    def __init__(self) -> None:
        logger.info("SAFETY_INIT", "安全驱力模块初始化", module="drive_engine")
        self.config = config_manager.config
        self.history = deque(maxlen=100)
        logger.info("SAFETY_INIT", "安全驱力模块初始化完成", module="drive_engine",
                   extra={"history_size": 100})
    
    @log_execution
    def compute(self, state_vector: Dict[str, float]) -> float:
        """计算安全驱力：当资源超过阈值时为负，正常时为正"""
        logger.debug("SAFETY_COMPUTE", "开始计算安全驱力", module="drive_engine",
                    extra={"state_keys": list(state_vector.keys())})
        
        cpu = _safe_float(state_vector.get(StateKeys.CPU_PERCENT, 0.0))
        mem = _safe_float(state_vector.get(StateKeys.MEMORY_PERCENT, 0.0))
        disk = _safe_float(state_vector.get(StateKeys.DISK_PERCENT, 0.0))
        error_rate = _safe_float(state_vector.get(StateKeys.ERROR_RATE, 0.0))
        load_avg = _safe_float(state_vector.get(StateKeys.LOAD_AVG, 0.0))
        
        logger.debug("SAFETY_COMPUTE", "获取原始指标值", module="drive_engine",
                    extra={"cpu": cpu, "mem": mem, "disk": disk,
                           "error_rate": error_rate, "load_avg": load_avg})
        
        # CPU 安全评分：超过阈值线性下降
        cpu_threshold = self.config.resource.max_cpu_percent / 100.0
        logger.debug("SAFETY_COMPUTE", f"CPU阈值: {cpu_threshold}", module="drive_engine")
        if cpu > cpu_threshold:
            cpu_score = -1.0 * (cpu - cpu_threshold) / (1.0 - cpu_threshold + 1e-6)
            logger.warning("SAFETY_COMPUTE", f"CPU超过阈值，评分为负: {cpu_score:.4f}", module="drive_engine",
                          extra={"cpu": cpu, "threshold": cpu_threshold})
        else:
            cpu_score = 1.0 - cpu / (cpu_threshold + 1e-6)
            logger.debug("SAFETY_COMPUTE", f"CPU正常，评分: {cpu_score:.4f}", module="drive_engine")
        
        # 内存安全评分
        mem_threshold = self.config.resource.max_memory_percent / 100.0
        logger.debug("SAFETY_COMPUTE", f"内存阈值: {mem_threshold}", module="drive_engine")
        if mem > mem_threshold:
            mem_score = -1.0 * (mem - mem_threshold) / (1.0 - mem_threshold + 1e-6)
            logger.warning("SAFETY_COMPUTE", f"内存超过阈值，评分为负: {mem_score:.4f}", module="drive_engine",
                          extra={"mem": mem, "threshold": mem_threshold})
        else:
            mem_score = 1.0 - mem / (mem_threshold + 1e-6)
            logger.debug("SAFETY_COMPUTE", f"内存正常，评分: {mem_score:.4f}", module="drive_engine")
        
        # 磁盘安全评分
        disk_threshold = 0.85
        logger.debug("SAFETY_COMPUTE", f"磁盘阈值: {disk_threshold}", module="drive_engine")
        if disk > disk_threshold:
            disk_score = -1.0 * (disk - disk_threshold) / (1.0 - disk_threshold + 1e-6)
            logger.warning("SAFETY_COMPUTE", f"磁盘超过阈值，评分为负: {disk_score:.4f}", module="drive_engine",
                          extra={"disk": disk, "threshold": disk_threshold})
        else:
            disk_score = 1.0 - disk / (disk_threshold + 1e-6)
            logger.debug("SAFETY_COMPUTE", f"磁盘正常，评分: {disk_score:.4f}", module="drive_engine")
        
        # 错误率安全评分：错误率越高越不安全
        logger.debug("SAFETY_COMPUTE", "计算错误率评分", module="drive_engine")
        error_score = 1.0 - 2.0 * error_rate
        if error_rate > 0.1:
            logger.warning("SAFETY_COMPUTE", f"错误率偏高: {error_rate:.4f}", module="drive_engine")
        else:
            logger.debug("SAFETY_COMPUTE", f"错误率正常: {error_rate:.4f}", module="drive_engine")
        
        # 负载评分
        logger.debug("SAFETY_COMPUTE", "计算负载评分", module="drive_engine")
        load_score = 1.0 - load_avg
        if load_avg > 0.8:
            logger.warning("SAFETY_COMPUTE", f"系统负载偏高: {load_avg:.4f}", module="drive_engine")
        else:
            logger.debug("SAFETY_COMPUTE", f"系统负载正常: {load_avg:.4f}", module="drive_engine")
        
        # 综合安全驱力（加权平均后归一化到 [-1, 1]）
        weights = {"cpu": 0.25, "mem": 0.25, "disk": 0.20, "error": 0.20, "load": 0.10}
        logger.debug("SAFETY_COMPUTE", f"权重配置: {weights}", module="drive_engine")
        
        safety = (
            weights["cpu"] * cpu_score +
            weights["mem"] * mem_score +
            weights["disk"] * disk_score +
            weights["error"] * error_score +
            weights["load"] * load_score
        )
        
        safety = max(-1.0, min(1.0, safety))
        logger.info("SAFETY_COMPUTE", f"安全驱力计算完成: {safety:.4f}", module="drive_engine",
                   extra={"safety": safety, "cpu_score": cpu_score, "mem_score": mem_score,
                          "disk_score": disk_score, "error_score": error_score, "load_score": load_score})
        
        self.history.append({"timestamp": time.time(), "value": safety})
        logger.debug("SAFETY_COMPUTE", "安全驱力历史记录已更新", module="drive_engine",
                    extra={"history_len": len(self.history)})
        
        return safety


class GainDrive:
    """收益驱力计算 —— 基于任务完成率、预测准确率提升等"""
    
    def __init__(self) -> None:
        logger.info("GAIN_INIT", "收益驱力模块初始化", module="drive_engine")
        self.history = deque(maxlen=100)
        self._last_prediction_error: Optional[float] = None
        self._last_task_success_rate: Optional[float] = None
        logger.info("GAIN_INIT", "收益驱力模块初始化完成", module="drive_engine",
                   extra={"history_size": 100})
    
    @log_execution
    def compute(self, state_vector: Dict[str, float],
                tasks_completed: int = 0,
                tasks_failed: int = 0,
                prediction_error: float = 0.0) -> float:
        """计算收益驱力：基于预测误差下降、任务成功率提升"""
        logger.debug("GAIN_COMPUTE", "开始计算收益驱力", module="drive_engine",
                    extra={"tasks_completed": tasks_completed, "tasks_failed": tasks_failed,
                           "prediction_error": prediction_error})
        
        # 预测误差改善评分
        logger.debug("GAIN_COMPUTE", "计算预测误差改善评分", module="drive_engine")
        if self._last_prediction_error is not None and self._last_prediction_error > 0:
            error_delta = self._last_prediction_error - prediction_error
            error_improvement = error_delta / (self._last_prediction_error + 1e-6)
            error_score = math.tanh(error_improvement * 2.0)
            logger.debug("GAIN_COMPUTE", f"预测误差改善: {error_improvement:.4f}, 评分: {error_score:.4f}",
                        module="drive_engine",
                        extra={"last_error": self._last_prediction_error, "current_error": prediction_error})
        else:
            error_score = 0.0
            logger.debug("GAIN_COMPUTE", "无历史预测误差，误差评分设为0", module="drive_engine")
        
        self._last_prediction_error = prediction_error
        logger.debug("GAIN_COMPUTE", f"更新历史预测误差: {prediction_error}", module="drive_engine")
        
        # 任务成功率评分
        logger.debug("GAIN_COMPUTE", "计算任务成功率评分", module="drive_engine")
        total_tasks = tasks_completed + tasks_failed
        if total_tasks > 0:
            success_rate = tasks_completed / total_tasks
            logger.debug("GAIN_COMPUTE", f"当前成功率: {success_rate:.4f}", module="drive_engine",
                        extra={"total": total_tasks, "completed": tasks_completed, "failed": tasks_failed})
            
            if self._last_task_success_rate is not None:
                success_delta = success_rate - self._last_task_success_rate
                success_score = math.tanh(success_delta * 5.0) + success_rate - 0.5
                logger.debug("GAIN_COMPUTE", f"成功率变化: {success_delta:.4f}, 评分: {success_score:.4f}",
                            module="drive_engine",
                            extra={"last_rate": self._last_task_success_rate, "current_rate": success_rate})
            else:
                success_score = success_rate - 0.5
                logger.debug("GAIN_COMPUTE", f"无历史成功率，基础评分: {success_score:.4f}", module="drive_engine")
            
            self._last_task_success_rate = success_rate
            logger.debug("GAIN_COMPUTE", f"更新历史成功率: {success_rate}", module="drive_engine")
        else:
            success_score = 0.0
            logger.debug("GAIN_COMPUTE", "无任务数据，成功率评分设为0", module="drive_engine")
        
        # 模型准确率评分（直接从状态向量获取）
        model_accuracy = state_vector.get(StateKeys.MODEL_ACCURACY, 0.0)
        logger.debug("GAIN_COMPUTE", f"模型准确率: {model_accuracy:.4f}", module="drive_engine")
        accuracy_score = 2.0 * (model_accuracy - 0.5)
        logger.debug("GAIN_COMPUTE", f"准确率评分: {accuracy_score:.4f}", module="drive_engine")
        
        # 综合收益驱力
        weights = {"error": 0.35, "success": 0.45, "accuracy": 0.20}
        logger.debug("GAIN_COMPUTE", f"权重配置: {weights}", module="drive_engine")
        
        gain = (
            weights["error"] * error_score +
            weights["success"] * success_score +
            weights["accuracy"] * accuracy_score
        )
        
        gain = max(-1.0, min(1.0, gain))
        logger.info("GAIN_COMPUTE", f"收益驱力计算完成: {gain:.4f}", module="drive_engine",
                   extra={"gain": gain, "error_score": error_score,
                          "success_score": success_score, "accuracy_score": accuracy_score})
        
        self.history.append({"timestamp": time.time(), "value": gain})
        logger.debug("GAIN_COMPUTE", "收益驱力历史记录已更新", module="drive_engine",
                    extra={"history_len": len(self.history)})
        
        return gain


class EfficiencyDrive:
    """能效驱力计算 —— 基于资源利用率、能耗比等"""
    
    def __init__(self) -> None:
        logger.info("EFF_INIT", "能效驱力模块初始化", module="drive_engine")
        self.history = deque(maxlen=100)
        self._last_resource_usage: Optional[float] = None
        logger.info("EFF_INIT", "能效驱力模块初始化完成", module="drive_engine",
                   extra={"history_size": 100})
    
    @log_execution
    def compute(self, state_vector: Dict[str, float]) -> float:
        """计算能效驱力：资源使用过高或过低都降低能效"""
        logger.debug("EFF_COMPUTE", "开始计算能效驱力", module="drive_engine",
                    extra={"state_keys": list(state_vector.keys())})
        
        cpu = _safe_float(state_vector.get(StateKeys.CPU_PERCENT, 0.0))
        mem = _safe_float(state_vector.get(StateKeys.MEMORY_PERCENT, 0.0))
        disk_io = _safe_float(state_vector.get(StateKeys.DISK_IO, 0.0))
        net_io = _safe_float(state_vector.get(StateKeys.NET_IO, 0.0))
        agent_cpu = _safe_float(state_vector.get(StateKeys.AGENT_CPU, 0.0))
        agent_mem = _safe_float(state_vector.get(StateKeys.AGENT_MEMORY, 0.0))
        
        logger.debug("EFF_COMPUTE", "获取原始指标值", module="drive_engine",
                    extra={"cpu": cpu, "mem": mem, "disk_io": disk_io,
                           "net_io": net_io, "agent_cpu": agent_cpu, "agent_mem": agent_mem})
        
        # 综合资源使用率（系统+自身）
        sys_usage = (cpu + mem) / 2.0
        agent_usage = (agent_cpu + agent_mem) / 2.0
        io_usage = min((disk_io + net_io) / 2.0, 1.0)
        
        logger.debug("EFF_COMPUTE", f"系统使用率: {sys_usage:.4f}, 自身使用率: {agent_usage:.4f}, IO使用率: {io_usage:.4f}",
                    module="drive_engine")
        
        # 资源利用率评分：理想区间 [0.3, 0.7]，偏离则下降
        logger.debug("EFF_COMPUTE", "计算资源利用率评分", module="drive_engine")
        ideal_low, ideal_high = 0.3, 0.7
        if sys_usage < ideal_low:
            util_score = -1.0 + 2.0 * sys_usage / ideal_low
            logger.debug("EFF_COMPUTE", f"系统使用率偏低: {sys_usage:.4f}, 评分: {util_score:.4f}", module="drive_engine")
        elif sys_usage > ideal_high:
            util_score = 1.0 - 2.0 * (sys_usage - ideal_high) / (1.0 - ideal_high + 1e-6)
            logger.warning("EFF_COMPUTE", f"系统使用率偏高: {sys_usage:.4f}, 评分: {util_score:.4f}", module="drive_engine")
        else:
            util_score = 2.0 * (sys_usage - ideal_low) / (ideal_high - ideal_low) - 1.0
            logger.debug("EFF_COMPUTE", f"系统使用率在理想区间: {sys_usage:.4f}, 评分: {util_score:.4f}", module="drive_engine")
        
        # 自身资源效率评分
        logger.debug("EFF_COMPUTE", "计算自身资源效率评分", module="drive_engine")
        if agent_usage < 0.1:
            agent_eff_score = -0.5 + 5.0 * agent_usage
            logger.debug("EFF_COMPUTE", f"自身使用率过低: {agent_usage:.4f}, 评分: {agent_eff_score:.4f}", module="drive_engine")
        elif agent_usage > 0.5:
            agent_eff_score = 1.0 - 2.0 * (agent_usage - 0.5) / 0.5
            logger.warning("EFF_COMPUTE", f"自身使用率过高: {agent_usage:.4f}, 评分: {agent_eff_score:.4f}", module="drive_engine")
        else:
            agent_eff_score = agent_usage * 2.0
            logger.debug("EFF_COMPUTE", f"自身使用率正常: {agent_usage:.4f}, 评分: {agent_eff_score:.4f}", module="drive_engine")
        
        # IO 效率评分
        logger.debug("EFF_COMPUTE", "计算IO效率评分", module="drive_engine")
        if io_usage < 0.05:
            io_score = -0.3 + 6.0 * io_usage
            logger.debug("EFF_COMPUTE", f"IO使用率过低: {io_usage:.4f}, 评分: {io_score:.4f}", module="drive_engine")
        elif io_usage > 0.8:
            io_score = 1.0 - 5.0 * (io_usage - 0.8) / 0.2
            logger.warning("EFF_COMPUTE", f"IO使用率过高: {io_usage:.4f}, 评分: {io_score:.4f}", module="drive_engine")
        else:
            io_score = io_usage * 1.25
            logger.debug("EFF_COMPUTE", f"IO使用率正常: {io_usage:.4f}, 评分: {io_score:.4f}", module="drive_engine")
        
        # 能效变化趋势（与上次比较）
        logger.debug("EFF_COMPUTE", "计算能效变化趋势", module="drive_engine")
        current_usage = (sys_usage + agent_usage + io_usage) / 3.0
        if self._last_resource_usage is not None:
            usage_delta = self._last_resource_usage - current_usage
            trend_score = math.tanh(usage_delta * 3.0)
            logger.debug("EFF_COMPUTE", f"资源使用变化: {usage_delta:.4f}, 趋势评分: {trend_score:.4f}",
                        module="drive_engine",
                        extra={"last_usage": self._last_resource_usage, "current_usage": current_usage})
        else:
            trend_score = 0.0
            logger.debug("EFF_COMPUTE", "无历史资源使用数据，趋势评分设为0", module="drive_engine")
        
        self._last_resource_usage = current_usage
        logger.debug("EFF_COMPUTE", f"更新历史资源使用率: {current_usage:.4f}", module="drive_engine")
        
        # 综合能效驱力
        weights = {"util": 0.35, "agent": 0.30, "io": 0.20, "trend": 0.15}
        logger.debug("EFF_COMPUTE", f"权重配置: {weights}", module="drive_engine")
        
        efficiency = (
            weights["util"] * util_score +
            weights["agent"] * agent_eff_score +
            weights["io"] * io_score +
            weights["trend"] * trend_score
        )
        
        efficiency = max(-1.0, min(1.0, efficiency))
        logger.info("EFF_COMPUTE", f"能效驱力计算完成: {efficiency:.4f}", module="drive_engine",
                   extra={"efficiency": efficiency, "util_score": util_score,
                          "agent_eff_score": agent_eff_score, "io_score": io_score, "trend_score": trend_score})
        
        self.history.append({"timestamp": time.time(), "value": efficiency})
        logger.debug("EFF_COMPUTE", "能效驱力历史记录已更新", module="drive_engine",
                    extra={"history_len": len(self.history)})
        
        return efficiency


class IntrinsicMotivation:
    """内在动机综合计算 —— 整合三大驱力，按权重加权"""
    
    def __init__(self) -> None:
        logger.info("MOTIVATION_INIT", "内在动机模块初始化", module="drive_engine")
        self.config = config_manager.config
        self.safety_drive = SafetyDrive()
        self.gain_drive = GainDrive()
        self.efficiency_drive = EfficiencyDrive()
        self.history = deque(maxlen=100)
        self._steady_state_counter = 0
        logger.info("MOTIVATION_INIT", "内在动机模块初始化完成", module="drive_engine",
                   extra={"weights": self.config.drive_weights.to_dict()})
    
    @log_execution
    def compute(self, state_vector: Dict[str, float],
                tasks_completed: int = 0,
                tasks_failed: int = 0,
                prediction_error: float = 0.0) -> DriveState:
        """计算内在动机 = safe_weight * safety + gain_weight * gain + eff_weight * efficiency"""
        logger.debug("MOTIVATION_COMPUTE", "开始计算内在动机", module="drive_engine")
        
        # 分别计算三大驱力
        logger.debug("MOTIVATION_COMPUTE", "调用安全驱力计算", module="drive_engine")
        safety = self.safety_drive.compute(state_vector)
        logger.debug("MOTIVATION_COMPUTE", f"安全驱力结果: {safety:.4f}", module="drive_engine")
        
        logger.debug("MOTIVATION_COMPUTE", "调用收益驱力计算", module="drive_engine")
        gain = self.gain_drive.compute(state_vector, tasks_completed, tasks_failed, prediction_error)
        logger.debug("MOTIVATION_COMPUTE", f"收益驱力结果: {gain:.4f}", module="drive_engine")
        
        logger.debug("MOTIVATION_COMPUTE", "调用能效驱力计算", module="drive_engine")
        efficiency = self.efficiency_drive.compute(state_vector)
        logger.debug("MOTIVATION_COMPUTE", f"能效驱力结果: {efficiency:.4f}", module="drive_engine")
        
        # 获取权重
        weights = self.config.drive_weights
        logger.debug("MOTIVATION_COMPUTE", f"驱力权重: safe={weights.safe}, gain={weights.gain}, eff={weights.eff}",
                    module="drive_engine")
        
        # 加权综合
        intrinsic = weights.safe * safety + weights.gain * gain + weights.eff * efficiency
        logger.debug("MOTIVATION_COMPUTE", f"加权内在动机原始值: {intrinsic:.4f}", module="drive_engine")
        
        intrinsic = max(-1.0, min(1.0, intrinsic))
        logger.debug("MOTIVATION_COMPUTE", f"归一化后内在动机: {intrinsic:.4f}", module="drive_engine")
        
        # 探索-利用平衡（默认 70/30）
        base_exploration = self.config.evolution.exploration_ratio
        logger.debug("MOTIVATION_COMPUTE", f"基础探索比例: {base_exploration:.4f}", module="drive_engine")
        
        # 当系统稳定时增加探索
        logger.debug("MOTIVATION_COMPUTE", "检查系统稳定性", module="drive_engine")
        steady_threshold = self.config.evolution.steady_state_error_threshold
        if abs(safety) < steady_threshold and abs(gain) < steady_threshold and abs(efficiency) < steady_threshold:
            self._steady_state_counter += 1
            logger.info("MOTIVATION_COMPUTE", f"系统趋于稳定，稳定计数器: {self._steady_state_counter}", module="drive_engine")
        else:
            if self._steady_state_counter > 0:
                logger.info("MOTIVATION_COMPUTE", f"系统脱离稳定状态，重置计数器", module="drive_engine",
                           extra={"previous_counter": self._steady_state_counter})
            self._steady_state_counter = 0
        
        exploration_boost = 0.0
        if self._steady_state_counter >= 5:
            exploration_boost = self.config.evolution.exploration_boost_on_steady
            logger.info("MOTIVATION_COMPUTE", f"触发探索增强: +{exploration_boost:.4f}", module="drive_engine",
                       extra={"steady_counter": self._steady_state_counter})
        else:
            logger.debug("MOTIVATION_COMPUTE", f"未触发探索增强，稳定计数: {self._steady_state_counter}", module="drive_engine")
        
        exploration_ratio = min(base_exploration + exploration_boost, 0.5)
        logger.debug("MOTIVATION_COMPUTE", f"最终探索比例: {exploration_ratio:.4f}", module="drive_engine")
        
        drive_state = DriveState(
            safety=safety,
            gain=gain,
            efficiency=efficiency,
            intrinsic_motivation=intrinsic,
            exploration_ratio=exploration_ratio,
            timestamp=time.time(),
        )
        
        self.history.append(drive_state.to_dict())
        logger.debug("MOTIVATION_COMPUTE", "内在动机历史记录已更新", module="drive_engine",
                    extra={"history_len": len(self.history)})
        
        logger.info("MOTIVATION_COMPUTE", f"内在动机计算完成: {intrinsic:.4f}, 探索比例: {exploration_ratio:.4f}",
                   module="drive_engine",
                   extra=drive_state.to_dict())
        
        return drive_state
    
    @log_execution
    def get_trend(self, window: int = 10) -> Dict[str, float]:
        """获取最近驱力变化趋势"""
        logger.debug("MOTIVATION_TREND", f"计算驱力趋势，窗口大小: {window}", module="drive_engine")
        
        if len(self.history) < 2:
            logger.warning("MOTIVATION_TREND", "历史数据不足，返回零趋势", module="drive_engine",
                          extra={"history_len": len(self.history)})
            return {"safety": 0.0, "gain": 0.0, "efficiency": 0.0, "intrinsic": 0.0}
        
        recent = list(self.history)[-window:]
        logger.debug("MOTIVATION_TREND", f"获取最近 {len(recent)} 条记录", module="drive_engine")
        
        safety_trend = recent[-1].get("safety", 0.0) - recent[0].get("safety", 0.0)
        gain_trend = recent[-1].get("gain", 0.0) - recent[0].get("gain", 0.0)
        efficiency_trend = recent[-1].get("efficiency", 0.0) - recent[0].get("efficiency", 0.0)
        intrinsic_trend = recent[-1].get("intrinsic_motivation", 0.0) - recent[0].get("intrinsic_motivation", 0.0)
        
        trends = {
            "safety": safety_trend,
            "gain": gain_trend,
            "efficiency": efficiency_trend,
            "intrinsic": intrinsic_trend,
        }
        
        logger.info("MOTIVATION_TREND", f"驱力趋势计算完成: {trends}", module="drive_engine", extra=trends)
        return trends


class DriveEngine(IDriveEngine):
    """驱力引擎主类 —— 统一接口，调度三大驱力计算"""
    
    def __init__(self) -> None:
        logger.info("ENGINE_INIT", "驱力引擎主类初始化", module="drive_engine")
        self.config = config_manager.config
        self.motivation = IntrinsicMotivation()
        self._last_state: Optional[DriveState] = None
        logger.info("ENGINE_INIT", "驱力引擎主类初始化完成", module="drive_engine",
                   extra={"exploration_ratio": self.config.evolution.exploration_ratio})
    
    @log_execution
    def compute(self, state_vector: Dict[str, float],
                tasks_completed: int = 0,
                tasks_failed: int = 0,
                prediction_error: float = 0.0) -> DriveState:
        """计算当前驱力状态"""
        logger.debug("ENGINE_COMPUTE", "驱力引擎开始计算", module="drive_engine",
                    extra={"tasks_completed": tasks_completed, "tasks_failed": tasks_failed,
                           "prediction_error": prediction_error, "state_keys": list(state_vector.keys())})
        
        drive_state = self.motivation.compute(state_vector, tasks_completed, tasks_failed, prediction_error)
        self._last_state = drive_state
        
        logger.info("ENGINE_COMPUTE", f"驱力引擎计算完成，内在动机: {drive_state.intrinsic_motivation:.4f}",
                   module="drive_engine", extra=drive_state.to_dict())
        
        return drive_state
    
    @log_execution
    def get_exploration_ratio(self) -> float:
        """获取当前探索比例"""
        logger.debug("ENGINE_EXPLORE", "获取探索比例", module="drive_engine")
        
        if self._last_state is not None:
            ratio = self._last_state.exploration_ratio
            logger.info("ENGINE_EXPLORE", f"当前探索比例: {ratio:.4f}", module="drive_engine", extra={"ratio": ratio})
            return ratio
        
        default_ratio = self.config.evolution.exploration_ratio
        logger.warning("ENGINE_EXPLORE", f"无历史状态，返回默认探索比例: {default_ratio:.4f}", module="drive_engine",
                      extra={"default_ratio": default_ratio})
        return default_ratio
    
    @log_execution
    def should_explore(self) -> bool:
        """根据探索比例决定是否进行探索"""
        logger.debug("ENGINE_DECIDE", "判断是否探索", module="drive_engine")
        
        import random
        ratio = self.get_exploration_ratio()
        decision = random.random() < ratio
        
        if decision:
            logger.info("ENGINE_DECIDE", f"决定探索 (概率 {ratio:.4f})", module="drive_engine", extra={"explore": True, "ratio": ratio})
        else:
            logger.info("ENGINE_DECIDE", f"决定利用 (概率 {1-ratio:.4f})", module="drive_engine", extra={"explore": False, "ratio": ratio})
        
        return decision
    
    @log_execution
    def get_drive_summary(self) -> Dict[str, Any]:
        """获取驱力摘要"""
        logger.debug("ENGINE_SUMMARY", "获取驱力摘要", module="drive_engine")
        
        if self._last_state is None:
            logger.warning("ENGINE_SUMMARY", "无驱力状态数据", module="drive_engine")
            return {"status": "no_data", "last_state": None}
        
        summary = {
            "status": "ok",
            "last_state": self._last_state.to_dict(),
            "trends": self.motivation.get_trend(),
            "exploration_ratio": self._last_state.exploration_ratio,
            "should_explore": self.should_explore(),
        }
        
        logger.info("ENGINE_SUMMARY", "驱力摘要生成完成", module="drive_engine", extra=summary)
        return summary
