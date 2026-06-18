"""
告警与通知系统 —— 规则驱动的多渠道告警，防抖动，静默期控制

设计原则（第一性原理）：
1. 告警即疼痛信号：系统感知到异常（误差过大、资源超限）时向外传递
2. 规则可配置：不同场景不同阈值，避免一刀切
3. 防抖动：连续触发N次才告警，避免噪声淹没信号
4. 静默期：同一告警在静默期内不重复，避免轰炸
5. 多渠道：日志/HTTP/脚本，确保通知可达
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_path: str  # 状态向量中的键路径，如 "cpu_percent"
    threshold: float
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    duration_sec: float = 0.0  # 持续多久才触发（0表示立即）
    level: AlertLevel = AlertLevel.WARNING
    cooldown_sec: float = 60.0  # 静默期
    consecutive_count: int = 1  # 连续触发次数
    channels: List[str] = field(default_factory=lambda: ["log"])  # log/http/script
    enabled: bool = True


@dataclass
class AlertEvent:
    """告警事件"""
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_value: float
    threshold: float


class AlertChannel:
    """告警通道基类"""

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError


class LogChannel(AlertChannel):
    """日志通道"""

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        extra = {
            "rule": event.rule_name,
            "value": event.metric_value,
            "threshold": event.threshold,
        }
        if context:
            extra.update(context)
        logger.warning(
            "ALERT",
            f"[{event.level.value.upper()}] {event.message}",
            module="alerting",
            extra=extra,
        )
        return True


class HTTPChannel(AlertChannel):
    """HTTP回调通道"""

    def __init__(self, webhook_url: str, timeout_sec: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import urllib.request
            payload = {
                "rule": event.rule_name,
                "level": event.level.value,
                "message": event.message,
                "timestamp": event.timestamp,
                "metric_value": event.metric_value,
                "threshold": event.threshold,
                "context": context or {},
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("ALERT", f"HTTP告警发送失败: {e}", module="alerting")
            return False


class ScriptChannel(AlertChannel):
    """脚本执行通道"""

    def __init__(self, script_path: str, timeout_sec: float = 10.0):
        self.script_path = script_path
        self.timeout_sec = timeout_sec

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            env = os.environ.copy()
            env["ALERT_RULE"] = event.rule_name
            env["ALERT_LEVEL"] = event.level.value
            env["ALERT_MESSAGE"] = event.message
            env["ALERT_VALUE"] = str(event.metric_value)
            env["ALERT_THRESHOLD"] = str(event.threshold)
            result = subprocess.run(
                [self.script_path],
                env=env,
                timeout=self.timeout_sec,
                capture_output=True,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning("ALERT", f"脚本告警执行失败: {e}", module="alerting")
            return False


class AlertManager:
    """告警管理器 —— 规则匹配、状态跟踪、多渠道分发"""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._channels: Dict[str, AlertChannel] = {
            "log": LogChannel(),
        }
        # 状态跟踪：rule_name -> {first_trigger_time, consecutive_count, last_alert_time}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._history: List[AlertEvent] = []
        self._max_history = 1000

    def register_rule(self, rule: AlertRule) -> None:
        """注册告警规则"""
        self._rules[rule.name] = rule
        logger.info("ALERT", f"告警规则注册: {rule.name}", module="alerting",
                    extra={"threshold": rule.threshold, "operator": rule.operator})

    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """添加通知通道"""
        self._channels[name] = channel
        logger.info("ALERT", f"通知通道添加: {name}", module="alerting")

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """评估条件"""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return False

    def _get_metric_value(self, state: Dict[str, Any], metric_path: str) -> Optional[float]:
        """从状态向量中提取指标值，支持嵌套路径如 'resource.cpu'"""
        keys = metric_path.split(".")
        current = state
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        if isinstance(current, (int, float)):
            return float(current)
        return None

    def check(self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[AlertEvent]:
        """检查所有规则，返回触发的告警事件"""
        triggered: List[AlertEvent] = []
        now = time.time()

        for name, rule in self._rules.items():
            if not rule.enabled:
                continue

            value = self._get_metric_value(state, rule.metric_path)
            if value is None:
                continue

            fired = self._evaluate_condition(value, rule.operator, rule.threshold)
            rule_state = self._state.setdefault(name, {
                "first_trigger_time": None,
                "consecutive_count": 0,
                "last_alert_time": 0,
            })

            if fired:
                rule_state["consecutive_count"] += 1
                if rule_state["first_trigger_time"] is None:
                    rule_state["first_trigger_time"] = now

                # 检查是否满足触发条件：持续时间 + 连续次数 + 静默期
                duration_ok = (now - rule_state["first_trigger_time"]) >= rule.duration_sec
                count_ok = rule_state["consecutive_count"] >= rule.consecutive_count
                cooldown_ok = (now - rule_state["last_alert_time"]) >= rule.cooldown_sec

                if duration_ok and count_ok and cooldown_ok:
                    event = AlertEvent(
                        rule_name=name,
                        level=rule.level,
                        message=f"{rule.metric_path}={value:.4f} 触发阈值 {rule.operator} {rule.threshold}",
                        timestamp=now,
                        metric_value=value,
                        threshold=rule.threshold,
                    )
                    self._dispatch(event, context)
                    triggered.append(event)
                    rule_state["last_alert_time"] = now
                    rule_state["consecutive_count"] = 0
                    rule_state["first_trigger_time"] = None
            else:
                # 未触发，重置所有触发状态
                rule_state["consecutive_count"] = 0
                rule_state["first_trigger_time"] = None

        return triggered

    def _dispatch(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> None:
        """分发告警到各通道"""
        rule = self._rules.get(event.rule_name)
        if not rule:
            return

        for channel_name in rule.channels:
            channel = self._channels.get(channel_name)
            if channel:
                try:
                    channel.send(event, context)
                except Exception as e:
                    logger.warning("ALERT", f"通道 {channel_name} 发送失败: {e}", module="alerting")

        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_history(self, limit: int = 100) -> List[AlertEvent]:
        """获取告警历史"""
        return self._history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            "rule_count": len(self._rules),
            "channel_count": len(self._channels),
            "history_count": len(self._history),
            "rules": {name: {"enabled": r.enabled, "level": r.level.value}
                      for name, r in self._rules.items()},
        }

    def silence_rule(self, name: str, duration_sec: float) -> bool:
        """手动静默某规则"""
        if name not in self._rules:
            return False
        self._state.setdefault(name, {})
        self._state[name]["last_alert_time"] = time.time() + duration_sec
        return True
