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


class DingTalkChannel(AlertChannel):
    """钉钉机器人通道 —— 通过webhook发送markdown格式告警"""

    def __init__(self, webhook_url: str, secret: str = "", timeout_sec: float = 5.0):
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout_sec = timeout_sec

    def _build_payload(self, event: AlertEvent) -> Dict[str, Any]:
        """构造钉钉消息体（markdown格式）"""
        level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
        emoji = level_emoji.get(event.level.value, "⚠️")
        title = f"{emoji} Agent告警 [{event.level.value.upper()}]"
        text = (
            f"### {title}\n\n"
            f"- **规则**: {event.rule_name}\n"
            f"- **消息**: {event.message}\n"
            f"- **指标值**: {event.metric_value}\n"
            f"- **阈值**: {event.threshold}\n"
            f"- **时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}\n"
        )
        return {"msgtype": "markdown", "markdown": {"title": title, "text": text}}

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import urllib.request
            url = self.webhook_url
            # 钉钉签名（如果配置了secret）
            if self.secret:
                import hmac
                import hashlib
                import base64
                import urllib.parse
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{self.secret}"
                hmac_code = hmac.new(
                    self.secret.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    digestmod=hashlib.sha256,
                ).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                url = f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"

            payload = self._build_payload(event)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("errcode") == 0
        except Exception as e:
            logger.warning("ALERT", f"钉钉告警发送失败: {e}", module="alerting")
            return False


class FeishuChannel(AlertChannel):
    """飞书机器人通道 —— 通过webhook发送交互卡片告警"""

    def __init__(self, webhook_url: str, secret: str = "", timeout_sec: float = 5.0):
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout_sec = timeout_sec

    def _build_payload(self, event: AlertEvent) -> Dict[str, Any]:
        """构造飞书消息体（交互卡片）"""
        level_color = {"info": "blue", "warning": "orange", "error": "red", "critical": "red"}
        color = level_color.get(event.level.value, "orange")
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"Agent告警 [{event.level.value.upper()}]"},
                    "template": color,
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**规则**: {event.rule_name}"}},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**消息**: {event.message}"}},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**指标值**: {event.metric_value} / **阈值**: {event.threshold}"}},
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}"}},
                ],
            },
        }

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import urllib.request
            payload = self._build_payload(event)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("code") == 0 or result.get("StatusCode") == 0
        except Exception as e:
            logger.warning("ALERT", f"飞书告警发送失败: {e}", module="alerting")
            return False


class WeComChannel(AlertChannel):
    """企业微信机器人通道 —— 通过webhook发送markdown告警"""

    def __init__(self, webhook_url: str, timeout_sec: float = 5.0):
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec

    def _build_payload(self, event: AlertEvent) -> Dict[str, Any]:
        """构造企业微信消息体（markdown）"""
        level_tag = {"info": "info", "warning": "warning", "error": "error", "critical": "critical"}
        color = level_tag.get(event.level.value, "warning")
        content = (
            f"<font color=\"{color}\">Agent告警 [{event.level.value.upper()}]</font>\n"
            f">规则: {event.rule_name}\n"
            f">消息: {event.message}\n"
            f">指标值: {event.metric_value}\n"
            f">阈值: {event.threshold}\n"
            f">时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}"
        )
        return {"msgtype": "markdown", "markdown": {"content": content}}

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import urllib.request
            payload = self._build_payload(event)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("errcode") == 0
        except Exception as e:
            logger.warning("ALERT", f"企业微信告警发送失败: {e}", module="alerting")
            return False


class EmailChannel(AlertChannel):
    """邮件通道 —— 通过SMTP发送告警邮件"""

    def __init__(self, smtp_host: str, smtp_port: int, username: str,
                 password: str, from_addr: str, to_addrs: List[str],
                 use_tls: bool = True):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs if isinstance(to_addrs, list) else [to_addrs]
        self.use_tls = use_tls

    def send(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> bool:
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Agent告警-{event.level.value.upper()}] {event.rule_name}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            html = (
                f"<h2>Agent告警 [{event.level.value.upper()}]</h2>"
                f"<p><b>规则:</b> {event.rule_name}</p>"
                f"<p><b>消息:</b> {event.message}</p>"
                f"<p><b>指标值:</b> {event.metric_value}</p>"
                f"<p><b>阈值:</b> {event.threshold}</p>"
                f"<p><b>时间:</b> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}</p>"
            )
            msg.attach(MIMEText(html, "html", "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            return True
        except Exception as e:
            logger.warning("ALERT", f"邮件告警发送失败: {e}", module="alerting")
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

    def dispatch(self, event: AlertEvent, context: Optional[Dict[str, Any]] = None) -> None:
        """H3新增：公开的告警分发接口，供外部直接触发告警

        与_dispatch的区别：不依赖rule存在，直接向所有已注册通道分发
        """
        logger.info("ALERT", f"分发告警: [{event.level.value}] {event.message}", module="alerting",
                    extra={"rule": event.rule_name, "channels": list(self._channels.keys())})
        for channel_name, channel in self._channels.items():
            try:
                channel.send(event, context)
            except Exception as e:
                logger.warning("ALERT", f"通道 {channel_name} 发送失败: {e}", module="alerting")
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    @staticmethod
    def create_channel_from_config(config: Dict[str, Any]) -> Optional[AlertChannel]:
        """H3新增：从配置字典创建通道实例

        配置格式:
            {"type": "dingtalk", "webhook_url": "...", "secret": "..."}
            {"type": "feishu", "webhook_url": "...", "secret": "..."}
            {"type": "wecom", "webhook_url": "..."}
            {"type": "email", "smtp_host": "...", "smtp_port": 465, ...}
            {"type": "http", "webhook_url": "..."}
        """
        ctype = config.get("type", "")
        if ctype == "dingtalk":
            return DingTalkChannel(
                webhook_url=config.get("webhook_url", ""),
                secret=config.get("secret", ""),
                timeout_sec=config.get("timeout", 5.0),
            )
        elif ctype == "feishu":
            return FeishuChannel(
                webhook_url=config.get("webhook_url", ""),
                secret=config.get("secret", ""),
                timeout_sec=config.get("timeout", 5.0),
            )
        elif ctype == "wecom":
            return WeComChannel(
                webhook_url=config.get("webhook_url", ""),
                timeout_sec=config.get("timeout", 5.0),
            )
        elif ctype == "email":
            return EmailChannel(
                smtp_host=config.get("smtp_host", ""),
                smtp_port=config.get("smtp_port", 465),
                username=config.get("username", ""),
                password=config.get("password", ""),
                from_addr=config.get("from_addr", ""),
                to_addrs=config.get("to_addrs", []),
                use_tls=config.get("use_tls", True),
            )
        elif ctype == "http":
            return HTTPChannel(
                webhook_url=config.get("webhook_url", ""),
                timeout_sec=config.get("timeout", 5.0),
            )
        logger.warning("ALERT", f"未知通道类型: {ctype}", module="alerting")
        return None

    def setup_channels_from_config(self, channels_config: List[Dict[str, Any]]) -> int:
        """H3新增：从配置列表批量创建通道并注册

        Returns: 成功创建的通道数
        """
        count = 0
        for cfg in channels_config:
            channel = self.create_channel_from_config(cfg)
            if channel is not None:
                name = cfg.get("name", cfg.get("type", f"channel_{count}"))
                self.add_channel(name, channel)
                count += 1
        logger.info("ALERT", f"从配置创建 {count} 个告警通道", module="alerting")
        return count

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
