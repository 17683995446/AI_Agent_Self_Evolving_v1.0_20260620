"""
U23：告警管理 —— 告警CLI辅助

第一性原理：
- 告警系统功能完整但CLI完全空白
- 用户需要查看告警规则、历史、统计
- 需要能静默规则、测试通道
- 告警数据应可格式化展示

设计：
1. AlertCliHelper：告警CLI辅助器
   - 从配置加载规则（不启动Agent）
   - 格式化规则列表/历史/统计
   - 静默规则
   - 测试通道
2. 复用AlertManager的核心能力
"""

import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import asdict


class AlertCliHelper:
    """告警CLI辅助器

    轻量级访问告警配置，不启动完整Agent
    """

    def __init__(self):
        from core.alerting import AlertManager
        self.manager = AlertManager()
        self._load_default_rules()

    def _load_default_rules(self):
        """加载默认告警规则（与main.py一致）"""
        from core.alerting import AlertRule, AlertLevel
        try:
            self.manager.register_rule(AlertRule(
                name="cpu_high",
                metric_path="cpu_percent",
                threshold=0.8,
                operator=">",
                duration_sec=10,
                level=AlertLevel.WARNING,
                channels=["log"],
            ))
            self.manager.register_rule(AlertRule(
                name="memory_high",
                metric_path="memory_percent",
                threshold=0.8,
                operator=">",
                duration_sec=10,
                level=AlertLevel.ERROR,
                channels=["log"],
            ))
        except Exception:
            pass

        # 尝试从配置加载通道
        try:
            from core.config import config_manager
            config_manager.load()
            alert_channels_config = getattr(config_manager.config, "alert_channels", [])
            if alert_channels_config:
                self.manager.setup_channels_from_config(alert_channels_config)
        except Exception:
            pass

    def list_rules(self) -> List[Dict[str, Any]]:
        """列出所有告警规则"""
        rules = []
        for name, rule in self.manager._rules.items():
            rules.append({
                "name": rule.name,
                "metric": rule.metric_path,
                "threshold": rule.threshold,
                "operator": rule.operator,
                "level": rule.level.value,
                "duration_sec": rule.duration_sec,
                "cooldown_sec": rule.cooldown_sec,
                "channels": rule.channels,
                "enabled": rule.enabled,
            })
        return rules

    def list_channels(self) -> List[Dict[str, Any]]:
        """列出所有告警通道"""
        channels = []
        for name, channel in self.manager._channels.items():
            channels.append({
                "name": name,
                "type": type(channel).__name__,
            })
        return channels

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警历史"""
        history = self.manager.get_history(limit)
        return [
            {
                "rule_name": e.rule_name,
                "level": e.level.value,
                "message": e.message,
                "timestamp": e.timestamp,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp)),
                "metric_value": e.metric_value,
                "threshold": e.threshold,
            }
            for e in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        return self.manager.get_stats()

    def silence_rule(self, name: str, duration_sec: float) -> bool:
        """静默规则"""
        return self.manager.silence_rule(name, duration_sec)

    def test_rule(self, rule_name: str, value: float) -> Dict[str, Any]:
        """测试规则是否会被触发

        Args:
            rule_name: 规则名称
            value: 模拟的指标值

        Returns:
            测试结果
        """
        rule = self.manager._rules.get(rule_name)
        if rule is None:
            return {"exists": False, "triggered": False, "message": f"规则不存在: {rule_name}"}

        triggered = self.manager._evaluate_condition(value, rule.operator, rule.threshold)
        return {
            "exists": True,
            "rule_name": rule_name,
            "metric": rule.metric_path,
            "value": value,
            "threshold": rule.threshold,
            "operator": rule.operator,
            "triggered": triggered,
            "level": rule.level.value,
            "message": f"指标 {rule.metric_path}={value} {rule.operator} {rule.threshold} -> {'触发' if triggered else '未触发'}",
        }

    def format_rules_list(self) -> str:
        """格式化规则列表"""
        rules = self.list_rules()
        if not rules:
            return "（无告警规则）"
        lines = [f"=== 告警规则（{len(rules)}条）==="]
        lines.append(f"{'名称':15s} {'指标':15s} {'条件':10s} {'阈值':8s} {'级别':10s} {'启用':6s} {'通道'}")
        lines.append("-" * 90)
        for r in rules:
            condition = f"{r['operator']} {r['threshold']}"
            lines.append(
                f"{r['name']:15s} {r['metric']:15s} {r['operator']:10s} {r['threshold']:8.2f} "
                f"{r['level']:10s} {'是' if r['enabled'] else '否':6s} {','.join(r['channels'])}"
            )
        return "\n".join(lines)

    def format_channels_list(self) -> str:
        """格式化通道列表"""
        channels = self.list_channels()
        if not channels:
            return "（无告警通道）"
        lines = [f"=== 告警通道（{len(channels)}个）==="]
        for ch in channels:
            lines.append(f"  {ch['name']:15s} {ch['type']}")
        return "\n".join(lines)

    def format_history(self, limit: int = 50) -> str:
        """格式化告警历史"""
        history = self.get_history(limit)
        if not history:
            return "（无告警历史）"
        lines = [f"=== 告警历史（最近{len(history)}条）==="]
        for h in history:
            lines.append(
                f"  [{h['time_str']}] [{h['level'].upper()}] {h['rule_name']}: {h['message']} "
                f"(值={h['metric_value']:.2f}, 阈值={h['threshold']:.2f})"
            )
        return "\n".join(lines)

    def format_stats(self) -> str:
        """格式化统计"""
        stats = self.get_stats()
        lines = ["=== 告警统计 ==="]
        lines.append(f"  规则数: {stats['rule_count']}")
        lines.append(f"  通道数: {stats['channel_count']}")
        lines.append(f"  历史告警数: {stats['history_count']}")
        if stats.get("rules"):
            lines.append("  规则详情:")
            for name, info in stats["rules"].items():
                status = "启用" if info["enabled"] else "禁用"
                lines.append(f"    {name}: {info['level']} [{status}]")
        return "\n".join(lines)

    def format_test_result(self, result: Dict[str, Any]) -> str:
        """格式化测试结果"""
        if not result.get("exists"):
            return result.get("message", "规则不存在")
        lines = ["=== 规则测试结果 ==="]
        lines.append(f"  规则: {result['rule_name']}")
        lines.append(f"  指标: {result['metric']}")
        lines.append(f"  模拟值: {result['value']}")
        lines.append(f"  阈值: {result['threshold']} (操作符: {result['operator']})")
        lines.append(f"  级别: {result['level']}")
        triggered = result["triggered"]
        lines.append(f"  结果: {'会触发告警' if triggered else '不会触发告警'}")
        return "\n".join(lines)
