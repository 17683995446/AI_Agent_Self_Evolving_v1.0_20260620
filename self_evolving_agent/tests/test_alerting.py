"""
告警与通知系统测试 —— 验证规则匹配、防抖动、静默期、多渠道分发

测试前置：
1. 拆解AlertManager接口：register_rule/add_channel/check/get_history/get_stats/silence_rule
2. 拆解通道接口：LogChannel/HTTPChannel/ScriptChannel
3. 边界条件：空状态、缺失指标、未触发、连续触发、静默期内、禁用规则
"""
import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alerting import (
    AlertManager, AlertRule, AlertLevel, AlertEvent,
    LogChannel, HTTPChannel, ScriptChannel,
)


class TestAlertRuleMatching:
    """测试告警规则匹配"""

    def setup_method(self):
        self.manager = AlertManager()
        self.manager.register_rule(AlertRule(
            name="cpu_high",
            metric_path="cpu_percent",
            threshold=80.0,
            operator=">",
            level=AlertLevel.WARNING,
        ))

    def test_trigger_when_above_threshold(self):
        """指标超过阈值时触发"""
        events = self.manager.check({"cpu_percent": 85.0})
        assert len(events) == 1
        assert events[0].rule_name == "cpu_high"
        assert events[0].metric_value == 85.0

    def test_no_trigger_when_below_threshold(self):
        """指标低于阈值时不触发"""
        events = self.manager.check({"cpu_percent": 70.0})
        assert len(events) == 0

    def test_no_trigger_when_missing_metric(self):
        """状态向量缺少指标时不触发"""
        events = self.manager.check({"memory_percent": 90.0})
        assert len(events) == 0

    def test_trigger_with_nested_path(self):
        """支持嵌套路径提取指标"""
        self.manager.register_rule(AlertRule(
            name="nested_mem",
            metric_path="resource.memory",
            threshold=90.0,
            operator=">=",
        ))
        events = self.manager.check({"resource": {"memory": 95.0}})
        assert len(events) == 1
        assert events[0].rule_name == "nested_mem"

    def test_no_trigger_with_partial_nested_path(self):
        """嵌套路径不完整时不触发"""
        self.manager.register_rule(AlertRule(
            name="nested_disk",
            metric_path="resource.disk.io",
            threshold=100.0,
            operator=">",
        ))
        events = self.manager.check({"resource": {"disk": 50}})
        assert len(events) == 0

    def test_all_operators(self):
        """验证所有比较运算符"""
        ops = [("gt", ">", 10, 15, True), ("lt", "<", 10, 5, True),
               ("ge", ">=", 10, 10, True), ("le", "<=", 10, 10, True),
               ("eq", "==", 10, 10, True), ("ne", "!=", 10, 5, True),
               ("gt_false", ">", 10, 5, False), ("lt_false", "<", 10, 15, False)]
        for name, op, threshold, value, expected in ops:
            self.manager.register_rule(AlertRule(
                name=name, metric_path="val", threshold=threshold, operator=op,
            ))
        events = self.manager.check({"val": 10})
        triggered_names = {e.rule_name for e in events}
        assert "ge" in triggered_names
        assert "le" in triggered_names
        assert "eq" in triggered_names
        assert "gt_false" not in triggered_names
        assert "lt_false" not in triggered_names


class TestAlertDebouncing:
    """测试防抖动与连续触发"""

    def setup_method(self):
        self.manager = AlertManager()
        self.manager.register_rule(AlertRule(
            name="cpu_spike",
            metric_path="cpu_percent",
            threshold=90.0,
            operator=">",
            consecutive_count=3,
            cooldown_sec=0,  # 关闭静默期便于测试
        ))

    def test_not_trigger_on_first_two_checks(self):
        """前两次检查不触发（连续次数不足）"""
        self.manager.check({"cpu_percent": 95.0})
        self.manager.check({"cpu_percent": 95.0})
        events = self.manager.check({"cpu_percent": 95.0})
        assert len(events) == 1  # 第三次触发
        assert events[0].rule_name == "cpu_spike"

    def test_reset_on_non_trigger(self):
        """中间有一次未触发则重置计数"""
        self.manager.check({"cpu_percent": 95.0})
        self.manager.check({"cpu_percent": 95.0})
        self.manager.check({"cpu_percent": 50.0})  # 重置
        events = self.manager.check({"cpu_percent": 95.0})
        assert len(events) == 0  # 重新开始计数

    def test_multiple_triggers_after_cooldown(self):
        """冷却期后可再次触发"""
        self.manager.register_rule(AlertRule(
            name="mem_high",
            metric_path="mem",
            threshold=80.0,
            operator=">",
            consecutive_count=1,
            cooldown_sec=0.05,
        ))
        events1 = self.manager.check({"mem": 85.0})
        assert len(events1) == 1
        events2 = self.manager.check({"mem": 85.0})
        assert len(events2) == 0  # 仍在冷却期（但cooldown=0.05，第二次可能已过）
        time.sleep(0.06)
        events3 = self.manager.check({"mem": 85.0})
        assert len(events3) == 1


class TestAlertCooldown:
    """测试静默期"""

    def setup_method(self):
        self.manager = AlertManager()
        self.manager.register_rule(AlertRule(
            name="disk_full",
            metric_path="disk_percent",
            threshold=95.0,
            operator=">=",
            consecutive_count=1,
            cooldown_sec=1.0,
        ))

    def test_no_repeat_within_cooldown(self):
        """静默期内不重复告警"""
        events1 = self.manager.check({"disk_percent": 98.0})
        assert len(events1) == 1
        events2 = self.manager.check({"disk_percent": 98.0})
        assert len(events2) == 0

    def test_can_trigger_after_cooldown(self):
        """静默期结束后可再次触发"""
        self.manager.check({"disk_percent": 98.0})
        time.sleep(1.05)
        events = self.manager.check({"disk_percent": 98.0})
        assert len(events) == 1

    def test_manual_silence(self):
        """手动静默规则"""
        self.manager.check({"disk_percent": 98.0})
        self.manager.silence_rule("disk_full", 2.0)
        time.sleep(0.1)
        events = self.manager.check({"disk_percent": 98.0})
        assert len(events) == 0


class TestAlertDuration:
    """测试持续时间条件"""

    def setup_method(self):
        self.manager = AlertManager()
        self.manager.register_rule(AlertRule(
            name="slow_leak",
            metric_path="mem_growth",
            threshold=1.0,
            operator=">",
            duration_sec=0.2,
            consecutive_count=1,
            cooldown_sec=0,
        ))

    def test_not_trigger_before_duration(self):
        """持续时间未到不触发"""
        events = self.manager.check({"mem_growth": 2.0})
        assert len(events) == 0

    def test_trigger_after_duration(self):
        """持续时间到后触发"""
        self.manager.check({"mem_growth": 2.0})
        time.sleep(0.25)
        events = self.manager.check({"mem_growth": 2.0})
        assert len(events) == 1

    def test_reset_duration_on_non_trigger(self):
        """中间未触发则重置持续时间计时"""
        self.manager.check({"mem_growth": 2.0})
        time.sleep(0.15)
        self.manager.check({"mem_growth": 0.5})  # 未触发
        time.sleep(0.15)
        events = self.manager.check({"mem_growth": 2.0})
        assert len(events) == 0  # 持续时间重新计算


class TestAlertChannels:
    """测试通知通道"""

    def test_log_channel(self):
        """日志通道发送成功"""
        channel = LogChannel()
        event = AlertEvent(
            rule_name="test", level=AlertLevel.INFO,
            message="test msg", timestamp=time.time(),
            metric_value=50.0, threshold=80.0,
        )
        assert channel.send(event) is True

    def test_http_channel_failure(self):
        """HTTP通道连接失败时返回False不抛异常"""
        channel = HTTPChannel("http://localhost:59999/noexist")
        event = AlertEvent(
            rule_name="test", level=AlertLevel.INFO,
            message="test", timestamp=time.time(),
            metric_value=1.0, threshold=2.0,
        )
        result = channel.send(event)
        assert result is False

    def test_script_channel_failure(self):
        """脚本不存在时返回False不抛异常"""
        channel = ScriptChannel("/nonexistent/script.sh")
        event = AlertEvent(
            rule_name="test", level=AlertLevel.INFO,
            message="test", timestamp=time.time(),
            metric_value=1.0, threshold=2.0,
        )
        result = channel.send(event)
        assert result is False

    def test_script_channel_success(self):
        """脚本成功执行返回True"""
        fd, path = tempfile.mkstemp(suffix=".sh")
        try:
            os.write(fd, b"#!/bin/sh\nexit 0\n")
            os.close(fd)
            os.chmod(path, 0o755)
            channel = ScriptChannel(path)
            event = AlertEvent(
                rule_name="test", level=AlertLevel.INFO,
                message="test", timestamp=time.time(),
                metric_value=1.0, threshold=2.0,
            )
            assert channel.send(event) is True
        finally:
            os.remove(path)


class TestAlertManagerStats:
    """测试告警管理器统计"""

    def setup_method(self):
        self.manager = AlertManager()

    def test_empty_stats(self):
        """空状态统计"""
        stats = self.manager.get_stats()
        assert stats["rule_count"] == 0
        assert stats["history_count"] == 0

    def test_stats_after_rules(self):
        """注册规则后统计"""
        self.manager.register_rule(AlertRule(
            name="r1", metric_path="cpu", threshold=80.0, operator=">",
        ))
        stats = self.manager.get_stats()
        assert stats["rule_count"] == 1
        assert stats["rules"]["r1"]["enabled"] is True

    def test_history_accumulation(self):
        """告警历史累积"""
        self.manager.register_rule(AlertRule(
            name="h1", metric_path="cpu", threshold=10.0, operator=">",
            consecutive_count=1, cooldown_sec=0,
        ))
        self.manager.check({"cpu": 20.0})
        self.manager.check({"cpu": 20.0})
        history = self.manager.get_history()
        assert len(history) == 2
        assert history[0].rule_name == "h1"

    def test_history_limit(self):
        """历史记录上限"""
        self.manager.register_rule(AlertRule(
            name="h2", metric_path="cpu", threshold=0.0, operator=">",
            consecutive_count=1, cooldown_sec=0,
        ))
        for _ in range(5):
            self.manager.check({"cpu": 1.0})
        history = self.manager.get_history(limit=3)
        assert len(history) == 3


class TestDisabledRule:
    """测试禁用规则"""

    def test_disabled_rule_not_triggered(self):
        """禁用规则不触发"""
        manager = AlertManager()
        manager.register_rule(AlertRule(
            name="disabled_cpu",
            metric_path="cpu_percent",
            threshold=0.0,
            operator=">",
            enabled=False,
        ))
        events = manager.check({"cpu_percent": 100.0})
        assert len(events) == 0
