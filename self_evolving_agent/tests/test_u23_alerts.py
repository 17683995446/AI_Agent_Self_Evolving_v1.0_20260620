"""
U23：告警管理 —— 告警CLI辅助

测试覆盖：
1. AlertCliHelper基础（实例化、规则加载）
2. 规则列表（list_rules/format_rules_list）
3. 通道列表（list_channels/format_channels_list）
4. 告警历史（get_history/format_history）
5. 统计（get_stats/format_stats）
6. 静默（silence_rule）
7. 测试规则（test_rule/format_test_result）
8. CLI集成（alerts子命令）
9. 无回归
"""

import os
import sys
import unittest
from unittest.mock import patch

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestU23AlertCliHelper(unittest.TestCase):
    """测试AlertCliHelper基础"""

    def test_helper_importable(self):
        from core.alert_cli import AlertCliHelper
        self.assertTrue(callable(AlertCliHelper))

    def test_helper_instantiable(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        self.assertIsNotNone(helper)

    def test_helper_has_manager(self):
        from core.alert_cli import AlertCliHelper
        from core.alerting import AlertManager
        helper = AlertCliHelper()
        self.assertIsInstance(helper.manager, AlertManager)

    def test_default_rules_loaded(self):
        """默认应加载cpu_high和memory_high规则"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        rules = helper.list_rules()
        rule_names = [r["name"] for r in rules]
        self.assertIn("cpu_high", rule_names)
        self.assertIn("memory_high", rule_names)


class TestU23RulesList(unittest.TestCase):
    """测试规则列表"""

    def test_list_rules_returns_list(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        rules = helper.list_rules()
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)

    def test_list_rules_has_required_fields(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        rules = helper.list_rules()
        for r in rules:
            self.assertIn("name", r)
            self.assertIn("metric", r)
            self.assertIn("threshold", r)
            self.assertIn("operator", r)
            self.assertIn("level", r)
            self.assertIn("enabled", r)
            self.assertIn("channels", r)

    def test_cpu_high_rule_details(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        rules = helper.list_rules()
        cpu_rule = next(r for r in rules if r["name"] == "cpu_high")
        self.assertEqual(cpu_rule["metric"], "cpu_percent")
        self.assertEqual(cpu_rule["operator"], ">")
        self.assertEqual(cpu_rule["level"], "warning")

    def test_format_rules_list(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        output = helper.format_rules_list()
        self.assertIn("告警规则", output)
        self.assertIn("cpu_high", output)
        self.assertIn("memory_high", output)


class TestU23Channels(unittest.TestCase):
    """测试通道列表"""

    def test_list_channels(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        channels = helper.list_channels()
        self.assertIsInstance(channels, list)
        self.assertGreater(len(channels), 0)

    def test_has_log_channel(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        channels = helper.list_channels()
        names = [ch["name"] for ch in channels]
        self.assertIn("log", names)

    def test_format_channels_list(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        output = helper.format_channels_list()
        self.assertIn("告警通道", output)
        self.assertIn("log", output)


class TestU23History(unittest.TestCase):
    """测试告警历史"""

    def test_get_history_empty(self):
        """无告警时应返回空列表"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        history = helper.get_history()
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 0)

    def test_format_history_empty(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        output = helper.format_history()
        self.assertIn("无告警历史", output)

    def test_get_history_with_events(self):
        """添加告警事件后应能获取"""
        from core.alert_cli import AlertCliHelper
        from core.alerting import AlertEvent, AlertLevel
        helper = AlertCliHelper()
        # 手动添加历史
        helper.manager._history.append(AlertEvent(
            rule_name="test_rule",
            level=AlertLevel.WARNING,
            message="测试告警",
            timestamp=1000.0,
            metric_value=0.9,
            threshold=0.8,
        ))
        history = helper.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["rule_name"], "test_rule")

    def test_format_history_with_events(self):
        from core.alert_cli import AlertCliHelper
        from core.alerting import AlertEvent, AlertLevel
        helper = AlertCliHelper()
        helper.manager._history.append(AlertEvent(
            rule_name="test_rule",
            level=AlertLevel.ERROR,
            message="测试告警",
            timestamp=1000.0,
            metric_value=0.9,
            threshold=0.8,
        ))
        output = helper.format_history()
        self.assertIn("告警历史", output)
        self.assertIn("test_rule", output)


class TestU23Stats(unittest.TestCase):
    """测试统计"""

    def test_get_stats(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        stats = helper.get_stats()
        self.assertIn("rule_count", stats)
        self.assertIn("channel_count", stats)
        self.assertIn("history_count", stats)
        self.assertGreater(stats["rule_count"], 0)

    def test_format_stats(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        output = helper.format_stats()
        self.assertIn("告警统计", output)
        self.assertIn("规则数", output)
        self.assertIn("通道数", output)


class TestU23Silence(unittest.TestCase):
    """测试静默"""

    def test_silence_existing_rule(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        success = helper.silence_rule("cpu_high", 300)
        self.assertTrue(success)

    def test_silence_nonexistent_rule(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        success = helper.silence_rule("nonexistent", 300)
        self.assertFalse(success)


class TestU23TestRule(unittest.TestCase):
    """测试规则测试"""

    def test_test_rule_triggered(self):
        """值超过阈值应触发"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("cpu_high", 0.9)
        self.assertTrue(result["exists"])
        self.assertTrue(result["triggered"])

    def test_test_rule_not_triggered(self):
        """值低于阈值不应触发"""
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("cpu_high", 0.5)
        self.assertTrue(result["exists"])
        self.assertFalse(result["triggered"])

    def test_test_rule_nonexistent(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("nonexistent", 0.9)
        self.assertFalse(result["exists"])

    def test_format_test_result_triggered(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("cpu_high", 0.9)
        output = helper.format_test_result(result)
        self.assertIn("规则测试结果", output)
        self.assertIn("会触发", output)

    def test_format_test_result_not_triggered(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("cpu_high", 0.5)
        output = helper.format_test_result(result)
        self.assertIn("不会触发", output)

    def test_format_test_result_nonexistent(self):
        from core.alert_cli import AlertCliHelper
        helper = AlertCliHelper()
        result = helper.test_rule("nonexistent", 0.9)
        output = helper.format_test_result(result)
        self.assertIn("不存在", output)


class TestU23CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_alerts_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "rules"])
        self.assertEqual(args.command, "alerts")
        self.assertEqual(args.alerts_action, "rules")

    def test_alerts_channels_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "channels"])
        self.assertEqual(args.alerts_action, "channels")

    def test_alerts_history_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "history"])
        self.assertEqual(args.alerts_action, "history")

    def test_alerts_history_with_limit(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "history", "--limit", "10"])
        self.assertEqual(args.limit, 10)

    def test_alerts_stats_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "stats"])
        self.assertEqual(args.alerts_action, "stats")

    def test_alerts_silence_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "silence", "cpu_high", "300"])
        self.assertEqual(args.alerts_action, "silence")
        self.assertEqual(args.name, "cpu_high")
        self.assertEqual(args.duration, 300.0)

    def test_alerts_test_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alerts", "test", "cpu_high", "0.9"])
        self.assertEqual(args.alerts_action, "test")
        self.assertEqual(args.name, "cpu_high")
        self.assertEqual(args.value, 0.9)

    def test_cli_alerts_rules_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "rules"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_channels_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "channels"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_stats_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "stats"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_history_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "history"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_silence_valid_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "silence", "cpu_high", "60"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_silence_invalid_returns_1(self):
        from cli import cli_main
        rc = cli_main(["alerts", "silence", "nonexistent", "60"])
        self.assertEqual(rc, 1)

    def test_cli_alerts_test_valid_returns_0(self):
        from cli import cli_main
        rc = cli_main(["alerts", "test", "cpu_high", "0.9"])
        self.assertEqual(rc, 0)

    def test_cli_alerts_test_invalid_returns_1(self):
        from cli import cli_main
        rc = cli_main(["alerts", "test", "nonexistent", "0.9"])
        self.assertEqual(rc, 1)


class TestU23NoRegression(unittest.TestCase):
    """验证无回归"""

    def test_all_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        commands = ["run", "status", "health", "snapshot", "config", "init",
                    "doctor", "shell", "alias", "completion", "theme",
                    "history", "bookmark", "help", "logs", "alerts"]
        for cmd in commands:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_logs_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "list"])
        self.assertEqual(args.command, "logs")


if __name__ == "__main__":
    unittest.main()
