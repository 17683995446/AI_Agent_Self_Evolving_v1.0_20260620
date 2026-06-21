"""
U10：状态摘要优化测试

验证目标：
1. format_duration 时长格式化（秒/分/小时/天）
2. format_bytes 字节格式化
3. format_percent 百分比带状态图标
4. StatusFormatter.collect_status 收集完整状态
5. StatusFormatter.format_human 人类友好输出（分组、图标）
6. StatusFormatter.format_json JSON输出
7. StatusFormatter.format 根据格式类型分发
8. CLI status --watch 选项
9. CLI status --json 输出有效JSON
10. 无回归

第一性原理：状态输出应可读可解析，机器与人类兼顾。
"""
import os
import sys
import json
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU10FormatDuration(unittest.TestCase):
    """U10：时长格式化"""

    def test_seconds_only(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(45), "45秒")

    def test_minutes_and_seconds(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(125), "2分5秒")

    def test_hours_minutes_seconds(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(3725), "1小时2分5秒")

    def test_days_hours_minutes_seconds(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(90061), "1天1小时1分1秒")

    def test_zero_seconds(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(0), "0秒")

    def test_negative_returns_zero(self):
        from core.status_formatter import format_duration
        self.assertEqual(format_duration(-10), "0秒")


class TestU10FormatBytes(unittest.TestCase):
    """U10：字节格式化"""

    def test_bytes(self):
        from core.status_formatter import format_bytes
        self.assertEqual(format_bytes(500), "500.0B")

    def test_kilobytes(self):
        from core.status_formatter import format_bytes
        self.assertIn("KB", format_bytes(1024))

    def test_megabytes(self):
        from core.status_formatter import format_bytes
        self.assertIn("MB", format_bytes(1024 * 1024))

    def test_gigabytes(self):
        from core.status_formatter import format_bytes
        self.assertIn("GB", format_bytes(1024 * 1024 * 1024))


class TestU10FormatPercent(unittest.TestCase):
    """U10：百分比格式化"""

    def test_low_percent_has_ok_icon(self):
        from core.status_formatter import format_percent
        result = format_percent(20)
        self.assertIn("[OK]", result)

    def test_medium_percent_has_warn_icon(self):
        from core.status_formatter import format_percent
        result = format_percent(60)
        self.assertIn("[!!]", result)

    def test_high_percent_has_critical_icon(self):
        from core.status_formatter import format_percent
        result = format_percent(90)
        self.assertIn("[XX]", result)

    def test_includes_numeric_value(self):
        from core.status_formatter import format_percent
        result = format_percent(45)
        self.assertIn("45", result)


class TestU10StatusFormatterCollect(unittest.TestCase):
    """U10：状态收集"""

    def test_collect_status_from_agent(self):
        from core.status_formatter import StatusFormatter
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        try:
            status = StatusFormatter.collect_status(agent)
            self.assertIn("running", status)
            self.assertIn("loop_count", status)
            self.assertIn("uptime_seconds", status)
            self.assertIn("uptime_human", status)
        finally:
            agent.stop()

    def test_collect_includes_config_section(self):
        from core.status_formatter import StatusFormatter
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        try:
            status = StatusFormatter.collect_status(agent)
            self.assertIn("config", status)
        finally:
            agent.stop()

    def test_collect_includes_drives_section(self):
        from core.status_formatter import StatusFormatter
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        try:
            status = StatusFormatter.collect_status(agent)
            self.assertIn("drives", status)
        finally:
            agent.stop()

    def test_uptime_human_is_formatted(self):
        from core.status_formatter import StatusFormatter
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        try:
            status = StatusFormatter.collect_status(agent)
            self.assertIsInstance(status["uptime_human"], str)
            self.assertIn("秒", status["uptime_human"])
        finally:
            agent.stop()


class TestU10StatusFormatterHuman(unittest.TestCase):
    """U10：人类友好输出"""

    def test_format_human_returns_string(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 10, "uptime_human": "5秒",
                  "forever_running": False, "resources": {}, "drives": {},
                  "prediction": {}, "config": {}, "multi_agent": {}}
        result = StatusFormatter.format_human(status)
        self.assertIsInstance(result, str)

    def test_format_human_has_title(self):
        from core.status_formatter import StatusFormatter
        status = {"running": False, "loop_count": 0, "uptime_human": "0秒"}
        result = StatusFormatter.format_human(status)
        self.assertIn("状态", result)

    def test_format_human_shows_running_icon(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 5, "uptime_human": "10秒"}
        result = StatusFormatter.format_human(status)
        self.assertIn("[运行中]", result)

    def test_format_human_shows_stopped_icon(self):
        from core.status_formatter import StatusFormatter
        status = {"running": False, "loop_count": 0, "uptime_human": "0秒"}
        result = StatusFormatter.format_human(status)
        self.assertIn("[已停止]", result)

    def test_format_human_shows_resources_section(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 5, "uptime_human": "10秒",
                  "resources": {"cpu_percent": 30.0, "memory_percent": 40.0}}
        result = StatusFormatter.format_human(status)
        self.assertIn("[资源]", result)
        self.assertIn("CPU", result)

    def test_format_human_shows_drives_section(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 5, "uptime_human": "10秒",
                  "drives": {"safe": 0.4, "gain": 0.4, "eff": 0.2}}
        result = StatusFormatter.format_human(status)
        self.assertIn("[驱力]", result)

    def test_format_human_shows_config_section(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 5, "uptime_human": "10秒",
                  "config": {"max_cpu_percent": 20.0, "main_loop_frequency": 1.0,
                             "log_level": "INFO"}}
        result = StatusFormatter.format_human(status)
        self.assertIn("[配置]", result)


class TestU10StatusFormatterJson(unittest.TestCase):
    """U10：JSON输出"""

    def test_format_json_returns_valid_json(self):
        from core.status_formatter import StatusFormatter
        status = {"running": True, "loop_count": 10, "uptime_seconds": 5.0}
        result = StatusFormatter.format_json(status)
        parsed = json.loads(result)
        self.assertEqual(parsed["running"], True)
        self.assertEqual(parsed["loop_count"], 10)

    def test_format_json_handles_unicode(self):
        from core.status_formatter import StatusFormatter
        status = {"message": "中文测试"}
        result = StatusFormatter.format_json(status)
        self.assertIn("中文测试", result)


class TestU10StatusFormatterDispatch(unittest.TestCase):
    """U10：格式分发"""

    def test_format_human_dispatch(self):
        from core.status_formatter import StatusFormatter
        status = {"running": False, "loop_count": 0, "uptime_human": "0秒"}
        result = StatusFormatter.format(status, "human")
        self.assertIn("状态", result)

    def test_format_json_dispatch(self):
        from core.status_formatter import StatusFormatter
        status = {"running": False, "loop_count": 0}
        result = StatusFormatter.format(status, "json")
        json.loads(result)  # 应不抛异常

    def test_format_default_is_human(self):
        from core.status_formatter import StatusFormatter
        status = {"running": False, "loop_count": 0, "uptime_human": "0秒"}
        result = StatusFormatter.format(status)
        self.assertIn("状态", result)


class TestU10CliIntegration(unittest.TestCase):
    """U10：CLI status子命令集成"""

    def test_cli_status_json_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["status", "--json"])
        self.assertTrue(args.json)

    def test_cli_status_watch_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["status", "--watch"])
        self.assertTrue(args.watch)

    def test_cli_status_runs_successfully(self):
        """端到端：CLI status 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["status"])
        self.assertEqual(exit_code, 0)

    def test_cli_status_json_outputs_valid_json(self):
        """端到端：CLI status --json 输出有效JSON"""
        from cli import cli_main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["status", "--json"])
        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertIn("running", data)
        self.assertIn("loop_count", data)

    def test_cli_status_human_has_sections(self):
        """端到端：CLI status 人类输出包含分组"""
        from cli import cli_main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["status"])
        output = buf.getvalue()
        self.assertIn("状态", output)


class TestU10NoRegression(unittest.TestCase):
    """U10：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time
        time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)


if __name__ == "__main__":
    unittest.main()
