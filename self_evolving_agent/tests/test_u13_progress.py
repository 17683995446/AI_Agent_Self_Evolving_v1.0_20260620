"""
U13：进度可视化测试

验证目标：
1. is_tty() 检测TTY环境
2. ProgressConfig 配置（刷新间隔/显示项/精简模式）
3. ProgressBar.format_progress_line 格式化进度行
4. ProgressBar.format_summary_line 格式化摘要行
5. ProgressBar.update TTY模式用\r覆盖
6. ProgressBar.update 非TTY模式降级为定期摘要
7. ProgressBar.finish 输出最终状态
8. ProgressBar.clear 清除进度行
9. MetricsSummary 定期摘要
10. CLI run --progress/--no-progress 选项
11. 无回归

第一性原理：运行状态应可视化，而非翻日志。
TTY用进度条，非TTY降级为摘要，不产生乱码。
"""
import os
import sys
import io
import time
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU13IsTty(unittest.TestCase):
    """U13：TTY检测"""

    def test_is_tty_callable(self):
        from core.progress import is_tty
        self.assertTrue(callable(is_tty))

    def test_is_tty_returns_bool(self):
        from core.progress import is_tty
        result = is_tty()
        self.assertIsInstance(result, bool)


class TestU13ProgressConfig(unittest.TestCase):
    """U13：进度配置"""

    def test_default_config(self):
        from core.progress import ProgressConfig
        cfg = ProgressConfig()
        self.assertEqual(cfg.refresh_interval, 1.0)
        self.assertTrue(cfg.show_cpu)
        self.assertTrue(cfg.show_memory)
        self.assertTrue(cfg.show_drives)
        self.assertTrue(cfg.show_loop_count)
        self.assertTrue(cfg.show_uptime)
        self.assertFalse(cfg.compact)

    def test_custom_config(self):
        from core.progress import ProgressConfig
        cfg = ProgressConfig(refresh_interval=0.5, compact=True, show_drives=False)
        self.assertEqual(cfg.refresh_interval, 0.5)
        self.assertTrue(cfg.compact)
        self.assertFalse(cfg.show_drives)


class TestU13ProgressBarFormatting(unittest.TestCase):
    """U13：进度条格式化"""

    def setUp(self):
        from core.progress import ProgressBar, ProgressConfig
        self.bar = ProgressBar(
            config=ProgressConfig(),
            tty_checker=lambda: True,  # 模拟TTY
        )
        self.sample_status = {
            "running": True,
            "loop_count": 42,
            "uptime_human": "5分30秒",
            "resources": {"cpu_percent": 25.5, "memory_percent": 30.0},
            "drives": {"safe": 0.4, "gain": 0.4, "eff": 0.2},
        }

    def test_format_progress_line_returns_string(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIsInstance(result, str)

    def test_progress_line_has_running_icon(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIn("●", result)  # 运行中图标

    def test_progress_line_has_stopped_icon(self):
        status = dict(self.sample_status, running=False)
        result = self.bar.format_progress_line(status)
        self.assertIn("○", result)

    def test_progress_line_includes_loop_count(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIn("42", result)

    def test_progress_line_includes_cpu(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIn("CPU", result)

    def test_progress_line_includes_memory(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIn("内存", result)

    def test_progress_line_includes_drives(self):
        result = self.bar.format_progress_line(self.sample_status)
        self.assertIn("驱力", result)

    def test_compact_mode_omits_drives(self):
        from core.progress import ProgressBar, ProgressConfig
        bar = ProgressBar(config=ProgressConfig(compact=True))
        result = bar.format_progress_line(self.sample_status)
        self.assertNotIn("驱力", result)

    def test_format_summary_line_has_timestamp(self):
        result = self.bar.format_summary_line(self.sample_status)
        self.assertIn(":", result)  # 时间格式 HH:MM:SS

    def test_format_summary_line_includes_status(self):
        result = self.bar.format_summary_line(self.sample_status)
        self.assertIn("运行中", result)


class TestU13ProgressBarUpdate(unittest.TestCase):
    """U13：进度条更新"""

    def test_tty_mode_uses_carriage_return(self):
        """TTY模式用\r覆盖"""
        from core.progress import ProgressBar, ProgressConfig
        buf = io.StringIO()
        bar = ProgressBar(
            config=ProgressConfig(refresh_interval=0),
            output=buf,
            tty_checker=lambda: True,
        )
        status = {"running": True, "loop_count": 1, "uptime_human": "1秒",
                  "resources": {"cpu_percent": 10, "memory_percent": 20}, "drives": {}}
        bar.update(status, force=True)
        output = buf.getvalue()
        self.assertIn("\r", output)

    def test_non_tty_mode_outputs_summary(self):
        """非TTY模式输出摘要行"""
        from core.progress import ProgressBar, ProgressConfig
        buf = io.StringIO()
        bar = ProgressBar(
            config=ProgressConfig(refresh_interval=0),
            output=buf,
            tty_checker=lambda: False,
        )
        status = {"running": True, "loop_count": 5, "uptime_human": "5秒",
                  "resources": {"cpu_percent": 15, "memory_percent": 25}, "drives": {}}
        bar.update(status, force=True)
        output = buf.getvalue()
        self.assertIn("\n", output)
        self.assertIn("循环", output)

    def test_refresh_interval_respected(self):
        """刷新间隔内不重复刷新"""
        from core.progress import ProgressBar, ProgressConfig
        buf = io.StringIO()
        bar = ProgressBar(
            config=ProgressConfig(refresh_interval=10.0),
            output=buf,
            tty_checker=lambda: True,
        )
        status = {"running": True, "loop_count": 1, "uptime_human": "1秒",
                  "resources": {}, "drives": {}}
        bar.update(status)  # 第一次应刷新
        first_len = len(buf.getvalue())
        bar.update(status)  # 第二次应被间隔阻止
        self.assertEqual(len(buf.getvalue()), first_len)

    def test_force_overrides_interval(self):
        from core.progress import ProgressBar, ProgressConfig
        buf = io.StringIO()
        bar = ProgressBar(
            config=ProgressConfig(refresh_interval=10.0),
            output=buf,
            tty_checker=lambda: True,
        )
        status = {"running": True, "loop_count": 1, "uptime_human": "1秒",
                  "resources": {}, "drives": {}}
        bar.update(status)
        bar.update(status, force=True)  # 强制刷新
        self.assertGreater(len(buf.getvalue()), 0)


class TestU13ProgressBarFinish(unittest.TestCase):
    """U13：进度条结束"""

    def test_finish_outputs_newline_in_tty(self):
        from core.progress import ProgressBar, ProgressConfig
        buf = io.StringIO()
        bar = ProgressBar(
            config=ProgressConfig(),
            output=buf,
            tty_checker=lambda: True,
        )
        status = {"running": True, "loop_count": 1, "uptime_human": "1秒",
                  "resources": {}, "drives": {}}
        bar.update(status, force=True)
        bar.finish(status)
        output = buf.getvalue()
        self.assertIn("\n", output)
        self.assertIn("最终状态", output)

    def test_finish_without_status(self):
        from core.progress import ProgressBar
        buf = io.StringIO()
        bar = ProgressBar(output=buf, tty_checker=lambda: True)
        bar.finish()
        # 不应崩溃

    def test_clear_removes_line(self):
        from core.progress import ProgressBar
        buf = io.StringIO()
        bar = ProgressBar(output=buf, tty_checker=lambda: True)
        status = {"running": True, "loop_count": 1, "uptime_human": "1秒",
                  "resources": {}, "drives": {}}
        bar.update(status, force=True)
        bar.clear()
        output = buf.getvalue()
        self.assertIn("\r", output)


class TestU13MetricsSummary(unittest.TestCase):
    """U13：指标摘要器"""

    def test_metrics_summary_initializable(self):
        from core.progress import MetricsSummary
        ms = MetricsSummary(interval_seconds=1.0)
        self.assertIsNotNone(ms)

    def test_format_returns_string(self):
        from core.progress import MetricsSummary
        ms = MetricsSummary()
        status = {"loop_count": 10, "resources": {"cpu_percent": 20, "memory_percent": 30},
                  "drives": {"safe": 0.4, "gain": 0.4, "eff": 0.2},
                  "prediction": {"confidence": 0.8}}
        result = ms.format(status)
        self.assertIsInstance(result, str)
        self.assertIn("循环", result)

    def test_maybe_output_respects_interval(self):
        from core.progress import MetricsSummary
        ms = MetricsSummary(interval_seconds=10.0)
        status = {"loop_count": 1}
        first = ms.maybe_output(status)
        self.assertIsNotNone(first)
        second = ms.maybe_output(status)
        self.assertIsNone(second)  # 间隔内不输出

    def test_force_output(self):
        from core.progress import MetricsSummary
        ms = MetricsSummary(interval_seconds=10.0)
        status = {"loop_count": 1}
        ms.maybe_output(status)
        forced = ms.maybe_output(status, force=True)
        self.assertIsNotNone(forced)

    def test_history_recorded(self):
        from core.progress import MetricsSummary
        ms = MetricsSummary(interval_seconds=0)
        status = {"loop_count": 1}
        ms.maybe_output(status, force=True)
        ms.maybe_output(status, force=True)
        self.assertEqual(len(ms.history), 2)


class TestU13CliIntegration(unittest.TestCase):
    """U13：CLI进度条集成"""

    def test_cli_run_progress_default_true(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run"])
        self.assertTrue(args.progress)

    def test_cli_run_no_progress(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run", "--no-progress"])
        self.assertFalse(args.progress)

    def test_cli_run_progress_interval(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run", "--progress-interval", "0.5"])
        self.assertEqual(args.progress_interval, 0.5)

    def test_cli_run_progress_explicit(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["run", "--progress"])
        self.assertTrue(args.progress)


class TestU13NoRegression(unittest.TestCase):
    """U13：无回归保护"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        self.assertIsNotNone(agent.perception)

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        agent.start()
        import time as _time
        _time.sleep(1.5)
        agent.stop()
        self.assertGreaterEqual(agent.loop_count, 0)


if __name__ == "__main__":
    unittest.main()
