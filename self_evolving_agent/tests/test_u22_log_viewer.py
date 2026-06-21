"""
U22：日志查看与过滤 —— 日志易用性提升

测试覆盖：
1. LogEntry数据类与parse_log_line
2. LogReader（读取/过滤/搜索/统计）
3. LogManager（list/clear/latest）
4. 日志着色
5. CLI集成（logs子命令）
6. 无回归
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestU22LogEntry(unittest.TestCase):
    """测试LogEntry与parse_log_line"""

    def test_entry_importable(self):
        from core.log_viewer import LogEntry
        self.assertTrue(callable(LogEntry))

    def test_parse_plain_text(self):
        from core.log_viewer import parse_log_line
        entry = parse_log_line("hello world", 1)
        self.assertEqual(entry.message, "hello world")
        self.assertEqual(entry.line_number, 1)

    def test_parse_empty_line(self):
        from core.log_viewer import parse_log_line
        entry = parse_log_line("", 5)
        self.assertEqual(entry.raw, "")

    def test_parse_event_type_format(self):
        """[EVENT_TYPE] message 格式"""
        from core.log_viewer import parse_log_line
        entry = parse_log_line("[CONFIG_LOAD] 配置文件解析成功", 1)
        self.assertEqual(entry.event_type, "CONFIG_LOAD")
        self.assertIn("配置文件解析成功", entry.message)

    def test_parse_error_event_infers_error_level(self):
        from core.log_viewer import parse_log_line
        entry = parse_log_line("[CONFIG_ERROR] 配置错误", 1)
        self.assertEqual(entry.level, "ERROR")

    def test_parse_warning_event(self):
        from core.log_viewer import parse_log_line
        entry = parse_log_line("[PERF_WARN] 性能警告", 1)
        self.assertEqual(entry.level, "WARNING")

    def test_parse_with_timestamp_and_level(self):
        """[timestamp] [LEVEL] [module] message 格式"""
        from core.log_viewer import parse_log_line
        line = "[2026-06-20 17:10:14] [ERROR   ] [CLI] 命令执行失败"
        entry = parse_log_line(line, 10)
        self.assertEqual(entry.timestamp, "2026-06-20 17:10:14")
        self.assertEqual(entry.level, "ERROR")
        self.assertIn("命令执行失败", entry.message)

    def test_entry_to_dict(self):
        from core.log_viewer import LogEntry
        entry = LogEntry(raw="test", level="INFO", line_number=5)
        d = entry.to_dict()
        self.assertEqual(d["raw"], "test")
        self.assertEqual(d["level"], "INFO")


class TestU22LogReader(unittest.TestCase):
    """测试LogReader"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.tmpdir, "test.log")
        with open(self.log_file, "w") as f:
            f.write("[CONFIG_INIT] 配置初始化\n")
            f.write("[CONFIG_LOAD] 加载成功\n")
            f.write("[PERF_WARN] 性能警告\n")
            f.write("[RUNTIME_ERROR] 运行时错误\n")
            f.write("[INFO] 普通信息\n")
            f.write("[DEBUG] 调试信息\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reader_importable(self):
        from core.log_viewer import LogReader
        self.assertTrue(callable(LogReader))

    def test_exists(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        self.assertTrue(reader.exists())

    def test_not_exists(self):
        from core.log_viewer import LogReader
        reader = LogReader("/nonexistent/file.log")
        self.assertFalse(reader.exists())

    def test_read_all(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        entries = reader.read_all()
        self.assertEqual(len(entries), 6)

    def test_read_tail(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        entries = reader.read_tail(3)
        self.assertEqual(len(entries), 3)
        self.assertIn("调试信息", entries[-1].message)

    def test_read_head(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        entries = reader.read_head(2)
        self.assertEqual(len(entries), 2)
        self.assertIn("配置初始化", entries[0].message)

    def test_search(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        results = reader.search("配置")
        self.assertEqual(len(results), 1)
        self.assertIn("配置初始化", results[0].message)

    def test_search_case_insensitive(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        results = reader.search("ERROR")
        self.assertGreaterEqual(len(results), 1)

    def test_search_no_results(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        results = reader.search("nonexistent_keyword")
        self.assertEqual(len(results), 0)

    def test_filter_by_level(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        entries = reader.read_all()
        errors = reader.filter_by_level(entries, "ERROR")
        self.assertEqual(len(errors), 1)

    def test_filter_by_levels(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        entries = reader.read_all()
        filtered = reader.filter_by_levels(entries, ["ERROR", "WARNING"])
        self.assertEqual(len(filtered), 2)

    def test_count(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        self.assertEqual(reader.count(), 6)

    def test_count_by_level(self):
        from core.log_viewer import LogReader
        reader = LogReader(self.log_file)
        counts = reader.count_by_level()
        self.assertGreater(counts.get("ERROR", 0), 0)
        self.assertGreater(counts.get("WARNING", 0), 0)

    def test_read_all_nonexistent(self):
        from core.log_viewer import LogReader
        reader = LogReader("/nonexistent/file.log")
        self.assertEqual(reader.read_all(), [])


class TestU22LogManager(unittest.TestCase):
    """测试LogManager"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # 创建测试日志文件
        for i in range(5):
            path = os.path.join(self.tmpdir, f"agent_2026062{i}_100000.log")
            with open(path, "w") as f:
                f.write(f"[INFO] test log {i}\n")
        for i in range(2):
            path = os.path.join(self.tmpdir, f"audit_2026062{i}.log")
            with open(path, "w") as f:
                f.write(f"[AUDIT] test audit {i}\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_manager_importable(self):
        from core.log_viewer import LogManager
        self.assertTrue(callable(LogManager))

    def test_manager_instantiable(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        self.assertIsNotNone(mgr)

    def test_get_latest_log(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        latest = mgr.get_latest_log()
        self.assertIsNotNone(latest)
        self.assertIn("agent_", os.path.basename(latest))

    def test_list_agent_logs(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        logs = mgr.list_logs("agent")
        self.assertEqual(len(logs), 5)

    def test_list_audit_logs(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        logs = mgr.list_logs("audit")
        self.assertEqual(len(logs), 2)

    def test_list_all_logs(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        logs = mgr.list_logs("all")
        self.assertEqual(len(logs), 7)

    def test_list_logs_has_file_info(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        logs = mgr.list_logs("agent")
        self.assertIn("name", logs[0])
        self.assertIn("size", logs[0])
        self.assertIn("lines", logs[0])

    def test_clear_logs_keep_latest(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        deleted = mgr.clear_logs("agent", keep_latest=2)
        self.assertEqual(deleted, 3)
        remaining = mgr.list_logs("agent")
        self.assertEqual(len(remaining), 2)

    def test_clear_logs_keep_zero(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        deleted = mgr.clear_logs("agent", keep_latest=0)
        self.assertEqual(deleted, 5)

    def test_clear_audit_logs(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        deleted = mgr.clear_logs("audit", keep_latest=1)
        self.assertEqual(deleted, 1)

    def test_format_list(self):
        from core.log_viewer import LogManager
        mgr = LogManager(self.tmpdir)
        output = mgr.format_list("agent")
        self.assertIn("agent_", output)
        self.assertIn("日志文件", output)

    def test_format_list_empty(self):
        from core.log_viewer import LogManager
        empty_dir = tempfile.mkdtemp()
        try:
            mgr = LogManager(empty_dir)
            output = mgr.format_list("agent")
            self.assertIn("无", output)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_get_latest_log_empty(self):
        from core.log_viewer import LogManager
        empty_dir = tempfile.mkdtemp()
        try:
            mgr = LogManager(empty_dir)
            self.assertIsNone(mgr.get_latest_log())
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


class TestU22Colorize(unittest.TestCase):
    """测试日志着色"""

    def setUp(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def tearDown(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def test_colorize_log_entry_importable(self):
        from core.log_viewer import colorize_log_entry
        self.assertTrue(callable(colorize_log_entry))

    def test_colorize_disabled_returns_raw(self):
        from core.log_viewer import LogEntry, colorize_log_entry
        entry = LogEntry(raw="test", level="ERROR")
        result = colorize_log_entry(entry, color_enabled=False)
        self.assertEqual(result, "test")

    def test_colorize_no_level_returns_raw(self):
        from core.log_viewer import LogEntry, colorize_log_entry
        entry = LogEntry(raw="test", level=None)
        result = colorize_log_entry(entry, color_enabled=True)
        self.assertEqual(result, "test")

    def test_format_log_entries(self):
        from core.log_viewer import LogEntry, format_log_entries
        entries = [
            LogEntry(raw="line1", level="INFO", line_number=1),
            LogEntry(raw="line2", level="ERROR", line_number=2),
        ]
        result = format_log_entries(entries, color_enabled=False)
        self.assertIn("line1", result)
        self.assertIn("line2", result)

    def test_format_log_entries_with_line_numbers(self):
        from core.log_viewer import LogEntry, format_log_entries
        entries = [LogEntry(raw="test", level="INFO", line_number=42)]
        result = format_log_entries(entries, color_enabled=False, show_line_numbers=True)
        self.assertIn("42", result)


class TestU22CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_logs_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "list"])
        self.assertEqual(args.command, "logs")
        self.assertEqual(args.logs_action, "list")

    def test_logs_tail_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "tail"])
        self.assertEqual(args.logs_action, "tail")
        self.assertEqual(args.lines, 20)

    def test_logs_tail_with_lines(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "tail", "--lines", "50"])
        self.assertEqual(args.lines, 50)

    def test_logs_tail_with_level(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "tail", "--level", "ERROR"])
        self.assertEqual(args.level, "ERROR")

    def test_logs_tail_with_file(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "tail", "--file", "/tmp/test.log"])
        self.assertEqual(args.file, "/tmp/test.log")

    def test_logs_search_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "search", "ERROR"])
        self.assertEqual(args.logs_action, "search")
        self.assertEqual(args.keyword, "ERROR")

    def test_logs_clear_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "clear", "--keep", "3"])
        self.assertEqual(args.logs_action, "clear")
        self.assertEqual(args.keep, 3)

    def test_logs_list_type(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["logs", "list", "--type", "audit"])
        self.assertEqual(args.type, "audit")

    def test_cli_logs_list_returns_0(self):
        from cli import cli_main
        rc = cli_main(["logs", "list"])
        self.assertEqual(rc, 0)

    def test_cli_logs_tail_no_file_returns_1(self):
        """无日志文件时应返回1"""
        from cli import cli_main
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            rc = cli_main(["logs", "tail"])
            self.assertEqual(rc, 1)
            os.chdir(old_cwd)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestU22NoRegression(unittest.TestCase):
    """验证无回归"""

    def test_all_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        commands = ["run", "status", "health", "snapshot", "config", "init",
                    "doctor", "shell", "alias", "completion", "theme",
                    "history", "bookmark", "help", "logs"]
        for cmd in commands:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_help_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["help"])
        self.assertEqual(args.command, "help")


if __name__ == "__main__":
    unittest.main()
