"""
U20：历史命令与书签 —— 重复操作回放

测试覆盖：
1. HistoryEntry数据类
2. CommandHistory（增删查、搜索、持久化）
3. Bookmark数据类
4. DEFAULT_BOOKMARKS预定义书签
5. BookmarkManager（增删查、预定义保护、持久化）
6. CLI集成（history/bookmark子命令）
7. 无回归
"""

import os
import sys
import json
import tempfile
import shutil
import time
import unittest
from unittest.mock import patch

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestU20HistoryEntry(unittest.TestCase):
    """测试HistoryEntry数据类"""

    def test_entry_importable(self):
        from core.history import HistoryEntry
        self.assertTrue(callable(HistoryEntry))

    def test_entry_instantiable(self):
        from core.history import HistoryEntry
        entry = HistoryEntry(command="status", timestamp=time.time())
        self.assertEqual(entry.command, "status")

    def test_entry_to_dict(self):
        from core.history import HistoryEntry
        entry = HistoryEntry(command="status", timestamp=1000.0, exit_code=0)
        d = entry.to_dict()
        self.assertEqual(d["command"], "status")
        self.assertEqual(d["timestamp"], 1000.0)

    def test_entry_from_dict(self):
        from core.history import HistoryEntry
        d = {"command": "run", "timestamp": 2000.0, "exit_code": 1}
        entry = HistoryEntry.from_dict(d)
        self.assertEqual(entry.command, "run")
        self.assertEqual(entry.exit_code, 1)

    def test_entry_format_time(self):
        from core.history import HistoryEntry
        entry = HistoryEntry(command="x", timestamp=time.time())
        time_str = entry.format_time()
        self.assertIsInstance(time_str, str)
        self.assertIn("-", time_str)  # 日期格式


class TestU20CommandHistory(unittest.TestCase):
    """测试CommandHistory"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.tmpdir, "history.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_history_importable(self):
        from core.history import CommandHistory
        self.assertTrue(callable(CommandHistory))

    def test_history_instantiable(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        self.assertIsNotNone(history)

    def test_add_command(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("status")
        self.assertEqual(history.count(), 1)

    def test_add_multiple(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("status")
        history.add("health")
        history.add("config show")
        self.assertEqual(history.count(), 3)

    def test_add_empty_ignored(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("")
        history.add("   ")
        self.assertEqual(history.count(), 0)

    def test_list(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        for cmd in ["a", "b", "c"]:
            history.add(cmd)
        entries = history.list()
        self.assertEqual(len(entries), 3)

    def test_list_with_limit(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        for i in range(10):
            history.add(f"cmd{i}")
        entries = history.list(limit=5)
        self.assertEqual(len(entries), 5)

    def test_search(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("status")
        history.add("health --detail")
        history.add("config show")
        results = history.search("status")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].command, "status")

    def test_search_case_insensitive(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("STATUS")
        results = history.search("status")
        self.assertEqual(len(results), 1)

    def test_search_no_results(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("status")
        results = history.search("nonexistent")
        self.assertEqual(len(results), 0)

    def test_search_by_prefix(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("config show")
        history.add("config validate")
        history.add("status")
        results = history.search_by_prefix("config")
        self.assertEqual(len(results), 2)

    def test_get_last(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("first")
        history.add("second")
        last = history.get_last()
        self.assertIsNotNone(last)
        self.assertEqual(last.command, "second")

    def test_get_last_empty(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        self.assertIsNone(history.get_last())

    def test_get_by_index(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("first")
        history.add("second")
        entry = history.get(0)
        self.assertEqual(entry.command, "first")

    def test_get_by_negative_index(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("first")
        history.add("second")
        entry = history.get(-1)
        self.assertEqual(entry.command, "second")

    def test_clear(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("a")
        history.add("b")
        history.clear()
        self.assertEqual(history.count(), 0)

    def test_persistence(self):
        """测试持久化"""
        from core.history import CommandHistory
        # 写入
        history1 = CommandHistory(history_file=self.history_file)
        history1.add("persisted_command")
        # 重新加载
        history2 = CommandHistory(history_file=self.history_file)
        self.assertEqual(history2.count(), 1)
        self.assertEqual(history2.get(0).command, "persisted_command")

    def test_max_entries_limit(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file, max_entries=5)
        for i in range(10):
            history.add(f"cmd{i}")
        self.assertEqual(history.count(), 5)
        # 应保留最后5条
        entries = history.list()
        self.assertEqual(entries[0].command, "cmd5")

    def test_format_list(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        history.add("status")
        output = history.format_list()
        self.assertIn("命令历史", output)
        self.assertIn("status", output)

    def test_format_list_empty(self):
        from core.history import CommandHistory
        history = CommandHistory(history_file=self.history_file)
        output = history.format_list()
        self.assertIn("无历史记录", output)


class TestU20Bookmark(unittest.TestCase):
    """测试Bookmark数据类"""

    def test_bookmark_importable(self):
        from core.history import Bookmark
        self.assertTrue(callable(Bookmark))

    def test_bookmark_instantiable(self):
        from core.history import Bookmark
        bm = Bookmark(name="test", command="status")
        self.assertEqual(bm.name, "test")
        self.assertEqual(bm.command, "status")

    def test_bookmark_to_dict(self):
        from core.history import Bookmark
        bm = Bookmark(name="test", command="status", description="测试")
        d = bm.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["command"], "status")

    def test_bookmark_from_dict(self):
        from core.history import Bookmark
        d = {"name": "test", "command": "run", "description": "", "created_at": 1000.0, "is_default": False}
        bm = Bookmark.from_dict(d)
        self.assertEqual(bm.name, "test")
        self.assertEqual(bm.command, "run")


class TestU20DefaultBookmarks(unittest.TestCase):
    """测试预定义书签"""

    def test_default_bookmarks_defined(self):
        from core.history import DEFAULT_BOOKMARKS
        self.assertIsInstance(DEFAULT_BOOKMARKS, dict)
        self.assertGreater(len(DEFAULT_BOOKMARKS), 0)

    def test_has_start_bookmark(self):
        from core.history import DEFAULT_BOOKMARKS
        self.assertIn("start", DEFAULT_BOOKMARKS)

    def test_has_check_bookmark(self):
        from core.history import DEFAULT_BOOKMARKS
        self.assertIn("check", DEFAULT_BOOKMARKS)

    def test_start_bookmark_runs_forever(self):
        from core.history import DEFAULT_BOOKMARKS
        self.assertIn("forever", DEFAULT_BOOKMARKS["start"])


class TestU20BookmarkManager(unittest.TestCase):
    """测试BookmarkManager"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bookmarks_file = os.path.join(self.tmpdir, "bookmarks.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_manager_importable(self):
        from core.history import BookmarkManager
        self.assertTrue(callable(BookmarkManager))

    def test_manager_instantiable(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        self.assertIsNotNone(mgr)

    def test_has_default_bookmarks(self):
        from core.history import BookmarkManager, DEFAULT_BOOKMARKS
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        for name in DEFAULT_BOOKMARKS:
            self.assertTrue(mgr.exists(name), f"缺少预定义书签: {name}")

    def test_add_user_bookmark(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        success = mgr.add("myrun", "run --forever")
        self.assertTrue(success)
        self.assertTrue(mgr.exists("myrun"))

    def test_cannot_override_default(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        success = mgr.add("start", "custom command")
        self.assertFalse(success)

    def test_remove_user_bookmark(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        mgr.add("myrun", "run --forever")
        success = mgr.remove("myrun")
        self.assertTrue(success)
        self.assertFalse(mgr.exists("myrun"))

    def test_cannot_remove_default(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        success = mgr.remove("start")
        self.assertFalse(success)

    def test_remove_nonexistent(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        success = mgr.remove("nonexistent")
        self.assertFalse(success)

    def test_get_bookmark(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        bm = mgr.get("start")
        self.assertIsNotNone(bm)
        self.assertEqual(bm.name, "start")

    def test_get_nonexistent(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        self.assertIsNone(mgr.get("nonexistent"))

    def test_get_command(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        cmd = mgr.get_command("start")
        self.assertIsNotNone(cmd)
        self.assertIn("run", cmd)

    def test_list_all(self):
        from core.history import BookmarkManager, DEFAULT_BOOKMARKS
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        all_bms = mgr.list_all()
        self.assertGreaterEqual(len(all_bms), len(DEFAULT_BOOKMARKS))

    def test_list_defaults(self):
        from core.history import BookmarkManager, DEFAULT_BOOKMARKS
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        defaults = mgr.list_defaults()
        self.assertEqual(len(defaults), len(DEFAULT_BOOKMARKS))

    def test_list_user_empty(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        user_bms = mgr.list_user()
        self.assertEqual(len(user_bms), 0)

    def test_list_user_with_custom(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        mgr.add("custom1", "status")
        mgr.add("custom2", "health")
        user_bms = mgr.list_user()
        self.assertEqual(len(user_bms), 2)

    def test_persistence(self):
        """测试用户书签持久化"""
        from core.history import BookmarkManager
        # 添加用户书签
        mgr1 = BookmarkManager(bookmarks_file=self.bookmarks_file)
        mgr1.add("persisted", "config show")
        # 重新加载
        mgr2 = BookmarkManager(bookmarks_file=self.bookmarks_file)
        self.assertTrue(mgr2.exists("persisted"))
        self.assertEqual(mgr2.get_command("persisted"), "config show")

    def test_format_list(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        output = mgr.format_list()
        self.assertIn("命令书签", output)
        self.assertIn("start", output)
        self.assertIn("预定义书签", output)

    def test_format_list_with_user_bookmarks(self):
        from core.history import BookmarkManager
        mgr = BookmarkManager(bookmarks_file=self.bookmarks_file)
        mgr.add("mycmd", "status", description="我的命令")
        output = mgr.format_list()
        self.assertIn("用户书签", output)
        self.assertIn("mycmd", output)


class TestU20CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_history_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["history", "list"])
        self.assertEqual(args.command, "history")
        self.assertEqual(args.history_action, "list")

    def test_history_search_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["history", "search", "status"])
        self.assertEqual(args.history_action, "search")
        self.assertEqual(args.keyword, "status")

    def test_history_clear_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["history", "clear"])
        self.assertEqual(args.history_action, "clear")

    def test_history_list_limit(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["history", "list", "--limit", "50"])
        self.assertEqual(args.limit, 50)

    def test_bookmark_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["bookmark", "list"])
        self.assertEqual(args.command, "bookmark")
        self.assertEqual(args.bookmark_action, "list")

    def test_bookmark_add_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["bookmark", "add", "myrun", "run --forever"])
        self.assertEqual(args.bookmark_action, "add")
        self.assertEqual(args.name, "myrun")
        self.assertEqual(args.command, "run --forever")

    def test_bookmark_remove_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["bookmark", "remove", "myrun"])
        self.assertEqual(args.bookmark_action, "remove")
        self.assertEqual(args.name, "myrun")

    def test_bookmark_run_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["bookmark", "run", "start"])
        self.assertEqual(args.bookmark_action, "run")
        self.assertEqual(args.name, "start")

    def test_cli_history_list_returns_0(self):
        from cli import cli_main
        rc = cli_main(["history", "list"])
        self.assertEqual(rc, 0)

    def test_cli_history_search_returns_0(self):
        from cli import cli_main
        rc = cli_main(["history", "search", "test"])
        self.assertEqual(rc, 0)

    def test_cli_bookmark_list_returns_0(self):
        from cli import cli_main
        rc = cli_main(["bookmark", "list"])
        self.assertEqual(rc, 0)


class TestU20NoRegression(unittest.TestCase):
    """验证无回归"""

    def test_all_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        commands = ["run", "status", "health", "snapshot", "config", "init",
                    "doctor", "shell", "alias", "completion", "theme",
                    "history", "bookmark"]
        for cmd in commands:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_config_subcommands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for action in ["validate", "show", "reload", "edit", "fields"]:
            args = parser.parse_args(["config", action])
            self.assertEqual(args.config_action, action)

    def test_theme_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "list"])
        self.assertEqual(args.command, "theme")


if __name__ == "__main__":
    unittest.main()
