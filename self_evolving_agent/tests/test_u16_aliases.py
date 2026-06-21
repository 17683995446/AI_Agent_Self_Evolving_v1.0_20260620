"""
U16：命令别名与快捷方式测试

验证目标：
1. DEFAULT_ALIASES 包含常用别名
2. AliasManager 类存在且可实例化
3. resolve() 正确展开别名
4. 链式展开（别名指向别名）
5. 防环检测
6. add() 添加自定义别名
7. remove() 删除自定义别名
8. 不允许覆盖预定义别名
9. 用户别名持久化
10. CLI alias子命令集成
11. CLI命令前别名解析生效
12. 无回归

第一性原理：高频操作应有短别名，减少按键成本。
"""
import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU16DefaultAliases(unittest.TestCase):
    """U16：预定义别名"""

    def test_default_aliases_exist(self):
        from core.aliases import DEFAULT_ALIASES
        self.assertIsInstance(DEFAULT_ALIASES, dict)
        self.assertGreater(len(DEFAULT_ALIASES), 5)

    def test_common_shortcuts_defined(self):
        from core.aliases import DEFAULT_ALIASES
        # 常用短别名应存在
        self.assertEqual(DEFAULT_ALIASES["s"], "status")
        self.assertEqual(DEFAULT_ALIASES["h"], "health")
        self.assertEqual(DEFAULT_ALIASES["c"], "config")
        self.assertEqual(DEFAULT_ALIASES["d"], "doctor")

    def test_default_aliases_are_valid_commands(self):
        """预定义别名应指向有效命令"""
        from core.aliases import DEFAULT_ALIASES
        valid_commands = {"run", "status", "health", "snapshot", "config",
                         "init", "doctor", "shell", "alias"}
        for alias, command in DEFAULT_ALIASES.items():
            first_word = command.split()[0]
            self.assertIn(first_word, valid_commands,
                         f"别名 {alias} 指向无效命令: {command}")


class TestU16AliasManagerBasics(unittest.TestCase):
    """U16：别名管理器基础"""

    def test_manager_importable(self):
        from core.aliases import AliasManager
        self.assertTrue(callable(AliasManager))

    def test_manager_instantiable(self):
        from core.aliases import AliasManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AliasManager(aliases_file=os.path.join(tmpdir, "aliases.json"))
            self.assertIsNotNone(mgr)

    def test_manager_has_default_aliases(self):
        from core.aliases import AliasManager, DEFAULT_ALIASES
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = AliasManager(aliases_file=os.path.join(tmpdir, "aliases.json"))
            for name in DEFAULT_ALIASES:
                self.assertIn(name, mgr.aliases)


class TestU16Resolve(unittest.TestCase):
    """U16：别名解析"""

    def setUp(self):
        from core.aliases import AliasManager
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = AliasManager(aliases_file=os.path.join(self.tmpdir, "aliases.json"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolve_simple_alias(self):
        result = self.mgr.resolve(["s"])
        self.assertEqual(result, ["status"])

    def test_resolve_alias_with_args(self):
        result = self.mgr.resolve(["h", "--detail"])
        self.assertEqual(result, ["health", "--detail"])

    def test_resolve_non_alias_unchanged(self):
        result = self.mgr.resolve(["status", "--json"])
        self.assertEqual(result, ["status", "--json"])

    def test_resolve_empty_list(self):
        result = self.mgr.resolve([])
        self.assertEqual(result, [])

    def test_resolve_alias_expanding_to_multiple(self):
        result = self.mgr.resolve(["cv"])
        self.assertEqual(result, ["config", "validate"])

    def test_resolve_chained_alias(self):
        """链式展开：别名指向另一个别名"""
        self.mgr.aliases["myalias"] = "s"  # 指向别名s
        result = self.mgr.resolve(["myalias"])
        self.assertEqual(result, ["status"])

    def test_resolve_cycle_detection(self):
        """防环检测：别名循环不应无限递归"""
        self.mgr.aliases["a"] = "b"
        self.mgr.aliases["b"] = "a"
        result = self.mgr.resolve(["a"])
        # 应在某处停止，不无限循环
        self.assertIsInstance(result, list)


class TestU16AddRemove(unittest.TestCase):
    """U16：添加与删除别名"""

    def setUp(self):
        from core.aliases import AliasManager
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = AliasManager(aliases_file=os.path.join(self.tmpdir, "aliases.json"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_custom_alias(self):
        self.assertTrue(self.mgr.add("myrun", "run --forever"))
        self.assertEqual(self.mgr.get("myrun"), "run --forever")

    def test_add_alias_resolvable(self):
        self.mgr.add("myrun", "run --forever")
        result = self.mgr.resolve(["myrun", "--duration", "60"])
        self.assertEqual(result, ["run", "--forever", "--duration", "60"])

    def test_cannot_override_default_alias(self):
        """不允许覆盖预定义别名"""
        self.assertFalse(self.mgr.add("s", "something_else"))
        # 原别名不变
        self.assertEqual(self.mgr.get("s"), "status")

    def test_remove_custom_alias(self):
        self.mgr.add("myrun", "run --forever")
        self.assertTrue(self.mgr.remove("myrun"))
        self.assertIsNone(self.mgr.get("myrun"))

    def test_cannot_remove_default_alias(self):
        """不允许删除预定义别名"""
        self.assertFalse(self.mgr.remove("s"))
        self.assertEqual(self.mgr.get("s"), "status")

    def test_remove_nonexistent_returns_false(self):
        self.assertFalse(self.mgr.remove("nonexistent"))


class TestU16Persistence(unittest.TestCase):
    """U16：别名持久化"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_custom_alias_persisted(self):
        from core.aliases import AliasManager
        aliases_file = os.path.join(self.tmpdir, "aliases.json")
        mgr1 = AliasManager(aliases_file=aliases_file)
        mgr1.add("myalias", "status --json")
        # 新manager加载
        mgr2 = AliasManager(aliases_file=aliases_file)
        self.assertEqual(mgr2.get("myalias"), "status --json")

    def test_default_aliases_not_persisted(self):
        """预定义别名不应写入用户文件"""
        from core.aliases import AliasManager
        aliases_file = os.path.join(self.tmpdir, "aliases.json")
        mgr = AliasManager(aliases_file=aliases_file)
        mgr.add("custom", "status")
        with open(aliases_file, "r") as f:
            data = json.load(f)
        self.assertIn("custom", data)
        self.assertNotIn("s", data)  # 预定义别名不写入


class TestU16ListAndFormat(unittest.TestCase):
    """U16：列表与格式化"""

    def setUp(self):
        from core.aliases import AliasManager
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = AliasManager(aliases_file=os.path.join(self.tmpdir, "aliases.json"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_aliases_returns_tuples(self):
        result = self.mgr.list_aliases()
        self.assertIsInstance(result, list)
        for item in result:
            self.assertEqual(len(item), 3)  # (name, command, is_default)

    def test_list_includes_defaults(self):
        result = self.mgr.list_aliases()
        names = [item[0] for item in result]
        self.assertIn("s", names)
        self.assertIn("h", names)

    def test_format_aliases_list(self):
        text = self.mgr.format_aliases_list()
        self.assertIn("命令别名", text)
        self.assertIn("s", text)
        self.assertIn("status", text)

    def test_is_alias(self):
        self.assertTrue(self.mgr.is_alias("s"))
        self.assertFalse(self.mgr.is_alias("nonexistent"))


class TestU16CliIntegration(unittest.TestCase):
    """U16：CLI alias子命令集成"""

    def test_cli_alias_list(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alias", "list"])
        self.assertEqual(args.alias_action, "list")

    def test_cli_alias_add(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alias", "add", "myrun", "run --forever"])
        self.assertEqual(args.alias_action, "add")
        self.assertEqual(args.name, "myrun")
        self.assertEqual(args.command, "run --forever")

    def test_cli_alias_remove(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alias", "remove", "myrun"])
        self.assertEqual(args.alias_action, "remove")
        self.assertEqual(args.name, "myrun")

    def test_cli_alias_list_runs(self):
        """端到端：CLI alias list 正常运行"""
        from cli import cli_main
        exit_code = cli_main(["alias", "list"])
        self.assertEqual(exit_code, 0)

    def test_cli_alias_resolved_before_parse(self):
        """端到端：别名在参数解析前展开"""
        from cli import cli_main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["s"])  # s -> status
        output = buf.getvalue()
        # status命令会输出状态信息
        self.assertTrue(len(output) > 0)


class TestU16NoRegression(unittest.TestCase):
    """U16：无回归保护"""

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

    def test_existing_cli_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "health", "config", "init", "doctor", "shell", "alias"]:
            args = parser.parse_args([cmd] if cmd not in ("config", "alias") else [cmd, "validate" if cmd == "config" else "list"])
            self.assertEqual(args.command, cmd)


if __name__ == "__main__":
    unittest.main()
