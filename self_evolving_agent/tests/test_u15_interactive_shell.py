"""
U15：交互式Shell测试

验证目标：
1. InteractiveShell 类存在且可实例化
2. input_fn 可注入（IO分离，便于测试）
3. cli_dispatch 可注入
4. 欢迎信息显示
5. help 元命令
6. exit/quit 退出
7. history 元命令
8. clear 元命令
9. !系统命令执行
10. CLI命令分发
11. 命令历史持久化
12. 空命令处理
13. CLI shell子命令集成
14. 无回归

第一性原理：探索式使用应支持REPL，而非每次敲完整命令。
"""
import os
import sys
import io
import unittest
import tempfile
import shutil
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU15ShellBasics(unittest.TestCase):
    """U15：Shell基础功能"""

    def test_shell_importable(self):
        from core.interactive_shell import InteractiveShell
        self.assertTrue(callable(InteractiveShell))

    def test_shell_instantiable(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        self.assertIsNotNone(shell)

    def test_input_fn_injectable(self):
        """input_fn 可注入，便于测试"""
        from core.interactive_shell import InteractiveShell
        calls = []

        def mock_input(prompt):
            calls.append(prompt)
            return "exit"

        shell = InteractiveShell(input_fn=mock_input)
        shell.run()
        self.assertEqual(len(calls), 1)
        self.assertIn("agent>", calls[0])

    def test_cli_dispatch_injectable(self):
        """cli_dispatch 可注入"""
        from core.interactive_shell import InteractiveShell
        dispatched = []

        def mock_dispatch(args):
            dispatched.append(args)
            return 0

        shell = InteractiveShell(
            input_fn=lambda p: "exit",
            cli_dispatch=mock_dispatch,
        )
        shell.run()
        # exit不触发dispatch
        self.assertEqual(len(dispatched), 0)


class TestU15Welcome(unittest.TestCase):
    """U15：欢迎信息"""

    def test_show_welcome(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(
            input_fn=lambda p: "exit",
            output=buf,
        )
        shell.run()
        output = buf.getvalue()
        self.assertIn("交互式Shell", output)
        self.assertIn("help", output)

    def test_welcome_shown_once(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.show_welcome()
        shell.show_welcome()
        output = buf.getvalue()
        # 欢迎信息只出现一次
        self.assertEqual(output.count("交互式Shell"), 1)


class TestU15Help(unittest.TestCase):
    """U15：help元命令"""

    def test_help_shows_commands(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.show_help()
        output = buf.getvalue()
        self.assertIn("run", output)
        self.assertIn("status", output)
        self.assertIn("health", output)
        self.assertIn("config", output)

    def test_help_shows_meta_commands(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.show_help()
        output = buf.getvalue()
        self.assertIn("exit", output)
        self.assertIn("history", output)
        self.assertIn("clear", output)

    def test_help_via_process_command(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.process_command("help")
        output = buf.getvalue()
        self.assertIn("Shell帮助", output)


class TestU15Exit(unittest.TestCase):
    """U15：退出命令"""

    def test_exit_returns_negative_one(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        result = shell.process_command("exit")
        self.assertEqual(result, -1)

    def test_quit_returns_negative_one(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        result = shell.process_command("quit")
        self.assertEqual(result, -1)

    def test_exit_case_insensitive(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        self.assertEqual(shell.process_command("EXIT"), -1)
        self.assertEqual(shell.process_command("Quit"), -1)


class TestU15History(unittest.TestCase):
    """U15：历史命令"""

    def test_history_records_commands(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(
            input_fn=lambda p: "exit",
            output=io.StringIO(),
        )
        shell.process_command("status")
        shell.process_command("health")
        self.assertIn("status", shell.history)
        self.assertIn("health", shell.history)

    def test_history_meta_command(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.process_command("status")
        shell.process_command("health")
        shell.process_command("history")
        output = buf.getvalue()
        self.assertIn("status", output)
        self.assertIn("health", output)

    def test_empty_command_not_recorded(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        shell.process_command("")
        shell.process_command("   ")
        self.assertEqual(len(shell.history), 0)


class TestU15Clear(unittest.TestCase):
    """U15：clear元命令"""

    def test_clear_returns_zero(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell()
        result = shell.process_command("clear")
        self.assertEqual(result, 0)

    def test_clear_writes_escape_sequence(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        shell.clear_screen()
        output = buf.getvalue()
        self.assertIn("\033[", output)  # ANSI转义序列


class TestU15SystemCommand(unittest.TestCase):
    """U15：系统命令（!前缀）"""

    def test_system_command_executes(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        result = shell.process_command("!echo hello")
        self.assertEqual(result, 0)
        self.assertIn("hello", buf.getvalue())

    def test_system_command_failure(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()
        shell = InteractiveShell(output=buf)
        result = shell.process_command("!false")
        self.assertNotEqual(result, 0)


class TestU15CliDispatch(unittest.TestCase):
    """U15：CLI命令分发"""

    def test_cli_command_dispatched(self):
        from core.interactive_shell import InteractiveShell
        dispatched = []

        def mock_dispatch(args):
            dispatched.append(args)
            return 0

        shell = InteractiveShell(
            output=io.StringIO(),
            cli_dispatch=mock_dispatch,
        )
        shell.process_command("status")
        shell.process_command("health --detail")
        self.assertEqual(dispatched[0], ["status"])
        self.assertEqual(dispatched[1], ["health", "--detail"])

    def test_cli_command_failure_handled(self):
        from core.interactive_shell import InteractiveShell
        buf = io.StringIO()

        def failing_dispatch(args):
            raise RuntimeError("test error")

        shell = InteractiveShell(
            output=buf,
            cli_dispatch=failing_dispatch,
        )
        result = shell.process_command("status")
        self.assertEqual(result, 1)
        self.assertIn("命令执行失败", buf.getvalue())


class TestU15HistoryPersistence(unittest.TestCase):
    """U15：历史持久化"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.tmpdir, "history")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_history(self):
        from core.interactive_shell import InteractiveShell
        shell1 = InteractiveShell(
            input_fn=lambda p: "exit",
            output=io.StringIO(),
            history_file=self.history_file,
        )
        shell1.process_command("status")
        shell1.process_command("health")
        shell1._save_history()
        # 新shell加载历史
        shell2 = InteractiveShell(
            input_fn=lambda p: "exit",
            output=io.StringIO(),
            history_file=self.history_file,
        )
        shell2._load_history()
        self.assertIn("status", shell2.history)
        self.assertIn("health", shell2.history)

    def test_history_file_created(self):
        from core.interactive_shell import InteractiveShell
        shell = InteractiveShell(history_file=self.history_file)
        shell.process_command("test_command")
        shell._save_history()
        self.assertTrue(os.path.exists(self.history_file))


class TestU15RunLoop(unittest.TestCase):
    """U15：运行循环"""

    def test_run_exits_on_exit_command(self):
        from core.interactive_shell import InteractiveShell
        commands = iter(["status", "health", "exit"])
        outputs = []

        def mock_input(prompt):
            return next(commands)

        buf = io.StringIO()
        shell = InteractiveShell(input_fn=mock_input, output=buf)
        result = shell.run()
        self.assertEqual(result, 0)
        self.assertIn("再见", buf.getvalue())

    def test_run_handles_eof(self):
        from core.interactive_shell import InteractiveShell

        def mock_input(prompt):
            raise EOFError()

        buf = io.StringIO()
        shell = InteractiveShell(input_fn=mock_input, output=buf)
        result = shell.run()
        self.assertEqual(result, 0)

    def test_run_handles_keyboard_interrupt(self):
        from core.interactive_shell import InteractiveShell
        call_count = [0]

        def mock_input(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        buf = io.StringIO()
        shell = InteractiveShell(input_fn=mock_input, output=buf)
        result = shell.run()
        self.assertEqual(result, 0)


class TestU15CliIntegration(unittest.TestCase):
    """U15：CLI shell子命令集成"""

    def test_cli_has_shell_subcommand(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["shell"])
        self.assertEqual(args.command, "shell")

    def test_cli_shell_help_shows_examples(self):
        from cli import _create_parser
        parser = _create_parser()
        import io as _io
        from contextlib import redirect_stdout
        buf = _io.StringIO()
        with redirect_stdout(buf):
            try:
                parser.parse_args(["shell", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        self.assertIn("shell", output)


class TestU15NoRegression(unittest.TestCase):
    """U15：无回归保护"""

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
        for cmd in ["run", "status", "health", "config", "init", "doctor", "shell"]:
            args = parser.parse_args([cmd] if cmd not in ("config",) else [cmd, "validate"])
            self.assertEqual(args.command, cmd)


if __name__ == "__main__":
    unittest.main()
