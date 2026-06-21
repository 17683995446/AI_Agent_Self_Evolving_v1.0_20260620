"""
U1阶段验证测试 —— CLI框架重构

验证点：
1. CLI模块存在且有完整子命令
2. --help可用
3. run子命令支持--duration/--forever/--scenario等参数
4. status/health/snapshot/config子命令存在
5. 默认行为合理（无参数=显示帮助）
6. 不破坏现有功能
"""

import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestU1CliModule:
    """验证CLI模块存在"""

    def test_cli_module_exists(self):
        """cli.py应存在"""
        cli_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cli.py"
        )
        assert os.path.exists(cli_path), "cli.py 应存在"

    def test_cli_importable(self):
        """cli模块应可导入"""
        from cli import cli_main, _create_parser
        assert cli_main is not None
        assert _create_parser is not None

    def test_parser_has_subcommands(self):
        """解析器应有run/status/health/snapshot/config子命令"""
        from cli import _create_parser
        parser = _create_parser()
        # 解析--help不应抛异常
        try:
            parser.parse_args(["--help"])
        except SystemExit:
            pass  # --help会触发SystemExit(0)


class TestU1HelpOutput:
    """验证--help输出"""

    def test_help_shows_subcommands(self):
        """--help应显示所有子命令"""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # --help会以exit code 0退出
        output = result.stdout + result.stderr
        assert "run" in output
        assert "status" in output
        assert "health" in output
        assert "snapshot" in output
        assert "config" in output

    def test_run_help_shows_options(self):
        """run --help应显示选项"""
        result = subprocess.run(
            [sys.executable, "cli.py", "run", "--help"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output = result.stdout + result.stderr
        assert "--duration" in output
        assert "--forever" in output
        assert "--scenario" in output
        assert "--enable-multi-agent" in output
        assert "--enable-forever" in output


class TestU1NoDefault30s:
    """验证默认行为不再是30秒测试模式"""

    def test_main_no_hardcoded_30s(self):
        """main.py不应再硬编码30秒默认运行"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # main()函数不应包含30秒测试模式
        assert "run_for_duration(30.0)" not in content, \
            "main.py不应再硬编码 run_for_duration(30.0)"

    def test_main_delegates_to_cli(self):
        """main()应委托给cli_main()"""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "cli_main" in content, "main.py应调用cli_main()"


class TestU1ConfigCommand:
    """验证config子命令"""

    def test_config_validate(self):
        """config validate应工作"""
        from cli import cli_main
        exit_code = cli_main(["config", "validate"])
        assert exit_code == 0

    def test_config_show(self):
        """config show应工作"""
        from cli import cli_main
        exit_code = cli_main(["config", "show"])
        assert exit_code == 0


class TestU1SnapshotCommand:
    """验证snapshot子命令"""

    def test_snapshot_save(self):
        """snapshot save应工作"""
        from cli import cli_main
        exit_code = cli_main(["snapshot", "save"])
        assert exit_code == 0

    def test_snapshot_list(self):
        """snapshot list应工作"""
        from cli import cli_main
        exit_code = cli_main(["snapshot", "list"])
        assert exit_code == 0


class TestU1NoRegression:
    """验证U1不破坏现有功能"""

    def test_agent_init_still_works(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        assert agent is not None
        agent.stop()

    def test_main_loop_still_runs(self):
        from main import SelfEvolvingAgent
        import time
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.8)
        assert agent.loop_count > 0
        agent.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
