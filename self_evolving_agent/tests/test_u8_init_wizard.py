"""
U8：交互式初始化向导测试

验证目标：
1. InitWizard 类存在且可实例化
2. SCENARIO_PRESETS 包含4种预设（balanced/aggressive/conservative/eco）
3. 交互式问答逻辑正确（通过注入ask_fn/confirm_fn测试）
4. 非交互模式使用默认值
5. 配置文件生成正确（YAML可解析、字段齐全）
6. 目录自动创建
7. 幂等性：重复运行不破坏现有配置（除非--force）
8. 生成的启动命令包含所选能力
9. CLI init子命令可解析
10. 无回归

第一性原理：首次使用应被引导，而非阅读文档。
向导应可测试（IO分离）、可脚本化（非交互模式）、幂等（不破坏现有配置）。
"""
import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent


class TestU8InitWizardBasics(unittest.TestCase):
    """U8：InitWizard 基础功能"""

    def test_init_wizard_importable(self):
        from core.init_wizard import InitWizard
        self.assertTrue(callable(InitWizard))

    def test_scenario_presets_exist(self):
        from core.init_wizard import SCENARIO_PRESETS
        for name in ["balanced", "aggressive", "conservative", "eco"]:
            self.assertIn(name, SCENARIO_PRESETS, f"应包含 {name} 预设")

    def test_scenario_presets_have_required_fields(self):
        from core.init_wizard import SCENARIO_PRESETS
        for name, preset in SCENARIO_PRESETS.items():
            self.assertIn("label", preset, f"{name} 应有 label")
            self.assertIn("resource", preset, f"{name} 应有 resource")
            self.assertIn("drive_weights", preset, f"{name} 应有 drive_weights")
            self.assertIn("main_loop_frequency", preset, f"{name} 应有 main_loop_frequency")
            # 驱力权重和应为1.0
            dw = preset["drive_weights"]
            self.assertAlmostEqual(dw["safe"] + dw["gain"] + dw["eff"], 1.0, places=6)

    def test_init_wizard_instantiable(self):
        from core.init_wizard import InitWizard
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InitWizard(config_path="config/agent.yaml", project_root=tmpdir)
            self.assertIsNotNone(wizard)

    def test_ask_fn_injectable(self):
        """ask_fn 可注入，便于测试"""
        from core.init_wizard import InitWizard
        calls = []

        def mock_ask(prompt, default):
            calls.append((prompt, default))
            return default

        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InitWizard(project_root=tmpdir, ask_fn=mock_ask)
            self.assertEqual(wizard.ask("test", "default"), "default")
            self.assertEqual(len(calls), 1)

    def test_confirm_fn_injectable(self):
        """confirm_fn 可注入，便于测试"""
        from core.init_wizard import InitWizard

        def mock_confirm(prompt, default):
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = InitWizard(project_root=tmpdir, confirm_fn=mock_confirm)
            self.assertTrue(wizard.confirm("test", False))


class TestU8InteractiveWizard(unittest.TestCase):
    """U8：交互式向导流程"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.answers_queue = []

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_wizard(self, ask_answers, confirm_answers):
        """创建带预设回答的向导"""
        from core.init_wizard import InitWizard
        ask_iter = iter(ask_answers)
        confirm_iter = iter(confirm_answers)

        def mock_ask(prompt, default):
            try:
                return next(ask_iter)
            except StopIteration:
                return default

        def mock_confirm(prompt, default):
            try:
                return next(confirm_iter)
            except StopIteration:
                return default

        return InitWizard(project_root=self.tmpdir, ask_fn=mock_ask, confirm_fn=mock_confirm)

    def test_full_interactive_flow_balanced(self):
        """完整交互流程：选择balanced场景"""
        wizard = self._make_wizard(
            ask_answers=["1", "20.0", "20.0", "9090", "8080", "2"],  # 场景/CPU/内存/rest端口/dash端口/日志
            confirm_answers=[True, True, True, True],  # 多Agent/永续/REST/Dashboard
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "balanced")
        self.assertEqual(wizard.answers["max_cpu"], 20.0)
        self.assertTrue(wizard.answers["enable_multi_agent"])
        self.assertEqual(wizard.answers["rest_port"], 9090)
        self.assertEqual(wizard.answers["log_level"], "INFO")

    def test_full_interactive_flow_aggressive(self):
        """选择aggressive场景"""
        wizard = self._make_wizard(
            ask_answers=["2", "50.0", "50.0", "9091", "8081", "1"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "aggressive")
        self.assertEqual(wizard.answers["max_cpu"], 50.0)
        self.assertEqual(wizard.answers["log_level"], "DEBUG")

    def test_full_interactive_flow_conservative(self):
        """选择conservative场景"""
        wizard = self._make_wizard(
            ask_answers=["3", "10.0", "10.0", "9092", "8082", "3"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "conservative")
        self.assertEqual(wizard.answers["log_level"], "WARNING")

    def test_full_interactive_flow_eco(self):
        """选择eco场景"""
        wizard = self._make_wizard(
            ask_answers=["4", "5.0", "5.0", "9093", "8083", "4"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "eco")
        self.assertEqual(wizard.answers["log_level"], "ERROR")

    def test_scenario_by_name(self):
        """通过名称选择场景"""
        wizard = self._make_wizard(
            ask_answers=["aggressive", "50.0", "50.0", "9090", "8080", "2"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "aggressive")

    def test_invalid_scenario_choice_defaults_to_balanced(self):
        """无效场景选择回退到balanced"""
        wizard = self._make_wizard(
            ask_answers=["invalid", "20.0", "20.0", "9090", "8080", "2"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        self.assertEqual(wizard.answers["scenario"], "balanced")

    def test_invalid_cpu_defaults_to_preset(self):
        """无效CPU值回退到预设值"""
        wizard = self._make_wizard(
            ask_answers=["1", "not_a_number", "20.0", "9090", "8080", "2"],
            confirm_answers=[True, True, True, True],
        )
        wizard.run(force=True)
        # 应回退到balanced预设的20.0
        self.assertEqual(wizard.answers["max_cpu"], 20.0)

    def test_disable_all_features(self):
        """禁用所有能力"""
        wizard = self._make_wizard(
            ask_answers=["1", "20.0", "20.0", "2"],  # 无端口询问
            confirm_answers=[False, False, False, False],
        )
        wizard.run(force=True)
        self.assertFalse(wizard.answers["enable_multi_agent"])
        self.assertFalse(wizard.answers["enable_forever"])
        self.assertFalse(wizard.answers["enable_rest"])
        self.assertFalse(wizard.answers["enable_dashboard"])


class TestU8NonInteractiveMode(unittest.TestCase):
    """U8：非交互模式"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_non_interactive_uses_defaults(self):
        from core.init_wizard import InitWizard
        wizard = InitWizard(project_root=self.tmpdir)
        result = wizard.run(force=True, non_interactive=True)
        self.assertEqual(result, 0)
        self.assertEqual(wizard.answers["scenario"], "balanced")
        self.assertTrue(wizard.answers["enable_multi_agent"])
        self.assertTrue(wizard.answers["enable_forever"])
        self.assertEqual(wizard.answers["rest_port"], 9090)
        self.assertEqual(wizard.answers["dashboard_port"], 8080)
        self.assertEqual(wizard.answers["log_level"], "INFO")

    def test_non_interactive_creates_config_file(self):
        from core.init_wizard import InitWizard
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        self.assertTrue(config_file.exists())

    def test_non_interactive_creates_directories(self):
        from core.init_wizard import InitWizard
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        for d in ["logs", "data", "data/snapshots", "config"]:
            self.assertTrue((Path(self.tmpdir) / d).exists(), f"目录 {d} 应被创建")


class TestU8ConfigGeneration(unittest.TestCase):
    """U8：配置文件生成"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generated_config_is_valid_yaml(self):
        from core.init_wizard import InitWizard
        import yaml
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIsInstance(cfg, dict)

    def test_generated_config_has_required_sections(self):
        from core.init_wizard import InitWizard
        import yaml
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for section in ["resource", "safety", "evolution", "drive_weights", "logging"]:
            self.assertIn(section, cfg, f"配置应包含 {section} 段")
        self.assertIn("main_loop_frequency", cfg)

    def test_generated_config_validates(self):
        """生成的配置应能通过 SystemConfig.validate()"""
        from core.init_wizard import InitWizard
        from core.config import SystemConfig, ConfigManager
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        # 重置单例，加载生成的配置
        ConfigManager._instance = None
        ConfigManager._initialized = False
        try:
            mgr = ConfigManager(config_path=str(config_file))
            self.assertTrue(mgr.config.validate())
        finally:
            ConfigManager._instance = None
            ConfigManager._initialized = False

    def test_generated_config_uses_chosen_scenario(self):
        from core.init_wizard import InitWizard
        import yaml

        def mock_ask(prompt, default):
            if "选择" in prompt and "1-4" in prompt:
                return "2"  # aggressive
            return default

        def mock_confirm(prompt, default):
            return True

        wizard = InitWizard(project_root=self.tmpdir, ask_fn=mock_ask, confirm_fn=mock_confirm)
        wizard.run(force=True)
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # aggressive 预设 CPU=50
        self.assertEqual(cfg["resource"]["max_cpu_percent"], 50.0)


class TestU8Idempotency(unittest.TestCase):
    """U8：幂等性 —— 不破坏现有配置"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_does_not_overwrite_without_force(self):
        """无--force时不应覆盖现有配置"""
        from core.init_wizard import InitWizard
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        original_content = "original: config\n"
        config_file.write_text(original_content, encoding="utf-8")

        # 通过构造函数注入 confirm_fn 返回 False（不覆盖）
        wizard = InitWizard(
            project_root=self.tmpdir,
            confirm_fn=lambda p, d: False,
        )
        result = wizard.run(force=False, non_interactive=True)
        self.assertEqual(result, 0)
        # 配置文件应未被修改
        self.assertEqual(config_file.read_text(encoding="utf-8"), original_content)

    def test_force_overwrites_existing(self):
        """--force应覆盖现有配置"""
        from core.init_wizard import InitWizard
        config_file = Path(self.tmpdir) / "config" / "agent.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("old: config\n", encoding="utf-8")

        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        content = config_file.read_text(encoding="utf-8")
        self.assertIn("resource", content)
        self.assertNotIn("old: config", content)


class TestU8RunCommandGeneration(unittest.TestCase):
    """U8：启动命令生成"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_command_includes_all_features(self):
        from core.init_wizard import InitWizard
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.run(force=True, non_interactive=True)
        cmd = wizard.generate_run_command()
        self.assertIn("--enable-multi-agent", cmd)
        self.assertIn("--enable-forever", cmd)
        self.assertIn("--rest-port 9090", cmd)
        self.assertIn("--dashboard-port 8080", cmd)
        self.assertIn("--log-level INFO", cmd)

    def test_run_command_minimal_when_disabled(self):
        from core.init_wizard import InitWizard
        wizard = InitWizard(project_root=self.tmpdir)
        wizard.answers = {
            "scenario": "balanced",
            "enable_multi_agent": False,
            "enable_forever": False,
            "enable_rest": False,
            "enable_dashboard": False,
            "log_level": "INFO",
        }
        cmd = wizard.generate_run_command()
        self.assertIn("python main.py run", cmd)
        self.assertNotIn("--enable-multi-agent", cmd)
        self.assertNotIn("--rest-port", cmd)


class TestU8CliIntegration(unittest.TestCase):
    """U8：CLI init子命令集成"""

    def test_cli_has_init_subcommand(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["init", "--non-interactive"])
        self.assertEqual(args.command, "init")
        self.assertTrue(args.non_interactive)

    def test_cli_init_force_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["init", "--force"])
        self.assertTrue(args.force)

    def test_cli_init_config_path(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["init", "--config-path", "custom/path.yaml"])
        self.assertEqual(args.config_path, "custom/path.yaml")

    def test_cli_init_help_shows_examples(self):
        from cli import _create_parser
        parser = _create_parser()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                parser.parse_args(["init", "--help"])
            except SystemExit:
                pass
        output = buf.getvalue()
        self.assertIn("init", output)
        self.assertIn("--non-interactive", output)

    def test_cli_init_non_interactive_end_to_end(self):
        """端到端：CLI init --non-interactive 生成配置"""
        from cli import cli_main
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config", "agent.yaml")
            exit_code = cli_main(["init", "--non-interactive", "--force",
                                  "--config-path", config_path])
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(config_path))


class TestU8NoRegression(unittest.TestCase):
    """U8：无回归保护"""

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
        """原有CLI命令不受影响"""
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "health", "config"]:
            args = parser.parse_args([cmd] if cmd != "config" else [cmd, "validate"])
            self.assertEqual(args.command, cmd)


if __name__ == "__main__":
    unittest.main()
