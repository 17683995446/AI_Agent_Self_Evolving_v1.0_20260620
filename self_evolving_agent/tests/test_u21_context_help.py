"""
U21：上下文感知帮助 —— 基于状态的帮助

测试覆盖：
1. HelpContext数据类
2. ContextDetector（状态检测）
3. ContextualHelpProvider（帮助生成）
   - 首次使用帮助
   - 通用帮助
   - 配置问题帮助
   - 迁移帮助
   - 错误帮助
   - 个性化建议
   - 故障排除
4. CLI集成（help子命令）
5. 无回归
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestU21HelpContext(unittest.TestCase):
    """测试HelpContext数据类"""

    def test_context_importable(self):
        from core.context_help import HelpContext
        self.assertTrue(callable(HelpContext))

    def test_context_instantiable(self):
        from core.context_help import HelpContext
        ctx = HelpContext()
        self.assertIsNotNone(ctx)

    def test_context_defaults(self):
        from core.context_help import HelpContext
        ctx = HelpContext()
        self.assertFalse(ctx.is_first_run)
        self.assertFalse(ctx.config_exists)
        self.assertFalse(ctx.config_valid)
        self.assertFalse(ctx.has_errors)

    def test_context_to_dict(self):
        from core.context_help import HelpContext
        ctx = HelpContext(is_first_run=True, config_exists=False)
        d = ctx.to_dict()
        self.assertTrue(d["is_first_run"])
        self.assertFalse(d["config_exists"])

    def test_context_with_errors(self):
        from core.context_help import HelpContext
        ctx = HelpContext(has_errors=True, error_type="config_invalid")
        self.assertTrue(ctx.has_errors)
        self.assertEqual(ctx.error_type, "config_invalid")


class TestU21ContextDetector(unittest.TestCase):
    """测试ContextDetector"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "agent.yaml")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detector_importable(self):
        from core.context_help import ContextDetector
        self.assertTrue(callable(ContextDetector))

    def test_detector_instantiable(self):
        from core.context_help import ContextDetector
        detector = ContextDetector(self.config_path)
        self.assertIsNotNone(detector)

    def test_detect_first_run(self):
        """配置文件不存在时应检测为首次使用"""
        from core.context_help import ContextDetector
        detector = ContextDetector(self.config_path)
        ctx = detector.detect()
        self.assertTrue(ctx.is_first_run)
        self.assertFalse(ctx.config_exists)

    def test_detect_config_exists(self):
        """配置文件存在时应检测到"""
        from core.context_help import ContextDetector
        # 创建空配置文件
        with open(self.config_path, "w") as f:
            f.write("# empty config")
        detector = ContextDetector(self.config_path)
        ctx = detector.detect()
        self.assertTrue(ctx.config_exists)
        self.assertFalse(ctx.is_first_run)

    def test_detect_returns_help_context(self):
        from core.context_help import ContextDetector, HelpContext
        detector = ContextDetector(self.config_path)
        ctx = detector.detect()
        self.assertIsInstance(ctx, HelpContext)


class TestU21ContextualHelpProvider(unittest.TestCase):
    """测试ContextualHelpProvider"""

    def test_provider_importable(self):
        from core.context_help import ContextualHelpProvider
        self.assertTrue(callable(ContextualHelpProvider))

    def test_provider_instantiable(self):
        from core.context_help import ContextualHelpProvider
        provider = ContextualHelpProvider()
        self.assertIsNotNone(provider)

    def test_get_help_first_run(self):
        """首次使用应返回引导帮助"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=True)
        help_text = provider.get_help(ctx)
        self.assertIn("首次使用", help_text)
        self.assertIn("init", help_text)

    def test_get_help_general(self):
        """非首次使用应返回通用帮助"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=False, config_exists=True, config_valid=True)
        help_text = provider.get_help(ctx)
        self.assertIn("命令参考", help_text)

    def test_get_help_with_config_problem(self):
        """配置无效时应包含配置问题帮助"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=False)
        help_text = provider.get_help(ctx)
        self.assertIn("配置问题", help_text)

    def test_get_help_with_migration_needed(self):
        """需要迁移时应包含迁移帮助"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=True, needs_migration=True)
        help_text = provider.get_help(ctx)
        self.assertIn("迁移", help_text)

    def test_get_help_with_errors(self):
        """有错误时应包含故障排除"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, has_errors=True, error_type="config_invalid")
        help_text = provider.get_help(ctx)
        self.assertIn("故障排除", help_text)

    def test_get_suggestions_first_run(self):
        """首次使用的建议应包含init"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=True)
        suggestions = provider.get_suggestions(ctx)
        self.assertTrue(any("init" in s for s in suggestions))

    def test_get_suggestions_config_problem(self):
        """配置问题的建议应包含doctor"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=False)
        suggestions = provider.get_suggestions(ctx)
        self.assertTrue(any("doctor" in s for s in suggestions))

    def test_get_suggestions_migration(self):
        """需要迁移的建议应包含migrate"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=True, needs_migration=True)
        suggestions = provider.get_suggestions(ctx)
        self.assertTrue(any("migrate" in s for s in suggestions))

    def test_get_suggestions_no_backups(self):
        """无备份的建议应包含backup"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=True, has_backups=False)
        suggestions = provider.get_suggestions(ctx)
        self.assertTrue(any("backup" in s for s in suggestions))

    def test_get_suggestions_completion(self):
        """建议应包含补全安装"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_exists=True, config_valid=True)
        suggestions = provider.get_suggestions(ctx)
        self.assertTrue(any("completion" in s for s in suggestions))

    def test_get_troubleshooting_no_errors(self):
        """无错误时故障排除应说明状态正常"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_valid=True, has_errors=False)
        result = provider.get_troubleshooting(ctx)
        self.assertIn("正常", result)

    def test_get_troubleshooting_with_errors(self):
        """有错误时应提供故障排除指南"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(has_errors=True, error_type="config_invalid")
        result = provider.get_troubleshooting(ctx)
        self.assertIn("故障排除", result)

    def test_get_quick_start_first_run(self):
        """首次使用的快速入门应包含init"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=True)
        result = provider.get_quick_start(ctx)
        self.assertIn("init", result)

    def test_get_quick_start_normal(self):
        """正常状态的快速入门应包含命令参考"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=False, config_exists=True)
        result = provider.get_quick_start(ctx)
        self.assertIn("命令参考", result)

    def test_format_context_report(self):
        """上下文报告应包含状态信息"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(is_first_run=True, config_exists=False)
        report = provider.format_context_report(ctx)
        self.assertIn("系统状态上下文", report)
        self.assertIn("首次使用", report)

    def test_format_context_report_with_version(self):
        """上下文报告应包含配置版本"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider()
        ctx = HelpContext(config_version=2)
        report = provider.format_context_report(ctx)
        self.assertIn("配置版本", report)
        self.assertIn("2", report)

    def test_help_contains_program_name(self):
        """帮助文本应包含程序名"""
        from core.context_help import ContextualHelpProvider, HelpContext
        provider = ContextualHelpProvider("my-agent")
        ctx = HelpContext(is_first_run=True)
        help_text = provider.get_help(ctx)
        self.assertIn("my-agent", help_text)


class TestU21CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_help_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["help"])
        self.assertEqual(args.command, "help")

    def test_help_context_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["help", "--context"])
        self.assertTrue(args.context)

    def test_help_troubleshoot_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["help", "--troubleshoot"])
        self.assertTrue(args.troubleshoot)

    def test_help_quick_start_flag(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["help", "--quick-start"])
        self.assertTrue(args.quick_start)

    def test_cli_help_returns_0(self):
        from cli import cli_main
        rc = cli_main(["help"])
        self.assertEqual(rc, 0)

    def test_cli_help_context_returns_0(self):
        from cli import cli_main
        rc = cli_main(["help", "--context"])
        self.assertEqual(rc, 0)

    def test_cli_help_troubleshoot_returns_0(self):
        from cli import cli_main
        rc = cli_main(["help", "--troubleshoot"])
        self.assertEqual(rc, 0)

    def test_cli_help_quick_start_returns_0(self):
        from cli import cli_main
        rc = cli_main(["help", "--quick-start"])
        self.assertEqual(rc, 0)


class TestU21NoRegression(unittest.TestCase):
    """验证无回归"""

    def test_all_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        commands = ["run", "status", "health", "snapshot", "config", "init",
                    "doctor", "shell", "alias", "completion", "theme",
                    "history", "bookmark", "help"]
        for cmd in commands:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_existing_subcommands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        # config子命令
        for action in ["validate", "show", "reload", "edit", "fields", "backup"]:
            args = parser.parse_args(["config", action])
            self.assertEqual(args.config_action, action)
        # history子命令
        for action in ["list", "clear"]:
            args = parser.parse_args(["history", action])
            self.assertEqual(args.history_action, action)
        # bookmark子命令
        for action in ["list"]:
            args = parser.parse_args(["bookmark", action])
            self.assertEqual(args.bookmark_action, action)

    def test_theme_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "list"])
        self.assertEqual(args.command, "theme")


if __name__ == "__main__":
    unittest.main()
