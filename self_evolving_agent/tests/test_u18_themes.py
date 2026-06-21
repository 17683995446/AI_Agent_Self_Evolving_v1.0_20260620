"""
U18：输出着色与主题 —— 信息层级视觉区分

测试覆盖：
1. Color常量（ANSI颜色码）
2. Theme数据类（级别颜色定义）
3. THEMES预定义主题（default/dark/light/plain）
4. 终端检测（_detect_color_support/_detect_theme）
5. ThemeManager（主题选择、着色、格式化）
6. 便捷函数（colorize/format_level）
7. CLI集成（theme子命令）
8. 无回归
"""

import os
import sys
import io
import unittest
from unittest.mock import patch, MagicMock

# 路径处理
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestU18ColorConstants(unittest.TestCase):
    """测试Color常量"""

    def test_color_importable(self):
        from core.themes import Color
        self.assertTrue(hasattr(Color, "RESET"))

    def test_color_has_basic_colors(self):
        from core.themes import Color
        for color in ["RED", "GREEN", "YELLOW", "BLUE", "CYAN", "MAGENTA", "WHITE", "BLACK"]:
            self.assertTrue(hasattr(Color, color), f"缺少颜色: {color}")

    def test_color_has_bright_colors(self):
        from core.themes import Color
        for color in ["BRIGHT_RED", "BRIGHT_GREEN", "BRIGHT_YELLOW", "BRIGHT_BLUE"]:
            self.assertTrue(hasattr(Color, color), f"缺少亮色: {color}")

    def test_color_has_styles(self):
        from core.themes import Color
        self.assertTrue(hasattr(Color, "BOLD"))
        self.assertTrue(hasattr(Color, "DIM"))

    def test_color_has_background(self):
        from core.themes import Color
        self.assertTrue(hasattr(Color, "BG_RED"))
        self.assertTrue(hasattr(Color, "BG_GREEN"))

    def test_reset_is_ansi_escape(self):
        from core.themes import Color
        self.assertTrue(Color.RESET.startswith("\033["))
        self.assertIn("0m", Color.RESET)

    def test_red_is_ansi_escape(self):
        from core.themes import Color
        self.assertTrue(Color.RED.startswith("\033["))
        self.assertIn("31m", Color.RED)


class TestU18Theme(unittest.TestCase):
    """测试Theme数据类"""

    def test_theme_importable(self):
        from core.themes import Theme
        self.assertTrue(callable(Theme))

    def test_theme_instantiable(self):
        from core.themes import Theme
        theme = Theme(
            name="test",
            description="测试主题",
            colors={"info": "\033[34m"},
        )
        self.assertEqual(theme.name, "test")
        self.assertEqual(theme.description, "测试主题")

    def test_theme_get_color(self):
        from core.themes import Theme
        theme = Theme(
            name="test",
            description="测试",
            colors={"info": "\033[34m", "error": "\033[31m"},
        )
        self.assertEqual(theme.get_color("info"), "\033[34m")
        self.assertEqual(theme.get_color("error"), "\033[31m")

    def test_theme_get_color_unknown_level(self):
        from core.themes import Theme
        theme = Theme(name="test", description="", colors={})
        self.assertEqual(theme.get_color("unknown"), "")

    def test_theme_get_label_color_defaults_to_color(self):
        from core.themes import Theme
        theme = Theme(
            name="test",
            description="",
            colors={"info": "\033[34m"},
        )
        # 没有定义label_colors时应回退到colors
        self.assertEqual(theme.get_label_color("info"), "\033[34m")


class TestU18PredefinedThemes(unittest.TestCase):
    """测试预定义主题"""

    def test_themes_defined(self):
        from core.themes import THEMES
        self.assertIsInstance(THEMES, dict)
        self.assertGreater(len(THEMES), 0)

    def test_default_theme_exists(self):
        from core.themes import THEMES
        self.assertIn("default", THEMES)

    def test_dark_theme_exists(self):
        from core.themes import THEMES
        self.assertIn("dark", THEMES)

    def test_light_theme_exists(self):
        from core.themes import THEMES
        self.assertIn("light", THEMES)

    def test_plain_theme_exists(self):
        from core.themes import THEMES
        self.assertIn("plain", THEMES)

    def test_all_themes_have_all_levels(self):
        """所有主题都应定义所有级别"""
        from core.themes import THEMES, LEVELS
        for name, theme in THEMES.items():
            for level in LEVELS:
                self.assertIn(level, theme.colors,
                    f"主题{name}缺少级别{level}的颜色定义")

    def test_plain_theme_has_no_colors(self):
        """plain主题所有颜色应为空字符串"""
        from core.themes import THEMES, LEVELS
        plain = THEMES["plain"]
        for level in LEVELS:
            self.assertEqual(plain.colors[level], "",
                f"plain主题的{level}颜色应为空")

    def test_default_theme_has_colors(self):
        """default主题应有非空颜色"""
        from core.themes import THEMES
        default = THEMES["default"]
        self.assertNotEqual(default.colors["error"], "")
        self.assertNotEqual(default.colors["success"], "")

    def test_themes_have_descriptions(self):
        from core.themes import THEMES
        for name, theme in THEMES.items():
            self.assertIsInstance(theme.description, str)
            self.assertGreater(len(theme.description), 0)


class TestU18Detection(unittest.TestCase):
    """测试终端检测"""

    def test_detect_color_support_no_color_env(self):
        from core.themes import _detect_color_support
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertFalse(_detect_color_support())

    def test_detect_color_support_force_color(self):
        from core.themes import _detect_color_support
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            # 即使非TTY，FORCE_COLOR也应启用
            self.assertTrue(_detect_color_support())

    def test_detect_color_support_non_tty(self):
        """非TTY应返回False"""
        from core.themes import _detect_color_support
        # StringIO不是TTY
        stream = io.StringIO()
        self.assertFalse(_detect_color_support(stream))

    def test_detect_color_support_dumb_term(self):
        from core.themes import _detect_color_support
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        with patch.dict(os.environ, {"TERM": "dumb"}):
            self.assertFalse(_detect_color_support(mock_stream))

    def test_detect_color_support_empty_term(self):
        from core.themes import _detect_color_support
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        with patch.dict(os.environ, {"TERM": ""}):
            self.assertFalse(_detect_color_support(mock_stream))

    def test_detect_color_support_tty_with_term(self):
        from core.themes import _detect_color_support
        mock_stream = MagicMock()
        mock_stream.isatty.return_value = True
        with patch.dict(os.environ, {"TERM": "xterm-256color", "NO_COLOR": ""}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("FORCE_COLOR", None)
            self.assertTrue(_detect_color_support(mock_stream))

    def test_detect_theme_default(self):
        from core.themes import _detect_theme
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_detect_theme(), "default")

    def test_detect_theme_dark_from_colorfgbg(self):
        from core.themes import _detect_theme
        with patch.dict(os.environ, {"COLORFGBG": "0;0"}):
            self.assertEqual(_detect_theme(), "dark")

    def test_detect_theme_light_from_colorfgbg(self):
        from core.themes import _detect_theme
        with patch.dict(os.environ, {"COLORFGBG": "0;15"}):
            self.assertEqual(_detect_theme(), "light")


class TestU18ThemeManager(unittest.TestCase):
    """测试ThemeManager"""

    def test_manager_importable(self):
        from core.themes import ThemeManager
        self.assertTrue(callable(ThemeManager))

    def test_manager_instantiable(self):
        from core.themes import ThemeManager
        mgr = ThemeManager()
        self.assertIsNotNone(mgr)

    def test_manager_default_theme(self):
        from core.themes import ThemeManager, THEMES
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        self.assertEqual(mgr.theme_name, "default")
        self.assertIs(mgr.theme, THEMES["default"])

    def test_manager_plain_when_color_disabled(self):
        from core.themes import ThemeManager, THEMES
        mgr = ThemeManager(color_enabled=False)
        self.assertIs(mgr.theme, THEMES["plain"])

    def test_manager_colorize_with_color(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        result = mgr.colorize("hello", "error")
        self.assertIn("hello", result)
        self.assertIn("\033[", result)  # 包含ANSI码

    def test_manager_colorize_without_color(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(color_enabled=False)
        result = mgr.colorize("hello", "error")
        self.assertEqual(result, "hello")  # 无颜色码

    def test_manager_colorize_unknown_level(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        result = mgr.colorize("hello", "unknown")
        self.assertEqual(result, "hello")  # 未知级别不着色

    def test_manager_format_level(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        result = mgr.format_level("出错了", "error")
        self.assertIn("[ERROR]", result)
        self.assertIn("出错了", result)

    def test_manager_format_level_custom_label(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        result = mgr.format_level("消息", "info", label="INFO")
        self.assertIn("[INFO]", result)

    def test_manager_format_level_plain(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(color_enabled=False)
        result = mgr.format_level("消息", "info")
        self.assertEqual(result, "[INFO] 消息")

    def test_manager_set_theme(self):
        from core.themes import ThemeManager, THEMES
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        success = mgr.set_theme("dark")
        self.assertTrue(success)
        self.assertEqual(mgr.theme_name, "dark")
        self.assertIs(mgr.theme, THEMES["dark"])

    def test_manager_set_invalid_theme(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        success = mgr.set_theme("nonexistent")
        self.assertFalse(success)

    def test_manager_enable_color(self):
        from core.themes import ThemeManager, THEMES
        mgr = ThemeManager(color_enabled=False)
        mgr.enable_color(True)
        self.assertTrue(mgr.color_enabled)

    def test_manager_disable_color(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        mgr.enable_color(False)
        self.assertFalse(mgr.color_enabled)

    def test_manager_strip_colors(self):
        from core.themes import ThemeManager
        mgr = ThemeManager()
        colored = "\033[31mhello\033[0m world \033[32mfoo\033[0m"
        stripped = mgr.strip_colors(colored)
        self.assertEqual(stripped, "hello world foo")

    def test_manager_list_themes(self):
        from core.themes import ThemeManager
        mgr = ThemeManager()
        themes = mgr.list_themes()
        self.assertIn("default", themes)
        self.assertIn("dark", themes)
        self.assertIn("light", themes)
        self.assertIn("plain", themes)

    def test_manager_format_themes_list(self):
        from core.themes import ThemeManager
        mgr = ThemeManager()
        output = mgr.format_themes_list()
        self.assertIsInstance(output, str)
        self.assertIn("default", output)
        self.assertIn("dark", output)

    def test_manager_preview(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(theme_name="default", color_enabled=True)
        preview = mgr.preview()
        self.assertIsInstance(preview, str)
        self.assertIn("debug", preview)
        self.assertIn("error", preview)

    def test_manager_preview_plain(self):
        from core.themes import ThemeManager
        mgr = ThemeManager(color_enabled=False)
        preview = mgr.preview()
        self.assertIn("[DEBUG]", preview)
        self.assertIn("[ERROR]", preview)


class TestU18ConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def setUp(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def tearDown(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def test_get_theme_manager(self):
        from core.themes import get_theme_manager, ThemeManager
        mgr = get_theme_manager()
        self.assertIsInstance(mgr, ThemeManager)

    def test_get_theme_manager_singleton(self):
        from core.themes import get_theme_manager
        mgr1 = get_theme_manager()
        mgr2 = get_theme_manager()
        self.assertIs(mgr1, mgr2)

    def test_colorize_function(self):
        from core.themes import colorize, reset_theme_manager
        reset_theme_manager()
        result = colorize("test", "info")
        self.assertIsInstance(result, str)
        self.assertIn("test", result)

    def test_format_level_function(self):
        from core.themes import format_level, reset_theme_manager
        reset_theme_manager()
        result = format_level("test", "error")
        self.assertIn("[ERROR]", result)
        self.assertIn("test", result)


class TestU18CliIntegration(unittest.TestCase):
    """测试CLI集成"""

    def test_theme_subcommand_exists(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "list"])
        self.assertEqual(args.command, "theme")
        self.assertEqual(args.theme_action, "list")

    def test_theme_show_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "show"])
        self.assertEqual(args.theme_action, "show")

    def test_theme_preview_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "preview"])
        self.assertEqual(args.theme_action, "preview")

    def test_theme_set_parses(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["theme", "set", "dark"])
        self.assertEqual(args.theme_action, "set")
        self.assertEqual(args.name, "dark")

    def test_cli_theme_list_returns_0(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme", "list"])
        self.assertEqual(rc, 0)

    def test_cli_theme_show_returns_0(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme", "show"])
        self.assertEqual(rc, 0)

    def test_cli_theme_preview_returns_0(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme", "preview"])
        self.assertEqual(rc, 0)

    def test_cli_theme_set_valid_returns_0(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme", "set", "dark"])
        self.assertEqual(rc, 0)
        reset_theme_manager()

    def test_cli_theme_set_invalid_returns_1(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme", "set", "nonexistent"])
        self.assertEqual(rc, 1)
        reset_theme_manager()

    def test_cli_theme_no_action(self):
        from cli import cli_main
        from core.themes import reset_theme_manager
        reset_theme_manager()
        rc = cli_main(["theme"])
        self.assertEqual(rc, 1)


class TestU18NoRegression(unittest.TestCase):
    """验证无回归"""

    def setUp(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def tearDown(self):
        from core.themes import reset_theme_manager
        reset_theme_manager()

    def test_existing_commands_still_work(self):
        from cli import _create_parser
        parser = _create_parser()
        for cmd in ["run", "status", "health", "snapshot", "config",
                    "init", "doctor", "shell", "alias", "completion", "theme"]:
            args = parser.parse_args([cmd])
            self.assertEqual(args.command, cmd)

    def test_completion_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["completion", "bash"])
        self.assertEqual(args.command, "completion")
        self.assertEqual(args.shell, "bash")

    def test_alias_still_works(self):
        from cli import _create_parser
        parser = _create_parser()
        args = parser.parse_args(["alias", "list"])
        self.assertEqual(args.command, "alias")


if __name__ == "__main__":
    unittest.main()
