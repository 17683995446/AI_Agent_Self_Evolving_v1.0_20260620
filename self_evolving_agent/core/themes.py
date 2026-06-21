"""
U18：输出着色与主题 —— 信息层级视觉区分

第一性原理：
- 用户在终端查看输出时，应能通过颜色快速区分信息层级
- 成功/警告/错误/信息/调试应有不同视觉标识
- 减少认知负担，提升可读性
- 非TTY环境（管道/重定向）应自动降级为无色

设计：
1. Color：ANSI颜色码常量
2. Theme：数据类，定义各信息级别的颜色方案
3. THEMES：预定义主题（default/dark/light/plain）
4. ThemeManager：管理主题，自动检测终端能力
5. colorize()：根据级别给文本着色
6. format_level()：格式化带级别标签的输出
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict


# ANSI颜色码
class Color:
    """ANSI颜色码常量"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 亮色前景色
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# 信息级别
LEVELS = ("debug", "info", "success", "warning", "error", "critical")


@dataclass
class Theme:
    """主题定义

    每个级别对应一个颜色码组合
    """
    name: str
    description: str
    colors: Dict[str, str]  # level -> color code
    label_colors: Dict[str, str] = field(default_factory=dict)  # 级别标签颜色

    def get_color(self, level: str) -> str:
        """获取指定级别的颜色码"""
        return self.colors.get(level, "")

    def get_label_color(self, level: str) -> str:
        """获取指定级别的标签颜色"""
        return self.label_colors.get(level, self.get_color(level))


# 预定义主题
THEMES: Dict[str, Theme] = {
    "default": Theme(
        name="default",
        description="标准ANSI颜色（适合大多数终端）",
        colors={
            "debug": Color.DIM + Color.CYAN,
            "info": Color.BLUE,
            "success": Color.GREEN,
            "warning": Color.YELLOW,
            "error": Color.RED,
            "critical": Color.BOLD + Color.BG_RED + Color.WHITE,
        },
        label_colors={
            "debug": Color.DIM + Color.CYAN,
            "info": Color.CYAN,
            "success": Color.GREEN,
            "warning": Color.YELLOW,
            "error": Color.RED,
            "critical": Color.BOLD + Color.BG_RED + Color.WHITE,
        },
    ),
    "dark": Theme(
        name="dark",
        description="暗色终端优化（使用亮色前景）",
        colors={
            "debug": Color.DIM + Color.BRIGHT_BLACK,
            "info": Color.BRIGHT_BLUE,
            "success": Color.BRIGHT_GREEN,
            "warning": Color.BRIGHT_YELLOW,
            "error": Color.BRIGHT_RED,
            "critical": Color.BOLD + Color.BG_RED + Color.BRIGHT_WHITE,
        },
        label_colors={
            "debug": Color.DIM + Color.BRIGHT_BLACK,
            "info": Color.BRIGHT_CYAN,
            "success": Color.BRIGHT_GREEN,
            "warning": Color.BRIGHT_YELLOW,
            "error": Color.BRIGHT_RED,
            "critical": Color.BOLD + Color.BG_RED + Color.BRIGHT_WHITE,
        },
    ),
    "light": Theme(
        name="light",
        description="亮色终端优化（使用深色前景）",
        colors={
            "debug": Color.DIM + Color.BLACK,
            "info": Color.BLUE,
            "success": Color.GREEN,
            "warning": Color.MAGENTA,
            "error": Color.RED,
            "critical": Color.BOLD + Color.BG_RED + Color.BLACK,
        },
        label_colors={
            "debug": Color.DIM + Color.BLACK,
            "info": Color.BLUE,
            "success": Color.GREEN,
            "warning": Color.MAGENTA,
            "error": Color.RED,
            "critical": Color.BOLD + Color.BG_RED + Color.BLACK,
        },
    ),
    "plain": Theme(
        name="plain",
        description="无颜色（适合管道/重定向/日志文件）",
        colors={level: "" for level in LEVELS},
        label_colors={level: "" for level in LEVELS},
    ),
}


def _detect_color_support(stream=None) -> bool:
    """检测终端是否支持颜色

    Args:
        stream: 输出流，默认为sys.stdout

    Returns:
        是否支持颜色
    """
    # NO_COLOR环境变量（https://no-color.org/）
    if os.environ.get("NO_COLOR") is not None:
        return False

    # 强制启用颜色
    if os.environ.get("FORCE_COLOR"):
        return True

    # 检查stream是否TTY
    if stream is None:
        stream = sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False

    # 检查TERM
    term = os.environ.get("TERM", "")
    if term in ("dumb", ""):
        return False

    return True


def _detect_theme() -> str:
    """根据终端背景色检测合适的主题

    Returns:
        主题名称
    """
    # COLORFGBG环境变量（如 "0;15" 表示前景0背景15）
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if len(parts) >= 2:
            try:
                bg = int(parts[-1])
                # 背景色 < 7 通常为暗色
                if bg <= 7:
                    return "dark"
                else:
                    return "light"
            except ValueError:
                pass

    # 默认使用default
    return "default"


class ThemeManager:
    """主题管理器

    负责主题选择、颜色检测、文本着色
    """

    def __init__(self, theme_name: Optional[str] = None,
                 color_enabled: Optional[bool] = None,
                 stream=None):
        """
        Args:
            theme_name: 主题名称，None则自动检测
            color_enabled: 是否启用颜色，None则自动检测
            stream: 输出流（用于检测TTY）
        """
        self._stream = stream
        self._color_enabled = color_enabled if color_enabled is not None else _detect_color_support(stream)
        self._theme_name = theme_name or (_detect_theme() if self._color_enabled else "plain")
        self._theme = THEMES.get(self._theme_name, THEMES["default"])
        if not self._color_enabled:
            self._theme = THEMES["plain"]

    @property
    def theme_name(self) -> str:
        return self._theme_name

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def color_enabled(self) -> bool:
        return self._color_enabled

    def set_theme(self, name: str) -> bool:
        """设置主题"""
        if name not in THEMES:
            return False
        self._theme_name = name
        if self._color_enabled:
            self._theme = THEMES[name]
        return True

    def enable_color(self, enabled: bool = True):
        """启用/禁用颜色"""
        self._color_enabled = enabled
        if enabled:
            self._theme = THEMES.get(self._theme_name, THEMES["default"])
        else:
            self._theme = THEMES["plain"]

    def colorize(self, text: str, level: str) -> str:
        """根据级别给文本着色

        Args:
            text: 原始文本
            level: 信息级别（debug/info/success/warning/error/critical）

        Returns:
            着色后的文本（如不支持颜色则返回原文）
        """
        color = self._theme.get_color(level)
        if not color:
            return text
        return f"{color}{text}{Color.RESET}"

    def colorize_label(self, text: str, level: str) -> str:
        """根据级别给标签文本着色"""
        color = self._theme.get_label_color(level)
        if not color:
            return text
        return f"{color}{text}{Color.RESET}"

    def format_level(self, text: str, level: str, label: Optional[str] = None) -> str:
        """格式化带级别标签的输出

        Args:
            text: 内容文本
            level: 信息级别
            label: 自定义标签文本，None则使用级别名大写

        Returns:
            格式化后的字符串，如 "[ERROR] 配置文件不存在"
        """
        if label is None:
            label = level.upper()
        colored_label = self.colorize_label(f"[{label}]", level)
        colored_text = self.colorize(text, level)
        return f"{colored_label} {colored_text}"

    def strip_colors(self, text: str) -> str:
        """去除文本中的所有ANSI颜色码"""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)

    def list_themes(self) -> Dict[str, str]:
        """列出所有可用主题"""
        return {name: theme.description for name, theme in THEMES.items()}

    def format_themes_list(self) -> str:
        """格式化主题列表"""
        lines = ["=== 可用主题 ==="]
        for name, theme in THEMES.items():
            marker = " *" if name == self._theme_name else ""
            lines.append(f"  {name:10s} {theme.description}{marker}")
        lines.append("")
        lines.append(f"当前主题: {self._theme_name}")
        lines.append(f"颜色启用: {'是' if self._color_enabled else '否'}")
        return "\n".join(lines)

    def preview(self) -> str:
        """生成主题预览（展示各级别颜色）"""
        lines = [f"=== 主题预览: {self._theme_name} ==="]
        for level in LEVELS:
            sample = f"这是{level}级别的示例消息"
            lines.append(self.format_level(sample, level))
        return "\n".join(lines)


# 全局ThemeManager实例（延迟初始化）
_global_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """获取全局ThemeManager实例"""
    global _global_theme_manager
    if _global_theme_manager is None:
        _global_theme_manager = ThemeManager()
    return _global_theme_manager


def reset_theme_manager():
    """重置全局ThemeManager（用于测试）"""
    global _global_theme_manager
    _global_theme_manager = None


# 便捷函数
def colorize(text: str, level: str) -> str:
    """便捷着色函数"""
    return get_theme_manager().colorize(text, level)


def format_level(text: str, level: str, label: Optional[str] = None) -> str:
    """便捷格式化函数"""
    return get_theme_manager().format_level(text, level, label)


def print_level(text: str, level: str, label: Optional[str] = None):
    """打印带级别颜色的消息"""
    print(format_level(text, level, label))
