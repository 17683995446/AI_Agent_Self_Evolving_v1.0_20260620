"""
U13：进度可视化

第一性原理：运行状态应可视化，而非翻日志。
- 运行时进度条：显示循环数、运行时长、资源使用
- 关键指标摘要：定期输出一行摘要
- 可配置刷新间隔与显示内容
- 支持TTY与非TTY环境（非TTY降级为定期摘要）

设计原则：
1. 进度条不干扰日志输出（使用回车\r覆盖）
2. 非TTY环境自动降级（避免乱码）
3. 刷新频率可配置（避免刷屏）
4. 显示内容可配置（精简/详细）
"""

import sys
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass


def is_tty() -> bool:
    """检测当前stdout是否为TTY"""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


@dataclass
class ProgressConfig:
    """进度显示配置"""
    refresh_interval: float = 1.0  # 刷新间隔（秒）
    show_cpu: bool = True
    show_memory: bool = True
    show_drives: bool = True
    show_loop_count: bool = True
    show_uptime: bool = True
    compact: bool = False  # 精简模式（单行）


class ProgressBar:
    """运行时进度条

    在TTY环境显示实时进度，非TTY环境降级为定期摘要。
    """

    def __init__(
        self,
        config: Optional[ProgressConfig] = None,
        output=None,
        tty_checker: Optional[Callable[[], bool]] = None,
    ):
        self.config = config or ProgressConfig()
        self.output = output or sys.stdout
        self._tty_checker = tty_checker or is_tty
        self._last_refresh: float = 0.0
        self._last_summary: float = 0.0
        self._line_length: int = 0

    def _is_tty(self) -> bool:
        return self._tty_checker()

    def format_progress_line(self, status: Dict[str, Any]) -> str:
        """格式化进度行

        Args:
            status: 状态字典（loop_count/uptime/cpu/memory/drives等）

        Returns:
            格式化的进度字符串
        """
        parts = []
        if self.config.show_loop_count:
            parts.append(f"循环:{status.get('loop_count', 0)}")
        if self.config.show_uptime:
            uptime = status.get("uptime_human", "0秒")
            parts.append(f"时长:{uptime}")
        if self.config.show_cpu:
            cpu = status.get("resources", {}).get("cpu_percent", 0)
            parts.append(f"CPU:{cpu:.0f}%")
        if self.config.show_memory:
            mem = status.get("resources", {}).get("memory_percent", 0)
            parts.append(f"内存:{mem:.0f}%")
        if self.config.show_drives and not self.config.compact:
            drives = status.get("drives", {})
            if drives:
                parts.append(f"驱力:S{drives.get('safe', 0):.1f}/G{drives.get('gain', 0):.1f}/E{drives.get('eff', 0):.1f}")
        running = status.get("running", False)
        icon = "●" if running else "○"
        line = f"{icon} " + " | ".join(parts)
        return line

    def format_summary_line(self, status: Dict[str, Any]) -> str:
        """格式化摘要行（用于非TTY或定期摘要）"""
        parts = []
        parts.append(f"[{time.strftime('%H:%M:%S')}]")
        parts.append(f"循环={status.get('loop_count', 0)}")
        uptime = status.get("uptime_human", "0秒")
        parts.append(f"时长={uptime}")
        resources = status.get("resources", {})
        if resources:
            parts.append(f"CPU={resources.get('cpu_percent', 0):.0f}%")
            parts.append(f"内存={resources.get('memory_percent', 0):.0f}%")
        running = status.get("running", False)
        parts.append("运行中" if running else "已停止")
        return " ".join(parts)

    def update(self, status: Dict[str, Any], force: bool = False) -> None:
        """更新进度显示

        Args:
            status: 状态字典
            force: 强制刷新（忽略间隔）
        """
        now = time.time()
        if not force and (now - self._last_refresh) < self.config.refresh_interval:
            return
        self._last_refresh = now

        if self._is_tty():
            line = self.format_progress_line(status)
            # 覆盖上一行
            padding = max(0, self._line_length - len(line))
            self.output.write("\r" + line + " " * padding)
            self.output.flush()
            self._line_length = len(line)
        else:
            # 非TTY：定期摘要（每5个刷新间隔输出一次）
            if force or (now - self._last_summary) >= self.config.refresh_interval * 5:
                line = self.format_summary_line(status)
                self.output.write(line + "\n")
                self.output.flush()
                self._last_summary = now

    def finish(self, status: Optional[Dict[str, Any]] = None) -> None:
        """结束进度显示，输出最终状态"""
        if self._is_tty() and self._line_length > 0:
            self.output.write("\n")
            self.output.flush()
            self._line_length = 0
        if status:
            line = self.format_summary_line(status)
            self.output.write("最终状态: " + line + "\n")
            self.output.flush()

    def clear(self) -> None:
        """清除当前进度行"""
        if self._is_tty() and self._line_length > 0:
            self.output.write("\r" + " " * self._line_length + "\r")
            self.output.flush()
            self._line_length = 0


class MetricsSummary:
    """关键指标摘要器

    定期输出一行关键指标摘要，适用于日志或非TTY环境。
    """

    def __init__(self, interval_seconds: float = 60.0):
        self.interval = interval_seconds
        self._last_output: float = 0.0
        self._history: list = []

    def maybe_output(self, status: Dict[str, Any], force: bool = False) -> Optional[str]:
        """如果到达间隔则输出摘要

        Returns:
            摘要字符串，或None（未到间隔）
        """
        now = time.time()
        if not force and (now - self._last_output) < self.interval:
            return None
        self._last_output = now
        summary = self.format(status)
        self._history.append({"time": now, "summary": summary})
        # 保留最近100条
        if len(self._history) > 100:
            self._history = self._history[-100:]
        return summary

    def format(self, status: Dict[str, Any]) -> str:
        """格式化指标摘要"""
        parts = [f"[{time.strftime('%H:%M:%S')}]"]
        parts.append(f"循环={status.get('loop_count', 0)}")
        resources = status.get("resources", {})
        if resources:
            parts.append(f"CPU={resources.get('cpu_percent', 0):.1f}%")
            parts.append(f"内存={resources.get('memory_percent', 0):.1f}%")
        drives = status.get("drives", {})
        if drives:
            parts.append(f"驱力={drives.get('safe', 0):.2f}/{drives.get('gain', 0):.2f}/{drives.get('eff', 0):.2f}")
        prediction = status.get("prediction", {})
        if prediction:
            parts.append(f"置信度={prediction.get('confidence', 0):.2f}")
        return " | ".join(parts)

    @property
    def history(self) -> list:
        return list(self._history)
