"""
U22：日志查看与过滤 —— 日志易用性提升

第一性原理：
- 用户排障时最常查看日志，但170+日志文件只能手动grep
- 应能通过CLI直接tail/search/list/clear日志
- 日志应支持按级别着色（复用U18 themes）
- 应支持按级别/时间/关键字过滤

设计：
1. LogEntry：解析单条日志为结构化数据
2. LogReader：读取日志文件，支持过滤/搜索/分页
3. LogManager：管理多个日志文件（list/clear/tail）
4. 日志着色：根据级别应用ANSI颜色
"""

import os
import re
import time
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterator, Any


# 日志级别到颜色的映射（复用U18 themes）
LEVEL_COLORS = {
    "DEBUG": "debug",
    "INFO": "info",
    "SUCCESS": "success",
    "WARNING": "warning",
    "WARN": "warning",
    "ERROR": "error",
    "CRITICAL": "critical",
    "FATAL": "critical",
}


@dataclass
class LogEntry:
    """单条日志条目"""
    raw: str  # 原始文本
    timestamp: Optional[str] = None  # 时间戳文本
    level: Optional[str] = None  # 日志级别
    event_type: Optional[str] = None  # 事件类型（如[CONFIG_LOAD]）
    message: str = ""  # 消息内容
    line_number: int = 0  # 行号

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "timestamp": self.timestamp,
            "level": self.level,
            "event_type": self.event_type,
            "message": self.message,
            "line_number": self.line_number,
        }


# 日志行解析正则
# 格式1: [2026-06-20 17:10:14] [ERROR   ] [Module] message
# 格式2: [EVENT_TYPE] message {...}
# 格式3: 纯文本
_LOG_PATTERN_1 = re.compile(
    r'^\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?\s*'
    r'\[?(DEBUG|INFO|SUCCESS|WARNING|WARN|ERROR|CRITICAL|FATAL)\s*\]?\s*'
    r'(?:\[([^\]]*)\])?\s*(.*)$'
)
_LOG_PATTERN_2 = re.compile(r'^\[([A-Z_][A-Z_]*)\]\s*(.*)$')


def parse_log_line(line: str, line_number: int = 0) -> LogEntry:
    """解析单行日志

    支持两种格式:
    1. [timestamp] [LEVEL] [module] message
    2. [EVENT_TYPE] message
    3. 纯文本
    """
    line = line.rstrip("\n\r")
    if not line.strip():
        return LogEntry(raw=line, line_number=line_number)

    # 尝试格式1: 带时间戳和级别
    m = _LOG_PATTERN_1.match(line)
    if m:
        return LogEntry(
            raw=line,
            timestamp=m.group(1),
            level=m.group(2).upper(),
            event_type=None,
            message=m.group(4).strip(),
            line_number=line_number,
        )

    # 尝试格式2: [EVENT_TYPE] message
    m = _LOG_PATTERN_2.match(line)
    if m:
        event_type = m.group(1)
        message = m.group(2).strip()
        # 从event_type推断级别
        level = _infer_level_from_event(event_type)
        return LogEntry(
            raw=line,
            timestamp=None,
            level=level,
            event_type=event_type,
            message=message,
            line_number=line_number,
        )

    # 格式3: 纯文本
    return LogEntry(raw=line, message=line, line_number=line_number)


def _infer_level_from_event(event_type: str) -> Optional[str]:
    """从事件类型推断日志级别"""
    event_upper = event_type.upper()
    if "ERROR" in event_upper or "FAIL" in event_upper or "EXCEPTION" in event_upper:
        return "ERROR"
    if "WARN" in event_upper:
        return "WARNING"
    if "SUCCESS" in event_upper:
        return "SUCCESS"
    if "DEBUG" in event_upper or "FUNC_" in event_upper:
        return "DEBUG"
    return "INFO"


class LogReader:
    """日志读取器

    支持读取单个日志文件，提供过滤/搜索/分页能力
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def exists(self) -> bool:
        return os.path.exists(self.file_path)

    def read_all(self) -> List[LogEntry]:
        """读取所有日志条目"""
        if not self.exists():
            return []
        entries = []
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                entries.append(parse_log_line(line, i))
        return entries

    def read_tail(self, n: int = 20) -> List[LogEntry]:
        """读取最后N条日志"""
        entries = self.read_all()
        return entries[-n:] if n > 0 else entries

    def read_head(self, n: int = 20) -> List[LogEntry]:
        """读取前N条日志"""
        entries = self.read_all()
        return entries[:n] if n > 0 else entries

    def filter_by_level(self, entries: List[LogEntry], level: str) -> List[LogEntry]:
        """按级别过滤"""
        level_upper = level.upper()
        return [e for e in entries if e.level and e.level.upper() == level_upper]

    def filter_by_levels(self, entries: List[LogEntry], levels: List[str]) -> List[LogEntry]:
        """按多个级别过滤"""
        levels_upper = {l.upper() for l in levels}
        return [e for e in entries if e.level and e.level.upper() in levels_upper]

    def search(self, keyword: str) -> List[LogEntry]:
        """搜索包含关键字的日志"""
        keyword_lower = keyword.lower()
        return [e for e in self.read_all() if keyword_lower in e.raw.lower()]

    def search_by_event(self, event_type: str) -> List[LogEntry]:
        """按事件类型搜索"""
        event_upper = event_type.upper()
        return [e for e in self.read_all() if e.event_type and e.event_type.upper() == event_upper]

    def count(self) -> int:
        """统计日志行数"""
        if not self.exists():
            return 0
        count = 0
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
        return count

    def count_by_level(self) -> Dict[str, int]:
        """按级别统计"""
        counts: Dict[str, int] = {}
        for entry in self.read_all():
            level = entry.level or "UNKNOWN"
            counts[level] = counts.get(level, 0) + 1
        return counts


class LogManager:
    """日志文件管理器

    管理logs目录下的多个日志文件
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir

    def _get_agent_logs(self) -> List[str]:
        """获取所有Agent日志文件（按时间排序）"""
        pattern = os.path.join(self.log_dir, "agent_*.log")
        files = glob.glob(pattern)
        return sorted(files)

    def _get_audit_logs(self) -> List[str]:
        """获取所有审计日志文件"""
        pattern = os.path.join(self.log_dir, "audit_*.log")
        files = glob.glob(pattern)
        return sorted(files)

    def get_latest_log(self) -> Optional[str]:
        """获取最新的日志文件"""
        files = self._get_agent_logs()
        return files[-1] if files else None

    def list_logs(self, log_type: str = "agent") -> List[Dict[str, Any]]:
        """列出日志文件

        Args:
            log_type: "agent" / "audit" / "all"

        Returns:
            日志文件信息列表
        """
        result = []
        if log_type in ("agent", "all"):
            for f in self._get_agent_logs():
                result.append(self._get_file_info(f, "agent"))
        if log_type in ("audit", "all"):
            for f in self._get_audit_logs():
                result.append(self._get_file_info(f, "audit"))
        return result

    def _get_file_info(self, path: str, log_type: str) -> Dict[str, Any]:
        """获取文件信息"""
        try:
            stat = os.stat(path)
            return {
                "path": path,
                "name": os.path.basename(path),
                "type": log_type,
                "size": stat.st_size,
                "size_human": _format_bytes(stat.st_size),
                "mtime": stat.st_mtime,
                "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "lines": LogReader(path).count(),
            }
        except Exception:
            return {
                "path": path,
                "name": os.path.basename(path),
                "type": log_type,
                "size": 0,
                "size_human": "0B",
                "mtime": 0,
                "mtime_str": "",
                "lines": 0,
            }

    def clear_logs(self, log_type: str = "agent", keep_latest: int = 1) -> int:
        """清理日志文件

        Args:
            log_type: "agent" / "audit" / "all"
            keep_latest: 保留最近N个文件

        Returns:
            删除的文件数
        """
        deleted = 0
        if log_type in ("agent", "all"):
            files = self._get_agent_logs()
            # 保留最近N个
            to_delete = files[:-keep_latest] if keep_latest > 0 else files
            for f in to_delete:
                try:
                    os.remove(f)
                    deleted += 1
                except Exception:
                    pass
        if log_type in ("audit", "all"):
            files = self._get_audit_logs()
            to_delete = files[:-keep_latest] if keep_latest > 0 else files
            for f in to_delete:
                try:
                    os.remove(f)
                    deleted += 1
                except Exception:
                    pass
        return deleted

    def format_list(self, log_type: str = "agent") -> str:
        """格式化日志文件列表"""
        logs = self.list_logs(log_type)
        if not logs:
            return f"（无{log_type}日志文件）"
        lines = [f"=== {log_type}日志文件（{len(logs)}个）==="]
        lines.append(f"{'名称':40s} {'大小':10s} {'行数':8s} {'修改时间':20s}")
        lines.append("-" * 80)
        for log in logs:
            lines.append(
                f"{log['name']:40s} {log['size_human']:10s} {log['lines']:8d} {log['mtime_str']:20s}"
            )
        return "\n".join(lines)


def _format_bytes(size: int) -> str:
    """格式化字节大小"""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def colorize_log_entry(entry: LogEntry, color_enabled: bool = True) -> str:
    """给日志条目着色

    Args:
        entry: 日志条目
        color_enabled: 是否启用颜色

    Returns:
        着色后的文本
    """
    if not color_enabled or not entry.level:
        return entry.raw

    try:
        from core.themes import get_theme_manager
        mgr = get_theme_manager()
        if not mgr.color_enabled:
            return entry.raw
        level_key = LEVEL_COLORS.get(entry.level.upper())
        if level_key is None:
            return entry.raw
        return mgr.colorize(entry.raw, level_key)
    except Exception:
        return entry.raw


def format_log_entries(entries: List[LogEntry], color_enabled: bool = True,
                       show_line_numbers: bool = False) -> str:
    """格式化日志条目列表"""
    lines = []
    for entry in entries:
        text = entry.raw
        if color_enabled:
            text = colorize_log_entry(entry, color_enabled)
        if show_line_numbers:
            text = f"{entry.line_number:6d} | {text}"
        lines.append(text)
    return "\n".join(lines)
