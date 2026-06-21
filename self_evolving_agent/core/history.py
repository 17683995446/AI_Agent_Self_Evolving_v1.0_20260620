"""
U20：历史命令与书签 —— 重复操作回放

第一性原理：
- 用户经常需要重复执行相同的命令序列
- 应能通过书签保存常用命令，快速回放历史命令
- 历史记录应持久化，跨会话可用
- 书签应有有意义的名称，便于记忆

设计：
1. CommandHistory：命令历史管理
   - 记录执行过的命令（带时间戳）
   - 持久化到JSON文件
   - 搜索（按关键字/前缀）
   - 获取最近N条
2. BookmarkManager：书签管理
   - 保存常用命令为命名书签
   - 预定义书签
   - 增删查改
   - 持久化
"""

import os
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any


@dataclass
class HistoryEntry:
    """历史命令条目"""
    command: str
    timestamp: float
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        return cls(
            command=data.get("command", ""),
            timestamp=data.get("timestamp", 0.0),
            exit_code=data.get("exit_code", 0),
        )

    def format_time(self) -> str:
        """格式化时间戳为可读字符串"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


# 预定义书签
DEFAULT_BOOKMARKS: Dict[str, str] = {
    "start": "run --forever",
    "start-full": "run --enable-multi-agent --enable-forever --rest-port 9090",
    "check": "doctor",
    "status": "status",
    "health": "health --detail",
    "save": "snapshot save",
    "validate": "config validate",
    "backup": "config backup",
    "edit": "config edit",
    "migrate": "config migrate",
}


@dataclass
class Bookmark:
    """书签条目"""
    name: str
    command: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bookmark":
        return cls(
            name=data.get("name", ""),
            command=data.get("command", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            is_default=data.get("is_default", False),
        )


def _get_data_dir() -> str:
    """获取数据目录"""
    home = os.path.expanduser("~")
    data_dir = os.path.join(home, ".self_evolving_agent")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


class CommandHistory:
    """命令历史管理器

    第一性原理：
    - 记录用户执行过的命令，便于回溯和重放
    - 历史应持久化，跨会话可用
    - 应支持搜索和过滤
    """

    def __init__(self, history_file: Optional[str] = None, max_entries: int = 1000):
        if history_file is None:
            history_file = os.path.join(_get_data_dir(), "history.json")
        self.history_file = history_file
        self.max_entries = max_entries
        self._entries: List[HistoryEntry] = []
        self._load()

    def _load(self):
        """从文件加载历史"""
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [HistoryEntry.from_dict(e) for e in data]
        except Exception:
            self._entries = []

    def _save(self):
        """保存历史到文件"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add(self, command: str, exit_code: int = 0):
        """添加一条历史记录"""
        if not command.strip():
            return
        entry = HistoryEntry(
            command=command.strip(),
            timestamp=time.time(),
            exit_code=exit_code,
        )
        self._entries.append(entry)
        # 限制最大数量
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        self._save()

    def list(self, limit: int = 20) -> List[HistoryEntry]:
        """获取最近的历史记录"""
        return self._entries[-limit:] if limit > 0 else list(self._entries)

    def search(self, keyword: str) -> List[HistoryEntry]:
        """按关键字搜索历史"""
        keyword_lower = keyword.lower()
        return [e for e in self._entries if keyword_lower in e.command.lower()]

    def search_by_prefix(self, prefix: str) -> List[HistoryEntry]:
        """按前缀搜索历史"""
        return [e for e in self._entries if e.command.startswith(prefix)]

    def get(self, index: int) -> Optional[HistoryEntry]:
        """按索引获取历史条目（支持负索引）"""
        try:
            return self._entries[index]
        except IndexError:
            return None

    def get_last(self) -> Optional[HistoryEntry]:
        """获取最后一条历史"""
        if not self._entries:
            return None
        return self._entries[-1]

    def clear(self):
        """清空历史"""
        self._entries = []
        self._save()

    def count(self) -> int:
        """返回历史记录数量"""
        return len(self._entries)

    def format_list(self, limit: int = 20) -> str:
        """格式化历史列表"""
        entries = self.list(limit)
        if not entries:
            return "（无历史记录）"
        lines = [f"=== 命令历史（最近{len(entries)}条，共{self.count()}条）==="]
        # 编号从1开始，倒序显示（最新在上）
        for i, entry in enumerate(reversed(entries), 1):
            time_str = entry.format_time()
            exit_marker = "" if entry.exit_code == 0 else f" [退出码:{entry.exit_code}]"
            lines.append(f"  {i:3d}. [{time_str}]{exit_marker} {entry.command}")
        return "\n".join(lines)


class BookmarkManager:
    """书签管理器

    第一性原理：
    - 常用命令应能保存为命名书签
    - 书签应有有意义的名称
    - 应支持预定义书签和用户自定义书签
    """

    def __init__(self, bookmarks_file: Optional[str] = None):
        if bookmarks_file is None:
            bookmarks_file = os.path.join(_get_data_dir(), "bookmarks.json")
        self.bookmarks_file = bookmarks_file
        self._bookmarks: Dict[str, Bookmark] = {}
        self._load()

    def _load(self):
        """从文件加载书签"""
        # 先加载预定义书签
        for name, command in DEFAULT_BOOKMARKS.items():
            self._bookmarks[name] = Bookmark(
                name=name,
                command=command,
                description=f"预定义书签",
                is_default=True,
            )
        # 再加载用户书签（覆盖同名预定义）
        if not os.path.exists(self.bookmarks_file):
            return
        try:
            with open(self.bookmarks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                bm = Bookmark.from_dict(item)
                if not bm.is_default:  # 不加载标记为default的
                    self._bookmarks[bm.name] = bm
        except Exception:
            pass

    def _save(self):
        """保存用户书签到文件（只保存非预定义的）"""
        user_bookmarks = [bm.to_dict() for bm in self._bookmarks.values() if not bm.is_default]
        try:
            with open(self.bookmarks_file, "w", encoding="utf-8") as f:
                json.dump(user_bookmarks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add(self, name: str, command: str, description: str = "") -> bool:
        """添加书签

        Returns:
            是否添加成功（预定义书签不可覆盖）
        """
        if name in DEFAULT_BOOKMARKS:
            return False
        self._bookmarks[name] = Bookmark(
            name=name,
            command=command,
            description=description,
            is_default=False,
        )
        self._save()
        return True

    def remove(self, name: str) -> bool:
        """删除书签

        Returns:
            是否删除成功（预定义书签不可删除）
        """
        if name in DEFAULT_BOOKMARKS:
            return False
        if name not in self._bookmarks:
            return False
        del self._bookmarks[name]
        self._save()
        return True

    def get(self, name: str) -> Optional[Bookmark]:
        """获取书签"""
        return self._bookmarks.get(name)

    def list_all(self) -> List[Bookmark]:
        """列出所有书签"""
        return list(self._bookmarks.values())

    def list_defaults(self) -> List[Bookmark]:
        """列出预定义书签"""
        return [bm for bm in self._bookmarks.values() if bm.is_default]

    def list_user(self) -> List[Bookmark]:
        """列出用户自定义书签"""
        return [bm for bm in self._bookmarks.values() if not bm.is_default]

    def exists(self, name: str) -> bool:
        """检查书签是否存在"""
        return name in self._bookmarks

    def get_command(self, name: str) -> Optional[str]:
        """获取书签对应的命令"""
        bm = self._bookmarks.get(name)
        return bm.command if bm else None

    def count(self) -> int:
        """返回书签总数"""
        return len(self._bookmarks)

    def format_list(self) -> str:
        """格式化书签列表"""
        if not self._bookmarks:
            return "（无书签）"
        lines = ["=== 命令书签 ==="]

        # 预定义书签
        defaults = self.list_defaults()
        if defaults:
            lines.append("\n[预定义书签]")
            for bm in sorted(defaults, key=lambda x: x.name):
                lines.append(f"  {bm.name:15s} -> {bm.command}")

        # 用户书签
        user_bms = self.list_user()
        if user_bms:
            lines.append("\n[用户书签]")
            for bm in sorted(user_bms, key=lambda x: x.name):
                desc = f"  # {bm.description}" if bm.description else ""
                lines.append(f"  {bm.name:15s} -> {bm.command}{desc}")

        lines.append(f"\n共 {self.count()} 个书签（{len(defaults)} 预定义 + {len(user_bms)} 用户）")
        return "\n".join(lines)
