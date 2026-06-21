"""
U16：命令别名与快捷方式

第一性原理：高频操作应有短别名，减少按键成本。
- 预定义常用别名（s=status, h=health, c=config等）
- 用户可自定义别名（持久化到配置）
- 别名在CLI和Shell中均可用
- 别名解析在参数解析前进行

设计原则：
1. 别名是语法糖：不引入新功能，只缩短输入
2. 别名不与现有命令冲突
3. 别名可链式展开（别名指向另一个别名时递归展开，防环）
4. 别名可带参数：alias_name args... 展开为 full_command args...
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


ALIASES_FILE = Path.home() / ".self_evolving_agent_aliases.json"

# 预定义别名（高频操作的短形式）
DEFAULT_ALIASES: Dict[str, str] = {
    "s": "status",
    "st": "status",
    "h": "health",
    "hd": "health --detail",
    "c": "config",
    "cv": "config validate",
    "cs": "config show",
    "cr": "config reload",
    "cb": "config backup",
    "cl": "config backlist",
    "v": "config validate",
    "d": "doctor",
    "dj": "doctor --json",
    "i": "init",
    "in": "init --non-interactive",
    "ss": "snapshot save",
    "sl": "snapshot list",
    "r": "run",
    "rf": "run --forever",
    "rm": "run --enable-multi-agent",
    "ra": "run --enable-multi-agent --enable-forever",
}


class AliasManager:
    """命令别名管理器"""

    def __init__(self, aliases_file: Optional[str] = None):
        self.aliases_file = aliases_file or str(ALIASES_FILE)
        self.aliases: Dict[str, str] = dict(DEFAULT_ALIASES)
        self._load_user_aliases()

    def _load_user_aliases(self) -> None:
        """加载用户自定义别名"""
        try:
            if os.path.exists(self.aliases_file):
                with open(self.aliases_file, "r", encoding="utf-8") as f:
                    user_aliases = json.load(f)
                if isinstance(user_aliases, dict):
                    self.aliases.update(user_aliases)
        except Exception:
            pass

    def _save_user_aliases(self, user_aliases: Dict[str, str]) -> None:
        """保存用户自定义别名"""
        try:
            with open(self.aliases_file, "w", encoding="utf-8") as f:
                json.dump(user_aliases, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def resolve(self, args: List[str]) -> List[str]:
        """解析别名，返回展开后的参数列表

        支持链式展开（别名指向别名），防环检测。

        Args:
            args: 原始参数列表

        Returns:
            展开后的参数列表
        """
        if not args:
            return args
        seen = set()
        result = list(args)
        # 只展开第一个token（命令名）
        while result and result[0] in self.aliases:
            alias_name = result[0]
            if alias_name in seen:
                break  # 防环
            seen.add(alias_name)
            expansion = self.aliases[alias_name].split()
            result = expansion + result[1:]
        return result

    def add(self, name: str, command: str) -> bool:
        """添加用户自定义别名

        Returns:
            True成功，False失败（与现有命令冲突）
        """
        # 不允许覆盖预定义别名
        if name in DEFAULT_ALIASES:
            return False
        self.aliases[name] = command
        # 保存用户别名
        user_aliases = {k: v for k, v in self.aliases.items() if k not in DEFAULT_ALIASES}
        self._save_user_aliases(user_aliases)
        return True

    def remove(self, name: str) -> bool:
        """删除用户自定义别名

        Returns:
            True成功，False失败（不存在或为预定义别名）
        """
        if name in DEFAULT_ALIASES:
            return False
        if name not in self.aliases:
            return False
        del self.aliases[name]
        user_aliases = {k: v for k, v in self.aliases.items() if k not in DEFAULT_ALIASES}
        self._save_user_aliases(user_aliases)
        return True

    def list_aliases(self) -> List[Tuple[str, str, bool]]:
        """列出所有别名

        Returns:
            [(name, command, is_default), ...] 按名称排序
        """
        result = []
        for name, command in sorted(self.aliases.items()):
            result.append((name, command, name in DEFAULT_ALIASES))
        return result

    def get(self, name: str) -> Optional[str]:
        """获取别名对应的命令"""
        return self.aliases.get(name)

    def is_alias(self, name: str) -> bool:
        """判断是否为别名"""
        return name in self.aliases

    def format_aliases_list(self) -> str:
        """格式化别名列表"""
        lines = ["=== 命令别名 ==="]
        lines.append(f"{'别名':<10} {'命令':<30} {'类型'}")
        lines.append("-" * 55)
        for name, command, is_default in self.list_aliases():
            kind = "预定义" if is_default else "自定义"
            lines.append(f"{name:<10} {command:<30} {kind}")
        return "\n".join(lines)
