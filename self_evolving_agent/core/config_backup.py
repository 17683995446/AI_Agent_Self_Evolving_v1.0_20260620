"""
U12：自动备份与回滚

第一性原理：配置变更不可逆是反人性的，应可回滚。
- 配置变更前自动备份（init/reload/preset应用前）
- 备份带时间戳，可列出与回滚
- 保留最近N个备份，自动清理旧备份
- 回滚操作本身也先备份当前（损坏）配置

设计原则：
1. 备份是安全网：变更前必备份，即使变更失败也能恢复
2. 备份有上限：保留最近N个，避免磁盘膨胀
3. 回滚可审计：记录回滚操作历史
4. 备份原子性：先写临时文件再rename，避免半写状态
"""

import os
import shutil
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BackupEntry:
    """备份条目"""
    path: str
    timestamp: float
    reason: str  # 备份原因（init/reload/preset/rollback/manual）
    size: int

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "timestamp": self.timestamp,
            "time_str": self.time_str,
            "reason": self.reason,
            "size": self.size,
        }


class ConfigBackupManager:
    """配置备份管理器

    管理配置文件的备份与回滚。
    备份目录默认为配置文件同级的 backups/ 子目录。
    """

    def __init__(
        self,
        config_path: str = "config/agent.yaml",
        backup_dir: Optional[str] = None,
        max_backups: int = 10,
    ):
        self.config_path = os.path.abspath(config_path)
        if backup_dir:
            self.backup_dir = os.path.abspath(backup_dir)
        else:
            self.backup_dir = os.path.join(os.path.dirname(self.config_path), "backups")
        self.max_backups = max_backups

    def _ensure_backup_dir(self) -> None:
        os.makedirs(self.backup_dir, exist_ok=True)

    def backup(self, reason: str = "manual") -> Optional[BackupEntry]:
        """备份当前配置文件

        Args:
            reason: 备份原因（init/reload/preset/rollback/manual）

        Returns:
            BackupEntry 或 None（配置文件不存在时）
        """
        if not os.path.exists(self.config_path):
            return None
        self._ensure_backup_dir()
        timestamp = time.time()
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        # 加毫秒避免同秒内多次备份文件名冲突
        ms = int((timestamp % 1) * 1000)
        backup_name = f"agent_{time_str}_{ms:03d}_{reason}.yaml"
        backup_path = os.path.join(self.backup_dir, backup_name)
        # 原子性备份：先复制到临时文件再rename
        tmp_path = backup_path + ".tmp"
        shutil.copy2(self.config_path, tmp_path)
        os.rename(tmp_path, backup_path)
        size = os.path.getsize(backup_path)
        entry = BackupEntry(
            path=backup_path, timestamp=timestamp, reason=reason, size=size
        )
        self._cleanup_old_backups()
        return entry

    def list_backups(self) -> List[BackupEntry]:
        """列出所有备份（按时间倒序）"""
        if not os.path.exists(self.backup_dir):
            return []
        entries = []
        for name in os.listdir(self.backup_dir):
            if not name.startswith("agent_") or not name.endswith(".yaml"):
                continue
            if name.endswith(".tmp"):
                continue
            full_path = os.path.join(self.backup_dir, name)
            if not os.path.isfile(full_path):
                continue
            # 解析原因：文件名格式 agent_YYYYMMDD_HHMMSS_MMM_reason.yaml
            # 去掉前缀 agent_ 和后缀 .yaml 后，按 _ 分割，最后一段是 reason
            stem = name[len("agent_"):-len(".yaml")]
            parts = stem.split("_")
            reason = parts[-1] if len(parts) >= 4 else "unknown"
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue
            entries.append(BackupEntry(
                path=full_path,
                timestamp=mtime,
                reason=reason,
                size=os.path.getsize(full_path),
            ))
        # 按文件名排序（文件名含时间戳+毫秒，比mtime更可靠，
        # 因为shutil.copy2会保留源文件mtime导致多个备份mtime相同）
        entries.sort(key=lambda e: os.path.basename(e.path), reverse=True)
        return entries

    def rollback(self, backup_name: Optional[str] = None) -> Optional[BackupEntry]:
        """回滚到指定备份（或最近一个备份）

        Args:
            backup_name: 备份文件名，None则回滚到最近一个

        Returns:
            回滚前的配置备份（BackupEntry），或None（无备份可回滚）

        Raises:
            FileNotFoundError: 指定的备份不存在
        """
        backups = self.list_backups()
        if not backups:
            return None
        # 确定目标备份
        if backup_name:
            target = None
            for b in backups:
                if b.name == backup_name:
                    target = b
                    break
            if target is None:
                raise FileNotFoundError(f"备份不存在: {backup_name}")
        else:
            target = backups[0]
        # 回滚前先备份当前配置（安全网）
        pre_rollback_backup = None
        if os.path.exists(self.config_path):
            pre_rollback_backup = self.backup(reason="pre_rollback")
        # 恢复
        self._ensure_backup_dir()
        tmp_path = self.config_path + ".tmp"
        shutil.copy2(target.path, tmp_path)
        os.rename(tmp_path, self.config_path)
        return pre_rollback_backup

    def _cleanup_old_backups(self) -> None:
        """清理旧备份，保留最近 max_backups 个"""
        backups = self.list_backups()
        if len(backups) <= self.max_backups:
            return
        for entry in backups[self.max_backups:]:
            try:
                os.remove(entry.path)
            except OSError:
                pass

    def format_backups_list(self) -> str:
        """格式化备份列表为可读文本"""
        backups = self.list_backups()
        if not backups:
            return "无可用备份"
        lines = [f"=== 配置备份列表 ({len(backups)}个) ==="]
        for i, b in enumerate(backups, 1):
            lines.append(f"  {i}. {b.name}")
            lines.append(f"     时间: {b.time_str}  原因: {b.reason}  大小: {b.size}B")
        return "\n".join(lines)
