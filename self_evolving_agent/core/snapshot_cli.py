"""
U26：快照管理增强 —— 快照CLI辅助

第一性原理：
- 快照是故障恢复的核心，但CLI只支持save/list/restore
- restore --id参数被解析但未使用（bug）
- 用户需要查看快照详情、对比快照差异、删除旧快照、导出快照数据
- 快照管理应轻量级，不需要启动完整Agent

设计：
1. SnapshotCliHelper：快照CLI辅助器
   - 封装PersistenceManager的CLI访问
   - 支持按ID/路径/索引解析快照
   - 格式化快照信息/差异/列表
   - 支持删除和导出快照
2. 修复restore --id bug：将--id参数传递给load_state
"""

import os
import json
import time
from typing import Dict, List, Optional, Any


class SnapshotCliHelper:
    """快照CLI辅助器

    轻量级访问PersistenceManager，不启动完整Agent
    """

    def __init__(self, snapshot_dir: Optional[str] = None):
        from core.persistence import PersistenceManager
        self.manager = PersistenceManager(snapshot_dir=snapshot_dir)

    def list_snapshots(self) -> List[str]:
        """列出所有快照路径（最新在前）"""
        return self.manager.list_snapshots()

    def resolve_snapshot_id(self, snapshot_id: str) -> Optional[str]:
        """解析快照ID为完整路径

        支持的ID格式：
        - 完整路径（如 /path/to/snapshot_xxx.pkl）
        - 文件名（如 snapshot_xxx.pkl）
        - 时间戳（如 123456.789012）
        - 索引（如 1, 2, 3，1=最新）

        Args:
            snapshot_id: 快照ID

        Returns:
            完整路径，不存在返回None
        """
        # 1. 完整路径
        if os.path.isabs(snapshot_id) and os.path.exists(snapshot_id):
            return snapshot_id

        # 2. 相对路径或文件名
        if snapshot_id.endswith(".pkl"):
            # 在快照目录中查找
            full_path = os.path.join(self.manager.snapshot_dir, snapshot_id)
            if os.path.exists(full_path):
                return full_path
            # 也可能是直接路径
            if os.path.exists(snapshot_id):
                return snapshot_id

        # 3. 索引（1-based）
        try:
            idx = int(snapshot_id)
            snapshots = self.list_snapshots()
            if 1 <= idx <= len(snapshots):
                return snapshots[idx - 1]
        except ValueError:
            pass

        # 4. 时间戳前缀匹配（仅匹配snapshot_后的时间戳前缀，避免误匹配）
        snapshots = self.list_snapshots()
        for path in snapshots:
            filename = os.path.basename(path)
            # filename: snapshot_{timestamp:.6f}.pkl
            # 仅当snapshot_id是时间戳前缀时才匹配
            if filename.startswith(f"snapshot_{snapshot_id}"):
                return path

        return None

    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """获取快照详情

        Args:
            snapshot_id: 快照ID

        Returns:
            快照信息字典，不存在返回None
        """
        path = self.resolve_snapshot_id(snapshot_id)
        if path is None:
            return None

        snapshot = self.manager.load_snapshot(path)
        if snapshot is None:
            return None

        info = snapshot.to_dict()
        info["path"] = path
        info["filename"] = os.path.basename(path)
        info["size_bytes"] = os.path.getsize(path)
        info["size_kb"] = round(os.path.getsize(path) / 1024, 2)
        info["checksum_valid"] = snapshot.verify_checksum()
        # 时间格式化
        info["time_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.timestamp))
        return info

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照

        Args:
            snapshot_id: 快照ID

        Returns:
            是否删除成功
        """
        path = self.resolve_snapshot_id(snapshot_id)
        if path is None or not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except Exception:
            return False

    def diff_snapshots(self, id1: str, id2: str) -> Optional[Dict[str, Any]]:
        """对比两个快照的差异

        Args:
            id1: 第一个快照ID
            id2: 第二个快照ID

        Returns:
            差异字典，任一快照不存在返回None
        """
        path1 = self.resolve_snapshot_id(id1)
        path2 = self.resolve_snapshot_id(id2)
        if path1 is None or path2 is None:
            return None

        snap1 = self.manager.load_snapshot(path1)
        snap2 = self.manager.load_snapshot(path2)
        if snap1 is None or snap2 is None:
            return None

        keys1 = set(snap1.data.keys())
        keys2 = set(snap2.data.keys())

        return {
            "snapshot1": {
                "path": path1,
                "timestamp": snap1.timestamp,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap1.timestamp)),
            },
            "snapshot2": {
                "path": path2,
                "timestamp": snap2.timestamp,
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snap2.timestamp)),
            },
            "keys_only_in_1": sorted(keys1 - keys2),
            "keys_only_in_2": sorted(keys2 - keys1),
            "common_keys": sorted(keys1 & keys2),
            "time_diff_sec": abs(snap2.timestamp - snap1.timestamp),
        }

    def export_snapshot(self, snapshot_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """导出快照为JSON

        Args:
            snapshot_id: 快照ID
            output_path: 输出文件路径，None则返回JSON字符串

        Returns:
            output_path为None时返回JSON字符串，否则返回写入的文件路径
        """
        path = self.resolve_snapshot_id(snapshot_id)
        if path is None:
            return None

        snapshot = self.manager.load_snapshot(path)
        if snapshot is None:
            return None

        # 构建可序列化的导出数据
        export_data = {
            "version": snapshot.version,
            "timestamp": snapshot.timestamp,
            "datetime": snapshot.datetime,
            "checksum": snapshot.checksum,
            "data_keys": list(snapshot.data.keys()),
        }

        # 尝试序列化data（可能包含不可JSON序列化的对象）
        serializable_data = {}
        non_serializable_keys = []
        for key, value in snapshot.data.items():
            try:
                json.dumps(value)
                serializable_data[key] = value
            except (TypeError, ValueError):
                non_serializable_keys.append(key)

        export_data["data"] = serializable_data
        export_data["non_serializable_keys"] = non_serializable_keys

        json_str = json.dumps(export_data, ensure_ascii=False, indent=2, default=str)

        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                return output_path
            except Exception:
                return None
        return json_str

    # ---------- 格式化方法 ----------

    def format_list(self) -> str:
        """格式化快照列表"""
        snapshots = self.list_snapshots()
        if not snapshots:
            return "（无快照）"

        lines = [f"=== 快照列表（{len(snapshots)}个）==="]
        lines.append(f"{'序号':<6} {'文件名':<35} {'大小':>10} {'时间'}")
        lines.append("-" * 80)
        for i, path in enumerate(snapshots, 1):
            filename = os.path.basename(path)
            try:
                size = os.path.getsize(path)
                size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"
            except Exception:
                size_str = "-"
            # 从文件名提取时间戳
            # snapshot_{timestamp:.6f}.pkl
            ts_str = filename.replace("snapshot_", "").replace(".pkl", "")
            try:
                ts = float(ts_str)
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            except ValueError:
                time_str = "-"
            lines.append(f"{i:<6} {filename:<35} {size_str:>10} {time_str}")
        return "\n".join(lines)

    def format_info(self, snapshot_id: str) -> str:
        """格式化快照详情"""
        info = self.get_snapshot_info(snapshot_id)
        if info is None:
            return f"快照 '{snapshot_id}' 不存在或加载失败"

        lines = [f"=== 快照详情: {info['filename']} ==="]
        lines.append(f"  路径: {info['path']}")
        lines.append(f"  版本: {info['version']}")
        lines.append(f"  时间: {info['time_str']}")
        lines.append(f"  大小: {info['size_kb']} KB ({info['size_bytes']} bytes)")
        lines.append(f"  校验和: {info['checksum']}")
        lines.append(f"  校验有效: {'是' if info['checksum_valid'] else '否'}")
        lines.append(f"  数据键: {', '.join(info['data_keys']) if info['data_keys'] else '（无）'}")
        return "\n".join(lines)

    def format_diff(self, id1: str, id2: str) -> str:
        """格式化快照差异"""
        diff = self.diff_snapshots(id1, id2)
        if diff is None:
            return "快照对比失败：一个或两个快照不存在"

        lines = ["=== 快照差异 ==="]
        lines.append(f"  快照1: {diff['snapshot1']['time_str']} ({os.path.basename(diff['snapshot1']['path'])})")
        lines.append(f"  快照2: {diff['snapshot2']['time_str']} ({os.path.basename(diff['snapshot2']['path'])})")
        lines.append(f"  时间差: {diff['time_diff_sec']:.1f} 秒")
        lines.append(f"  仅快照1有的键: {diff['keys_only_in_1'] if diff['keys_only_in_1'] else '（无）'}")
        lines.append(f"  仅快照2有的键: {diff['keys_only_in_2'] if diff['keys_only_in_2'] else '（无）'}")
        lines.append(f"  共有键: {diff['common_keys'] if diff['common_keys'] else '（无）'}")
        return "\n".join(lines)
