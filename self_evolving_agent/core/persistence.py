"""
数据持久化模块 —— 实现状态快照、恢复和备份

设计原则：
1. 快照应包含足够信息恢复系统到最近状态
2. 恢复过程必须幂等，可重复执行
3. 备份策略应可配置（频率、保留数量）
"""

import os
import sys
import time
import json
import pickle
import shutil
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.config import config_manager


class Snapshot:
    """状态快照 —— 包含某一时刻的系统完整状态"""

    def __init__(self, version: str = "1.0"):
        self.version = version
        self.timestamp = time.time()
        self.datetime = datetime.now().isoformat()
        self.data: Dict[str, Any] = {}
        self.checksum: str = ""

    def set_data(self, key: str, value: Any) -> None:
        """存储快照数据"""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """获取快照数据"""
        return self.data.get(key, default)

    def compute_checksum(self) -> str:
        """计算数据校验和"""
        try:
            serialized = pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)
            self.checksum = hashlib.sha256(serialized).hexdigest()[:16]
            return self.checksum
        except Exception as e:
            logger.warning("SNAPSHOT", f"校验和计算失败: {e}", module="persistence")
            self.checksum = ""
            return ""

    def verify_checksum(self) -> bool:
        """验证数据完整性"""
        if not self.checksum:
            return True  # 无校验和视为通过
        try:
            serialized = pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)
            computed = hashlib.sha256(serialized).hexdigest()[:16]
            return computed == self.checksum
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "datetime": self.datetime,
            "data_keys": list(self.data.keys()),
            "checksum": self.checksum,
        }


class PersistenceManager:
    """持久化管理器 —— 负责快照的保存、加载、恢复和备份"""

    def __init__(self, snapshot_dir: Optional[str] = None, max_snapshots: int = 5):
        self.config = config_manager.config
        self.snapshot_dir = snapshot_dir or getattr(self.config, 'model_snapshot_dir', './data/snapshots')
        self.max_snapshots = max_snapshots
        os.makedirs(self.snapshot_dir, exist_ok=True)
        logger.info("PERSISTENCE_INIT", "持久化管理器初始化", module="persistence",
                    extra={"snapshot_dir": self.snapshot_dir, "max_snapshots": max_snapshots})

    def _snapshot_path(self, timestamp: float) -> str:
        """生成快照文件路径"""
        filename = f"snapshot_{timestamp:.6f}.pkl"
        return os.path.join(self.snapshot_dir, filename)

    def save_snapshot(self, snapshot: Snapshot) -> str:
        """保存快照到磁盘，返回文件路径"""
        snapshot.compute_checksum()
        path = self._snapshot_path(snapshot.timestamp)
        try:
            with open(path, 'wb') as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("PERSISTENCE_SAVE", "快照保存成功", module="persistence",
                        extra={"path": path, "keys": list(snapshot.data.keys()), "checksum": snapshot.checksum})
            self._cleanup_old_snapshots()
            return path
        except Exception as e:
            logger.error("PERSISTENCE_SAVE", f"快照保存失败: {e}", module="persistence")
            return ""

    def load_snapshot(self, path: str) -> Optional[Snapshot]:
        """从磁盘加载快照"""
        try:
            with open(path, 'rb') as f:
                snapshot = pickle.load(f)
            if not snapshot.verify_checksum():
                logger.error("PERSISTENCE_LOAD", "快照校验和验证失败", module="persistence",
                            extra={"path": path})
                return None
            logger.info("PERSISTENCE_LOAD", "快照加载成功", module="persistence",
                        extra={"path": path, "keys": list(snapshot.data.keys())})
            return snapshot
        except Exception as e:
            logger.error("PERSISTENCE_LOAD", f"快照加载失败: {e}", module="persistence")
            return None

    def list_snapshots(self) -> List[str]:
        """列出所有可用的快照文件，按时间排序（最新的在前）"""
        try:
            files = [f for f in os.listdir(self.snapshot_dir) if f.startswith("snapshot_") and f.endswith(".pkl")]
            files.sort(reverse=True)
            return [os.path.join(self.snapshot_dir, f) for f in files]
        except Exception:
            return []

    def get_latest_snapshot(self) -> Optional[str]:
        """获取最新的快照路径"""
        snapshots = self.list_snapshots()
        return snapshots[0] if snapshots else None

    def _cleanup_old_snapshots(self) -> None:
        """清理旧快照，保留最新的max_snapshots个"""
        snapshots = self.list_snapshots()
        if len(snapshots) > self.max_snapshots:
            for old_path in snapshots[self.max_snapshots:]:
                try:
                    os.remove(old_path)
                    logger.debug("PERSISTENCE_CLEAN", f"清理旧快照: {old_path}", module="persistence")
                except Exception as e:
                    logger.warning("PERSISTENCE_CLEAN", f"清理旧快照失败: {e}", module="persistence")

    def create_backup(self, source_path: str, backup_dir: Optional[str] = None) -> str:
        """创建文件备份"""
        backup_dir = backup_dir or os.path.join(self.snapshot_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.basename(source_path)
        backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}.bak")
        try:
            shutil.copy2(source_path, backup_path)
            logger.info("PERSISTENCE_BACKUP", "备份创建成功", module="persistence",
                        extra={"source": source_path, "backup": backup_path})
            return backup_path
        except Exception as e:
            logger.error("PERSISTENCE_BACKUP", f"备份创建失败: {e}", module="persistence")
            return ""

    def restore_from_latest(self) -> Optional[Snapshot]:
        """从最新快照恢复"""
        latest = self.get_latest_snapshot()
        if not latest:
            logger.warning("PERSISTENCE_RESTORE", "无可用快照", module="persistence")
            return None
        return self.load_snapshot(latest)


class AgentStateSerializer:
    """Agent状态序列化器 —— 将Agent各模块状态提取为可持久化格式"""

    @staticmethod
    def serialize_prediction(prediction) -> Dict[str, Any]:
        """序列化预测引擎状态"""
        return {
            "base_history": list(prediction.base_predictor.history),
            "mid_history": list(prediction.mid_predictor.history),
            "high_history": list(prediction.high_predictor.history),
            "error_history": list(prediction.error_history),
            "consecutive_errors": prediction.consecutive_errors,
        }

    @staticmethod
    def deserialize_prediction(prediction, data: Dict[str, Any]) -> None:
        """恢复预测引擎状态"""
        prediction.base_predictor.history.clear()
        prediction.base_predictor.history.extend(data.get("base_history", []))
        prediction.mid_predictor.history.clear()
        prediction.mid_predictor.history.extend(data.get("mid_history", []))
        prediction.high_predictor.history.clear()
        prediction.high_predictor.history.extend(data.get("high_history", []))
        prediction.error_history.clear()
        prediction.error_history.extend(data.get("error_history", []))
        prediction.consecutive_errors = data.get("consecutive_errors", 0)

    @staticmethod
    def serialize_memory(memory) -> Dict[str, Any]:
        """序列化记忆系统状态"""
        return {
            "short_term": [(item.content, item.timestamp, item.importance) for item in memory.short_term.items],
            "long_term_keys": list(memory.long_term.items.keys()),
            "replay_size": memory._replay.size(),
        }

    @staticmethod
    def serialize_full_agent(agent) -> Snapshot:
        """序列化完整Agent状态"""
        snapshot = Snapshot(version="1.0")
        snapshot.set_data("loop_count", agent.loop_count)
        snapshot.set_data("start_time", agent.start_time)
        snapshot.set_data("prediction", AgentStateSerializer.serialize_prediction(agent.prediction))
        snapshot.set_data("memory", AgentStateSerializer.serialize_memory(agent.memory))
        snapshot.set_data("config", config_manager.config.to_dict() if hasattr(config_manager.config, 'to_dict') else {})
        logger.info("SERIALIZER", "Agent状态序列化完成", module="persistence",
                    extra={"keys": list(snapshot.data.keys())})
        return snapshot

    @staticmethod
    def deserialize_full_agent(agent, snapshot: Snapshot) -> bool:
        """恢复完整Agent状态"""
        try:
            agent.loop_count = snapshot.get_data("loop_count", 0)
            agent.start_time = snapshot.get_data("start_time", time.time())

            pred_data = snapshot.get_data("prediction")
            if pred_data:
                AgentStateSerializer.deserialize_prediction(agent.prediction, pred_data)

            logger.info("SERIALIZER", "Agent状态恢复完成", module="persistence")
            return True
        except Exception as e:
            logger.error("SERIALIZER", f"Agent状态恢复失败: {e}", module="persistence")
            return False
