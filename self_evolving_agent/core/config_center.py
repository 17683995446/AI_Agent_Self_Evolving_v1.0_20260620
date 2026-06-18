"""
配置中心 —— 集中式配置管理，支持环境变量覆盖、热加载、版本控制、多环境隔离

设计原则（第一性原理）：
1. 配置是代码：变更必须可追溯、可回滚
2. 环境优先：环境变量 > 配置文件 > 默认值
3. 变更安全：热加载必须验证通过才生效
"""

import os
import sys
import json
import yaml
import time
import copy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.config import SystemConfig, ConfigManager, ResourceConstraints, SafetyConstraints, EvolutionConstraints, DriveWeights


@dataclass
class ConfigVersion:
    """配置版本记录"""
    version_id: str
    timestamp: float
    config_dict: Dict[str, Any]
    source: str  # "default", "file", "env", "hot_reload"
    checksum: str


class EnvironmentOverride:
    """环境变量覆盖器 —— 将环境变量映射到配置项"""

    # 环境变量前缀
    PREFIX = "AGENT_"

    # 映射表：环境变量名 -> 配置路径（点分隔）
    MAPPING = {
        "AGENT_MAX_CPU": "resource.max_cpu_percent",
        "AGENT_MAX_MEMORY": "resource.max_memory_percent",
        "AGENT_MAX_RECONSTRUCT": "resource.max_reconstruction_iterations",
        "AGENT_STORAGE_RETAIN": "resource.storage_retain_ratio",
        "AGENT_EXPLORATION_RATIO": "evolution.exploration_ratio",
        "AGENT_REPLAY_INTERVAL": "evolution.idle_replay_interval_hours",
        "AGENT_SAFE_WEIGHT": "drive_weights.safe",
        "AGENT_GAIN_WEIGHT": "drive_weights.gain",
        "AGENT_EFF_WEIGHT": "drive_weights.eff",
        "AGENT_PERCEPTION_INTERVAL": "perception_interval",
        "AGENT_DECISION_INTERVAL": "decision_interval",
        "AGENT_LOOP_FREQUENCY": "main_loop_frequency",
        "AGENT_SNAPSHOT_DIR": "model_snapshot_dir",
        "AGENT_DB_PATH": "database_path",
        "AGENT_LOG_DIR": "log_dir",
        "AGENT_ENV": "_env",  # 特殊：指定运行环境
    }

    @classmethod
    def apply(cls, config: SystemConfig) -> Dict[str, Any]:
        """应用环境变量覆盖，返回被覆盖的项"""
        overridden = {}
        for env_key, config_path in cls.MAPPING.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            if config_path == "_env":
                overridden[env_key] = value
                continue

            try:
                # 尝试数值转换
                if "." in value:
                    converted = float(value)
                else:
                    converted = int(value)
            except ValueError:
                converted = value

            # 设置到配置对象
            parts = config_path.split(".")
            target = config
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], converted)
            overridden[env_key] = converted
            logger.info("CONFIG_ENV", f"环境变量覆盖: {env_key}={converted}", module="config_center")

        return overridden


class ConfigVersionControl:
    """配置版本控制 —— 保存历史版本，支持回滚"""

    def __init__(self, max_versions: int = 10):
        self.max_versions = max_versions
        self._versions: List[ConfigVersion] = []
        self._current_index = -1

    def _compute_checksum(self, config_dict: Dict[str, Any]) -> str:
        import hashlib
        content = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def save(self, config: SystemConfig, source: str = "manual") -> ConfigVersion:
        """保存当前配置为一个版本"""
        config_dict = config.to_dict()
        version = ConfigVersion(
            version_id=f"v{len(self._versions) + 1}",
            timestamp=time.time(),
            config_dict=config_dict,
            source=source,
            checksum=self._compute_checksum(config_dict),
        )
        self._versions.append(version)
        self._current_index = len(self._versions) - 1

        # 限制版本数量
        if len(self._versions) > self.max_versions:
            self._versions = self._versions[-self.max_versions:]
            self._current_index = len(self._versions) - 1

        logger.info("CONFIG_VERSION", f"配置版本保存: {version.version_id}", module="config_center",
                    extra={"source": source, "checksum": version.checksum})
        return version

    def rollback(self, steps: int = 1) -> Optional[SystemConfig]:
        """回滚到之前的版本"""
        target_index = self._current_index - steps
        if target_index < 0:
            logger.warning("CONFIG_VERSION", "无法回滚：已到最早版本", module="config_center")
            return None

        version = self._versions[target_index]
        self._current_index = target_index

        # 从字典重建配置
        config = SystemConfig()
        self._apply_dict_to_config(config, version.config_dict)

        logger.info("CONFIG_VERSION", f"配置回滚到: {version.version_id}", module="config_center",
                    extra={"target_index": target_index, "checksum": version.checksum})
        return config

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return [
            {
                "version_id": v.version_id,
                "timestamp": v.timestamp,
                "source": v.source,
                "checksum": v.checksum,
                "is_current": i == self._current_index,
            }
            for i, v in enumerate(self._versions)
        ]

    def _apply_dict_to_config(self, config: SystemConfig, data: Dict[str, Any]) -> None:
        """从字典应用配置"""
        if "resource" in data:
            for k, v in data["resource"].items():
                if hasattr(config.resource, k):
                    setattr(config.resource, k, v)
        if "safety" in data:
            for k, v in data["safety"].items():
                if hasattr(config.safety, k):
                    setattr(config.safety, k, v)
        if "evolution" in data:
            for k, v in data["evolution"].items():
                if hasattr(config.evolution, k):
                    setattr(config.evolution, k, v)
        if "drive_weights" in data:
            for k, v in data["drive_weights"].items():
                if hasattr(config.drive_weights, k):
                    setattr(config.drive_weights, k, v)
        for field_name in ["perception_interval", "decision_interval", "main_loop_frequency",
                          "model_snapshot_dir", "database_path", "log_dir"]:
            if field_name in data:
                setattr(config, field_name, data[field_name])


class ConfigCenter:
    """配置中心 —— 统一管理配置加载、覆盖、热更新、版本控制"""

    def __init__(self, base_config_path: Optional[str] = None):
        self.base_config_path = base_config_path or "./config/agent.yaml"
        self.env = os.environ.get("AGENT_ENV", "dev")
        self.env_config_path = self._get_env_config_path()

        # 独立配置对象（不依赖ConfigManager单例）
        self._config = SystemConfig()
        self._last_modified = 0

        # 版本控制
        self._version_control = ConfigVersionControl(max_versions=20)

        # 保存初始版本
        self._version_control.save(self._config, source="default")

        # 加载基础配置
        self._load_base_config()

        # 加载环境特定配置
        self._load_env_config()

        # 应用环境变量覆盖
        self._env_overrides = EnvironmentOverride.apply(self._config)

        # 保存加载后的版本
        self._version_control.save(self._config, source="initial_load")

        self._last_hot_reload_time = 0.0
        self._hot_reload_count = 0

        logger.info("CONFIG_CENTER", "配置中心初始化完成", module="config_center",
                    extra={"env": self.env, "env_overrides": list(self._env_overrides.keys())})

    def _get_env_config_path(self) -> str:
        """获取环境特定配置文件路径"""
        base = os.path.splitext(self.base_config_path)[0]
        return f"{base}.{self.env}.yaml"

    def _load_base_config(self) -> None:
        """加载基础配置文件"""
        if not os.path.exists(self.base_config_path):
            logger.info("CONFIG_CENTER", f"基础配置不存在，使用默认值: {self.base_config_path}", module="config_center")
            return
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data:
                self._update_config_from_dict(data)
                self._last_modified = os.path.getmtime(self.base_config_path)
                logger.info("CONFIG_CENTER", "基础配置加载成功", module="config_center")
        except Exception as e:
            logger.error("CONFIG_CENTER", f"基础配置加载失败: {e}", module="config_center")

    def _load_env_config(self) -> None:
        """加载环境特定配置（覆盖基础配置）"""
        if not os.path.exists(self.env_config_path):
            logger.info("CONFIG_CENTER", f"环境配置不存在，使用基础配置: {self.env_config_path}", module="config_center")
            return

        try:
            with open(self.env_config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data:
                self._update_config_from_dict(data)
                self._version_control.save(self._config, source=f"env_{self.env}")
                logger.info("CONFIG_CENTER", f"环境配置加载成功: {self.env}", module="config_center")
        except Exception as e:
            logger.error("CONFIG_CENTER", f"环境配置加载失败: {e}", module="config_center")

    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典更新配置（独立实现，不依赖ConfigManager）"""
        if "resource" in data:
            for k, v in data["resource"].items():
                if hasattr(self._config.resource, k):
                    setattr(self._config.resource, k, v)
        if "safety" in data:
            for k, v in data["safety"].items():
                if hasattr(self._config.safety, k):
                    setattr(self._config.safety, k, v)
        if "evolution" in data:
            for k, v in data["evolution"].items():
                if hasattr(self._config.evolution, k):
                    setattr(self._config.evolution, k, v)
        if "drive_weights" in data:
            for k, v in data["drive_weights"].items():
                if hasattr(self._config.drive_weights, k):
                    setattr(self._config.drive_weights, k, v)
        for field_name in ["perception_interval", "decision_interval", "main_loop_frequency",
                          "model_snapshot_dir", "database_path", "log_dir"]:
            if field_name in data:
                setattr(self._config, field_name, data[field_name])

    @property
    def config(self) -> SystemConfig:
        """获取当前生效配置"""
        return self._config

    def hot_reload(self) -> Dict[str, Any]:
        """执行热重载 —— 加载最新配置并验证"""
        result = {"reloaded": False, "changes": [], "error": None}

        try:
            # 1. 检查文件是否修改
            if not os.path.exists(self.base_config_path):
                result["reason"] = "配置文件不存在"
                return result

            current_mtime = os.path.getmtime(self.base_config_path)
            if current_mtime <= self._last_modified:
                result["reason"] = "文件未修改"
                return result

            # 2. 保存当前配置（用于回滚）
            old_config_dict = copy.deepcopy(self._config.to_dict())
            self._version_control.save(self._config, source="pre_hot_reload")

            # 3. 重新加载配置
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data:
                self._update_config_from_dict(data)
            self._last_modified = current_mtime

            # 4. 重新应用环境变量（环境变量优先级最高）
            EnvironmentOverride.apply(self._config)

            # 5. 验证新配置
            self._config.validate()

            # 6. 计算变更
            new_config_dict = self._config.to_dict()
            changes = self._diff_configs(old_config_dict, new_config_dict)

            # 7. 保存新版本
            if changes:
                self._version_control.save(self._config, source="hot_reload")
                self._hot_reload_count += 1
                result["reloaded"] = True
                result["changes"] = changes
                logger.info("CONFIG_CENTER", "热重载成功", module="config_center",
                            extra={"changes": len(changes)})
            else:
                result["reloaded"] = False
                result["reason"] = "无变更"

        except Exception as e:
            # 热重载失败，回滚到上一个版本
            logger.error("CONFIG_CENTER", f"热重载失败，回滚: {e}", module="config_center")
            prev = self._version_control.rollback(steps=1)
            if prev:
                self._config = prev
            result["error"] = str(e)

        return result

    def _diff_configs(self, old: Dict[str, Any], new: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """对比两个字典，返回变更列表"""
        changes = []
        for key in set(list(old.keys()) + list(new.keys())):
            old_val = old.get(key)
            new_val = new.get(key)
            current_path = f"{path}.{key}" if path else key

            if isinstance(old_val, dict) and isinstance(new_val, dict):
                changes.extend(self._diff_configs(old_val, new_val, current_path))
            elif old_val != new_val:
                changes.append({
                    "path": current_path,
                    "old": old_val,
                    "new": new_val,
                })
        return changes

    def rollback(self, steps: int = 1) -> bool:
        """回滚配置"""
        prev = self._version_control.rollback(steps=steps)
        if prev:
            self._config = prev
            logger.info("CONFIG_CENTER", f"配置回滚成功，步数: {steps}", module="config_center")
            return True
        return False

    def get_versions(self) -> List[Dict[str, Any]]:
        """获取配置版本历史"""
        return self._version_control.list_versions()

    def get_stats(self) -> Dict[str, Any]:
        """获取配置中心统计"""
        return {
            "env": self.env,
            "base_config_path": self.base_config_path,
            "env_config_path": self.env_config_path,
            "hot_reload_count": self._hot_reload_count,
            "version_count": len(self._version_control.list_versions()),
            "env_overrides": self._env_overrides,
        }


# 全局配置中心实例
config_center = ConfigCenter()
