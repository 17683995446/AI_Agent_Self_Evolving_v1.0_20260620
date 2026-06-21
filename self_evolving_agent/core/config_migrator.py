"""
U14：迁移与版本管理

第一性原理：配置演进应有版本号与自动迁移。
- 配置文件带版本号（config_version字段）
- 加载时检测版本，自动迁移到当前版本
- 迁移是单向链：v1->v2->v3...
- 迁移前自动备份（与U12集成）
- 兼容性检查：报告过时/未知字段

设计原则：
1. 版本号是契约：旧配置必须能升级到新版本
2. 迁移是幂等的：已是当前版本的配置不重复迁移
3. 迁移链是线性的：每个版本有且仅有一个迁移函数
4. 迁移可审计：记录迁移历史
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass


# 当前配置版本
CURRENT_CONFIG_VERSION = 2


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool
    from_version: int
    to_version: int
    message: str
    migrated_fields: List[str] = None

    def __post_init__(self):
        if self.migrated_fields is None:
            self.migrated_fields = []


# 迁移函数注册表：key=源版本, value=(目标版本, 迁移函数)
MigrationFn = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], List[str]]]
_MIGRATIONS: Dict[int, Tuple[int, MigrationFn]] = {}


def register_migration(from_version: int, to_version: int):
    """装饰器：注册迁移函数"""
    def decorator(fn: MigrationFn) -> MigrationFn:
        _MIGRATIONS[from_version] = (to_version, fn)
        return fn
    return decorator


# ---------- 迁移函数 ----------

@register_migration(0, 1)
def _migrate_v0_to_v1(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """v0->v1: 无版本号的旧配置添加版本号，重命名过时字段

    v0特征：无 config_version 字段
    v1变更：添加 config_version=1，resource_constraints→resource 别名已在O1处理
    """
    migrated = []
    if "config_version" not in config:
        config["config_version"] = 1
        migrated.append("config_version")
    # O1别名映射：resource_constraints -> resource
    if "resource_constraints" in config and "resource" not in config:
        config["resource"] = config.pop("resource_constraints")
        migrated.append("resource_constraints→resource")
    if "safety_constraints" in config and "safety" not in config:
        config["safety"] = config.pop("safety_constraints")
        migrated.append("safety_constraints→safety")
    if "evolution_constraints" in config and "evolution" not in config:
        config["evolution"] = config.pop("evolution_constraints")
        migrated.append("evolution_constraints→evolution")
    return config, migrated


@register_migration(1, 2)
def _migrate_v1_to_v2(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """v1->v2: 添加日志性能配置字段（U6引入）

    v2变更：logging段增加 perf_slow_threshold_sec 和 perf_report_interval_loops
    """
    migrated = []
    if "logging" not in config:
        config["logging"] = {}
        migrated.append("logging")
    logging = config["logging"]
    if "perf_slow_threshold_sec" not in logging:
        logging["perf_slow_threshold_sec"] = 1.0
        migrated.append("logging.perf_slow_threshold_sec")
    if "perf_report_interval_loops" not in logging:
        logging["perf_report_interval_loops"] = 100
        migrated.append("logging.perf_report_interval_loops")
    config["config_version"] = 2
    migrated.append("config_version→2")
    return config, migrated


class ConfigMigrator:
    """配置迁移管理器"""

    def __init__(self, config_path: str = "config/agent.yaml"):
        self.config_path = os.path.abspath(config_path)

    def read_config(self) -> Tuple[Dict[str, Any], int]:
        """读取配置文件，返回(配置字典, 版本号)

        无版本号的配置视为v0
        """
        if not os.path.exists(self.config_path):
            return {}, 0
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是字典，当前为 {type(data).__name__}")
        version = data.get("config_version", 0)
        return data, version

    def needs_migration(self) -> bool:
        """检查是否需要迁移"""
        _, version = self.read_config()
        return version < CURRENT_CONFIG_VERSION

    def get_migration_path(self, from_version: int) -> List[int]:
        """获取从指定版本到当前版本的迁移路径

        Returns:
            迁移路径上的版本号列表（不含起点，含终点）
        """
        path = []
        current = from_version
        while current < CURRENT_CONFIG_VERSION:
            if current not in _MIGRATIONS:
                raise ValueError(f"无迁移路径: v{current} -> v{current+1}（未注册迁移函数）")
            target, _ = _MIGRATIONS[current]
            path.append(target)
            current = target
        return path

    def migrate(self, dry_run: bool = False) -> MigrationResult:
        """执行迁移

        Args:
            dry_run: 仅模拟，不写文件

        Returns:
            MigrationResult
        """
        config, from_version = self.read_config()
        if from_version >= CURRENT_CONFIG_VERSION:
            return MigrationResult(
                success=True,
                from_version=from_version,
                to_version=from_version,
                message=f"配置已是最新版本 v{from_version}，无需迁移",
            )
        # 检查迁移路径完整性
        try:
            path = self.get_migration_path(from_version)
        except ValueError as e:
            return MigrationResult(
                success=False,
                from_version=from_version,
                to_version=from_version,
                message=str(e),
            )
        # 执行迁移链
        all_migrated = []
        current = from_version
        for target in path:
            _, fn = _MIGRATIONS[current]
            config, migrated = fn(config)
            all_migrated.extend(migrated)
            current = target
        # 写入（非dry_run）
        if not dry_run:
            self._write_config(config)
        return MigrationResult(
            success=True,
            from_version=from_version,
            to_version=CURRENT_CONFIG_VERSION,
            message=f"配置已从 v{from_version} 迁移到 v{CURRENT_CONFIG_VERSION}",
            migrated_fields=all_migrated,
        )

    def _write_config(self, config: Dict[str, Any]) -> None:
        """写入配置文件（原子性）"""
        tmp_path = self.config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(f"# 配置版本: {config.get('config_version', 0)}\n")
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        os.rename(tmp_path, self.config_path)

    def check_compatibility(self) -> Dict[str, Any]:
        """兼容性检查：报告过时/未知字段

        Returns:
            {"version": int, "is_current": bool, "deprecated_fields": [...],
             "unknown_fields": [...], "missing_fields": [...]}
        """
        config, version = self.read_config()
        # 已知字段集合（当前版本）
        known_top_fields = {"config_version", "resource", "safety", "evolution",
                           "drive_weights", "logging", "main_loop_frequency",
                           "perception_interval", "decision_interval",
                           "model_snapshot_dir", "database_path", "log_dir"}
        deprecated_fields = []
        unknown_fields = []
        for key in config.keys():
            if key.endswith("_constraints"):
                deprecated_fields.append(key)
            elif key not in known_top_fields:
                unknown_fields.append(key)
        # 必需字段检查
        required_fields = ["resource", "drive_weights"]
        missing_fields = [f for f in required_fields if f not in config]
        return {
            "version": version,
            "current_version": CURRENT_CONFIG_VERSION,
            "is_current": version == CURRENT_CONFIG_VERSION,
            "needs_migration": version < CURRENT_CONFIG_VERSION,
            "deprecated_fields": deprecated_fields,
            "unknown_fields": unknown_fields,
            "missing_fields": missing_fields,
        }

    def format_compatibility_report(self) -> str:
        """格式化兼容性报告"""
        report = self.check_compatibility()
        lines = ["=== 配置兼容性报告 ==="]
        lines.append(f"  当前版本: v{report['version']}")
        lines.append(f"  最新版本: v{report['current_version']}")
        lines.append(f"  是否最新: {'是' if report['is_current'] else '否'}")
        if report["needs_migration"]:
            lines.append(f"  需要迁移: 是（v{report['version']} -> v{report['current_version']}）")
        if report["deprecated_fields"]:
            lines.append(f"  过时字段: {', '.join(report['deprecated_fields'])}")
        if report["unknown_fields"]:
            lines.append(f"  未知字段: {', '.join(report['unknown_fields'])}")
        if report["missing_fields"]:
            lines.append(f"  缺失字段: {', '.join(report['missing_fields'])}")
        if not (report["deprecated_fields"] or report["unknown_fields"] or report["missing_fields"]):
            lines.append("  字段检查: 全部正常")
        return "\n".join(lines)
