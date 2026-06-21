"""
配置管理模块 —— 加载和管理系统所有配置参数
支持热更新，所有配置变更记录审计日志
"""
import os
import sys
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

# O6：统一sys.path策略 —— 仅在直接运行时添加路径（避免import副作用）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution
from core.exceptions import ConfigurationException


@dataclass
class ResourceConstraints:
    """资源约束配置 —— 系统的硬性红线，不可突破

    这些约束是Agent系统的"生存底线"，一旦触发，系统将自动采取保护性措施：
    - 降低数据采集频率
    - 跳过本轮模型更新
    - 优先执行资源释放动作

    Attributes:
        max_cpu_percent: Agent自身CPU使用上限（百分比）。
            取值范围: (0, 100]，默认: 20.0
            影响: 超过此值时，决策引擎将拒绝执行非保护性动作
        max_memory_percent: Agent自身内存使用上限（百分比）。
            取值范围: (0, 100]，默认: 20.0
            影响: 超过此值时，记忆系统将暂停写入新经验
        max_reconstruction_iterations: 模型重构最大轮次。
            取值范围: [1, +inf)，默认: 3
            影响: 防止模型在误差持续增大时无限重构，避免资源耗尽
        storage_retain_ratio: 历史快照保留比例。
            取值范围: (0, 1.0]，默认: 0.30
            影响: 控制磁盘占用，旧快照按此比例自动清理
    """
    max_cpu_percent: float = 20.0
    max_memory_percent: float = 20.0
    max_reconstruction_iterations: int = 3
    storage_retain_ratio: float = 0.30
    
    def validate(self) -> bool:
        """验证约束参数的合法性

        U5：使用 ConfigurationException 替代 assert ——
        assert 可被 python -O 禁用且错误信息不友好；配置错误是不可降级的，
        必须抛出带上下文的类型化异常，让用户明确知道哪里错了、如何修。
        """
        logger.debug("CONFIG_VALIDATE", "验证资源约束配置", module="config",
                    extra={"cpu": self.max_cpu_percent, "mem": self.max_memory_percent})
        if not (0 < self.max_cpu_percent <= 100):
            raise ConfigurationException(
                f"CPU 约束必须在 0-100 之间，当前为 {self.max_cpu_percent}",
                context={"field": "max_cpu_percent", "value": self.max_cpu_percent},
            )
        if not (0 < self.max_memory_percent <= 100):
            raise ConfigurationException(
                f"内存约束必须在 0-100 之间，当前为 {self.max_memory_percent}",
                context={"field": "max_memory_percent", "value": self.max_memory_percent},
            )
        if self.max_reconstruction_iterations <= 0:
            raise ConfigurationException(
                f"重构迭代次数必须大于 0，当前为 {self.max_reconstruction_iterations}",
                context={"field": "max_reconstruction_iterations", "value": self.max_reconstruction_iterations},
            )
        if not (0 < self.storage_retain_ratio <= 1.0):
            raise ConfigurationException(
                f"存储保留比例必须在 0-1 之间，当前为 {self.storage_retain_ratio}",
                context={"field": "storage_retain_ratio", "value": self.storage_retain_ratio},
            )
        logger.info("CONFIG_VALIDATE", "资源约束配置验证通过", module="config")
        return True


@dataclass
class SafetyConstraints:
    """行为安全约束配置 —— 防止Agent"越界"操作

    这些约束确保Agent在自主决策时不会危害宿主系统：
    - 禁止访问敏感系统路径
    - 限制连续探索失败的次数
    - 监控性能退化并触发保护

    Attributes:
        forbidden_paths: 禁止访问的文件/目录路径列表。
            默认: ["/boot", "/etc/passwd", "/etc/shadow", "/proc/sys", "/sys/kernel"]
            影响: 文件感知模块将跳过这些路径，决策引擎不会执行涉及这些路径的动作
        max_consecutive_exploration_failures: 最大连续探索失败次数。
            取值范围: [1, +inf)，默认: 5
            影响: 超过此值时，探索比例自动降为0，强制进入利用模式
        performance_degradation_threshold: 性能退化检测阈值。
            取值范围: (0, 1.0)，默认: 0.05
            影响: 当预测准确率下降超过此比例时，触发模型保护性冻结
    """
    forbidden_paths: list = field(default_factory=lambda: [
        "/boot", "/etc/passwd", "/etc/shadow",
        "/proc/sys", "/sys/kernel"
    ])
    max_consecutive_exploration_failures: int = 5
    performance_degradation_threshold: float = 0.05
    
    def validate(self) -> bool:
        logger.debug("CONFIG_VALIDATE", "验证行为安全约束配置", module="config")
        if self.max_consecutive_exploration_failures <= 0:
            raise ConfigurationException(
                f"最大连续探索失败次数必须大于 0，当前为 {self.max_consecutive_exploration_failures}",
                context={"field": "max_consecutive_exploration_failures",
                         "value": self.max_consecutive_exploration_failures},
            )
        if not (0 < self.performance_degradation_threshold < 1.0):
            raise ConfigurationException(
                f"性能退化阈值必须在 0-1 之间（开区间），当前为 {self.performance_degradation_threshold}",
                context={"field": "performance_degradation_threshold",
                         "value": self.performance_degradation_threshold},
            )
        logger.info("CONFIG_VALIDATE", "行为安全约束配置验证通过", module="config")
        return True


@dataclass
class EvolutionConstraints:
    """进化节奏约束配置 —— 控制Agent"学习速度"和"探索欲望"

    这些参数调节Agent在"探索新策略"和"利用已知策略"之间的平衡：
    - 探索比例决定随机尝试新动作的概率
    - 离线复盘间隔控制模型进化的频率
    - 稳态检测让Agent在环境稳定时更敢于探索

    Attributes:
        exploration_ratio: 基础探索比例（epsilon-greedy）。
            取值范围: [0, 1.0]，默认: 0.30
            影响: 30%概率随机选择动作（探索），70%概率选择最优动作（利用）
        idle_replay_interval_hours: 离线复盘间隔（小时）。
            取值范围: (0, +inf)，默认: 6.0
            影响: 每隔此时间，记忆系统执行一次离线复盘，分析历史经验
        steady_state_error_threshold: 稳态判定阈值。
            取值范围: [0, +inf)，默认: 0.01
            影响: 当预测误差连续低于此值时，判定为"稳态"，探索比例自动提升
        exploration_boost_on_steady: 稳态时探索比例提升量。
            取值范围: [0, 1.0]，默认: 0.10
            影响: 稳态时探索比例从30%提升到40%，鼓励发现更优策略
    """
    exploration_ratio: float = 0.30
    idle_replay_interval_hours: float = 6.0
    steady_state_error_threshold: float = 0.01
    exploration_boost_on_steady: float = 0.10
    
    def validate(self) -> bool:
        logger.debug("CONFIG_VALIDATE", "验证进化节奏约束配置", module="config")
        if not (0 <= self.exploration_ratio <= 1.0):
            raise ConfigurationException(
                f"探索比例必须在 0-1 之间，当前为 {self.exploration_ratio}",
                context={"field": "exploration_ratio", "value": self.exploration_ratio},
            )
        if self.idle_replay_interval_hours <= 0:
            raise ConfigurationException(
                f"离线复盘间隔必须大于 0，当前为 {self.idle_replay_interval_hours}",
                context={"field": "idle_replay_interval_hours",
                         "value": self.idle_replay_interval_hours},
            )
        logger.info("CONFIG_VALIDATE", "进化节奏约束配置验证通过", module="config")
        return True


@dataclass
class DriveWeights:
    """驱力权重配置 —— 决定Agent的"性格"和"价值观"

    三大驱力的加权组合构成Agent的内在动机：
    - 安全驱力（Safe）：追求系统稳定，厌恶风险
    - 收益驱力（Gain）：追求预测准确、任务成功
    - 能效驱力（Eff）：追求资源高效利用

    权重分配决定Agent的行为倾向：
    - 保守型: safe=0.6, gain=0.2, eff=0.2 → 优先保证系统安全
    - 进取型: safe=0.2, gain=0.6, eff=0.2 → 优先提升预测性能
    - 节能型: safe=0.2, gain=0.2, eff=0.6 → 优先降低资源消耗

    Attributes:
        safe: 安全驱力权重。
            取值范围: [0, 1.0]，默认: 0.4
            影响: 权重越高，Agent在资源紧张时越保守
        gain: 收益驱力权重。
            取值范围: [0, 1.0]，默认: 0.4
            影响: 权重越高，Agent越积极优化预测模型
        eff: 能效驱力权重。
            取值范围: [0, 1.0]，默认: 0.2
            影响: 权重越高，Agent越注重资源使用效率
            注意: safe + gain + eff 必须等于 1.0
    """
    safe: float = 0.4
    gain: float = 0.4
    eff: float = 0.2

    def validate(self) -> bool:
        logger.debug("CONFIG_VALIDATE", "验证驱力权重配置", module="config",
                    extra={"safe": self.safe, "gain": self.gain, "eff": self.eff})
        total = self.safe + self.gain + self.eff
        if abs(total - 1.0) >= 1e-6:
            raise ConfigurationException(
                f"驱力权重之和必须等于 1.0，当前为 {total}",
                context={"safe": self.safe, "gain": self.gain, "eff": self.eff, "total": total},
            )
        for name, w in [("safe", self.safe), ("gain", self.gain), ("eff", self.eff)]:
            if not (0 <= w <= 1.0):
                raise ConfigurationException(
                    f"驱力权重 {name} 必须在 0-1 之间，当前为 {w}",
                    context={"field": name, "value": w},
                )
        logger.info("CONFIG_VALIDATE", "驱力权重配置验证通过", module="config")
        return True

    def to_dict(self) -> Dict[str, float]:
        return {"safe": self.safe, "gain": self.gain, "eff": self.eff}


@dataclass
class LoggingConfig:
    """H6：日志与性能监控配置 —— 控制日志输出级别和噪声

    第一性原理：
    1. 日志是给人看的，级别应匹配场景（生产INFO/开发DEBUG）
    2. 高频低价值日志应采样，避免淹没关键信息
    3. 函数追踪开销大，生产环境应关闭

    Attributes:
        console_level: 控制台日志级别。默认 "INFO"
        file_level: 文件日志级别。默认 "DEBUG"（保留排查细节）
        func_trace_enabled: 是否启用函数进入/退出追踪。默认 False（生产静默）
        sample_rates: 事件采样率映射 {event_type: every_n}。默认对FUNC_ENTER/FUNC_EXIT每10次记1次
        perf_monitor_enabled: 是否启用性能监控。默认 True
        perf_slow_threshold_sec: 慢操作阈值（秒）。默认 0.1
        perf_report_interval_loops: 每隔多少轮主循环输出性能报告。0=不输出。默认 100
    """
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    func_trace_enabled: bool = False
    sample_rates: Dict[str, int] = field(default_factory=lambda: {
        "FUNC_ENTER": 10,
        "FUNC_EXIT": 10,
    })
    perf_monitor_enabled: bool = True
    perf_slow_threshold_sec: float = 0.1
    perf_report_interval_loops: int = 100

    def validate(self) -> bool:
        logger.debug("CONFIG_VALIDATE", "验证日志配置", module="config")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.console_level.upper() not in valid_levels:
            raise ConfigurationException(
                f"console_level 非法: {self.console_level}，合法值: {valid_levels}",
                context={"field": "console_level", "value": self.console_level},
            )
        if self.file_level.upper() not in valid_levels:
            raise ConfigurationException(
                f"file_level 非法: {self.file_level}，合法值: {valid_levels}",
                context={"field": "file_level", "value": self.file_level},
            )
        if self.perf_slow_threshold_sec < 0:
            raise ConfigurationException(
                f"慢操作阈值不能为负，当前为 {self.perf_slow_threshold_sec}",
                context={"field": "perf_slow_threshold_sec",
                         "value": self.perf_slow_threshold_sec},
            )
        if self.perf_report_interval_loops < 0:
            raise ConfigurationException(
                f"报告间隔不能为负，当前为 {self.perf_report_interval_loops}",
                context={"field": "perf_report_interval_loops",
                         "value": self.perf_report_interval_loops},
            )
        logger.info("CONFIG_VALIDATE", "日志配置验证通过", module="config")
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "console_level": self.console_level,
            "file_level": self.file_level,
            "func_trace_enabled": self.func_trace_enabled,
            "sample_rates": dict(self.sample_rates),
            "perf_monitor_enabled": self.perf_monitor_enabled,
            "perf_slow_threshold_sec": self.perf_slow_threshold_sec,
            "perf_report_interval_loops": self.perf_report_interval_loops,
        }


@dataclass
class SystemConfig:
    """系统总配置 —— Agent系统的"中央控制器"

    整合所有子配置，构成Agent的完整运行参数。修改此配置后，
    系统支持热重载（每10轮自动检测配置文件变更）。

    Attributes:
        resource: 资源约束配置（硬性红线）。
            影响: 控制Agent的资源使用上限，防止拖垮宿主系统
        safety: 行为安全约束配置。
            影响: 限制Agent的操作范围，防止越界访问
        evolution: 进化节奏约束配置。
            影响: 调节学习速度和探索欲望
        drive_weights: 驱力权重配置。
            影响: 决定Agent的决策倾向（保守/进取/节能）
        perception_interval: 数据采集间隔（秒）。
            取值范围: (0, +inf)，默认: 1.0
            影响: 间隔越短，状态感知越实时，但CPU消耗越高
        decision_interval: 决策间隔（秒）。
            取值范围: (0, +inf)，默认: 1.0
            影响: 间隔越短，决策越频繁，但可能过度反应
        model_snapshot_dir: 模型快照保存目录。
            默认: "./data/snapshots"
            影响: 优化器在此目录保存/加载模型状态
        database_path: 长期记忆数据库路径。
            默认: "./data/agent_memory.db"
            影响: SQLite数据库存储所有历史经验
        log_dir: 日志文件保存目录。
            默认: "./logs"
            影响: 结构化日志和审计日志的输出目录
        main_loop_frequency: 主循环频率（Hz）。
            取值范围: (0, +inf)，默认: 1.0
            影响: 频率越高，响应越快，但CPU消耗越高
                1.0Hz = 每秒1轮，10.0Hz = 每秒10轮
    """
    resource: ResourceConstraints = field(default_factory=ResourceConstraints)
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    evolution: EvolutionConstraints = field(default_factory=EvolutionConstraints)
    drive_weights: DriveWeights = field(default_factory=DriveWeights)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    perception_interval: float = 1.0
    decision_interval: float = 1.0
    model_snapshot_dir: str = "./data/snapshots"
    database_path: str = "./data/agent_memory.db"
    log_dir: str = "./logs"
    main_loop_frequency: float = 1.0
    
    def validate(self) -> bool:
        """验证所有配置"""
        logger.info("CONFIG_VALIDATE", "开始验证全部系统配置", module="config")
        self.resource.validate()
        self.safety.validate()
        self.evolution.validate()
        self.drive_weights.validate()
        self.logging.validate()
        if self.perception_interval <= 0:
            raise ConfigurationException(
                f"感知间隔必须大于 0，当前为 {self.perception_interval}",
                context={"field": "perception_interval", "value": self.perception_interval},
            )
        if self.decision_interval <= 0:
            raise ConfigurationException(
                f"决策间隔必须大于 0，当前为 {self.decision_interval}",
                context={"field": "decision_interval", "value": self.decision_interval},
            )
        if self.main_loop_frequency <= 0:
            raise ConfigurationException(
                f"主循环频率必须大于 0，当前为 {self.main_loop_frequency}",
                context={"field": "main_loop_frequency", "value": self.main_loop_frequency},
            )
        os.makedirs(self.model_snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.database_path) or ".", exist_ok=True)
        logger.info("CONFIG_VALIDATE", "全部系统配置验证通过", module="config")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": asdict(self.resource),
            "safety": asdict(self.safety),
            "evolution": asdict(self.evolution),
            "drive_weights": self.drive_weights.to_dict(),
            "logging": self.logging.to_dict(),
            "perception_interval": self.perception_interval,
            "decision_interval": self.decision_interval,
            "model_snapshot_dir": self.model_snapshot_dir,
            "database_path": self.database_path,
            "log_dir": self.log_dir,
            "main_loop_frequency": self.main_loop_frequency,
        }


class ConfigManager:
    """配置管理器 —— 支持加载、验证、热更新"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if ConfigManager._initialized:
            return
        ConfigManager._initialized = True
        
        self.config_path = config_path or os.environ.get("AGENT_CONFIG", "./config/agent.yaml")
        self._config = SystemConfig()
        self._watcher = None
        self._last_modified = 0
        
        logger.info("CONFIG_INIT", "配置管理器初始化", module="config",
                    extra={"config_path": self.config_path})
        
        # 尝试加载配置文件
        self.load()
    
    @log_execution
    def load(self) -> SystemConfig:
        """加载配置文件

        U5：配置加载失败拒绝启动（第一性原理——配置是契约）。
        - 配置文件不存在：可接受，使用默认配置（仅警告）。
        - YAML 语法错误：不可接受，抛出 ConfigurationException 拒绝启动，
          避免静默降级到默认值导致行为偏离用户预期。
        - 配置项非法：由 validate() 抛出 ConfigurationException。
        环境变量覆盖优先级最高：文件加载后再应用环境变量。
        """
        logger.info("CONFIG_LOAD", f"尝试加载配置文件: {self.config_path}", module="config")

        if not os.path.exists(self.config_path):
            logger.warning("CONFIG_LOAD", "配置文件不存在，使用默认配置", module="config",
                          extra={"config_path": self.config_path})
            self._apply_env_overrides()
            self._config.validate()
            return self._config

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            logger.debug("CONFIG_LOAD", "配置文件解析成功", module="config",
                        extra={"keys": list(data.keys()) if data else []})

            # 更新配置
            if data:
                if not isinstance(data, dict):
                    raise ConfigurationException(
                        f"配置文件顶层必须是字典，当前为 {type(data).__name__}",
                        context={"config_path": self.config_path, "got_type": type(data).__name__},
                    )
                self._update_from_dict(data)

            # U5：环境变量覆盖（优先级高于配置文件）
            self._apply_env_overrides()

            self._config.validate()
            self._last_modified = os.path.getmtime(self.config_path)

            logger.info("CONFIG_LOAD", "配置文件加载成功", module="config",
                       extra={"config_path": self.config_path})

        except ConfigurationException:
            # 配置异常不可降级，直接向上抛出拒绝启动
            raise
        except yaml.YAMLError as e:
            # YAML 语法错误：拒绝启动，不静默降级
            logger.error("CONFIG_LOAD", f"配置文件YAML语法错误，拒绝启动: {e}", module="config",
                        extra={"config_path": self.config_path, "error": str(e)})
            raise ConfigurationException(
                f"配置文件YAML语法错误: {e}",
                context={"config_path": self.config_path, "yaml_error": str(e)},
            )
        except Exception as e:
            # 其他IO或未知错误：拒绝启动
            logger.error("CONFIG_LOAD", f"配置文件加载失败，拒绝启动: {e}", module="config",
                        extra={"error": str(e)})
            raise ConfigurationException(
                f"配置文件加载失败: {e}",
                context={"config_path": self.config_path, "error": str(e)},
            )

        return self._config

    def _apply_env_overrides(self) -> None:
        """U5：应用环境变量覆盖

        第一性原理：环境变量 > 配置文件 > 默认值。
        环境变量是部署时的契约，应优先于文件配置。
        支持的环境变量（与 config_center.EnvironmentOverride 对齐）：
        - AGENT_MAX_CPU: 资源CPU上限
        - AGENT_MAX_MEMORY: 资源内存上限
        - AGENT_EXPLORATION_RATIO: 探索比例
        - AGENT_PERCEPTION_INTERVAL: 感知间隔
        - AGENT_DECISION_INTERVAL: 决策间隔
        - AGENT_LOOP_FREQUENCY: 主循环频率
        - AGENT_LOG_DIR: 日志目录
        - AGENT_DB_PATH: 数据库路径
        - AGENT_SNAPSHOT_DIR: 快照目录
        - AGENT_LOG_LEVEL: 日志级别（console_level）
        """
        _ENV_MAPPING = {
            "AGENT_MAX_CPU": ("resource", "max_cpu_percent", float),
            "AGENT_MAX_MEMORY": ("resource", "max_memory_percent", float),
            "AGENT_MAX_RECONSTRUCT": ("resource", "max_reconstruction_iterations", int),
            "AGENT_STORAGE_RETAIN": ("resource", "storage_retain_ratio", float),
            "AGENT_EXPLORATION_RATIO": ("evolution", "exploration_ratio", float),
            "AGENT_REPLAY_INTERVAL": ("evolution", "idle_replay_interval_hours", float),
            "AGENT_SAFE_WEIGHT": ("drive_weights", "safe", float),
            "AGENT_GAIN_WEIGHT": ("drive_weights", "gain", float),
            "AGENT_EFF_WEIGHT": ("drive_weights", "eff", float),
            "AGENT_PERCEPTION_INTERVAL": (None, "perception_interval", float),
            "AGENT_DECISION_INTERVAL": (None, "decision_interval", float),
            "AGENT_LOOP_FREQUENCY": (None, "main_loop_frequency", float),
            "AGENT_SNAPSHOT_DIR": (None, "model_snapshot_dir", str),
            "AGENT_DB_PATH": (None, "database_path", str),
            "AGENT_LOG_DIR": (None, "log_dir", str),
            "AGENT_LOG_LEVEL": ("logging", "console_level", str),
        }

        for env_key, (sub_cfg, field_name, caster) in _ENV_MAPPING.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                value = caster(raw)
            except (ValueError, TypeError) as e:
                raise ConfigurationException(
                    f"环境变量 {env_key}={raw!r} 无法转换为 {caster.__name__}: {e}",
                    context={"env_key": env_key, "raw_value": raw, "caster": caster.__name__},
                )
            target = self._config if sub_cfg is None else getattr(self._config, sub_cfg)
            setattr(target, field_name, value)
            logger.info("CONFIG_ENV", f"环境变量覆盖: {env_key}={value}", module="config")
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """从字典更新配置

        支持别名映射，使场景配置（scenarios.yaml）与主配置（agent.yaml）键名兼容：
        - resource_constraints → resource
        - safety_constraints → safety
        - evolution_constraints → evolution
        """
        # 键名别名映射：场景配置使用 _constraints 后缀，主配置使用简短名
        ALIAS = {
            "resource_constraints": "resource",
            "safety_constraints": "safety",
            "evolution_constraints": "evolution",
        }
        normalized = {}
        for k, v in data.items():
            normalized[ALIAS.get(k, k)] = v

        if "resource" in normalized:
            for k, v in normalized["resource"].items():
                if hasattr(self._config.resource, k):
                    setattr(self._config.resource, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新资源约束: {k}={v}", module="config")

        if "safety" in normalized:
            for k, v in normalized["safety"].items():
                if hasattr(self._config.safety, k):
                    setattr(self._config.safety, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新安全约束: {k}={v}", module="config")

        if "evolution" in normalized:
            for k, v in normalized["evolution"].items():
                if hasattr(self._config.evolution, k):
                    setattr(self._config.evolution, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新进化约束: {k}={v}", module="config")

        if "drive_weights" in normalized:
            for k, v in normalized["drive_weights"].items():
                if hasattr(self._config.drive_weights, k):
                    setattr(self._config.drive_weights, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新驱力权重: {k}={v}", module="config")

        # H6: 日志与性能监控配置
        if "logging" in normalized:
            log_data = normalized["logging"]
            if isinstance(log_data, dict):
                for k, v in log_data.items():
                    if hasattr(self._config.logging, k):
                        setattr(self._config.logging, k, v)
                        logger.debug("CONFIG_UPDATE", f"更新日志配置: {k}={v}", module="config")

        # 更新其他字段
        for field_name in ["perception_interval", "decision_interval", "main_loop_frequency",
                          "model_snapshot_dir", "database_path", "log_dir"]:
            if field_name in normalized:
                setattr(self._config, field_name, normalized[field_name])
                logger.debug("CONFIG_UPDATE", f"更新配置项: {field_name}={normalized[field_name]}", module="config")
    
    @property
    def config(self) -> SystemConfig:
        """获取当前配置"""
        return self._config
    
    @log_execution
    def check_hot_reload(self) -> bool:
        """检查配置文件是否变更，如变更则热加载"""
        if not os.path.exists(self.config_path):
            return False
        
        current_mtime = os.path.getmtime(self.config_path)
        if current_mtime > self._last_modified:
            logger.info("CONFIG_HOT_RELOAD", "检测到配置文件变更，执行热加载", module="config")
            self.load()
            return True
        return False


# 全局配置实例
config_manager = ConfigManager()
