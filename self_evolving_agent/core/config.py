"""
配置管理模块 —— 加载和管理系统所有配置参数
支持热更新，所有配置变更记录审计日志
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 导入日志模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger, log_execution


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
        """验证约束参数的合法性"""
        logger.debug("CONFIG_VALIDATE", "验证资源约束配置", module="config",
                    extra={"cpu": self.max_cpu_percent, "mem": self.max_memory_percent})
        assert 0 < self.max_cpu_percent <= 100, "CPU 约束必须在 0-100 之间"
        assert 0 < self.max_memory_percent <= 100, "内存约束必须在 0-100 之间"
        assert self.max_reconstruction_iterations > 0, "重构迭代次数必须大于 0"
        assert 0 < self.storage_retain_ratio <= 1.0, "存储保留比例必须在 0-1 之间"
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
        assert self.max_consecutive_exploration_failures > 0
        assert 0 < self.performance_degradation_threshold < 1.0
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
        assert 0 <= self.exploration_ratio <= 1.0
        assert self.idle_replay_interval_hours > 0
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
        assert abs(total - 1.0) < 1e-6, f"驱力权重之和必须等于 1.0，当前为 {total}"
        assert all(0 <= w <= 1.0 for w in [self.safe, self.gain, self.eff])
        logger.info("CONFIG_VALIDATE", "驱力权重配置验证通过", module="config")
        return True
    
    def to_dict(self) -> Dict[str, float]:
        return {"safe": self.safe, "gain": self.gain, "eff": self.eff}


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
        assert self.perception_interval > 0
        assert self.decision_interval > 0
        assert self.main_loop_frequency > 0
        os.makedirs(self.model_snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        logger.info("CONFIG_VALIDATE", "全部系统配置验证通过", module="config")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": asdict(self.resource),
            "safety": asdict(self.safety),
            "evolution": asdict(self.evolution),
            "drive_weights": self.drive_weights.to_dict(),
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
        """加载配置文件"""
        logger.info("CONFIG_LOAD", f"尝试加载配置文件: {self.config_path}", module="config")
        
        if not os.path.exists(self.config_path):
            logger.warning("CONFIG_LOAD", "配置文件不存在，使用默认配置", module="config",
                          extra={"config_path": self.config_path})
            self._config.validate()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            logger.debug("CONFIG_LOAD", "配置文件解析成功", module="config",
                        extra={"keys": list(data.keys()) if data else []})
            
            # 更新配置
            if data:
                self._update_from_dict(data)
            
            self._config.validate()
            self._last_modified = os.path.getmtime(self.config_path)
            
            logger.info("CONFIG_LOAD", "配置文件加载成功", module="config",
                       extra={"config_path": self.config_path})
            
        except Exception as e:
            logger.error("CONFIG_LOAD", f"配置文件加载失败: {str(e)}", module="config",
                        extra={"error": str(e)})
            # 使用默认配置
            self._config.validate()
        
        return self._config
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """从字典更新配置"""
        if "resource" in data:
            for k, v in data["resource"].items():
                if hasattr(self._config.resource, k):
                    setattr(self._config.resource, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新资源约束: {k}={v}", module="config")
        
        if "safety" in data:
            for k, v in data["safety"].items():
                if hasattr(self._config.safety, k):
                    setattr(self._config.safety, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新安全约束: {k}={v}", module="config")
        
        if "evolution" in data:
            for k, v in data["evolution"].items():
                if hasattr(self._config.evolution, k):
                    setattr(self._config.evolution, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新进化约束: {k}={v}", module="config")
        
        if "drive_weights" in data:
            for k, v in data["drive_weights"].items():
                if hasattr(self._config.drive_weights, k):
                    setattr(self._config.drive_weights, k, v)
                    logger.debug("CONFIG_UPDATE", f"更新驱力权重: {k}={v}", module="config")
        
        # 更新其他字段
        for field_name in ["perception_interval", "decision_interval", "main_loop_frequency",
                          "model_snapshot_dir", "database_path", "log_dir"]:
            if field_name in data:
                setattr(self._config, field_name, data[field_name])
                logger.debug("CONFIG_UPDATE", f"更新配置项: {field_name}={data[field_name]}", module="config")
    
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
