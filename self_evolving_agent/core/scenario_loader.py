"""
H5：场景配置加载器 —— 从scenarios.yaml加载预置场景，一键切换运行模式

第一性原理：
1. 不同部署环境需要不同配置（监控阈值/驱力权重/告警通道）
2. 配置应与代码分离，运维人员改yaml不改代码
3. 场景 = 模块选择 + 配置覆盖 + 适配器声明

用法：
    loader = ScenarioLoader()
    scenario = loader.load("system_monitor")
    agent = SelfEvolvingAgent(
        module_config=scenario.module_config,
        container=scenario.build_container(),
    )
"""

import os
import sys
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logger import logger
from core.config import config_manager, SystemConfig


@dataclass
class Scenario:
    """场景定义"""
    name: str
    description: str
    module_config: Dict[str, str] = field(default_factory=dict)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    adapters: List[Dict[str, Any]] = field(default_factory=list)
    alert_channels: List[Dict[str, Any]] = field(default_factory=list)
    allowed_services: List[str] = field(default_factory=list)
    allowed_kill_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "module_config": self.module_config,
            "config_overrides": self.config_overrides,
            "adapters": self.adapters,
            "alert_channels": self.alert_channels,
            "allowed_services": self.allowed_services,
            "allowed_kill_patterns": self.allowed_kill_patterns,
        }


class ScenarioLoader:
    """场景加载器 —— 读取scenarios.yaml，解析并应用场景配置"""

    DEFAULT_SCENARIOS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "scenarios.yaml"
    )

    def __init__(self, scenarios_path: Optional[str] = None):
        self.scenarios_path = scenarios_path or self.DEFAULT_SCENARIOS_PATH
        self._scenarios: Dict[str, Dict[str, Any]] = {}
        self._load_scenarios_file()

    def _load_scenarios_file(self):
        """加载scenarios.yaml"""
        if not os.path.exists(self.scenarios_path):
            logger.warning("SCENARIO", f"场景配置文件不存在: {self.scenarios_path}", module="scenario")
            return

        try:
            with open(self.scenarios_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._scenarios = data.get("scenarios", {})
            logger.info("SCENARIO", f"加载 {len(self._scenarios)} 个场景配置",
                        module="scenario", extra={"names": list(self._scenarios.keys())})
        except Exception as e:
            logger.error("SCENARIO", f"场景配置加载失败: {e}", module="scenario")

    def list_scenarios(self) -> List[str]:
        """列出所有可用场景"""
        return list(self._scenarios.keys())

    def get_scenario_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取场景信息（不应用）"""
        return self._scenarios.get(name)

    def load(self, name: str) -> Scenario:
        """加载并解析场景，返回Scenario对象

        Args:
            name: 场景名称，如 "system_monitor"

        Returns:
            Scenario对象，包含module_config和config_overrides

        Raises:
            ValueError: 场景不存在
        """
        if name not in self._scenarios:
            available = list(self._scenarios.keys())
            raise ValueError(f"场景 '{name}' 不存在，可用场景: {available}")

        raw = self._scenarios[name]
        logger.info("SCENARIO", f"加载场景: {name} - {raw.get('description', '')}",
                    module="scenario")

        scenario = Scenario(
            name=name,
            description=raw.get("description", ""),
            module_config=raw.get("modules", {}),
            config_overrides=raw.get("config_overrides", {}),
            adapters=raw.get("adapters", []),
            alert_channels=raw.get("alert_channels", []),
            allowed_services=raw.get("allowed_services", []),
            allowed_kill_patterns=raw.get("allowed_kill_patterns", []),
        )

        # 应用配置覆盖到全局config_manager
        self._apply_config_overrides(scenario.config_overrides)

        logger.info("SCENARIO", f"场景 '{name}' 加载完成",
                    module="scenario",
                    extra={"module_config": scenario.module_config,
                           "overrides_keys": list(scenario.config_overrides.keys())})
        return scenario

    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """应用配置覆盖到config_manager"""
        if not overrides:
            return

        config = config_manager.config

        for key, value in overrides.items():
            try:
                if key == "resource_constraints" or key == "resource":
                    if isinstance(value, dict):
                        if "max_cpu_percent" in value:
                            config.resource.max_cpu_percent = float(value["max_cpu_percent"])
                        if "max_memory_percent" in value:
                            config.resource.max_memory_percent = float(value["max_memory_percent"])
                        logger.debug("SCENARIO", f"应用资源约束覆盖: {value}", module="scenario")

                elif key == "safety_constraints" or key == "safety":
                    if isinstance(value, dict):
                        for sk, sv in value.items():
                            setattr(config.safety, sk, sv)

                elif key == "evolution_constraints" or key == "evolution":
                    if isinstance(value, dict):
                        for ek, ev in value.items():
                            setattr(config.evolution, ek, ev)

                elif key == "drive_weights":
                    if isinstance(value, dict):
                        # 兼容两种key格式：safe/safety_weight
                        if "safe" in value:
                            config.drive_weights.safe = float(value["safe"])
                        if "safety_weight" in value:
                            config.drive_weights.safe = float(value["safety_weight"])
                        if "gain" in value:
                            config.drive_weights.gain = float(value["gain"])
                        if "gain_weight" in value:
                            config.drive_weights.gain = float(value["gain_weight"])
                        if "eff" in value:
                            config.drive_weights.eff = float(value["eff"])
                        if "efficiency_weight" in value:
                            config.drive_weights.eff = float(value["efficiency_weight"])
                        logger.debug("SCENARIO", f"应用驱力权重覆盖: {config.drive_weights}", module="scenario")

                elif key == "alert_channels":
                    # 告警通道配置
                    if isinstance(value, list):
                        setattr(config, "alert_channels", value)

                elif key == "collaboration":
                    if isinstance(value, dict):
                        setattr(config, "collaboration", value)

                elif key == "logging":
                    # H6: 日志与性能监控配置覆盖
                    if isinstance(value, dict):
                        for lk, lv in value.items():
                            if hasattr(config.logging, lk):
                                setattr(config.logging, lk, lv)
                        logger.debug("SCENARIO", f"应用日志配置覆盖: {value}", module="scenario")

                else:
                    # 通用覆盖：直接设置属性
                    setattr(config, key, value)
                    logger.debug("SCENARIO", f"应用配置覆盖: {key}={value}", module="scenario")

            except Exception as e:
                logger.warning("SCENARIO", f"应用配置覆盖失败 {key}: {e}", module="scenario")

    def build_agent_kwargs(self, scenario: Scenario) -> Dict[str, Any]:
        """根据场景构造SelfEvolvingAgent的构造参数

        Returns:
            包含module_config和executor白名单的kwargs字典
        """
        kwargs = {
            "module_config": scenario.module_config,
        }
        return kwargs

    def setup_alert_channels(self, scenario: Scenario, alert_manager) -> int:
        """根据场景配置告警通道

        Args:
            scenario: 场景对象
            alert_manager: AlertManager实例

        Returns:
            成功创建的通道数
        """
        if not scenario.alert_channels:
            return 0
        return alert_manager.setup_channels_from_config(scenario.alert_channels)

    def setup_executor_whitelist(self, scenario: Scenario, executor) -> None:
        """根据场景配置执行器白名单

        Args:
            scenario: 场景对象
            executor: ActionExecutor实例
        """
        if scenario.allowed_services:
            executor.allowed_services = scenario.allowed_services
            logger.info("SCENARIO", f"执行器服务白名单已设置: {scenario.allowed_services}",
                        module="scenario")
        if scenario.allowed_kill_patterns:
            executor.allowed_kill_patterns = scenario.allowed_kill_patterns
            logger.info("SCENARIO", f"执行器进程白名单已设置: {scenario.allowed_kill_patterns}",
                        module="scenario")


def create_agent_from_scenario(scenario_name: str,
                                scenarios_path: Optional[str] = None):
    """H5便捷函数：从场景名创建Agent

    用法:
        agent = create_agent_from_scenario("system_monitor")
        agent.start()
    """
    from main import SelfEvolvingAgent

    loader = ScenarioLoader(scenarios_path)
    scenario = loader.load(scenario_name)

    agent = SelfEvolvingAgent(module_config=scenario.module_config)

    # 应用场景到执行器白名单
    if hasattr(agent.decision, "executor"):
        loader.setup_executor_whitelist(scenario, agent.decision.executor)

    # 应用场景到告警通道
    if hasattr(agent, "alert_manager") and agent.alert_manager is not None:
        loader.setup_alert_channels(scenario, agent.alert_manager)

    logger.info("SCENARIO", f"Agent已从场景 '{scenario_name}' 创建完成",
                module="scenario")
    return agent


if __name__ == "__main__":
    # 演示用法
    loader = ScenarioLoader()
    print(f"可用场景: {loader.list_scenarios()}")
    for name in loader.list_scenarios():
        info = loader.get_scenario_info(name)
        print(f"  - {name}: {info.get('description', '')}")
