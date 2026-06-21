"""
H5阶段验证测试 —— 场景配置加载器

验证点：
1. ScenarioLoader类存在且可创建
2. 加载scenarios.yaml，列出5个预置场景
3. load()返回Scenario对象，字段完整
4. config_overrides应用到config_manager
5. setup_alert_channels正确创建通道
6. setup_executor_whitelist正确设置白名单
7. create_agent_from_scenario便捷函数
8. 不存在场景抛ValueError
9. 空配置/缺字段容错
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scenario_loader import (
    ScenarioLoader, Scenario, create_agent_from_scenario,
)
from core.config import config_manager


@pytest.fixture
def loader():
    """默认场景加载器（指向项目内置scenarios.yaml）"""
    return ScenarioLoader()


@pytest.fixture
def isolated_config():
    """隔离config_manager状态，避免用例间污染"""
    original = config_manager.config
    # 备份关键字段
    backup = {
        "max_cpu": original.resource.max_cpu_percent,
        "max_mem": original.resource.max_memory_percent,
        "safe": original.drive_weights.safe,
        "gain": original.drive_weights.gain,
        "eff": original.drive_weights.eff,
    }
    yield config_manager
    # 恢复
    original.resource.max_cpu_percent = backup["max_cpu"]
    original.resource.max_memory_percent = backup["max_mem"]
    original.drive_weights.safe = backup["safe"]
    original.drive_weights.gain = backup["gain"]
    original.drive_weights.eff = backup["eff"]


class TestH5LoaderCreation:
    """验证加载器创建"""

    def test_scenario_loader_exists(self):
        assert ScenarioLoader is not None

    def test_scenario_dataclass_exists(self):
        assert Scenario is not None

    def test_create_default_loader(self, loader):
        assert loader is not None
        assert loader.scenarios_path.endswith("scenarios.yaml")

    def test_create_loader_with_custom_path(self, tmp_path):
        """指定自定义路径"""
        p = tmp_path / "scenarios.yaml"
        p.write_text("scenarios:\n  test:\n    description: 'test'\n")
        l = ScenarioLoader(scenarios_path=str(p))
        assert "test" in l.list_scenarios()

    def test_loader_handles_missing_file(self, tmp_path):
        """文件不存在时不崩溃"""
        l = ScenarioLoader(scenarios_path=str(tmp_path / "nonexistent.yaml"))
        assert l.list_scenarios() == []


class TestH5ListScenarios:
    """验证场景列表"""

    def test_list_scenarios_returns_list(self, loader):
        result = loader.list_scenarios()
        assert isinstance(result, list)

    def test_five_preset_scenarios_loaded(self, loader):
        """5个预置场景全部加载"""
        names = loader.list_scenarios()
        assert len(names) >= 5
        for expected in [
            "system_monitor",
            "log_anomaly_detection",
            "microservice_guardian",
            "dev_env_optimizer",
            "multi_agent_cluster",
        ]:
            assert expected in names, f"缺少预置场景: {expected}"

    def test_get_scenario_info(self, loader):
        info = loader.get_scenario_info("system_monitor")
        assert info is not None
        assert "description" in info
        assert "modules" in info


class TestH5LoadScenario:
    """验证load方法"""

    def test_load_returns_scenario_object(self, loader, isolated_config):
        s = loader.load("system_monitor")
        assert isinstance(s, Scenario)

    def test_scenario_name_correct(self, loader, isolated_config):
        s = loader.load("system_monitor")
        assert s.name == "system_monitor"

    def test_scenario_description_nonempty(self, loader, isolated_config):
        s = loader.load("system_monitor")
        assert s.description != ""

    def test_scenario_module_config_present(self, loader, isolated_config):
        s = loader.load("system_monitor")
        assert isinstance(s.module_config, dict)
        assert "perception" in s.module_config
        assert "prediction" in s.module_config
        assert "decision" in s.module_config

    def test_scenario_adapters_list(self, loader, isolated_config):
        s = loader.load("system_monitor")
        assert isinstance(s.adapters, list)
        assert len(s.adapters) > 0
        assert s.adapters[0]["type"] in ("log_adapter", "action_adapter")

    def test_load_nonexistent_raises(self, loader):
        with pytest.raises(ValueError, match="不存在"):
            loader.load("nonexistent_scenario_xyz")

    def test_load_nonexistent_lists_available(self, loader):
        """错误信息中包含可用场景"""
        try:
            loader.load("xxx")
        except ValueError as e:
            assert "system_monitor" in str(e)


class TestH5ConfigOverrides:
    """验证配置覆盖应用"""

    def test_resource_constraints_overridden(self, loader, isolated_config):
        """resource_constraints覆盖到config_manager"""
        s = loader.load("system_monitor")
        # system_monitor设置max_cpu_percent=80
        assert config_manager.config.resource.max_cpu_percent == 80.0
        assert config_manager.config.resource.max_memory_percent == 85.0

    def test_drive_weights_overridden(self, loader, isolated_config):
        """drive_weights覆盖到config_manager"""
        loader.load("system_monitor")
        # system_monitor: safety_weight=0.5, gain_weight=0.3, efficiency_weight=0.2
        assert config_manager.config.drive_weights.safe == 0.5
        assert config_manager.config.drive_weights.gain == 0.3
        assert config_manager.config.drive_weights.eff == 0.2

    def test_dev_env_optimizer_weights(self, loader, isolated_config):
        """dev_env_optimizer: efficiency优先"""
        loader.load("dev_env_optimizer")
        assert config_manager.config.drive_weights.eff == 0.5
        assert config_manager.config.drive_weights.safe == 0.2

    def test_empty_overrides_no_crash(self, tmp_path, isolated_config):
        """空config_overrides不崩溃"""
        p = tmp_path / "s.yaml"
        p.write_text("scenarios:\n  empty:\n    description: 'empty'\n")
        l = ScenarioLoader(scenarios_path=str(p))
        s = l.load("empty")
        assert s.config_overrides == {}


class TestH5SetupAlertChannels:
    """验证告警通道设置"""

    def test_setup_alert_channels_empty(self, loader, isolated_config):
        """无alert_channels时返回0"""
        s = loader.load("system_monitor")
        am = MagicMock()
        count = loader.setup_alert_channels(s, am)
        # system_monitor未配置alert_channels字段
        assert count == 0
        am.setup_channels_from_config.assert_not_called()

    def test_setup_alert_channels_with_config(self, loader, isolated_config):
        """有alert_channels时调用alert_manager"""
        s = loader.load("system_monitor")
        s.alert_channels = [
            {"type": "dingtalk", "webhook_url": "http://x", "name": "dt"},
        ]
        am = MagicMock()
        am.setup_channels_from_config.return_value = 1
        count = loader.setup_alert_channels(s, am)
        assert count == 1
        am.setup_channels_from_config.assert_called_once_with(s.alert_channels)


class TestH5SetupExecutorWhitelist:
    """验证执行器白名单设置"""

    def test_setup_whitelist_applies_services(self, loader, isolated_config):
        s = loader.load("system_monitor")
        s.allowed_services = ["nginx", "redis"]
        executor = MagicMock()
        executor.allowed_services = []
        loader.setup_executor_whitelist(s, executor)
        assert executor.allowed_services == ["nginx", "redis"]

    def test_setup_whitelist_applies_kill_patterns(self, loader, isolated_config):
        s = loader.load("system_monitor")
        s.allowed_kill_patterns = ["zombie_proc"]
        executor = MagicMock()
        executor.allowed_kill_patterns = []
        loader.setup_executor_whitelist(s, executor)
        assert executor.allowed_kill_patterns == ["zombie_proc"]

    def test_setup_whitelist_empty_no_change(self, loader, isolated_config):
        """无白名单字段时不修改executor"""
        s = loader.load("system_monitor")
        # system_monitor未配置allowed_services
        executor = MagicMock()
        executor.allowed_services = ["existing"]
        executor.allowed_kill_patterns = ["existing_kill"]
        loader.setup_executor_whitelist(s, executor)
        # 没有字段时不应该覆盖
        assert executor.allowed_services == ["existing"]


class TestH5BuildAgentKwargs:
    """验证构造Agent参数"""

    def test_build_kwargs_contains_module_config(self, loader, isolated_config):
        s = loader.load("system_monitor")
        kwargs = loader.build_agent_kwargs(s)
        assert "module_config" in kwargs
        assert kwargs["module_config"] == s.module_config


class TestH5CreateAgentFromScenario:
    """验证便捷函数"""

    def test_create_agent_from_scenario_returns_agent(self, isolated_config):
        """端到端：从场景名创建Agent"""
        agent = create_agent_from_scenario("system_monitor")
        assert agent is not None
        assert hasattr(agent, "decision")
        assert hasattr(agent, "alert_manager")

    def test_create_agent_applies_whitelist(self, isolated_config):
        """场景白名单应用到executor"""
        # 先给system_monitor加白名单字段
        from core.scenario_loader import ScenarioLoader, Scenario
        loader = ScenarioLoader()
        s = loader.load("system_monitor")
        s.allowed_services = ["nginx"]
        s.allowed_kill_patterns = ["defunct"]
        agent = MagicMock()
        agent.decision.executor = MagicMock()
        agent.decision.executor.allowed_services = []
        loader.setup_executor_whitelist(s, agent.decision.executor)
        assert agent.decision.executor.allowed_services == ["nginx"]


class TestH5ScenarioDataclass:
    """验证Scenario数据类"""

    def test_scenario_to_dict(self):
        s = Scenario(
            name="test",
            description="desc",
            module_config={"perception": "default"},
            config_overrides={"k": "v"},
            adapters=[{"type": "log_adapter"}],
            alert_channels=[{"type": "dingtalk"}],
            allowed_services=["svc"],
            allowed_kill_patterns=["pat"],
        )
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"
        assert d["module_config"]["perception"] == "default"
        assert d["config_overrides"]["k"] == "v"
        assert d["adapters"][0]["type"] == "log_adapter"
        assert d["alert_channels"][0]["type"] == "dingtalk"
        assert d["allowed_services"] == ["svc"]
        assert d["allowed_kill_patterns"] == ["pat"]

    def test_scenario_default_factories(self):
        """默认值是独立容器"""
        s1 = Scenario(name="a", description="")
        s2 = Scenario(name="b", description="")
        s1.module_config["x"] = "y"
        # s2不应受影响
        assert "x" not in s2.module_config


class TestH5ScenarioIntegration:
    """验证场景加载与Agent集成"""

    def test_all_scenarios_loadable(self, isolated_config):
        """所有5个场景都能成功加载"""
        loader = ScenarioLoader()
        for name in loader.list_scenarios():
            s = loader.load(name)
            assert isinstance(s, Scenario)
            assert s.name == name
            assert s.description != ""

    def test_microservice_guardian_has_adapters(self, loader, isolated_config):
        s = loader.load("microservice_guardian")
        assert len(s.adapters) > 0
        # 应该有action_adapter
        types = [a.get("type") for a in s.adapters]
        assert "action_adapter" in types

    def test_multi_agent_cluster_has_collaboration(self, loader, isolated_config):
        s = loader.load("multi_agent_cluster")
        assert "collaboration" in s.config_overrides
        assert s.config_overrides["collaboration"]["enabled"] is True
