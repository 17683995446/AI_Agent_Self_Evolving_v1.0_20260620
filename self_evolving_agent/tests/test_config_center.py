"""
配置中心测试 —— 验证环境变量覆盖、热加载、版本控制、多环境隔离
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_center import ConfigCenter, EnvironmentOverride, ConfigVersionControl
from core.config import SystemConfig


class TestEnvironmentOverride:
    """测试环境变量覆盖"""

    def test_env_override_cpu(self):
        os.environ["AGENT_MAX_CPU"] = "50"
        config = SystemConfig()
        original = config.resource.max_cpu_percent
        overridden = EnvironmentOverride.apply(config)
        assert config.resource.max_cpu_percent == 50.0
        assert "AGENT_MAX_CPU" in overridden
        # 清理
        del os.environ["AGENT_MAX_CPU"]
        # 恢复
        config.resource.max_cpu_percent = original

    def test_env_override_exploration(self):
        os.environ["AGENT_EXPLORATION_RATIO"] = "0.15"
        config = SystemConfig()
        EnvironmentOverride.apply(config)
        assert config.evolution.exploration_ratio == 0.15
        del os.environ["AGENT_EXPLORATION_RATIO"]

    def test_env_override_weights(self):
        os.environ["AGENT_SAFE_WEIGHT"] = "0.5"
        os.environ["AGENT_GAIN_WEIGHT"] = "0.3"
        os.environ["AGENT_EFF_WEIGHT"] = "0.2"
        config = SystemConfig()
        EnvironmentOverride.apply(config)
        assert config.drive_weights.safe == 0.5
        assert config.drive_weights.gain == 0.3
        assert config.drive_weights.eff == 0.2
        del os.environ["AGENT_SAFE_WEIGHT"]
        del os.environ["AGENT_GAIN_WEIGHT"]
        del os.environ["AGENT_EFF_WEIGHT"]

    def test_no_env_no_override(self):
        # 确保没有设置环境变量
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_"):
                del os.environ[key]
        config = SystemConfig()
        original = config.resource.max_cpu_percent
        overridden = EnvironmentOverride.apply(config)
        assert len(overridden) == 0
        assert config.resource.max_cpu_percent == original


class TestConfigVersionControl:
    """测试配置版本控制"""

    def test_save_and_list(self):
        vc = ConfigVersionControl(max_versions=5)
        config = SystemConfig()
        config.resource.max_cpu_percent = 25.0
        vc.save(config, source="test")
        versions = vc.list_versions()
        assert len(versions) == 1
        assert versions[0]["source"] == "test"
        assert versions[0]["is_current"] is True

    def test_rollback(self):
        vc = ConfigVersionControl(max_versions=5)
        config = SystemConfig()
        config.resource.max_cpu_percent = 10.0
        vc.save(config, source="v1")
        config.resource.max_cpu_percent = 20.0
        vc.save(config, source="v2")

        rolled = vc.rollback(steps=1)
        assert rolled is not None
        assert rolled.resource.max_cpu_percent == 10.0

    def test_rollback_too_far(self):
        vc = ConfigVersionControl(max_versions=5)
        config = SystemConfig()
        vc.save(config, source="v1")
        rolled = vc.rollback(steps=5)
        assert rolled is None

    def test_max_versions(self):
        vc = ConfigVersionControl(max_versions=3)
        config = SystemConfig()
        for i in range(5):
            config.resource.max_cpu_percent = float(i)
            vc.save(config, source=f"v{i}")
        versions = vc.list_versions()
        assert len(versions) == 3

    def test_checksum_consistency(self):
        vc = ConfigVersionControl(max_versions=5)
        config = SystemConfig()
        v1 = vc.save(config, source="test")
        v2 = vc.save(config, source="test2")
        assert v1.checksum == v2.checksum  # 相同配置应有相同checksum


class TestConfigCenter:
    """测试配置中心"""

    def setup_method(self):
        # 每个测试前清理AGENT_环境变量
        for key in list(os.environ.keys()):
            if key.startswith("AGENT_"):
                del os.environ[key]

    def test_default_initialization(self):
        # 使用不存在的路径，确保使用默认值
        cc = ConfigCenter(base_config_path="/nonexistent/config.yaml")
        assert cc.config is not None
        assert cc.config.resource.max_cpu_percent == 20.0

    def test_env_config_loading(self):
        # 创建临时环境配置文件
        temp_dir = tempfile.mkdtemp()
        base_path = os.path.join(temp_dir, "agent.yaml")
        env_path = os.path.join(temp_dir, "agent.test.yaml")

        with open(base_path, 'w') as f:
            f.write("resource:\n  max_cpu_percent: 15.0\n")
        with open(env_path, 'w') as f:
            f.write("resource:\n  max_cpu_percent: 25.0\n")

        os.environ["AGENT_ENV"] = "test"
        cc = ConfigCenter(base_config_path=base_path)
        # 环境配置应覆盖基础配置
        assert cc.config.resource.max_cpu_percent == 25.0

        del os.environ["AGENT_ENV"]
        shutil.rmtree(temp_dir)

    def test_hot_reload_no_change(self):
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "agent.yaml")
        with open(config_path, 'w') as f:
            f.write("resource:\n  max_cpu_percent: 30.0\n")

        cc = ConfigCenter(base_config_path=config_path)
        assert cc.config.resource.max_cpu_percent == 30.0

        # 未修改文件，热重载应无变更
        result = cc.hot_reload()
        assert result["reloaded"] is False

        shutil.rmtree(temp_dir)

    def test_hot_reload_with_change(self):
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "agent.yaml")
        with open(config_path, 'w') as f:
            f.write("resource:\n  max_cpu_percent: 30.0\n")

        cc = ConfigCenter(base_config_path=config_path)
        assert cc.config.resource.max_cpu_percent == 30.0

        # 修改文件
        import time
        time.sleep(0.1)
        with open(config_path, 'w') as f:
            f.write("resource:\n  max_cpu_percent: 35.0\n")

        result = cc.hot_reload()
        assert result["reloaded"] is True
        assert len(result["changes"]) > 0
        assert cc.config.resource.max_cpu_percent == 35.0

        shutil.rmtree(temp_dir)

    def test_rollback_integration(self):
        cc = ConfigCenter(base_config_path="/nonexistent/config.yaml")
        original_cpu = cc.config.resource.max_cpu_percent
        cc.config.resource.max_cpu_percent = 99.0
        cc._version_control.save(cc.config, source="modified")

        success = cc.rollback(steps=1)
        assert success is True
        # 回滚后应恢复
        assert cc.config.resource.max_cpu_percent == original_cpu

    def test_get_versions(self):
        cc = ConfigCenter(base_config_path="/nonexistent/config.yaml")
        versions = cc.get_versions()
        assert len(versions) >= 1
        assert versions[-1]["is_current"] is True

    def test_get_stats(self):
        cc = ConfigCenter(base_config_path="/nonexistent/config.yaml")
        stats = cc.get_stats()
        assert "env" in stats
        assert "version_count" in stats
        assert "env_overrides" in stats
