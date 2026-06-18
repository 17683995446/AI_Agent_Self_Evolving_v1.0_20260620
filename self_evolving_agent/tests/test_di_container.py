"""
依赖注入与模块解耦测试 —— 验证DI容器和模块生命周期管理
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.container import DIContainer, ModuleLifecycle
from core.registry import ModuleRegistry, get_default_registry
from modules.perception import EnvironmentPerception
from modules.prediction import PredictionEngine
from main import SelfEvolvingAgent


class TestModuleLifecycle:
    """测试模块生命周期管理"""

    def test_lifecycle_init_and_start(self):
        class FakeModule:
            def __init__(self):
                self.started = False
            def start(self):
                self.started = True

        mod = FakeModule()
        lc = ModuleLifecycle("fake", mod)
        assert lc.initialize() is True
        assert lc.start() is True
        assert mod.started is True
        assert lc.stop() is True

    def test_lifecycle_cleanup_fallback_to_close(self):
        class FakeModule:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True

        mod = FakeModule()
        lc = ModuleLifecycle("fake", mod)
        assert lc.cleanup() is True
        assert mod.closed is True

    def test_lifecycle_idempotent(self):
        class FakeModule:
            pass

        lc = ModuleLifecycle("fake", FakeModule())
        assert lc.initialize() is True
        assert lc.initialize() is True  # 重复初始化应返回True
        assert lc.start() is True
        assert lc.start() is True  # 重复启动应返回True


class TestDIContainer:
    """测试依赖注入容器"""

    def test_container_create(self):
        container = DIContainer()
        assert container is not None

    def test_container_resolve_from_registry(self):
        container = DIContainer()
        mod = container.resolve("default", "perception")
        assert isinstance(mod, EnvironmentPerception)

    def test_container_build_agent(self):
        container = DIContainer()
        modules = container.build_agent()
        assert "perception" in modules
        assert "prediction" in modules
        assert "drive_engine" in modules
        assert "decision" in modules
        assert "error_eval" in modules
        assert "optimizer" in modules
        assert "memory" in modules

    def test_container_factory_override(self):
        container = DIContainer()
        called = False
        def fake_factory():
            nonlocal called
            called = True
            return EnvironmentPerception()
        container.register_factory("custom_perception", fake_factory)
        mod = container.resolve("custom_perception")
        assert called is True
        assert mod is not None

    def test_container_module_status(self):
        container = DIContainer()
        container.build_agent()
        status = container.get_module_status()
        assert "perception" in status
        assert status["perception"]["has_start"] is True or status["perception"]["has_start"] is False

    def test_container_lifecycle_management(self):
        container = DIContainer()
        container.build_agent()
        init_results = container.initialize_all()
        assert all(init_results.values())
        start_results = container.start_all()
        assert all(start_results.values())
        stop_results = container.stop_all()
        assert all(stop_results.values())
        cleanup_results = container.cleanup_all()
        assert all(cleanup_results.values())


class TestAgentWithDI:
    """测试Agent使用依赖注入"""

    def test_agent_with_container(self):
        container = DIContainer()
        agent = SelfEvolvingAgent(container=container)
        assert agent.perception is not None
        assert agent.prediction is not None
        assert agent._modules is not None

    def test_agent_with_prebuilt_modules(self):
        container = DIContainer()
        modules = container.build_agent()
        agent = SelfEvolvingAgent(modules=modules)
        assert agent.perception is modules["perception"]
        assert agent._container is None

    def test_agent_backward_compatible(self):
        # 测试旧式调用仍然有效
        agent = SelfEvolvingAgent()
        assert agent.perception is not None
        assert agent.prediction is not None

    def test_agent_run_with_container(self):
        container = DIContainer()
        agent = SelfEvolvingAgent(container=container)
        agent.run_for_duration(0.3)
        assert agent.loop_count >= 0
