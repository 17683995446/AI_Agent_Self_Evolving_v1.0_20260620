"""
插件化可扩展性测试 —— 验证是否像乐高/电子元器件一样可插拔
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import IPrediction, MetricKeys
from core.registry import ModuleRegistry, get_default_registry
from modules.prediction import PredictionResult


class MockPredictionPlugin(IPrediction):
    """模拟一个外部插件预测器 —— 用于测试可替换性"""

    def __init__(self):
        self.updated = False
        self.learned = False
        self._fixed_prediction = {MetricKeys.CPU: 42.0, MetricKeys.MEMORY: 42.0, MetricKeys.DISK: 42.0}

    def update(self, state: dict) -> None:
        self.updated = True

    def predict(self):
        return PredictionResult(
            predicted_values=self._fixed_prediction,
            confidence=0.99,
            layer="mock_plugin"
        )

    def online_learn(self, actual: dict) -> None:
        self.learned = True


class TestPluginReplaceability(unittest.TestCase):
    """测试模块是否可像乐高一样替换"""

    def test_can_instantiate_mock_plugin(self):
        plugin = MockPredictionPlugin()
        self.assertIsNotNone(plugin)

    def test_mock_plugin_implements_interface(self):
        plugin = MockPredictionPlugin()
        self.assertTrue(isinstance(plugin, IPrediction))

    def test_mock_plugin_predict_returns_result(self):
        plugin = MockPredictionPlugin()
        result = plugin.predict()
        self.assertEqual(result.predicted_values[MetricKeys.CPU], 42.0)

    def test_registry_can_register_plugin(self):
        """注册中心可以注册插件"""
        registry = ModuleRegistry()
        registry.register_prediction("mock", MockPredictionPlugin)
        self.assertTrue(registry.is_registered("prediction", "mock"))

    def test_registry_can_create_plugin(self):
        """注册中心可以通过名称创建插件实例"""
        registry = ModuleRegistry()
        registry.register_prediction("mock", MockPredictionPlugin)
        plugin = registry.create_prediction("mock")
        self.assertIsInstance(plugin, MockPredictionPlugin)
        result = plugin.predict()
        self.assertEqual(result.predicted_values[MetricKeys.CPU], 42.0)

    def _copy_default_registry(self, registry: ModuleRegistry):
        """辅助方法：复制默认注册表的所有模块"""
        default = get_default_registry()
        for name in default.list_predictions():
            registry._prediction[name] = default._prediction[name]
        # 复制其他模块类型（通过内部属性访问）
        for name, cls in default._perception.items():
            registry._perception[name] = cls
        for name, cls in default._drive_engine.items():
            registry._drive_engine[name] = cls
        for name, cls in default._decision.items():
            registry._decision[name] = cls
        for name, cls in default._error_eval.items():
            registry._error_eval[name] = cls
        for name, cls in default._optimizer.items():
            registry._optimizer[name] = cls
        for name, cls in default._memory.items():
            registry._memory[name] = cls

    def test_agent_can_use_plugin_via_config(self):
        """关键测试：Agent可以通过配置使用插件，无需修改main.py"""
        from main import SelfEvolvingAgent

        # 创建自定义注册表并复制默认模块 + 注册插件
        registry = ModuleRegistry()
        self._copy_default_registry(registry)
        registry.register_prediction("mock", MockPredictionPlugin)

        # 通过配置让Agent使用插件
        agent = SelfEvolvingAgent(
            registry=registry,
            module_config={"prediction": "mock"}
        )

        # 验证Agent确实使用了插件
        self.assertIsInstance(agent.prediction, MockPredictionPlugin)

        # 验证插件可以正常工作
        agent.prediction.update({MetricKeys.CPU: 10.0})
        result = agent.prediction.predict()
        self.assertEqual(result.predicted_values[MetricKeys.CPU], 42.0)

    def test_agent_default_behavior_unchanged(self):
        """默认行为不变 —— 不传配置时使用默认模块"""
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        from modules.prediction import PredictionEngine
        self.assertIsInstance(agent.prediction, PredictionEngine)

    def test_plugin_integration_in_full_loop(self):
        """插件可以在完整循环中工作"""
        from main import SelfEvolvingAgent

        registry = ModuleRegistry()
        self._copy_default_registry(registry)
        registry.register_prediction("mock", MockPredictionPlugin)

        agent = SelfEvolvingAgent(
            registry=registry,
            module_config={"prediction": "mock"}
        )

        # 运行Agent几秒验证插件集成
        agent.start()
        import time
        time.sleep(2.0)
        agent.stop()
        self.assertGreater(agent.loop_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
