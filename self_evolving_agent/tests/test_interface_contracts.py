"""
接口契约验证测试 —— 确保所有模块返回值符合接口规范
运行: python -m pytest tests/test_interface_contracts.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import (
    StateKeys, MetricKeys, DriveKeys, ErrorReportKeys,
    IPerception, IPrediction, IDriveEngine, IDecision,
    IErrorEvaluator, IOptimizer, IMemory,
)
from modules.perception import EnvironmentPerception
from modules.drive_engine import DriveEngine
from modules.decision import DecisionEngine
from modules.prediction import PredictionEngine
from modules.error_eval import ErrorEvaluator
from modules.optimizer import Optimizer
from modules.memory import MemorySystem


class TestStateVectorContract(unittest.TestCase):
    """验证状态向量包含所有标准键名"""

    def test_perception_returns_all_state_keys(self):
        p = EnvironmentPerception()
        state = p.get_state_vector()
        for key in StateKeys.all_keys():
            self.assertIn(key, state, f"状态向量缺少标准键: {key}")

    def test_state_values_in_range(self):
        p = EnvironmentPerception()
        state = p.get_state_vector()
        for key in [StateKeys.CPU_PERCENT, StateKeys.MEMORY_PERCENT,
                    StateKeys.DISK_PERCENT, StateKeys.ERROR_RATE]:
            self.assertGreaterEqual(state[key], 0.0)
            self.assertLessEqual(state[key], 1.0)


class TestErrorReportContract(unittest.TestCase):
    """验证误差报告包含所有标准键名"""

    def test_evaluate_returns_all_error_keys(self):
        ee = ErrorEvaluator()
        report = ee.evaluate({"cpu": 10.0}, {"cpu": 12.0})
        for key in [
            ErrorReportKeys.MSE, ErrorReportKeys.MAE,
            ErrorReportKeys.RECONSTRUCTION_SIGNAL,
            ErrorReportKeys.SAMPLE_COUNT, ErrorReportKeys.TIMESTAMP,
        ]:
            self.assertIn(key, report, f"误差报告缺少标准键: {key}")

    def test_empty_evaluate_returns_defaults(self):
        ee = ErrorEvaluator()
        report = ee.evaluate({}, {})
        self.assertEqual(report[ErrorReportKeys.SAMPLE_COUNT], 0)
        self.assertEqual(report[ErrorReportKeys.MSE], 0.0)
        self.assertEqual(report[ErrorReportKeys.MAE], 0.0)
        self.assertFalse(report[ErrorReportKeys.RECONSTRUCTION_SIGNAL])


class TestDriveStateContract(unittest.TestCase):
    """验证驱力状态符合契约"""

    def test_drive_ranges(self):
        de = DriveEngine()
        state = {
            StateKeys.CPU_PERCENT: 0.1, StateKeys.MEMORY_PERCENT: 0.1,
            StateKeys.DISK_PERCENT: 0.1, StateKeys.ERROR_RATE: 0.0,
            StateKeys.LOAD_AVG: 0.1, StateKeys.MODEL_ACCURACY: 0.5,
            StateKeys.AGENT_CPU: 0.1, StateKeys.AGENT_MEMORY: 0.1,
            StateKeys.DISK_IO: 0.1, StateKeys.NET_IO: 0.1,
        }
        ds = de.compute(state, 0, 0, 0.0)
        self.assertGreaterEqual(ds.safety, -1.0)
        self.assertLessEqual(ds.safety, 1.0)
        self.assertGreaterEqual(ds.gain, -1.0)
        self.assertLessEqual(ds.gain, 1.0)
        self.assertGreaterEqual(ds.efficiency, -1.0)
        self.assertLessEqual(ds.efficiency, 1.0)
        self.assertGreaterEqual(ds.intrinsic_motivation, -1.0)
        self.assertLessEqual(ds.intrinsic_motivation, 1.0)
        self.assertGreaterEqual(ds.exploration_ratio, 0.0)
        self.assertLessEqual(ds.exploration_ratio, 1.0)

    def test_drive_summary_has_status(self):
        de = DriveEngine()
        summary = de.get_drive_summary()
        self.assertIn("status", summary)


class TestPredictionResultContract(unittest.TestCase):
    """验证预测结果符合契约"""

    def test_predict_has_metric_keys(self):
        pe = PredictionEngine()
        for i in range(15):
            pe.update({MetricKeys.CPU: float(i), MetricKeys.MEMORY: float(i+5), MetricKeys.DISK: float(i+10)})
        result = pe.predict()
        for key in MetricKeys.all_keys():
            self.assertIn(key, result.predicted_values, f"预测结果缺少指标: {key}")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsNotNone(result.layer)

    def test_predict_all_returns_all_layers(self):
        pe = PredictionEngine()
        for i in range(15):
            pe.update({MetricKeys.CPU: float(i), MetricKeys.MEMORY: float(i+5), MetricKeys.DISK: float(i+10)})
        all_results = pe.predict_all()
        self.assertIn("base", all_results)
        self.assertIn("mid", all_results)
        self.assertIn("high", all_results)


class TestActionResultContract(unittest.TestCase):
    """验证动作结果符合契约"""

    def test_action_result_has_required_fields(self):
        from modules.decision import ActionExecutor, Action, ActionType
        ex = ActionExecutor()
        action = Action(action_type=ActionType.COLLECT_DATA, name="测试", description="test")
        result = ex.execute(action)
        self.assertIsNotNone(result.action)
        self.assertIsInstance(result.success, bool)
        self.assertGreaterEqual(result.elapsed_time, 0.0)
        self.assertIsNotNone(result.timestamp)


class TestMemoryStatsContract(unittest.TestCase):
    """验证记忆统计符合契约"""

    def test_stats_has_required_keys(self):
        ms = MemorySystem()
        stats = ms.get_stats()
        required = ["short_term_size", "long_term_count"]
        for key in required:
            self.assertIn(key, stats, f"统计缺少键: {key}")
        ms.close()


class TestOptimizationResultContract(unittest.TestCase):
    """验证优化结果符合契约"""

    def test_result_has_required_fields(self):
        opt = Optimizer()
        result = opt.online_optimize({"cpu": 10.0}, {"cpu": 12.0}, {"mae": 2.0})
        self.assertIn(result.optimization_type, ["online", "online_skipped"])
        self.assertIsInstance(result.loss, float)
        self.assertGreaterEqual(result.iterations, 0)
        self.assertIsNotNone(result.to_dict())


class TestModuleReplaceabilityContract(unittest.TestCase):
    """验证模块替换后契约仍然成立"""

    def test_mock_plugin_maintains_contract(self):
        """替换预测模块后，状态向量契约仍然成立"""
        from core.registry import ModuleRegistry
        from tests.test_plugin_system import MockPredictionPlugin
        from main import SelfEvolvingAgent

        registry = ModuleRegistry()
        default_reg = __import__('core.registry', fromlist=['get_default_registry']).get_default_registry()
        for name in default_reg.list_predictions():
            registry._prediction[name] = default_reg._prediction[name]
        for name, cls in default_reg._perception.items():
            registry._perception[name] = cls
        for name, cls in default_reg._drive_engine.items():
            registry._drive_engine[name] = cls
        for name, cls in default_reg._decision.items():
            registry._decision[name] = cls
        for name, cls in default_reg._error_eval.items():
            registry._error_eval[name] = cls
        for name, cls in default_reg._optimizer.items():
            registry._optimizer[name] = cls
        for name, cls in default_reg._memory.items():
            registry._memory[name] = cls

        registry.register_prediction("mock", MockPredictionPlugin)

        agent = SelfEvolvingAgent(registry=registry, module_config={"prediction": "mock"})

        # 验证状态向量契约
        state = agent.perception.get_state_vector()
        for key in StateKeys.all_keys():
            self.assertIn(key, state)

        # 验证预测结果契约
        agent.prediction.update({MetricKeys.CPU: 10.0, MetricKeys.MEMORY: 20.0, MetricKeys.DISK: 30.0})
        pred = agent.prediction.predict()
        for key in MetricKeys.all_keys():
            self.assertIn(key, pred.predicted_values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
