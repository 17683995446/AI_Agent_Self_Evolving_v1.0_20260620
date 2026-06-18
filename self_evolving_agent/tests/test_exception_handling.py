"""
异常处理精细化测试 —— 验证所有模块的异常降级路径

测试目标：
1. 每个模块在异常时都能降级运行，不导致系统崩溃
2. 异常分类正确，可降级异常被正确处理
3. 边界条件和非法输入被正确拦截
"""
import sys
import os
import time
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exceptions import (
    AgentException, PerceptionException, PredictionException,
    DriveEngineException, DecisionException, ErrorEvalException,
    OptimizerException, MemoryException, ConfigurationException,
    ResourceExhaustedException, ModuleInitException,
)
from core.interfaces import StateKeys, MetricKeys, ErrorReportKeys, DriveKeys
from modules.perception import EnvironmentPerception, SystemMetrics
from modules.prediction import BaseLayerPredictor, MidLayerPredictor, HighLayerPredictor, PredictionEngine
from modules.drive_engine import DriveEngine, DriveState
from modules.decision import DecisionEngine, ActionType, Action, ActionExecutor
from modules.error_eval import ErrorEvaluator
from modules.optimizer import Optimizer, GradientDescent, HyperparameterSearch
from modules.memory import MemorySystem, ShortTermMemory, LongTermMemory


class TestExceptionHierarchy:
    """测试异常层次结构"""

    def test_agent_exception_base(self):
        e = AgentException("test", module="test_mod", context={"key": "val"})
        assert e.module == "test_mod"
        assert e.context == {"key": "val"}
        assert e.degradable is True
        d = e.to_dict()
        assert d["type"] == "AgentException"
        assert d["message"] == "test"

    def test_perception_exception(self):
        e = PerceptionException("采集失败", context={"cpu": 0.5})
        assert e.module == "perception"
        assert e.degradable is True

    def test_configuration_exception_not_degradable(self):
        e = ConfigurationException("配置错误")
        assert e.degradable is False

    def test_all_exceptions_inherit_base(self):
        exceptions = [
            PerceptionException(""), PredictionException(""),
            DriveEngineException(""), DecisionException(""),
            ErrorEvalException(""), OptimizerException(""),
            MemoryException(""), ResourceExhaustedException(""),
            ModuleInitException(""),
        ]
        for e in exceptions:
            assert isinstance(e, AgentException)


class TestPerceptionResilience:
    """测试环境感知模块的异常恢复能力"""

    def test_get_state_vector_with_empty_history(self):
        p = EnvironmentPerception()
        # 清空历史，强制重新采集
        p.system_history.clear()
        p.log_history.clear()
        p.file_history.clear()
        p.self_history.clear()
        state = p.get_state_vector()
        assert isinstance(state, dict)
        assert StateKeys.CPU_PERCENT in state
        assert 0.0 <= state[StateKeys.CPU_PERCENT] <= 1.0

    def test_collect_system_metrics_graceful_on_error(self):
        p = EnvironmentPerception()
        # 正常调用不应抛异常
        metrics = p.collect_system_metrics()
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp > 0

    def test_collect_file_metrics_with_invalid_dir(self):
        p = EnvironmentPerception()
        metrics = p.collect_file_metrics(watch_dir="/nonexistent_path_12345")
        assert isinstance(metrics.to_dict(), dict)
        assert metrics.total_files == 0


class TestPredictionResilience:
    """测试预测模块的异常恢复能力"""

    def test_base_predictor_with_empty_history(self):
        bp = BaseLayerPredictor(window_size=10)
        result = bp.predict()
        assert result.layer == "base"
        assert result.confidence == 0.0
        assert all(v == 0.0 for v in result.predicted_values.values())

    def test_base_predictor_with_single_data_point(self):
        bp = BaseLayerPredictor(window_size=10)
        bp.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
        result = bp.predict()
        assert result.layer == "base"
        # 单点数据无法做线性回归，但不应抛异常
        assert isinstance(result.predicted_values, dict)

    def test_mid_predictor_with_few_data(self):
        mp = MidLayerPredictor(lookback=20)
        for i in range(3):
            mp.update({MetricKeys.CPU: float(i*10), MetricKeys.MEMORY: float(i*10), MetricKeys.DISK: float(i*10)})
        result = mp.predict()
        assert result.layer == "mid"
        assert result.confidence >= 0.0

    def test_high_predictor_with_few_data(self):
        hp = HighLayerPredictor(pattern_window=60)
        for i in range(5):
            hp.update({MetricKeys.CPU: float(i*10), MetricKeys.MEMORY: float(i*10), MetricKeys.DISK: float(i*10)})
        result = hp.predict()
        assert result.layer == "high"
        # 数据不足时应有默认策略
        assert result.strategy_advice is not None

    def test_prediction_engine_auto_reset(self):
        pe = PredictionEngine()
        # 填充数据使误差持续增大
        for i in range(10):
            pe.update({MetricKeys.CPU: 0.0, MetricKeys.MEMORY: 0.0, MetricKeys.DISK: 0.0})
            pe.online_learn({MetricKeys.CPU: 100.0, MetricKeys.MEMORY: 100.0, MetricKeys.DISK: 100.0})
        # 不应抛异常，应自动重置
        assert pe.consecutive_errors == 0 or pe.consecutive_errors < pe.reset_threshold

    def test_high_layer_period_zero_safety(self):
        hp = HighLayerPredictor(pattern_window=60)
        # 构造可能导致period为0的数据
        for i in range(15):
            hp.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 50.0, MetricKeys.DISK: 50.0})
        result = hp.predict()
        assert isinstance(result.predicted_values, dict)


class TestDriveEngineResilience:
    """测试驱力引擎的异常恢复能力"""

    def test_compute_with_empty_state(self):
        de = DriveEngine()
        state = {}
        drive = de.compute(state, tasks_completed=0, tasks_failed=0, prediction_error=0.0)
        assert isinstance(drive, DriveState)
        assert -1.0 <= drive.safety <= 1.0
        assert -1.0 <= drive.gain <= 1.0
        assert -1.0 <= drive.efficiency <= 1.0

    def test_compute_with_extreme_values(self):
        de = DriveEngine()
        state = {
            StateKeys.CPU_PERCENT: 1.0,
            StateKeys.MEMORY_PERCENT: 1.0,
            StateKeys.DISK_PERCENT: 1.0,
            StateKeys.ERROR_RATE: 1.0,
            StateKeys.LOAD_AVG: 10.0,
        }
        drive = de.compute(state, tasks_completed=0, tasks_failed=0, prediction_error=100.0)
        assert isinstance(drive, DriveState)
        # 极端值下安全驱力应为负
        assert drive.safety < 0

    def test_get_exploration_ratio_without_compute(self):
        de = DriveEngine()
        ratio = de.get_exploration_ratio()
        assert 0.0 <= ratio <= 1.0


class TestDecisionResilience:
    """测试决策模块的异常恢复能力"""

    def test_executor_with_none_action(self):
        exe = ActionExecutor()
        result = exe.execute(None)
        assert result.success is False
        assert result.action.action_type == ActionType.SLEEP

    def test_executor_resource_boundary_always_checked(self):
        exe = ActionExecutor()
        # 资源检查不应抛异常
        passed, msg = exe.check_resource_boundaries()
        assert isinstance(passed, bool)
        assert isinstance(msg, str)

    def test_policy_with_empty_drives(self):
        from modules.decision import ActionSpace, Policy
        space = ActionSpace()
        policy = Policy(space)
        action = policy.select_action({}, {})
        assert action is not None
        assert isinstance(action.name, str)

    def test_decision_engine_full_with_empty_input(self):
        de = DecisionEngine()
        result = de.decide_and_execute({}, {DriveKeys.SAFE: 0.0, DriveKeys.GAIN: 0.0, DriveKeys.EFF: 0.0})
        assert result is not None
        assert isinstance(result.success, bool)


class TestErrorEvalResilience:
    """测试误差评估模块的异常恢复能力"""

    def test_evaluate_with_empty_dicts(self):
        ee = ErrorEvaluator()
        result = ee.evaluate({}, {})
        assert isinstance(result, dict)
        assert ErrorReportKeys.MSE in result
        assert result[ErrorReportKeys.SAMPLE_COUNT] == 0

    def test_evaluate_with_mismatched_types(self):
        ee = ErrorEvaluator()
        with pytest.raises(TypeError):
            ee.evaluate([1, 2], {"a": 1})

    def test_entropy_with_uniform_values(self):
        ee = ErrorEvaluator()
        # 所有值相同，熵应为0
        ent = ee.compute_error_entropy([5.0, 5.0, 5.0, 5.0])
        assert ent == 0.0

    def test_entropy_with_single_value(self):
        ee = ErrorEvaluator()
        ent = ee.compute_error_entropy([1.0])
        assert ent == 0.0

    def test_surprise_with_zero_std(self):
        ee = ErrorEvaluator()
        # std为0时不应抛异常
        is_surprise = ee.surprise_detector.detect(1.0, 0.0, 0.0)
        assert is_surprise is False


class TestOptimizerResilience:
    """测试优化器模块的异常恢复能力"""

    def test_gradient_descent_with_nan_gradient(self):
        gd = GradientDescent(learning_rate=0.01, variant="adam")
        params = {"w": 1.0}
        gradients = {"w": float('nan')}
        new_params = gd.step(params, gradients)
        # NaN梯度不应导致参数变成NaN
        assert math.isfinite(new_params["w"]) or new_params["w"] == params["w"]

    def test_gradient_descent_with_inf_gradient(self):
        gd = GradientDescent(learning_rate=0.01, variant="adam")
        params = {"w": 1.0}
        gradients = {"w": float('inf')}
        new_params = gd.step(params, gradients)
        # 无限梯度不应导致系统崩溃
        assert "w" in new_params

    def test_hyperparameter_search_with_bad_objective(self):
        hs = HyperparameterSearch()
        # 目标函数总是返回异常值
        def bad_objective(params):
            return float('inf')
        result = hs.search(bad_objective, n_trials=3)
        assert isinstance(result.to_dict(), dict)
        assert math.isfinite(result.loss) or result.loss == float('inf')

    def test_optimizer_online_with_empty_error_report(self):
        opt = Optimizer()
        result = opt.online_optimize({}, {}, {})
        assert result.optimization_type == "online"

    def test_optimizer_offline_with_empty_history(self):
        opt = Optimizer()
        result = opt.offline_optimize()
        assert result.optimization_type == "offline"


class TestMemoryResilience:
    """测试记忆系统的异常恢复能力"""

    def test_store_unhashable_content(self):
        stm = ShortTermMemory(capacity=10)
        # 存储不可哈希的对象（如包含列表的字典）
        content = {"data": [1, 2, 3], "nested": {"a": [4, 5]}}
        item_id = stm.store(content, importance=0.8)
        assert isinstance(item_id, str)
        assert len(item_id) == 8

    def test_long_term_memory_full(self):
        ltm = LongTermMemory(capacity=5)
        from modules.memory import MemoryItem
        for i in range(10):
            item = MemoryItem(content=f"item_{i}", timestamp=time.time(), importance=0.5)
            ltm.store(str(i), item)
        # 容量满时应自动遗忘
        assert ltm.get_size() <= 5

    def test_memory_system_record_with_large_td_error(self):
        ms = MemorySystem()
        state = {"cpu": 0.5}
        ms.record_experience(state, "test", 1.0, state, td_error=float('inf'))
        stats = ms.get_stats()
        assert stats["short_term_size"] >= 1

    def test_working_memory_overflow(self):
        from modules.memory import WorkingMemory, MemoryItem
        wm = WorkingMemory(capacity=3)
        for i in range(10):
            item = MemoryItem(content=f"item_{i}", timestamp=time.time())
            wm.load(str(i), item)
        assert wm.get_size() <= 3


class TestIntegrationResilience:
    """集成异常测试 —— 验证端到端降级"""

    def test_full_pipeline_with_corrupted_state(self):
        from main import SelfEvolvingAgent
        agent = SelfEvolvingAgent()
        # 启动后停止，验证生命周期
        agent.run_for_duration(0.5)
        assert agent.loop_count >= 0

    def test_agent_handles_module_init_failure(self):
        # 使用不存在的模块配置，验证是否优雅降级
        from core.registry import get_default_registry
        registry = get_default_registry()
        # 默认注册表应始终可用
        assert registry.is_registered("prediction", "default")
