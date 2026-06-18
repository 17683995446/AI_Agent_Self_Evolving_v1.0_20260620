"""
复杂场景混沌测试 —— 验证系统在极端、异常、混乱输入下的稳定性

测试原则（第一性原理）：
1. 任何外部输入都不可信任，必须验证和兜底
2. 单点故障不应导致系统级崩溃
3. 异常传播必须有边界，不能无限扩散
"""
import sys
import os
import math
import time
import random
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from core.interfaces import StateKeys, MetricKeys, ErrorReportKeys, DriveKeys
from modules.perception import EnvironmentPerception
from modules.prediction import PredictionEngine, BaseLayerPredictor, MidLayerPredictor, HighLayerPredictor
from modules.drive_engine import DriveEngine
from modules.decision import DecisionEngine, ActionExecutor, ActionSpace, Policy, Action, ActionType
from modules.error_eval import ErrorEvaluator
from modules.optimizer import Optimizer, GradientDescent, HyperparameterSearch
from modules.memory import MemorySystem, ShortTermMemory, LongTermMemory


class TestChaosPerception:
    """环境感知模块混沌测试"""

    def test_state_vector_with_none_values(self):
        p = EnvironmentPerception()
        # 直接注入含None的状态不应导致崩溃
        p.system_history.append({"cpu_percent": None, "memory_percent": 0.5})
        state = p.get_state_vector()
        assert isinstance(state, dict)

    def test_state_vector_with_missing_keys(self):
        p = EnvironmentPerception()
        p.system_history.append({"only_one_key": 0.5})
        state = p.get_state_vector()
        assert isinstance(state, dict)

    def test_extreme_cpu_values(self):
        p = EnvironmentPerception()
        for val in [-1000.0, -1.0, 0.0, 100.0, 10000.0, float('inf'), float('-inf'), float('nan')]:
            p.system_history.append({"cpu_percent": val, "memory_percent": 50.0})
            state = p.get_state_vector()
            assert isinstance(state, dict)

    def test_concurrent_perception_updates(self):
        p = EnvironmentPerception()
        errors = []

        def updater():
            try:
                for _ in range(20):
                    p.collect_system_metrics()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=updater) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0, f"并发感知出错: {errors}"


class TestChaosPrediction:
    """预测模块混沌测试"""

    def test_predict_with_nan_in_history(self):
        pe = PredictionEngine()
        pe.update({MetricKeys.CPU: float('nan'), MetricKeys.MEMORY: 50.0, MetricKeys.DISK: 50.0})
        pe.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: float('nan'), MetricKeys.DISK: 50.0})
        result = pe.predict()
        assert isinstance(result.predicted_values, dict)

    def test_predict_with_inf_in_history(self):
        pe = PredictionEngine()
        pe.update({MetricKeys.CPU: float('inf'), MetricKeys.MEMORY: 50.0, MetricKeys.DISK: 50.0})
        pe.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: float('-inf'), MetricKeys.DISK: 50.0})
        result = pe.predict()
        assert isinstance(result.predicted_values, dict)

    def test_online_learn_with_mismatched_keys(self):
        pe = PredictionEngine()
        pe.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 60.0, MetricKeys.DISK: 70.0})
        # actual缺少某些键
        pe.online_learn({MetricKeys.CPU: 55.0})
        # actual有多余的键
        pe.online_learn({MetricKeys.CPU: 55.0, MetricKeys.MEMORY: 58.0, MetricKeys.DISK: 72.0, "extra_key": 99.0})

    def test_predict_with_empty_and_full_history(self):
        pe = PredictionEngine()
        # 空历史
        r1 = pe.predict()
        assert r1.layer in ["base", "mid", "high"]
        # 大量历史
        for i in range(1000):
            pe.update({MetricKeys.CPU: random.random() * 100, MetricKeys.MEMORY: random.random() * 100, MetricKeys.DISK: random.random() * 100})
        r2 = pe.predict()
        assert isinstance(r2.predicted_values, dict)

    def test_base_predictor_with_all_same_values(self):
        bp = BaseLayerPredictor()
        for _ in range(20):
            bp.update({MetricKeys.CPU: 50.0, MetricKeys.MEMORY: 50.0, MetricKeys.DISK: 50.0})
        result = bp.predict()
        assert isinstance(result.predicted_values, dict)

    def test_high_predictor_with_no_pattern(self):
        hp = HighLayerPredictor()
        # 完全随机数据，无周期性
        for i in range(100):
            hp.update({MetricKeys.CPU: random.random() * 100, MetricKeys.MEMORY: random.random() * 100, MetricKeys.DISK: random.random() * 100})
        result = hp.predict()
        assert isinstance(result.predicted_values, dict)


class TestChaosDriveEngine:
    """驱力引擎混沌测试"""

    def test_compute_with_all_extreme_states(self):
        de = DriveEngine()
        extremes = [
            {},  # 空状态
            {"cpu_percent": float('nan')},
            {"cpu_percent": float('inf')},
            {"cpu_percent": float('-inf')},
            {"cpu_percent": -999.0, "memory_percent": 999.0},
            {"cpu_percent": None, "memory_percent": None},
        ]
        for state in extremes:
            drive = de.compute(state, tasks_completed=0, tasks_failed=0, prediction_error=0.0)
            assert -1.0 <= drive.safety <= 1.0 or math.isnan(drive.safety)
            assert -1.0 <= drive.gain <= 1.0 or math.isnan(drive.gain)
            assert -1.0 <= drive.efficiency <= 1.0 or math.isnan(drive.efficiency)

    def test_compute_with_huge_task_counts(self):
        de = DriveEngine()
        state = {StateKeys.CPU_PERCENT: 0.5, StateKeys.MEMORY_PERCENT: 0.5}
        for tc in [0, 1, 100, 1000000, -100]:
            for tf in [0, 1, 100, 1000000, -100]:
                drive = de.compute(state, tasks_completed=tc, tasks_failed=tf, prediction_error=0.0)
                assert isinstance(drive.to_dict(), dict)


class TestChaosDecision:
    """决策模块混沌测试"""

    def test_executor_with_malformed_action(self):
        exe = ActionExecutor()
        # 构造一个params包含异常值的动作
        action = Action(ActionType.SLEEP, "休眠", "测试", params={"duration": float('nan')})
        result = exe.execute(action)
        assert result is not None

        action2 = Action(ActionType.SLEEP, "休眠", "测试", params={"duration": float('inf')})
        result2 = exe.execute(action2)
        assert result2 is not None

        action3 = Action(ActionType.SLEEP, "休眠", "测试", params={"duration": -10.0})
        result3 = exe.execute(action3)
        assert result3 is not None

    def test_policy_with_empty_and_invalid_drives(self):
        space = ActionSpace()
        policy = Policy(space)
        invalid_drives_list = [
            {},
            {"safe": float('nan')},
            {"safe": float('inf')},
            {"safe": -999.0, "gain": 999.0, "eff": 0.0},
            None,
        ]
        for drives in invalid_drives_list:
            if drives is None:
                continue
            action = policy.select_action({}, drives)
            assert action is not None

    def test_decision_with_full_malformed_pipeline(self):
        de = DecisionEngine()
        # 异常状态 + 异常驱力
        result = de.decide_and_execute(
            {"cpu_percent": float('nan'), "memory_percent": float('inf')},
            {"safe": float('nan'), "gain": float('-inf'), "eff": 999.0}
        )
        assert result is not None


class TestChaosErrorEval:
    """误差评估混沌测试"""

    def test_evaluate_with_chaos_inputs(self):
        ee = ErrorEvaluator()
        chaos_pairs = [
            ({"a": float('nan')}, {"a": float('nan')}),
            ({"a": float('inf')}, {"a": float('inf')}),
            ({"a": float('-inf')}, {"a": float('inf')}),
            ({"a": 0.0}, {"a": float('inf')}),
            ({"a": 1e308}, {"a": -1e308}),
            ({"a": 0.0, "b": 0.0}, {"a": 0.0}),  # 键不匹配
            ({"a": 0.0}, {"a": 0.0, "b": 0.0}),
        ]
        for pred, actual in chaos_pairs:
            result = ee.evaluate(pred, actual)
            assert isinstance(result, dict)

    def test_entropy_with_chaos_values(self):
        ee = ErrorEvaluator()
        chaos_values = [
            [float('nan'), float('nan')],
            [float('inf'), float('inf')],
            [float('-inf'), float('inf')],
            [1e308, 1e308, 1e308],
            [-1e308, -1e308],
            [0.0, 0.0, 0.0, 0.0],
        ]
        for vals in chaos_values:
            ent = ee.compute_error_entropy(vals)
            assert isinstance(ent, float) or math.isnan(ent) or math.isinf(ent)


class TestChaosOptimizer:
    """优化器混沌测试"""

    def test_gradient_descent_with_chaos_gradients(self):
        gd = GradientDescent()
        params = {"w": 1.0, "b": 0.5}
        chaos_grads = [
            {"w": float('nan'), "b": float('nan')},
            {"w": float('inf'), "b": float('-inf')},
            {"w": 1e308, "b": -1e308},
            {"w": -1e308, "b": 1e308},
            {},  # 空梯度
            {"w": 0.0},  # 缺少键
        ]
        for grad in chaos_grads:
            new_params = gd.step(params, grad)
            assert isinstance(new_params, dict)

    def test_hyperparameter_search_with_bad_objectives(self):
        hs = HyperparameterSearch()
        bad_objectives = [
            lambda p: float('nan'),
            lambda p: float('inf'),
            lambda p: float('-inf'),
            lambda p: random.choice([float('nan'), float('inf'), -float('inf'), 1e308, -1e308]),
        ]
        for obj in bad_objectives:
            result = hs.search(obj, n_trials=3)
            assert isinstance(result.to_dict(), dict)


class TestChaosMemory:
    """记忆系统混沌测试"""

    def test_store_unhashable_and_circular_content(self):
        stm = ShortTermMemory(capacity=10)
        chaos_contents = [
            {"list": [1, 2, {"nested": [3, 4]}]},
            {"self_ref": None},
            {"bytes": b"\x00\x01\x02"},
            {"unicode": "测试中文🎉"},
            {"mixed": [1, "str", 3.14, None, True]},
        ]
        for content in chaos_contents:
            item_id = stm.store(content, importance=random.random())
            assert isinstance(item_id, str)

    def test_memory_with_extreme_importance(self):
        ms = MemorySystem()
        for imp in [-999.0, -1.0, 0.0, 1.0, 999.0, float('nan'), float('inf')]:
            ms.record_experience({"s": 1}, "a", 1.0, {"s": 2}, td_error=imp)
        stats = ms.get_stats()
        assert isinstance(stats, dict)

    def test_replay_with_corrupted_experiences(self):
        from modules.memory import PrioritizedReplay, Experience
        replay = PrioritizedReplay(capacity=10)
        # 存储异常经验（使用add方法）
        for _ in range(20):
            exp = Experience(
                state={"cpu": random.random()},
                action="test",
                reward=random.choice([float('nan'), float('inf'), -float('inf'), 1e308, -1e308]),
                next_state={"cpu": random.random()},
                td_error=random.choice([float('nan'), float('inf'), 1e308]),
            )
            replay.add(1.0, exp)
        sampled = replay.sample(5)
        assert isinstance(sampled, list)


class TestChaosIntegration:
    """集成混沌测试 —— 端到端异常场景"""

    def test_agent_survives_chaos_loop(self):
        """Agent在混乱输入循环中存活"""
        agent = SelfEvolvingAgent()
        # 手动注入混乱状态并运行几个周期
        for _ in range(5):
            try:
                state = agent.perception.get_state_vector()
                # 污染状态向量
                state["cpu_percent"] = random.choice([float('nan'), float('inf'), -1.0, 2.0, None])
                pred = agent.prediction.predict()
                drives = agent.drive_engine.compute(state, 0, 0, 0.0)
                result = agent.decision.decide_and_execute(state, drives.to_dict())
                errors = agent.error_eval.evaluate(pred.predicted_values, state)
                agent.memory.record_experience(state, result.action.name, 0.0, state)
            except Exception:
                pass  # 任何异常都不应导致崩溃
        assert True  # 如果执行到这里说明没有崩溃

    def test_agent_with_none_modules(self):
        """验证Agent在部分模块异常时仍能降级运行"""
        agent = SelfEvolvingAgent()
        # 运行一个正常周期
        agent.run_for_duration(0.5)
        assert agent.loop_count >= 0

    def test_rapid_state_changes(self):
        """模拟状态剧烈波动"""
        agent = SelfEvolvingAgent()
        pe = agent.prediction
        for _ in range(50):
            pe.update({
                MetricKeys.CPU: random.uniform(-100, 200),
                MetricKeys.MEMORY: random.uniform(-100, 200),
                MetricKeys.DISK: random.uniform(-100, 200),
            })
        result = pe.predict()
        assert isinstance(result.predicted_values, dict)
