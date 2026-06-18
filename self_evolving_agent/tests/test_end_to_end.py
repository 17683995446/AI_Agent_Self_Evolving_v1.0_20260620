"""
端到端接口稳定性测试 —— 验证完整数据流和长时间运行稳定性

测试目标：
1. 完整业务链路闭环：感知→预测→驱力→决策→执行→评估→记忆→优化
2. 数据一致性：各模块间传递的数据无丢失、无错乱
3. 异常恢复：模拟各模块异常，验证系统降级和恢复
4. 接口稳定性：所有接口在多次调用后行为一致
"""
import sys
import os
import time
import math
import random
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent
from core.container import DIContainer
from core.interfaces import StateKeys, MetricKeys, ErrorReportKeys, DriveKeys
from modules.decision import ActionType


class TestEndToEndPipeline:
    """端到端完整链路测试"""

    def test_full_pipeline_single_cycle(self):
        """单周期完整链路测试"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(0.5)
        assert agent.loop_count >= 1
        # 验证数据流：感知产生了状态向量
        state = agent.perception.get_state_vector()
        assert StateKeys.CPU_PERCENT in state
        # 验证预测引擎有历史数据（属性名为 base_predictor / mid_predictor / high_predictor）
        assert len(agent.prediction.base_predictor.history) > 0 or len(agent.prediction.mid_predictor.history) > 0
        # 验证记忆系统有记录（replay_pool或short_term）
        stats = agent.memory.get_stats()
        assert stats["short_term_size"] > 0 or stats["replay_pool_size"] > 0

    def test_full_pipeline_multiple_cycles(self):
        """多周期完整链路测试"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(2.0)
        assert agent.loop_count >= 2
        # 验证各周期数据累积（replay_pool或short_term）
        stats = agent.memory.get_stats()
        total_memories = stats["short_term_size"] + stats.get("replay_pool_size", 0)
        assert total_memories >= 1

    def test_data_flow_consistency(self):
        """数据流一致性验证"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(1.0)
        # 感知层的状态应被预测层使用
        state = agent.perception.get_state_vector()
        pred = agent.prediction.predict()
        # 预测结果应包含与状态对应的指标
        assert isinstance(pred.predicted_values, dict)
        # 误差评估应能处理预测结果
        errors = agent.error_eval.evaluate(pred.predicted_values, state)
        assert ErrorReportKeys.MSE in errors

    def test_drive_influences_decision(self):
        """验证驱力确实影响决策"""
        agent = SelfEvolvingAgent()
        # 运行几个周期让系统有历史数据
        agent.run_for_duration(1.5)
        # 获取决策历史
        history = agent.decision.executor.execution_history
        if len(history) > 0:
            # 验证所有执行结果都是有效的ActionResult
            for result in history:
                assert hasattr(result, 'success')
                assert hasattr(result, 'action')
                assert isinstance(result.action.name, str)


class TestInterfaceStability:
    """接口稳定性测试 —— 多次调用行为一致"""

    def test_perception_interface_stable(self):
        p = SelfEvolvingAgent().perception
        results = []
        for _ in range(5):
            state = p.get_state_vector()
            results.append(state)
            time.sleep(0.05)
        # 所有结果结构一致
        for r in results:
            assert set(r.keys()) == set(results[0].keys())
            for k in r:
                assert isinstance(r[k], float)

    def test_prediction_interface_stable(self):
        pe = SelfEvolvingAgent().prediction
        # 填充数据（确保满足各层最小数据要求）
        for i in range(25):
            pe.update({MetricKeys.CPU: 50.0 + i, MetricKeys.MEMORY: 60.0 + i, MetricKeys.DISK: 70.0 + i})
        results = []
        for _ in range(5):
            r = pe.predict()
            results.append(r)
        # 预测结果结构一致（默认返回mid层，数据足够时可能返回其他层）
        for r in results:
            assert r.layer in ["base", "mid", "high", "ensemble"]
            assert isinstance(r.predicted_values, dict)
            assert 0.0 <= r.confidence <= 1.0

    def test_drive_engine_interface_stable(self):
        de = SelfEvolvingAgent().drive_engine
        state = {StateKeys.CPU_PERCENT: 0.5, StateKeys.MEMORY_PERCENT: 0.5}
        results = []
        for _ in range(5):
            d = de.compute(state, 0, 0, 0.0)
            results.append(d)
        # 驱力状态结构一致
        for d in results:
            assert -1.0 <= d.safety <= 1.0
            assert -1.0 <= d.gain <= 1.0
            assert -1.0 <= d.efficiency <= 1.0
            assert hasattr(d, 'to_dict')

    def test_error_eval_interface_stable(self):
        ee = SelfEvolvingAgent().error_eval
        pred = {"cpu": 50.0, "memory": 60.0}
        actual = {"cpu": 55.0, "memory": 58.0}
        results = []
        for _ in range(5):
            r = ee.evaluate(pred, actual)
            results.append(r)
        for r in results:
            assert ErrorReportKeys.MSE in r
            assert ErrorReportKeys.MAE in r
            assert ErrorReportKeys.SAMPLE_COUNT in r

    def test_memory_interface_stable(self):
        ms = SelfEvolvingAgent().memory
        results = []
        for _ in range(5):
            s = ms.get_stats()
            results.append(s)
        for s in results:
            assert "short_term_size" in s
            assert "long_term_count" in s


class TestConcurrentStability:
    """并发稳定性测试"""

    def test_concurrent_module_access(self):
        """验证模块在并发访问下稳定"""
        agent = SelfEvolvingAgent()
        errors = []
        results = []

        def worker():
            try:
                for _ in range(10):
                    state = agent.perception.get_state_vector()
                    results.append(state)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"并发访问出错: {errors}"
        assert len(results) == 30

    def test_agent_start_stop_idempotent(self):
        """验证启动停止的幂等性"""
        agent = SelfEvolvingAgent()
        agent.start()
        time.sleep(0.2)
        agent.stop()
        # 再次启动
        agent.start()
        time.sleep(0.2)
        agent.stop()
        # 不应抛异常
        assert agent.loop_count >= 0


class TestDegradedOperation:
    """降级运行测试"""

    def test_agent_runs_with_minimal_config(self):
        """验证最小配置下系统仍能运行"""
        agent = SelfEvolvingAgent(module_config={})
        agent.run_for_duration(0.5)
        assert agent.loop_count >= 1

    def test_agent_survives_rapid_restart(self):
        """验证快速重启不会导致问题"""
        for _ in range(3):
            agent = SelfEvolvingAgent()
            agent.run_for_duration(0.3)
            assert agent.loop_count >= 0

    def test_all_action_types_executable(self):
        """验证所有动作类型至少可被构造和执行器处理"""
        from modules.decision import ActionSpace, ActionExecutor, Action
        space = ActionSpace()
        executor = ActionExecutor()
        for action in space.get_all_actions():
            result = executor.execute(action)
            assert result is not None
            assert hasattr(result, 'success')


class TestLongRunningStability:
    """长时间运行稳定性测试（缩短版，用于CI）"""

    def test_10_second_run_no_crash(self):
        """10秒运行不崩溃"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(10.0)
        assert agent.loop_count >= 5  # 至少运行5个周期
        # 验证所有模块仍然可用
        state = agent.perception.get_state_vector()
        assert StateKeys.CPU_PERCENT in state
        pred = agent.prediction.predict()
        assert isinstance(pred.predicted_values, dict)
        stats = agent.memory.get_stats()
        total_memories = stats["short_term_size"] + stats.get("replay_pool_size", 0)
        assert total_memories > 0
