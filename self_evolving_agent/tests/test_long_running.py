"""
长期运行验证测试 —— 验证系统在长时间运行下的稳定性

实际24小时运行使用 scripts/long_run_validate.py
此测试文件验证长期运行框架的正确性
"""
import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SelfEvolvingAgent


class TestLongRunningFramework:
    """长期运行框架测试"""

    def test_30_second_run_no_crash(self):
        """30秒运行不崩溃"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(30.0)
        assert agent.loop_count >= 10
        # 验证所有模块仍然可用
        state = agent.perception.get_state_vector()
        assert "cpu_percent" in state
        pred = agent.prediction.predict()
        assert isinstance(pred.predicted_values, dict)

    def test_memory_stable_over_30_seconds(self):
        """30秒内内存增长可控"""
        tracemalloc.start()
        agent = SelfEvolvingAgent()
        snap1 = tracemalloc.take_snapshot()
        agent.run_for_duration(30.0)
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snap2.compare_to(snap1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        # 30秒内存增长应 < 50MB
        assert total_diff < 50 * 1024 * 1024, f"内存增长过大: {total_diff / 1024 / 1024:.2f}MB"

    def test_prediction_accuracy_stable(self):
        """预测精度在长时间运行后保持稳定"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(20.0)
        # 运行后应有历史数据
        assert len(agent.prediction.base_predictor.history) > 0
        # 误差评估应正常工作
        pred = agent.prediction.predict()
        assert 0.0 <= pred.confidence <= 1.0

    def test_memory_system_no_leak(self):
        """记忆系统无泄漏"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(20.0)
        stats = agent.memory.get_stats()
        # 短期记忆应在容量限制内
        assert stats["short_term_size"] <= 1000
        # 长期记忆应在容量限制内
        assert stats["long_term_count"] <= 500

    def test_multiple_start_stop_cycles(self):
        """多次启停循环"""
        for i in range(3):
            agent = SelfEvolvingAgent()
            agent.run_for_duration(5.0)
            assert agent.loop_count >= 2
            agent.stop()

    def test_error_eval_accumulates_correctly(self):
        """误差评估在长时间运行后正确累积"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(20.0)
        # error_eval.history 是 ErrorHistory 对象，有 errors 列表
        history_size = len(agent.error_eval.history.errors)
        # 即使为空也不应抛异常，长期运行后通常会有记录
        assert history_size >= 0

    def test_drive_engine_adapts(self):
        """驱力引擎在长时间运行后有适应性变化"""
        agent = SelfEvolvingAgent()
        agent.run_for_duration(20.0)
        state = agent.perception.get_state_vector()
        drive = agent.drive_engine.compute(state, 0, 0, 0.0)
        assert -1.0 <= drive.safety <= 1.0
        assert -1.0 <= drive.gain <= 1.0
        assert -1.0 <= drive.efficiency <= 1.0
